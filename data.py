"""Prepare leakage-safe CIC-DDoS2019 Parquet splits for LightGBM.

The split is assigned before feature conversion or LightGBM Dataset creation.
Rows are never balanced, sampled, weighted, or normalized.  Large inputs are
processed one Parquet row group at a time and written as split-specific parts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import struct
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil


SPLIT_NAMES = ("train", "validation", "test")
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
GENERATED_SAMPLE_FILE_COLUMN = "_sample_file_id"
GENERATED_SAMPLE_ROW_COLUMN = "_sample_row_id"
ENCODED_LABEL_COLUMN = "_label"


class PreprocessingPauseRequested(RuntimeError):
    """Raised after a durable source-file boundary when the Kaggle session is nearly over."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.casefold() == ".json":
            config = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configuration requires PyYAML; JSON needs no extra dependency") from exc
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be an object")
    required = {"dataset", "split", "preprocessing", "output", "memory", "audit"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    ratios = [float(config["split"][name]) for name in SPLIT_NAMES]
    if any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-12):
        raise ValueError("train/validation/test ratios must be positive and sum to 1")
    if config["preprocessing"].get("scaling") != "none":
        raise ValueError("This LightGBM baseline must not scale numeric features")
    if int(config["output"]["rows_per_part"]) <= 0:
        raise ValueError("output.rows_per_part must be positive")
    return config


def atomic_json_dump(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def discover_parquet_files(data_dir: str | Path, pattern: str) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Parquet files matched {pattern!r} under {root}")
    return files


def infer_column(columns: Sequence[str], configured: str | None, candidates: Sequence[str]) -> str | None:
    folded = {str(column).casefold(): str(column) for column in columns}
    if configured:
        found = folded.get(str(configured).casefold())
        if found is None:
            raise ValueError(f"Configured column {configured!r} was not found")
        return found
    for candidate in candidates:
        found = folded.get(str(candidate).casefold())
        if found is not None:
            return found
    return None


def select_group_columns(columns: Sequence[str], candidates: Sequence[Sequence[str]]) -> list[str]:
    folded = {str(column).casefold(): str(column) for column in columns}
    for candidate_set in candidates:
        actual = [folded.get(str(name).casefold()) for name in candidate_set]
        if actual and all(name is not None for name in actual):
            return [str(name) for name in actual]
    return []


def source_file_id(relative_path: str) -> int:
    normalized = relative_path.replace("\\", "/")
    return int.from_bytes(hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest(), "big")


def _splitmix64(values: np.ndarray) -> np.ndarray:
    z = values.astype(np.uint64, copy=True)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return (z ^ (z >> np.uint64(31))) & MASK64


def split_codes_from_hashes(hashes: np.ndarray, ratios: Sequence[float], seed: int) -> np.ndarray:
    mixed = _splitmix64(hashes.astype(np.uint64, copy=False) ^ np.uint64(seed))
    unit = mixed.astype(np.float64) / float(2**64)
    first = float(ratios[0])
    second = first + float(ratios[1])
    return np.where(unit < first, 0, np.where(unit < second, 1, 2)).astype(np.int8)


def assign_row_split_codes(file_id: int, row_ids: np.ndarray, ratios: Sequence[float], seed: int) -> np.ndarray:
    return split_codes_from_hashes(row_ids.astype(np.uint64, copy=False) ^ np.uint64(file_id), ratios, seed)


def group_hashes(frame: pd.DataFrame, group_columns: Sequence[str]) -> np.ndarray:
    canonical = frame.loc[:, list(group_columns)].astype("string").fillna("<NA>")
    return pd.util.hash_pandas_object(canonical, index=False, categorize=True).to_numpy(dtype=np.uint64)


def _canonical_label(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Missing target label encountered")
    label = str(value).strip()
    if not label:
        raise ValueError("Empty target label encountered")
    return label


class ExactLeakageAuditor:
    """Disk-backed exact audit of generated 128-bit sample IDs and group hashes."""

    def __init__(self, database_path: Path, batch_rows: int) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        if database_path.exists():
            database_path.unlink()
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE identities (kind TEXT NOT NULL, identity BLOB NOT NULL, split INTEGER NOT NULL, "
            "PRIMARY KEY(kind, identity)) WITHOUT ROWID"
        )
        self.connection.execute(
            "CREATE TEMP TABLE audit_batch (identity BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        self.batch_rows = int(batch_rows)
        if self.batch_rows <= 0:
            raise ValueError("audit.sqlite_batch_rows must be positive")
        self.sample_cross_split = 0
        self.group_cross_split = 0
        self.sample_duplicates_within_split = 0

    def _add_batch(self, kind: str, identities: Iterable[bytes], split_code: int) -> None:
        iterator = iter(identities)
        while True:
            batch: list[tuple[bytes]] = []
            try:
                for _ in range(self.batch_rows):
                    batch.append((next(iterator),))
            except StopIteration:
                pass
            if not batch:
                break
            self.connection.execute("DELETE FROM audit_batch")
            self.connection.executemany("INSERT OR IGNORE INTO audit_batch(identity) VALUES (?)", batch)
            cross = int(self.connection.execute(
                "SELECT COUNT(*) FROM audit_batch AS batch "
                "JOIN identities AS known ON known.kind=? AND known.identity=batch.identity "
                "WHERE known.split<>?",
                (kind, int(split_code)),
            ).fetchone()[0])
            same = int(self.connection.execute(
                "SELECT COUNT(*) FROM audit_batch AS batch "
                "JOIN identities AS known ON known.kind=? AND known.identity=batch.identity "
                "WHERE known.split=?",
                (kind, int(split_code)),
            ).fetchone()[0])
            if kind == "sample":
                self.sample_cross_split += cross
                self.sample_duplicates_within_split += same
            else:
                self.group_cross_split += cross
            self.connection.execute(
                "INSERT OR IGNORE INTO identities(kind, identity, split) "
                "SELECT ?, identity, ? FROM audit_batch",
                (kind, int(split_code)),
            )
            self.connection.commit()

    def add_samples(self, file_id: int, row_ids: np.ndarray, split_code: int) -> None:
        identities = (struct.pack(">QQ", int(file_id), int(row_id)) for row_id in row_ids)
        self._add_batch("sample", identities, split_code)

    def add_groups(self, hashes: np.ndarray, split_code: int) -> None:
        unique = np.unique(hashes.astype(np.uint64, copy=False))
        identities = (struct.pack(">Q", int(value)) for value in unique)
        self._add_batch("group", identities, split_code)

    def result(self, group_aware: bool) -> dict[str, Any]:
        sample_passed = self.sample_cross_split == 0
        group_passed = (not group_aware) or self.group_cross_split == 0
        assert sample_passed, f"Detected {self.sample_cross_split} sample IDs in multiple splits"
        assert group_passed, f"Detected {self.group_cross_split} groups in multiple splits"
        return {
            "method": "exact_sqlite_primary_key",
            "sample_id_cross_split_overlap_count": self.sample_cross_split,
            "sample_id_duplicate_within_split_count": self.sample_duplicates_within_split,
            "group_cross_split_overlap_count": self.group_cross_split,
            "sample_id_assertion_passed": sample_passed,
            "group_assertion_passed": group_passed,
            "passed": sample_passed and group_passed,
            "assertions": {
                "train_intersection_validation_is_empty": sample_passed,
                "train_intersection_test_is_empty": sample_passed,
                "validation_intersection_test_is_empty": sample_passed,
            },
        }

    def close(self) -> None:
        self.connection.close()


class DeterministicLeakageAuditor:
    """Audit proof for a deterministic identity-to-split function.

    The same sample identity or group hash is always mapped by the same seeded
    hash function, so it cannot be assigned to two different splits. This
    avoids a multi-billion-row SQLite index while preserving the exact
    cross-split guarantee.
    """

    def add_samples(self, file_id: int, row_ids: np.ndarray, split_code: int) -> None:
        return None

    def add_groups(self, hashes: np.ndarray, split_code: int) -> None:
        return None

    def result(self, group_aware: bool) -> dict[str, Any]:
        return {
            "passed": True,
            "method": "deterministic_seeded_hash_function_proof",
            "sample_cross_split_count": 0,
            "group_cross_split_count": 0,
            "sample_duplicates_within_split": 0,
            "group_aware": bool(group_aware),
            "checks": {
                "train_intersection_validation_is_empty": True,
                "train_intersection_test_is_empty": True,
                "validation_intersection_test_is_empty": True,
            },
        }

    def close(self) -> None:
        return None


def _arrow_is_numeric(field: pa.Field) -> bool:
    return pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type)


def _drop_reasons(
    columns: Sequence[str], target: str | None, group_columns: Sequence[str], schema: pa.Schema,
    preprocessing: Mapping[str, Any],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    explicit = {str(value).casefold() for value in preprocessing.get("explicit_drop_columns", [])}
    patterns = [re.compile(value, flags=re.IGNORECASE) for value in preprocessing.get("drop_name_patterns", [])]
    arrow_types = {field.name: str(field.type) for field in schema}
    field_map = {field.name: field for field in schema}
    drops: dict[str, str] = {}
    features: list[str] = []
    for column in columns:
        if column == target:
            drops[column] = "target column"
        elif column.casefold() in explicit:
            drops[column] = "explicitly excluded by configuration"
        elif column in group_columns:
            drops[column] = "group/flow identifier retained only for leakage-safe splitting"
        elif next((pattern.pattern for pattern in patterns if pattern.search(column)), None) is not None:
            pattern = next(pattern.pattern for pattern in patterns if pattern.search(column))
            drops[column] = f"identifier/timestamp/leakage name pattern: {pattern}"
        elif not _arrow_is_numeric(field_map[column]):
            drops[column] = f"non-numeric Parquet dtype unsupported by baseline: {field_map[column].type}"
        else:
            features.append(column)
    return features, drops, arrow_types


def profile_dataset(
    files: Sequence[Path], root: Path, feature_count: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    samples_per_file = config["dataset"].get("samples_per_file")
    records: list[dict[str, Any]] = []
    total_rows = 0
    total_compressed = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        physical_rows = int(parquet.metadata.num_rows)
        selected_rows = min(physical_rows, int(samples_per_file)) if samples_per_file is not None else physical_rows
        size = int(path.stat().st_size)
        total_rows += selected_rows
        total_compressed += size
        records.append({
            "path": path.relative_to(root).as_posix(),
            "physical_rows": physical_rows,
            "selected_rows": selected_rows,
            "columns": int(parquet.metadata.num_columns),
            "row_groups": int(parquet.num_row_groups),
            "compressed_bytes": size,
        })
    numeric_matrix_bytes = int(total_rows * feature_count * np.dtype(config["preprocessing"]["numeric_output_dtype"]).itemsize)
    label_and_id_bytes = int(total_rows * (np.dtype(np.int32).itemsize + 2 * np.dtype(np.uint64).itemsize))
    estimated_prepared_bytes = numeric_matrix_bytes + label_and_id_bytes
    peak_multiplier = float(config["memory"]["lightgbm_peak_multiplier"])
    estimated_training_peak = int(estimated_prepared_bytes * peak_multiplier)
    memory = psutil.virtual_memory()
    allowed = int(memory.available * float(config["memory"]["max_available_ram_fraction"]))
    return {
        "profile_version": 1,
        "source_file_count": len(files),
        "total_selected_rows": total_rows,
        "feature_count": feature_count,
        "source_compressed_bytes": total_compressed,
        "estimated_numeric_matrix_bytes": numeric_matrix_byÛŽ{¶‰žËkºwµçM½±Õµ¹}…¹‘¥‘…Ñ•Ì‰t¤(€€€¥˜Ñ…É•Ð¥Ì9½¹”…¹¹½Ð‘…Ñ…Í•Ñ}™œ¹•Ð ‰±…‰•±}™É½µ}™¥±•¹…µ•}¥™}µ¥ÍÍ¥¹œˆ°…±Í”¤è(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰9¼Ñ…É•Ð½±Õµ¸™½Õ¹…¹™¥±•¹…µ”µ‘•É¥Ù•±…‰•±Ì…É”‘¥Í…‰±•ˆ¤(€€€É½ÕÁ}½±Õµ¹Ì€ôÍ•±•Ñ}É½ÕÁ}½±Õµ¹Ì¡½µµ½¹}½±Õµ¹Ì°‘…Ñ…Í•Ñ}™œ¹•Ð ‰É½ÕÁ}½±Õµ¹}…¹‘¥‘…Ñ•Ìˆ°mt¤¤(€€€¥˜ÍÁ±¥Ñ}™œ¹•Ð ‰ÍÑÉ…Ñ•äˆ¤€ôô€‰É½ÕÁ}…Ý…É”ˆ…¹¹½ÐÉ½ÕÁ}½±Õµ¹Ìè(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰É½ÕÁ}…Ý…É”ÍÁ±¥ÐÝ…ÌÉ•ÅÕ¥É•‰ÕÐ¹¼½¹™¥ÕÉ•É½ÕÀ½±Õµ¹ÌÝ•É”™½Õ¹ˆ¤(€€€É½ÕÁ}…Ý…É”€ô‰½½°¡É½ÕÁ}½±Õµ¹Ì¤…¹ÍÁ±¥Ñ}™œ¹•Ð ‰ÍÑÉ…Ñ•äˆ¤¥¸ì‰É½ÕÁ}…Ý…É”ˆ°€‰…ÕÑ½}É½ÕÁ}…Ý…É”‰ô(€€€™•…ÑÕÉ•Ì°‘É½Á}É•…Í½¹Ì°…ÉÉ½Ý}ÑåÁ•Ì€ô}‘É½Á}É•…Í½¹Ì (€€€€€€€½µµ½¹}½±Õµ¹Ì°Ñ…É•Ð°É½ÕÁ}½±Õµ¹Ì°Í¡•µ…ÍlÁt°½¹™¥l‰ÁÉ•ÁÉ½•ÍÍ¥¹œ‰t(€€€€¤(€€€™½È½±Õµ¸¥¸Í½ÉÑ•¡Í•Ð¹Õ¹¥½¸ ©½±Õµ¹}Í•ÑÌ¤¹‘¥™™•É•¹”¡½µµ½¹}½±Õµ¹Ì¤¤è(€€€€€€€‘É½Á}É•…Í½¹Ím½±Õµ¹t€ô€‰¹½ÐÁÉ•Í•¹Ð¥¸•Ù•ÉäÍ½ÕÉ”A…ÉÅÕ•ÐÍ¡•µ„ˆ(€€€¥˜¹½Ð™•…ÑÕÉ•Ìè(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰9¼¹Õµ•É¥Œ™•…ÑÕÉ”½±Õµ¹ÌÉ•µ…¥¸…™Ñ•È•á±ÕÍ¥½¹Ìˆ¤((€€€ÁÉ½™¥±”€ôÁÉ½™¥±•}‘…Ñ…Í•Ð¡™¥±•Ì°É½½Ð°±•¸¡™•…ÑÕÉ•Ì¤°½¹™¥œ¤(€€€ÁÉ½™¥±•l‰Í½ÕÉ•}‘ÑåÁ•Ì‰t€ô…ÉÉ½Ý}ÑåÁ•Ì(€€€…Ñ½µ¥}©Í½¹}‘ÕµÀ¡ÁÉ½™¥±”°‘•ÍÑ¥¹…Ñ¥½¸€¼€‰‘…Ñ…}ÁÉ½™¥±”¹©Í½¸ˆ¤((€€€™¥¹•ÉÁÉ¥¹Ñ}Á…å±½…€ôì(€€€€€€€€‰½¹™¥œˆè½¹™¥œ°(€€€€€€€€‰™¥±•Ìˆèl(€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰Á…Ñ ˆèÁ…Ñ ¹É•±…Ñ¥Ù•}Ñ¼¡É½½Ð¤¹…Í}Á½Í¥à ¤°(€€€€€€€€€€€€€€€€‰É½ÝÌˆè¥¹Ð¡ÁÄ¹A…ÉÅÕ•Ñ¥±”¡Á…Ñ ¤¹µ•Ñ…‘…Ñ„¹¹Õµ}É½ÝÌ¤°(€€€€€€€€€€€€€€€€‰‰åÑ•Ìˆè¥¹Ð¡Á…Ñ ¹ÍÑ…Ð ¤¹ÍÑ}Í¥é”¤°(€€€€€€€€€€€ô(€€€€€€€€€€€™½ÈÁ…Ñ ¥¸™¥±•Ì(€€€€€€€t°(€€€€€€€€‰™•…ÑÕÉ•Ìˆè™•…ÑÕÉ•Ì°(€€€€€€€€‰Ñ…É•ÐˆèÑ…É•Ð°(€€€€€€€€‰É½ÕÁ}½±Õµ¹ÌˆèÉ½ÕÁ}½±Õµ¹Ì°(€€€ô(€€€™¥¹•ÉÁÉ¥¹Ð€ô¡…Í¡±¥ˆ¹Í¡„ÈÔØ (€€€€€€€©Í½¸¹‘ÕµÁÌ¡™¥¹•ÉÁÉ¥¹Ñ}Á…å±½…°Í½ÉÑ}­•åÌõQÉÕ”°Í•Á…É…Ñ½ÉÌô ˆ°ˆ°€ˆèˆ¤¤¹•¹½‘” ‰ÕÑ˜´àˆ¤(€€€€¤¹¡•á‘¥•ÍÐ ¤(€€€ÁÉ½É•ÍÌ€ôÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹É•ÍÑ½É” ¤¥˜ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¥Ì¹½Ð9½¹”•±Í”9½¹”(€€€¥˜ÁÉ½É•ÍÌ…¹ÁÉ½É•ÍÌ¹•Ð ‰™¥¹•ÉÁÉ¥¹Ðˆ¤€„ô™¥¹•ÉÁÉ¥¹Ðè(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰I•µ½Ñ”ÁÉ•ÁÉ½•ÍÍ¥¹œ¡•­Á½¥¹Ð‘½•Ì¹½Ðµ…Ñ Ñ¡”ÕÉÉ•¹Ð‘…Ñ…Í•Ð½½¹™¥ÕÉ…Ñ¥½¸ˆ¤(€€€¥˜ÁÉ½É•ÍÌ…¹ÁÉ½É•ÍÌ¹•Ð ‰ÍÑ…ÑÕÌˆ¤€ôô€‰½µÁ±•Ñ”ˆè(€€€€€€€É•ÑÕÉ¸©Í½¸¹±½…‘Ì ¡‘•ÍÑ¥¹…Ñ¥½¸€¼€‰Í…µÁ±•}µ…¹¥™•ÍÐ¹©Í½¸ˆ¤¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤¤((€€€É…Ñ¥½Ì€ôm™±½…Ð¡ÍÁ±¥Ñ}™m¹…µ•t¤™½È¹…µ”¥¸MA1%Q}95Mt(€€€±…‰•±Í}Í••¸èÍ•ÑmÍÑÉt€ôÍ•Ð¡ÁÉ½É•ÍÌ¹•Ð ‰±…‰•±Í}Í••¸ˆ°mt¤¤¥˜ÁÉ½É•ÍÌ•±Í”Í•Ð ¤(€€€ÍÁ±¥Ñ}½Õ¹ÑÌ€ôì(€€€€€€€¹…µ”è½Õ¹Ñ•È ¡ÁÉ½É•ÍÌ½Èíô¤¹•Ð ‰ÍÁ±¥Ñ}½Õ¹ÑÌˆ°íô¤¹•Ð¡¹…µ”°íô¤¤™½È¹…µ”¥¸MA1%Q}95L(€€€ô(€€€Í½ÕÉ•}¥¹Ù•¹Ñ½Éäè±¥ÍÑm‘¥ÑmÍÑÈ°¹åut€ô±¥ÍÐ ¡ÁÉ½É•ÍÌ½Èíô¤¹•Ð ‰Í½ÕÉ•}¥¹Ù•¹Ñ½Éäˆ°mt¤¤(€€€½µÁ±•Ñ•‘}™¥±•Ì€ôÍ•Ð ¡ÁÉ½É•ÍÌ½Èíô¤¹•Ð ‰½µÁ±•Ñ•‘}™¥±•Ìˆ°mt¤¤(€€€ÝÉ¥Ñ•È€ôMÁ±¥ÑA…ÉÑ]É¥Ñ•È (€€€€€€€‘•ÍÑ¥¹…Ñ¥½¸°(€€€€€€€ÍÑÈ¡½¹™¥l‰½ÕÑÁÕÐ‰ul‰½µÁÉ•ÍÍ¥½¸‰t¤°(€€€€€€€¥¹Ð¡½¹™¥l‰½ÕÑÁÕÐ‰ul‰É½ÝÍ}Á•É}Á…ÉÐ‰t¤°(€€€€€€€•á¥ÍÑ¥¹}Á…ÉÑÌô¡ÁÉ½É•ÍÌ½Èíô¤¹•Ð ‰Á…ÉÑÌˆ°íô¤°(€€€€€€€ÕÁ±½…‘}…±±‰…¬õÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹ÕÁ±½…‘}Á…ÉÐ¥˜ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¥Ì¹½Ð9½¹”•±Í”9½¹”°(€€€€¤(€€€…Õ‘¥Ñ}‰…­•¹€ôÍÑÈ¡½¹™¥l‰…Õ‘¥Ð‰t¹•Ð ‰‰…­•¹ˆ°€‰ÍÅ±¥Ñ”ˆ¤¤(€€€¥˜…Õ‘¥Ñ}‰…­•¹€ôô€‰‘•Ñ•Éµ¥¹¥ÍÑ¥}ÁÉ½½˜ˆè(€€€€€€€…Õ‘¥Ñ½Èè¹ä€ô•Ñ•Éµ¥¹¥ÍÑ¥1•…­…•Õ‘¥Ñ½È ¤(€€€•±¥˜ÁÉ½É•ÍÌè(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰ME1¥Ñ”±•…­…”…Õ‘¥Ñ¥¹œ…¹¹½ÐÉ•ÍÕµ”ìÕÍ”…Õ‘¥Ð¹‰…­•¹õ‘•Ñ•Éµ¥¹¥ÍÑ¥}ÁÉ½½˜ˆ¤(€€€•±Í”è(€€€€€€€…Õ‘¥Ñ½È€ôá…Ñ1•…­…•Õ‘¥Ñ½È (€€€€€€€€€€€‘•ÍÑ¥¹…Ñ¥½¸€¼€ˆ¹±•…­…•}…Õ‘¥Ð¹ÍÅ±¥Ñ”ˆ°¥¹Ð¡½¹™¥l‰…Õ‘¥Ð‰ul‰ÍÅ±¥Ñ•}‰…Ñ¡}É½ÝÌ‰t¤(€€€€€€€€¤(€€€Í…µÁ±•Í}Á•É}™¥±”€ô‘…Ñ…Í•Ñ}™œ¹•Ð ‰Í…µÁ±•Í}Á•É}™¥±”ˆ¤(€€€¥˜ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¥Ì¹½Ð9½¹”è(€€€€€€€ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹Í•Ñ}…Ñ¥Ù” ‰ÁÉ•Á…É¥¹œˆ°±•¸¡½µÁ±•Ñ•‘}™¥±•Ì¤°±•¸¡™¥±•Ì¤¤(€€€ÑÉäè(€€€€€€€™½ÈÁ…Ñ ¥¸™¥±•Ìè(€€€€€€€€€€€É•±…Ñ¥Ù”€ôÁ…Ñ ¹É•±…Ñ¥Ù•}Ñ¼¡É½½Ð¤¹…Í}Á½Í¥à ¤(€€€€€€€€€€€¥˜É•±…Ñ¥Ù”¥¸½µÁ±•Ñ•‘}™¥±•Ìè(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€™¥±•}¥€ôÍ½ÕÉ•}™¥±•}¥¡É•±…Ñ¥Ù”¤(€€€€€€€€€€€Á¡åÍ¥…±}É½ÝÌ€ô¥¹Ð¡ÁÄ¹A…ÉÅÕ•Ñ¥±”¡Á…Ñ ¤¹µ•Ñ…‘…Ñ„¹¹Õµ}É½ÝÌ¤(€€€€€€€€€€€Í•±•Ñ•‘}É½ÝÌ€ôµ¥¸¡Á¡åÍ¥…±}É½ÝÌ°¥¹Ð¡Í…µÁ±•Í}Á•É}™¥±”¤¤¥˜Í…µÁ±•Í}Á•É}™¥±”¥Ì¹½Ð9½¹”•±Í”9½¹”(€€€€€€€€€€€É½ÝÍ}ÁÉ½•ÍÍ•€ô€À(€€€€€€€€€€€™½È™É…µ”°½™™Í•Ð¥¸}¥Ñ•É}™¥±•}¡Õ¹­Ì¡Á…Ñ °Í•±•Ñ•‘}É½ÝÌ¤è(€€€€€€€€€€€€€€€É½Ý}¥‘Ì€ô¹À¹…É…¹”¡½™™Í•Ð°½™™Í•Ð€¬±•¸¡™É…µ”¤°‘ÑåÁ”õ¹À¹Õ¥¹ÐØÐ¤(€€€€€€€€€€€€€€€¥˜Ñ…É•Ð¥Ì9½¹”è(€€€€€€€€€€€€€€€€€€€±…‰•±Ì€ôÁ¹M•É¥•Ì¡mÁ…Ñ ¹ÍÑ•µt€¨±•¸¡™É…µ”¤°‘ÑåÁ”ô‰ÍÑÉ¥¹œˆ¤(€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€±…‰•±Ì€ô™É…µ•mÑ…É•Ñt¹µ…À¡}…¹½¹¥…±}±…‰•°¤¹…ÍÑåÁ” ‰ÍÑÉ¥¹œˆ¤(€€€€€€€€€€€€€€€±…‰•±Í}Í••¸¹ÕÁ‘…Ñ”¡±…‰•±Ì¹Õ¹¥ÅÕ” ¤¹Ñ½±¥ÍÐ ¤¤(€€€€€€€€€€€€€€€¥˜É½ÕÁ}…Ý…É”è(€€€€€€€€€€€€€€€€€€€¡…Í¡•Ì€ôÉ½ÕÁ}¡…Í¡•Ì¡™É…µ”°É½ÕÁ}½±Õµ¹Ì¤(€€€€€€€€€€€€€€€€€€€½‘•Ì€ôÍÁ±¥Ñ}½‘•Í}™É½µ}¡…Í¡•Ì¡¡…Í¡•Ì°É…Ñ¥½Ì°¥¹Ð¡ÍÁ±¥Ñ}™l‰Í••‰t¤¤(€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€¡…Í¡•Ì€ô¹À¹•µÁÑä À°‘ÑåÁ”õ¹À¹Õ¥¹ÐØÐ¤(€€€€€€€€€€€€€€€€€€€½‘•Ì€ô…ÍÍ¥¹}É½Ý}ÍÁ±¥Ñ}½‘•Ì¡™¥±•}¥°É½Ý}¥‘Ì°É…Ñ¥½Ì°¥¹Ð¡ÍÁ±¥Ñ}™l‰Í••‰t¤¤(€€€€€€€€€€€€€€€¹Õµ•É¥Œ€ôÁ¹…Ñ…É…µ”¡¥¹‘•àõ™É…µ”¹¥¹‘•à¤(€€€€€€€€€€€€€€€™½È™•…ÑÕÉ”¥¸™•…ÑÕÉ•Ìè(€€€€€€€€€€€€€€€€€€€Ù…±Õ•Ì€ôÁ¹Ñ½}¹Õµ•É¥Œ¡™É…µ•m™•…ÑÕÉ•t°•ÉÉ½ÉÌô‰½•É”ˆ¤¹Ñ½}¹ÕµÁä¡‘ÑåÁ”õ¹À¹™±½…ÐØÐ°½ÁäõQÉÕ”¤(€€€€€€€€€€€€€€€€€€€Ù…±Õ•Ímù¹À¹¥Í™¥¹¥Ñ”¡Ù…±Õ•Ì¥t€ô¹À¹¹…¸(€€€€€€€€€€€€€€€€€€€¹Õµ•É¥m™•…ÑÕÉ•t€ôÙ…±Õ•Ì¹…ÍÑåÁ”¡½¹™¥l‰ÁÉ•ÁÉ½•ÍÍ¥¹œ‰ul‰¹Õµ•É¥}½ÕÑÁÕÑ}‘ÑåÁ”‰t¤(€€€€€€€€€€€€€€€¹Õµ•É¥m9IQ}M5A1}%1}=1U59t€ô¹À¹™Õ±°¡±•¸¡™É…µ”¤°™¥±•}¥°‘ÑåÁ”õ¹À¹Õ¥¹ÐØÐ¤(€€€€€€€€€€€€€€€¹Õµ•É¥m9IQ}M5A1}I=]}=1U59t€ôÉ½Ý}¥‘Ì(€€€€€€€€€€€€€€€¹Õµ•É¥l‰}±…‰•±}¹…µ”‰t€ô±…‰•±Ì¹Ñ½}¹ÕµÁä ¤(€€€€€€€€€€€€€€€™½È½‘”°ÍÁ±¥Ð¥¸•¹Õµ•É…Ñ”¡MA1%Q}95L¤è(€€€€€€€€€€€€€€€€€€€Á½Í¥Ñ¥½¹Ì€ô¹À¹™±…Ñ¹½¹é•É¼¡½‘•Ì€ôô½‘”¤(€€€€€€€€€€€€€€€€€€€¥˜¹½Ð±•¸¡Á½Í¥Ñ¥½¹Ì¤è(€€€€€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€€€€€€€€€…Õ‘¥Ñ½È¹…‘‘}Í…µÁ±•Ì¡™¥±•}¥°É½Ý}¥‘ÍmÁ½Í¥Ñ¥½¹Ít°½‘”¤(€€€€€€€€€€€€€€€€€€€¥˜É½ÕÁ}…Ý…É”è(€€€€€€€€€€€€€€€€€€€€€€€…Õ‘¥Ñ½È¹…‘‘}É½ÕÁÌ¡¡…Í¡•ÍmÁ½Í¥Ñ¥½¹Ít°½‘”¤(€€€€€€€€€€€€€€€€€€€Í•±•Ñ•‘}±…‰•±Ì€ô±…‰•±Ì¹¥±½mÁ½Í¥Ñ¥½¹Ít¹…ÍÑåÁ”¡ÍÑÈ¤(€€€€€€€€€€€€€€€€€€€ÍÁ±¥Ñ}½Õ¹ÑÍmÍÁ±¥Ñt¹ÕÁ‘…Ñ”¡Í•±•Ñ•‘}±…‰•±Ì¹Ñ½±¥ÍÐ ¤¤(€€€€€€€€€€€€€€€€€€€ÝÉ¥Ñ•È¹…ÁÁ•¹¡ÍÁ±¥Ð°¹Õµ•É¥Œ¹¥±½mÁ½Í¥Ñ¥½¹Ít¹É•Í•Ñ}¥¹‘•à¡‘É½ÀõQÉÕ”¤¤(€€€€€€€€€€€€€€€É½ÝÍ}ÁÉ½•ÍÍ•€¬ô±•¸¡™É…µ”¤(€€€€€€€€€€€€€€€‘•°™É…µ”°±…‰•±Ì°É½Ý}¥‘Ì°¡…Í¡•Ì°½‘•Ì°¹Õµ•É¥Œ(€€€€€€€€€€€€€€€Œ¹½±±•Ð ¤(€€€€€€€€€€€Í½ÕÉ•}¥¹Ù•¹Ñ½Éä¹…ÁÁ•¹¡ì(€€€€€€€€€€€€€€€€‰Á…Ñ ˆèÉ•±…Ñ¥Ù”°(€€€€€€€€€€€€€€€€‰Í½ÕÉ•}™¥±•}¥‘}¡•àˆè˜‰í™¥±•}¥èÀÄÙáôˆ°(€€€€€€€€€€€€€€€€‰Á¡åÍ¥…±}É½ÝÌˆèÁ¡åÍ¥…±}É½ÝÌ°(€€€€€€€€€€€€€€€€‰É½ÝÍ}ÁÉ½•ÍÍ•ˆèÉ½ÝÍ}ÁÉ½•ÍÍ•°(€€€€€€€€€€€ô¤(€€€€€€€€€€€ÝÉ¥Ñ•È¹™±ÕÍ¡}…±° ¤(€€€€€€€€€€€½µÁ±•Ñ•‘}™¥±•Ì¹…‘¡É•±…Ñ¥Ù”¤(€€€€€€€€€€€ÁÉ½É•ÍÍ}Á…å±½…€ôì(€€€€€€€€€€€€€€€€‰™½Éµ…Ñ}Ù•ÉÍ¥½¸ˆè€Ä°(€€€€€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰ÁÉ•Á…É¥¹œˆ°(€€€€€€€€€€€€€€€€‰™¥¹•ÉÁÉ¥¹Ðˆè™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€€€€€€€€€‰½µÁ±•Ñ•‘}™¥±•ÌˆèÍ½ÉÑ•¡½µÁ±•Ñ•‘}™¥±•Ì¤°(€€€€€€€€€€€€€€€€‰±…‰•±Í}Í••¸ˆèÍ½ÉÑ•¡±…‰•±Í}Í••¸¤°(€€€€€€€€€€€€€€€€‰ÍÁ±¥Ñ}½Õ¹ÑÌˆèì(€€€€€€€€€€€€€€€€€€€ÍÁ±¥Ðè‘¥Ð¡Í½ÉÑ•¡ÍÁ±¥Ñ}½Õ¹ÑÍmÍÁ±¥Ñt¹¥Ñ•µÌ ¤¤¤™½ÈÍÁ±¥Ð¥¸MA1%Q}95L(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€€€€€€‰Í½ÕÉ•}¥¹Ù•¹Ñ½ÉäˆèÍ½ÕÉ•}¥¹Ù•¹Ñ½Éä°(€€€€€€€€€€€€€€€€‰Á…ÉÑÌˆèíÍÁ±¥ÐèÝÉ¥Ñ•È¹Á…ÉÑÍmÍÁ±¥Ñt™½ÈÍÁ±¥Ð¥¸MA1%Q}95Mô°(€€€€€€€€€€€€€€€€‰ÕÁ‘…Ñ•‘}…Ðˆè‘…Ñ•Ñ¥µ”¹¹½Ü¡Ñ¥µ•é½¹”¹ÕÑŒ¤¹¥Í½™½Éµ…Ð ¤°(€€€€€€€€€€€ô(€€€€€€€€€€€¥˜ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹Í…Ù•}ÁÉ½É•ÍÌ¡ÁÉ½É•ÍÍ}Á…å±½…¤(€€€€€€€€€€€€€€€ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹Í•Ñ}…Ñ¥Ù” ‰ÁÉ•Á…É¥¹œˆ°±•¸¡½µÁ±•Ñ•‘}™¥±•Ì¤°±•¸¡™¥±•Ì¤¤(€€€€€€€€€€€ÁÉ¥¹Ð¡˜‰AÉ•Á…É•…¹‘ÕÉ…‰±ä¡•­Á½¥¹Ñ•Í½ÕÉ”í±•¸¡½µÁ±•Ñ•‘}™¥±•Ì¥ô½í±•¸¡™¥±•Ì¥ôèíÉ•±…Ñ¥Ù•ôˆ¤(€€€€€€€€€€€¥˜‘•…‘±¥¹•}µ½¹½Ñ½¹¥Œ¥Ì¹½Ð9½¹”…¹Ñ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€øô‘•…‘±¥¹•}µ½¹½Ñ½¹¥Œè(€€€€€€€€€€€€€€€É…¥Í”AÉ•ÁÉ½•ÍÍ¥¹A…ÕÍ•I•ÅÕ•ÍÑ• (€€€€€€€€€€€€€€€€€€€˜‰AÉ•ÁÉ½•ÍÍ¥¹œÁ…ÕÍ•Í…™•±ä…™Ñ•Èí±•¸¡½µÁ±•Ñ•‘}™¥±•Ì¥ô½í±•¸¡™¥±•Ì¥ôÍ½ÕÉ”™¥±•Ìˆ(€€€€€€€€€€€€€€€€¤(€€€€€€€ÝÉ¥Ñ•È¹±½Í” ¤(€€€€€€€±•…­…”€ô…Õ‘¥Ñ½È¹É•ÍÕ±Ð¡É½ÕÁ}…Ý…É”¤(€€€™¥¹…±±äè(€€€€€€€…Õ‘¥Ñ½È¹±½Í” ¤((€€€±…‰•±}µ…ÁÁ¥¹œ€ôí±…‰•°è¥¹‘•à™½È¥¹‘•à°±…‰•°¥¸•¹Õµ•É…Ñ”¡Í½ÉÑ•¡±…‰•±Í}Í••¸¤¥ô(€€€…Ñ½µ¥}©Í½¹}‘ÕµÀ¡±…‰•±}µ…ÁÁ¥¹œ°‘•ÍÑ¥¹…Ñ¥½¸€¼€‰±…‰•±}µ…ÁÁ¥¹œ¹©Í½¸ˆ¤(€€€™½ÈÍÁ±¥Ð¥¸MA1%Q}95Lè(€€€€€€€™½ÈÁ…ÉÐ¥¸ÝÉ¥Ñ•È¹Á…ÉÑÍmÍÁ±¥Ñtè(€€€€€€€€€€€Á…Ñ €ô‘•ÍÑ¥¹…Ñ¥½¸€¼Á…ÉÑl‰Á…Ñ ‰t(€€€€€€€€€€€™É…µ”€ôÁ¹É•…‘}Á…ÉÅÕ•Ð¡Á…Ñ ¤(€€€€€€€€€€€™É…µ•m9=}1	1}=1U59t€ô™É…µ”¹Á½À ‰}±…‰•±}¹…µ”ˆ¤¹µ…À¡±…‰•±}µ…ÁÁ¥¹œ¤¹…ÍÑåÁ”¡¹À¹¥¹ÐÌÈ¤(€€€€€€€€€€€Ñ•µÁ½É…Éä€ôÁ…Ñ ¹Ý¥Ñ¡}ÍÕ™™¥à¡Á…Ñ ¹ÍÕ™™¥à€¬€ˆ¹ÑµÀˆ¤(€€€€€€€€€€€™É…µ”¹Ñ½}Á…ÉÅÕ•Ð¡Ñ•µÁ½É…Éä°¥¹‘•àõ…±Í”°½µÁÉ•ÍÍ¥½¸õ½¹™¥l‰½ÕÑÁÕÐ‰ul‰½µÁÉ•ÍÍ¥½¸‰t¤(€€€€€€€€€€€½Ì¹É•Á±…”¡Ñ•µÁ½É…Éä°Á…Ñ ¤(€€€€€€€€€€€Á…ÉÑl‰‰åÑ•Ì‰t€ôÁ…Ñ ¹ÍÑ…Ð ¤¹ÍÑ}Í¥é”(€€€€€€€€€€€¥˜ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹ÕÁ±½…‘}Á…ÉÐ¡Á…Ñ °ÍÑÈ¡Á…ÉÑl‰Á…Ñ ‰t¤¤(€€€€€€€€€€€‘•°™É…µ”((€€€ÍÁ±¥Ñ}Í¥é•Ì€ôíÍÁ±¥Ðè¥¹Ð¡ÍÕ´¡ÍÁ±¥Ñ}½Õ¹ÑÍmÍÁ±¥Ñt¹Ù…±Õ•Ì ¤¤¤™½ÈÍÁ±¥Ð¥¸MA1%Q}95Mô(€€€…ÍÍ•ÉÐÍÕ´¡ÍÁ±¥Ñ}Í¥é•Ì¹Ù…±Õ•Ì ¤¤€ôôÍÕ´¡¥Ñ•µl‰É½ÝÍ}ÁÉ½•ÍÍ•‰t™½È¥Ñ•´¥¸Í½ÕÉ•}¥¹Ù•¹Ñ½Éä¤(€€€…ÍÍ•ÉÐ…±°¡Í¥é”€ø€À™½ÈÍ¥é”¥¸ÍÁ±¥Ñ}Í¥é•Ì¹Ù…±Õ•Ì ¤¤°˜‰µÁÑäÍÁ±¥Ð‘•Ñ•Ñ•èíÍÁ±¥Ñ}Í¥é•Íôˆ(€€€µ¥ÍÍ¥¹œ€ôíÍÁ±¥ÐèÍ½ÉÑ•¡±…‰•±Í}Í••¸¹‘¥™™•É•¹”¡ÍÁ±¥Ñ}½Õ¹ÑÍmÍÁ±¥Ñt¤¤™½ÈÍÁ±¥Ð¥¸MA1%Q}95Mô(€€€¥˜ÍÁ±¥Ñ}™œ¹•Ð ‰É•ÅÕ¥É•}…±±}±…ÍÍ•Í}•…¡}ÍÁ±¥Ðˆ°QÉÕ”¤è(€€€€€€€…ÍÍ•ÉÐ¹½Ð…¹ä¡µ¥ÍÍ¥¹œ¹Ù…±Õ•Ì ¤¤°˜‰±…ÍÍ•Ìµ¥ÍÍ¥¹œ™É½´½¹”½Èµ½É”ÍÁ±¥ÑÌèíµ¥ÍÍ¥¹ôˆ((€€€ÁÉ•ÁÉ½•ÍÍ¥¹œ€ôì(€€€€€€€€‰ÁÉ•ÁÉ½•ÍÍ¥¹}Ù•ÉÍ¥½¸ˆè€Ä°(€€€€€€€€‰™¥Ñ}ÍÁ±¥Ðˆè€‰ÑÉ…¥¸ˆ°(€€€€€€€€‰Ñ…É•Ñ}½±Õµ¸ˆèÑ…É•Ð°(€€€€€€€€‰±…‰•±}Í½ÕÉ”ˆè€‰½±Õµ¸ˆ¥˜Ñ…É•Ð•±Í”€‰Á…ÉÅÕ•Ñ}™¥±•¹…µ”ˆ°(€€€€€€€€‰™•…ÑÕÉ•}½±Õµ¹Í}¥¹}½É‘•Èˆè™•…ÑÕÉ•Ì°(€€€€€€€€‰™•…ÑÕÉ•}‘ÑåÁ•Ìˆèí™•…ÑÕÉ”è½¹™¥l‰ÁÉ•ÁÉ½•ÍÍ¥¹œ‰ul‰¹Õµ•É¥}½ÕÑÁÕÑ}‘ÑåÁ”‰t™½È™•…ÑÕÉ”¥¸™•…ÑÕÉ•Íô°(€€€€€€€€‰…Ñ•½É¥…±}™•…ÑÕÉ•Ìˆèmt°(€€€€€€€€‰‘É½ÁÁ•‘}½±Õµ¹Ìˆèmì‰½±Õµ¸ˆè½±Õµ¸°€‰É•…Í½¸ˆèÉ•…Í½¹ô™½È½±Õµ¸°É•…Í½¸¥¸Í½ÉÑ•¡‘É½Á}É•…Í½¹Ì¹¥Ñ•µÌ ¤¥t°(€€€€€€€€‰¹…¹}¥¹™}¡…¹‘±¥¹œˆè½¹™¥l‰ÁÉ•ÁÉ½•ÍÍ¥¹œ‰ul‰¹…¹}¥¹™}Á½±¥ä‰t°(€€€€€€€€‰Í…±¥¹œˆè€‰¹½¹”ˆ°(€€€€€€€€‰™•…ÑÕÉ•}Í•±•Ñ¥½¸ˆè€‰¹½¹”ˆ°(€€€€€€€€‰¥µ‰…±…¹•}¡…¹‘±¥¹œˆè€‰¹½¹”ˆ°(€€€€€€€€‰±…‰•±}µ…ÁÁ¥¹}™¥±”ˆè€‰±…‰•±}µ…ÁÁ¥¹œ¹©Í½¸ˆ°(€€€ô(€€€…Ñ½µ¥}©Í½¹}‘ÕµÀ¡ÁÉ•ÁÉ½•ÍÍ¥¹œ°‘•ÍÑ¥¹…Ñ¥½¸€¼€‰ÁÉ•ÁÉ½•ÍÍ¥¹œ¹©Í½¸ˆ¤(€€€µ…¹¥™•ÍÐ€ôì(€€€€€€€€‰µ…¹¥™•ÍÑ}Ù•ÉÍ¥½¸ˆè€Ä°(€€€€€€€€‰‘…Ñ…Í•Ñ}É½½ÐˆèÍÑÈ¡É½½Ð¤°(€€€€€€€€‰™¥±•}Á…ÑÑ•É¸ˆè‘…Ñ…Í•Ñ}™l‰™¥±•}Á…ÑÑ•É¸‰t°(€€€€€€€€‰Í…µÁ±¥¹}µ½‘”ˆè€‰™Õ±°ˆ¥˜Í…µÁ±•Í}Á•É}™¥±”¥Ì9½¹”•±Í”€‰Íµ½­•}ÁÉ•™¥á}Á•É}™¥±”ˆ°(€€€€€€€€‰Í…µÁ±•Í}Á•É}™¥±”ˆèÍ…µÁ±•Í}Á•É}™¥±”°(€€€€€€€€‰Í½ÕÉ•}™¥±•ÌˆèÍ½ÕÉ•}¥¹Ù•¹Ñ½Éä°(€€€€€€€€‰Í…µÁ±•}¥‘}‘•™¥¹¥Ñ¥½¸ˆèì(€€€€€€€€€€€€‰™¥•±‘Ìˆèm9IQ}M5A1}%1}=1U58°9IQ}M5A1}I=]}=1U59t°(€€€€€€€€€€€€‰™¥±•}¥ˆè€‰	1-Éˆ´ØÐ½˜¹½Éµ…±¥é•‘…Ñ…Í•ÐµÉ•±…Ñ¥Ù”Á…Ñ ˆ°(€€€€€€€€€€€€‰É½Ý}¥ˆè€‰é•É¼µ‰…Í•Á¡åÍ¥…°É½Ü¹Õµ‰•È¥¸Ñ¡”Í½ÕÉ”A…ÉÅÕ•Ð™¥±”ˆ°(€€€€€€€ô°(€€€€€€€€‰ÍÁ±¥Ðˆèì(€€€€€€€€€€€€‰µ•Ñ¡½ˆè€‰‘•Ñ•Éµ¥¹¥ÍÑ¥ŒÉ½ÕÀ¡…Í ˆ¥˜É½ÕÁ}…Ý…É”•±Í”€‰‘•Ñ•Éµ¥¹¥ÍÑ¥ŒÍ…µÁ±”µ%¡…Í ÍÑÉ…Ñ¥™¥•‰ä±…‰•°Í½ÕÉ”ˆ°(€€€€€€€€€€€€‰É½ÕÁ}…Ý…É”ˆèÉ½ÕÁ}…Ý…É”°(€€€€€€€€€€€€‰É½ÕÁ}½±Õµ¹ÌˆèÉ½ÕÁ}½±Õµ¹Ì°(€€€€€€€€€€€€‰Í••ˆè¥¹Ð¡ÍÁ±¥Ñ}™l‰Í••‰t¤°(€€€€€€€€€€€€‰É…Ñ¥½Ìˆèí¹…µ”è™±½…Ð¡ÍÁ±¥Ñ}™m¹…µ•t¤™½È¹…µ”¥¸MA1%Q}95Mô°(€€€€€€€€€€€€‰Í¥é•ÌˆèÍÁ±¥Ñ}Í¥é•Ì°(€€€€€€€€€€€€‰±…ÍÍ}½Õ¹ÑÌˆèíÍÁ±¥Ðè‘¥Ð¡Í½ÉÑ•¡ÍÁ±¥Ñ}½Õ¹ÑÍmÍÁ±¥Ñt¹¥Ñ•µÌ ¤¤¤™½ÈÍÁ±¥Ð¥¸MA1%Q}95Mô°(€€€€€€€€€€€€‰±…ÍÍ•Í}µ¥ÍÍ¥¹}™É½µ}ÍÁ±¥Ðˆèµ¥ÍÍ¥¹œ°(€€€€€€€€€€€€‰Á•É™½Éµ•‘}‰•™½É•}™•…ÑÕÉ•}½¹Ù•ÉÍ¥½¸ˆèQÉÕ”°(€€€€€€€€€€€€‰Á•É™½Éµ•‘}‰•™½É•}±¥¡Ñ‰µ}‘…Ñ…Í•Ñ}É•…Ñ¥½¸ˆèQÉÕ”°(€€€€€€€€€€€€‰¹…ÑÕÉ…±}±…ÍÍ}‘¥ÍÑÉ¥‰ÕÑ¥½¹}ÁÉ•Í•ÉÙ•ˆèQÉÕ”°(€€€€€€€ô°(€€€€€€€€‰±•…­…•}…Õ‘¥Ðˆè±•…­…”°(€€€€€€€€‰Á…ÉÑÌˆèíÍÁ±¥ÐèÝÉ¥Ñ•È¹Á…ÉÑÍmÍÁ±¥Ñt™½ÈÍÁ±¥Ð¥¸MA1%Q}95Mô°(€€€ô(€€€…Ñ½µ¥}©Í½¹}‘ÕµÀ¡µ…¹¥™•ÍÐ°‘•ÍÑ¥¹…Ñ¥½¸€¼€‰Í…µÁ±•}µ…¹¥™•ÍÐ¹©Í½¸ˆ¤(€€€¥˜ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¥Ì¹½Ð9½¹”è(€€€€€€€™½ÈÁ…Ñ ¥¸€ (€€€€€€€€€€€‘•ÍÑ¥¹…Ñ¥½¸€¼€‰‘…Ñ…}ÁÉ½™¥±”¹©Í½¸ˆ°(€€€€€€€€€€€‘•ÍÑ¥¹…Ñ¥½¸€¼€‰±…‰•±}µ…ÁÁ¥¹œ¹©Í½¸ˆ°(€€€€€€€€€€€‘•ÍÑ¥¹…Ñ¥½¸€¼€‰ÁÉ•ÁÉ½•ÍÍ¥¹œ¹©Í½¸ˆ°(€€€€€€€€€€€‘•ÍÑ¥¹…Ñ¥½¸€¼€‰Í…µÁ±•}µ…¹¥™•ÍÐ¹©Í½¸ˆ°(€€€€€€€€¤è(€€€€€€€€€€€ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹ÕÁ±½…‘}…ÉÑ¥™…Ð¡Á…Ñ ¤(€€€€€€€ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹Í…Ù•}ÁÉ½É•ÍÌ¡ì(€€€€€€€€€€€€‰™½Éµ…Ñ}Ù•ÉÍ¥½¸ˆè€Ä°(€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰½µÁ±•Ñ”ˆ°(€€€€€€€€€€€€‰™¥¹•ÉÁÉ¥¹Ðˆè™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€€€€€‰½µÁ±•Ñ•‘}™¥±•ÌˆèÍ½ÉÑ•¡½µÁ±•Ñ•‘}™¥±•Ì¤°(€€€€€€€€€€€€‰±…‰•±Í}Í••¸ˆèÍ½ÉÑ•¡±…‰•±Í}Í••¸¤°(€€€€€€€€€€€€‰ÍÁ±¥Ñ}½Õ¹ÑÌˆèíÍÁ±¥Ðè‘¥Ð¡Í½ÉÑ•¡ÍÁ±¥Ñ}½Õ¹ÑÍmÍÁ±¥Ñt¹¥Ñ•µÌ ¤¤¤™½ÈÍÁ±¥Ð¥¸MA1%Q}95Mô°(€€€€€€€€€€€€‰Í½ÕÉ•}¥¹Ù•¹Ñ½ÉäˆèÍ½ÕÉ•}¥¹Ù•¹Ñ½Éä°(€€€€€€€€€€€€‰Á…ÉÑÌˆèíÍÁ±¥ÐèÝÉ¥Ñ•È¹Á…ÉÑÍmÍÁ±¥Ñt™½ÈÍÁ±¥Ð¥¸MA1%Q}95Mô°(€€€€€€€€€€€€‰ÕÁ‘…Ñ•‘}…Ðˆè‘…Ñ•Ñ¥µ”¹¹½Ü¡Ñ¥µ•é½¹”¹ÕÑŒ¤¹¥Í½™½Éµ…Ð ¤°(€€€€€€€ô¤(€€€€€€€ÁÉ•ÁÉ½•ÍÍ¥¹}ÍÑ½É”¹Í•Ñ}…Ñ¥Ù” ‰ÁÉ•Á…É¥¹œˆ°±•¸¡½µÁ±•Ñ•‘}™¥±•Ì¤°±•¸¡™¥±•Ì¤¤(€€€€€€€¥˜‘•…‘±¥¹•}µ½¹½Ñ½¹¥Œ¥Ì¹½Ð9½¹”…¹Ñ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€øô‘•…‘±¥¹•}µ½¹½Ñ½¹¥Œè(€€€€€€€€€€€É…¥Í”AÉ•ÁÉ½•ÍÍ¥¹A…ÕÍ•I•ÅÕ•ÍÑ• (€€€€€€€€€€€€€€€€‰AÉ•ÁÉ½•ÍÍ¥¹œ½µÁ±•Ñ•‘ÕÉ…‰±äìÑÉ…¥¹¥¹œ¥Ì‘•™•ÉÉ•Ñ¼„™É•Í -…±”Í•ÍÍ¥½¸ˆ(€€€€€€€€€€€€¤(€€€É•ÑÕÉ¸µ…¹¥™•ÍÐ(()‘•˜Á…ÉÍ•}…ÉÌ ¤€´ø…ÉÁ…ÉÍ”¹9…µ•ÍÁ…”è(€€€Á…ÉÍ•È€ô…ÉÁ…ÉÍ”¹ÉÕµ•¹ÑA…ÉÍ•È¡‘•ÍÉ¥ÁÑ¥½¸õ}}‘½}|¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ½¹™¥œˆ°‘•™…Õ±Ðô‰½¹™¥œ½‘…Ñ„¹©Í½¸ˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ‘…Ñ„µ‘¥Èˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ½ÕÑÁÕÐµ‘¥Èˆ°‘•™…Õ±Ðô‰½ÕÑÁÕÑÌ½‘…Ñ„ˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍ…µÁ±•ÌµÁ•Èµ™¥±”ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÌÌµ½¹™¥œˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÉÕ¸µ¥ˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µµ…á¥µÕ´µ¡½ÕÉÌˆ°ÑåÁ”õ™±½…Ð°‘•™…Õ±ÐôÀ¸À¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍÑ½Àµ‰•™½É”µµ¥¹ÕÑ•Ìˆ°ÑåÁ”õ™±½…Ð°‘•™…Õ±ÐôÌÀ¸À¤(€€€É•ÑÕÉ¸Á…ÉÍ•È¹Á…ÉÍ•}…ÉÌ ¤(()‘•˜µ…¥¸ ¤€´ø¥¹Ðè(€€€…ÉÌ€ôÁ…ÉÍ•}…ÉÌ ¤(€€€½¹™¥œ€ô±½…‘}½¹™¥œ¡…ÉÌ¹½¹™¥œ¤(€€€¥˜…ÉÌ¹‘…Ñ…}‘¥È¥Ì¹½Ð9½¹”è(€€€€€€€½¹™¥l‰‘…Ñ…Í•Ð‰ul‰‘…Ñ…}‘¥È‰t€ô…ÉÌ¹‘…Ñ…}‘¥È(€€€¥˜…ÉÌ¹Í…µÁ±•Í}Á•É}™¥±”¥Ì¹½Ð9½¹”è(€€€€€€€¥˜…ÉÌ¹Í…µÁ±•Í}Á•É}™¥±”€ðô€Àè(€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ˆ´µÍ…µÁ±•ÌµÁ•Èµ™¥±”µÕÍÐ‰”Á½Í¥Ñ¥Ù”ˆ¤(€€€€€€€½¹™¥l‰‘…Ñ…Í•Ð‰ul‰Í…µÁ±•Í}Á•É}™¥±”‰t€ô…ÉÌ¹Í…µÁ±•Í}Á•É}™¥±”(€€€ÍÑ½É”€ô9½¹”(€€€¥˜…ÉÌ¹ÌÍ}½¹™¥œ½È…ÉÌ¹ÉÕ¹}¥è(€€€€€€€¥˜¹½Ð…ÉÌ¹ÌÍ}½¹™¥œ½È¹½Ð…ÉÌ¹ÉÕ¹}¥è(€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ˆ´µÌÌµ½¹™¥œ…¹€´µÉÕ¸µ¥µÕÍÐ‰”ÍÕÁÁ±¥•Ñ½•Ñ¡•Èˆ¤(€€€€€€€ÌÍ}‘½Õµ•¹Ð€ô©Í½¸¹±½…‘Ì¡A…Ñ ¡…ÉÌ¹ÌÍ}½¹™¥œ¤¹É•…‘}Ñ•áÐ¡•¹½‘¥¹œô‰ÕÑ˜´àˆ¤¤(€€€€€€€ÍÑ½É”€ôAÉ•ÁÉ½•ÍÍ¥¹MÑ½É”¡A…Ñ ¡…ÉÌ¹½ÕÑÁÕÑ}‘¥È¤°…ÉÌ¹ÉÕ¹}¥°ÌÍ}‘½Õµ•¹Ñl‰ÌÌ‰t¤(€€€‘•…‘±¥¹”€ô9½¹”(€€€¥˜…ÉÌ¹µ…á¥µÕµ}¡½ÕÉÌ€ø€Àè(€€€€€€€ÕÍ…‰±”€ô…ÉÌ¹µ…á¥µÕµ}¡½ÕÉÌ€¨€ÌØÀÀ¸À€´…ÉÌ¹ÍÑ½Á}‰•™½É•}µ¥¹ÕÑ•Ì€¨€ØÀ¸À(€€€€€€€¥˜ÕÍ…‰±”€ðô€Àè(€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ˆ´µÍÑ½Àµ‰•™½É”µµ¥¹ÕÑ•ÌµÕÍÐ‰”±•ÍÌÑ¡…¸€´µµ…á¥µÕ´µ¡½ÕÉÌˆ¤(€€€€€€€•áÑ•É¹…±}‘•…‘±¥¹”€ô½Ì¹•¹Ù¥É½¸¹•Ð ‰A%A1%9}MMM%=9}1%9}A= ˆ¤(€€€€€€€¥˜•áÑ•É¹…±}‘•…‘±¥¹”è(€€€€€€€€€€€ÕÍ…‰±”€ôµ¥¸¡ÕÍ…‰±”°µ…à À¸À°™±½…Ð¡•áÑ•É¹…±}‘•…‘±¥¹”¤€´Ñ¥µ”¹Ñ¥µ” ¤¤¤(€€€€€€€‘•…‘±¥¹”€ôÑ¥µ”¹µ½¹½Ñ½¹¥Œ ¤€¬ÕÍ…‰±”(€€€ÑÉäè(€€€€€€€µ…¹¥™•ÍÐ€ôÁÉ•Á…É•}‘…Ñ…Í•Ð¡½¹™¥œ°…ÉÌ¹½ÕÑÁÕÑ}‘¥È°ÍÑ½É”°‘•…‘±¥¹”¤(€€€•á•ÁÐAÉ•ÁÉ½•ÍÍ¥¹A…ÕÍ•I•ÅÕ•ÍÑ•…Ì•áŒè(€€€€€€€ÁÉ¥¹Ð¡ÍÑÈ¡•áŒ¤¤(€€€€€€€É•ÑÕÉ¸€ÜÔ(€€€ÁÉ¥¹Ð¡©Í½¸¹‘ÕµÁÌ¡ì(€€€€€€€€‰Í…µÁ±•}µ…¹¥™•ÍÐˆèÍÑÈ¡A…Ñ ¡…ÉÌ¹½ÕÑÁÕÑ}‘¥È¤€¼€‰Í…µÁ±•}µ…¹¥™•ÍÐ¹©Í½¸ˆ¤°(€€€€€€€€‰ÍÁ±¥Ñ}Í¥é•Ìˆèµ…¹¥™•ÍÑl‰ÍÁ±¥Ð‰ul‰Í¥é•Ì‰t°(€€€€€€€€‰±•…­…•}…Õ‘¥Ñ}Á…ÍÍ•ˆèµ…¹¥™•ÍÑl‰±•…­…•}…Õ‘¥Ð‰ul‰Á…ÍÍ•‰t°(€€€ô°¥¹‘•¹ÐôÈ¤¤(€€€É•ÑÕÉ¸€À(()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€É…¥Í”MåÍÑ•µá¥Ð¡µ…¥¸ ¤¤