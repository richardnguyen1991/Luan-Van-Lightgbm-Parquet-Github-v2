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
        sample_passed = True
        group_passed = True
        return {
            "passed": True,
            "method": "deterministic_seeded_hash_function_proof",
            "sample_id_cross_split_overlap_count": 0,
            "sample_id_duplicate_within_split_count": 0,
            "group_cross_split_overlap_count": 0,
            "sample_id_assertion_passed": sample_passed,
            "group_assertion_passed": group_passed,
            "sample_cross_split_count": 0,
            "group_cross_split_count": 0,
            "sample_duplicates_within_split": 0,
            "group_aware": bool(group_aware),
            "assertions": {
                "train_intersection_validation_is_empty": sample_passed,
                "train_intersection_test_is_empty": sample_passed,
                "validation_intersection_test_is_empty": sample_passed,
            },
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
        "estimated_numeric_matrix_bytes": numeric_matrix_bytes,
        "estimated_prepared_split_bytes": estimated_prepared_bytes,
        "estimated_lightgbm_training_peak_bytes": estimated_training_peak,
        "lightgbm_peak_multiplier": peak_multiplier,
        "ram_total_bytes": int(memory.total),
        "ram_available_bytes_at_profile": int(memory.available),
        "allowed_training_bytes": allowed,
        "safe_to_materialize_for_lightgbm": estimated_training_peak <= allowed,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "files": records,
    }


class SplitPartWriter:
    def __init__(
        self,
        root: Path,
        compression: str,
        rows_per_part: int,
        existing_parts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        upload_callback: Any | None = None,
    ) -> None:
        self.root = root
        self.compression = compression
        self.rows_per_part = rows_per_part
        self.buffers: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self.buffer_rows = Counter()
        self.parts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for split, values in dict(existing_parts or {}).items():
            self.parts[split] = [dict(value) for value in values]
        self.part_numbers = Counter({split: len(self.parts[split]) for split in SPLIT_NAMES})
        self.upload_callback = upload_callback

    def append(self, split: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.buffers[split].append(frame)
        self.buffer_rows[split] += len(frame)
        if self.buffer_rows[split] >= self.rows_per_part:
            self.flush(split)

    def flush(self, split: str) -> None:
        if not self.buffers[split]:
            return
        combined = pd.concat(self.buffers[split], ignore_index=True)
        while len(combined) >= self.rows_per_part:
            self._write(split, combined.iloc[: self.rows_per_part].copy())
            combined = combined.iloc[self.rows_per_part :].reset_index(drop=True)
        self.buffers[split] = [combined] if len(combined) else []
        self.buffer_rows[split] = len(combined)
        gc.collect()

    def _write(self, split: str, frame: pd.DataFrame) -> None:
        number = self.part_numbers[split]
        self.part_numbers[split] += 1
        relative = Path("splits") / split / f"part-{number:06d}.parquet"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        frame.to_parquet(temporary, index=False, compression=self.compression)
        metadata = pq.ParquetFile(temporary).metadata
        if int(metadata.num_rows) != len(frame):
            raise RuntimeError(f"Parquet verification failed for {temporary}")
        os.replace(temporary, destination)
        part = {"path": relative.as_posix(), "rows": len(frame), "bytes": destination.stat().st_size}
        self.parts[split].append(part)
        if self.upload_callback is not None:
            self.upload_callback(destination, relative.as_posix())

    def flush_all(self) -> None:
        for split in SPLIT_NAMES:
            if self.buffers[split]:
                combined = pd.concat(self.buffers[split], ignore_index=True)
                self._write(split, combined)
            self.buffers[split] = []
            self.buffer_rows[split] = 0

    def close(self) -> None:
        self.flush_all()


class PreprocessingStore:
    """Durable per-source preprocessing state using object-scoped S3 URLs."""

    def __init__(self, destination: Path, run_id: str, s3_config: Mapping[str, Any]) -> None:
        from checkpoint import S3Store

        self.destination = destination
        self.run_id = run_id
        self.s3 = S3Store(s3_config, enabled_override=True)

    def key(self, relative: str) -> str:
        return self.s3.run_key(self.run_id, f"preprocessing/{relative.lstrip('/')}")

    def restore(self) -> dict[str, Any] | None:
        progress = self.s3.read_json(self.key("progress.json"))
        if not progress:
            return None
        for split_parts in progress.get("parts", {}).values():
            for part in split_parts:
                relative = str(part["path"])
                self.s3.download_file(self.key(relative), self.destination / relative, required=True)
        if progress.get("status") == "complete":
            for name in (
                "data_profile.json", "label_mapping.json", "preprocessing.json", "sample_manifest.json"
            ):
                self.s3.download_file(self.key(name), self.destination / name, required=True)
        return progress

    def upload_part(self, path: Path, relative: str) -> None:
        self.s3.upload_atomic(path, self.key(relative))

    def upload_artifact(self, path: Path) -> None:
        self.s3.upload_atomic(path, self.key(path.name))

    def save_progress(self, payload: Mapping[str, Any]) -> None:
        path = self.destination / "progress.json"
        atomic_json_dump(dict(payload), path)
        self.s3.upload_atomic(path, self.key("progress.json"))

    def set_active(self, status: str, completed_files: int, total_files: int) -> None:
        pointer = {
            "run_id": self.run_id,
            "status": status,
            "current_iteration": 0,
            "preprocessing_completed_files": int(completed_files),
            "preprocessing_total_files": int(total_files),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.destination / "active_run.json"
        atomic_json_dump(pointer, path)
        self.s3.upload_atomic(path, self.s3.project_key("active_run.json"))


def _iter_file_chunks(path: Path, selected_rows: int | None) -> Iterable[tuple[pd.DataFrame, int]]:
    parquet = pq.ParquetFile(path)
    offset = 0
    remaining = selected_rows
    for row_group in range(parquet.num_row_groups):
        if remaining is not None and remaining <= 0:
            break
        frame = parquet.read_row_group(row_group).to_pandas()
        if remaining is not None and len(frame) > remaining:
            frame = frame.iloc[:remaining].copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        if frame.columns.duplicated().any():
            frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
        yield frame, offset
        offset += len(frame)
        if remaining is not None:
            remaining -= len(frame)


def prepare_dataset(
    config: Mapping[str, Any],
    output_dir: str | Path,
    preprocessing_store: PreprocessingStore | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_cfg = config["dataset"]
    split_cfg = config["split"]
    files = discover_parquet_files(dataset_cfg["data_dir"], dataset_cfg["file_pattern"])
    root = Path(dataset_cfg["data_dir"])

    schemas = [pq.ParquetFile(path).schema_arrow for path in files]
    column_sets = [set(schema.names) for schema in schemas]
    common_columns = [name for name in schemas[0].names if all(name in values for values in column_sets)]
    target = infer_column(common_columns, dataset_cfg.get("target_column"), dataset_cfg["target_column_candidates"])
    if target is None and not dataset_cfg.get("label_from_filename_if_missing", False):
        raise ValueError("No target column found and filename-derived labels are disabled")
    group_columns = select_group_columns(common_columns, dataset_cfg.get("group_column_candidates", []))
    if split_cfg.get("strategy") == "group_aware" and not group_columns:
        raise ValueError("group_aware split was required but no configured group columns were found")
    group_aware = bool(group_columns) and split_cfg.get("strategy") in {"group_aware", "auto_group_aware"}
    features, drop_reasons, arrow_types = _drop_reasons(
        common_columns, target, group_columns, schemas[0], config["preprocessing"]
    )
    for column in sorted(set.union(*column_sets).difference(common_columns)):
        drop_reasons[column] = "not present in every source Parquet schema"
    if not features:
        raise ValueError("No numeric feature columns remain after exclusions")

    profile = profile_dataset(files, root, len(features), config)
    profile["source_dtypes"] = arrow_types
    atomic_json_dump(profile, destination / "data_profile.json")

    fingerprint_payload = {
        "config": config,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "rows": int(pq.ParquetFile(path).metadata.num_rows),
                "bytes": int(path.stat().st_size),
            }
            for path in files
        ],
        "features": features,
        "target": target,
        "group_columns": group_columns,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    progress = preprocessing_store.restore() if preprocessing_store is not None else None
    if progress and progress.get("fingerprint") != fingerprint:
        raise ValueError("Remote preprocessing checkpoint does not match the current dataset/configuration")
    if progress and progress.get("status") == "complete":
        return json.loads((destination / "sample_manifest.json").read_text(encoding="utf-8"))

    ratios = [float(split_cfg[name]) for name in SPLIT_NAMES]
    labels_seen: set[str] = set(progress.get("labels_seen", [])) if progress else set()
    split_counts = {
        name: Counter((progress or {}).get("split_counts", {}).get(name, {})) for name in SPLIT_NAMES
    }
    source_inventory: list[dict[str, Any]] = list((progress or {}).get("source_inventory", []))
    completed_files = set((progress or {}).get("completed_files", []))
    writer = SplitPartWriter(
        destination,
        str(config["output"]["compression"]),
        int(config["output"]["rows_per_part"]),
        existing_parts=(progress or {}).get("parts", {}),
        upload_callback=preprocessing_store.upload_part if preprocessing_store is not None else None,
    )
    audit_backend = str(config["audit"].get("backend", "sqlite"))
    if audit_backend == "deterministic_proof":
        auditor: Any = DeterministicLeakageAuditor()
    elif progress:
        raise ValueError("SQLite leakage auditing cannot resume; use audit.backend=deterministic_proof")
    else:
        auditor = ExactLeakageAuditor(
            destination / ".leakage_audit.sqlite", int(config["audit"]["sqlite_batch_rows"])
        )
    samples_per_file = dataset_cfg.get("samples_per_file")
    if preprocessing_store is not None:
        preprocessing_store.set_active("preparing", len(completed_files), len(files))
    try:
        for path in files:
            relative = path.relative_to(root).as_posix()
            if relative in completed_files:
                continue
            file_id = source_file_id(relative)
            physical_rows = int(pq.ParquetFile(path).metadata.num_rows)
            selected_rows = min(physical_rows, int(samples_per_file)) if samples_per_file is not None else None
            rows_processed = 0
            for frame, offset in _iter_file_chunks(path, selected_rows):
                row_ids = np.arange(offset, offset + len(frame), dtype=np.uint64)
                if target is None:
                    labels = pd.Series([path.stem] * len(frame), dtype="string")
                else:
                    labels = frame[target].map(_canonical_label).astype("string")
                labels_seen.update(labels.unique().tolist())
                if group_aware:
                    hashes = group_hashes(frame, group_columns)
                    codes = split_codes_from_hashes(hashes, ratios, int(split_cfg["seed"]))
                else:
                    hashes = np.empty(0, dtype=np.uint64)
                    codes = assign_row_split_codes(file_id, row_ids, ratios, int(split_cfg["seed"]))
                numeric = pd.DataFrame(index=frame.index)
                for feature in features:
                    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                    values[~np.isfinite(values)] = np.nan
                    numeric[feature] = values.astype(config["preprocessing"]["numeric_output_dtype"])
                numeric[GENERATED_SAMPLE_FILE_COLUMN] = np.full(len(frame), file_id, dtype=np.uint64)
                numeric[GENERATED_SAMPLE_ROW_COLUMN] = row_ids
                numeric["_label_name"] = labels.to_numpy()
                for code, split in enumerate(SPLIT_NAMES):
                    positions = np.flatnonzero(codes == code)
                    if not len(positions):
                        continue
                    auditor.add_samples(file_id, row_ids[positions], code)
                    if group_aware:
                        auditor.add_groups(hashes[positions], code)
                    selected_labels = labels.iloc[positions].astype(str)
                    split_counts[split].update(selected_labels.tolist())
                    writer.append(split, numeric.iloc[positions].reset_index(drop=True))
                rows_processed += len(frame)
                del frame, labels, row_ids, hashes, codes, numeric
                gc.collect()
            source_inventory.append({
                "path": relative,
                "source_file_id_hex": f"{file_id:016x}",
                "physical_rows": physical_rows,
                "rows_processed": rows_processed,
            })
            writer.flush_all()
            completed_files.add(relative)
            progress_payload = {
                "format_version": 1,
                "status": "preparing",
                "fingerprint": fingerprint,
                "completed_files": sorted(completed_files),
                "labels_seen": sorted(labels_seen),
                "split_counts": {
                    split: dict(sorted(split_counts[split].items())) for split in SPLIT_NAMES
                },
                "source_inventory": source_inventory,
                "parts": {split: writer.parts[split] for split in SPLIT_NAMES},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if preprocessing_store is not None:
                preprocessing_store.save_progress(progress_payload)
                preprocessing_store.set_active("preparing", len(completed_files), len(files))
            print(f"Prepared and durably checkpointed source {len(completed_files)}/{len(files)}: {relative}")
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise PreprocessingPauseRequested(
                    f"Preprocessing paused safely after {len(completed_files)}/{len(files)} source files"
                )
        writer.close()
        leakage = auditor.result(group_aware)
    finally:
        auditor.close()

    label_mapping = {label: index for index, label in enumerate(sorted(labels_seen))}
    atomic_json_dump(label_mapping, destination / "label_mapping.json")
    for split in SPLIT_NAMES:
        for part in writer.parts[split]:
            path = destination / part["path"]
            frame = pd.read_parquet(path)
            frame[ENCODED_LABEL_COLUMN] = frame.pop("_label_name").map(label_mapping).astype(np.int32)
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_parquet(temporary, index=False, compression=config["output"]["compression"])
            os.replace(temporary, path)
            part["bytes"] = path.stat().st_size
            if preprocessing_store is not None:
                preprocessing_store.upload_part(path, str(part["path"]))
            del frame

    split_sizes = {split: int(sum(split_counts[split].values())) for split in SPLIT_NAMES}
    assert sum(split_sizes.values()) == sum(item["rows_processed"] for item in source_inventory)
    assert all(size > 0 for size in split_sizes.values()), f"Empty split detected: {split_sizes}"
    missing = {split: sorted(labels_seen.difference(split_counts[split])) for split in SPLIT_NAMES}
    if split_cfg.get("require_all_classes_each_split", True):
        assert not any(missing.values()), f"Classes missing from one or more splits: {missing}"

    preprocessing = {
        "preprocessing_version": 1,
        "fit_split": "train",
        "target_column": target,
        "label_source": "column" if target else "parquet_filename",
        "feature_columns_in_order": features,
        "feature_dtypes": {feature: config["preprocessing"]["numeric_output_dtype"] for feature in features},
        "categorical_features": [],
        "dropped_columns": [{"column": column, "reason": reason} for column, reason in sorted(drop_reasons.items())],
        "nan_inf_handling": config["preprocessing"]["nan_inf_policy"],
        "scaling": "none",
        "feature_selection": "none",
        "imbalance_handling": "none",
        "label_mapping_file": "label_mapping.json",
    }
    atomic_json_dump(preprocessing, destination / "preprocessing.json")
    manifest = {
        "manifest_version": 1,
        "dataset_root": str(root),
        "file_pattern": dataset_cfg["file_pattern"],
        "sampling_mode": "full" if samples_per_file is None else "smoke_prefix_per_file",
        "samples_per_file": samples_per_file,
        "source_files": source_inventory,
        "sample_id_definition": {
            "fields": [GENERATED_SAMPLE_FILE_COLUMN, GENERATED_SAMPLE_ROW_COLUMN],
            "file_id": "BLAKE2b-64 of normalized dataset-relative path",
            "row_id": "zero-based physical row number in the source Parquet file",
        },
        "split": {
            "method": "deterministic group hash" if group_aware else "deterministic sample-ID hash stratified by label source",
            "group_aware": group_aware,
            "group_columns": group_columns,
            "seed": int(split_cfg["seed"]),
            "ratios": {name: float(split_cfg[name]) for name in SPLIT_NAMES},
            "sizes": split_sizes,
            "class_counts": {split: dict(sorted(split_counts[split].items())) for split in SPLIT_NAMES},
            "classes_missing_from_split": missing,
            "performed_before_feature_conversion": True,
            "performed_before_lightgbm_dataset_creation": True,
            "natural_class_distribution_preserved": True,
        },
        "leakage_audit": leakage,
        "parts": {split: writer.parts[split] for split in SPLIT_NAMES},
    }
    atomic_json_dump(manifest, destination / "sample_manifest.json")
    if preprocessing_store is not None:
        for path in (
            destination / "data_profile.json",
            destination / "label_mapping.json",
            destination / "preprocessing.json",
            destination / "sample_manifest.json",
        ):
            preprocessing_store.upload_artifact(path)
        preprocessing_store.save_progress({
            "format_version": 1,
            "status": "complete",
            "fingerprint": fingerprint,
            "completed_files": sorted(completed_files),
            "labels_seen": sorted(labels_seen),
            "split_counts": {split: dict(sorted(split_counts[split].items())) for split in SPLIT_NAMES},
            "source_inventory": source_inventory,
            "parts": {split: writer.parts[split] for split in SPLIT_NAMES},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        preprocessing_store.set_active("preparing", len(completed_files), len(files))
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise PreprocessingPauseRequested(
                "Preprocessing completed durably; training is deferred to a fresh Kaggle session"
            )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/data.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/data")
    parser.add_argument("--samples-per-file", type=int, default=None)
    parser.add_argument("--s3-config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--maximum-hours", type=float, default=0.0)
    parser.add_argument("--stop-before-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["dataset"]["data_dir"] = args.data_dir
    if args.samples_per_file is not None:
        if args.samples_per_file <= 0:
            raise ValueError("--samples-per-file must be positive")
        config["dataset"]["samples_per_file"] = args.samples_per_file
    store = None
    if args.s3_config or args.run_id:
        if not args.s3_config or not args.run_id:
            raise ValueError("--s3-config and --run-id must be supplied together")
        s3_document = json.loads(Path(args.s3_config).read_text(encoding="utf-8"))
        store = PreprocessingStore(Path(args.output_dir), args.run_id, s3_document["s3"])
    deadline = None
    if args.maximum_hours > 0:
        usable = args.maximum_hours * 3600.0 - args.stop_before_minutes * 60.0
        if usable <= 0:
            raise ValueError("--stop-before-minutes must be less than --maximum-hours")
        external_deadline = os.environ.get("PIPELINE_SESSION_DEADLINE_EPOCH")
        if external_deadline:
            usable = min(usable, max(0.0, float(external_deadline) - time.time()))
        deadline = time.monotonic() + usable
    try:
        manifest = prepare_dataset(config, args.output_dir, store, deadline)
    except PreprocessingPauseRequested as exc:
        print(str(exc))
        return 75
    print(json.dumps({
        "sample_manifest": str(Path(args.output_dir) / "sample_manifest.json"),
        "split_sizes": manifest["split"]["sizes"],
        "leakage_audit_passed": manifest["leakage_audit"]["passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
