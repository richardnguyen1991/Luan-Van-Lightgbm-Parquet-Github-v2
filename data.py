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
from collections import Counter, defaultdict
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
    def __init__(self, root: Path, compression: str, rows_per_part: int) -> None:
        self.root = root
        self.compression = compression
        self.rows_per_part = rows_per_part
        self.buffers: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self.buffer_rows = Counter()
        self.part_numbers = Counter()
        self.parts: dict[str, list[dict[str, Any]]] = defaultdict(list)

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
        self.parts[split].append({"path": relative.as_posix(), "rows": len(frame), "bytes": destination.stat().st_size})

    def close(self) -> None:
        for split in SPLIT_NAMES:
            if self.buffers[split]:
                combined = pd.concat(self.buffers[split], ignore_index=True)
                self._write(split, combined)
            self.buffers[split] = []
            self.buffer_rows[split] = 0


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


def prepare_dataset(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
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

    ratios = [float(split_cfg[name]) for name in SPLIT_NAMES]
    labels_seen: set[str] = set()
    split_counts = {name: Counter() for name in SPLIT_NAMES}
    source_inventory: list[dict[str, Any]] = []
    writer = SplitPartWriter(
        destination, str(config["output"]["compression"]), int(config["output"]["rows_per_part"])
    )
    auditor = ExactLeakageAuditor(
        destination / ".leakage_audit.sqlite", int(config["audit"]["sqlite_batch_rows"])
    )
    samples_per_file = dataset_cfg.get("samples_per_file")
    try:
        for path in files:
            relative = path.relative_to(root).as_posix()
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
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/data.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/data")
    parser.add_argument("--samples-per-file", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["dataset"]["data_dir"] = args.data_dir
    if args.samples_per_file is not None:
        if args.samples_per_file <= 0:
            raise ValueError("--samples-per-file must be positive")
        config["dataset"]["samples_per_file"] = args.samples_per_file
    manifest = prepare_dataset(config, args.output_dir)
    print(json.dumps({
        "sample_manifest": str(Path(args.output_dir) / "sample_manifest.json"),
        "split_sizes": manifest["split"]["sizes"],
        "leakage_audit_passed": manifest["leakage_audit"]["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
