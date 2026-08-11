"""LightGBM Dataset construction, baseline parameters, metric, and callbacks."""

from __future__ import annotations

import gc
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from checkpoint import canonical_hash


LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")
INTERNAL_COLUMNS = ("_sample_file_id", "_sample_row_id", "_label")


class TrainingPauseRequested(RuntimeError):
    """Raised only after a durable checkpoint requests a new Kaggle session."""


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_training_config(config: Mapping[str, Any]) -> None:
    if int(config["num_boost_round"]) != 100:
        raise ValueError("Baseline num_boost_round must be exactly 100")
    if bool(config["early_stopping"]):
        raise ValueError("Early stopping is forbidden for the baseline")
    if config["device"] != "cpu":
        raise ValueError("The baseline must run on CPU")
    if config["imbalance_handling"] != "none" or config["feature_selection"] != "none":
        raise ValueError("Baseline imbalance handling and feature selection must both be 'none'")
    if not bool(config["use_all_train_rows"]):
        raise ValueError("The baseline must use every row in the train split")
    params = config["model_params"]
    required_exact = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "learning_rate": 0.001,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 0.1,
        "min_gain_to_split": 0.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "device_type": "cpu",
        "deterministic": True,
        "histogram_pool_size": 128.0,
        "use_quantized_grad": True,
        "num_grad_quant_bins": 16,
        "quant_train_renew_leaf": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": params.get(key)}
        for key, expected in required_exact.items()
        if params.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Baseline parameter contract violated: {mismatches}")
    if bool(params.get("force_col_wise")) == bool(params.get("force_row_wise")):
        raise ValueError("Exactly one of force_col_wise and force_row_wise must be true")
    if int(params.get("num_threads", 0)) <= 0:
        raise ValueError("model_params.num_threads must be a positive CPU thread count")
    for seed_name in ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
        if seed_name not in params or not isinstance(params[seed_name], int):
            raise ValueError(f"model_params.{seed_name} must be an integer")
    forbidden = {"class_weight", "scale_pos_weight", "is_unbalance", "sample_weight"}
    present = sorted(forbidden.intersection(params))
    if present:
        raise ValueError(f"Imbalance-handling parameters are forbidden: {present}")
    metrics = set(params.get("metric", []))
    if metrics != {"multi_logloss", "multi_error"}:
        raise ValueError("Configured metrics must be exactly multi_logloss and multi_error")
    if int(config["checkpoint"]["interval_rounds"]) != 10:
        raise ValueError("checkpoint.interval_rounds must be exactly 10")


def effective_model_params(config: Mapping[str, Any], num_classes: int) -> dict[str, Any]:
    validate_training_config(config)
    if num_classes <= 1:
        raise ValueError("Multiclass LightGBM requires at least two classes")
    params = dict(config["model_params"])
    params["num_class"] = int(num_classes)
    return params


@dataclass
class DatasetBundle:
    train_dataset: Any
    validation_dataset: Any
    test_dataset: Any
    features: dict[str, Any]
    labels: dict[str, np.ndarray]
    feature_names: list[str]
    model_feature_names: list[str]
    label_mapping: dict[str, int]
    params: dict[str, Any]
    params_hash: str
    feature_schema_hash: str


class _ILocIndexer:
    def __init__(self, owner: "LazyParquetFeatures") -> None:
        self.owner = owner

    def __getitem__(self, index: Any) -> pd.DataFrame | pd.Series:
        return self.owner.read(index, as_frame=True)


class LazyParquetFeatures:
    """DataFrame-like, row-addressable view over prepared Parquet parts."""

    def __init__(
        self, prepared_dir: Path, parts: Sequence[Mapping[str, Any]], feature_names: Sequence[str],
        output_feature_names: Sequence[str] | None = None,
    ) -> None:
        if not parts:
            raise ValueError("Prepared split contains no Parquet parts")
        self.prepared_dir = prepared_dir
        self.parts = [dict(part) for part in parts]
        self.feature_names = list(feature_names)
        self.output_feature_names = list(output_feature_names or feature_names)
        if len(self.output_feature_names) != len(self.feature_names):
            raise ValueError("Input and output feature-name counts differ")
        self._lengths = np.asarray([int(part["rows"]) for part in self.parts], dtype=np.int64)
        self._offsets = np.concatenate(([0], np.cumsum(self._lengths)))
        self.iloc = _ILocIndexer(self)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self.feature_names)

    def _read_part(self, part_index: int) -> pd.DataFrame:
        frame = pd.read_parquet(
            self.prepared_dir / str(self.parts[part_index]["path"]), columns=self.feature_names
        )
        if list(frame.columns) != self.feature_names:
            raise AssertionError("Feature order does not match preprocessing.json")
        return frame.astype(np.float32)

    def read(self, index: Any, as_frame: bool = False) -> Any:
        scalar = isinstance(index, (int, np.integer))
        if scalar:
            positions = np.asarray([int(index)], dtype=np.int64)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            positions = np.arange(start, stop, step, dtype=np.int64)
        else:
            positions = np.asarray(index, dtype=np.int64).reshape(-1)
            positions = np.where(positions < 0, positions + len(self), positions)
        if np.any((positions < 0) | (positions >= len(self))):
            raise IndexError("Parquet row index is out of range")
        if not len(positions):
            empty = pd.DataFrame(columns=self.output_feature_names, dtype=np.float32)
            return empty if as_frame else empty.to_numpy(dtype=np.float32)

        result = np.empty((len(positions), len(self.feature_names)), dtype=np.float32)
        part_indices = np.searchsorted(self._offsets[1:], positions, side="right")
        for part_index in np.unique(part_indices):
            output_positions = np.flatnonzero(part_indices == part_index)
            local_positions = positions[output_positions] - self._offsets[part_index]
            frame = self._read_part(int(part_index))
            result[output_positions] = frame.iloc[local_positions].to_numpy(dtype=np.float32, copy=False)
        if scalar:
            if as_frame:
                return pd.Series(result[0], index=self.output_feature_names)
            return result[0]
        if as_frame:
            return pd.DataFrame(result, columns=self.output_feature_names)
        return result


def lightgbm_safe_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Create stable, unique ASCII names accepted by LightGBM's JSON serializer."""
    safe = []
    for index, original in enumerate(feature_names):
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(original)).strip("_") or "feature"
        safe.append(f"f{index:04d}_{slug[:96]}")
    if len(set(safe)) != len(safe):
        raise AssertionError("Generated LightGBM feature names are not unique")
    return safe


def _sequence_for_part(lgb: Any, prepared_dir: Path, part: Mapping[str, Any], feature_names: Sequence[str], batch_size: int) -> Any:
    class ParquetPartSequence(lgb.Sequence):
        def __init__(self) -> None:
            self.path = prepared_dir / str(part["path"])
            self.rows = int(part["rows"])
            self.columns = list(feature_names)
            self.batch_size = int(batch_size)
            self._cache: pd.DataFrame | None = None

        def __len__(self) -> int:
            return self.rows

        def __getitem__(self, index: Any) -> np.ndarray:
            if self._cache is None:
                self._cache = pd.read_parquet(self.path, columns=self.columns).astype(np.float32)
                if len(self._cache) != self.rows or list(self._cache.columns) != self.columns:
                    raise AssertionError(f"Prepared Parquet metadata mismatch: {self.path}")
            selected = self._cache.iloc[index]
            # LightGBM's random Sequence sampler requires double precision rows.
            # Sequential construction accepts float32, halving the transient batch.
            dtype = np.float32 if isinstance(index, slice) else np.float64
            values = selected.to_numpy(dtype=dtype, copy=True)
            release = not isinstance(index, slice) or index.stop is None or int(index.stop) >= self.rows
            if release:
                self._cache = None
                gc.collect()
                if sys.platform.startswith("linux"):
                    try:
                        import ctypes
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except (ImportError, OSError):
                        pass
            return values

    return ParquetPartSequence()


def _read_split_labels(prepared_dir: Path, parts: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not parts:
        raise ValueError("Prepared split contains no Parquet parts")
    labels = np.empty(sum(int(part["rows"]) for part in parts), dtype=np.int32)
    offset = 0
    for part in parts:
        values = pd.read_parquet(
            prepared_dir / str(part["path"]), columns=["_label"]
        )["_label"].to_numpy(dtype=np.int32, copy=False)
        expected_rows = int(part["rows"])
        if len(values) != expected_rows:
            raise AssertionError(f"Prepared label metadata mismatch: {part['path']}")
        labels[offset:offset + expected_rows] = values
        offset += expected_rows
    return labels


def build_datasets(prepared_data_dir: str | Path, config: Mapping[str, Any]) -> DatasetBundle:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Install lightgbm>=4.0,<5 before training") from exc
    prepared = Path(prepared_data_dir)
    manifest = _read_json(prepared / "sample_manifest.json")
    preprocessing = _read_json(prepared / "preprocessing.json")
    label_mapping = {str(key): int(value) for key, value in _read_json(prepared / "label_mapping.json").items()}
    profile = _read_json(prepared / "data_profile.json")
    feature_names = list(preprocessing["feature_columns_in_order"])
    model_feature_names = lightgbm_safe_feature_names(feature_names)
    if preprocessing.get("scaling") != "none" or preprocessing.get("imbalance_handling") != "none":
        raise ValueError("Prepared data violates the unscaled/unbalanced baseline contract")
    batch_size = int(config["dataset"].get("sequence_batch_rows", 8192))
    if batch_size <= 0:
        raise ValueError("dataset.sequence_batch_rows must be positive")
    if not profile["safe_to_materialize_for_lightgbm"]:
        LOGGER.info(
            "Raw full-matrix materialization is unsafe; constructing LightGBM Datasets from "
            "Parquet Sequences with %d rows per batch", batch_size,
        )
    features: dict[str, LazyParquetFeatures] = {}
    labels: dict[str, np.ndarray] = {}
    sequences: dict[str, list[Any]] = {}
    for split in SPLIT_NAMES:
        parts = manifest["parts"][split]
        features[split] = LazyParquetFeatures(prepared, parts, feature_names, model_feature_names)
        labels[split] = _read_split_labels(prepared, parts)
        sequences[split] = [
            _sequence_for_part(lgb, prepared, part, feature_names, batch_size) for part in parts
        ]
    observed_set: set[int] = set()
    for split in SPLIT_NAMES:
        observed_set.update(int(value) for value in np.unique(labels[split]))
    observed = sorted(observed_set)
    expected = list(range(len(label_mapping)))
    if observed != expected:
        raise AssertionError(f"label_mapping.json indices {expected} do not match observed labels {observed}")
    for split in SPLIT_NAMES:
        if len(features[split]) != int(manifest["split"]["sizes"][split]):
            raise AssertionError(f"Prepared {split} row count disagrees with sample_manifest.json")
    params = effective_model_params(config, len(label_mapping))
    free_raw = bool(config["dataset"].get("free_raw_data", True))
    train_dataset = lgb.Dataset(
        sequences["train"], label=labels["train"], feature_name=model_feature_names,
        categorical_feature=[], free_raw_data=free_raw,
    )
    validation_dataset = lgb.Dataset(
        sequences["validation"], label=labels["validation"], reference=train_dataset,
        feature_name=model_feature_names, categorical_feature=[], free_raw_data=free_raw,
    )
    test_dataset = lgb.Dataset(
        sequences["test"], label=labels["test"], reference=train_dataset,
        feature_name=model_feature_names, categorical_feature=[], free_raw_data=free_raw,
    )
    schema_payload = {
        "feature_names": feature_names,
        "model_feature_names": model_feature_names,
        "feature_dtypes": preprocessing["feature_dtypes"],
        "categorical_features": preprocessing["categorical_features"],
        "label_mapping": label_mapping,
    }
    return DatasetBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        features=features,
        labels=labels,
        feature_names=feature_names,
        model_feature_names=model_feature_names,
        label_mapping=label_mapping,
        params=params,
        params_hash=canonical_hash(params),
        feature_schema_hash=canonical_hash(schema_payload),
    )


def macro_f1_metric(num_classes: int) -> Callable[[np.ndarray, Any], tuple[str, float, bool]]:
    labels_order = list(range(num_classes))

    def evaluate(predictions: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
        probabilities = np.asarray(predictions)
        labels = np.asarray(dataset.get_label(), dtype=np.int32)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(num_classes, -1).T
        if probabilities.shape != (len(labels), num_classes):
            raise ValueError(
                f"Unexpected multiclass prediction shape {probabilities.shape}; "
                f"expected {(len(labels), num_classes)}"
            )
        predicted = np.argmax(probabilities, axis=1)
        score = f1_score(labels, predicted, labels=labels_order, average="macro", zero_division=0)
        return "macro_f1", float(score), True

    return evaluate


class IterationRecorder:
    """Append-only LightGBM callback with periodic durable checkpoints."""

    order = 50
    before_iteration = False

    def __init__(
        self,
        history: list[dict[str, Any]],
        session_id: str,
        target_iteration: int,
        learning_rate: float,
        checkpoint_interval: int,
        checkpoint_hook: Callable[[Any, list[dict[str, Any]], str], float],
        deadline_monotonic: float | None,
        max_rounds_this_session: int | None,
        session_start_iteration: int,
        maximum_session_hours: float,
        stop_before_minutes: float,
    ) -> None:
        self.history = history
        self.session_id = session_id
        self.target_iteration = int(target_iteration)
        self.learning_rate = float(learning_rate)
        self.checkpoint_interval = int(checkpoint_interval)
        self.checkpoint_hook = checkpoint_hook
        self.deadline_monotonic = deadline_monotonic
        self.max_rounds_this_session = max_rounds_this_session
        self.session_start_iteration = int(session_start_iteration)
        self.maximum_session_hours = float(maximum_session_hours)
        self.stop_before_minutes = float(stop_before_minutes)
        self.last_perf = time.perf_counter()
        self.last_timestamp = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _metrics(evaluation_result_list: Sequence[Sequence[Any]]) -> dict[str, float]:
        return {
            f"{str(item[0])}_{str(item[1])}": float(item[2])
            for item in evaluation_result_list
        }

    def __call__(self, env: Any) -> None:
        now_perf = time.perf_counter()
        now_timestamp = datetime.now(timezone.utc).isoformat()
        current = int(env.model.current_iteration())
        expected = len(self.history) + 1
        if current != expected:
            raise RuntimeError(f"Non-contiguous LightGBM iteration: expected {expected}, observed {current}")
        metrics = self._metrics(env.evaluation_result_list)
        record = {
            "iteration": current,
            "session_id": self.session_id,
            "timestamp_start": self.last_timestamp,
            "timestamp_end": now_timestamp,
            "learning_rate": self.learning_rate,
            "train_multi_logloss": metrics.get("train_multi_logloss"),
            "val_multi_logloss": metrics.get("validation_multi_logloss"),
            "train_multi_error": metrics.get("train_multi_error"),
            "val_multi_error": metrics.get("validation_multi_error"),
            "train_macro_f1": metrics.get("train_macro_f1"),
            "val_macro_f1": metrics.get("validation_macro_f1"),
            "iteration_seconds": now_perf - self.last_perf,
            "checkpoint_seconds": 0.0,
            "is_final_round": current == self.target_iteration,
        }
        required_metrics = [key for key, value in record.items() if key.startswith(("train_", "val_")) and value is None]
        if required_metrics:
            raise RuntimeError(f"LightGBM callback did not receive required metrics: {required_metrics}")
        self.history.append(record)
        self.last_perf = now_perf
        self.last_timestamp = now_timestamp

        time_limit = self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic
        round_limit = (
            self.max_rounds_this_session is not None
            and current - self.session_start_iteration >= int(self.max_rounds_this_session)
        )
        final = current == self.target_iteration
        scheduled = current % self.checkpoint_interval == 0
        should_pause = not final and (time_limit or round_limit)
        if scheduled or final or should_pause:
            status = "ready_for_report" if final else ("paused" if should_pause else "running")
            checkpoint_seconds = float(self.checkpoint_hook(env.model, self.history, status))
            record["checkpoint_seconds"] = checkpoint_seconds
            # Exclude checkpoint/S3 synchronization from the following boosting
            # iteration's duration and from accumulated model-training time.
            self.last_perf = time.perf_counter()
            self.last_timestamp = datetime.now(timezone.utc).isoformat()
            if current == self.checkpoint_interval:
                average_iteration = float(np.mean([item["iteration_seconds"] for item in self.history[:current]]))
                blocks = math.ceil(self.target_iteration / self.checkpoint_interval)
                estimated_total = average_iteration * self.target_iteration + checkpoint_seconds * blocks
                usable_session_seconds = max(1.0, self.maximum_session_hours * 3600 - self.stop_before_minutes * 60)
                estimated_sessions = max(1, math.ceil(estimated_total / usable_session_seconds))
                LOGGER.info(
                    "Round-10 timing: avg_iteration=%.3fs checkpoint_block=%.3fs estimated_total_100=%.1fs estimated_kaggle_sessions=%d",
                    average_iteration, checkpoint_seconds, estimated_total, estimated_sessions,
                )
            if should_pause:
                raise TrainingPauseRequested(f"Session paused safely after iteration {current}")
