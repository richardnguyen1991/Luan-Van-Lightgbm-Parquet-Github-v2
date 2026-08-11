"""Rebuild LightGBM metrics, explainability tables, and all figures from artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from checkpoint import S3Store, atomic_json_dump, sha256_file
from viz import generate_final_figures


LOGGER = logging.getLogger("make_report")
ArtifactCallback = Callable[[Path, str], None]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_report_config(run_dir: Path) -> dict[str, Any]:
    captured = run_dir / "config" / "report_config.json"
    source = captured if captured.exists() else Path(__file__).resolve().parent / "config" / "report.json"
    return read_json(source)


def ordered_class_names(mapping: Mapping[str, int]) -> list[str]:
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    if sorted(int(value) for value in mapping.values()) != list(range(len(names))):
        raise ValueError("label_mapping.json IDs must be contiguous from zero")
    return names


def notify(path: Path, category: str, callback: ArtifactCallback | None) -> None:
    if callback:
        callback(path, category)


class DirectS3Run:
    def __init__(self, uri: str) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"Invalid S3 run URI: {uri}")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for an s3:// run directory") from exc
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.client = boto3.client("s3", region_name=region or None)

    def download(self, destination: Path) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        found = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}/"):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                relative = key[len(self.prefix):].lstrip("/")
                if not relative or relative.endswith("/"):
                    continue
                local = destination / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, key, str(local))
                found += 1
        if not found:
            raise FileNotFoundError(f"No objects found below s3://{self.bucket}/{self.prefix}/")

    def upload_atomic(self, path: Path, category: str) -> None:
        final_key = f"{self.prefix}/{category}/{path.name}"
        temporary_key = f"{final_key}.tmp-{uuid.uuid4().hex}"
        checksum = sha256_file(path)
        size = path.stat().st_size
        try:
            self.client.upload_file(str(path), self.bucket, temporary_key, ExtraArgs={"Metadata": {"sha256": checksum}})
            temporary = self.client.head_object(Bucket=self.bucket, Key=temporary_key)
            if int(temporary["ContentLength"]) != size or temporary.get("Metadata", {}).get("sha256") != checksum:
                raise IOError("Temporary S3 report artifact failed checksum/size verification")
            self.client.copy_object(
                Bucket=self.bucket, Key=final_key,
                CopySource={"Bucket": self.bucket, "Key": temporary_key}, MetadataDirective="COPY",
            )
            final = self.client.head_object(Bucket=self.bucket, Key=final_key)
            if int(final["ContentLength"]) != size or final.get("Metadata", {}).get("sha256") != checksum:
                raise IOError("Final S3 report artifact failed checksum/size verification")
        finally:
            try:
                self.client.delete_object(Bucket=self.bucket, Key=temporary_key)
            except Exception:
                LOGGER.warning("Could not remove temporary report key %s", temporary_key)


@contextmanager
def materialize_run(run_dir: str) -> Iterator[tuple[Path, ArtifactCallback | None]]:
    if not run_dir.startswith("s3://"):
        local = Path(run_dir).resolve()
        if not local.exists():
            raise FileNotFoundError(local)
        yield local, None
        return
    remote = DirectS3Run(run_dir)
    with tempfile.TemporaryDirectory(prefix="lightgbm-report-") as temporary:
        local = Path(temporary) / "run"
        local.mkdir()
        remote.download(local)
        yield local, remote.upload_atomic


def local_s3_callback(run_dir: Path, enabled: bool | None, required: bool) -> ArtifactCallback | None:
    ready = bool(os.environ.get("S3_BUCKET") and os.environ.get("S3_PREFIX"))
    should_enable = ready if enabled is None else enabled
    if not should_enable:
        return None
    s3_config = read_json(Path(__file__).resolve().parent / "config" / "train.json")["s3"]
    s3_config["enabled"] = True
    s3_config["upload_required"] = required
    store = S3Store(s3_config, True)
    run_id = str(read_json(run_dir / "config" / "run_config.json").get("run_id", run_dir.name))
    return lambda path, category: store.upload_atomic(path, store.run_key(run_id, f"{category}/{path.name}"))


def combine_callbacks(first: ArtifactCallback | None, second: ArtifactCallback | None) -> ArtifactCallback | None:
    callbacks = [item for item in (first, second) if item]
    if not callbacks:
        return None

    def combined(path: Path, category: str) -> None:
        for item in callbacks:
            item(path, category)

    return combined


def stratified_indices(y_true: np.ndarray, num_classes: int, maximum: int, seed: int) -> np.ndarray:
    if len(y_true) <= maximum:
        return np.arange(len(y_true), dtype=np.int64)
    counts = np.bincount(np.asarray(y_true, dtype=np.int64), minlength=num_classes)
    raw = counts / counts.sum() * maximum
    quotas = np.floor(raw).astype(np.int64)
    quotas[(counts > 0) & (quotas == 0)] = 1
    while quotas.sum() > maximum:
        candidates = np.flatnonzero(quotas > 1)
        quotas[candidates[np.argmax(quotas[candidates] - raw[candidates])]] -= 1
    while quotas.sum() < maximum:
        candidates = np.flatnonzero(quotas < counts)
        quotas[candidates[np.argmax(raw[candidates] - quotas[candidates])]] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for class_index in range(num_classes):
        positions = np.flatnonzero(y_true == class_index)
        chosen = rng.choice(positions, size=min(int(quotas[class_index]), len(positions)), replace=False)
        selected.append(np.sort(chosen))
        del positions, chosen
        gc.collect()
    result = np.concatenate(selected)
    result.sort()
    return result


def create_prediction_artifacts(
    run_dir: Path,
    booster: Any,
    test_features: Any,
    test_labels: np.ndarray,
    report_config: Mapping[str, Any],
    callback: ArtifactCallback | None = None,
) -> list[Path]:
    num_classes = int(booster.num_model_per_iteration())
    if int(booster.current_iteration()) != 100:
        raise ValueError("Prediction artifacts require final_model_round_100")
    raw_dir = run_dir / "raw"
    metrics_dir = run_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    true_dtype = np.int16 if num_classes <= np.iinfo(np.int16).max else np.int32
    y_true_path = raw_dir / "y_true.npy"
    np.save(y_true_path, np.asarray(test_labels, dtype=true_dtype), allow_pickle=False)
    y_prob_path = raw_dir / "y_prob.npy"
    probabilities = np.lib.format.open_memmap(
        y_prob_path, mode="w+", dtype=np.float32, shape=(len(test_labels), num_classes)
    )
    chunk_rows = int(report_config["prediction_chunk_rows"])
    for start in range(0, len(test_labels), chunk_rows):
        stop = min(start + chunk_rows, len(test_labels))
        prediction = booster.predict(test_features.iloc[start:stop], num_iteration=100)
        probabilities[start:stop] = np.asarray(prediction, dtype=np.float32)
        probabilities.flush()
        del prediction
        gc.collect()
    del probabilities
    notify(y_true_path, "raw", callback)
    notify(y_prob_path, "raw", callback)

    indices = stratified_indices(
        np.asarray(test_labels), num_classes, int(report_config["explain_max_samples"]), int(report_config["seed"])
    )
    explain = test_features.iloc[indices].copy()
    explain["_label"] = np.asarray(test_labels)[indices].astype(true_dtype)
    explain_path = raw_dir / "explain_sample.parquet"
    explain.to_parquet(explain_path, index=False, compression="zstd")
    support = np.bincount(explain["_label"].to_numpy(dtype=np.int64), minlength=num_classes)
    manifest = {
        "source": "held-out test split",
        "sampling": "deterministic stratified sample preserving natural class proportions",
        "seed": int(report_config["seed"]),
        "maximum_rows": int(report_config["explain_max_samples"]),
        "actual_rows": len(explain),
        "class_counts": support.tolist(),
        "feature_count": test_features.shape[1],
        "used_for_model_metrics": False,
    }
    manifest_path = raw_dir / "explain_sample_manifest.json"
    atomic_json_dump(manifest, manifest_path)
    notify(explain_path, "raw", callback)
    notify(manifest_path, "raw", callback)
    del explain, indices
    gc.collect()

    benchmark = benchmark_booster(booster, test_features, report_config["benchmark"])
    benchmark_path = metrics_dir / "deployment_benchmark.json"
    atomic_json_dump(benchmark, benchmark_path)
    notify(benchmark_path, "metrics", callback)
    return [y_true_path, y_prob_path, explain_path, manifest_path, benchmark_path]


def benchmark_booster(booster: Any, features: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    import lightgbm as lgb
    import sklearn

    batch_size = min(int(config["batch_size"]), len(features))
    warmups = int(config["warmup_iterations"])
    measurements = int(config["measurement_iterations"])
    if batch_size <= 0 or warmups != 50 or measurements != 500:
        raise ValueError("Benchmark contract requires a positive batch and exactly 50 warm-ups / 500 measurements")
    frame = features.iloc[:batch_size]
    for _ in range(warmups):
        array = frame.to_numpy(dtype=np.float32, copy=True)
        booster.predict(array, num_iteration=100)
        del array
    preprocess_ms, predict_ms, total_ms = [], [], []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    for _ in range(measurements):
        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        array = frame.to_numpy(dtype=np.float32, copy=True)
        predict_start = time.perf_counter()
        prediction = booster.predict(array, num_iteration=100)
        end = time.perf_counter()
        preprocess_ms.append((predict_start - preprocess_start) * 1000)
        predict_ms.append((end - predict_start) * 1000)
        total_ms.append((end - total_start) * 1000)
        peak_rss = max(peak_rss, process.memory_info().rss)
        del array, prediction
    total_seconds = sum(total_ms) / 1000
    percentile = lambda values, q: float(np.percentile(values, q))
    model_path_size = len(booster.model_to_string(num_iteration=100).encode("utf-8")) / 1024**2
    return {
        "protocol": {"warmup_iterations": warmups, "measurement_iterations": measurements},
        "batch_size": batch_size,
        "cpu_threads": int(booster.params.get("num_threads", 0)),
        "model_size_mb": model_path_size,
        "t_preprocess": {"p50_ms_per_batch": percentile(preprocess_ms, 50), "p95_ms_per_batch": percentile(preprocess_ms, 95)},
        "t_predict": {"p50_ms_per_batch": percentile(predict_ms, 50), "p95_ms_per_batch": percentile(predict_ms, 95)},
        "t_total": {"p50_ms_per_batch": percentile(total_ms, 50), "p95_ms_per_batch": percentile(total_ms, 95)},
        "latency_p50_ms_per_sample": percentile(total_ms, 50) / batch_size,
        "latency_p95_ms_per_sample": percentile(total_ms, 95) / batch_size,
        "throughput_samples_per_second": batch_size * measurements / total_seconds,
        "peak_rss_mb": peak_rss / 1024**2,
        "lightgbm_version": lgb.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cpu": platform.processor() or platform.machine(),
        "total_ram_bytes": int(psutil.virtual_memory().total),
    }


def cumulative_predictions(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int, chunk_rows: int) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.int16 if num_classes <= np.iinfo(np.int16).max else np.int32
    predicted_all = np.empty(len(y_true), dtype=dtype)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for start in range(0, len(y_true), chunk_rows):
        stop = min(start + chunk_rows, len(y_true))
        predicted = np.asarray(y_prob[start:stop]).argmax(axis=1).astype(dtype)
        predicted_all[start:stop] = predicted
        np.add.at(confusion, (np.asarray(y_true[start:stop], dtype=np.int64), predicted), 1)
        del predicted
    return predicted_all, confusion


def compute_auc_pr(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int, support: np.ndarray) -> tuple[list[Any], list[Any], dict[str, Any]]:
    roc_values, pr_values, valid_roc, valid_pr, valid_support = [], [], [], [], []
    for class_index in range(num_classes):
        y_true_binary = (np.asarray(y_true) == class_index).astype(np.int8)
        positives = int(y_true_binary.sum())
        if positives == 0 or positives == len(y_true_binary):
            roc_value = pr_value = None
        else:
            y_prob_c = np.asarray(y_prob[:, class_index], dtype=np.float32)
            roc_value = float(roc_auc_score(y_true_binary, y_prob_c))
            pr_value = float(average_precision_score(y_true_binary, y_prob_c))
            valid_roc.append(roc_value)
            valid_pr.append(pr_value)
            valid_support.append(int(support[class_index]))
            del y_prob_c
        roc_values.append(roc_value)
        pr_values.append(pr_value)
        del y_true_binary
        gc.collect()
    weights = np.asarray(valid_support, dtype=np.float64)
    y_true_binary = np.equal.outer(np.asarray(y_true), np.arange(num_classes)).astype(np.int8).ravel()
    y_prob_c = np.asarray(y_prob, dtype=np.float32).ravel()
    summary = {
        "roc_macro_ovr": float(np.mean(valid_roc)) if valid_roc else None,
        "roc_weighted_ovr": float(np.average(valid_roc, weights=weights)) if valid_roc and weights.sum() else None,
        "roc_micro": float(roc_auc_score(y_true_binary, y_prob_c)),
        "pr_macro": float(np.mean(valid_pr)) if valid_pr else None,
        "pr_weighted": float(np.average(valid_pr, weights=weights)) if valid_pr and weights.sum() else None,
        "pr_micro": float(average_precision_score(y_true_binary, y_prob_c)),
    }
    del y_true_binary, y_prob_c
    gc.collect()
    return roc_values, pr_values, summary


def _importance_tables(
    run_dir: Path,
    booster: Any,
    class_names: list[str],
    report_config: Mapping[str, Any],
    callback: ArtifactCallback | None,
) -> dict[str, pd.DataFrame]:
    explain_path = run_dir / "raw" / "explain_sample.parquet"
    explain = pd.read_parquet(explain_path)
    labels = explain.pop("_label").to_numpy(dtype=np.int32)
    model_feature_names = list(booster.feature_name())
    preprocessing = read_json(run_dir / "config" / "preprocessing.json")
    feature_names = list(preprocessing["feature_columns_in_order"])
    run_config = read_json(run_dir / "config" / "run_config.json")
    configured_model_names = list(run_config.get("model_feature_names", feature_names))
    if model_feature_names != configured_model_names or model_feature_names != list(explain.columns):
        raise ValueError("Booster, run_config.json, and explain_sample.parquet feature order differ")
    explainability = run_dir / "explainability"
    explainability.mkdir(parents=True, exist_ok=True)

    gain = np.asarray(booster.feature_importance("gain", iteration=100), dtype=np.float64)
    gain_total = gain.sum()
    gain_table = pd.DataFrame({"feature": feature_names, "gain": gain})
    gain_table["gain_percent"] = np.where(gain_total > 0, gain / gain_total * 100, 0)
    gain_table = gain_table.sort_values("gain", ascending=False).reset_index(drop=True)
    gain_table["cumulative_gain_percent"] = gain_table["gain_percent"].cumsum()
    gain_table["rank_gain"] = np.arange(1, len(gain_table) + 1)

    split = np.asarray(booster.feature_importance("split", iteration=100), dtype=np.int64)
    split_total = split.sum()
    split_table = pd.DataFrame({"feature": feature_names, "split_count": split})
    split_table["split_percent"] = np.where(split_total > 0, split / split_total * 100, 0)
    split_table = split_table.sort_values("split_count", ascending=False).reset_index(drop=True)
    split_table["rank_split"] = np.arange(1, len(split_table) + 1)

    actual_explain = explain
    while True:
        try:
            shap_sum = np.zeros(len(feature_names), dtype=np.float64)
            rows_seen = 0
            chunk_rows = int(report_config["shap_chunk_rows"])
            for start in range(0, len(actual_explain), chunk_rows):
                stop = min(start + chunk_rows, len(actual_explain))
                raw_contributions = booster.predict(
                    actual_explain.iloc[start:stop], pred_contrib=True, num_iteration=100
                )
                if isinstance(raw_contributions, list):
                    contributions = np.stack(
                        [np.asarray(value) for value in raw_contributions], axis=1
                    )
                else:
                    contributions = np.asarray(raw_contributions)
                if contributions.ndim == 2:
                    contributions = contributions.reshape(len(contributions), len(class_names), len(feature_names) + 1)
                elif contributions.ndim == 3 and contributions.shape[0] == len(class_names):
                    contributions = np.transpose(contributions, (1, 0, 2))
                if contributions.shape[1:] != (len(class_names), len(feature_names) + 1):
                    raise ValueError(f"Unexpected multiclass SHAP shape: {contributions.shape}")
                shap_sum += np.abs(contributions[:, :, :-1]).sum(axis=(0, 1))
                rows_seen += len(contributions) * len(class_names)
                del raw_contributions, contributions
                gc.collect()
            mean_abs = shap_sum / max(rows_seen, 1)
            break
        except MemoryError:
            if len(actual_explain) <= 1000:
                raise
            reduced_indices = stratified_indices(labels[:len(actual_explain)], len(class_names), max(1000, len(actual_explain) // 2), int(report_config["seed"]))
            actual_explain = actual_explain.iloc[reduced_indices].reset_index(drop=True)
            labels = labels[reduced_indices]
            LOGGER.warning("Reduced explain sample to %d rows after SHAP MemoryError", len(actual_explain))
    if len(actual_explain) != len(explain):
        persisted = actual_explain.copy()
        persisted["_label"] = labels
        persisted.to_parquet(explain_path, index=False, compression="zstd")
        manifest_path = run_dir / "raw" / "explain_sample_manifest.json"
        manifest = read_json(manifest_path)
        manifest["actual_rows"] = len(actual_explain)
        manifest["automatic_reduction_reason"] = "SHAP MemoryError"
        atomic_json_dump(manifest, manifest_path)
        notify(explain_path, "raw", callback)
        notify(manifest_path, "raw", callback)
    shap_total = mean_abs.sum()
    shap_table = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    shap_table["shap_percent"] = np.where(shap_total > 0, mean_abs / shap_total * 100, 0)
    shap_table = shap_table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_table["rank_shap"] = np.arange(1, len(shap_table) + 1)

    baseline_prediction = np.asarray(booster.predict(actual_explain, num_iteration=100)).argmax(axis=1)
    baseline_f1 = f1_score(labels, baseline_prediction, labels=list(range(len(class_names))), average="macro", zero_division=0)
    repeats = int(report_config["permutation_repeats"])
    rng = np.random.default_rng(int(report_config["seed"]))
    decreases: dict[str, list[float]] = {name: [] for name in feature_names}
    for feature, model_feature in zip(feature_names, model_feature_names):
        original = actual_explain[model_feature].to_numpy(copy=True)
        for _ in range(repeats):
            actual_explain[model_feature] = rng.permutation(original)
            predicted = np.asarray(booster.predict(actual_explain, num_iteration=100)).argmax(axis=1)
            score = f1_score(labels, predicted, labels=list(range(len(class_names))), average="macro", zero_division=0)
            decreases[feature].append(float(baseline_f1 - score))
            del predicted
        actual_explain[model_feature] = original
        del original
        gc.collect()
    permutation_table = pd.DataFrame({
        "feature": feature_names,
        "mean_decrease": [float(np.mean(decreases[name])) for name in feature_names],
        "std_decrease": [float(np.std(decreases[name], ddof=1)) if repeats > 1 else 0.0 for name in feature_names],
    }).sort_values("mean_decrease", ascending=False).reset_index(drop=True)
    permutation_table["rank_permutation"] = np.arange(1, len(permutation_table) + 1)

    tables = {
        "feature_importance_gain": gain_table,
        "feature_importance_split": split_table,
        "permutation_importance": permutation_table,
        "shap_feature_importance": shap_table,
    }
    for name, table in tables.items():
        path = explainability / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.6f")
        notify(path, "explainability", callback)
    comparison = gain_table[["feature", "rank_gain"]].merge(
        split_table[["feature", "rank_split"]], on="feature"
    ).merge(permutation_table[["feature", "rank_permutation"]], on="feature").merge(
        shap_table[["feature", "rank_shap"]], on="feature"
    )
    rank_columns = ["rank_gain", "rank_split", "rank_permutation", "rank_shap"]
    comparison["top10_method_count"] = (comparison[rank_columns] <= 10).sum(axis=1)
    comparison["top10_consensus"] = comparison["top10_method_count"] >= 3
    comparison_path = explainability / "feature_importance_comparison.csv"
    comparison.to_csv(comparison_path, index=False, float_format="%.6f")
    notify(comparison_path, "explainability", callback)
    del explain, actual_explain, labels, baseline_prediction
    gc.collect()
    return tables


def generate_report(run_dir: Path, callback: ArtifactCallback | None = None) -> list[Path]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required to rebuild model-dependent report artifacts") from exc
    report_config = load_report_config(run_dir)
    config_dir, metrics_dir = run_dir / "config", run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = config_dir / "run_config.json"
    run_config = read_json(run_config_path)
    history = read_json(metrics_dir / "history.json")
    mapping = read_json(config_dir / "label_mapping.json")
    class_names = ordered_class_names(mapping)
    labels_order = list(range(len(class_names)))
    model_path = run_dir / "checkpoints" / "final_model_round_100.txt"
    booster = lgb.Booster(model_file=str(model_path))
    if int(booster.current_iteration()) != 100:
        raise ValueError("final_model_round_100.txt does not contain exactly 100 iterations")
    y_true = np.load(run_dir / "raw" / "y_true.npy", mmap_mode="r")
    y_prob = np.load(run_dir / "raw" / "y_prob.npy", mmap_mode="r")
    if y_true.dtype not in (np.int16, np.int32) or y_prob.dtype != np.float32:
        raise TypeError("y_true must be int16/int32 and y_prob must be float32")
    if y_prob.shape != (len(y_true), len(class_names)):
        raise ValueError("Prediction arrays disagree with label_mapping.json")
    y_pred, confusion = cumulative_predictions(y_true, y_prob, len(class_names), int(report_config["prediction_chunk_rows"]))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, zero_division=0
    )
    roc_values, pr_values, auc_summary = compute_auc_pr(y_true, y_prob, len(class_names), support)
    per_class = pd.DataFrame({
        "class": class_names, "support": support.astype(np.int64), "precision": precision,
        "recall": recall, "f1": f1, "roc_auc": roc_values, "pr_auc": pr_values,
    })
    per_class_path = metrics_dir / "per_class_metrics.csv"
    per_class.to_csv(per_class_path, index=False, float_format="%.6f")
    notify(per_class_path, "metrics", callback)
    report_path = metrics_dir / "classification_report.txt"
    report_path.write_text(classification_report(
        y_true, y_pred, labels=labels_order, target_names=class_names, digits=4, zero_division=0
    ), encoding="utf-8")
    notify(report_path, "metrics", callback)
    minority = int(np.argmin(np.where(support > 0, support, np.iinfo(np.int64).max)))
    benchmark_path = metrics_dir / "deployment_benchmark.json"
    if benchmark_path.exists():
        benchmark = read_json(benchmark_path)
    else:
        benchmark_features = pd.read_parquet(run_dir / "raw" / "explain_sample.parquet").drop(columns=["_label"])
        benchmark = benchmark_booster(booster, benchmark_features, report_config["benchmark"])
        atomic_json_dump(benchmark, benchmark_path)
        notify(benchmark_path, "metrics", callback)
        del benchmark_features
        gc.collect()
    training_seconds = float(sum(float(item["iteration_seconds"]) for item in history))
    test_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels_order, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "minority_class": class_names[minority],
        "minority_class_support": int(support[minority]),
        "minority_class_f1": float(f1[minority]),
        "log_loss": float(log_loss(y_true, y_prob, labels=labels_order)),
        "auc_roc_macro_ovr": auc_summary["roc_macro_ovr"],
        "auc_roc_weighted_ovr": auc_summary["roc_weighted_ovr"],
        "auc_roc_micro": auc_summary["roc_micro"],
        "pr_auc_macro": auc_summary["pr_macro"],
        "pr_auc_weighted": auc_summary["pr_weighted"],
        "pr_auc_micro": auc_summary["pr_micro"],
        "final_iteration": 100,
        "num_trees": int(booster.num_trees()),
        "model_size_mb": model_path.stat().st_size / 1024**2,
        "training_time_seconds": training_seconds,
        "test_samples": len(y_true),
        "deployment_benchmark": benchmark,
    }
    test_metrics_path = metrics_dir / "test_metrics.json"
    atomic_json_dump(test_metrics, test_metrics_path)
    notify(test_metrics_path, "metrics", callback)
    summary = {
        "Accuracy": test_metrics["accuracy"], "Balanced Accuracy": test_metrics["balanced_accuracy"],
        "Macro Precision": test_metrics["macro_precision"], "Macro Recall": test_metrics["macro_recall"],
        "Macro F1": test_metrics["macro_f1"], "Weighted F1": test_metrics["weighted_f1"],
        "MCC": test_metrics["mcc"], "Minority Class": test_metrics["minority_class"],
        "Minority Class F1": test_metrics["minority_class_f1"], "Log Loss": test_metrics["log_loss"],
        "AUC-ROC macro-OVR": test_metrics["auc_roc_macro_ovr"],
        "AUC-ROC weighted-OVR": test_metrics["auc_roc_weighted_ovr"], "AUC-ROC micro": test_metrics["auc_roc_micro"],
        "PR-AUC macro": test_metrics["pr_auc_macro"], "PR-AUC weighted": test_metrics["pr_auc_weighted"],
        "PR-AUC micro": test_metrics["pr_auc_micro"], "final_iteration": 100,
        "num_trees": test_metrics["num_trees"], "model_size_mb": test_metrics["model_size_mb"],
        "training_time_seconds": training_seconds,
        "t_preprocess_p50_ms_batch": benchmark["t_preprocess"]["p50_ms_per_batch"],
        "t_preprocess_p95_ms_batch": benchmark["t_preprocess"]["p95_ms_per_batch"],
        "t_predict_p50_ms_batch": benchmark["t_predict"]["p50_ms_per_batch"],
        "t_predict_p95_ms_batch": benchmark["t_predict"]["p95_ms_per_batch"],
        "t_total_p50_ms_batch": benchmark["t_total"]["p50_ms_per_batch"],
        "t_total_p95_ms_batch": benchmark["t_total"]["p95_ms_per_batch"],
        "latency_p50_ms_sample": benchmark["latency_p50_ms_per_sample"],
        "latency_p95_ms_sample": benchmark["latency_p95_ms_per_sample"],
        "throughput_samples_second": benchmark["throughput_samples_per_second"],
        "peak_rss_mb": benchmark["peak_rss_mb"], "inference_batch_size": benchmark["batch_size"],
        "cpu_threads": benchmark["cpu_threads"],
    }
    summary_path = metrics_dir / "summary_metrics.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False, float_format="%.6f")
    notify(summary_path, "metrics", callback)

    importance = _importance_tables(run_dir, booster, class_names, report_config, callback)
    plot_indices = stratified_indices(y_true, len(class_names), int(report_config["maximum_roc_pr_plot_samples"]), int(report_config["seed"]))
    y_true_plot = np.asarray(y_true[plot_indices], dtype=np.int32)
    y_prob_plot = np.asarray(y_prob[plot_indices], dtype=np.float32)
    run_config["plotting"] = {
        "maximum_roc_pr_samples": int(report_config["maximum_roc_pr_plot_samples"]),
        "roc_pr_samples_used": len(plot_indices),
        "sampling": "full test" if len(plot_indices) == len(y_true) else "deterministic stratified sample",
        "seed": int(report_config["seed"]),
        "metrics_use_full_test": True,
    }
    atomic_json_dump(run_config, run_config_path)
    notify(run_config_path, "config", callback)
    figures = generate_final_figures(
        run_dir, y_true_plot, y_prob_plot, confusion, per_class, importance, callback
    )
    del y_true_plot, y_prob_plot, plot_indices, y_pred, confusion, y_true, y_prob
    gc.collect()
    return [per_class_path, report_path, test_metrics_path, summary_path, run_config_path, *figures]


def evaluate_final_model(
    run_dir: Path,
    booster: Any,
    test_features: Any,
    test_labels: np.ndarray,
    callback: ArtifactCallback | None = None,
) -> list[Path]:
    report_config = load_report_config(run_dir)
    created = create_prediction_artifacts(run_dir, booster, test_features, test_labels, report_config, callback)
    return [*created, *generate_report(run_dir, callback)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Local run directory or s3://bucket/prefix/run_id")
    parser.add_argument("--upload-to-s3", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--s3-upload-required", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    with materialize_run(args.run_dir) as (run_dir, direct_callback):
        environment_callback = None if args.run_dir.startswith("s3://") else local_s3_callback(
            run_dir, args.upload_to_s3, args.s3_upload_required
        )
        generated = generate_report(run_dir, combine_callbacks(direct_callback, environment_callback))
        LOGGER.info("Generated %d report artifacts under %s", len(set(generated)), run_dir)


if __name__ == "__main__":
    main()
