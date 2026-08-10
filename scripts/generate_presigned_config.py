"""Generate compact, short-lived S3 operations for one Kaggle session."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


AWS_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "AWS_DEFAULT_REGION",
)
FIGURE_NAMES = (
    "learning_curves", "lr_schedule", "iteration_time", "class_distribution",
    "confusion_matrix", "confusion_matrix_normalized", "confusion_matrix_raw",
    "roc_curves", "pr_curves", "per_class_f1", "feature_importance_gain",
    "feature_importance_split", "permutation_importance", "shap_feature_importance",
)
METRIC_FILES = (
    "history.json", "history.csv", "deployment_benchmark.json", "per_class_metrics.csv",
    "classification_report.txt", "test_metrics.json", "summary_metrics.csv",
    "confusion_matrix.csv", "confusion_matrix_normalized.csv",
)
CONFIG_FILES = (
    "run_config.json", "model_params.json", "sample_manifest.json", "preprocessing.json",
    "data_profile.json", "label_mapping.json", "report_config.json",
)
RAW_FILES = ("y_true.npy", "y_prob.npy", "explain_sample.parquet", "explain_sample_manifest.json")
EXPLAIN_FILES = (
    "feature_importance_gain.csv", "feature_importance_split.csv", "permutation_importance.csv",
    "shap_feature_importance.csv", "feature_importance_comparison.csv",
)


def normalize_aws_environment() -> None:
    for name in AWS_ENV_NAMES:
        value = os.environ.get(name)
        if value is not None:
            os.environ[name] = value.replace("\r", "").replace("\n", "").replace("\\r", "").replace("\\n", "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--s3-prefix", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expires", type=int, default=86400)
    parser.add_argument("--region", default=None)
    return parser.parse_args()


def resolve_run_id(client: object, bucket: str, prefix: str, requested: str | None) -> str:
    if requested:
        return requested
    try:
        response = client.get_object(Bucket=bucket, Key=f"{prefix}/active_run.json")
        pointer = json.loads(response["Body"].read().decode("utf-8"))
        if pointer.get("status") in {"preparing", "running", "paused", "ready_for_report"}:
            return str(pointer["run_id"])
    except Exception as exc:
        response = getattr(exc, "response", {}) or {}
        if response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
    return f"lightgbm_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"


def run_keys(prefix: str, run_id: str) -> set[str]:
    root = f"{prefix}/{run_id}"
    keys = {f"{prefix}/active_run.json"}
    keys.update(f"{root}/checkpoints/model_round_{iteration:03d}.txt" for iteration in range(10, 101, 10))
    keys.update(f"{root}/checkpoints/{name}" for name in (
        "last_model.txt", "training_state.json", "final_model_round_100.txt",
    ))
    keys.update(f"{root}/metrics/{name}" for name in METRIC_FILES)
    keys.update(f"{root}/metrics/{name}.csv" for name in FIGURE_NAMES)
    keys.update(f"{root}/figures/{name}.{suffix}" for name in FIGURE_NAMES for suffix in ("png", "pdf"))
    keys.update(f"{root}/config/{name}" for name in CONFIG_FILES)
    keys.update(f"{root}/raw/{name}" for name in RAW_FILES)
    keys.update(f"{root}/explainability/{name}" for name in EXPLAIN_FILES)
    preprocessing_root = f"{root}/preprocessing"
    keys.update(f"{preprocessing_root}/{name}" for name in (
        "progress.json", "data_profile.json", "label_mapping.json",
        "preprocessing.json", "sample_manifest.json",
    ))
    return keys


def existing_preprocessing_part_keys(client: object, bucket: str, prefix: str, run_id: str) -> set[str]:
    """Return only part objects committed by the durable progress manifest."""
    progress_key = f"{prefix}/{run_id}/preprocessing/progress.json"
    try:
        response = client.get_object(Bucket=bucket, Key=progress_key)
        progress = json.loads(response["Body"].read().decode("utf-8"))
    except Exception as exc:
        response = getattr(exc, "response", {}) or {}
        if response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return set()
        raise
    keys: set[str] = set()
    preprocessing_root = f"{prefix}/{run_id}/preprocessing"
    for split, entries in progress.get("parts", {}).items():
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unexpected preprocessing split in progress.json: {split}")
        expected_prefix = f"splits/{split}/"
        for entry in entries:
            relative = str(entry["path"]).lstrip("/")
            if not relative.startswith(expected_prefix) or "/" in relative[len(expected_prefix):]:
                raise ValueError(f"Unsafe preprocessing part path in progress.json: {relative}")
            keys.add(f"{preprocessing_root}/{relative}")
    return keys


def part_upload_operations(client: object, bucket: str, prefix: str, run_id: str, expires: int) -> dict[str, object]:
    """Create three prefix-limited POST policies instead of thousands of URLs."""
    result: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        part_prefix = f"{prefix}/{run_id}/preprocessing/splits/{split}"
        result[part_prefix] = client.generate_presigned_post(
            Bucket=bucket,
            Key=f"{part_prefix}/${{filename}}",
            Fields={"success_action_status": "204"},
            Conditions=[
                ["starts-with", "$key", f"{part_prefix}/"],
                {"success_action_status": "204"},
            ],
            ExpiresIn=expires,
        )
    return result


def presigned_operations(client: object, bucket: str, final_key: str, expires: int) -> dict[str, object]:
    return {
        "put_final": client.generate_presigned_url(
            "put_object", Params={"Bucket": bucket, "Key": final_key}, ExpiresIn=expires
        ),
        "head_final": client.generate_presigned_url(
            "head_object", Params={"Bucket": bucket, "Key": final_key}, ExpiresIn=expires
        ),
        "get_final": client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": final_key}, ExpiresIn=expires
        ),
    }


def main() -> None:
    normalize_aws_environment()
    import boto3
    from botocore.config import Config

    args = parse_args()
    if not 1 <= args.expires <= 604800:
        raise ValueError("--expires must be between 1 and 604800 seconds")
    prefix = args.s3_prefix.strip("/")
    kwargs: dict[str, object] = {"config": Config(signature_version="s3v4", s3={"addressing_style": "virtual"})}
    if args.region:
        kwargs["region_name"] = args.region
    client = boto3.client("s3", **kwargs)
    run_id = resolve_run_id(client, args.bucket, prefix, args.run_id)
    keys = run_keys(prefix, run_id)
    existing_parts = existing_preprocessing_part_keys(client, args.bucket, prefix, run_id)
    download_keys = keys | existing_parts
    payload = {
        "format_version": 4,
        "bucket": args.bucket,
        "s3_prefix": prefix,
        "run_id": run_id,
        "aws_region": client.meta.region_name,
        "expires_at_utc": (datetime.now(timezone.utc) + timedelta(seconds=args.expires)).isoformat(),
        "uploads": {key: presigned_operations(client, args.bucket, key, args.expires) for key in sorted(keys)},
        "part_uploads": part_upload_operations(client, args.bucket, prefix, run_id, args.expires),
        "downloads": {
            key: {
                "get": client.generate_presigned_url("get_object", Params={"Bucket": args.bucket, "Key": key}, ExpiresIn=args.expires),
                "head": client.generate_presigned_url("head_object", Params={"Bucket": args.bucket, "Key": key}, ExpiresIn=args.expires),
            }
            for key in sorted(download_keys)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        f"Generated compact S3 session operations for {run_id}: "
        f"{len(keys)} fixed keys, {len(existing_parts)} existing parts, 3 part-upload policies"
    )


if __name__ == "__main__":
    main()

