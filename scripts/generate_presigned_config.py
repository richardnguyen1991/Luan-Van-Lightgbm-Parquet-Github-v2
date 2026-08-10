"""Generate short-lived atomic S3 operations for one Kaggle session."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote


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
    # Part names are deterministic. 512 slots per split supports up to
    # 402,653,184 rows at the configured 262,144 rows/part while keeping all
    # Kaggle credentials object-scoped and short-lived.
    keys.update(
        f"{preprocessing_root}/splits/{split}/part-{number:06d}.parquet"
        for split in ("train", "validation", "test")
        for number in range(512)
    )
    return keys


def presigned_operations(client: object, bucket: str, final_key: str, expires: int) -> dict[str, object]:
    temporary_key = f"{final_key}.staging"
    copy_source = f"/{bucket}/{quote(temporary_key, safe='/')}"
    return {
        "temporary_key": temporary_key,
        "put_temporary": client.generate_presigned_url(
            "put_object", Params={"Bucket": bucket, "Key": temporary_key}, ExpiresIn=expires
        ),
        "head_temporary": client.generate_presigned_url(
            "head_object", Params={"Bucket": bucket, "Key": temporary_key}, ExpiresIn=expires
        ),
        "copy_final": client.generate_presigned_url(
            "copy_object",
            Params={"Bucket": bucket, "Key": final_key, "CopySource": {"Bucket": bucket, "Key": temporary_key}, "MetadataDirective": "COPY"},
            ExpiresIn=expires,
        ),
        "copy_headers": {"x-amz-copy-source": copy_source, "x-amz-metadata-directive": "COPY"},
        "head_final": client.generate_presigned_url(
            "head_object", Params={"Bucket": bucket, "Key": final_key}, ExpiresIn=expires
        ),
        "get_final": client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": final_key}, ExpiresIn=expires
        ),
        "delete_temporary": client.generate_presigned_url(
            "delete_object", Params={"Bucket": bucket, "Key": temporary_key}, ExpiresIn=expires
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
    payload = {
        "format_version": 2,
        "bucket": args.bucket,
        "s3_prefix": prefix,
        "run_id": run_id,
        "aws_region": client.meta.region_name,
        "expires_at_utc": (datetime.now(timezone.utc) + timedelta(seconds=args.expires)).isoformat(),
        "uploads": {key: presigned_operations(client, args.bucket, key, args.expires) for key in sorted(keys)},
        "downloads": {
            key: {
                "get": client.generate_presigned_url("get_object", Params={"Bucket": args.bucket, "Key": key}, ExpiresIn=args.expires),
                "head": client.generate_presigned_url("head_object", Params={"Bucket": args.bucket, "Key": key}, ExpiresIn=args.expires),
            }
            for key in sorted(keys)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Generated object-scoped S3 session operations for {run_id}: {len(keys)} keys")


if __name__ == "__main__":
    main()
