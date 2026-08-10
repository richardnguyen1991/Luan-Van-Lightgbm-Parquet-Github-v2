"""Build a self-contained CPU Kaggle notebook from versioned project files."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDED_FILES = (
    "data.py", "model.py", "checkpoint.py", "train.py", "viz.py", "make_report.py",
    "config/data.json", "config/data.smoke.json", "config/train.json",
    "config/train.smoke.json", "config/report.json", "config/orchestration.json",
)


def code_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.rstrip().splitlines()]}


def markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.rstrip().splitlines()]}


def encoded_sources() -> dict[str, str]:
    return {
        relative: base64.b64encode(zlib.compress((PROJECT_ROOT / relative).read_bytes(), level=9)).decode("ascii")
        for relative in EMBEDDED_FILES
    }


def build_notebook(
    profile: str = "production",
    presigned_config: str | Path | None = None,
    max_rounds_this_session: int | None = None,
) -> dict[str, Any]:
    if profile not in {"production", "smoke"}:
        raise ValueError(f"Unsupported notebook profile: {profile}")
    suffix = "" if profile == "production" else ".smoke"
    data_config = f"config/data{suffix}.json"
    train_config = f"config/train{suffix}.json"
    embedded = json.dumps(encoded_sources(), sort_keys=True)
    presigned_bytes = Path(presigned_config).read_bytes() if presigned_config else b""
    presigned_b64 = base64.b64encode(zlib.compress(presigned_bytes, level=9)).decode("ascii") if presigned_bytes else ""
    description = (
        "Production: full natural-distribution train split and exactly 100 boosting iterations."
        if profile == "production"
        else "Smoke: 2,000 rows per source file and at most 10 new iterations in this session; target remains exactly 100."
    )
    extract = f'''from pathlib import Path
import base64
import json
import os
import subprocess
import sys
import time
import zlib

PROJECT_NAME = "Luan-Van-LightGBM-Parquet-Github-v2"
SESSION_MAXIMUM_HOURS = 8.5
SESSION_STOP_BEFORE_MINUTES = 30.0
os.environ["PIPELINE_SESSION_DEADLINE_EPOCH"] = str(
    time.time() + SESSION_MAXIMUM_HOURS * 3600.0 - SESSION_STOP_BEFORE_MINUTES * 60.0
)
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
PROJECT_DIR = Path("/kaggle/working") / PROJECT_NAME
SOURCE_DIR = PROJECT_DIR / "source"
PREPARED_DIR = PROJECT_DIR / "prepared"
RUNS_DIR = PROJECT_DIR / "runs"
for directory in (SOURCE_DIR, PREPARED_DIR, RUNS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

encoded_files = json.loads({json.dumps(embedded)})
for relative, encoded_content in encoded_files.items():
    destination = SOURCE_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(zlib.decompress(base64.b64decode(encoded_content)))
print(f"Extracted {{len(encoded_files)}} versioned source/config files")
'''
    secrets = f'''PRESIGNED_CONFIG_ZLIB_B64 = {presigned_b64!r}
if PRESIGNED_CONFIG_ZLIB_B64:
    presigned_path = PROJECT_DIR / "s3_presigned_config.json"
    presigned_path.write_bytes(zlib.decompress(base64.b64decode(PRESIGNED_CONFIG_ZLIB_B64)))
    presigned = json.loads(presigned_path.read_text(encoding="utf-8"))
    os.environ["S3_PRESIGNED_CONFIG_PATH"] = str(presigned_path)
    os.environ["S3_BUCKET"] = presigned["bucket"]
    os.environ["S3_PREFIX"] = presigned["s3_prefix"]
    os.environ["RUN_ID"] = presigned["run_id"]
    os.environ["AWS_REGION"] = presigned["aws_region"]
    os.environ["AWS_DEFAULT_REGION"] = presigned["aws_region"]
    print("Loaded short-lived object-scoped S3 operations; no AWS key is embedded.")
else:
    from kaggle_secrets import UserSecretsClient
    client = UserSecretsClient()
    aliases = {{
        "AWS_ACCESS_KEY_ID": ("AWS_ACCESS_KEY_ID",),
        "AWS_SECRET_ACCESS_KEY": ("AWS_SECRET_ACCESS_KEY",),
        "AWS_REGION": ("AWS_REGION", "AWS_DEFAULT_REGION"),
        "AWS_DEFAULT_REGION": ("AWS_DEFAULT_REGION", "AWS_REGION"),
        "S3_BUCKET": ("S3_BUCKET",),
        "S3_PREFIX": ("S3_PREFIX",),
    }}
    missing = []
    for environment_name, candidates in aliases.items():
        value = None
        for candidate in candidates:
            try:
                value = client.get_secret(candidate)
            except Exception:
                value = None
            if value:
                break
        if value:
            os.environ[environment_name] = value
        else:
            missing.append("/".join(candidates))
    if missing:
        raise RuntimeError("Missing S3 configuration: " + ", ".join(sorted(set(missing))))
os.environ["PYTHONHASHSEED"] = "2026"
print("S3 environment configured; credential values were not printed.")
'''
    dependencies = '''required = {
    "lightgbm": "lightgbm>=4.0,<5",
    "boto3": "boto3>=1.34,<2",
    "requests": "requests>=2.31,<3",
}
missing = []
for module, requirement in required.items():
    try:
        imported = __import__(module)
        if module == "lightgbm" and not str(imported.__version__).startswith("4."):
            missing.append(requirement)
    except ImportError:
        missing.append(requirement)
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
import lightgbm as lgb
import psutil
host_ram_gib = psutil.virtual_memory().total / (1024 ** 3)
print(f"LightGBM={lgb.__version__}; LightGBM device=CPU; Kaggle host=TPU v3-8 VM; RAM={host_ram_gib:.1f} GiB")
'''
    prepare = f'''preferred = Path("/kaggle/input/cicddos2019-parquet-per-classes")
if preferred.exists():
    data_dir = preferred
else:
    parquet_files = sorted(Path("/kaggle/input").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No Parquet files found in attached Kaggle inputs")
    data_dir = Path(os.path.commonpath([str(path.parent) for path in parquet_files]))
print(f"Preparing deterministic leakage-safe splits from {{data_dir}}")
data_command = [
    sys.executable, str(SOURCE_DIR / "data.py"),
    "--config", str(SOURCE_DIR / "{data_config}"),
    "--data-dir", str(data_dir),
    "--output-dir", str(PREPARED_DIR),
]
if os.environ.get("RUN_ID"):
    data_command.extend([
        "--s3-config", str(SOURCE_DIR / "{train_config}"),
        "--run-id", os.environ["RUN_ID"],
        "--maximum-hours", "8.5",
        "--stop-before-minutes", "30",
    ])
data_result = subprocess.run(data_command, cwd=SOURCE_DIR, check=False)
if data_result.returncode not in (0, 75):
    raise subprocess.CalledProcessError(data_result.returncode, data_command)
PREPROCESSING_PAUSED = data_result.returncode == 75
if PREPROCESSING_PAUSED:
    print("Preprocessing paused after a durable source-file checkpoint; training is deferred to the next session.")
'''
    round_limit = (
        f'\n    train_command.extend(["--max-rounds-this-session", "{max_rounds_this_session}"])'
        if max_rounds_this_session is not None
        else ""
    )
    training = f'''if PREPROCESSING_PAUSED:
    print("Skipping training in this session because preprocessing will resume first.")
else:
    train_command = [
    sys.executable, str(SOURCE_DIR / "train.py"),
    "--config", str(SOURCE_DIR / "{train_config}"),
    "--prepared-data-dir", str(PREPARED_DIR),
    "--output-dir", str(RUNS_DIR),
    "--upload-checkpoints-to-s3",
    ]
{round_limit}
    if os.environ.get("RUN_ID"):
        train_command.extend(["--run-id", os.environ["RUN_ID"]])
    result = subprocess.run(train_command, cwd=SOURCE_DIR, check=False)
    if result.returncode not in (0, 75):
        raise subprocess.CalledProcessError(result.returncode, train_command)
    if result.returncode == 75:
        print("Session paused only after a verified checkpoint; the watchdog may launch the next session.")
    else:
        print("Training reached iteration 100 and final reporting completed or remains durably retryable.")
'''
    summary = '''active_path = RUNS_DIR / "active_run.json"
if active_path.exists():
    active = json.loads(active_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "run_id": active.get("run_id"),
        "status": active.get("status"),
        "current_iteration": active.get("current_iteration"),
    }, indent=2))
else:
    print("No active run pointer was created.")
'''
    return {
        "cells": [
            markdown_cell(f"# CIC-DDoS2019 LightGBM CPU baseline\n\n{description} Resume/checkpoint state is synchronized with S3."),
            code_cell(extract), code_cell(secrets), code_cell(dependencies),
            code_cell(prepare), code_cell(training), code_cell(summary),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"accelerator": "none", "dataSources": ["dungnguyen28101991/cicddos2019-parquet-per-classes"]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "kaggle_notebook.ipynb"))
    parser.add_argument("--profile", choices=("production", "smoke"), default="production")
    parser.add_argument("--presigned-config", default=None)
    parser.add_argument("--max-rounds-this-session", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rounds_this_session is not None and args.max_rounds_this_session <= 0:
        raise ValueError("--max-rounds-this-session must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            build_notebook(args.profile, args.presigned_config, args.max_rounds_this_session),
            ensure_ascii=False,
            indent=1,
        ) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
