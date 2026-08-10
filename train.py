"""Train the fixed 100-round CPU LightGBM baseline with resumable checkpoints.

This module intentionally contains no matplotlib imports or plotting code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
import sklearn

from checkpoint import CheckpointManager, atomic_json_dump, canonical_hash
from model import (
    IterationRecorder,
    TrainingPauseRequested,
    build_datasets,
    macro_f1_metric,
    validate_training_config,
)


LOGGER = logging.getLogger(__name__)
PAUSED_EXIT_CODE = 75


def load_train_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        if config_path.suffix.casefold() == ".json":
            config = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configuration requires PyYAML; JSON needs no extra dependency") from exc
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training configuration root must be an object")
    required = {
        "project_name", "model_name", "experiment_role", "seed", "device",
        "num_boost_round", "early_stopping", "imbalance_handling", "feature_selection",
        "use_all_train_rows", "model_params", "dataset", "checkpoint", "session", "s3", "logging",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing training configuration keys: {missing}")
    validate_training_config(config)
    return config


def session_deadline(config: Mapping[str, Any]) -> float | None:
    maximum_hours = float(config["session"]["maximum_hours"])
    stop_before_minutes = float(config["session"]["stop_before_minutes"])
    if maximum_hours <= 0:
        return None
    usable_seconds = maximum_hours * 3600.0 - stop_before_minutes * 60.0
    if usable_seconds <= 0:
        raise ValueError("session.stop_before_minutes must be less than session.maximum_hours")
    return time.monotonic() + usable_seconds


def validate_resume_state(
    state: Mapping[str, Any], run_id: str, params_hash: str, feature_schema_hash: str, target: int
) -> int:
    checks = {
        "run_id": (state.get("run_id"), run_id),
        "params_hash": (state.get("params_hash"), params_hash),
        "feature_schema_hash": (state.get("feature_schema_hash"), feature_schema_hash),
        "target_iteration": (int(state.get("target_iteration", -1)), int(target)),
    }
    failures = {key: {"observed": observed, "expected": expected} for key, (observed, expected) in checks.items() if observed != expected}
    if failures:
        raise ValueError(f"Checkpoint is incompatible with this run: {failures}")
    current = int(state["current_iteration"])
    if not 0 <= current <= int(target):
        raise ValueError(f"Checkpoint iteration {current} is outside 0..{target}")
    return current


def remaining_rounds(current_iteration: int, target_iteration: int) -> int:
    remaining = int(target_iteration) - int(current_iteration)
    if remaining < 0:
        raise ValueError("Checkpoint already exceeds the configured target iteration")
    return remaining


def _session_id() -> str:
    return f"session_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _environment_metadata(lightgbm_version: str, num_threads: int) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "lightgbm_version": lightgbm_version,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "configured_num_threads": int(num_threads),
        "total_ram_bytes": int(memory.total),
    }


def _write_run_configuration(
    run_dir: Path,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    params_hash: str,
    feature_schema_hash: str,
    label_mapping: Mapping[str, int],
    feature_names: list[str],
    lightgbm_version: str,
) -> tuple[Path, Path]:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(config)
    run_config.update({
        "run_id": run_dir.name,
        "experiment_role": "baseline_control",
        "num_boost_round": 100,
        "early_stopping": False,
        "imbalance_handling": "none",
        "feature_selection": "none",
        "use_all_train_rows": True,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "learning_rate": 0.001,
        "params_hash": params_hash,
        "feature_schema_hash": feature_schema_hash,
        "feature_count": len(feature_names),
        "num_classes": len(label_mapping),
        "environment": _environment_metadata(lightgbm_version, int(params["num_threads"])),
    })
    run_path = config_dir / "run_config.json"
    params_path = config_dir / "model_params.json"
    atomic_json_dump(run_config, run_path)
    atomic_json_dump(dict(params), params_path)
    return run_path, params_path


def _copy_prepared_metadata(prepared: Path, run_dir: Path) -> list[Path]:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in ("preprocessing.json", "data_profile.json", "sample_manifest.json", "label_mapping.json"):
        source = prepared / name
        if not source.exists():
            raise FileNotFoundError(f"Required prepared-data artifact is missing: {source}")
        destination = config_dir / name
        shutil.copyfile(source, destination)
        copied.append(destination)
    report_source = Path(__file__).resolve().parent / "config" / "report.json"
    report_destination = config_dir / "report_config.json"
    shutil.copyfile(report_source, report_destination)
    copied.append(report_destination)
    return copied


def _run_final_reporting(
    manager: CheckpointManager,
    run_id: str,
    run_dir: Path,
    booster: Any,
    test_features: Any,
    test_labels: np.ndarray,
) -> bool:
    try:
        from make_report import evaluate_final_model

        evaluate_final_model(
            run_dir,
            booster,
            test_features,
            test_labels,
            callback=lambda path, category: manager.sync_artifact(run_id, path, category),
        )
        return True
    except Exception:
        LOGGER.warning(
            "Final evaluation/reporting failed; final_model_round_100.txt and checkpoints remain intact",
            exc_info=True,
        )
        return False


def train(args: argparse.Namespace) -> int:
    config = load_train_config(args.config)
    if args.prepared_data_dir is not None:
        config["dataset"]["prepared_data_dir"] = args.prepared_data_dir
    if args.max_rounds_this_session is not None:
        if args.max_rounds_this_session <= 0:
            raise ValueError("--max-rounds-this-session must be positive")
        config["session"]["max_rounds_this_session"] = args.max_rounds_this_session
    logging.basicConfig(
        level=getattr(logging, str(config["logging"]["level"]).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Install lightgbm>=4.0,<5 before training") from exc

    output_root = Path(args.output_dir)
    manager = CheckpointManager(
        output_root=output_root,
        model_name=str(config["model_name"]),
        checkpoint_config=config["checkpoint"],
        s3_config=config["s3"],
        s3_enabled_override=True if args.upload_checkpoints_to_s3 else None,
    )
    run_id = manager.resolve_run_id(args.run_id)
    run_dir = manager.run_dir(run_id)
    prepared = Path(config["dataset"]["prepared_data_dir"])
    LOGGER.info("Preparing exactly one LightGBM Dataset for each train/validation/test split")
    bundle = build_datasets(prepared, config)
    target = int(config["num_boost_round"])
    manager.download_resume_state(run_id)
    loaded = manager.load_state(run_id)
    if loaded is None:
        current_iteration = 0
        history: list[dict[str, Any]] = []
        init_model: str | None = None
        _copy_prepared_metadata(prepared, run_dir)
        _write_run_configuration(
            run_dir, config, bundle.params, bundle.params_hash, bundle.feature_schema_hash,
            bundle.label_mapping, bundle.feature_names, lgb.__version__,
        )
        for path in sorted((run_dir / "config").glob("*.json")):
            manager.sync_artifact(run_id, path, "config")
    else:
        state, history = loaded
        current_iteration = validate_resume_state(
            state, run_id, bundle.params_hash, bundle.feature_schema_hash, target
        )
        init_model = str(run_dir / "checkpoints" / "last_model.txt")
        LOGGER.info("Resuming run %s from completed iteration %d", run_id, current_iteration)

    if remaining_rounds(current_iteration, target) == 0:
        booster = lgb.Booster(model_file=str(run_dir / "checkpoints" / "last_model.txt"))
        if int(booster.current_iteration()) != target:
            raise AssertionError("Serialized final Booster does not contain exactly 100 iterations")
        manager.save_final_model(run_id, booster, target)
        manager.set_run_status(run_id, "ready_for_report", target)
        reporting_complete = _run_final_reporting(
            manager, run_id, run_dir, booster, bundle.features["test"], bundle.labels["test"]
        )
        manager.set_run_status(
            run_id, "complete" if reporting_complete else "ready_for_report", target
        )
        LOGGER.info("Run %s was already complete at iteration %d", run_id, target)
        return 0

    session_id = _session_id()
    manager.set_run_status(run_id, "running", current_iteration)
    checkpoint_state: dict[str, Any] = {}

    def checkpoint_hook(booster: Any, current_history: list[dict[str, Any]], status: str) -> float:
        nonlocal checkpoint_state
        checkpoint_state, checkpoint_seconds = manager.save_checkpoint(
            run_id=run_id,
            session_id=session_id,
            booster=booster,
            history=current_history,
            params_hash=bundle.params_hash,
            feature_schema_hash=bundle.feature_schema_hash,
            target_iteration=target,
            status=status,
        )
        current_history[-1]["checkpoint_seconds"] = checkpoint_seconds
        manager.update_history_after_checkpoint(run_id, checkpoint_state, current_history)
        manager.set_run_status(run_id, status, int(booster.current_iteration()))
        try:
            from viz import generate_incremental_reports

            generate_incremental_reports(
                run_dir,
                callback=lambda path, category: manager.sync_artifact(run_id, path, category),
            )
        except Exception:
            LOGGER.warning(
                "Incremental plotting failed after iteration %d; checkpoint remains valid",
                int(booster.current_iteration()),
                exc_info=True,
            )
        return checkpoint_seconds

    recorder = IterationRecorder(
        history=history,
        session_id=session_id,
        target_iteration=target,
        learning_rate=float(bundle.params["learning_rate"]),
        checkpoint_interval=int(config["checkpoint"]["interval_rounds"]),
        checkpoint_hook=checkpoint_hook,
        deadline_monotonic=session_deadline(config),
        max_rounds_this_session=config["session"].get("max_rounds_this_session"),
        session_start_iteration=current_iteration,
        maximum_session_hours=float(config["session"]["maximum_hours"]),
        stop_before_minutes=float(config["session"]["stop_before_minutes"]),
    )
    callbacks = [recorder]
    period = int(config["logging"]["lightgbm_period"])
    if period > 0:
        callbacks.append(lgb.log_evaluation(period=period))
    try:
        booster = lgb.train(
            params=bundle.params,
            train_set=bundle.train_dataset,
            num_boost_round=remaining_rounds(current_iteration, target),
            valid_sets=[bundle.train_dataset, bundle.validation_dataset],
            valid_names=["train", "validation"],
            feval=macro_f1_metric(len(bundle.label_mapping)),
            init_model=init_model,
            keep_training_booster=True,
            callbacks=callbacks,
        )
    except TrainingPauseRequested as exc:
        LOGGER.info("%s; next session will resume at iteration %d", exc, len(history) + 1)
        return PAUSED_EXIT_CODE

    final_iteration = int(booster.current_iteration())
    if final_iteration != target or len(history) != target:
        raise AssertionError(
            f"Training returned without exactly 100 iterations: booster={final_iteration}, history={len(history)}"
        )
    final_path = manager.save_final_model(run_id, booster, target)
    manager.set_run_status(run_id, "ready_for_report", target)
    LOGGER.info("Completed fixed baseline at iteration 100: %s", final_path)
    reporting_complete = _run_final_reporting(
        manager, run_id, run_dir, booster, bundle.features["test"], bundle.labels["test"]
    )
    manager.set_run_status(
        run_id, "complete" if reporting_complete else "ready_for_report", target
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.json")
    parser.add_argument("--prepared-data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-rounds-this-session", type=int, default=None)
    parser.add_argument("--upload-checkpoints-to-s3", action="store_true")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(train(parse_args()))


if __name__ == "__main__":
    main()
