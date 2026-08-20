"""Decide whether GitHub Actions should launch the next Kaggle session."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from .kaggle_http import get_kernel_status
except ImportError:
    from kaggle_http import get_kernel_status

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET_ENV_NAMES = (
    "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_SECRET", "KAGGLE_KEY",
    "KAGGLE_KERNEL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "AWS_DEFAULT_REGION", "S3_BUCKET", "S3_PREFIX",
)


def normalize_secret_environment() -> None:
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value is not None:
            os.environ[name] = value.replace("\r", "").replace("\n", "").replace("\\r", "").replace("\\n", "").strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def normalize_kernel_status(output: str) -> str:
    text = output.casefold()
    for status, tokens in (
        ("running", ("running", "kernelworkerstatus.running")),
        ("queued", ("queued", "pending", "kernelworkerstatus.queued")),
        ("complete", ("complete", "completed", "kernelworkerstatus.complete")),
        ("cancelled", ("cancelled", "canceled")),
        ("error", ("error", "failed", "failure")),
    ):
        if any(token in text for token in tokens):
            return status
    return "unknown"


@dataclass(frozen=True)
class Decision:
    should_push: bool
    reason: str
    current_iteration: int
    kernel_status: str
    session_attempts: int
    stagnant_restarts: int
    report_restarts: int


def made_durable_progress(active: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    current = int(active.get("current_iteration", 0))
    previous_iteration = int(state.get("last_observed_iteration", 0))
    preprocessing_progress = int(active.get("preprocessing_completed_files", 0))
    previous_preprocessing_progress = int(state.get("last_preprocessing_completed_files", 0))
    active_status = str(active.get("status", "missing")).casefold()
    previous_status = str(state.get("last_active_status", "missing")).casefold()
    lifecycle_progress = (
        active_status != previous_status
        and active_status in {"paused", "ready_for_report", "complete"}
    )
    return (
        current > previous_iteration
        or preprocessing_progress > previous_preprocessing_progress
        or lifecycle_progress
    )


def orchestration_state_after_push(
    previous: Mapping[str, Any],
    active: Mapping[str, Any],
    reason: str,
    observed_iteration: int,
    now_utc: str,
) -> dict[str, Any]:
    active_status = str(active.get("status", "missing")).casefold()
    preprocessing_progress = int(active.get("preprocessing_completed_files", 0))
    progress = made_durable_progress(active, previous)
    previous_report_restarts = int(previous.get("report_restarts", 0))
    return {
        "last_push_at": now_utc,
        "last_push_reason": reason,
        "last_observed_iteration": observed_iteration,
        "last_preprocessing_completed_files": preprocessing_progress,
        "last_active_status": active_status,
        "session_attempts": 1 if progress else int(previous.get("session_attempts", 0)) + 1,
        "stagnant_restarts": 0 if progress else int(previous.get("stagnant_restarts", 0)) + 1,
        "report_restarts": previous_report_restarts + 1 if active_status == "ready_for_report" else 0,
    }


def decide_next_session(
    active: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
    kernel_status: str,
    config: Mapping[str, Any],
    now_timestamp: float,
    force: bool = False,
) -> Decision:
    active, state = dict(active or {}), dict(state or {})
    current = int(active.get("current_iteration", 0))
    active_status = str(active.get("status", "missing")).casefold()
    configured_run_id = str(config.get("run_id", "")).strip()
    active_run_id = str(active.get("run_id", "")).strip()
    different_run = bool(configured_run_id and active_run_id and configured_run_id != active_run_id)
    attempts = int(state.get("session_attempts", 0))
    stagnant = int(state.get("stagnant_restarts", 0))
    report_restarts = int(state.get("report_restarts", 0))
    if different_run:
        current = 0
        active_status = "different_run"
        attempts = 0
        stagnant = 0
        report_restarts = 0
    made_progress = made_durable_progress(active, state)
    if made_progress:
        # These limits protect against consecutive launches that make no durable
        # progress. They must not become lifetime counters that permanently lock
        # a healthy long-running job after enough successful resume sessions.
        attempts = 0
        stagnant = 0
    if force:
        return Decision(True, "manual force", current, kernel_status, attempts, stagnant, report_restarts)
    if active_status == "complete" and current == int(config["target_iteration"]):
        return Decision(False, "iteration 100 and final report are complete", current, kernel_status, attempts, stagnant, report_restarts)
    if kernel_status in {"running", "queued"}:
        return Decision(False, f"Kaggle notebook is {kernel_status}", current, kernel_status, attempts, stagnant, report_restarts)
    if active_status == "running" and kernel_status == "unknown":
        heartbeat = parse_timestamp(active.get("updated_at"))
        stale_seconds = float(config["running_heartbeat_stale_hours"]) * 3600
        if heartbeat is None or now_timestamp - heartbeat < stale_seconds:
            return Decision(False, "Kaggle status unknown and S3 heartbeat is not stale", current, kernel_status, attempts, stagnant, report_restarts)
    last_push = parse_timestamp(state.get("last_push_at"))
    if last_push is not None and now_timestamp - last_push < int(config["recent_push_guard_minutes"]) * 60:
        return Decision(False, "recent push guard is active", current, kernel_status, attempts, stagnant, report_restarts)
    if attempts >= int(config["maximum_session_attempts"]):
        return Decision(False, "maximum session attempts reached", current, kernel_status, attempts, stagnant, report_restarts)
    if different_run:
        return Decision(
            True,
            f"configured run {configured_run_id} has not started",
            current,
            kernel_status,
            attempts,
            stagnant,
            report_restarts,
        )
    if active_status == "ready_for_report":
        if report_restarts >= int(config["maximum_report_restarts"]):
            return Decision(False, "maximum report restarts reached", current, kernel_status, attempts, stagnant, report_restarts)
        return Decision(
            True,
            "iteration 100 is durable but final reporting needs a session",
            current,
            kernel_status,
            attempts,
            stagnant,
            report_restarts,
        )
    if kernel_status == "cancelled":
        return Decision(
            True,
            "previous Kaggle session is cancelled and run is incomplete",
            current,
            kernel_status,
            attempts,
            stagnant,
            report_restarts,
        )
    if stagnant >= int(config["maximum_stagnant_restarts"]):
        return Decision(False, "maximum stagnant restarts reached", current, kernel_status, attempts, stagnant, report_restarts)
    if active_status == "paused":
        reason = "verified checkpoint is resumable"
    elif kernel_status in {"complete", "error"}:
        reason = f"previous Kaggle session is {kernel_status} and run is incomplete"
    elif not active:
        reason = "no active run exists"
    else:
        reason = "run is incomplete and no Kaggle session is active"
    return Decision(True, reason, current, kernel_status, attempts, stagnant, report_restarts)


class S3State:
    def __init__(self) -> None:
        import boto3
        self.bucket = os.environ["S3_BUCKET"].strip()
        self.prefix = os.environ["S3_PREFIX"].strip().strip("/")
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.client = boto3.client("s3", region_name=region or None)

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def read_json(self, name: str) -> dict[str, Any] | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key(name))
        except Exception as exc:
            response = getattr(exc, "response", {}) or {}
            if response.get("Error", {}).get("Code") in {"NoSuchKey", "NotFound", "404"} or response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return None
            raise
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        body = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=self.key(name), Body=body)


def kernel_status(kernel: str) -> str:
    try:
        output = f'Kernel has status "{get_kernel_status(kernel)}"'
    except Exception as exc:
        return "unknown"
    return normalize_kernel_status(output)


def write_github_output(path: str | None, decision: Decision) -> None:
    values = {
        "should_push": str(decision.should_push).lower(), "reason": decision.reason.replace("\n", " "),
        "current_iteration": str(decision.current_iteration), "kernel_status": decision.kernel_status,
    }
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            for key, value in values.items():
                handle.write(f"{key}={value}\n")
    print(json.dumps(asdict(decision), indent=2))


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command_decide(args: argparse.Namespace) -> None:
    config, store = load_config(args.config), S3State()
    decision = decide_next_session(
        store.read_json("active_run.json"), store.read_json("orchestration_state.json"),
        kernel_status(args.kernel), config, time.time(), args.force,
    )
    write_github_output(args.github_output, decision)


def command_record(args: argparse.Namespace) -> None:
    store = S3State()
    previous = store.read_json("orchestration_state.json") or {}
    active = store.read_json("active_run.json") or {}
    observed = int(active.get("current_iteration", args.observed_iteration))
    state = orchestration_state_after_push(previous, active, args.reason, observed, utc_now())
    store.write_json("orchestration_state.json", state)
    print(json.dumps(state, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "orchestration.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    decide = commands.add_parser("decide")
    decide.add_argument("--kernel", required=True)
    decide.add_argument("--github-output", default=None)
    decide.add_argument("--force", action="store_true")
    decide.set_defaults(func=command_decide)
    record = commands.add_parser("record-push")
    record.add_argument("--reason", required=True)
    record.add_argument("--observed-iteration", type=int, default=0)
    record.set_defaults(func=command_record)
    return parser.parse_args()


def main() -> None:
    normalize_secret_environment()
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
