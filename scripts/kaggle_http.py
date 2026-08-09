"""Minimal Kaggle notebook status and push client for OAuth or legacy tokens."""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService"


def clean_secret(value: str) -> str:
    return value.replace("\r", "").replace("\n", "").replace("\\r", "").replace("\\n", "").strip()


def authorization_header() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return f"Bearer {clean_secret(token)}"
    username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError("KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY is missing")
    encoded = base64.b64encode(f"{clean_secret(username)}:{clean_secret(key)}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def post(method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{API_ROOT}/{method}",
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Authorization": authorization_header(), "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kaggle API {method} failed with HTTP {exc.code}: {body[:1000]}") from exc


def split_kernel(kernel: str) -> tuple[str, str]:
    values = kernel.strip().split("/", 1)
    if len(values) != 2 or not all(values):
        raise ValueError("Kaggle kernel must use owner/kernel-slug")
    return values[0], values[1]


def get_kernel_status(kernel: str) -> str:
    owner, slug = split_kernel(kernel)
    response = post("GetKernelSessionStatus", {"userName": owner, "kernelSlug": slug})
    return str(response.get("status", "UNKNOWN"))


def metadata_bool(metadata: Mapping[str, Any], name: str, default: bool) -> bool:
    value = metadata.get(name, default)
    return value if isinstance(value, bool) else str(value).casefold() in {"1", "true", "yes", "on"}


def push_kernel(bundle: Path, timeout: int) -> dict[str, Any]:
    metadata = json.loads((bundle / "kernel-metadata.json").read_text(encoding="utf-8"))
    owner, slug = split_kernel(str(metadata["id"]))
    notebook = json.loads((bundle / str(metadata["code_file"])).read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    payload = {
        "slug": f"{owner}/{slug}", "newTitle": str(metadata["title"]),
        "text": json.dumps(notebook), "language": str(metadata.get("language", "python")),
        "kernelType": str(metadata.get("kernel_type", "notebook")),
        "datasetDataSources": list(metadata.get("dataset_sources", [])),
        "kernelDataSources": list(metadata.get("kernel_sources", [])),
        "competitionDataSources": list(metadata.get("competition_sources", [])),
        "modelDataSources": list(metadata.get("model_sources", [])),
        "isPrivate": metadata_bool(metadata, "is_private", True),
        "enableGpu": metadata_bool(metadata, "enable_gpu", False),
        "enableTpu": metadata_bool(metadata, "enable_tpu", False),
        "enableInternet": metadata_bool(metadata, "enable_internet", True),
        "sessionTimeoutSeconds": int(timeout),
    }
    machine_shape = str(metadata.get("machine_shape", "")).strip()
    if machine_shape:
        payload["machineShape"] = machine_shape
    return post("SaveKernel", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--kernel", required=True)
    push = subparsers.add_parser("push")
    push.add_argument("--path", type=Path, required=True)
    push.add_argument("--timeout", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "status":
        print(f'Kernel has status "{get_kernel_status(args.kernel)}"')
    else:
        response = push_kernel(args.path, args.timeout)
        print(json.dumps({key: response.get(key) for key in ("ref", "url", "versionNumber", "kernelId")}))


if __name__ == "__main__":
    main()
