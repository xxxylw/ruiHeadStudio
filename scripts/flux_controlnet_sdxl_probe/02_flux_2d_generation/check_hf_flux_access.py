#!/usr/bin/env python
"""Check Hugging Face access required for Slice 02 without loading model weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, whoami


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--controlnet-model", default="Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0")
    return parser.parse_args()


def try_download(repo_id: str, filename: str) -> dict[str, object]:
    try:
        path = hf_hub_download(repo_id, filename)
    except Exception as exc:  # pragma: no cover - report script
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "path": path}


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        identity = whoami()
    except Exception as exc:  # pragma: no cover - report script
        identity = {"error": f"{type(exc).__name__}: {exc}"}

    report = {
        "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "whoami": identity,
        "base_model": {
            "repo": args.base_model,
            "model_index": try_download(args.base_model, "model_index.json"),
        },
        "controlnet_model": {
            "repo": args.controlnet_model,
            "config": try_download(args.controlnet_model, "config.json"),
        },
    }
    report["ok"] = bool(
        report["base_model"]["model_index"]["ok"]
        and report["controlnet_model"]["config"]["ok"]
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
