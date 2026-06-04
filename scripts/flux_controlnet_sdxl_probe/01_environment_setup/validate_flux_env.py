#!/usr/bin/env python
"""Validate the isolated FLUX ControlNet experiment environment."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    "torch",
    "diffusers",
    "transformers",
    "huggingface_hub",
    "mediapipe",
    "cv2",
    "pytorch3d",
    "pytorch3d.renderer",
    "xformers",
    "nerfacc",
    "nvdiffrast.torch",
    "tinycudann",
    "diff_gaussian_rasterization",
    "simple_knn._C",
    "smplx",
    "controlnet_aux",
]

REQUIRED_DIFFUSERS_SYMBOLS = [
    ("diffusers", "FluxControlNetPipeline"),
    ("diffusers", "FluxControlNetModel"),
]


def import_status(module_name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - report script
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    version = getattr(module, "__version__", None)
    return {"ok": True, "version": str(version) if version is not None else ""}


def symbol_status(module_name: str, symbol_name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(module_name)
        getattr(module, symbol_name)
    except Exception as exc:  # pragma: no cover - report script
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "symbol": symbol_name}


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    imports = {name: import_status(name) for name in REQUIRED_IMPORTS}
    symbols = {
        f"{module}.{symbol}": symbol_status(module, symbol)
        for module, symbol in REQUIRED_DIFFUSERS_SYMBOLS
    }

    report = {
        "python": sys.executable,
        "version": sys.version,
        "imports": imports,
        "symbols": symbols,
    }
    report["ok"] = all(item["ok"] for item in imports.values()) and all(
        item["ok"] for item in symbols.values()
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
