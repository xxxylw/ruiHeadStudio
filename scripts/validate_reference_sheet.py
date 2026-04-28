#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_reference_sheet_func():
    module_path = Path(__file__).resolve().parents[1] / "threestudio" / "utils" / "reference_sheet.py"
    spec = importlib.util.spec_from_file_location("reference_sheet_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_reference_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RuiHeadStudio reference sheet metadata.")
    parser.add_argument("metadata", help="Path to reference sheet metadata.json")
    args = parser.parse_args()

    sheet = load_reference_sheet_func()(args.metadata)
    print(f"identity_mode: {sheet.identity_mode}")
    print(f"prompt: {sheet.prompt}")
    print(f"references: {len(sheet.references)}")
    for ref in sheet.references:
        print(f"- {ref.view}: {ref.image_path} weight={ref.weight}")


if __name__ == "__main__":
    main()
