#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict

import torch
from plyfile import PlyData


def load_opacity_diagnostics_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "threestudio"
        / "utils"
        / "opacity_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("opacity_diagnostics_runtime", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


opacity_diagnostics = load_opacity_diagnostics_module()
summarize_gaussian_regions = opacity_diagnostics.summarize_gaussian_regions


def _format_float(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def read_ply_gaussian_tensors(ply_path: Path) -> Dict[str, torch.Tensor]:
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]

    opacity_logits = torch.tensor(vertex["opacity"], dtype=torch.float32).reshape(-1, 1)
    scale_names = sorted(
        [prop.name for prop in vertex.properties if prop.name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if not scale_names:
        raise ValueError(f"No scale_* properties found in {ply_path}")

    scale_logits = torch.stack(
        [torch.tensor(vertex[name], dtype=torch.float32) for name in scale_names],
        dim=1,
    )
    return {
        "opacity": torch.sigmoid(opacity_logits),
        "scaling": torch.exp(scale_logits),
    }


def write_summary_markdown(path: Path, stats: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = stats["total"]
    lines = [
        "# Opacity Thickness Diagnostics",
        "",
        "## Total",
        "",
        f"- count: {total['count']}",
        f"- opacity mean: {_format_float(total['opacity']['mean'])}",
        f"- scaling max mean: {_format_float(total['scaling_max']['mean'])}",
        "",
        "## Regions",
        "",
        "| region | count | opacity mean | scaling max mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for region in ("front", "side", "rear"):
        region_stats = stats["regions"][region]
        lines.append(
            "| {region} | {count} | {opacity} | {scaling} |".format(
                region=region,
                count=region_stats["count"],
                opacity=_format_float(region_stats["opacity"]["mean"]),
                scaling=_format_float(region_stats["scaling_max"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "Region labels are `unknown` in this PLY-only pass; model-bound normals can be wired in a later GPU pass.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostics(ply_path: Path, output_dir: Path) -> Dict[str, object]:
    tensors = read_ply_gaussian_tensors(ply_path)
    labels = ["unknown"] * int(tensors["opacity"].numel())
    stats = summarize_gaussian_regions(tensors["opacity"], tensors["scaling"], labels)
    stats["label_source"] = "unknown"
    stats["ply"] = str(ply_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "gaussian_region_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_markdown(output_dir / "summary.md", stats)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Gaussian opacity and thickness stats.")
    parser.add_argument("--ply", required=True, type=Path, help="Path to a saved Gaussian PLY.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for gaussian_region_stats.json and summary.md.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_diagnostics(args.ply, args.output)


if __name__ == "__main__":
    main()
