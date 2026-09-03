#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASELINE = {
    "ViT-L/14 CLIP": 0.2784,
    "ViT-B/16 CLIP": 0.3130,
    "ViT-B/32 CLIP": 0.3131,
    "PIQE": 59.93,
    "MUSIQ": 51.36,
}

HIGHER_IS_BETTER = {
    "ViT-L/14 CLIP": True,
    "ViT-B/16 CLIP": True,
    "ViT-B/32 CLIP": True,
    "PIQE": False,
    "MUSIQ": True,
}


def mean(summary: dict, metric: str) -> float | None:
    if metric.endswith("CLIP"):
        model = metric[: -len(" CLIP")]
        return summary.get("clip", {}).get(model, {}).get("clip_score", {}).get("mean")
    return summary.get("quality", {}).get(metric, {}).get("score", {}).get("mean")


def run_name(path: Path) -> str:
    if path.parent.name == "all_metrics":
        return path.parent.parent.parent.name
    if path.parent.name.endswith("_metrics"):
        return path.parent.name
    parts = path.parts
    if "eval" in parts:
        index = parts.index("eval")
        if index > 0:
            return parts[index - 1]
    return path.parent.name


def collect(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        name = run_name(path)
        for metric, baseline in BASELINE.items():
            value = mean(summary, metric)
            if value is None:
                continue
            delta = value - baseline
            improved = delta > 0 if HIGHER_IS_BETTER[metric] else delta < 0
            rows.append(
                {
                    "run": name,
                    "metric": metric,
                    "value": f"{value:.6f}",
                    "baseline": f"{baseline:.6f}",
                    "delta": f"{delta:.6f}",
                    "improved": "yes" if improved else "no",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "metric", "value", "baseline", "delta", "improved"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], source_paths: list[Path]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["run"], []).append(row)
    lines = [
        "# Text-GS Alignment Dashboard",
        "",
        "Baseline is the supplied HeadStudio table. CLIP and MUSIQ are higher-is-better; PIQE is lower-is-better.",
        "",
        "![Metric bars](metrics_bars.svg)",
        "",
    ]
    for run, run_rows in sorted(grouped.items()):
        lines.extend([f"## {run}", "", "| Metric | Value | Baseline | Delta | Improved |", "| --- | ---: | ---: | ---: | :---: |"])
        for row in run_rows:
            lines.append(
                f"| {row['metric']} | {row['value']} | {row['baseline']} | {row['delta']} | {row['improved']} |"
            )
        lines.append("")
    lines.extend(["## Sources", ""])
    for source in source_paths:
        lines.append(f"- `{source}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(path: Path, rows: list[dict[str, str]]) -> None:
    clip_rows = [row for row in rows if row["metric"].endswith("CLIP")]
    if not clip_rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"640\" height=\"120\"></svg>\n")
        return
    width = 900
    row_height = 28
    left = 220
    chart_width = 560
    height = 70 + len(clip_rows) * row_height
    max_value = max(float(row["value"]) for row in clip_rows + [{"value": "0.34"}])
    lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        "<text x=\"24\" y=\"32\" font-family=\"Arial\" font-size=\"20\" font-weight=\"700\">CLIP metrics vs HeadStudio baseline</text>",
    ]
    y = 62
    for row in clip_rows:
        value = float(row["value"])
        baseline = float(row["baseline"])
        value_w = int(chart_width * value / max_value)
        base_x = left + int(chart_width * baseline / max_value)
        color = "#2f6fed" if row["improved"] == "yes" else "#c44747"
        label = f"{row['run']} / {row['metric']}"
        lines.extend(
            [
                f"<text x=\"24\" y=\"{y + 16}\" font-family=\"Arial\" font-size=\"12\">{label}</text>",
                f"<rect x=\"{left}\" y=\"{y}\" width=\"{value_w}\" height=\"16\" fill=\"{color}\" rx=\"2\"/>",
                f"<line x1=\"{base_x}\" y1=\"{y - 3}\" x2=\"{base_x}\" y2=\"{y + 20}\" stroke=\"#111\" stroke-width=\"1\"/>",
                f"<text x=\"{left + value_w + 8}\" y=\"{y + 13}\" font-family=\"Arial\" font-size=\"12\">{value:.4f}</text>",
            ]
        )
        y += row_height
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RuiHeadStudio alignment metrics.")
    parser.add_argument("summary_json", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = collect(args.summary_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics_comparison.csv", rows)
    write_markdown(args.output_dir / "README.md", rows, args.summary_json)
    write_svg(args.output_dir / "metrics_bars.svg", rows)


if __name__ == "__main__":
    main()
