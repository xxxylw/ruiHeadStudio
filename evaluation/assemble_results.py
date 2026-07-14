from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_evaluation import _summary, _write_csv
from evaluation.src.report import render_chinese_report, render_markdown


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assemble(
    clip_rows: list[dict],
    quality_rows: list[dict],
    batch_root: Path,
    output_dir: Path,
    device: str,
) -> dict:
    """Write final reproducible artifacts from separately computed metric rows."""
    if not clip_rows or not quality_rows:
        raise ValueError("CLIP and quality inputs must both be non-empty")
    for row in clip_rows + quality_rows:
        row["score"] = float(row["score"])
        if "rank" in row and row["rank"] != "":
            row["rank"] = int(row["rank"])

    summary = _summary(clip_rows, quality_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_image_metrics.csv", clip_rows + quality_rows)
    _write_csv(output_dir / "clip_retrieval.csv", clip_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "device": device,
                "batch_root": str(batch_root),
                "image_count": len({row["image_path"] for row in quality_rows}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_chinese_report(summary, str(batch_root), str(output_dir)), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble HeadStudio metric artifacts.")
    parser.add_argument("--clip-dir", type=Path, action="append", required=True)
    parser.add_argument("--quality-dir", type=Path, action="append", required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()

    clip_rows = [row for directory in args.clip_dir for row in _read_csv(directory / "per_image_metrics.csv")]
    quality_rows = [
        row
        for directory in args.quality_dir
        for row in _read_csv(directory / "per_image_metrics.csv")
    ]
    assemble(clip_rows, quality_rows, args.batch_root, args.output_dir, args.device)


if __name__ == "__main__":
    main()
