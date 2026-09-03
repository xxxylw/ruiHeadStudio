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

from evaluation.src.clip_metrics import CLIP_MODELS, score_clip
from evaluation.src.dataset import load_examples
from evaluation.src.quality_metric import score_musiq, score_piqe
from evaluation.src.report import render_chinese_report, render_markdown
from evaluation.src.statistics import summarize, summarize_retrieval


def selected_clip_models(metric_name: str) -> tuple[str, ...]:
    selections = {
        "all": CLIP_MODELS,
        "clip_b32": ("ViT-B/32",),
        "clip_b16": ("ViT-B/16",),
        "clip_l14": ("ViT-L/14",),
        "piqe": (),
        "musiq": (),
    }
    return selections[metric_name]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(clip_rows: list[dict], quality_rows: list[dict]) -> dict:
    result = {"clip": {}, "quality": {}}
    for model_name in CLIP_MODELS:
        rows = [row for row in clip_rows if row["model"] == model_name]
        result["clip"][model_name] = {
            "clip_score": summarize([row["score"] for row in rows]),
            "retrieval": summarize_retrieval([row["rank"] for row in rows]),
        }
    for quality_name in sorted({row["model"] for row in quality_rows}):
        rows = [row for row in quality_rows if row["model"] == quality_name]
        result["quality"][quality_name] = {"score": summarize([row["score"] for row in rows])}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final HeadStudio images.")
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--metrics", choices=("all", "clip_b32", "clip_b16", "clip_l14", "piqe", "musiq"), default="all"
    )
    args = parser.parse_args()

    examples = load_examples(args.batch_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics != "all":
        if args.metrics == "piqe":
            _write_csv(args.output_dir / "per_image_metrics.csv", score_piqe(examples, args.device))
            return
        if args.metrics == "musiq":
            _write_csv(args.output_dir / "per_image_metrics.csv", score_musiq(examples, args.device))
            return
        clip_rows = score_clip(examples, selected_clip_models(args.metrics)[0], args.device, args.batch_size)
        _write_csv(args.output_dir / "clip_retrieval.csv", clip_rows)
        _write_csv(args.output_dir / "per_image_metrics.csv", clip_rows)
        return

    clip_rows: list[dict] = []
    for model_name in selected_clip_models(args.metrics):
        clip_rows.extend(score_clip(examples, model_name, args.device, args.batch_size))
    quality_rows = score_piqe(examples, args.device) + score_musiq(examples, args.device)
    summary = _summary(clip_rows, quality_rows)

    _write_csv(args.output_dir / "per_image_metrics.csv", clip_rows + quality_rows)
    _write_csv(args.output_dir / "clip_retrieval.csv", clip_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {"python": sys.version, "platform": platform.platform(), "device": args.device,
             "batch_root": str(args.batch_root), "image_count": len(examples)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_chinese_report(summary, str(args.batch_root), str(args.output_dir)), encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
