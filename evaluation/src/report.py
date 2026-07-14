from __future__ import annotations

from typing import Any


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# HeadStudio Quantitative Evaluation", "", "## Main metrics", ""]
    clip = summary.get("clip", {})
    if clip:
        lines.extend(["| Model | CLIP Score |", "| --- | ---: |"])
        for model_name, model_summary in clip.items():
            lines.append(f"| {model_name} | {model_summary['clip_score']['mean']:.6f} |")
    quality = summary.get("quality", {})
    if quality:
        lines.extend(["", "## No-reference image quality", "", "| Metric | Score |", "| --- | ---: |"])
        for name, values in quality.items():
            lines.append(f"| {name} | {values['score']['mean']:.6f} |")
    return "\n".join(lines) + "\n"


def render_chinese_report(summary: dict[str, Any], batch_root: str, output_dir: str) -> str:
    lines = [
        "# HeadStudio 量化实验结果",
        "",
        "## 评测协议",
        "",
        f"- 评测集：`{batch_root}` 中 26 条 prompt 的最终 4 个视角，共 104 张 `it10000-*.png`。",
        "- CLIP Score：图像与其原始 prompt 的 L2 归一化特征余弦相似度；每种模型在 104 个图文对上取均值。",
        "- 检索：每张图在全部 26 条原始 prompt 中对自身 prompt 的排名；报告 Recall@1、MRR 与平均排名。",
        "- PIQE 和 MUSIQ 均为无参考画面质量辅助指标，不能解释为语义正确性或几何重建精度。",
        "",
        "## 主要结果",
        "",
        "| CLIP 模型 | CLIP Score ↑ | Recall@1 ↑ | MRR ↑ | Mean Rank ↓ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name, values in summary["clip"].items():
        score = values["clip_score"]["mean"]
        retrieval = values["retrieval"]
        lines.append(
            f"| {model_name} | {score:.6f} | {retrieval['recall_at_1']:.6f} | "
            f"{retrieval['mrr']:.6f} | {retrieval['mean_rank']:.6f} |"
        )
    lines.extend(["", "| 画面质量指标 | Score | Std | 95% CI |", "| --- | ---: | ---: | ---: |"])
    for name, values in summary["quality"].items():
        score = values["score"]
        direction = "越低越好" if name == "PIQE" else "越高越好"
        lines.append(
            f"| {name}（{direction}） | {score['mean']:.6f} | {score['std']:.6f} | "
            f"[{score['ci95_low']:.6f}, {score['ci95_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## 限制与复现",
            "",
            "本批数据没有真值 mesh、扫描或配对参考图，因此未报告 Chamfer Distance、F-score、法线一致性、PSNR、SSIM 或 LPIPS；这些指标不能由当前数据有效计算。",
            f"逐图明细、汇总 JSON 与运行环境记录位于：`{output_dir}`。",
        ]
    )
    return "\n".join(lines) + "\n"
