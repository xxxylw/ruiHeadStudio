# Text GS Alignment Nightly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an overnight RuiHeadStudio experiment that improves HeadStudio text-to-3D head metrics while leaving committed method, data, and visualization records.

**Architecture:** Add a backward-compatible multi-component CLIP alignment loss to the existing HeadStudio training system. Launch three parallel single-prompt ablations on the server GPUs, evaluate each run with the existing evaluator, and summarize results in Markdown, CSV, and SVG artifacts.

**Tech Stack:** Python, PyTorch, threestudio/RuiHeadStudio, Bash, conda environment `ruiheadstudio`, existing `evaluation/run_evaluation.py`.

---

### Task 1: Multi-Component CLIP Loss

**Files:**
- Modify: `threestudio/systems/Head3DGSLKs.py`
- Modify: `configs/headstudio.yaml`
- Modify: `tests/test_clip_alignment.py`

- [ ] Add `clip_global_weight`, `clip_foreground_weight`, and `clip_view_weight` config fields with default `0.0`.
- [ ] Keep legacy `clip_foreground_only` and `clip_use_view_prompt` behavior when all component weights are zero.
- [ ] When component weights are positive, compute weighted global full-frame, foreground crop, and foreground view-conditioned CLIP losses, normalized by positive weight sum.
- [ ] Log `train/loss_clip_global`, `train/loss_clip_foreground`, and `train/loss_clip_view`.
- [ ] Run `pytest tests/test_clip_alignment.py -q`.

### Task 2: Experiment Documentation

**Files:**
- Create: `docs/experiments/2026-08-27-text-gs-alignment-nightly.md`

- [ ] Record baseline metrics supplied by the project owner.
- [ ] Record the method hypothesis: region-aware multi-view text alignment for 3D Gaussian heads.
- [ ] Record the three overnight ablations, exact prompt, exact flags, output root, metrics, and success gates.

### Task 3: Dashboard Generator

**Files:**
- Create: `scripts/summarize_alignment_dashboard.py`

- [ ] Read one or more `summary.json` metric outputs.
- [ ] Compare each run against the fixed HeadStudio baseline table.
- [ ] Write `metrics_comparison.csv`, `README.md`, and `metrics_bars.svg` under a requested output directory.
- [ ] Run the script on existing `outputs/clip_dual_scale_20260804/eval/*_metrics/summary.json` files as a smoke test.

### Task 4: Overnight Launcher

**Files:**
- Create: `scripts/run_text_gs_alignment_nightly.sh`

- [ ] Launch three variants in parallel on GPUs 0, 1, and 2.
- [ ] Use conda env `ruiheadstudio`, offline model flags, fixed Elon Musk prompt, `trainer.max_steps=10000`, and `guidance_scale=25`.
- [ ] Write one manifest per variant so `evaluation/run_evaluation.py` can score final renders.
- [ ] Run all metrics for each successful variant and call the dashboard generator at the end.

### Task 5: Keepalive

**Files:**
- Create locally: `D:\_materials\_code\VocabularyLearning\scripts\keep_server_ssh_alive.ps1`

- [ ] Start a PowerShell process that sends one tiny SSH command to `server` every 60 seconds.
- [ ] Log timestamps to `D:\_materials\_code\VocabularyLearning\logs\server_keepalive.log`.

### Task 6: Commit and Launch

**Files:**
- All modified server repository files.

- [ ] Run tests.
- [ ] Commit method, scripts, and docs on `codex/text-gs-alignment-nightly`.
- [ ] Start keepalive.
- [ ] Start `scripts/run_text_gs_alignment_nightly.sh` with `nohup`.
- [ ] Verify log files, background PIDs, and GPU processes.
