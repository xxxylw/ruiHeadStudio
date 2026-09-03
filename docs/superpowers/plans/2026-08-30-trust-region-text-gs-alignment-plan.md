# Trust-Region Text-GS Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a normalized Gaussian trust-region loss and late CLIP decay so semantic refinement improves alignment while preserving perceptual quality.

**Architecture:** Keep the existing three-component CLIP path in `Head3DGSLKsRig`. Snapshot the continuation checkpoint's Gaussian parameters after loading, compute a detached-anchor drift penalty during training, and expose a configurable linear CLIP decay window. Put the pure penalty math in `clip_alignment.py` for focused unit tests.

**Tech Stack:** Python 3.9, PyTorch, pytest, threestudio YAML overrides, SSH/conda experiment scripts.

---

### Task 1: Add pure trust-region math tests

**Files:**
- Create: `tests/test_clip_alignment.py` if absent, otherwise extend its existing alignment tests
- Test: `tests/test_clip_alignment.py`

- [ ] **Step 1: Write failing tests** for `clip_decay_weight` and `normalized_parameter_drift`: decay equals the base before the window, reaches zero at the end, zero drift returns zero, gradients flow to current tensors, and anchor tensors receive no gradient.
- [ ] **Step 2: Run `pytest -q tests/test_clip_alignment.py`** and confirm the new symbols fail before implementation.
- [ ] **Step 3: Implement the two pure helpers in `threestudio/models/clip_alignment.py`** with explicit shape validation and epsilon-stabilized normalization.
- [ ] **Step 4: Run the focused test file** and require all tests to pass.
- [ ] **Step 5: Commit** with `test: cover trust-region alignment math`.

### Task 2: Integrate anchored drift and CLIP decay

**Files:**
- Modify: `threestudio/systems/Head3DGSLKs.py`
- Modify: `threestudio/models/clip_alignment.py`

- [ ] **Step 1: Add config fields** `lambda_trust`, `trust_clip_decay_steps`, `trust_xyz_weight`, `trust_scaling_weight`, `trust_opacity_weight`, and `trust_feature_weight` with zero-preserving defaults.
- [ ] **Step 2: Snapshot detached clones** of `get_xyz`, `_scaling`, `_opacity`, and `_features_dc`/feature tensor after Gaussian initialization, with a clear runtime error if a requested tensor is unavailable.
- [ ] **Step 3: Add the normalized drift term** after CLIP alignment activates, log total and component losses, and multiply by `lambda_trust`.
- [ ] **Step 4: Replace the constant CLIP multiplier** with `clip_decay_weight(...)` over the configured final window; keep old behavior when decay is zero.
- [ ] **Step 5: Run the existing alpha/CLIP focused tests plus a Python import/compile check.**
- [ ] **Step 6: Commit** with `feat: add trust-region semantic refinement`.

### Task 3: Add a reproducible smoke launcher

**Files:**
- Create: `scripts/launch_trust_region_smoke.sh`
- Create: `scripts/evaluate_trust_region_smoke.sh`

- [ ] **Step 1: Launch one batch-1, 3,000-step continuation** from the known round-2 checkpoint with `lambda_clip=0.003`, component weights `0.20/0.55/0.25`, `lambda_trust=0.02`, and a 1,000-step decay window.
- [ ] **Step 2: Wait for the PID, build the exact manifest, and run all five offline metrics on CPU.**
- [ ] **Step 3: Verify logs include `loss_trust` and the decayed CLIP weight, then commit the scripts** with `exp: add trust-region smoke experiment`.

### Task 4: Publish evidence

**Files:**
- Modify: `docs/experiments/2026-08-27-text-gs-alignment-nightly.md`
- Modify: `outputs/text_gs_alignment_trust_region_20260830/dashboard/README.md`
- Modify: `outputs/text_gs_alignment_trust_region_20260830/dashboard/metrics_comparison.csv`
- Modify: `outputs/text_gs_alignment_trust_region_20260830/dashboard/metrics_bars.svg`

- [ ] **Step 1: Run the smoke evaluation and record exact means, counts, and comparison deltas.**
- [ ] **Step 2: Regenerate the dashboard with the repository summarizer.**
- [ ] **Step 3: Add a concise method/result/limitation section to the experiment Markdown.**
- [ ] **Step 4: Verify git diff, summary JSON, CSV, SVG, and keepalive log; commit with `docs: record trust-region results` only after metrics exist.**
