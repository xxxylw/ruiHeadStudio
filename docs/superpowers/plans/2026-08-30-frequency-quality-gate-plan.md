# Frequency-Gated Text-GS Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a late-stage, rendered-image frequency quality gate to text-to-3DGS alignment and evaluate three small ablations against the five HeadStudio metrics.

**Architecture:** Keep CLIP alignment and SDS unchanged, add pure differentiable image-frequency helpers, and apply the quality loss in `Head3DGSLKs.training_step` with a ramp schedule. The launcher starts independent continuations from the strongest semantic checkpoint; the evaluator uses `HEADSTUDIO_FINAL_STEP` so continuation endpoints are explicit.

**Tech Stack:** PyTorch, Lightning/Threestudio, pytest, Bash, offline CLIP/PIQE/MUSIQ evaluator, CSV/SVG dashboard.

---

### Task 1: Add failing helper tests

**Files:**
- Modify: `tests/test_clip_alignment.py`
- Test: `tests/test_clip_alignment.py`

- [ ] **Step 1: Write the failing tests**

Add tests for `frequency_quality_loss`:

```python
def test_frequency_quality_loss_is_zero_for_constant_image():
    image = torch.full((1, 3, 8, 8), 0.5, requires_grad=True)
    loss = frequency_quality_loss(image)
    assert torch.isclose(loss, torch.zeros_like(loss))

def test_frequency_quality_loss_has_finite_image_gradients_and_accepts_alpha():
    image = torch.rand((1, 3, 8, 8), requires_grad=True)
    alpha = torch.zeros((1, 8, 8, 1))
    alpha[:, 2:6, 2:6] = 1.0
    loss = frequency_quality_loss(image, alpha)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(image.grad).all()

def test_frequency_quality_loss_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        frequency_quality_loss(torch.zeros((3, 8, 8)))
```

- [ ] **Step 2: Run the focused test and verify the expected missing-symbol failure**

Run: `source miniconda; conda activate ruiheadstudio; PYTHONPATH=. pytest -q tests/test_clip_alignment.py -k frequency_quality_loss`

Expected: FAIL because `frequency_quality_loss` is not defined.

### Task 2: Implement the pure frequency helper

**Files:**
- Modify: `threestudio/models/clip_alignment.py`
- Test: `tests/test_clip_alignment.py`

- [ ] **Step 1: Implement minimal image and alpha normalization**

Use `[B,3,H,W]` RGB input, accept alpha as `[B,H,W]`, `[B,H,W,1]`, or `[B,1,H,W]`, multiply each pixel by `0.25 + 0.75 * detached_alpha`, and compute mean absolute horizontal/vertical differences plus a 3x3 Laplacian response at full and half resolution.

- [ ] **Step 2: Run focused tests**

Run: `source miniconda; conda activate ruiheadstudio; PYTHONPATH=. pytest -q tests/test_clip_alignment.py -k frequency_quality_loss`

Expected: all frequency tests pass with finite gradients.

- [ ] **Step 3: Commit the helper and tests**

Run: `git add threestudio/models/clip_alignment.py tests/test_clip_alignment.py && git commit -m "feat: add differentiable frequency quality loss"`

### Task 3: Wire the scheduled loss into Head3DGSLKs

**Files:**
- Modify: `threestudio/systems/Head3DGSLKs.py`
- Modify: `configs/headstudio_retry.yaml`
- Test: `tests/test_clip_alignment.py`

- [ ] **Step 1: Add config fields with disabled defaults**

Add `quality_start_step: int = 0`, `quality_ramp_end_step: int = 0`, and `lambda_frequency_quality: float = 0.0` to the system config and expose matching YAML keys. Keep the default weight zero.

- [ ] **Step 2: Add a small linear ramp helper test**

Test that the quality weight is zero before `quality_start_step`, reaches the configured weight at `quality_ramp_end_step`, and stays constant after it.

- [ ] **Step 3: Implement the scheduled loss**

After the CLIP block, compute `quality_weight`, call `frequency_quality_loss(images.permute(0, 3, 1, 2), out["opacity"])`, log `train/loss_frequency_quality`, and add `quality_weight * loss_frequency_quality` to the objective. Reuse the existing `clip_decay_weight` schedule semantics.

- [ ] **Step 4: Run the complete focused test file and compile changed modules**

Run: `source miniconda; conda activate ruiheadstudio; PYTHONPATH=. pytest -q tests/test_clip_alignment.py && python -m py_compile threestudio/models/clip_alignment.py threestudio/systems/Head3DGSLKs.py`

Expected: existing and new tests pass.

- [ ] **Step 5: Commit the integration**

Run: `git add threestudio/models/clip_alignment.py threestudio/systems/Head3DGSLKs.py configs/headstudio_retry.yaml tests/test_clip_alignment.py && git commit -m "feat: schedule rendered frequency quality gate"`

### Task 4: Launch the three-cardinal ablation sweep

**Files:**
- Create: `scripts/launch_frequency_quality_sweep.sh`
- Create: `scripts/evaluate_frequency_quality_sweep.sh`

- [ ] **Step 1: Add reproducible launch parameters**

Use output root `outputs/text_gs_alignment_frequency_quality_sweep_20260830`, source checkpoint `outputs/text_gs_alignment_refine_alpha_20260828/refine_semantic/runs/refine_semantic/save/last.ply`, exact prompt, batch size 1, 2,000 continuation steps, `lambda_clip=0.001`, CLIP decay 11000 to 12000, quality ramp 11000 to 12000, and frequency weights `0.0005`, `0.0010`, `0.0020` on GPUs 0, 1, and 2.

- [ ] **Step 2: Add evaluation waiting and explicit endpoint**

Wait on each training PID, write a manifest, export `HEADSTUDIO_FINAL_STEP=12000`, and invoke `evaluation/run_evaluation.py --metrics all` in the `ruiheadstudio` environment. Keep each run's log and summary in its own directory.

- [ ] **Step 3: Syntax-check and launch**

Run: `bash -n scripts/launch_frequency_quality_sweep.sh scripts/evaluate_frequency_quality_sweep.sh` and then launch the sweep remotely. Record PIDs and start time in the experiment Markdown.

### Task 5: Evaluate, visualize, and document

**Files:**
- Modify: `docs/experiments/2026-08-27-text-gs-alignment-nightly.md`
- Create: `outputs/text_gs_alignment_frequency_quality_sweep_20260830/dashboard/README.md`
- Create: `outputs/text_gs_alignment_frequency_quality_sweep_20260830/dashboard/metrics_comparison.csv`
- Create: `outputs/text_gs_alignment_frequency_quality_sweep_20260830/dashboard/metrics_bars.svg`

- [ ] **Step 1: Parse all five metrics**

Read each `eval/all_metrics/summary.json`, compare to `0.2784`, `0.3130`, `0.3131`, `59.93`, and `51.36`, and compute per-metric deltas with PIQE treated as lower-is-better.

- [ ] **Step 2: Generate the consolidated dashboard**

Run `scripts/summarize_alignment_dashboard.py` with the three summaries and force-add the generated dashboard because output directories are ignored.

- [ ] **Step 3: Append exact experimental evidence**

Document checkpoint, command parameters, endpoint fix, all metrics, best run, and whether the full five-metric gate passed. Do not claim success for partial improvements.

- [ ] **Step 4: Verify and commit**

Run the focused tests, `git status --short`, inspect that only the four pre-existing unrelated untracked paths remain, then commit with `exp: record frequency quality sweep`.
