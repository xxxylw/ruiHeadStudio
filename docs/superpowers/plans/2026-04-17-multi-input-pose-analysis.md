# Multi-Input Pose Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the pose distribution analyzer so it can compare multiple RuiHeadStudio pose collections in one shared UMAP/HDBSCAN space and optionally sample frames with the same sequence-then-frame logic used during training.

**Architecture:** Keep `scripts/analyze_pose_distribution.py` as the single entry point, but refactor its data loading and sampling helpers so it can expand multiple file or directory inputs into one combined feature table with per-input and per-group metadata. Add one focused `unittest` module to lock the new multi-input expansion and `train_like` frame sampling behavior before changing the script.

**Tech Stack:** Python 3.9, NumPy, Matplotlib, UMAP, HDBSCAN, standard library `argparse` and `unittest`

---

### Task 1: Add failing tests for multi-input loading and train-like sampling

**Files:**
- Create: `tests/test_analyze_pose_distribution.py`
- Test: `tests/test_analyze_pose_distribution.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run `conda run -n ruiheadstudio python -m unittest tests.test_analyze_pose_distribution -v` and verify it fails because the new helper functions do not exist**
- [ ] **Step 3: Implement the minimal helpers in `scripts/analyze_pose_distribution.py`**
- [ ] **Step 4: Re-run the test module and verify it passes**

### Task 2: Wire the CLI to combined analysis outputs

**Files:**
- Modify: `scripts/analyze_pose_distribution.py`

- [ ] **Step 1: Extend the CLI to accept multiple file/directory inputs and optional top-level group labels**
- [ ] **Step 2: Add `--sampler-mode=all_frames|train_like` and `--samples-per-input` for frame-level sampling**
- [ ] **Step 3: Emit combined metadata, group-colored UMAP output, and richer summary counts without breaking the existing source-colored and cluster-colored outputs**
- [ ] **Step 4: Run the updated analyzer on `project_converted_exp.npy` and `talkshow/collection/synthetic_aug/` to verify the new workflow end-to-end**
