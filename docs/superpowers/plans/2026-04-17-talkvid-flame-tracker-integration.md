# TalkVid FLAME Tracker Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conversion path from `flame-head-tracker` monocular video tracking outputs to RuiHeadStudio training `.npy` collections and document the expected TalkVid directory layout.

**Architecture:** Keep `flame-head-tracker` as an external preprocessor with its own environment and define a stable on-disk contract for its outputs inside this repo. Add one standalone conversion script under `scripts/` that reads per-frame `.npz` tracking files from a clip directory, stacks the FLAME parameters into RuiHeadStudio sequence dictionaries, and writes an object-array `.npy` collection. Pair it with focused `unittest` coverage and a small integration guide under `docs/`.

**Tech Stack:** Python 3.9, NumPy, standard library `argparse`, `json`, `pathlib`, `unittest`

---

### Task 1: Add contract tests for tracker-output conversion

**Files:**
- Create: `tests/test_convert_talkvid_to_ruiheadstudio.py`
- Test: `tests/test_convert_talkvid_to_ruiheadstudio.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run `conda run -n ruiheadstudio python -m unittest tests.test_convert_talkvid_to_ruiheadstudio -v` and verify it fails because the conversion script does not exist**
- [ ] **Step 3: Implement the minimal conversion helpers in `scripts/convert_talkvid_to_ruiheadstudio.py`**
- [ ] **Step 4: Re-run the test module and verify it passes**

### Task 2: Implement the conversion CLI

**Files:**
- Create: `scripts/convert_talkvid_to_ruiheadstudio.py`

- [ ] **Step 1: Support tracker result directories as input, including parent directories searched recursively**
- [ ] **Step 2: Parse per-frame `.npz` files and map `exp`, `jaw_pose`, `neck_pose`, and `eye_pose` to RuiHeadStudio keys**
- [ ] **Step 3: Support optional `metadata.json` per clip directory for stable `video_name`, `clip_name`, and source-video paths**
- [ ] **Step 4: Export either one merged `.npy` file or append to an existing collection**

### Task 3: Document the directory structure and usage

**Files:**
- Create: `docs/talkvid_flame_tracker_integration.md`

- [ ] **Step 1: Document the recommended layout for raw TalkVid clips, tracker outputs, and converted RuiHeadStudio collections**
- [ ] **Step 2: Show the minimal commands to run flame-head-tracker and then the new conversion script**
- [ ] **Step 3: Verify the docs against a small synthetic conversion example**
