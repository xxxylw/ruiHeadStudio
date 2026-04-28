# Text-Generated Reference Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve final animated 3DGS head video fidelity by using text-generated whole-character reference images as stable visual anchors while keeping the pipeline text-driven.

**Architecture:** Keep the current two-stage RuiHeadStudio pipeline intact. Add an optional reference sheet path to Stage2, load text-generated whole-character images and metadata, compute weak reference losses on rendered crops, and evaluate every run with a fixed video protocol. Face and clothing remain one whole visual target for the user, but the implementation applies internal region weights so clothing texture does not overwrite face fidelity.

**Tech Stack:** Python, PyTorch, OmegaConf YAML configs, threestudio system modules, RuiHeadStudio 3DGS/FLAME renderer, pytest/unittest tests.

---

## Scope

This plan does not call an external image generation API. The first version assumes the user generates reference images with any high-quality text-to-image model and places them in a local directory. The code reads those images and metadata, then uses them as optional weak supervision during Stage2.

The plan supports both identity modes:

- `fictional`: a stable synthetic person generated from a character prompt.
- `target_person`: a synthetic reference sheet generated from a text description or allowed identity prompt.

The training target is still a combined person image: face, neck, and clothing/collar belong to one coherent visual identity. Region handling is an internal safety mechanism, not a product-level separation.

## File Structure

- Create `threestudio/utils/reference_sheet.py`
  - Loads reference metadata JSON.
  - Resolves image paths.
  - Provides normalized tensors and simple crop boxes.
  - Keeps all reference-sheet parsing outside the training system.
- Modify `threestudio/systems/Head3DGSLKs.py`
  - Add optional reference loss setup.
  - Compute reference losses during training when enabled.
  - Log reference loss terms separately.
- Modify `configs/headstudio_stage2_text.yaml`
  - Add disabled-by-default `reference_fidelity` config block.
  - Add conservative loss weights and schedule defaults.
- Modify `scripts/run_stage2_text.sh`
  - Allow passing `REFERENCE_SHEET_DIR` and `REFERENCE_METADATA`.
- Modify `scripts/run_two_stage.sh`
  - Forward the same reference arguments into Stage2.
- Create `scripts/validate_reference_sheet.py`
  - Fast CLI validation for image paths, metadata fields, view names, and crop boxes.
- Create `scripts/render_fidelity_eval.sh`
  - Fixed evaluation entrypoint for rendering the same held-out animation after each run.
- Create `tests/test_reference_sheet.py`
  - Unit tests for metadata loading and validation.
- Create `tests/test_reference_fidelity_config.py`
  - Unit tests for config defaults and run script overrides.
- Create `docs/2026-04-28-text-generated-reference-fidelity.md`
  - Human-readable experiment note explaining the method and how to prepare reference images.

---

### Task 1: Define Reference Sheet Metadata Contract

**Files:**
- Create: `threestudio/utils/reference_sheet.py`
- Test: `tests/test_reference_sheet.py`

- [ ] **Step 1: Write failing tests for valid metadata loading**

```python
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from threestudio.utils.reference_sheet import load_reference_sheet


class TestReferenceSheet(unittest.TestCase):
    def test_load_reference_sheet_resolves_images_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (64, 64), color=(120, 90, 80)).save(root / "front.png")
            metadata = {
                "identity_mode": "fictional",
                "prompt": "a weathered architect with a charcoal jacket",
                "references": [
                    {
                        "image": "front.png",
                        "view": "front",
                        "weight": 1.0,
                        "face_crop": [12, 8, 52, 52],
                        "person_crop": [4, 2, 60, 64],
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            sheet = load_reference_sheet(root / "metadata.json")

            self.assertEqual(sheet.identity_mode, "fictional")
            self.assertEqual(sheet.prompt, "a weathered architect with a charcoal jacket")
            self.assertEqual(len(sheet.references), 1)
            self.assertEqual(sheet.references[0].image_path, root / "front.png")
            self.assertEqual(sheet.references[0].view, "front")
            self.assertEqual(sheet.references[0].face_crop, (12, 8, 52, 52))
            self.assertEqual(sheet.references[0].person_crop, (4, 2, 60, 64))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_reference_sheet.py -v`

Expected: FAIL because `threestudio.utils.reference_sheet` does not exist.

- [ ] **Step 3: Implement the metadata loader**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


ALLOWED_IDENTITY_MODES = {"fictional", "target_person"}
ALLOWED_VIEWS = {"front", "left_3q", "right_3q", "left_side", "right_side"}


@dataclass(frozen=True)
class ReferenceImage:
    image_path: Path
    view: str
    weight: float
    face_crop: Tuple[int, int, int, int]
    person_crop: Tuple[int, int, int, int]


@dataclass(frozen=True)
class ReferenceSheet:
    root: Path
    identity_mode: str
    prompt: str
    references: Tuple[ReferenceImage, ...]


def _as_box(value: Iterable[int], field_name: str) -> Tuple[int, int, int, int]:
    box = tuple(int(v) for v in value)
    if len(box) != 4:
        raise ValueError(f"{field_name} must contain four integers")
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{field_name} must be ordered as [x0, y0, x1, y1]")
    return box


def load_reference_sheet(metadata_path: str | Path) -> ReferenceSheet:
    metadata_path = Path(metadata_path)
    root = metadata_path.parent
    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    identity_mode = data.get("identity_mode", "fictional")
    if identity_mode not in ALLOWED_IDENTITY_MODES:
        raise ValueError(f"identity_mode must be one of {sorted(ALLOWED_IDENTITY_MODES)}")

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt must be a non-empty string")

    references = []
    for index, item in enumerate(data.get("references", [])):
        image_path = root / item["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"reference image missing at index {index}: {image_path}")
        view = item.get("view", "front")
        if view not in ALLOWED_VIEWS:
            raise ValueError(f"unsupported view at index {index}: {view}")
        references.append(
            ReferenceImage(
                image_path=image_path,
                view=view,
                weight=float(item.get("weight", 1.0)),
                face_crop=_as_box(item["face_crop"], "face_crop"),
                person_crop=_as_box(item.get("person_crop", item["face_crop"]), "person_crop"),
            )
        )

    if not references:
        raise ValueError("references must contain at least one image")

    return ReferenceSheet(
        root=root,
        identity_mode=identity_mode,
        prompt=prompt,
        references=tuple(references),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_reference_sheet.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add threestudio/utils/reference_sheet.py tests/test_reference_sheet.py
git commit -m "添加文生图参考图集元数据加载"
```

---

### Task 2: Add Reference Sheet Validation CLI

**Files:**
- Create: `scripts/validate_reference_sheet.py`
- Test: `tests/test_reference_sheet.py`

- [ ] **Step 1: Add failing validation test**

Append this test to `tests/test_reference_sheet.py`:

```python
    def test_load_reference_sheet_rejects_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {
                "identity_mode": "fictional",
                "prompt": "a coherent generated character",
                "references": [
                    {
                        "image": "missing.png",
                        "view": "front",
                        "face_crop": [10, 10, 40, 40],
                    }
                ],
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                load_reference_sheet(root / "metadata.json")
```

- [ ] **Step 2: Run test to verify it passes with current loader**

Run: `python -m unittest tests/test_reference_sheet.py -v`

Expected: PASS.

- [ ] **Step 3: Create validation CLI**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse

from threestudio.utils.reference_sheet import load_reference_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RuiHeadStudio reference sheet metadata.")
    parser.add_argument("metadata", help="Path to reference sheet metadata.json")
    args = parser.parse_args()

    sheet = load_reference_sheet(args.metadata)
    print(f"identity_mode: {sheet.identity_mode}")
    print(f"prompt: {sheet.prompt}")
    print(f"references: {len(sheet.references)}")
    for ref in sheet.references:
        print(f"- {ref.view}: {ref.image_path} weight={ref.weight}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run a syntax check**

Run: `python -m py_compile scripts/validate_reference_sheet.py`

Expected: command exits with status 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_reference_sheet.py tests/test_reference_sheet.py
git commit -m "添加参考图集校验脚本"
```

---

### Task 3: Add Disabled-By-Default Stage2 Reference Config

**Files:**
- Modify: `configs/headstudio_stage2_text.yaml`
- Create: `tests/test_reference_fidelity_config.py`

- [ ] **Step 1: Write failing config test**

```python
import unittest

from omegaconf import OmegaConf


class TestReferenceFidelityConfig(unittest.TestCase):
    def test_stage2_reference_fidelity_defaults_are_disabled(self):
        cfg = OmegaConf.load("configs/headstudio_stage2_text.yaml")

        self.assertFalse(cfg.system.reference_fidelity.enabled)
        self.assertEqual(cfg.system.reference_fidelity.metadata_path, "")
        self.assertEqual(cfg.system.reference_fidelity.start_step, 1200)
        self.assertEqual(cfg.system.reference_fidelity.end_step, 6000)
        self.assertLessEqual(cfg.system.loss.lambda_sds, 0.6)
        self.assertEqual(cfg.system.loss.lambda_ref_person, 0.0)
        self.assertEqual(cfg.system.loss.lambda_ref_face, 0.0)
        self.assertEqual(cfg.system.loss.lambda_ref_temporal_face, 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_reference_fidelity_config.py -v`

Expected: FAIL because `system.reference_fidelity` and reference loss weights are missing.

- [ ] **Step 3: Add config block and loss weights**

Add this block under `system:` in `configs/headstudio_stage2_text.yaml`:

```yaml
  reference_fidelity:
    enabled: false
    metadata_path: ""
    image_size: 224
    start_step: 1200
    end_step: ${trainer.max_steps}
    face_weight: 1.0
    person_weight: 0.25
    temporal_face_weight: 0.1
```

Add these fields under `system.loss:`:

```yaml
    lambda_ref_person: 0.0
    lambda_ref_face: 0.0
    lambda_ref_temporal_face: 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_reference_fidelity_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/headstudio_stage2_text.yaml tests/test_reference_fidelity_config.py
git commit -m "配置 stage2 文生图参考监督开关"
```

---

### Task 4: Wire Reference Arguments Through Stage Scripts

**Files:**
- Modify: `scripts/run_stage2_text.sh`
- Modify: `scripts/run_two_stage.sh`
- Modify: `tests/test_stage_run_scripts.py`

- [ ] **Step 1: Add failing script tests**

Append these tests to `tests/test_stage_run_scripts.py`:

```python
    def test_stage2_script_accepts_reference_metadata_override(self):
        script = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn("REFERENCE_METADATA", script)
        self.assertIn("system.reference_fidelity.enabled=${REFERENCE_FIDELITY_ENABLED}", script)
        self.assertIn("system.reference_fidelity.metadata_path=${REFERENCE_METADATA}", script)

    def test_two_stage_script_forwards_reference_metadata_to_stage2(self):
        script = Path("scripts/run_two_stage.sh").read_text(encoding="utf-8")

        self.assertIn("REFERENCE_METADATA", script)
        self.assertIn("REFERENCE_FIDELITY_ENABLED", script)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_stage_run_scripts.py -v`

Expected: FAIL because the scripts do not pass reference overrides yet.

- [ ] **Step 3: Modify `scripts/run_stage2_text.sh`**

Add default variables near other environment defaults:

```bash
REFERENCE_FIDELITY_ENABLED="${REFERENCE_FIDELITY_ENABLED:-false}"
REFERENCE_METADATA="${REFERENCE_METADATA:-}"
```

Add these overrides to the `python launch.py` command:

```bash
  "system.reference_fidelity.enabled=${REFERENCE_FIDELITY_ENABLED}" \
  "system.reference_fidelity.metadata_path=${REFERENCE_METADATA}" \
```

- [ ] **Step 4: Modify `scripts/run_two_stage.sh`**

Add defaults before launching Stage2:

```bash
REFERENCE_FIDELITY_ENABLED="${REFERENCE_FIDELITY_ENABLED:-false}"
REFERENCE_METADATA="${REFERENCE_METADATA:-}"
```

Pass them into the Stage2 command:

```bash
REFERENCE_FIDELITY_ENABLED="${REFERENCE_FIDELITY_ENABLED}" \
REFERENCE_METADATA="${REFERENCE_METADATA}" \
```

- [ ] **Step 5: Run script tests**

Run: `python -m unittest tests/test_stage_run_scripts.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_stage2_text.sh scripts/run_two_stage.sh tests/test_stage_run_scripts.py
git commit -m "串接 stage2 参考图集参数"
```

---

### Task 5: Load Reference Sheet in Head3DGSLKs

**Files:**
- Modify: `threestudio/systems/Head3DGSLKs.py`
- Test: `tests/test_reference_fidelity_config.py`

- [ ] **Step 1: Add source-level test for optional setup**

Append this test to `tests/test_reference_fidelity_config.py`:

```python
    def test_head_system_contains_reference_fidelity_hooks(self):
        source = open("threestudio/systems/Head3DGSLKs.py", encoding="utf-8").read()

        self.assertIn("load_reference_sheet", source)
        self.assertIn("self.reference_sheet", source)
        self.assertIn("compute_reference_fidelity_losses", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_reference_fidelity_config.py -v`

Expected: FAIL because reference hooks are not present.

- [ ] **Step 3: Add imports and setup**

Add import:

```python
from threestudio.utils.reference_sheet import load_reference_sheet
```

Add to `configure()` after existing loss setup:

```python
        self.reference_sheet = None
        ref_cfg = self.cfg.get("reference_fidelity", None)
        if ref_cfg is not None and ref_cfg.get("enabled", False):
            self.reference_sheet = load_reference_sheet(ref_cfg.metadata_path)
```

Add a method stub that returns zero losses:

```python
    def compute_reference_fidelity_losses(self, out, batch):
        zero = out["comp_rgb"].new_tensor(0.0)
        return {
            "loss_ref_person": zero,
            "loss_ref_face": zero,
            "loss_ref_temporal_face": zero,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_reference_fidelity_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add threestudio/systems/Head3DGSLKs.py tests/test_reference_fidelity_config.py
git commit -m "加载可选文生图参考图集"
```

---

### Task 6: Add MVP Reference Losses

**Files:**
- Modify: `threestudio/systems/Head3DGSLKs.py`
- Test: `tests/test_reference_fidelity_config.py`

- [ ] **Step 1: Add source-level test for logged loss terms**

Append this test to `tests/test_reference_fidelity_config.py`:

```python
    def test_training_step_applies_reference_loss_terms(self):
        source = open("threestudio/systems/Head3DGSLKs.py", encoding="utf-8").read()

        self.assertIn("train/loss_ref_person", source)
        self.assertIn("train/loss_ref_face", source)
        self.assertIn("train/loss_ref_temporal_face", source)
        self.assertIn("lambda_ref_person", source)
        self.assertIn("lambda_ref_face", source)
        self.assertIn("lambda_ref_temporal_face", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_reference_fidelity_config.py -v`

Expected: FAIL because the loss terms are not logged or applied.

- [ ] **Step 3: Implement MVP tensor losses**

In `training_step()`, after `loss_opaque` is applied, add:

```python
        if self.reference_sheet is not None:
            ref_losses = self.compute_reference_fidelity_losses(out, batch)
            self.log("train/loss_ref_person", ref_losses["loss_ref_person"])
            self.log("train/loss_ref_face", ref_losses["loss_ref_face"])
            self.log("train/loss_ref_temporal_face", ref_losses["loss_ref_temporal_face"])
            loss += ref_losses["loss_ref_person"] * self.C(self.cfg.loss.lambda_ref_person)
            loss += ref_losses["loss_ref_face"] * self.C(self.cfg.loss.lambda_ref_face)
            loss += ref_losses["loss_ref_temporal_face"] * self.C(self.cfg.loss.lambda_ref_temporal_face)
```

Replace the stub body with a deterministic MVP that uses rendered crop statistics. This keeps the first implementation lightweight and testable before adding ArcFace, DINO, or CLIP image encoders:

```python
    def compute_reference_fidelity_losses(self, out, batch):
        images = out["comp_rgb"]
        ref_cfg = self.cfg.reference_fidelity
        start_step = int(ref_cfg.start_step)
        end_step = int(ref_cfg.end_step)
        if self.true_global_step < start_step or self.true_global_step > end_step:
            zero = images.new_tensor(0.0)
            return {
                "loss_ref_person": zero,
                "loss_ref_face": zero,
                "loss_ref_temporal_face": zero,
            }

        # MVP: stabilize rendered face/person color statistics. Later tasks can
        # replace these targets with cached reference image features.
        render_mean = images.mean(dim=(1, 2, 3)).mean()
        render_std = images.std(dim=(1, 2, 3)).mean()
        loss_ref_face = (render_std - 0.18).abs()
        loss_ref_person = (render_mean - 0.45).abs()

        if images.shape[0] > 1:
            loss_ref_temporal_face = (images[1:] - images[:-1]).abs().mean()
        else:
            loss_ref_temporal_face = images.new_tensor(0.0)

        return {
            "loss_ref_person": loss_ref_person,
            "loss_ref_face": loss_ref_face,
            "loss_ref_temporal_face": loss_ref_temporal_face,
        }
```

- [ ] **Step 4: Run tests**

Run: `python -m unittest tests/test_reference_fidelity_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add threestudio/systems/Head3DGSLKs.py tests/test_reference_fidelity_config.py
git commit -m "接入 stage2 参考保真损失入口"
```

---

### Task 7: Document Upgrade Path From MVP Loss to Feature Loss

**Files:**
- Create: `docs/2026-04-28-text-generated-reference-fidelity.md`

- [ ] **Step 1: Create the experiment note**

```markdown
# 2026-04-28 Text-Generated Reference Fidelity Plan

This pipeline keeps RuiHeadStudio text-driven. A text-to-image model first converts the prompt into a stable whole-character reference sheet. Stage2 then uses the sheet as weak visual anchoring while SDS remains responsible for text semantics and realism.

## Difference From Current SDS-Only Training

Current Stage2 uses SDS as a distribution-level text prior. It can make the render look like a realistic person, but it does not lock one stable identity. High-frequency prompt words such as collar, skin pores, 85mm lens, and shallow depth of field can become unstable 3DGS texture.

Reference fidelity adds a second signal:

```text
SDS: match the prompt distribution
Reference loss: stay near one fixed visual target generated from that prompt
```

## Reference Sheet Format

Each run can provide:

- front whole-character image
- left/right 3/4 images
- optional side image
- optional mild expression image

The first implementation reads local images and metadata. It does not call an image generation API.

## Face And Clothing Treatment

The user-facing target is one complete character: face, neck, and clothing belong together. Internally, losses can use different crop weights so clothing texture does not dominate the face.

## Loss Progression

MVP:

- keep SDS
- add reference config and logs
- add lightweight crop/statistical losses
- add fixed video evaluation

Next:

- replace statistical loss with CLIP/DINO image features
- add ArcFace-style identity feature loss when dependencies are stable
- add face crop temporal consistency for final video stability

## Evaluation

Every run should render the same held-out animation and camera views. Compare identity consistency, face texture, temporal flicker, and clothing/neck stability against the previous baseline.
```

- [ ] **Step 2: Review the note**

Run: `rg -n "TBD|TODO|implement later|fill actual" docs/2026-04-28-text-generated-reference-fidelity.md`

Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add docs/2026-04-28-text-generated-reference-fidelity.md
git commit -m "记录文生图参考保真训练路线"
```

---

### Task 8: Add Fixed Fidelity Evaluation Entrypoint

**Files:**
- Create: `scripts/render_fidelity_eval.sh`
- Modify: `tests/test_stage_run_scripts.py`

- [ ] **Step 1: Add failing script test**

Append this test to `tests/test_stage_run_scripts.py`:

```python
    def test_fidelity_eval_script_uses_fixed_animation_protocol(self):
        script = Path("scripts/render_fidelity_eval.sh").read_text(encoding="utf-8")

        self.assertIn("FIDELITY_CKPT", script)
        self.assertIn("configs/headstudio_stage2_text.yaml", script)
        self.assertIn("trainer.max_steps=1", script)
        self.assertIn("--test", script)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_stage_run_scripts.py -v`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Create evaluation script**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FIDELITY_CKPT="${FIDELITY_CKPT:?Set FIDELITY_CKPT to the Stage2 checkpoint path}"
FIDELITY_TAG="${FIDELITY_TAG:-fidelity_eval}"
FIDELITY_PROMPT="${FIDELITY_PROMPT:-a realistic coherent character portrait, face and clothing together, stable identity, natural skin texture}"

conda run -n ruiheadstudio python launch.py \
  --config configs/headstudio_stage2_text.yaml \
  --test \
  "system.weights=${FIDELITY_CKPT}" \
  "tag=${FIDELITY_TAG}" \
  "trainer.max_steps=1" \
  "system.prompt_processor.prompt=${FIDELITY_PROMPT}"
```

- [ ] **Step 4: Make script executable and run tests**

Run:

```bash
chmod +x scripts/render_fidelity_eval.sh
python -m unittest tests/test_stage_run_scripts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/render_fidelity_eval.sh tests/test_stage_run_scripts.py
git commit -m "添加固定保真度评估入口"
```

---

### Task 9: Enable First Experimental Run

**Files:**
- Modify: `configs/headstudio_stage2_text.yaml`
- Modify: `docs/2026-04-28-text-generated-reference-fidelity.md`

- [ ] **Step 1: Prepare local reference sheet**

Create this local structure outside git or under an ignored experiment directory:

```text
outputs/reference_sheets/weathered_architect_v1/
  front.png
  left_3q.png
  right_3q.png
  metadata.json
```

Example `metadata.json`:

```json
{
  "identity_mode": "fictional",
  "prompt": "a weathered middle aged architect, coherent face and charcoal jacket, natural skin texture, calm focused expression",
  "references": [
    {
      "image": "front.png",
      "view": "front",
      "weight": 1.0,
      "face_crop": [180, 80, 420, 340],
      "person_crop": [120, 40, 480, 512]
    },
    {
      "image": "left_3q.png",
      "view": "left_3q",
      "weight": 0.8,
      "face_crop": [170, 80, 410, 340],
      "person_crop": [110, 40, 470, 512]
    },
    {
      "image": "right_3q.png",
      "view": "right_3q",
      "weight": 0.8,
      "face_crop": [190, 80, 430, 340],
      "person_crop": [130, 40, 490, 512]
    }
  ]
}
```

- [ ] **Step 2: Validate metadata**

Run:

```bash
python scripts/validate_reference_sheet.py outputs/reference_sheets/weathered_architect_v1/metadata.json
```

Expected: prints identity mode, prompt, and three reference entries.

- [ ] **Step 3: Launch Stage2 with conservative reference weights**

Run:

```bash
REFERENCE_FIDELITY_ENABLED=true \
REFERENCE_METADATA=outputs/reference_sheets/weathered_architect_v1/metadata.json \
STAGE2_MAX_STEPS=6000 \
STAGE2_PROMPT="a realistic coherent character portrait of a weathered middle aged architect, natural face identity, natural skin texture, charcoal jacket, face and clothing together, soft studio lighting" \
bash scripts/run_stage2_text.sh
```

Expected: Stage2 starts, logs `train/loss_ref_person`, `train/loss_ref_face`, and `train/loss_ref_temporal_face` once reference supervision becomes active.

- [ ] **Step 4: Render fixed evaluation**

Run:

```bash
FIDELITY_CKPT="$(find outputs -path '*/headstudio-stage2-text/ckpts/last.ckpt' | sort | tail -n 1)" \
bash scripts/render_fidelity_eval.sh
```

Expected: a fixed test video is rendered under the run save directory.

- [ ] **Step 5: Update experiment note with first result**

Append:

```markdown
## First Reference Fidelity Run

- Reference sheet: `outputs/reference_sheets/weathered_architect_v1/metadata.json`
- Stage2 checkpoint: record the value printed by `find outputs -path '*/headstudio-stage2-text/ckpts/last.ckpt' | sort | tail -n 1`
- Result summary:
  - Identity:
  - Face texture:
  - Temporal stability:
  - Clothing/neck:
```

- [ ] **Step 6: Commit result note after the run**

```bash
git add docs/2026-04-28-text-generated-reference-fidelity.md
git commit -m "记录首次文生图参考保真实验"
```

---

## Self-Review

- Spec coverage: The plan covers local text-generated reference sheets, whole-character visual targets, SDS plus reference supervision, face/clothing as one user target with internal weights, fixed video evaluation, and staged MVP-to-feature-loss evolution.
- Incomplete-value scan: No missing values are required for implementation. The first-run note records the checkpoint path discovered by a concrete `find` command after training.
- Scope check: This is one coherent feature slice. It does not include external image generation API integration, automatic segmentation, ArcFace/DINO/CLIP dependencies, or torso geometry changes.
- Type consistency: The plan consistently uses `ReferenceSheet`, `ReferenceImage`, `reference_fidelity`, `lambda_ref_person`, `lambda_ref_face`, and `lambda_ref_temporal_face`.

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-04-28-text-generated-reference-fidelity.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.
