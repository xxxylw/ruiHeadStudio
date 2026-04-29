# Opacity and Reference Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a staged fix path for rear-head transparency: first add opacity diagnostics, then region-aware opacity repair, then prepare reference feature-fidelity upgrades.

**Architecture:** Add small, testable utilities for Gaussian region classification and stats, keep GPU rendering/eval in scripts, and keep training behavior unchanged unless new config flags are enabled. Opacity repair lives in the existing Gaussian/Head system boundary; reference feature upgrades extend the existing `reference_fidelity` config without replacing the current statistics loss.

**Tech Stack:** Python 3.9, PyTorch, NumPy, plyfile, unittest, existing threestudio/3DGS training scripts.

---

## File Structure

- Create `threestudio/utils/opacity_diagnostics.py`
  - Pure utility functions for region classification, tensor summaries, and JSON-serializable Gaussian stats.
  - No dependency on Lightning or launch code.
- Create `scripts/diagnose_opacity_thickness.py`
  - CLI entry for Phase A. Loads `save/last.ply`, computes Gaussian opacity/scaling/region stats, writes `gaussian_region_stats.json` and `summary.md`.
  - Optional render-view extension can be added after stats are stable.
- Modify `gaussiansplatting/scene/gaussian_flame_model.py`
  - Add region-aware helpers to classify Gaussian bindings by neutral FLAME face normals.
  - Add optional region-aware prune thresholds.
- Modify `threestudio/systems/Head3DGSLKs.py`
  - Add disabled-by-default config dictionaries: `opacity_coverage`, `rear_opacity`, `prune_region_guard`.
  - Add loss hooks for opacity coverage and rear opacity.
  - Route prune calls through region guard only when enabled.
- Modify `configs/headstudio_stage1_prior.yaml` and `configs/headstudio_stage2_text.yaml`
  - Add disabled defaults for new opacity repair config.
- Modify `threestudio/utils/reference_sheet.py` and `threestudio/systems/Head3DGSLKs.py`
  - Extend reference metadata handling to support optional `neck_crop` and `global_crop`.
  - Add config skeleton for feature/identity losses, default disabled.
- Create tests:
  - `tests/test_opacity_diagnostics.py`
  - `tests/test_region_prune_guard.py`
  - Extend `tests/test_opacity_alpha_pipeline.py`
  - Extend `tests/test_reference_fidelity_config.py`
  - Extend `tests/test_stage_run_scripts.py`

---

### Task 1: Add CPU-Safe Opacity Diagnostics Utilities

**Files:**
- Create: `threestudio/utils/opacity_diagnostics.py`
- Test: `tests/test_opacity_diagnostics.py`

- [ ] **Step 1: Write failing tests for quantiles, region labels, and summaries**

Create `tests/test_opacity_diagnostics.py`:

```python
import unittest

import torch

from threestudio.utils.opacity_diagnostics import (
    classify_by_face_normals,
    summarize_tensor,
    summarize_gaussian_regions,
)


class TestOpacityDiagnostics(unittest.TestCase):
    def test_summarize_tensor_returns_json_safe_stats(self):
        stats = summarize_tensor(torch.tensor([0.1, 0.2, 0.9], dtype=torch.float32))

        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 0.4, places=6)
        self.assertIn("p10", stats)
        self.assertIn("p50", stats)
        self.assertIn("p90", stats)

    def test_summarize_tensor_handles_empty_tensor(self):
        stats = summarize_tensor(torch.empty(0))

        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["mean"])
        self.assertIsNone(stats["p50"])

    def test_classify_by_face_normals_uses_x_axis_regions(self):
        face_normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        labels = classify_by_face_normals(face_normals, front_threshold=0.35, rear_threshold=-0.35)

        self.assertEqual(labels, ["front", "rear", "side", "side"])

    def test_summarize_gaussian_regions_groups_opacity_and_scaling(self):
        opacity = torch.tensor([[0.9], [0.2], [0.5]], dtype=torch.float32)
        scaling = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=torch.float32,
        )
        labels = ["front", "rear", "rear"]

        summary = summarize_gaussian_regions(opacity, scaling, labels)

        self.assertEqual(summary["total"]["count"], 3)
        self.assertEqual(summary["regions"]["front"]["count"], 1)
        self.assertEqual(summary["regions"]["rear"]["count"], 2)
        self.assertEqual(summary["regions"]["side"]["count"], 0)
        self.assertAlmostEqual(summary["regions"]["rear"]["opacity"]["mean"], 0.35, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests/test_opacity_diagnostics.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'threestudio.utils.opacity_diagnostics'
```

- [ ] **Step 3: Implement diagnostics utility**

Create `threestudio/utils/opacity_diagnostics.py`:

```python
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch


def _to_1d_float_tensor(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    return values.detach().float().reshape(-1).cpu()


def summarize_tensor(values: torch.Tensor) -> Dict[str, Optional[float]]:
    tensor = _to_1d_float_tensor(values)
    if tensor.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }

    quantiles = torch.quantile(tensor, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def classify_by_face_normals(
    face_normals: torch.Tensor,
    front_threshold: float = 0.35,
    rear_threshold: float = -0.35,
) -> List[str]:
    normals = face_normals.detach().float().cpu()
    if normals.ndim != 2 or normals.shape[-1] != 3:
        raise ValueError("face_normals must have shape [N, 3]")

    labels: List[str] = []
    for normal in normals:
        x = float(normal[0].item())
        if x >= front_threshold:
            labels.append("front")
        elif x <= rear_threshold:
            labels.append("rear")
        else:
            labels.append("side")
    return labels


def _mask_for_labels(labels: Iterable[str], target: str) -> torch.Tensor:
    return torch.tensor([label == target for label in labels], dtype=torch.bool)


def summarize_gaussian_regions(
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    labels: List[str],
) -> Dict[str, object]:
    opacity_flat = _to_1d_float_tensor(opacity)
    scaling_tensor = scaling.detach().float().cpu()
    if scaling_tensor.ndim == 2:
        scaling_max = scaling_tensor.max(dim=1).values
    else:
        scaling_max = scaling_tensor.reshape(-1)

    if opacity_flat.numel() != len(labels) or scaling_max.numel() != len(labels):
        raise ValueError("opacity, scaling, and labels must describe the same number of gaussians")

    regions: Dict[str, object] = {}
    for region in ("front", "side", "rear"):
        mask = _mask_for_labels(labels, region)
        regions[region] = {
            "count": int(mask.sum().item()),
            "opacity": summarize_tensor(opacity_flat[mask]),
            "scaling_max": summarize_tensor(scaling_max[mask]),
        }

    return {
        "total": {
            "count": int(opacity_flat.numel()),
            "opacity": summarize_tensor(opacity_flat),
            "scaling_max": summarize_tensor(scaling_max),
        },
        "regions": regions,
    }
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
python -m unittest tests/test_opacity_diagnostics.py -v
```

Expected:

```text
Ran 4 tests
OK
```

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add tests/test_opacity_diagnostics.py threestudio/utils/opacity_diagnostics.py
git commit -m "添加 opacity 诊断统计工具"
```

---

### Task 2: Add Gaussian Region Classification on GaussianFlameModel

**Files:**
- Modify: `gaussiansplatting/scene/gaussian_flame_model.py`
- Test: `tests/test_region_prune_guard.py`

- [ ] **Step 1: Write failing source tests for region helpers**

Create `tests/test_region_prune_guard.py`:

```python
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "gaussiansplatting" / "scene" / "gaussian_flame_model.py"


class TestRegionPruneGuard(unittest.TestCase):
    def test_gaussian_flame_model_exposes_region_helpers(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn("def get_bound_face_normals", source)
        self.assertIn("def get_gaussian_region_labels", source)
        self.assertIn("classify_by_face_normals", source)

    def test_prune_methods_accept_region_thresholds(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn("region_min_opacity=None", source)
        self.assertIn("def _region_opacity_prune_mask", source)
        self.assertIn('region_min_opacity.get("rear"', source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests/test_region_prune_guard.py -v
```

Expected:

```text
FAIL: test_gaussian_flame_model_exposes_region_helpers
```

- [ ] **Step 3: Implement region helpers and optional threshold mask**

Modify `gaussiansplatting/scene/gaussian_flame_model.py`.

Add import near other imports:

```python
from threestudio.utils.opacity_diagnostics import classify_by_face_normals
```

Add methods inside `GaussianFlameModel` after `get_anchor_world_xyz`:

```python
    def get_bound_face_normals(self):
        T, R, S = self.get_face_transform_components()
        return F.normalize(R[:, :, 2], dim=-1)

    def get_gaussian_region_labels(self, front_threshold=0.35, rear_threshold=-0.35):
        if self._faces.numel() == 0:
            return []
        normals = self.get_bound_face_normals()
        return classify_by_face_normals(normals, front_threshold, rear_threshold)

    def _region_opacity_prune_mask(self, region_min_opacity):
        labels = self.get_gaussian_region_labels()
        if not labels:
            return torch.zeros((self.get_opacity.shape[0],), dtype=torch.bool, device=self.get_opacity.device)

        opacity = self.get_opacity.squeeze()
        prune_mask = torch.zeros_like(opacity, dtype=torch.bool)
        for region in ("front", "side", "rear"):
            threshold = region_min_opacity.get(region)
            if threshold is None:
                continue
            region_mask = torch.tensor([label == region for label in labels], dtype=torch.bool, device=opacity.device)
            prune_mask = torch.logical_or(prune_mask, torch.logical_and(region_mask, opacity < float(threshold)))
        return prune_mask
```

Change signatures:

```python
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, region_min_opacity=None):
```

and:

```python
    def prune_only(self, min_opacity=0.005, extent=0.01, region_min_opacity=None):
```

In `densify_and_prune`, replace:

```python
        prune_mask = (self.get_opacity < min_opacity).squeeze()
```

with:

```python
        if region_min_opacity:
            prune_mask = self._region_opacity_prune_mask(region_min_opacity)
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
```

In `prune_only`, replace:

```python
        unseen_points = (self.get_opacity < min_opacity).squeeze()
```

with:

```python
        if region_min_opacity:
            unseen_points = self._region_opacity_prune_mask(region_min_opacity)
        else:
            unseen_points = (self.get_opacity < min_opacity).squeeze()
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests/test_opacity_diagnostics.py tests/test_region_prune_guard.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add gaussiansplatting/scene/gaussian_flame_model.py tests/test_region_prune_guard.py
git commit -m "添加高斯区域分类与分区 prune 接口"
```

---

### Task 3: Add Phase A Opacity Diagnostics CLI

**Files:**
- Create: `scripts/diagnose_opacity_thickness.py`
- Test: `tests/test_opacity_diagnostics.py`

- [ ] **Step 1: Extend tests for CLI source and summary writer**

Append to `tests/test_opacity_diagnostics.py`:

```python
from pathlib import Path
import tempfile

from scripts.diagnose_opacity_thickness import write_summary_markdown


class TestOpacityDiagnosticsCli(unittest.TestCase):
    def test_diagnostics_script_has_expected_cli_arguments(self):
        source = Path("scripts/diagnose_opacity_thickness.py").read_text(encoding="utf-8")

        self.assertIn("--ply", source)
        self.assertIn("--output", source)
        self.assertIn("gaussian_region_stats.json", source)
        self.assertIn("summary.md", source)

    def test_write_summary_markdown_writes_region_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.md"
            stats = {
                "total": {"count": 3, "opacity": {"mean": 0.5}, "scaling_max": {"mean": 0.2}},
                "regions": {
                    "front": {"count": 1, "opacity": {"mean": 0.9}, "scaling_max": {"mean": 0.1}},
                    "side": {"count": 0, "opacity": {"mean": None}, "scaling_max": {"mean": None}},
                    "rear": {"count": 2, "opacity": {"mean": 0.3}, "scaling_max": {"mean": 0.4}},
                },
            }

            write_summary_markdown(output, stats)

            text = output.read_text(encoding="utf-8")
            self.assertIn("# Opacity Thickness Diagnostics", text)
            self.assertIn("| rear | 2 | 0.300000 | 0.400000 |", text)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests/test_opacity_diagnostics.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'scripts.diagnose_opacity_thickness'
```

- [ ] **Step 3: Implement CLI**

Create `scripts/diagnose_opacity_thickness.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from threestudio.utils.opacity_diagnostics import summarize_gaussian_regions


def _read_vertex_array(ply_path: Path, prefix: str) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    vertex = ply.elements[0]
    names = sorted(
        [prop.name for prop in vertex.properties if prop.name.startswith(prefix)],
        key=lambda name: int(name.split("_")[-1]),
    )
    if not names:
        raise ValueError(f"No properties with prefix {prefix!r} found in {ply_path}")
    return np.stack([np.asarray(vertex[name]) for name in names], axis=1)


def load_ply_stats_inputs(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    vertex = ply.elements[0]
    opacity = torch.tensor(np.asarray(vertex["opacity"])[..., None], dtype=torch.float32)
    scaling = torch.tensor(_read_vertex_array(ply_path, "scale_"), dtype=torch.float32)
    face = _read_vertex_array(ply_path, "face_").astype(np.int64)
    labels = ["unknown"] * face.shape[0]
    return opacity, scaling, labels


def _format_float(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_summary_markdown(path: Path, stats):
    lines = [
        "# Opacity Thickness Diagnostics",
        "",
        f"- Total gaussians: {stats['total']['count']}",
        f"- Total opacity mean: {_format_float(stats['total']['opacity']['mean'])}",
        f"- Total scaling max mean: {_format_float(stats['total']['scaling_max']['mean'])}",
        "",
        "| region | count | opacity mean | scaling max mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for region in ("front", "side", "rear"):
        region_stats = stats["regions"][region]
        lines.append(
            f"| {region} | {region_stats['count']} | "
            f"{_format_float(region_stats['opacity']['mean'])} | "
            f"{_format_float(region_stats['scaling_max']['mean'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize Gaussian opacity and thickness diagnostics from a PLY.")
    parser.add_argument("--ply", required=True, type=Path, help="Path to save/last.ply")
    parser.add_argument("--output", required=True, type=Path, help="Output diagnostics directory")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    opacity, scaling, labels = load_ply_stats_inputs(args.ply)
    stats = summarize_gaussian_regions(opacity, scaling, labels)

    stats_path = args.output / "gaussian_region_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_summary_markdown(args.output / "summary.md", stats)
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
```

This first CLI reads PLY stats without GPU. Region labels are `unknown` until Task 4 wires model-based face normal classification.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests/test_opacity_diagnostics.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add scripts/diagnose_opacity_thickness.py tests/test_opacity_diagnostics.py
git commit -m "添加 opacity 厚实度诊断脚本"
```

---

### Task 4: Add Config Defaults and Disabled Training Hooks for Opacity Repair

**Files:**
- Modify: `threestudio/systems/Head3DGSLKs.py`
- Modify: `configs/headstudio_stage1_prior.yaml`
- Modify: `configs/headstudio_stage2_text.yaml`
- Test: `tests/test_opacity_alpha_pipeline.py`

- [ ] **Step 1: Add failing source tests for config and hooks**

Append to `tests/test_opacity_alpha_pipeline.py`:

```python
    def test_stage_configs_define_disabled_opacity_repair_defaults(self):
        stage1 = (REPO_ROOT / "configs" / "headstudio_stage1_prior.yaml").read_text(encoding="utf-8")
        stage2 = (REPO_ROOT / "configs" / "headstudio_stage2_text.yaml").read_text(encoding="utf-8")

        for cfg in (stage1, stage2):
            self.assertIn("opacity_coverage:", cfg)
            self.assertIn("rear_opacity:", cfg)
            self.assertIn("prune_region_guard:", cfg)
            self.assertIn("lambda_opacity_coverage: 0.0", cfg)
            self.assertIn("lambda_rear_opacity: 0.0", cfg)

    def test_head_system_contains_disabled_opacity_repair_hooks(self):
        source = (REPO_ROOT / "threestudio" / "systems" / "Head3DGSLKs.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("opacity_coverage: dict", source)
        self.assertIn("rear_opacity: dict", source)
        self.assertIn("prune_region_guard: dict", source)
        self.assertIn("compute_opacity_coverage_loss", source)
        self.assertIn("compute_rear_opacity_loss", source)
        self.assertIn("train/loss_opacity_coverage", source)
        self.assertIn("train/loss_rear_opacity", source)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests/test_opacity_alpha_pipeline.py -v
```

Expected:

```text
FAIL: test_stage_configs_define_disabled_opacity_repair_defaults
```

- [ ] **Step 3: Add config dataclass fields and loss helpers**

Modify `Head3DGSLKsRig.Config` in `threestudio/systems/Head3DGSLKs.py`:

```python
        opacity_coverage: dict = field(default_factory=dict)
        rear_opacity: dict = field(default_factory=dict)
        prune_region_guard: dict = field(default_factory=dict)
```

Add helper methods before `training_step`:

```python
    def compute_opacity_coverage_loss(self, out):
        cfg = self.cfg.get("opacity_coverage", {})
        if not cfg.get("enabled", False):
            return out["opacity"].new_tensor(0.0)
        opacity = out["opacity"]
        min_alpha = float(cfg.get("min_alpha", 0.85))
        return F.relu(min_alpha - opacity).mean()

    def compute_rear_opacity_loss(self):
        cfg = self.cfg.get("rear_opacity", {})
        if not cfg.get("enabled", False):
            return self.gaussian.get_opacity.new_tensor(0.0)
        labels = self.gaussian.get_gaussian_region_labels()
        if not labels:
            return self.gaussian.get_opacity.new_tensor(0.0)
        rear_mask = torch.tensor([label == "rear" for label in labels], dtype=torch.bool, device=self.gaussian.get_opacity.device)
        if not rear_mask.any():
            return self.gaussian.get_opacity.new_tensor(0.0)
        min_mean_opacity = float(cfg.get("min_mean_opacity", 0.35))
        rear_mean = self.gaussian.get_opacity.squeeze()[rear_mask].mean()
        return F.relu(rear_mean.new_tensor(min_mean_opacity) - rear_mean)
```

In `training_step`, after `loss_opaque`:

```python
        lambda_opacity_coverage = self.cfg.loss.get("lambda_opacity_coverage", 0.0)
        if lambda_opacity_coverage > 0.0:
            loss_opacity_coverage = self.compute_opacity_coverage_loss(out)
            self.log("train/loss_opacity_coverage", loss_opacity_coverage)
            loss += loss_opacity_coverage * self.C(lambda_opacity_coverage)

        lambda_rear_opacity = self.cfg.loss.get("lambda_rear_opacity", 0.0)
        if lambda_rear_opacity > 0.0:
            loss_rear_opacity = self.compute_rear_opacity_loss()
            self.log("train/loss_rear_opacity", loss_rear_opacity)
            loss += loss_rear_opacity * self.C(lambda_rear_opacity)
```

- [ ] **Step 4: Add disabled defaults to both stage configs**

In both stage YAML files, under `system:`, add:

```yaml
  opacity_coverage:
    enabled: false
    min_alpha: 0.85
    mode: head_region
  rear_opacity:
    enabled: false
    min_mean_opacity: 0.35
  prune_region_guard:
    enabled: false
    rear_min_opacity_scale: 0.5
```

Under `system.loss:`, add:

```yaml
    lambda_opacity_coverage: 0.0
    lambda_rear_opacity: 0.0
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m unittest tests/test_opacity_alpha_pipeline.py tests/test_stage_configs.py -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add threestudio/systems/Head3DGSLKs.py configs/headstudio_stage1_prior.yaml configs/headstudio_stage2_text.yaml tests/test_opacity_alpha_pipeline.py
git commit -m "加入默认关闭的 opacity 修复配置"
```

---

### Task 5: Wire Region-Aware Prune Guard Through Training

**Files:**
- Modify: `threestudio/systems/Head3DGSLKs.py`
- Test: `tests/test_opacity_alpha_pipeline.py`

- [ ] **Step 1: Add failing source test for prune guard wiring**

Append to `tests/test_opacity_alpha_pipeline.py`:

```python
    def test_head_system_passes_region_min_opacity_to_prune(self):
        source = (REPO_ROOT / "threestudio" / "systems" / "Head3DGSLKs.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("build_region_min_opacity", source)
        self.assertIn("region_min_opacity=region_min_opacity", source)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m unittest tests/test_opacity_alpha_pipeline.py -v
```

Expected:

```text
FAIL: test_head_system_passes_region_min_opacity_to_prune
```

- [ ] **Step 3: Implement prune threshold builder**

Add method to `Head3DGSLKsRig` before `on_before_optimizer_step`:

```python
    def build_region_min_opacity(self, base_min_opacity):
        cfg = self.cfg.get("prune_region_guard", {})
        if not cfg.get("enabled", False):
            return None
        rear_scale = float(cfg.get("rear_min_opacity_scale", 0.5))
        return {
            "front": float(base_min_opacity),
            "side": float(base_min_opacity),
            "rear": float(base_min_opacity) * rear_scale,
        }
```

In both prune calls inside `on_before_optimizer_step`, pass region thresholds.

For `densify_and_prune`:

```python
                    region_min_opacity = self.build_region_min_opacity(self.cfg.densify_min_opacity)
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.densify_min_opacity,
                        self.cameras_extent,
                        size_threshold,
                        region_min_opacity=region_min_opacity,
                    )
```

For `prune_only`:

```python
                    region_min_opacity = self.build_region_min_opacity(self.cfg.prune_only_min_opacity)
                    self.gaussian.prune_only(
                        min_opacity=self.cfg.prune_only_min_opacity,
                        extent=self.cameras_extent,
                        region_min_opacity=region_min_opacity,
                    )
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests/test_opacity_alpha_pipeline.py tests/test_region_prune_guard.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add threestudio/systems/Head3DGSLKs.py tests/test_opacity_alpha_pipeline.py
git commit -m "接入后脑勺分区 prune guard"
```

---

### Task 6: Extend Reference Metadata and Config Skeleton for Feature Losses

**Files:**
- Modify: `threestudio/utils/reference_sheet.py`
- Modify: `threestudio/systems/Head3DGSLKs.py`
- Modify: `configs/headstudio_stage2_text.yaml`
- Test: `tests/test_reference_sheet.py`
- Test: `tests/test_reference_fidelity_config.py`

- [ ] **Step 1: Add failing tests for optional neck/global crops and feature config**

Append to `tests/test_reference_fidelity_config.py`:

```python
    def test_stage2_reference_feature_loss_defaults_are_disabled(self):
        cfg = Path("configs/headstudio_stage2_text.yaml").read_text(encoding="utf-8")

        self.assertIn("feature_loss:", cfg)
        self.assertIn("backbone: dino", cfg)
        self.assertIn("lambda_ref_face_feature: 0.0", cfg)
        self.assertIn("lambda_ref_person_feature: 0.0", cfg)
        self.assertIn("identity_loss:", cfg)
        self.assertIn("lambda_ref_identity: 0.0", cfg)

    def test_head_system_contains_reference_feature_config_hooks(self):
        source = Path("threestudio/systems/Head3DGSLKs.py").read_text(encoding="utf-8")

        self.assertIn("loss_ref_face_feature", source)
        self.assertIn("loss_ref_person_feature", source)
        self.assertIn("loss_ref_identity", source)
```

Append to `tests/test_reference_sheet.py`:

```python
    def test_reference_sheet_supports_optional_neck_and_global_crops(self):
        metadata = {
            "identity_mode": "target_person",
            "references": [
                {
                    "image": "reference_sheet.png",
                    "view": "front",
                    "weight": 1.0,
                    "face_crop": [1, 2, 3, 4],
                    "person_crop": [5, 6, 7, 8],
                    "neck_crop": [9, 10, 11, 12],
                    "global_crop": [13, 14, 15, 16],
                }
            ],
        }

        sheet = load_reference_sheet_from_metadata(metadata, Path("/tmp/reference"))

        self.assertEqual(sheet.references[0].neck_crop, [9, 10, 11, 12])
        self.assertEqual(sheet.references[0].global_crop, [13, 14, 15, 16])
```

If `load_reference_sheet_from_metadata` does not exist yet, first expose it from current parsing code rather than duplicating parser logic.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests/test_reference_sheet.py tests/test_reference_fidelity_config.py -v
```

Expected:

```text
FAIL
```

- [ ] **Step 3: Extend reference sheet dataclass and parser**

In `threestudio/utils/reference_sheet.py`, add optional fields to the reference item dataclass:

```python
    neck_crop: list[int] | None = None
    global_crop: list[int] | None = None
```

When parsing each metadata reference:

```python
        neck_crop=data.get("neck_crop")
        global_crop=data.get("global_crop")
```

If Python 3.9 rejects `list[int] | None`, use:

```python
from typing import List, Optional

neck_crop: Optional[List[int]] = None
global_crop: Optional[List[int]] = None
```

- [ ] **Step 4: Add disabled feature-loss config skeleton**

In `configs/headstudio_stage2_text.yaml`, under `system.reference_fidelity:`, add:

```yaml
    feature_loss:
      enabled: false
      backbone: dino
    identity_loss:
      enabled: false
      backbone: arcface
```

Under `system.loss:`, add:

```yaml
    lambda_ref_face_feature: 0.0
    lambda_ref_person_feature: 0.0
    lambda_ref_identity: 0.0
```

- [ ] **Step 5: Add zero-valued placeholder hooks in reference losses**

In `compute_reference_fidelity_losses`, include zero values in the early return:

```python
                "loss_ref_face_feature": zero,
                "loss_ref_person_feature": zero,
                "loss_ref_identity": zero,
```

and in the normal return:

```python
            "loss_ref_face_feature": images.new_tensor(0.0),
            "loss_ref_person_feature": images.new_tensor(0.0),
            "loss_ref_identity": images.new_tensor(0.0),
```

In `training_step`, when `self.reference_sheet is not None`, log them:

```python
            self.log("train/loss_ref_face_feature", ref_losses["loss_ref_face_feature"])
            self.log("train/loss_ref_person_feature", ref_losses["loss_ref_person_feature"])
            self.log("train/loss_ref_identity", ref_losses["loss_ref_identity"])
            loss += ref_losses["loss_ref_face_feature"] * self.C(self.cfg.loss.lambda_ref_face_feature)
            loss += ref_losses["loss_ref_person_feature"] * self.C(self.cfg.loss.lambda_ref_person_feature)
            loss += ref_losses["loss_ref_identity"] * self.C(self.cfg.loss.lambda_ref_identity)
```

This task only creates the disabled config and logging surface. Actual DINO/CLIP/ArcFace backbones are separate follow-up tasks after opacity repair is validated.

- [ ] **Step 6: Run tests**

Run:

```bash
python -m unittest tests/test_reference_sheet.py tests/test_reference_fidelity_config.py -v
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add threestudio/utils/reference_sheet.py threestudio/systems/Head3DGSLKs.py configs/headstudio_stage2_text.yaml tests/test_reference_sheet.py tests/test_reference_fidelity_config.py
git commit -m "扩展 reference feature loss 配置表面"
```

---

### Task 7: Add Run Script Knobs for Opacity Repair Experiments

**Files:**
- Modify: `scripts/run_stage2_text.sh`
- Modify: `scripts/run_two_stage.sh`
- Test: `tests/test_stage_run_scripts.py`

- [ ] **Step 1: Add failing tests for env knobs**

Append to `tests/test_stage_run_scripts.py`:

```python
    def test_stage2_script_accepts_opacity_repair_overrides(self):
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")

        self.assertIn("OPACITY_COVERAGE_ENABLED", stage2)
        self.assertIn("REAR_OPACITY_ENABLED", stage2)
        self.assertIn("PRUNE_REGION_GUARD_ENABLED", stage2)
        self.assertIn("system.opacity_coverage.enabled=${OPACITY_COVERAGE_ENABLED}", stage2)
        self.assertIn("system.loss.lambda_opacity_coverage=${LAMBDA_OPACITY_COVERAGE}", stage2)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests/test_stage_run_scripts.py -v
```

Expected:

```text
FAIL: test_stage2_script_accepts_opacity_repair_overrides
```

- [ ] **Step 3: Add env defaults and Hydra overrides to Stage2 script**

In `scripts/run_stage2_text.sh`, near reference env defaults:

```bash
OPACITY_COVERAGE_ENABLED="${OPACITY_COVERAGE_ENABLED:-false}"
LAMBDA_OPACITY_COVERAGE="${LAMBDA_OPACITY_COVERAGE:-0.0}"
REAR_OPACITY_ENABLED="${REAR_OPACITY_ENABLED:-false}"
LAMBDA_REAR_OPACITY="${LAMBDA_REAR_OPACITY:-0.0}"
PRUNE_REGION_GUARD_ENABLED="${PRUNE_REGION_GUARD_ENABLED:-false}"
```

In the `launch.py` command add:

```bash
  "system.opacity_coverage.enabled=${OPACITY_COVERAGE_ENABLED}" \
  "system.rear_opacity.enabled=${REAR_OPACITY_ENABLED}" \
  "system.prune_region_guard.enabled=${PRUNE_REGION_GUARD_ENABLED}" \
  "system.loss.lambda_opacity_coverage=${LAMBDA_OPACITY_COVERAGE}" \
  "system.loss.lambda_rear_opacity=${LAMBDA_REAR_OPACITY}" \
```

- [ ] **Step 4: Forward env in two-stage script**

In `scripts/run_two_stage.sh`, ensure Stage2 invocation preserves:

```bash
OPACITY_COVERAGE_ENABLED="${OPACITY_COVERAGE_ENABLED:-false}" \
LAMBDA_OPACITY_COVERAGE="${LAMBDA_OPACITY_COVERAGE:-0.0}" \
REAR_OPACITY_ENABLED="${REAR_OPACITY_ENABLED:-false}" \
LAMBDA_REAR_OPACITY="${LAMBDA_REAR_OPACITY:-0.0}" \
PRUNE_REGION_GUARD_ENABLED="${PRUNE_REGION_GUARD_ENABLED:-false}" \
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m unittest tests/test_stage_run_scripts.py -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit Task 7**

Run:

```bash
git add scripts/run_stage2_text.sh scripts/run_two_stage.sh tests/test_stage_run_scripts.py
git commit -m "暴露 opacity 修复实验开关"
```

---

### Task 8: Final Verification and Documentation Update

**Files:**
- Modify: `docs/2026-04-29-group-meeting-progress-report.md` if desired
- Create or modify: `docs/2026-04-29-opacity-reference-implementation-status.md`

- [ ] **Step 1: Run full relevant unit tests**

Run:

```bash
python -m unittest \
  tests/test_opacity_diagnostics.py \
  tests/test_region_prune_guard.py \
  tests/test_opacity_alpha_pipeline.py \
  tests/test_reference_sheet.py \
  tests/test_reference_fidelity_config.py \
  tests/test_stage_run_scripts.py \
  -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run compile checks**

Run:

```bash
python -m py_compile \
  scripts/diagnose_opacity_thickness.py \
  threestudio/utils/opacity_diagnostics.py \
  gaussiansplatting/scene/gaussian_flame_model.py \
  threestudio/systems/Head3DGSLKs.py
```

Expected:

```text
No traceback
```

The existing `SyntaxWarning` for `(\d+)\.png` may still appear until that unrelated string is cleaned up.

- [ ] **Step 3: Run diagnostics on existing output**

Run:

```bash
python scripts/diagnose_opacity_thickness.py \
  --ply outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/save/last.ply \
  --output outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/diagnostics/opacity_thickness
```

Expected:

```text
Wrote outputs/cristiano_ronaldo_ref_v120260428-161859/headstudio-stage2-text/diagnostics/opacity_thickness/gaussian_region_stats.json
```

- [ ] **Step 4: Write implementation status doc**

Create `docs/2026-04-29-opacity-reference-implementation-status.md`:

```markdown
# 2026-04-29 Opacity 与 Reference Fidelity 实现状态

## 已完成

- Phase A CPU 诊断工具：`scripts/diagnose_opacity_thickness.py`
- Gaussian opacity/scaling 统计：`threestudio/utils/opacity_diagnostics.py`
- FLAME face binding 区域分类接口：`GaussianFlameModel.get_gaussian_region_labels`
- 默认关闭的 opacity coverage / rear opacity / prune region guard 配置
- Stage2 脚本 opacity 修复实验开关
- Reference feature / identity loss 的默认关闭配置表面

## 尚未完成

- GPU 固定视角 RGB / opacity / depth 渲染诊断
- DINO / CLIP feature loss 实际 backbone
- ArcFace identity loss
- temporal feature consistency

## 下一步实验

先对现有 C 罗 Stage2 输出运行 opacity 诊断，再决定是否开启：

```bash
OPACITY_COVERAGE_ENABLED=true \
LAMBDA_OPACITY_COVERAGE=0.02 \
PRUNE_REGION_GUARD_ENABLED=true \
bash scripts/run_stage2_text.sh ...
```
```

- [ ] **Step 5: Commit Task 8**

Run:

```bash
git add docs/2026-04-29-opacity-reference-implementation-status.md
git commit -m "记录 opacity 与 reference 实现状态"
```

---

## Self-Review Notes

- Spec coverage:
  - Phase A diagnostics: Tasks 1, 3, and 8.
  - Region stats/classification: Tasks 1 and 2.
  - Phase B opacity coverage/rear opacity/prune guard: Tasks 4, 5, and 7.
  - Phase C feature-loss upgrade surface: Task 6.
  - Testing and docs: Task 8.
- Scope intentionally excludes actual DINO/CLIP/ArcFace model loading in this plan. The spec says these come after opacity repair is validated; this plan creates the config/logging surface and leaves backbone implementation for a follow-up plan.
