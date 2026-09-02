"""Focused regression tests for Head3DGSLKs.on_train_end parameter-drift stats.

Dynamic densification/pruning changes the gaussian point count (the initial
cloud is pts_num=100000; after prune/densify the trained cloud has a different
size), which made the old on_train_end per-point subtraction
(final _xyz - initial _xyz) raise a RuntimeError at the very end of every full
10000-step run. The fix compares only the common prefix, records both counts
and the comparison mode, and never raises, so the closing statistics can never
block a completed run.

These tests execute the real on_train_end method body on lightweight fakes.
The module under test is imported with minimal stubs for the heavy
threestudio/gaussiansplatting plugin trees (same approach as the
set_min_max_steps regression test).
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "threestudio/systems/Head3DGSLKs.py"

_STUB_MODULE_NAMES = [
    "threestudio",
    "threestudio.utils",
    "threestudio.utils.ops",
    "threestudio.utils.typing",
    "threestudio.systems",
    "threestudio.systems.base",
    "threestudio.models",
    "threestudio.models.clip_alignment",
    "gaussiansplatting",
    "gaussiansplatting.gaussian_renderer",
    "gaussiansplatting.scene",
    "gaussiansplatting.scene.cameras",
    "gaussiansplatting.scene.gaussian_flame_model",
    "gaussiansplatting.arguments",
]


def _make_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _install_stub_modules():
    import typing as _typing

    threestudio = _make_package("threestudio")
    threestudio.register = lambda *a, **k: (lambda cls: cls)
    threestudio.warn = lambda *a, **k: None

    utils = _make_package("threestudio.utils")
    ops = _make_package("threestudio.utils.ops")
    ops.binary_cross_entropy = lambda *a, **k: None
    ops.dot = lambda *a, **k: None
    typing_module = _make_package("threestudio.utils.typing")
    for _name in dir(_typing):
        if not _name.startswith("_"):
            setattr(typing_module, _name, getattr(_typing, _name))

    systems = _make_package("threestudio.systems")
    base = _make_package("threestudio.systems.base")

    class _BaseLift3DSystem:
        class Config:
            pass

    base.BaseLift3DSystem = _BaseLift3DSystem

    models = _make_package("threestudio.models")
    clip_alignment = _make_package("threestudio.models.clip_alignment")
    for _name in [
        "CLIPAlignment",
        "blend_alignment_losses",
        "artifact_suppression_loss",
        "clip_alignment_weight",
        "clip_decay_weight",
        "frequency_quality_loss",
        "normalized_parameter_drift",
        "quality_ramp_weight",
        "reference_statistics_loss",
        "rendered_reference_loss",
    ]:
        setattr(clip_alignment, _name, lambda *a, **k: None)

    _make_package("gaussiansplatting")
    gaussian_renderer = _make_package("gaussiansplatting.gaussian_renderer")
    gaussian_renderer.render = lambda *a, **k: None
    scene = _make_package("gaussiansplatting.scene")
    scene.GaussianModel = type("GaussianModel", (), {})
    cameras = _make_package("gaussiansplatting.scene.cameras")
    cameras.Camera = type("Camera", (), {})
    cameras.MiniCam = type("MiniCam", (), {})
    flame = _make_package("gaussiansplatting.scene.gaussian_flame_model")
    flame.GaussianFlameModel = type("GaussianFlameModel", (), {})
    arguments = _make_package("gaussiansplatting.arguments")
    for _name in ["ModelParams", "PipelineParams", "OptimizationParams"]:
        setattr(arguments, _name, type(_name, (), {}))
    arguments.get_combined_args = lambda *a, **k: None

    return systems, models


def _load_system_module():
    """Execute Head3DGSLKs.py with stubbed heavy imports and return the module."""
    _install_stub_modules()
    spec = importlib.util.spec_from_file_location("head3dgslks_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name in _STUB_MODULE_NAMES:
            sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def system_module():
    return _load_system_module()


def _make_system(system_module, initial_xyz, final_xyz, save_dir):
    system = system_module.Head3DGSLKsRig.__new__(system_module.Head3DGSLKsRig)
    system._initial_gaussian_xyz = initial_xyz
    gaussian = types.SimpleNamespace()
    gaussian._xyz = final_xyz
    system.gaussian = gaussian
    system.get_save_path = lambda name: str(save_dir / name)
    return system


def _run_on_train_end(system_module, system):
    # Unbound call exercises the real method body on the fake system.
    system_module.Head3DGSLKsRig.on_train_end(system)


def test_on_train_end_point_count_shrink_does_not_raise(system_module, tmp_path):
    # Real failure mode: 100000 initial gaussians pruned down to 35093.
    initial = torch.zeros((100000, 3))
    final = torch.full((35093, 3), 1.0)
    system = _make_system(system_module, initial, final, tmp_path)

    _run_on_train_end(system_module, system)  # must not raise

    record = json.loads((tmp_path / "parameter_drift.json").read_text())
    assert record["initial_point_count"] == 100000
    assert record["final_point_count"] == 35093
    assert record["compared_point_count"] == 35093
    assert record["drift_compare_mode"] == "prefix"
    assert record["max_abs_xyz_drift"] == 1.0
    assert record["mean_abs_xyz_drift"] == 1.0


def test_on_train_end_point_count_growth_does_not_raise(system_module, tmp_path):
    # Densification may also grow the cloud beyond the initial count.
    initial = torch.zeros((100000, 3))
    final = torch.full((120000, 3), 0.5)
    system = _make_system(system_module, initial, final, tmp_path)

    _run_on_train_end(system_module, system)  # must not raise

    record = json.loads((tmp_path / "parameter_drift.json").read_text())
    assert record["initial_point_count"] == 100000
    assert record["final_point_count"] == 120000
    assert record["compared_point_count"] == 100000
    assert record["drift_compare_mode"] == "prefix"
    assert record["max_abs_xyz_drift"] == 0.5
    assert record["mean_abs_xyz_drift"] == 0.5


def test_on_train_end_equal_counts_keep_full_drift_semantics(system_module, tmp_path):
    initial = torch.zeros((5, 3))
    final = torch.full((5, 3), 0.25)
    system = _make_system(system_module, initial, final, tmp_path)

    _run_on_train_end(system_module, system)  # must not raise

    record = json.loads((tmp_path / "parameter_drift.json").read_text())
    assert record["initial_point_count"] == 5
    assert record["final_point_count"] == 5
    assert record["compared_point_count"] == 5
    assert record["drift_compare_mode"] == "full"
    assert record["max_abs_xyz_drift"] == 0.25
    assert record["mean_abs_xyz_drift"] == 0.25


def test_on_train_end_empty_final_cloud_records_status(system_module, tmp_path):
    initial = torch.zeros((100000, 3))
    final = torch.zeros((0, 3))
    system = _make_system(system_module, initial, final, tmp_path)

    _run_on_train_end(system_module, system)  # must not raise

    record = json.loads((tmp_path / "parameter_drift.json").read_text())
    assert record["initial_point_count"] == 100000
    assert record["final_point_count"] == 0
    assert record["compared_point_count"] == 0
    assert record["drift_compare_mode"] == "empty"
    assert record["max_abs_xyz_drift"] is None
    assert record["mean_abs_xyz_drift"] is None


def test_on_train_end_without_initial_snapshot_is_noop(system_module, tmp_path):
    system = system_module.Head3DGSLKsRig.__new__(system_module.Head3DGSLKsRig)
    gaussian = types.SimpleNamespace()
    gaussian._xyz = torch.zeros((10, 3))
    system.gaussian = gaussian
    system.get_save_path = lambda name: str(tmp_path / name)

    _run_on_train_end(system_module, system)  # must not raise

    assert not (tmp_path / "parameter_drift.json").exists()

