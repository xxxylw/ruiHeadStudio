"""Focused regression test for FluxReferenceGuidance.set_min_max_steps.

Head3DGSLKs.training_step calls set_min_max_steps once training passes
half_scheduler_max_step (step 3001 with the default config). The method was
missing from FluxReferenceGuidance, so every full run crashed with
AttributeError exactly at that step. This test pins the method on the real
class definition without importing the heavy threestudio plugin tree: the
module under test is executed with minimal stubs for threestudio, and the
method is exercised on an instance created without __init__ so no model or
scheduler machinery is touched.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "threestudio/models/guidance/flux_reference_guidance.py"


def _load_flux_module_with_threestudio_stub():
    """Execute the guidance module with stubbed threestudio so FluxReferenceGuidance is defined."""
    threestudio = types.ModuleType("threestudio")
    threestudio.register = lambda *args, **kwargs: (lambda cls: cls)

    utils = types.ModuleType("threestudio.utils")
    utils.__path__ = []
    base = types.ModuleType("threestudio.utils.base")

    class BaseObject:
        class Config:
            pass

    base.BaseObject = BaseObject
    typing = types.ModuleType("threestudio.utils.typing")

    stubs = {
        "threestudio": threestudio,
        "threestudio.utils": utils,
        "threestudio.utils.base": base,
        "threestudio.utils.typing": typing,
    }
    for name, module in stubs.items():
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location(
        "flux_reference_guidance_stub_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return module


def test_set_min_max_steps_regression():
    saved_skip = os.environ.pop("FLUX_SKIP_THREESTUDIO", None)
    try:
        flux = _load_flux_module_with_threestudio_stub()
        assert flux.FluxReferenceGuidance is not None
        assert callable(getattr(flux.FluxReferenceGuidance, "set_min_max_steps", None))

        # Custom values from the Head3DGSLKs call site (step 3001+):
        # set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55).
        guidance = flux.FluxReferenceGuidance.__new__(flux.FluxReferenceGuidance)
        guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)
        assert guidance.min_step_percent == 0.02
        assert guidance.max_step_percent == 0.55
        assert isinstance(guidance.min_step_percent, float)
        assert isinstance(guidance.max_step_percent, float)
        # The method must only store the two percent attributes; it must not
        # introduce timestep/scheduler state.
        assert not hasattr(guidance, "min_step")
        assert not hasattr(guidance, "max_step")

        # Defaults and float coercion.
        defaults = flux.FluxReferenceGuidance.__new__(flux.FluxReferenceGuidance)
        defaults.set_min_max_steps()
        assert defaults.min_step_percent == 0.02
        assert defaults.max_step_percent == 0.98

        coerced = flux.FluxReferenceGuidance.__new__(flux.FluxReferenceGuidance)
        coerced.set_min_max_steps(min_step_percent=1, max_step_percent=2)
        assert coerced.min_step_percent == 1.0
        assert coerced.max_step_percent == 2.0
    finally:
        if saved_skip is not None:
            os.environ["FLUX_SKIP_THREESTUDIO"] = saved_skip
