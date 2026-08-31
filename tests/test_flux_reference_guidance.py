from pathlib import Path
import importlib.util

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "flux_reference_guidance_under_test",
    ROOT / "threestudio/models/guidance/flux_reference_guidance.py",
)
FLUX = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(FLUX)


def test_flux_prompt_includes_view_without_changing_identity_text():
    prompt = FLUX.build_flux_view_prompt("a realistic 3D head avatar", 45.0, -30.0)
    assert "a realistic 3D head avatar" in prompt
    assert "azimuth" in prompt
    assert "elevation" in prompt


def test_flux_reference_loss_is_zero_for_identical_images_and_has_gradients():
    image = torch.rand((1, 3, 8, 8), requires_grad=True)
    reference = image.detach().clone()
    loss = FLUX.flux_reference_loss(image, reference)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1.0e-6)
    noisy = torch.rand((1, 3, 8, 8), requires_grad=True)
    FLUX.flux_reference_loss(noisy, reference).backward()
    assert torch.isfinite(noisy.grad).all()


def test_flux_reference_loss_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="BCHW"):
        FLUX.flux_reference_loss(torch.zeros((3, 8, 8)), torch.zeros((1, 3, 8, 8)))


def test_backend_uses_single_file_strategy_when_checkpoint_exists(tmp_path):
    checkpoint = tmp_path / "flux1-schnell.safetensors"
    checkpoint.write_bytes(b"placeholder")
    backend = FLUX.FluxReferenceBackend("repo", "cpu", single_file_path=str(checkpoint))
    assert backend._load_strategy() == "single_file"


def test_backend_falls_back_to_pipeline_when_checkpoint_missing(tmp_path):
    backend = FLUX.FluxReferenceBackend("repo", "cpu", single_file_path=str(tmp_path / "missing.safetensors"))
    assert backend._load_strategy() == "pipeline"


def test_backend_falls_back_to_pipeline_without_single_file_path():
    backend = FLUX.FluxReferenceBackend("repo", "cpu")
    assert backend._load_strategy() == "pipeline"
