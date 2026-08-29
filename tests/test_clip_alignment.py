from pathlib import Path
import importlib.util

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

_MODULE_SPEC = importlib.util.spec_from_file_location(
    "clip_alignment_under_test", ROOT / "threestudio/models/clip_alignment.py"
)
_CLIP_ALIGNMENT = importlib.util.module_from_spec(_MODULE_SPEC)
assert _MODULE_SPEC.loader is not None
_MODULE_SPEC.loader.exec_module(_CLIP_ALIGNMENT)
clip_decay_weight = _CLIP_ALIGNMENT.clip_decay_weight
normalized_parameter_drift = _CLIP_ALIGNMENT.normalized_parameter_drift
frequency_quality_loss = _CLIP_ALIGNMENT.frequency_quality_loss
quality_ramp_weight = _CLIP_ALIGNMENT.quality_ramp_weight
rendered_reference_loss = _CLIP_ALIGNMENT.rendered_reference_loss
reference_statistics_loss = _CLIP_ALIGNMENT.reference_statistics_loss
blend_alignment_losses = _CLIP_ALIGNMENT.blend_alignment_losses


def test_clip_alignment_has_warmup_and_cosine_distance_contract():
    source = (ROOT / "threestudio/models/clip_alignment.py").read_text(encoding="utf-8")

    assert "def clip_alignment_weight(" in source
    assert "return 0.0 if global_step < start_step else base_weight" in source
    assert "def cosine_alignment_loss(" in source
    assert "1.0 - F.cosine_similarity" in source
    assert 'download_root=os.path.expanduser("~/.cache/clip")' in source


def test_blend_alignment_losses_interpolates_two_frozen_teachers():
    primary = torch.tensor(0.2)
    recovery = torch.tensor(0.8)
    assert blend_alignment_losses(primary, recovery, 0.0).item() == pytest.approx(0.2)
    assert blend_alignment_losses(primary, recovery, 1.0).item() == pytest.approx(0.8)
    assert blend_alignment_losses(primary, recovery, 0.25).item() == pytest.approx(0.35)


def test_blend_alignment_losses_rejects_invalid_weight():
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend_alignment_losses(torch.tensor(0.2), torch.tensor(0.8), 1.1)


def test_clip_alignment_supports_foreground_crops_and_view_conditioned_text():
    source = (ROOT / "threestudio/models/clip_alignment.py").read_text(encoding="utf-8")

    assert "def foreground_crop(" in source
    assert "opacity = opacity.permute(0, 3, 1, 2)" in source
    assert "foreground = alpha[0] > 0.1" in source
    assert "def text_features_for_azimuth(" in source
    assert '"front": f"front view of {prompt}"' in source
    assert '"back": f"backside view of {prompt}"' in source
    assert "foreground_only: bool = False" in source
    assert "view_dependent: bool = False" in source


def test_headstudio_only_loads_clip_when_its_loss_is_enabled():
    system_source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text(encoding="utf-8")
    config_source = (ROOT / "configs/headstudio.yaml").read_text(encoding="utf-8")

    assert "clip_model_name: str = \"ViT-L/14\"" in system_source
    assert "clip_start_step: int = 2000" in system_source
    assert "clip_foreground_only: bool = False" in system_source
    assert "clip_use_view_prompt: bool = False" in system_source
    assert "clip_global_weight: float = 0.0" in system_source
    assert "clip_foreground_weight: float = 0.0" in system_source
    assert "clip_view_weight: float = 0.0" in system_source
    assert "clip_recovery_model_name: str = \"\"" in system_source
    assert "clip_recovery_weight: float = 0.0" in system_source
    assert "blend_alignment_losses" in system_source
    assert "if self.C(self.cfg.loss.lambda_clip) > 0.0:" in system_source
    assert "self.clip_alignment = CLIPAlignment(" in system_source
    assert "clip_weight = clip_alignment_weight(" in system_source
    assert "opacity=out[\"opacity\"]" in system_source
    assert "azimuth=batch[\"azimuth\"]" in system_source
    assert 'alpha = render_pkg.get("alpha_3dgs")' in system_source
    assert "opacities = []" in system_source
    assert "render_pkg[\"opacity\"] = opacities" in system_source
    assert "component_weights = {" in system_source
    assert "loss_clip = loss_clip / component_total" in system_source
    assert 'self.log("train/loss_clip", loss_clip)' in system_source
    assert 'self.log("train/loss_clip_global", loss_clip_global)' in system_source
    assert "clip_global_weight: 0.0" in config_source
    assert "clip_foreground_weight: 0.0" in config_source
    assert "clip_view_weight: 0.0" in config_source
    assert "clip_recovery_model_name: \"\"" in config_source
    assert "clip_recovery_weight: 0.0" in config_source
    assert "clip_decay_start_step: int = 0" in system_source
    assert "clip_decay_end_step: int = 0" in system_source
    assert "lambda_trust: float = 0.0" in system_source
    assert "self.trust_region_anchor" in system_source
    assert "normalized_parameter_drift" in system_source
    assert "clip_decay_weight" in system_source
    assert "frequency_quality_loss" in system_source
    assert "quality_ramp_weight" in system_source
    assert "quality_start_step: int = 0" in system_source
    assert "quality_ramp_end_step: int = 0" in system_source
    assert "lambda_frequency_quality: float = 0.0" in system_source
    assert "lambda_rendered_reference: float = 0.0" in system_source
    assert "lambda_reference_statistics: float = 0.0" in system_source
    assert "reference_statistics_loss" in system_source
    assert "self.reference_gaussian = None" in system_source
    assert "track_stats=False" in system_source
    assert "lambda_clip: 0.0" in config_source


def test_gaussian_ply_reload_resets_pointwise_optimizer_buffers():
    source = (ROOT / "gaussiansplatting/scene/gaussian_flame_model.py").read_text(encoding="utf-8")
    load_source = source.split("def load_ply", 1)[1].split("def training_setup", 1)[0]

    assert "self.max_radii2D = torch.zeros((self.num_gs), device=\"cuda\")" in load_source


def test_training_writes_parameter_drift_provenance():
    source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text(encoding="utf-8")

    assert "parameter_drift.json" in source
    assert "max_abs_xyz_drift" in source


def test_training_writes_gradient_probe_provenance():
    source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text(encoding="utf-8")

    assert "gradient_probe.json" in source
    assert "grad_is_none" in source


def test_clip_decay_weight_has_stable_linear_window():
    assert clip_decay_weight(0.006, 7000, 7000, 8000) == pytest.approx(0.006)
    assert clip_decay_weight(0.006, 7500, 7000, 8000) == pytest.approx(0.003)
    assert clip_decay_weight(0.006, 8000, 7000, 8000) == pytest.approx(0.0)
    assert clip_decay_weight(0.006, 9000, 7000, 8000) == pytest.approx(0.0)


def test_normalized_parameter_drift_has_gradients_but_detaches_anchor():
    current = torch.tensor([[2.0, 4.0]], requires_grad=True)
    anchor = torch.tensor([[1.0, 2.0]], requires_grad=True)
    normalizer = torch.tensor([[1.0, 2.0]])

    loss = normalized_parameter_drift(current, anchor, normalizer)
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert current.grad is not None
    assert anchor.grad is None


def test_normalized_parameter_drift_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        normalized_parameter_drift(torch.zeros(2, 3), torch.zeros(2, 2))


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


def test_quality_ramp_weight_has_stable_linear_window():
    assert quality_ramp_weight(0.002, 10000, 11000, 12000) == pytest.approx(0.0)
    assert quality_ramp_weight(0.002, 11500, 11000, 12000) == pytest.approx(0.001)
    assert quality_ramp_weight(0.002, 12000, 11000, 12000) == pytest.approx(0.002)
    assert quality_ramp_weight(0.002, 13000, 11000, 12000) == pytest.approx(0.002)


def test_rendered_reference_loss_is_zero_for_identical_images():
    image = torch.rand((1, 3, 8, 8), requires_grad=True)
    loss = rendered_reference_loss(image, image.detach())
    assert torch.isclose(loss, torch.zeros_like(loss))


def test_rendered_reference_loss_has_finite_gradients_and_alpha_support():
    image = torch.rand((1, 3, 8, 8), requires_grad=True)
    reference = torch.zeros_like(image)
    alpha = torch.zeros((1, 8, 8, 1))
    alpha[:, 2:6, 2:6] = 1.0
    loss = rendered_reference_loss(image, reference, alpha)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(image.grad).all()


def test_rendered_reference_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        rendered_reference_loss(torch.zeros((1, 3, 8, 8)), torch.zeros((1, 3, 7, 8)))


def test_reference_statistics_loss_is_zero_for_identical_images():
    image = torch.rand((1, 3, 16, 16), requires_grad=True)
    loss = reference_statistics_loss(image, image.detach())
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1.0e-6)


def test_reference_statistics_loss_has_finite_gradients_and_alpha_support():
    image = torch.rand((1, 3, 16, 16), requires_grad=True)
    reference = torch.zeros_like(image)
    alpha = torch.zeros((1, 16, 16, 1))
    alpha[:, 4:12, 4:12] = 1.0
    loss = reference_statistics_loss(image, reference, alpha)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(image.grad).all()


def test_reference_statistics_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        reference_statistics_loss(torch.zeros((1, 3, 16, 16)), torch.zeros((1, 3, 15, 16)))
