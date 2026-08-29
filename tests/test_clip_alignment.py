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


def test_clip_alignment_has_warmup_and_cosine_distance_contract():
    source = (ROOT / "threestudio/models/clip_alignment.py").read_text(encoding="utf-8")

    assert "def clip_alignment_weight(" in source
    assert "return 0.0 if global_step < start_step else base_weight" in source
    assert "def cosine_alignment_loss(" in source
    assert "1.0 - F.cosine_similarity" in source
    assert 'download_root=os.path.expanduser("~/.cache/clip")' in source


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
    assert "clip_decay_start_step: int = 0" in system_source
    assert "clip_decay_end_step: int = 0" in system_source
    assert "lambda_trust: float = 0.0" in system_source
    assert "self.trust_region_anchor" in system_source
    assert "normalized_parameter_drift" in system_source
    assert "clip_decay_weight" in system_source
    assert "lambda_clip: 0.0" in config_source


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
