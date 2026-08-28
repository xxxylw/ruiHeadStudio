from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]



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
    assert "component_weights = {" in system_source
    assert "loss_clip = loss_clip / component_total" in system_source
    assert 'self.log("train/loss_clip", loss_clip)' in system_source
    assert 'self.log("train/loss_clip_global", loss_clip_global)' in system_source
    assert "clip_global_weight: 0.0" in config_source
    assert "clip_foreground_weight: 0.0" in config_source
    assert "clip_view_weight: 0.0" in config_source
    assert "lambda_clip: 0.0" in config_source
