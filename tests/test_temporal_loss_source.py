from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="ignore")


def test_local_scale_ratio_loss_is_configured_and_logged():
    system_source = read_source("threestudio/systems/Head3DGSLKs.py")
    config = read_source("configs/headstudio.yaml")

    assert "scale_ratio_threshold: float = 0.5" in system_source
    assert "scale_ratio_threshold: 0.5" in config
    assert "lambda_scaling: 10.0" in config
    assert "lambda_scale_ratio: 5.0" in config

    assert "self.cfg.loss.lambda_scale_ratio = 0.01 * self.cfg.loss.lambda_scale_ratio" in system_source
    assert "lambda_scaling = self.C(self.cfg.loss.lambda_scaling)" in system_source
    assert "lambda_scale_ratio = self.C(self.cfg.loss.lambda_scale_ratio)" in system_source
    assert "if lambda_scaling > 0.0:" in system_source
    assert "if lambda_scale_ratio > 0.0:" in system_source
    assert "scale_ratio = scaling / (tris_scaling.unsqueeze(-1) + 1e-10)" in system_source
    assert "scale_ratio_excess = F.relu(scale_ratio - self.cfg.scale_ratio_threshold)" in system_source
    assert "loss_scale_ratio = (scale_ratio_excess ** 2).mean()" in system_source
    assert 'self.log("train/loss_scale_ratio", loss_scale_ratio)' in system_source
    assert "loss += loss_scale_ratio * lambda_scale_ratio" in system_source


def test_local_position_loss_replaces_old_position_default():
    system_source = read_source("threestudio/systems/Head3DGSLKs.py")
    config = read_source("configs/headstudio.yaml")

    assert "lambda_position: 0.0" in config
    assert "lambda_local_position: 5.0" in config
    assert "self.cfg.loss.lambda_local_position = 0.01 * self.cfg.loss.lambda_local_position" in system_source
    assert "lambda_position = self.C(self.cfg.loss.lambda_position)" in system_source
    assert "lambda_local_position = self.C(self.cfg.loss.lambda_local_position)" in system_source
    assert "if lambda_position > 0.0:" in system_source
    assert "if lambda_local_position > 0.0:" in system_source
    assert 'self.log("train/loss_local_position", loss_position)' in system_source
    assert "loss += loss_position * lambda_local_position" in system_source


def test_surface_constraint_losses_are_configured_and_logged():
    gaussian_source = read_source("gaussiansplatting/scene/gaussian_flame_model.py")
    system_source = read_source("threestudio/systems/Head3DGSLKs.py")
    config = read_source("configs/headstudio.yaml")

    assert "surface_constraint_start_step: int = 2400" in system_source
    assert "surface_constraint_start_step: 2400" in config
    assert "lambda_barycentric_inside: 1.0" in config
    assert "lambda_normal_offset: 0.5" in config
    assert "def get_surface_constraint_terms(self):" in gaussian_source
    assert "barycentric = torch.stack([u, v, w], dim=1)" in gaussian_source
    assert "return barycentric, normal_offset" in gaussian_source
    assert "lambda_barycentric_inside = self.C(self.cfg.loss.lambda_barycentric_inside)" in system_source
    assert "lambda_normal_offset = self.C(self.cfg.loss.lambda_normal_offset)" in system_source
    assert "loss_barycentric_inside = F.relu(-barycentric).mean()" in system_source
    assert "loss_normal_offset = torch.abs(normal_offset).mean()" in system_source
    assert 'self.log("train/loss_barycentric_inside", loss_barycentric_inside)' in system_source
    assert 'self.log("train/loss_normal_offset", loss_normal_offset)' in system_source


def test_scale_ratio_loss_documents_opacity_weighting_as_future_experiment():
    system_source = read_source("threestudio/systems/Head3DGSLKs.py")

    assert "Opacity weighting is intentionally left out for the first ratio-loss experiment." in system_source
