from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_scale_ratio_loss_is_configured_and_logged():
    system_source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text()
    config = (ROOT / "configs/headstudio.yaml").read_text()

    assert "scale_ratio_threshold: float = 0.5" in system_source
    assert "scale_ratio_threshold: 0.5" in config
    assert "lambda_scale_ratio: 2.0" in config

    assert "self.cfg.loss.lambda_scale_ratio = 0.01 * self.cfg.loss.lambda_scale_ratio" in system_source
    assert "scale_ratio = scaling / (tris_scaling.unsqueeze(-1) + 1e-10)" in system_source
    assert "scale_ratio_excess = F.relu(scale_ratio - self.cfg.scale_ratio_threshold)" in system_source
    assert "loss_scale_ratio = (scale_ratio_excess ** 2).mean()" in system_source
    assert 'self.log("train/loss_scale_ratio", loss_scale_ratio)' in system_source
    assert "loss += loss_scale_ratio * self.C(self.cfg.loss.lambda_scale_ratio)" in system_source


def test_scale_ratio_loss_documents_opacity_weighting_as_future_experiment():
    system_source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text()

    assert "Opacity weighting is intentionally left out for the first ratio-loss experiment." in system_source


if __name__ == "__main__":
    test_local_scale_ratio_loss_is_configured_and_logged()
    test_scale_ratio_loss_documents_opacity_weighting_as_future_experiment()
