from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gaussian_flame_exposes_triangle_surface_terms():
    source = (ROOT / "gaussiansplatting/scene/gaussian_flame_model.py").read_text()

    assert "def get_bound_triangles(self):" in source
    assert "def get_surface_constraint_terms(self):" in source
    assert "barycentric = torch.stack([u, v, w], dim=1)" in source
    assert "normal_offset = torch.sum((points - projected) * normal, dim=-1)" in source


def test_headstudio_uses_soft_barycentric_and_normal_offset_losses():
    config = (ROOT / "configs/headstudio.yaml").read_text()
    source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text()

    assert "lambda_barycentric_inside: 1.0" in config
    assert "lambda_normal_offset: 0.5" in config
    assert "surface_constraint_start_step: 2400" in config

    assert "surface_constraint_start_step: int = 2400" in source
    assert "barycentric, normal_offset = self.gaussian.get_surface_constraint_terms()" in source
    assert "loss_barycentric_inside" in source
    assert "loss_normal_offset" in source
    assert "self.log(\"train/loss_barycentric_inside\", loss_barycentric_inside)" in source
    assert "self.log(\"train/loss_normal_offset\", loss_normal_offset)" in source


if __name__ == "__main__":
    test_gaussian_flame_exposes_triangle_surface_terms()
    test_headstudio_uses_soft_barycentric_and_normal_offset_losses()
