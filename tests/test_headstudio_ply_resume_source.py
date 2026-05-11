from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_headstudio_supports_ply_resume_initialization():
    source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text()

    assert "gaussian_init_ply: Optional[str] = None" in source
    assert "gaussian_init_step: int = 0" in source
    assert "self.gaussian.load_ply(self.cfg.gaussian_init_ply)" in source
    assert "return super().true_global_step + self.cfg.gaussian_init_step" in source


def test_gaussian_flame_load_ply_updates_point_count():
    source = (ROOT / "gaussiansplatting/scene/gaussian_flame_model.py").read_text()

    assert "self.num_gs = self._xyz.shape[0]" in source


if __name__ == "__main__":
    test_headstudio_supports_ply_resume_initialization()
    test_gaussian_flame_load_ply_updates_point_count()
