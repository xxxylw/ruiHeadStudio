from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_temporal_window_config_and_batch_contract_are_declared():
    data_source = (ROOT / "threestudio/data/uncond_rand_exp.py").read_text()
    config = (ROOT / "configs/headstudio.yaml").read_text()

    for field in [
        "temporal_window_enabled: bool = False",
        "temporal_window_length: int = 2",
        "temporal_window_stride: int = 1",
        "temporal_primary_index: int = 0",
        "temporal_same_camera: bool = True",
    ]:
        assert field in data_source

    for key in [
        "temporal_window_enabled: false",
        "temporal_window_length: 2",
        "temporal_window_stride: 1",
        "temporal_primary_index: 0",
        "temporal_same_camera: true",
    ]:
        assert key in config

    for field in [
        '"temporal_enabled": temporal_enabled',
        '"temporal_source_name": temporal_source_name',
        '"temporal_source_index": temporal_source_index',
        '"temporal_sequence_index": temporal_sequence_index',
        '"temporal_frame_indices": temporal_frame_indices',
        '"temporal_primary_index": temporal_primary_index',
        '"temporal_window_length": temporal_window_length',
        '"temporal_expression": temporal_expression',
        '"temporal_jaw_pose": temporal_jaw_pose',
        '"temporal_leye_pose": temporal_leye_pose',
        '"temporal_reye_pose": temporal_reye_pose',
        '"temporal_neck_pose": temporal_neck_pose',
    ]:
        assert field in data_source


def test_gaussian_temporal_state_helper_preserves_pose_and_exposes_scale_ratio():
    source = (ROOT / "gaussiansplatting/scene/gaussian_flame_model.py").read_text()

    assert "def get_temporal_surface_states(" in source
    assert "saved_expression = self._expression" in source
    assert "try:" in source
    assert "finally:" in source
    assert "self._expression = saved_expression" in source
    assert '"xyz": self.get_xyz' in source
    assert '"triangle_centroid": triangle_centroid' in source
    assert '"triangle_area": triangle_area' in source
    assert '"scaling": scaling' in source
    assert '"scale_ratio": scaling / ((triangle_area + 1e-10).sqrt().unsqueeze(-1))' in source


def test_system_temporal_losses_are_configured_and_gated():
    system_source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text()
    config = (ROOT / "configs/headstudio.yaml").read_text()

    assert "temporal_loss_start_step: int = 2400" in system_source
    assert "temporal_loss_start_step: 2400" in config
    assert "lambda_temporal_motion: 0.0" in config
    assert "lambda_temporal_scale_ratio: 0.0" in config

    assert "self.cfg.loss.lambda_temporal_motion = 0.01 * self.cfg.loss.lambda_temporal_motion" in system_source
    assert "self.cfg.loss.lambda_temporal_scale_ratio = 0.01 * self.cfg.loss.lambda_temporal_scale_ratio" in system_source
    assert "def compute_temporal_losses(self, batch):" in system_source
    assert "states = self.gaussian.get_temporal_surface_states(" in system_source
    assert "loss_temporal_motion" in system_source
    assert "loss_temporal_scale_ratio" in system_source
    assert 'batch.get("temporal_enabled", False)' in system_source
    assert "self.true_global_step >= self.cfg.temporal_loss_start_step" in system_source
    assert 'self.log("train/loss_temporal_motion", temporal_losses["motion"])' in system_source
    assert 'self.log("train/loss_temporal_scale_ratio", temporal_losses["scale_ratio"])' in system_source


if __name__ == "__main__":
    test_temporal_window_config_and_batch_contract_are_declared()
    test_gaussian_temporal_state_helper_preserves_pose_and_exposes_scale_ratio()
    test_system_temporal_losses_are_configured_and_gated()
