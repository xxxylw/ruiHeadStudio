from pathlib import Path
import sys
from dataclasses import dataclass

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
        "temporal_window_enabled: true",
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
    assert "lambda_temporal_motion: 0.1" in config
    assert "lambda_temporal_scale_ratio: 0.02" in config

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


def test_temporal_window_sampler_wraps_single_sequence_after_end():
    source = (ROOT / "threestudio/data/uncond_rand_exp.py").read_text()
    sampler_source = source[
        source.index("@dataclass\nclass PoseSource") : source.index(
            "\n@dataclass\nclass RandomCameraDataModuleConfig"
        )
    ]
    namespace = {
        "dataclass": dataclass,
        "np": np,
        "random": __import__("random"),
        "List": list,
        "Dict": dict,
    }
    exec(sampler_source, namespace)
    PoseSource = namespace["PoseSource"]
    create_pose_source_cursors = namespace["create_pose_source_cursors"]
    sample_pose_window_from_source = namespace[
        "sample_pose_window_from_source"
    ]

    sequence = {
        "expression": np.zeros((3, 100), dtype=np.float32),
        "jaw_pose": np.zeros((3, 3), dtype=np.float32),
        "leye_pose": np.zeros((3, 3), dtype=np.float32),
        "reye_pose": np.zeros((3, 3), dtype=np.float32),
        "neck_pose": np.zeros((3, 3), dtype=np.float32),
    }
    sources = [PoseSource(name="talkshow", weight=1.0, sequences=[sequence])]
    cursors = create_pose_source_cursors(sources)

    first = sample_pose_window_from_source(sources, cursors, 0, window_length=2)
    second = sample_pose_window_from_source(sources, cursors, 0, window_length=2)
    wrapped = sample_pose_window_from_source(sources, cursors, 0, window_length=2)

    assert first["frame_indices"] == [0, 1]
    assert second["frame_indices"] == [1, 2]
    assert wrapped["frame_indices"] == [0, 1]


if __name__ == "__main__":
    test_temporal_window_config_and_batch_contract_are_declared()
    test_gaussian_temporal_state_helper_preserves_pose_and_exposes_scale_ratio()
    test_system_temporal_losses_are_configured_and_gated()
    test_temporal_window_sampler_wraps_single_sequence_after_end()
