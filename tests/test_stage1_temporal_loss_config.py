import unittest

from omegaconf import OmegaConf


class TestStage1TemporalLossConfig(unittest.TestCase):
    def test_stage1_enables_adjacent_frame_temporal_stability(self):
        stage1 = OmegaConf.load("configs/headstudio_stage1_prior.yaml")

        self.assertTrue(stage1.data.sample_adjacent_frame)
        self.assertEqual(stage1.data.adjacent_frame_max_offset, 1)
        self.assertGreater(stage1.system.loss.lambda_anchor, 0.0)
        self.assertGreater(stage1.system.loss.lambda_temporal_xyz, 0.0)


if __name__ == "__main__":
    unittest.main()
