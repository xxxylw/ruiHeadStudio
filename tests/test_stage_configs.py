import unittest

from omegaconf import OmegaConf


class TestStageConfigs(unittest.TestCase):
    def test_stage1_and_stage2_configs_define_distinct_modes(self):
        stage1 = OmegaConf.load("configs/headstudio_stage1_prior.yaml")
        stage2 = OmegaConf.load("configs/headstudio_stage2_text.yaml")

        self.assertEqual(stage1.data.difficulty_sampling_mode, "curriculum")
        self.assertEqual(stage2.data.difficulty_sampling_mode, "curriculum")
        self.assertFalse(stage1.system.guidance.use_nfsd)
        self.assertTrue(stage2.system.guidance.use_nfsd)


if __name__ == "__main__":
    unittest.main()
