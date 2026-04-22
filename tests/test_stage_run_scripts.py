import os
import unittest


class TestStageRunScripts(unittest.TestCase):
    def test_stage_run_scripts_exist(self):
        self.assertTrue(os.path.exists("scripts/run_stage1_prior.sh"))
        self.assertTrue(os.path.exists("scripts/run_stage2_text.sh"))


if __name__ == "__main__":
    unittest.main()
