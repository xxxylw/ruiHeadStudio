import unittest
from pathlib import Path


class TestReferenceFidelityConfig(unittest.TestCase):
    def test_stage2_reference_fidelity_defaults_are_disabled(self):
        cfg = Path("configs/headstudio_stage2_text.yaml").read_text(encoding="utf-8")

        self.assertIn("reference_fidelity:", cfg)
        self.assertIn("enabled: false", cfg)
        self.assertIn('metadata_path: ""', cfg)
        self.assertIn("start_step: 1200", cfg)
        self.assertIn("end_step: ${trainer.max_steps}", cfg)
        self.assertIn("lambda_ref_person: 0.0", cfg)
        self.assertIn("lambda_ref_face: 0.0", cfg)
        self.assertIn("lambda_ref_temporal_face: 0.0", cfg)

    def test_stage_scripts_accept_reference_metadata_override(self):
        stage2 = Path("scripts/run_stage2_text.sh").read_text(encoding="utf-8")
        two_stage = Path("scripts/run_two_stage.sh").read_text(encoding="utf-8")

        self.assertIn("REFERENCE_METADATA", stage2)
        self.assertIn("system.reference_fidelity.enabled=${REFERENCE_FIDELITY_ENABLED}", stage2)
        self.assertIn("system.reference_fidelity.metadata_path=${REFERENCE_METADATA}", stage2)
        self.assertIn("system.loss.lambda_ref_person=${REFERENCE_LAMBDA_REF_PERSON}", stage2)
        self.assertIn("system.loss.lambda_ref_face=${REFERENCE_LAMBDA_REF_FACE}", stage2)
        self.assertIn("system.loss.lambda_ref_temporal_face=${REFERENCE_LAMBDA_REF_TEMPORAL_FACE}", stage2)
        self.assertIn("REFERENCE_METADATA", two_stage)
        self.assertIn("REFERENCE_FIDELITY_ENABLED", two_stage)

    def test_head_system_contains_reference_fidelity_hooks(self):
        source = Path("threestudio/systems/Head3DGSLKs.py").read_text(encoding="utf-8")

        self.assertIn("load_reference_sheet", source)
        self.assertIn("reference_fidelity: dict", source)
        self.assertIn("self.reference_sheet", source)
        self.assertIn("compute_reference_fidelity_losses", source)
        self.assertIn("target.to(device=images.device, dtype=images.dtype)", source)
        self.assertIn("train/loss_ref_person", source)
        self.assertIn("train/loss_ref_face", source)
        self.assertIn("train/loss_ref_temporal_face", source)


if __name__ == "__main__":
    unittest.main()
