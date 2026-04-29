import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestOpacityAlphaPipeline(unittest.TestCase):
    def test_head_system_uses_rasterizer_alpha_as_opacity(self):
        source = (REPO_ROOT / "threestudio" / "systems" / "Head3DGSLKs.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('render_pkg["opacity"] = alphas', source)
        self.assertNotIn('render_pkg["opacity"] = depths /', source)

    def test_gaussian_flame_densify_and_prune_uses_min_opacity(self):
        source = (
            REPO_ROOT / "gaussiansplatting" / "scene" / "gaussian_flame_model.py"
        ).read_text(encoding="utf-8")
        method_start = source.index("    def densify_and_prune")
        method_end = source.index("    def prune_only", method_start)
        method_source = source[method_start:method_end]

        self.assertIn("self.get_opacity < min_opacity", method_source)
        self.assertIn("torch.logical_or(prune_mask", method_source)

    def test_stage_configs_define_disabled_opacity_repair_defaults(self):
        stage1 = (REPO_ROOT / "configs" / "headstudio_stage1_prior.yaml").read_text(
            encoding="utf-8"
        )
        stage2 = (REPO_ROOT / "configs" / "headstudio_stage2_text.yaml").read_text(
            encoding="utf-8"
        )

        for cfg in (stage1, stage2):
            self.assertIn("opacity_coverage:", cfg)
            self.assertIn("rear_opacity:", cfg)
            self.assertIn("prune_region_guard:", cfg)
            self.assertIn("lambda_opacity_coverage: 0.0", cfg)
            self.assertIn("lambda_rear_opacity: 0.0", cfg)

    def test_head_system_contains_disabled_opacity_repair_hooks(self):
        source = (REPO_ROOT / "threestudio" / "systems" / "Head3DGSLKs.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("opacity_coverage: dict", source)
        self.assertIn("rear_opacity: dict", source)
        self.assertIn("prune_region_guard: dict", source)
        self.assertIn("compute_opacity_coverage_loss", source)
        self.assertIn("compute_rear_opacity_loss", source)
        self.assertIn("train/loss_opacity_coverage", source)
        self.assertIn("train/loss_rear_opacity", source)

    def test_head_system_passes_region_min_opacity_to_prune(self):
        source = (REPO_ROOT / "threestudio" / "systems" / "Head3DGSLKs.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("build_region_min_opacity", source)
        self.assertIn("region_min_opacity=region_min_opacity", source)


if __name__ == "__main__":
    unittest.main()
