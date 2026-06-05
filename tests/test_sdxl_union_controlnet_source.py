from pathlib import Path
import importlib.util
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_sdxl_union_guidance_declares_named_control_modes():
    spec = importlib.util.spec_from_file_location(
        "controlnet_union_sdxl_contract",
        ROOT / "threestudio/models/guidance/controlnet_union_sdxl_contract.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.CONTROL_MODE_IDS["openpose"] == 0
    assert module.CONTROL_MODE_IDS["depth"] == 1
    assert module.resolve_control_modes(["openpose", "depth"]) == [0, 1]

    with pytest.raises(ValueError, match="Unsupported Union Control Mode"):
        module.resolve_control_modes(["openpose", "gray"])


def test_headstudio_config_uses_sdxl_union_guidance_contract():
    config = (ROOT / "configs/headstudio.yaml").read_text()

    assert 'guidance_type: "controlnet-union-sdxl-guidance"' in config
    assert 'prompt_processor_type: "stable-diffusion-xl-prompt-processor"' in config
    assert 'pretrained_model_name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"' in config
    assert 'vae_model_name_or_path: "madebyollin/sdxl-vae-fp16-fix"' in config
    assert 'controlnet_model_name_or_path: "xinsir/controlnet-union-sdxl-1.0"' in config
    assert 'control_modes: ["openpose", "depth"]' in config
    assert "guidance_resolution: 512" in config


def test_old_sd15_guidance_runtime_is_removed_from_package_imports():
    guidance_init = (ROOT / "threestudio/models/guidance/__init__.py").read_text()

    assert "controlnet_union_sdxl_guidance" in guidance_init
    assert "controlnet_guidance" not in guidance_init


def test_sdxl_union_guidance_sds_path_uses_union_and_sdxl_contracts():
    source = (
        ROOT / "threestudio/models/guidance/controlnet_union_sdxl_guidance.py"
    ).read_text()

    assert "def compute_grad_sds(" in source
    assert "def sample_timesteps(" in source
    assert "self.scheduler.timesteps" in source
    assert "self.scheduler.add_noise" in source
    assert "self.scheduler.scale_model_input" in source
    assert '"text_embeds":' in source
    assert '"time_ids":' in source
    assert "control_type=" in source
    assert "control_type_idx=self.control_mode_ids" in source
    assert "controlnet_cond=image_cond" in source
    assert "conditioning_scale=self.conditioning_scale" in source
    assert "noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)" in source
    assert "self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)" in source
    assert "grad = noise_pred - noise" in source
    assert "torch.randint(\n            self.min_step_index" in source


def test_sdxl_union_guidance_supports_local_cache_and_cpu_offload_loading():
    source = (
        ROOT / "threestudio/models/guidance/controlnet_union_sdxl_guidance.py"
    ).read_text()
    smoke_script = (ROOT / "scripts/run_sdxl_union_thor_smoke.sh").read_text()

    assert "local_files_only: bool = False" in source
    assert "local_files_only=self.cfg.local_files_only" in source
    assert "if self.cfg.enable_model_cpu_offload:" in source
    assert "self.pipe.enable_model_cpu_offload()" in source
    assert "else:" in source
    assert "self.pipe = self.pipe.to(self.device)" in source
    assert "system.guidance.local_files_only=True" in smoke_script
    assert "system.guidance.enable_model_cpu_offload=True" in smoke_script


def test_training_step_logs_sdxl_union_sds_loss():
    system_source = (ROOT / "threestudio/systems/Head3DGSLKs.py").read_text()

    assert 'self.log("train/loss_sds", guidance_out["loss_sds"])' in system_source
