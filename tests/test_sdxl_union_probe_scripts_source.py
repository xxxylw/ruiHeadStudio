from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


THOR_PROMPT = "a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation"


def test_sdxl_union_probe_scripts_exist_and_use_expected_models():
    env_script = (
        ROOT
        / "scripts/sdxl_union_controlnet_probe/01_environment_validation/validate_sdxl_union_env.py"
    ).read_text()
    cache_script = (
        ROOT
        / "scripts/sdxl_union_controlnet_probe/01_environment_validation/check_sdxl_union_cache.py"
    ).read_text()
    generation_script = (
        ROOT
        / "scripts/sdxl_union_controlnet_probe/02_sdxl_union_2d_generation/probe_sdxl_union_2d_generation.py"
    ).read_text()

    for source in [env_script, cache_script, generation_script]:
        assert "xinsir/controlnet-union-sdxl-1.0" in source
        assert "stabilityai/stable-diffusion-xl-base-1.0" in source
        assert "madebyollin/sdxl-vae-fp16-fix" in source
        assert "ruiheadstudio-sdxl-union-controlnet" in source

    assert "ControlNetUnionModel" in env_script
    assert "StableDiffusionXLControlNetUnionPipeline" in env_script
    assert "EulerAncestralDiscreteScheduler" in env_script
    assert "AutoencoderKL" in env_script
    assert "snapshot_download" in cache_script
    assert "local_files_only" in cache_script
    assert "missing" in cache_script

    assert "resolve_control_modes" in generation_script
    assert "control_image=[pose_pil, depth_pil]" in generation_script
    assert "control_mode=control_mode_ids" in generation_script
    assert "flame_pose.png" in generation_script
    assert "flame_depth.png" in generation_script
    assert "sdxl_union_pose_depth.png" in generation_script


def test_sdxl_union_thor_smoke_script_uses_thor_prompt_and_smoke_limits():
    smoke_script = (ROOT / "scripts/run_sdxl_union_thor_smoke.sh").read_text()

    assert THOR_PROMPT in smoke_script
    assert "trainer.max_steps=3" in smoke_script
    assert "data.batch_size=1" in smoke_script
    assert "system.guidance.guidance_resolution=512" in smoke_script
    assert 'system.guidance_type="controlnet-union-sdxl-guidance"' in smoke_script
    assert 'system.prompt_processor_type="stable-diffusion-xl-prompt-processor"' in smoke_script
