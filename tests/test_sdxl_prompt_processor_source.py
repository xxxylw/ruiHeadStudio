from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sdxl_prompt_processor_declares_required_output_contract():
    source_path = ROOT / "threestudio/models/prompt_processors/stable_diffusion_xl_prompt_processor.py"
    source = source_path.read_text()
    prompt_init = (ROOT / "threestudio/models/prompt_processors/__init__.py").read_text()

    assert "SDXLPromptProcessorOutput" in source
    assert "def get_sdxl_text_embeddings(" in source
    for field in [
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
        "prompt_embeds_vd",
        "negative_prompt_embeds_vd",
        "pooled_prompt_embeds_vd",
        "negative_pooled_prompt_embeds_vd",
    ]:
        assert field in source

    assert '@threestudio.register("stable-diffusion-xl-prompt-processor")' in source
    assert "CLIPTextModelWithProjection" in source
    assert "text_encoder_2" in source
    assert "stable_diffusion_xl_prompt_processor" in prompt_init
