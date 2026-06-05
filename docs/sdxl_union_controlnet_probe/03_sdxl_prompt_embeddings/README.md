# 03 SDXL Prompt Embeddings

Goal: validate the dedicated SDXL prompt processor contract before it is consumed by guidance.

The prompt processor should produce positive and negative sequence embeddings plus positive and negative pooled embeddings, including view-dependent variants. The SDXL Union guidance should fail fast if it receives the old SD1.5 `PromptProcessorOutput`.

The matching result directory is `outputs/sdxl_union_controlnet_probe/03_sdxl_prompt_embeddings/`.
