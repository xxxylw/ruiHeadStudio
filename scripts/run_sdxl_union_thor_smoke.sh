#!/bin/bash
set -euo pipefail

CUDA_DEVICE="${1:-0}"
THOR_PROMPT="a DSLR portrait of Thor in Marvel, masterpiece, Studio Quality, 8k, ultra-HD, next generation"
THOR_NEGATIVE="sculpture, statue, shadow, dark face, eyeglass, glasses, noise,pattern, strange color, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions,long neck"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python3 launch.py \
  --config configs/headstudio.yaml --train \
  system.guidance_type="controlnet-union-sdxl-guidance" \
  system.prompt_processor_type="stable-diffusion-xl-prompt-processor" \
  system.prompt_processor.prompt="${THOR_PROMPT}" \
  system.prompt_processor.negative_prompt="${THOR_NEGATIVE}" \
  system.guidance.guidance_resolution=512 \
  system.guidance.local_files_only=True \
  system.guidance.enable_model_cpu_offload=True \
  data.batch_size=1 \
  trainer.max_steps=3 \
  system.max_grad=0.001 \
  system.area_relax=True
