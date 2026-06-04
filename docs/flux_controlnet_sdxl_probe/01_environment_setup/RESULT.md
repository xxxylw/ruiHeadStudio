# 01 Environment Setup Result

Status: passed on host/GPU.

## Environment

- Conda env: `ruiheadstudio-flux-controlnet`
- Created by cloning: `ruiheadstudio-bnbfix`
- Validation report: `outputs/flux_controlnet_sdxl_probe/01_environment_setup/validate_after_diffusers034_host.json`

## Final Package Set

- `torch==2.1.2+cu118`
- `torchvision==0.16.2+cu118`
- `torchaudio==2.1.2+cu118`
- `pytorch3d==0.7.7=py39_cu118_pyt212`
- `diffusers==0.34.0`
- `transformers==4.46.3`
- `accelerate==1.1.1`
- `huggingface-hub==0.36.0`
- `xformers==0.0.23.post1`

## Validation

The validation script imports the existing RuiHeadStudio 3DGS runtime dependencies plus FLUX ControlNet diffusers symbols:

- `diffusers.FluxControlNetPipeline`: ok
- `diffusers.FluxControlNetModel`: ok
- `pytorch3d`: ok
- `pytorch3d.renderer`: ok
- `tinycudann`: ok
- `diff_gaussian_rasterization`: ok
- `simple_knn._C`: ok
- `nvdiffrast.torch`: ok
- `nerfacc`: ok

## Notes

- `diffusers==0.35.1` was rejected because it requires `torch.library.custom_op`, which is absent in `torch==2.1.2`.
- `diffusers==0.34.0` keeps FLUX ControlNet symbols available while avoiding that torch 2.4+ API.
- `pytorch3d` had to be reinstalled from the `py39_cu118_pyt212` tarball after upgrading torch; the previous `pyt201` build imported shallowly but failed in `pytorch3d.renderer`.
- `xformers==0.0.23.post1` imports, but warns that its CUDA extensions are unavailable because the wheel was built for CUDA 12.1 while this environment uses CUDA 11.8. This is a performance/attention-kernel risk for later training, not a Slice 01 import blocker.

## Next Dependency

Slice 02 can now attempt ordinary FLUX ControlNet 2D generation with fixed prompt plus independent FLAME pose/depth conditions.
