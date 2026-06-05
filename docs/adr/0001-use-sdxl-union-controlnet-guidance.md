# Use SDXL Union ControlNet Guidance

We will use SDXL Union ControlNet Guidance as this branch's replacement Guidance Backend: keep the training shell, FLAME-derived Pose Condition and Depth Condition construction, Multi-View Supervision, and 3DGS optimization flow unchanged, while replacing the SD1.5 Multi-ControlNet runtime path with an SDXL base model plus `xinsir/controlnet-union-sdxl-1.0`.

This branch deliberately deletes the old SD1.5 guidance implementation instead of preserving a runtime switch, because the experiment is a focused replacement probe and SDXL Union needs a different prompt embedding contract, Control Mode contract, scheduler handling, and SDS forward path. The first implementation will use pose+depth only, 512-pixel smoke-test resolution, a dedicated SDXL prompt processor, standard two-way CFG, Euler scheduling, and constant SDS weighting before exploring higher resolution, ProMax, extra Control Conditions, or alternative SDS weighting.
