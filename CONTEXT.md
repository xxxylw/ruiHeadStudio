# RuiHeadStudio

RuiHeadStudio generates text-guided, animatable head avatars by optimizing FLAME-bound 3D Gaussians under multi-view image guidance and local geometry regularization.

## Language

**3D Avatar**:
A FLAME-rigged head representation optimized as 3D Gaussians so it can be rendered from multiple camera views and animated by expression or pose changes.
_Avoid_: 2D image, single-view portrait

**Guidance Backend**:
The frozen generative model stack that turns rendered RGB, text prompt, and control conditions into a training gradient for the 3D Avatar.
_Avoid_: ControlNet when referring to the whole guidance stack

**Control Condition**:
A view-aligned conditioning image derived from FLAME state and camera, such as pose or depth, used by the Guidance Backend to constrain generated structure.
_Avoid_: reference image, target image

**Pose Condition**:
A Control Condition built from projected FLAME joints or landmarks for the sampled camera view. It is passed independently from depth so its guidance strength and schedule can be tuned separately.
_Avoid_: merged pose-depth image

**Depth Condition**:
A Control Condition built from FLAME mesh depth for the sampled camera view. It is passed independently from pose so its guidance strength and schedule can be tuned separately.
_Avoid_: merged pose-depth image

**Multi-View Supervision**:
The training pattern where each step samples camera views across azimuth, elevation, and distance so the 3D Avatar is optimized for view consistency rather than a single 2D view.
_Avoid_: single-view optimization

**FLUX ControlNet Guidance**:
A Guidance Backend based on FLUX.1-dev and a FLUX ControlNet that consumes rendered RGB, text prompt, and one or more Control Conditions to produce an SDS-like gradient.
_Avoid_: SD1.5 ControlNet, classic DDPM guidance

**FLUX Guidance Scale**:
The scalar guidance input used by FLUX guidance embedding during single conditional prediction. The first FLUX ControlNet Guidance implementation uses this instead of true classifier-free guidance with separate conditional and unconditional predictions.
_Avoid_: true CFG when only one conditional prediction is run

**FLUX Smoke Test**:
A small-resolution probe run that verifies FLUX ControlNet Guidance can load, consume independent pose and depth Control Conditions, produce a nonzero gradient for a Rendered RGB Sample, and fit within available GPU memory before full 3D Avatar training.
_Avoid_: full training run

**SDXL Union ControlNet Guidance**:
A Guidance Backend based on an SDXL image prior and a union ControlNet that consumes the Rendered RGB Sample, text prompt, and one or more Control Conditions to produce an SDS-like gradient.
_Avoid_: FLUX guidance, two separate SD1.5 ControlNets

**Union Control Mode**:
The declared control-type selection used by a union ControlNet to interpret each supplied Control Condition, such as pose or depth.
_Avoid_: guessing control type from pixels, merging control conditions

**Rendered RGB Sample**:
The current image rendered from the 3D Avatar for a sampled camera view. In guidance, it is encoded to a latent as the optimization sample that receives gradients, not used as a reference image to preserve.
_Avoid_: reference image, init image

**SD1.5 ControlNet Baseline**:
The existing Guidance Backend based on Stable Diffusion 1.5 ControlNet with pose and depth control conditions.
_Avoid_: old ControlNet when the exact baseline matters

## Example Dialogue

Developer: Are we replacing the whole training pipeline with FLUX?

Domain expert: No. Keep Multi-View Supervision and the FLAME-bound 3D Avatar. Replace the Guidance Backend from the SD1.5 ControlNet Baseline to FLUX ControlNet Guidance.

Developer: Do pose and depth become target images?

Domain expert: No. They remain Control Conditions. They are generated from the current FLAME state and camera view, then passed alongside the rendered RGB and text prompt.

Developer: Should pose and depth be combined into one image for FLUX ControlNet?

Domain expert: No. Use independent Pose Condition and Depth Condition inputs so their conditioning scales and guidance schedules remain separately controllable.

Developer: Is the current rendered RGB a reference image for FLUX?

Domain expert: No. It is the Rendered RGB Sample: encode it to the latent being optimized and backpropagate the SDS-like gradient through it into the 3D Avatar.

Developer: Does first-pass FLUX guidance use true classifier-free guidance?

Domain expert: No. Use single conditional prediction with FLUX Guidance Scale first, then add true classifier-free guidance only as a later ablation once the SDS-like gradient path is working.

Developer: Should the first FLUX probe use native high FLUX resolution?

Domain expert: No. Start with a 512-pixel FLUX Smoke Test so memory and gradient correctness are validated before increasing resolution.
