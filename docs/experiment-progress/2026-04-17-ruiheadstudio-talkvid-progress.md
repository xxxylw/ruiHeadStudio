# 2026-04-17 Experiment Progress

## Scope

This progress note summarizes the work completed today around:

- RuiHeadStudio environment setup
- TalkSHOW parameter conversion and synthetic FLAME augmentation
- Multi-input FLAME distribution analysis
- TalkVid to FLAME tracking integration
- Minimal TalkVid clip download validation

## 1. RuiHeadStudio Environment

- Rebuilt the `ruiheadstudio` environment manually instead of relying on `conda env create -f environment.yml`.
- Installed the core training stack required by `ENVIRONMENT.md`, including:
  - `torch 2.0.1`
  - `torchvision`
  - `torchaudio`
  - `pytorch3d`
  - `xformers`
  - `envlight`
  - `nerfacc`
  - `nvdiffrast`
  - `tinycudann`
  - `diff-gaussian-rasterization`
- Verified that the project entrypoint and major imports work in the prepared environment.
- Confirmed that CUDA availability depends on whether commands are executed inside or outside the restricted sandbox.

## 2. TalkSHOW to RuiHeadStudio Parameter Conversion

- Located and used the existing TalkSHOW conversion workflow.
- Converted the available project `.pkl` assets into a RuiHeadStudio-compatible FLAME parameter archive:
  - `collection/ruiheadstudio/flame_collections/talkshow/project_converted_exp.npy`
- Verified the converted structure contains the expected training fields:
  - `expression`
  - `jaw_pose`
  - `leye_pose`
  - `reye_pose`
  - `neck_pose`

## 3. Synthetic FLAME Data Augmentation

- Added a synthetic augmentation generator:
  - `scripts/generate_augmented_pose_collections.py`
- Generated 10 mixed synthetic `.npy` files under:
  - `collection/ruiheadstudio/flame_collections/talkshow/synthetic_aug/`
- The generated files blend:
  - perturbation-style local variation
  - moderate outward expansion of expression, jaw, and neck motion
- Added tests for the augmentation path:
  - `tests/test_generate_augmented_pose_collections.py`

## 4. Distribution Analysis Upgrade

- Extended `scripts/analyze_pose_distribution.py` to support:
  - multiple `.npy` inputs
  - group labels such as `base` vs `aug`
  - `sampler-mode=train_like`
  - joint UMAP embedding across all inputs
  - HDBSCAN clustering over the shared embedding
- Added tests:
  - `tests/test_analyze_pose_distribution.py`
- Ran a combined comparison between:
  - `project_converted_exp.npy`
  - the 10 synthetic augmentation files
- Generated outputs in:
  - `collection/ruiheadstudio/analysis/pose_distribution/talkshow/combined_distribution_test/`

## 5. TalkVid Integration Path

- Added direct conversion support for FLAME tracker outputs to RuiHeadStudio format:
  - `scripts/convert_talkvid_to_ruiheadstudio.py`
- Wrote integration notes:
  - `docs/talkvid_flame_tracker_integration.md`
- Created the working directory structure for:
  - TalkVid clips
  - FLAME tracker outputs
  - converted RuiHeadStudio archives

## 6. External FLAME Tracker Setup

- Downloaded the `flame-head-tracker` source into:
  - `external/flame-head-tracker`
- Created and configured a separate `flame-tracker` conda environment.
- Downloaded the required public model assets and verified local model layout.
- Confirmed the missing FLAME assets were placed correctly:
  - `models/FLAME2020/generic_model.pkl`
  - `models/FLAME_albedo_from_BFM.npz`

## 7. FLAME Tracker Verification

- Ran a minimal single-image reconstruction test using the tracker and verified that it returns:
  - `shape`
  - `exp`
  - `jaw_pose`
  - `eye_pose`
- Ran a minimal 4-frame video test with `photometric_fitting=False` and confirmed that the tracker writes per-frame `.npz` files.
- Observed that the non-photometric path does not export `neck_pose`.
- Switched to `photometric_fitting=True` and identified a bug in the upstream tracker:
  - the photometric path referenced `gt_ear_landmarks` even when ear landmarks were disabled
- Fixed the issue in:
  - `external/flame-head-tracker/tracker_base.py`
- Re-ran the 4-frame photometric test and confirmed that the output `.npz` files now contain:
  - `exp`
  - `jaw_pose`
  - `neck_pose`
  - `eye_pose`
  - `shape`

## 8. TalkVid Metadata and Clip Download Validation

- Installed and configured the tooling required for TalkVid metadata and clip handling:
  - `huggingface-cli`
  - `yt-dlp`
  - `ffmpeg`
- Downloaded TalkVid metadata JSONs into:
  - `data/talkvid/meta/filtered_video_clips.json`
  - `data/talkvid/meta/filtered_video_clips_with_captions.json`
- Created a small filtered sample file:
  - `data/talkvid/meta/talkvid_input_small.json`
- Probed candidate YouTube links and found that some TalkVid entries are already unavailable.
- Created a vetted subset with currently reachable samples:
  - `data/talkvid/meta/talkvid_input_vetted.json`
- Validated the download pipeline on one live TalkVid sample by:
  - downloading the full source video with `yt-dlp`
  - trimming the target segment locally with `ffmpeg`
- Saved:
  - source video in `data/talkvid/source_videos/`
  - trimmed clip in `data/talkvid/clips/`

## 9. Current Status

The project is now in a state where:

- RuiHeadStudio has a working local environment
- FLAME parameter conversion and synthetic augmentation are available
- multi-input FLAME distribution analysis is implemented
- TalkVid metadata intake is working
- FLAME tracker photometric exports are verified to include `neck_pose`
- at least one real TalkVid clip has been downloaded and trimmed successfully

## 10. Next Recommended Step

The next practical step is:

1. Download the remaining vetted TalkVid clips.
2. Run `flame-head-tracker` on those clips with `photometric_fitting=True`.
3. Convert the tracker outputs with `scripts/convert_talkvid_to_ruiheadstudio.py`.
4. Compare the resulting TalkVid-derived FLAME distribution against the existing TalkSHOW and synthetic training pools.
