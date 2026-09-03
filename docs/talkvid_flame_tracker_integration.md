# TalkVid To RuiHeadStudio Via FLAME Head Tracker

This project treats `flame-head-tracker` as an external preprocessing step. Each TalkVid clip is tracked independently, then converted into RuiHeadStudio's training `.npy` format.

## Recommended Layout

Keep the tracker in its own environment and store its outputs inside this repo with a stable directory layout:

```text
ruiHeadStudio/
├── external/
│   └── flame-head-tracker/          # separate clone + separate conda env
├── data/
│   └── talkvid/
│       ├── clips/
│       │   ├── clip_0001.mp4
│       │   └── clip_0002.mp4
│       └── flame_tracker/
│           ├── clip_0001/
│           │   ├── metadata.json
│           │   ├── 000000.npz
│           │   ├── 000001.npz
│           │   └── ...
│           └── clip_0002/
│               ├── metadata.json
│               ├── 000000.npz
│               └── ...
└── collection/
    └── ruiheadstudio/
        └── flame_collections/
            └── talkvid/
                ├── talkvid_all.npy
                └── per_clip/
                    ├── clip_0001.npy
                    └── clip_0002.npy
```

`metadata.json` is optional but recommended. The converter reads:

```json
{
  "video_name": "speaker_001",
  "clip_name": "clip_0001",
  "source_video": "/abs/path/to/data/talkvid/clips/clip_0001.mp4",
  "source_file": "clip_0001"
}
```

If `metadata.json` is missing, the converter falls back to directory names.

## Expected Tracker Frame Format

Each frame `.npz` must contain the FLAME parameters produced by `flame-head-tracker`:

- `exp`: `(1, 100)` or `(100,)`
- `jaw_pose`: `(1, 3)` or `(3,)`
- `neck_pose`: `(1, 3)` or `(3,)`
- `eye_pose`: `(1, 6)` or `(6,)`

The converter maps them to RuiHeadStudio as:

- `expression <- exp`
- `jaw_pose <- jaw_pose`
- `neck_pose <- neck_pose`
- `leye_pose <- eye_pose[:3]`
- `reye_pose <- eye_pose[3:]`

## Conversion Command

Convert one tracker clip directory:

```bash
conda run -n ruiheadstudio python scripts/convert_talkvid_to_ruiheadstudio.py \
  data/talkvid/flame_tracker/clip_0001 \
  --output collection/ruiheadstudio/flame_collections/talkvid/talkvid_all.npy
```

Convert all clip directories under a parent folder:

```bash
conda run -n ruiheadstudio python scripts/convert_talkvid_to_ruiheadstudio.py \
  data/talkvid/flame_tracker \
  --output collection/ruiheadstudio/flame_collections/talkvid/talkvid_all.npy
```

Append new clips to an existing collection:

```bash
conda run -n ruiheadstudio python scripts/convert_talkvid_to_ruiheadstudio.py \
  data/talkvid/flame_tracker/new_batch \
  --output collection/ruiheadstudio/flame_collections/talkvid/talkvid_all.npy \
  --append
```

## Running The External Tracker

The tracker itself should stay isolated in its own environment because it depends on Python 3.10, CUDA 11.7, and additional FLAME/DECA/MICA assets.

Reference:
- `flame-head-tracker`: https://github.com/peizhiyan/flame-head-tracker

Suggested workflow:

1. Put a TalkVid clip in `data/talkvid/clips/clip_0001.mp4`
2. Run `flame-head-tracker` and save frame results into `data/talkvid/flame_tracker/clip_0001/`
3. Add `metadata.json` beside the frame `.npz` files
4. Run `scripts/convert_talkvid_to_ruiheadstudio.py`
5. Point `configs/headstudio.yaml` to the converted `.npy`
