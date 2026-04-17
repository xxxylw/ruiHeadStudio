# Synthetic Pose Collections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 10 medium-extent augmented RuiHeadStudio training `.npy` pose collections from the converted TalkSHOW mother collection.

**Architecture:** Add one standalone generation script under `scripts/` that loads the converted collection, applies deterministic sequence-level augmentation with moderate expansion in expression/jaw/neck dynamics, and writes 10 output `.npy` files. Add a small `unittest` test file to lock the output contract and basic augmentation behavior.

**Tech Stack:** Python 3.9, NumPy, standard library `unittest`, RuiHeadStudio pose-collection object-array format

---

### Task 1: Add contract tests for synthetic collection generation

**Files:**
- Create: `tests/test_generate_augmented_pose_collections.py`
- Test: `tests/test_generate_augmented_pose_collections.py`

- [ ] **Step 1: Write the failing test**

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_augmented_pose_collections import (
    POSE_KEYS,
    generate_augmented_collection,
    save_collections,
)


class SyntheticPoseCollectionTests(unittest.TestCase):
    def _make_sequence(self, frames: int = 24):
        t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
        return {
            "expression": np.stack([t for _ in range(100)], axis=1).astype(np.float32),
            "jaw_pose": np.stack([t, t * 0.5, t * 0.25], axis=1).astype(np.float32),
            "leye_pose": np.stack([t * 0.2, t * 0.1, t * 0.05], axis=1).astype(np.float32),
            "reye_pose": np.stack([t * 0.15, t * 0.05, t * 0.08], axis=1).astype(np.float32),
            "neck_pose": np.stack([t * 0.3, t * 0.2, t * 0.1], axis=1).astype(np.float32),
            "video_name": "src_video",
            "clip_name": "src_clip",
            "source_file": "src.pkl",
            "source_path": "/tmp/src.pkl",
        }

    def test_generate_augmented_collection_preserves_contract(self):
        source = [self._make_sequence()]
        augmented = generate_augmented_collection(source, file_index=0, rng_seed=7)

        self.assertEqual(len(augmented), 1)
        item = augmented[0]
        for key in POSE_KEYS:
            self.assertEqual(item[key].dtype, np.float32)
            self.assertEqual(item[key].shape[0], source[0][key].shape[0])
        self.assertIn("synthetic_medium", item["video_name"])
        self.assertNotEqual(
            float(np.mean(item["expression"])),
            float(np.mean(source[0]["expression"])),
        )

    def test_save_collections_writes_expected_number_of_files(self):
        source = [self._make_sequence()]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = save_collections(
                source,
                Path(tmpdir),
                num_files=3,
                seed=11,
            )

            self.assertEqual(len(output_paths), 3)
            for path in output_paths:
                self.assertTrue(path.exists())
                arr = np.load(path, allow_pickle=True)
                self.assertEqual(arr.dtype, object)
                self.assertEqual(arr.shape, (1,))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n ruiheadstudio python -m unittest tests.test_generate_augmented_pose_collections -v`
Expected: FAIL with `ModuleNotFoundError` for `scripts.generate_augmented_pose_collections`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/generate_augmented_pose_collections.py` with:
- `POSE_KEYS`
- `generate_augmented_collection(...)`
- `save_collections(...)`
- a CLI entry point that reads `talkshow/collection/project_converted_exp.npy`
- deterministic medium-expansion augmentation for `expression`, `jaw_pose`, `leye_pose`, `reye_pose`, `neck_pose`

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n ruiheadstudio python -m unittest tests.test_generate_augmented_pose_collections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_generate_augmented_pose_collections.py scripts/generate_augmented_pose_collections.py docs/superpowers/plans/2026-04-17-generate-synthetic-pose-collections.md
git commit -m "feat: generate synthetic pose collections"
```

### Task 2: Generate the 10 augmented `.npy` collections

**Files:**
- Modify: `scripts/generate_augmented_pose_collections.py`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_01.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_02.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_03.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_04.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_05.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_06.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_07.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_08.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_09.npy`
- Create: `talkshow/collection/synthetic_aug/aug_pose_mix_10.npy`

- [ ] **Step 1: Run the generation script**

Run:
`conda run -n ruiheadstudio python scripts/generate_augmented_pose_collections.py --input talkshow/collection/project_converted_exp.npy --output-dir talkshow/collection/synthetic_aug --num-files 10 --seed 20260417`

Expected: prints 10 saved output paths

- [ ] **Step 2: Verify structure of generated files**

Run:
`conda run -n ruiheadstudio python -c "import numpy as np; from pathlib import Path; paths=sorted(Path('talkshow/collection/synthetic_aug').glob('*.npy')); print(len(paths)); arr=np.load(paths[0], allow_pickle=True); print(arr.dtype, arr.shape, sorted(arr.tolist()[0].keys()))"`

Expected:
- `10`
- `object`
- `(3,)`
- keys include `expression`, `jaw_pose`, `leye_pose`, `reye_pose`, `neck_pose`

- [ ] **Step 3: Verify metadata marks synthetic origin**

Run:
`conda run -n ruiheadstudio python -c "import numpy as np; arr=np.load('talkshow/collection/synthetic_aug/aug_pose_mix_01.npy', allow_pickle=True).tolist(); print(arr[0]['video_name'], arr[0]['clip_name'])"`

Expected: names include `synthetic_medium`

- [ ] **Step 4: Commit**

```bash
git add talkshow/collection/synthetic_aug/*.npy
git commit -m "data: add synthetic medium-extent pose collections"
```
