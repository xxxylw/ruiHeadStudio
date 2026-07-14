from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def batch_root(tmp_path: Path) -> Path:
    root = tmp_path / "batch"
    root.mkdir()
    (root / "manifest.tsv").write_text(
        "1\t01_alien\tok\tstart\tend\t0\t/run/01_alien\ta head of an alien\t\n",
        encoding="utf-8",
    )
    save_dir = root / "runs" / "01_alien" / "save"
    save_dir.mkdir(parents=True)
    for view_index in range(4):
        Image.new("RGB", (4, 4), color=(view_index, 0, 0)).save(
            save_dir / f"it10000-{view_index}.png"
        )
    return root
