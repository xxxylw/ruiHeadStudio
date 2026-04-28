from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


ALLOWED_IDENTITY_MODES = {"fictional", "target_person"}
ALLOWED_VIEWS = {"front", "left_3q", "right_3q", "left_side", "right_side"}


@dataclass(frozen=True)
class ReferenceImage:
    image_path: Path
    view: str
    weight: float
    face_crop: Tuple[int, int, int, int]
    person_crop: Tuple[int, int, int, int]


@dataclass(frozen=True)
class ReferenceSheet:
    root: Path
    identity_mode: str
    prompt: str
    references: Tuple[ReferenceImage, ...]


def _as_box(value: Iterable[int], field_name: str) -> Tuple[int, int, int, int]:
    box = tuple(int(v) for v in value)
    if len(box) != 4:
        raise ValueError(f"{field_name} must contain four integers")
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{field_name} must be ordered as [x0, y0, x1, y1]")
    return box


def load_reference_sheet(metadata_path: str | Path) -> ReferenceSheet:
    metadata_path = Path(metadata_path)
    root = metadata_path.parent
    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    identity_mode = data.get("identity_mode", "fictional")
    if identity_mode not in ALLOWED_IDENTITY_MODES:
        raise ValueError(f"identity_mode must be one of {sorted(ALLOWED_IDENTITY_MODES)}")

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt must be a non-empty string")

    references = []
    for index, item in enumerate(data.get("references", [])):
        image_path = root / item["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"reference image missing at index {index}: {image_path}")
        view = item.get("view", "front")
        if view not in ALLOWED_VIEWS:
            raise ValueError(f"unsupported view at index {index}: {view}")
        face_crop = _as_box(item["face_crop"], "face_crop")
        references.append(
            ReferenceImage(
                image_path=image_path,
                view=view,
                weight=float(item.get("weight", 1.0)),
                face_crop=face_crop,
                person_crop=_as_box(item.get("person_crop", face_crop), "person_crop"),
            )
        )

    if not references:
        raise ValueError("references must contain at least one image")

    return ReferenceSheet(
        root=root,
        identity_mode=identity_mode,
        prompt=prompt,
        references=tuple(references),
    )
