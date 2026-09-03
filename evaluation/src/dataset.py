from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


FINAL_STEP = int(os.environ.get("HEADSTUDIO_FINAL_STEP", "10000"))
VIEW_INDICES = (0, 1, 2, 3)


class DatasetValidationError(ValueError):
    """Raised when a batch cannot form the fixed prompt/image evaluation set."""


@dataclass(frozen=True)
class Example:
    run_name: str
    prompt: str
    image_path: Path
    view_index: int


def load_examples(batch_root: Path) -> list[Example]:
    manifest_path = batch_root / "manifest.tsv"
    if not manifest_path.is_file():
        raise DatasetValidationError(f"manifest not found: {manifest_path}")

    examples: list[Example] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        columns = line.split("\t")
        if line_number == 1 and columns[:3] == ["index", "tag", "status"]:
            continue
        if len(columns) < 8:
            raise DatasetValidationError(
                f"manifest line {line_number} has fewer than 8 tab-separated columns"
            )
        run_name, status, prompt = columns[1], columns[2], columns[7]
        if status != "ok":
            raise DatasetValidationError(
                f"manifest line {line_number} has non-success status: {status}"
            )
        if not prompt:
            raise DatasetValidationError(f"manifest line {line_number} has an empty prompt")

        save_dir = batch_root / "runs" / run_name / "save"
        expected = [save_dir / f"it{FINAL_STEP}-{index}.png" for index in VIEW_INDICES]
        present = [path for path in expected if path.is_file()]
        if len(present) != len(VIEW_INDICES):
            raise DatasetValidationError(
                f"run {run_name}: expected 4 final views at step {FINAL_STEP}, "
                f"found {len(present)}"
            )
        examples.extend(
            Example(run_name, prompt, image_path, view_index)
            for view_index, image_path in zip(VIEW_INDICES, expected)
        )

    if not examples:
        raise DatasetValidationError("manifest contains no evaluable runs")
    return examples
