"""Lightweight SDXL Union ControlNet contracts shared by probes and guidance."""

from __future__ import annotations


CONTROL_MODE_IDS = {
    "openpose": 0,
    "depth": 1,
    "softedge": 2,
    "canny": 3,
    "lineart": 3,
    "normal": 4,
    "segment": 5,
}


def resolve_control_modes(control_modes: list[str]) -> list[int]:
    resolved: list[int] = []
    for mode in control_modes:
        if mode not in CONTROL_MODE_IDS:
            supported = ", ".join(sorted(CONTROL_MODE_IDS))
            raise ValueError(
                f"Unsupported Union Control Mode {mode!r}. Supported modes: {supported}"
            )
        resolved.append(CONTROL_MODE_IDS[mode])
    return resolved
