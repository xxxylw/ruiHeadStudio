from pathlib import Path

import numpy as np


FLASHAVATAR_FLAME_MESH_PATH = Path("third_party/FlashAvatar-code/flame/FlameMesh.obj")
FLAME_VERTEX_COUNT = 5023


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_obj_face_vertex_index(face_token: str) -> int:
    vertex_index = int(face_token.split("/", 1)[0])
    if vertex_index <= 0:
        raise ValueError(f"OBJ face indices must be positive: {face_token}")
    return vertex_index - 1


def load_obj_tri_faces(path: Path) -> np.ndarray:
    faces = []
    vertex_count = 0
    with path.open("r", encoding="utf-8") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                vertex_count += 1
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                if len(tokens) != 3:
                    raise ValueError(f"Only triangular OBJ faces are supported in {path}: {line.strip()}")
                faces.append([_parse_obj_face_vertex_index(token) for token in tokens])

    if vertex_count != FLAME_VERTEX_COUNT:
        raise ValueError(
            f"Expected {FLAME_VERTEX_COUNT} FLAME vertices in {path}, found {vertex_count}"
        )
    if not faces:
        raise ValueError(f"No triangular faces found in {path}")

    faces_array = np.asarray(faces, dtype=np.int32)
    if faces_array.min() < 0 or faces_array.max() >= vertex_count:
        raise ValueError(f"Face indices in {path} are outside the FLAME vertex range")
    return faces_array


def load_flashavatar_flame_faces() -> np.ndarray:
    return load_obj_tri_faces(_repo_root() / FLASHAVATAR_FLAME_MESH_PATH)
