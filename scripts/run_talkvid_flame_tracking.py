import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ClipJob:
    clip_id: str
    video_path: Path
    output_dir: Path
    source_metadata: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flame-head-tracker on a vetted TalkVid clip manifest and store per-frame .npz outputs."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a TalkVid manifest JSON, e.g. data/talkvid/meta/talkvid_input_vetted_20.json",
    )
    parser.add_argument(
        "--clips-root",
        default="data/talkvid/clips",
        help="Root directory containing per-clip mp4 files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/talkvid/flame_tracker",
        help="Root directory where tracker output folders are written.",
    )
    parser.add_argument(
        "--tracker-root",
        default="external/flame-head-tracker",
        help="Path to the local flame-head-tracker checkout.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--photometric-fitting", action="store_true", default=False)
    parser.add_argument("--realign", action="store_true", default=False)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for how many manifest entries to process. 0 means all.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips whose tracker output directory already contains frame .npz files.",
    )
    return parser.parse_args()


def load_clip_jobs(manifest_path: Path, clips_root: Path, output_root: Path) -> List[ClipJob]:
    raw_items = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    jobs: List[ClipJob] = []
    for item in raw_items:
        clip_id = item["id"]
        clip_dir = clips_root / clip_id
        video_path = clip_dir / f"{clip_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Clip video missing: {video_path}")

        metadata_path = clip_dir / "metadata.json"
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        jobs.append(
            ClipJob(
                clip_id=clip_id,
                video_path=video_path,
                output_dir=output_root / clip_id,
                source_metadata=source_metadata,
            )
        )
    return jobs


def write_tracking_metadata(job: ClipJob) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "video_name": job.source_metadata.get("video_name", job.clip_id),
        "clip_name": job.source_metadata.get("clip_name", job.clip_id),
        "source_file": job.source_metadata.get("source_file", job.video_path.name),
        "source_path": job.source_metadata.get("source_path", str(job.video_path)),
    }
    (job.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def infer_video_fps(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Unable to infer fps for video: {video_path}")
    return fps


def build_tracker_cfg(job: ClipJob, tracker_root: Path, device: str, batch_size: int, photometric_fitting: bool, realign: bool) -> Dict[str, object]:
    fps = infer_video_fps(job.video_path)
    return {
        "video_path": str(job.video_path),
        "original_fps": fps,
        "subsample_fps": fps,
        "save_path": str(job.output_dir.parent),
        "photometric_fitting": photometric_fitting,
        "realign": realign,
        "batch_size": batch_size,
        "mediapipe_face_landmarker_v2_path": str(tracker_root / "models" / "face_landmarker.task"),
        "flame_model_path": str(tracker_root / "models" / "FLAME2020" / "generic_model.pkl"),
        "flame_lmk_embedding_path": str(tracker_root / "models" / "landmark_embedding.npy"),
        "tex_space_path": str(tracker_root / "models" / "FLAME_albedo_from_BFM.npz"),
        "face_parsing_model_path": str(tracker_root / "models" / "79999_iter.pth"),
        "template_mesh_file_path": str(tracker_root / "models" / "head_template.obj"),
        "result_img_size": 512,
        "optimize_fov": False,
        "device": device,
    }


def should_skip(job: ClipJob) -> bool:
    return any(job.output_dir.glob("*.npz"))


def prepare_tracker_import(tracker_root: Path) -> None:
    tracker_root = Path(tracker_root).resolve()
    if str(tracker_root) not in sys.path:
        sys.path.insert(0, str(tracker_root))


def run_jobs(args: argparse.Namespace) -> None:
    tracker_root = Path(args.tracker_root).resolve()
    clips_root = Path(args.clips_root).resolve()
    output_root = Path(args.output_root).resolve()
    jobs = load_clip_jobs(Path(args.manifest), clips_root, output_root)
    if args.limit > 0:
        jobs = jobs[: args.limit]

    os.chdir(tracker_root)
    prepare_tracker_import(tracker_root)

    import torch
    from tracker_video import track_video

    torch.cuda.set_device(args.device)

    for index, job in enumerate(jobs, start=1):
        if args.skip_existing and should_skip(job):
            print(f"[skip] {job.clip_id} ({index}/{len(jobs)})")
            write_tracking_metadata(job)
            continue

        print(f"[track] {job.clip_id} ({index}/{len(jobs)})")
        write_tracking_metadata(job)
        cfg = build_tracker_cfg(
            job=job,
            tracker_root=tracker_root,
            device=args.device,
            batch_size=args.batch_size,
            photometric_fitting=args.photometric_fitting,
            realign=args.realign,
        )
        track_video(cfg)


def main() -> None:
    args = parse_args()
    run_jobs(args)


if __name__ == "__main__":
    main()
