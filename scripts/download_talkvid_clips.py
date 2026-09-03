import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select, vet, and download TalkVid clips directly from YouTube "
            "using yt-dlp's Android client."
        )
    )
    parser.add_argument(
        "--metadata",
        default="data/talkvid/meta/filtered_video_clips.json",
        help="Path to TalkVid metadata JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="data/talkvid/meta/talkvid_input_vetted_20.json",
        help="Where to save the vetted subset JSON.",
    )
    parser.add_argument(
        "--clips-dir",
        default="data/talkvid/clips",
        help="Directory where downloaded clips will be stored.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="Number of successfully downloaded clips to keep.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["English", "Chinese"],
        help="Allowed language labels from TalkVid metadata.",
    )
    parser.add_argument(
        "--min-dover",
        type=float,
        default=8.8,
        help="Minimum TalkVid dover score.",
    )
    parser.add_argument(
        "--min-cotracker",
        type=float,
        default=0.90,
        help="Minimum co-tracker ratio.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=1280,
        help="Minimum source width.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=400,
        help="Maximum metadata entries to probe before stopping.",
    )
    parser.add_argument(
        "--yt-dlp-bin",
        default="conda run -n ruiheadstudio yt-dlp",
        help="yt-dlp command prefix. Supports shell-style quoting.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only vet entries, do not download clips.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_key(item: Dict) -> tuple:
    return (
        float(item.get("dover_scores", 0.0)),
        float(item.get("cotracker_ratio", 0.0)),
        int(item.get("width", 0)),
        float(item.get("end-time", 0.0)) - float(item.get("start-time", 0.0)),
    )


def select_candidates(items: Iterable[Dict], args: argparse.Namespace) -> List[Dict]:
    selected: List[Dict] = []
    seen_links = set()
    allowed_languages = set(args.languages)

    sorted_items = sorted(items, key=score_key, reverse=True)
    for item in sorted_items:
        info = item.get("info", {})
        link = info.get("Video Link")
        if not link or link in seen_links:
            continue
        if info.get("Language") not in allowed_languages:
            continue
        if float(item.get("dover_scores", 0.0)) < args.min_dover:
            continue
        if float(item.get("cotracker_ratio", 0.0)) < args.min_cotracker:
            continue
        if int(item.get("width", 0)) < args.min_width:
            continue
        selected.append(item)
        seen_links.add(link)
        if len(selected) >= args.max_candidates:
            break
    return selected


def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def yt_dlp_prefix(spec: str) -> List[str]:
    return shlex.split(spec)


def probe_candidate(item: Dict, yt_prefix: Sequence[str]) -> bool:
    url = item["info"]["Video Link"]
    cmd = list(yt_prefix) + ["--skip-download", "--no-playlist", "--print", "id", url]
    result = run_command(cmd)
    return result.returncode == 0


def clip_output_path(clips_dir: Path, clip_id: str) -> Path:
    return clips_dir / clip_id / f"{clip_id}.mp4"


def metadata_output_path(clips_dir: Path, clip_id: str) -> Path:
    return clips_dir / clip_id / "metadata.json"


def download_clip(item: Dict, clips_dir: Path, yt_prefix: Sequence[str]) -> bool:
    clip_id = item["id"]
    output_path = clip_output_path(clips_dir, clip_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    url = item["info"]["Video Link"]
    start = float(item["start-time"])
    end = float(item["end-time"])
    out_tmpl = str(output_path.parent / "%(id)s.%(ext)s")

    cmd = list(yt_prefix) + [
        "--no-playlist",
        "--extractor-args",
        "youtube:player_client=android",
        "-f",
        "18/b",
        "--download-sections",
        f"*{start:.3f}-{end:.3f}",
        "-o",
        out_tmpl,
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return False

    # yt-dlp expands %(id)s with the raw YouTube id, not the TalkVid clip id.
    # Normalize any downloaded mp4 to the clip_id naming convention expected by
    # the downstream tracker and conversion scripts.
    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    candidates = sorted(
        p for p in output_path.parent.glob("*.mp4") if p.name != output_path.name and p.stat().st_size > 0
    )
    if not candidates:
        return False

    candidates[0].rename(output_path)
    return output_path.exists() and output_path.stat().st_size > 0


def write_clip_metadata(item: Dict, clips_dir: Path) -> None:
    clip_id = item["id"]
    metadata_path = metadata_output_path(clips_dir, clip_id)
    payload = {
        "clip_id": clip_id,
        "video_name": item["info"].get("Video Link", "").split("v=")[-1],
        "clip_name": clip_id,
        "source_file": clip_output_path(clips_dir, clip_id).name,
        "source_path": item["info"].get("Video Link"),
        "start_time": item.get("start-time"),
        "end_time": item.get("end-time"),
        "dover_scores": item.get("dover_scores"),
        "cotracker_ratio": item.get("cotracker_ratio"),
        "info": item.get("info", {}),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_vetted(items: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata = load_metadata(Path(args.metadata))
    candidates = select_candidates(metadata, args)
    yt_prefix = yt_dlp_prefix(args.yt_dlp_bin)
    clips_dir = Path(args.clips_dir)

    vetted: List[Dict] = []
    attempts = 0

    for item in candidates:
        if len(vetted) >= args.target_count:
            break
        attempts += 1

        clip_id = item["id"]
        print(f"[probe] {clip_id}")
        if not probe_candidate(item, yt_prefix):
            print(f"[skip] {clip_id} probe failed")
            continue

        if args.probe_only:
            vetted.append(item)
            print(f"[keep] {clip_id} probe ok")
            continue

        print(f"[download] {clip_id}")
        if not download_clip(item, clips_dir, yt_prefix):
            print(f"[skip] {clip_id} download failed")
            continue

        write_clip_metadata(item, clips_dir)
        vetted.append(item)
        print(f"[keep] {clip_id} downloaded ({len(vetted)}/{args.target_count})")

    save_vetted(vetted, Path(args.output_json))
    print(f"saved {len(vetted)} vetted clips to {args.output_json}")
    print(f"probed {attempts} candidates")

    if len(vetted) < args.target_count:
        print(
            f"warning: target_count={args.target_count} not reached; "
            f"only {len(vetted)} clips were vetted/downloaded."
        )


if __name__ == "__main__":
    main()
