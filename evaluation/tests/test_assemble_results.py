import json


def test_assemble_writes_complete_artifacts(tmp_path):
    from evaluation.assemble_results import assemble
    from evaluation.src.clip_metrics import CLIP_MODELS

    clip_rows = [
        {
            "model": model,
            "run_name": "sample",
            "prompt": "a sample",
            "image_path": "sample.png",
            "view_index": 0,
            "score": 0.2,
            "rank": 1,
        }
        for model in CLIP_MODELS
    ]
    quality_rows = [
        {
            "model": "PIQE",
            "run_name": "sample",
            "prompt": "a sample",
            "image_path": "sample.png",
            "view_index": 0,
            "score": 50.0,
        }
    ]

    summary = assemble(clip_rows, quality_rows, tmp_path / "batch", tmp_path / "out", "cpu")

    assert set(summary["clip"]) == set(CLIP_MODELS)
    assert summary["quality"]["PIQE"]["score"]["mean"] == 50.0
    assert (tmp_path / "out" / "per_image_metrics.csv").exists()
    assert (tmp_path / "out" / "clip_retrieval.csv").exists()
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["image_count"] == 1
