def test_render_markdown_names_every_main_metric():
    from evaluation.src.report import render_markdown

    text = render_markdown(
        {
            "clip": {"ViT-B/32": {"clip_score": {"mean": 0.3}}},
            "quality": {"PIQE": {"score": {"mean": 65.0}}},
        }
    )

    assert "CLIP Score" in text
    assert "ViT-B/32" in text
    assert "PIQE" in text
