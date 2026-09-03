import pytest


def test_summary_returns_mean_sample_std_and_confidence_interval():
    from evaluation.src.statistics import summarize

    result = summarize([0.2, 0.4, 0.6, 0.8])

    assert result["count"] == 4
    assert result["mean"] == pytest.approx(0.5)
    assert result["std"] == pytest.approx(0.2581988897)
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]


def test_retrieval_uses_one_based_ranks():
    from evaluation.src.statistics import summarize_retrieval

    result = summarize_retrieval([1, 2, 4])

    assert result["count"] == 3
    assert result["recall_at_1"] == pytest.approx(1 / 3)
    assert result["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 3)
    assert result["mean_rank"] == pytest.approx(7 / 3)


def test_rank_for_target_is_one_based_and_tie_optimistic():
    from evaluation.src.clip_metrics import rank_for_target

    assert rank_for_target([0.9, 0.2, 0.5], 0) == 1
    assert rank_for_target([0.9, 0.2, 0.5], 2) == 2
    assert rank_for_target([0.9, 0.9, 0.5], 1) == 1


def test_single_model_metric_names_select_the_expected_checkpoint():
    from evaluation.run_evaluation import selected_clip_models

    assert selected_clip_models("clip_b32") == ("ViT-B/32",)
    assert selected_clip_models("clip_b16") == ("ViT-B/16",)
    assert selected_clip_models("clip_l14") == ("ViT-L/14",)
    assert selected_clip_models("all") == ("ViT-L/14", "ViT-B/16", "ViT-B/32")


def test_piqe_only_mode_does_not_select_a_clip_checkpoint():
    from evaluation.run_evaluation import selected_clip_models

    assert selected_clip_models("piqe") == ()


def test_musiq_only_mode_does_not_select_a_clip_checkpoint():
    from evaluation.run_evaluation import selected_clip_models

    assert selected_clip_models("musiq") == ()
