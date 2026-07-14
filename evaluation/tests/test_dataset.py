import pytest


def test_load_examples_pairs_one_prompt_to_four_final_views(batch_root):
    from evaluation.src.dataset import load_examples

    examples = load_examples(batch_root)

    assert len(examples) == 4
    assert {item.view_index for item in examples} == {0, 1, 2, 3}
    assert {item.run_name for item in examples} == {"01_alien"}
    assert {item.prompt for item in examples} == {"a head of an alien"}


def test_load_examples_rejects_a_missing_final_view(batch_root):
    from evaluation.src.dataset import DatasetValidationError, load_examples

    (batch_root / "runs" / "01_alien" / "save" / "it10000-3.png").unlink()

    with pytest.raises(DatasetValidationError, match="expected 4 final views"):
        load_examples(batch_root)
