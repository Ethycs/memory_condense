import numpy as np
import pytest

from tools.run_attention_tfidf_artifact_campaign import (
    _aggregate,
    _array_receipt,
    _atomic_npy,
    _empty_counts,
    _feature_input_fingerprint,
    _parse_histories,
    _validated_array,
)


def test_history_selection_is_sorted_unique_and_bounded():
    assert _parse_histories("3,1-2,2", available=4) == (1, 2, 3)
    with pytest.raises(ValueError, match="ends before"):
        _parse_histories("3-1")
    with pytest.raises(ValueError, match="lie in"):
        _parse_histories("0,1")


def test_feature_fingerprint_binds_ordered_model_inputs_only():
    encoder = {"checkpoint": "fixed", "layer": 4}
    first = _feature_input_fingerprint(
        (("s1", "summary one"), ("s2", "summary two")),
        (("q1", "question one"),),
        encoder,
    )
    assert first == _feature_input_fingerprint(
        (("s1", "summary one"), ("s2", "summary two")),
        (("q1", "question one"),),
        encoder,
    )
    assert first != _feature_input_fingerprint(
        (("s2", "summary two"), ("s1", "summary one")),
        (("q1", "question one"),),
        encoder,
    )
    assert first != _feature_input_fingerprint(
        (("s1", "summary changed"), ("s2", "summary two")),
        (("q1", "question one"),),
        encoder,
    )
    assert first != _feature_input_fingerprint(
        (("s1", "summary one"), ("s2", "summary two")),
        (("q1", "question changed"),),
        encoder,
    )


def test_array_receipt_reuses_exact_file_and_rejects_tampering(tmp_path):
    path = tmp_path / "features.npy"
    value = np.arange(12, dtype=np.float16).reshape(3, 4)
    _atomic_npy(path, value)
    receipt = _array_receipt(path)
    cached = _validated_array(path, receipt)
    assert cached.dtype == np.float16
    assert np.array_equal(cached, value)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        _validated_array(path, receipt)


def test_aggregate_sums_counts_before_computing_rates():
    one = _empty_counts()
    one.update(
        questions=2,
        support_total=4,
        baseline_supports=2,
        candidate_supports=3,
        union_supports=4,
        novel_supports_over_baseline=2,
        baseline_complete_questions=1,
        candidate_complete_questions=1,
        union_complete_questions=2,
        newly_complete_questions=1,
    )
    two = dict(one)
    result = _aggregate(
        (
            {"arms": {"tfidf": {"8": one}}},
            {"arms": {"tfidf": {"8": two}}},
        )
    )["tfidf"]["8"]
    assert result["questions"] == 4
    assert result["support_total"] == 8
    assert result["candidate_support_recall"] == pytest.approx(0.75)
    assert result["union_complete_rate"] == pytest.approx(1.0)
