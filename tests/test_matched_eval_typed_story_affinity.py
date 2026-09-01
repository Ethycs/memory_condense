import hashlib
import json

from tools.matched_eval.typed_story_affinity import (
    derive_evidence_story_affinity,
    evidence_history_story_key_sha256,
    evidence_source_story_key_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_selected_sources_in_one_history_share_only_the_history_key() -> None:
    left = derive_evidence_story_affinity(
        _sha("namespace"), "d23cf73b::answer_5a0d28f8_1"
    )
    right = derive_evidence_story_affinity(
        _sha("namespace"), "d23cf73b::answer_5a0d28f8_4"
    )
    contaminant = derive_evidence_story_affinity(
        _sha("namespace"), "f685340e_abs::a268827b_3"
    )

    assert left.source_story_key_sha256 != right.source_story_key_sha256
    assert left.history_story_key_sha256 == right.history_story_key_sha256
    assert contaminant.history_story_key_sha256 != left.history_story_key_sha256
    assert set(left.story_keys) & set(right.story_keys) == {
        left.history_story_key_sha256
    }
    assert not (set(left.story_keys) & set(contaminant.story_keys))


def test_raw_locators_remain_out_of_the_opaque_projection() -> None:
    affinity = derive_evidence_story_affinity(
        _sha("namespace"), "question-prefix::answer-source"
    )
    encoded = json.dumps(affinity.opaque_projection(), sort_keys=True)

    assert "question-prefix" not in encoded
    assert "answer-source" not in encoded
    assert "source_id" not in affinity.opaque_projection()
    assert "namespace_id" not in affinity.opaque_projection()
    assert affinity.provider_visible_raw_locator_count == 0
    assert affinity.retained_transformer_token_state_bytes == 0


def test_unpartitioned_source_does_not_invent_a_broader_component() -> None:
    affinity = derive_evidence_story_affinity(
        _sha("namespace"), "standalone-source"
    )

    assert affinity.story_keys == (affinity.source_story_key_sha256,)
    assert affinity.history_key_distinct_from_source is False
    assert affinity.receipt_sha256 == derive_evidence_story_affinity(
        _sha("namespace"), "standalone-source"
    ).receipt_sha256


def test_exported_source_and_history_helpers_are_the_derivation_domain() -> None:
    namespace = _sha("namespace")
    affinity = derive_evidence_story_affinity(
        namespace, "history-a::source-3"
    )

    assert affinity.source_story_key_sha256 == evidence_source_story_key_sha256(
        namespace, "history-a::source-3"
    )
    assert affinity.history_story_key_sha256 == (
        evidence_history_story_key_sha256(namespace, "history-a")
    )
