import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.native_spine_summary import body_identity, fragment_body
from tools.native_spine_namespace import materialize_namespace
from tests.test_native_hierarchy_occurrence import fixture


def inputs():
    template, occurrence = fixture()
    source, history = occurrence("2026-09-12")
    turns = list(history.turns.values())
    body = {"turns": [{"role": t.role, "text": t.text} for t in turns]}
    # Reconstruct the original exact source-pointer descriptors, not new cuts.
    ordinal = {t.turn_id: i for i, t in enumerate(turns)}
    summaries = [{"pointer": {
        "body_sha256": body_identity(body), "turn_ordinal": ordinal[a.spans[0].turn_id],
        "role": a.spans[0].role, "start_char": a.spans[0].start_char, "end_char": a.spans[0].end_char,
        "turn_text_sha256": a.spans[0].turn_text_sha256, "span_text_sha256": a.spans[0].span_text_sha256,
        "token_count": a.spans[0].token_count}, "summary": a.summary} for a in history.atoms]
    return template, source, body, summaries


def test_complete_namespace_rebinds_exact_coverage_but_small_fixture_cannot_pass_1m_admission():
    template, source, body, summaries = inputs()
    namespace = materialize_namespace([source], body_ids={source["body_sha256"]},
        templates={source["body_sha256"]: template}, load_body=lambda _: body,
        load_summaries=lambda _: summaries, compiler_identity="fixture")
    assert namespace.audit["complete_namespace"]
    assert namespace.audit["admitted_occurrences"] == namespace.audit["hierarchy_occurrences"] == 1
    assert namespace.audit["body_tokens"] > 0
    with pytest.raises(ValueError, match="token scale"):
        namespace.require_complete()
    assert namespace.require_complete(minimum_body_tokens=1) == namespace


def test_pending_tree_does_not_discard_admitted_atomic_addresses_and_partial_is_explicit():
    _, source, body, summaries = inputs()
    kwargs = dict(body_ids={source["body_sha256"]}, templates={}, load_body=lambda _: body,
                  load_summaries=lambda _: summaries, compiler_identity="fixture")
    with pytest.raises(ValueError, match="missing summaries or hierarchies"):
        materialize_namespace([source], **kwargs)
    namespace = materialize_namespace([source], allow_partial=True, **kwargs)
    assert len(namespace.atomic_index.sections) == len(summaries)
    assert not namespace.hierarchy.sections
    assert namespace.audit["missing_hierarchy_occurrence_ids"] == [source["occurrence_id"]]
    with pytest.raises(ValueError):
        namespace.require_complete(minimum_body_tokens=1)


def test_missing_body_is_reported_and_never_loaded_or_hidden_by_another_namespace():
    template, source, body, summaries = inputs()
    missing = dict(source, body_sha256="uncompiled-body", session_id="other")
    missing["occurrence_id"] = identity_sha256({k: v for k, v in missing.items() if k != "occurrence_id"})
    calls = []
    def load(sha):
        calls.append(sha)
        assert sha == source["body_sha256"]
        return body
    namespace = materialize_namespace([source, missing], body_ids={source["body_sha256"]},
        templates={source["body_sha256"]: template}, load_body=load,
        load_summaries=lambda _: summaries, compiler_identity="fixture", allow_partial=True)
    assert calls == [source["body_sha256"]]
    assert namespace.history.occurrence_ids == (source["occurrence_id"],)
    assert namespace.audit["missing_summary_occurrence_ids"] == [missing["occurrence_id"]]
    assert not namespace.audit["complete_namespace"]
