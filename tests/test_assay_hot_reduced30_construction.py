from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tools import assay_hot_reduced30_construction as assay


def _locked_minimal_rows() -> list[dict[str, object]]:
    return [
        {
            "ordinal": ordinal,
            "prompt_question_sha256": question_sha,
            "question_id": question_id,
        }
        for ordinal, question_id, question_sha in assay.LOCKED_QUESTIONS
    ]


def test_locked_population_is_exact_ordered_and_unique() -> None:
    assert assay.QUESTION_COUNT == 30
    assert assay.LOCKED_QUESTIONS[0][1:] == (
        "06878be2",
        "488d098964e1c68638411404eb2126164877d443ba18a64dd995820663777319",
    )
    assert next(row for row in assay.LOCKED_QUESTIONS if row[0] == 14)[1] == "d23cf73b"
    assert next(row for row in assay.LOCKED_QUESTIONS if row[0] == 83)[1] == "c14c00dd"
    assert [row[0] for row in assay.LOCKED_QUESTIONS] == [
        5,
        6,
        14,
        15,
        17,
        25,
        36,
        40,
        42,
        43,
        48,
        49,
        51,
        52,
        53,
        59,
        61,
        66,
        67,
        69,
        75,
        77,
        79,
        81,
        82,
        83,
        86,
        87,
        94,
        97,
    ]
    assert len({row[1] for row in assay.LOCKED_QUESTIONS}) == 30
    assert len({row[2] for row in assay.LOCKED_QUESTIONS}) == 30
    assert len(assay.LOCKED_IDENTITY_SHA256) == 64


def test_locked_rows_are_selected_by_identity_not_input_order() -> None:
    source = list(reversed(_locked_minimal_rows()))
    selected = assay._locked_rows(source)  # noqa: SLF001
    assert [ordinal for ordinal, _row in selected] == [
        value[0] for value in assay.LOCKED_QUESTIONS
    ]
    assert [row["question_id"] for _ordinal, row in selected] == [
        value[1] for value in assay.LOCKED_QUESTIONS
    ]


def test_locked_rows_fail_closed_when_a_question_hash_changes() -> None:
    source = _locked_minimal_rows()
    source[7]["prompt_question_sha256"] = "0" * 64
    with pytest.raises(assay.Reduced30ConstructionError, match="hash changed"):
        assay._locked_rows(source)  # noqa: SLF001


def test_arm_discovery_prefers_diagnostic_arm_without_format_assumption() -> None:
    parent = {
        "provider_messages": [{"role": "user", "content": "parent"}],
        "context_token_proxy": 10,
    }
    diagnostic = {
        "provider_messages": [{"role": "user", "content": "diagnostic"}],
        "context_token_proxy": 12,
        "episode_selection": {},
        "fact_ledger": {},
        "rendered_raw_chunk_ids": [],
    }
    path, selected = assay.find_provider_arm(
        {"new_format": {"fallback": parent, "packet": diagnostic}}
    )
    assert path == "new_format.packet"
    assert selected["provider_messages"] == diagnostic["provider_messages"]
    explicit_path, explicit = assay.find_provider_arm(
        {"new_format": {"fallback": parent, "packet": diagnostic}},
        "new_format.fallback",
    )
    assert explicit_path == "new_format.fallback"
    assert explicit["provider_messages"] == parent["provider_messages"]


def test_wrapper_successor_resolves_nested_source_and_shadow_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_calls: list[object] = []

    def load_source(root: object) -> tuple[dict[str, object], str, str, str]:
        source_calls.append(root)
        return {}, "construction", "runtime", "replay"

    cache = SimpleNamespace(
        physical_store_row_count=17,
        projection=lambda: {"format": "cache"},
        source_database_sha256="d" * 64,
        source_store_receipt_sha256="s" * 64,
    )
    shadow_calls: list[tuple[object, object]] = []

    def build_cache(context: object, namespace: object):
        shadow_calls.append((context, namespace))
        return cache, {"cache_build_ns": 123, "stable_count": 17}

    index = SimpleNamespace(projection=lambda: {"format": "index"})
    monkeypatch.setattr(
        assay,
        "build_user_led_envelope_shadow_index",
        lambda observed: index if observed is cache else None,
    )
    successor = SimpleNamespace(
        legacy=SimpleNamespace(
            shadow=SimpleNamespace(_build_cache=build_cache),
            source_assay=SimpleNamespace(_load_source=load_source),
        )
    )

    assert assay._source_loader(successor) is load_source  # noqa: SLF001
    assert assay._source_loader(successor)("sealed-root")[1:] == (  # noqa: SLF001
        "construction",
        "runtime",
        "replay",
    )
    context = object()
    namespace = SimpleNamespace(namespace_id="namespace-1")
    observed_index, binding = assay._candidate_universe(  # noqa: SLF001
        successor, context, namespace
    )
    assert source_calls == ["sealed-root"]
    assert shadow_calls == [(context, namespace)]
    assert observed_index is index
    assert binding["cache_build_binding"] == {"stable_count": 17}
    assert binding["namespace_id"] == "namespace-1"


def test_direct_component_precedes_legacy_and_invalid_contract_fails() -> None:
    direct_loader = lambda _root: (None, None, None, None)
    nested_loader = lambda _root: (None, None, None, None)
    module = SimpleNamespace(
        source_assay=SimpleNamespace(_load_source=direct_loader),
        legacy=SimpleNamespace(
            source_assay=SimpleNamespace(_load_source=nested_loader)
        ),
    )
    assert assay._source_loader(module) is direct_loader  # noqa: SLF001

    invalid = SimpleNamespace(
        legacy=SimpleNamespace(source_assay=SimpleNamespace(_load_source=None))
    )
    with pytest.raises(
        assay.Reduced30ConstructionError,
        match="module.legacy.source_assay has no callable _load_source",
    ):
        assay._source_loader(invalid)  # noqa: SLF001


def test_aggregate_reports_slots_lanes_facts_and_context() -> None:
    report = {
        "compiled_fact_count": 7,
        "context_token_proxy": 9000,
        "lane_decision_counts": {
            "typed_operation": {"rejected_global_episode_raw_budget": 2, "selected": 1}
        },
        "lane_proposal_count": 12,
        "lane_rejection_count": 2,
        "lane_used_chunks": {"typed_operation": 2},
        "lane_used_tokens": {"typed_operation": 300},
        "prompt_workspace_token_proxy": 9900,
        "rendered_episode_raw_chunk_count": 2,
        "rendered_fact_count": 1,
        "required_slot_coverage": [
            {
                "compiled_fact_match": True,
                "rendered_episode_match": True,
                "rendered_fact_match": False,
            }
        ],
    }
    aggregate = assay._aggregate([report, copy.deepcopy(report)])  # noqa: SLF001
    assert aggregate["compiled_fact_count"] == 14
    assert aggregate["fact_rendered_count"] == 2
    assert aggregate["context_token_max"] == 9000
    assert aggregate["lane_rejection_count"] == 4
    assert aggregate["lane_decision_counts"]["typed_operation"] == {
        "rejected_global_episode_raw_budget": 4,
        "selected": 2,
    }
    assert aggregate["required_slot_totals"] == {
        "compiled_fact_match": 2,
        "rendered_episode_match": 2,
        "rendered_fact_match": 0,
        "required": 2,
    }
    assert aggregate["questions_with_required_slots"] == 2
    assert aggregate["questions_without_required_slots"] == 0
    assert aggregate["questions_all_rendered_fact_slots_matched"] == 0
