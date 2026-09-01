from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.run_locked_specialist_final_reconcile_v3 import (
    DEFAULT_PREFLIGHT,
    DEFAULT_V2_REPLAY,
    DEFAULT_V2_RUN,
    EXPECTED_QUESTION_COUNT,
    EXPECTED_V2_PREFLIGHT_SHA256,
    EXPECTED_V2_REPLAY_SHA256,
    EXPECTED_V2_RUN_SHA256,
    REPLAY_NAME,
    RUN_NAME,
    _authority_composition_eligibility,
    _build_lane_audits,
    _compose_rows,
    _load_verified_sources,
    _materialization_projection,
    _numeric_prediction_matches,
    _question_row_is_self_hashed,
    _render_numeric_prediction,
    build_parser,
    run_materialize,
    run_replay,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


@pytest.fixture(scope="module")
def official_state():
    if not DEFAULT_PREFLIGHT.is_file():
        pytest.skip("sealed locked V2 answer artifacts are not present")
    bundle = _load_verified_sources(
        preflight_path=DEFAULT_PREFLIGHT,
        run_path=DEFAULT_V2_RUN,
        replay_path=DEFAULT_V2_REPLAY,
        expected_preflight_sha256=EXPECTED_V2_PREFLIGHT_SHA256,
        expected_run_sha256=EXPECTED_V2_RUN_SHA256,
        expected_replay_sha256=EXPECTED_V2_REPLAY_SHA256,
    )
    return bundle, _build_lane_audits(bundle)


def _authority(basis: str, count: int = 2, *, duplicate: bool = False):
    evidence = []
    for index in range(count):
        receipt = _SHA_A if duplicate else f"{index + 1:064x}"
        evidence.append({"contract_item_receipt_sha256s": [receipt]})
    return SimpleNamespace(basis=basis, proof={"parent_evidence": evidence})


def _numeric(*, value: float = 4.0, unit: str | None = None,
             boolean: bool | None = None):
    return SimpleNamespace(
        supported=True,
        boolean_result=boolean,
        numeric_result=value,
        unit=unit,
    )


def test_exact_v2_run_and_replay_are_the_only_answer_sources(official_state) -> None:
    bundle, _audits = official_state

    assert bundle.preflight.sha256 == EXPECTED_V2_PREFLIGHT_SHA256
    assert bundle.run.sha256 == bundle.replay.sha256 == EXPECTED_V2_RUN_SHA256
    assert bundle.run.payload == bundle.replay.payload
    assert len(bundle.rows) == len(bundle.plans) == EXPECTED_QUESTION_COUNT
    assert len(bundle.providers_by_ordinal) == 72
    assert bundle.preflight.payload["hard_complete_chat_token_cap"] == 8000
    assert bundle.preflight.payload["observed_max_complete_envelope_tokens"] <= 8000


def test_gold_blind_lane_audits_are_full_and_deterministic(official_state) -> None:
    _bundle, audits = official_state

    for audit in (audits.temporal, audits.numeric, audits.authority):
        assert len(audit.status_rows) == 72
        assert audit.status_population_sha256 == identity_sha256(
            list(audit.status_rows)
        )
        assert audit.resolved_population_sha256 == identity_sha256(
            list(audit.resolved_rows)
        )
        assert all(row["provider_calls"] == 0 for row in audit.status_rows)
        assert all(
            row["retained_transformer_token_state_bytes"] == 0
            for row in audit.status_rows
        )
        assert all(row["gold_loaded"] is False for row in audit.status_rows)

    assert [row["ordinal"] for row in audits.temporal.resolved_rows] == [15, 16, 17, 25]
    assert [row["ordinal"] for row in audits.numeric.resolved_rows] == [
        34, 44, 87, 90
    ]
    assert [row["ordinal"] for row in audits.authority.resolved_rows] == [
        12, 15, 44, 67, 76, 87, 90
    ]
    assert set(audits.authority.resolutions_by_ordinal) == {12, 15, 44, 76, 90}


@pytest.mark.parametrize(
    "basis",
    ["exact_current_total", "explicit_duration", "exact_declared_total"],
)
def test_complete_authority_bases_are_composition_safe(basis: str) -> None:
    eligible, reason = _authority_composition_eligibility(
        _authority(basis), dated_question="How many widgets are current?"
    )
    assert eligible is True
    assert reason == "complete_cross_plane_authority"


def test_bounded_authority_requires_a_real_composite() -> None:
    eligible, reason = _authority_composition_eligibility(
        _authority("bounded_cardinality_lower_bound", count=1),
        dated_question="How many classes are there?",
    )
    assert eligible is False
    assert reason == "bounded_cardinality_requires_composite_operands"

    eligible, reason = _authority_composition_eligibility(
        _authority("bounded_cardinality_lower_bound", count=3),
        dated_question="How many babies were born?",
    )
    assert eligible is True
    assert reason == "independent_composite_cardinality_lower_bound"


def test_bounded_authority_rejects_missing_identity_or_frequency_closure() -> None:
    distinct, distinct_reason = _authority_composition_eligibility(
        _authority("bounded_cardinality_lower_bound", count=3),
        dated_question="How many different museums did I visit?",
    )
    recurring, recurring_reason = _authority_composition_eligibility(
        _authority("bounded_cardinality_lower_bound", count=3),
        dated_question="How many classes do I attend per week?",
    )
    duplicate, duplicate_reason = _authority_composition_eligibility(
        _authority("bounded_cardinality_lower_bound", count=3, duplicate=True),
        dated_question="How many babies were born?",
    )

    assert distinct is recurring is duplicate is False
    assert distinct_reason == "distinct_cardinality_requires_identity_dedup_proof"
    assert recurring_reason == "recurring_cardinality_requires_frequency_closure"
    assert duplicate_reason == "bounded_cardinality_operands_not_independent"


def test_numeric_renderer_prefers_a_matching_candidate_then_parent() -> None:
    receipt = _numeric(value=4)

    assert _numeric_prediction_matches(
        "4", dated_question="How many bikes?", receipt=receipt
    )
    assert _render_numeric_prediction(
        candidate_prediction="4",
        parent_prediction="9",
        dated_question="How many bikes?",
        receipt=receipt,
    ) == ("4", "candidate")
    assert _render_numeric_prediction(
        candidate_prediction="2",
        parent_prediction="4",
        dated_question="How many bikes?",
        receipt=receipt,
    ) == ("4", "parent")


def test_numeric_renderer_only_computes_when_neither_sealed_answer_matches() -> None:
    assert _render_numeric_prediction(
        candidate_prediction="2",
        parent_prediction="3",
        dated_question="How many bikes?",
        receipt=_numeric(value=4),
    ) == ("4", "computed")
    assert _render_numeric_prediction(
        candidate_prediction="No.",
        parent_prediction="No.",
        dated_question="Is the left total greater?",
        receipt=_numeric(value=10, boolean=True),
    ) == ("Yes.", "computed")
    assert _render_numeric_prediction(
        candidate_prediction="2 classes per week.",
        parent_prediction="3 classes per week.",
        dated_question="How many fitness classes do I attend in a typical week?",
        receipt=_numeric(value=5, unit="fitness_class/week"),
    ) == ("5 fitness classes per week", "computed")


def test_composition_is_temporal_then_numeric_then_authority_then_v2(
    official_state,
) -> None:
    bundle, audits = official_state
    rows = _compose_rows(bundle, audits)
    by_ordinal = {row["ordinal"]: row for row in rows}

    assert {o for o, row in by_ordinal.items() if row["decision_lane"] == "question_bound_temporal"} == {15, 16, 17, 25}
    assert {o for o, row in by_ordinal.items() if row["decision_lane"] == "sealed_numeric"} == {34, 44, 87, 90}
    assert {o for o, row in by_ordinal.items() if row["decision_lane"] == "cross_plane_parent_protection"} == {12, 76}
    assert sum(row["decision_lane"] == "v2_fallback" for row in rows) == 90
    assert all(_question_row_is_self_hashed(row) for row in rows)
    assert all(row["physical_provider_calls"] == 0 for row in rows)
    assert all(row["retained_transformer_token_state_bytes"] == 0 for row in rows)


def test_materialization_has_100_judge_rows_and_exact_source_bindings(
    official_state,
) -> None:
    bundle, audits = official_state
    payload = _materialization_projection(
        bundle,
        audits,
        expected_temporal_status_population_sha256=(
            audits.temporal.status_population_sha256
        ),
        expected_numeric_status_population_sha256=(
            audits.numeric.status_population_sha256
        ),
        expected_authority_status_population_sha256=(
            audits.authority.status_population_sha256
        ),
    )

    assert payload["question_count"] == len(payload["questions"]) == 100
    assert len(payload["judge_rows"]) == 100
    assert payload["gold_loaded"] is False
    assert payload["physical_provider_calls_during_materialization"] == 0
    assert payload["retained_transformer_token_state_bytes"] == 0
    assert payload["hard_complete_chat_token_cap"] == 8000
    assert payload["observed_max_complete_envelope_tokens"] <= 8000
    assert payload["v2_preflight_artifact_sha256"] == bundle.preflight.sha256
    assert payload["v2_run_artifact_sha256"] == bundle.run.sha256
    assert payload["v2_replay_artifact_sha256"] == bundle.replay.sha256
    assert all(_question_row_is_self_hashed(row) for row in payload["questions"])
    assert [row["source_row_sha256"] for row in payload["judge_rows"]] == [
        row["source_row_sha256"] for row in payload["questions"]
    ]


def test_materialize_and_replay_are_byte_identical(
    official_state, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    bundle, audits = official_state
    import tools.run_locked_specialist_final_reconcile_v3 as lifecycle

    monkeypatch.setattr(lifecycle, "_load_from_args", lambda _args: bundle)
    monkeypatch.setattr(lifecycle, "_build_lane_audits", lambda _bundle: audits)
    args = Namespace(
        output_root=tmp_path,
        expected_temporal_status_population_sha256=(
            audits.temporal.status_population_sha256
        ),
        expected_numeric_status_population_sha256=(
            audits.numeric.status_population_sha256
        ),
        expected_authority_status_population_sha256=(
            audits.authority.status_population_sha256
        ),
    )
    created = run_materialize(args)
    args.expected_v3_run_sha256 = created["run_sha256"]
    replayed = run_replay(args)

    assert replayed["byte_identical"] is True
    assert replayed["run_sha256"] == replayed["replay_sha256"]
    assert read_sealed_json(tmp_path / RUN_NAME).payload == read_sealed_json(
        tmp_path / REPLAY_NAME
    ).payload


def test_cli_cannot_materialize_without_all_frozen_lane_receipts() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["materialize"])

    parsed = parser.parse_args(
        [
            "materialize",
            "--expected-temporal-status-population-sha256", _SHA_A,
            "--expected-numeric-status-population-sha256", _SHA_B,
            "--expected-authority-status-population-sha256", _SHA_C,
        ]
    )
    assert parsed.expected_temporal_status_population_sha256 == _SHA_A


def test_valid_but_wrong_frozen_population_sha_is_rejected(official_state) -> None:
    bundle, audits = official_state

    with pytest.raises(
        ValueError, match="temporal full-72 status population differs"
    ):
        _materialization_projection(
            bundle,
            audits,
            expected_temporal_status_population_sha256=_SHA_A,
            expected_numeric_status_population_sha256=(
                audits.numeric.status_population_sha256
            ),
            expected_authority_status_population_sha256=(
                audits.authority.status_population_sha256
            ),
        )
