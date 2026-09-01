from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_em_fact_memory import (
    EMFact,
    EMFactCitation,
    EMFactCompression,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastEvidence,
)
from tests.test_matched_eval_source_history_fact_union import (
    _NAMESPACE_ID,
    _hydrate_all,
    _parent,
    _selection,
    _write_store,
)
from tools.matched_eval.post_selection_em_fact_lane import (
    PostSelectionEMFactLaneError,
    bind_post_selection_em_neighborhood,
    map_post_selection_em_facts,
    parse_sealed_em_fact_compression,
)
from tools.matched_eval.source_history_fact_union import (
    FactLane,
    build_post_map_fact_union,
    plan_source_history_hydration,
)


def _sha(value: str) -> str:
    return quote_sha256(value)


def _one_chunk_plan(tmp_path: Path):
    text = "On 2026-08-02, Alpha selected the blue token."
    path = tmp_path / "memory.db"
    memberships = _write_store(path, {"history-a": [text]})
    history = _hydrate_all(path, memberships)["history-a"]
    plan = plan_source_history_hydration(
        _parent(),
        selections=(_selection("em-a", FactLane.EM, "history-a"),),
        histories=(history,),
    )
    return plan, text


def _neighborhood(plan, text: str, *, duplicate: bool = False):
    evidence = (
        FastEvidence(_sha("evidence-1"), "history-a", text),
        *(
            (FastEvidence(_sha("evidence-2"), "history-a", text),)
            if duplicate
            else ()
        ),
    )
    return bind_post_selection_em_neighborhood(
        plan,
        question_id="question-0",
        question_sha256=_sha("question"),
        dated_question_sha256=_sha("dated-question"),
        source_stage_id="selected-em-stage",
        upstream_selection_receipt_sha256=_sha("em-stage-selection"),
        evidence=evidence,
        selection_ids=("em-a",) * len(evidence),
    )


def _compression(neighborhood, *, duplicate_facts: bool = False):
    quote = "blue token"
    citations = tuple(
        EMFactCitation(
            evidence_alias=row.evidence_alias,
            evidence_id=row.evidence_id,
            source_id=row.source_id,
            quote=quote,
            quote_sha256=quote_sha256(quote),
        )
        for row in neighborhood.evidence
    )
    facts = (
        EMFact("F1", "Alpha selected a blue token.", (citations[0],)),
        *(
            (EMFact("F2", "The selected token was blue.", (citations[1],)),)
            if duplicate_facts
            else ()
        ),
    )
    return EMFactCompression(
        question_id=neighborhood.question_id,
        source_stage_id=neighborhood.source_stage_id,
        neighborhood_evidence_ids=neighborhood.evidence_ids,
        facts=facts,
        response_sha256=_sha("sealed-completion"),
    )


def test_selected_em_facts_keep_duplicates_until_post_map_union(
    tmp_path: Path,
) -> None:
    plan, text = _one_chunk_plan(tmp_path)
    neighborhood = _neighborhood(plan, text, duplicate=True)
    compression = _compression(neighborhood, duplicate_facts=True)

    lane = map_post_selection_em_facts(plan, neighborhood, compression)

    assert lane.provider_calls == 0
    assert lane.retained_transformer_token_state_bytes == 0
    assert lane.source_fact_count == 2
    assert lane.accepted_before_dedup_count == 2
    assert len(lane.batches) == 1
    assert not lane.batches[0].rejected
    accepted = lane.batches[0].accepted
    assert tuple(row.lane for row in accepted) == (FactLane.EM, FactLane.EM)
    assert tuple(row.source_role for row in accepted) == ("user", "user")
    assert all(row.source_created_at.startswith("2026-08-01") for row in accepted)
    assert accepted[0].chunk_id == accepted[1].chunk_id

    union = build_post_map_fact_union(plan, batches=lane.batches)
    assert union.accepted_before_dedup_count == 2
    assert len(union.union_facts_before_direct_exclusion) == 1
    assert union.retained_facts[0].owner_lane is FactLane.EM
    assert union.retained_facts[0].fact_variants == (
        "Alpha selected a blue token.",
        "The selected token was blue.",
    )
    assert neighborhood.evidence_ids == (
        _sha("evidence-1"),
        _sha("evidence-2"),
    )


def test_empty_valid_compression_completes_only_selected_em_window(
    tmp_path: Path,
) -> None:
    plan, text = _one_chunk_plan(tmp_path)
    neighborhood = _neighborhood(plan, text)
    compression = EMFactCompression(
        question_id=neighborhood.question_id,
        source_stage_id=neighborhood.source_stage_id,
        neighborhood_evidence_ids=neighborhood.evidence_ids,
        facts=(),
        response_sha256=_sha("empty-completion"),
    )

    lane = map_post_selection_em_facts(plan, neighborhood, compression)
    union = build_post_map_fact_union(plan, batches=lane.batches)

    assert lane.source_fact_count == lane.accepted_before_dedup_count == 0
    assert len(lane.batches) == 1
    assert lane.batches[0].source_item_count == 0
    assert union.completed_window_ids == neighborhood.completed_window_ids
    assert union.pending_window_ids == ()


def test_sealed_compression_round_trips_then_binds_exact_quotes(
    tmp_path: Path,
) -> None:
    plan, text = _one_chunk_plan(tmp_path)
    neighborhood = _neighborhood(plan, text)
    compression = _compression(neighborhood)

    restored = parse_sealed_em_fact_compression(
        compression.identity_payload()
    )
    lane = map_post_selection_em_facts(plan, neighborhood, restored)

    assert restored == compression
    assert lane.batches[0].accepted[0].quote == "blue token"
    assert lane.batches[0].accepted[0].quote_start_char == text.index(
        "blue token"
    )
    tampered = compression.identity_payload()
    tampered["facts"][0]["citations"][0]["quote"] = "green token"
    with pytest.raises(ValueError, match="digest"):
        parse_sealed_em_fact_compression(tampered)


def test_selection_binding_rejects_foreign_lane_and_partial_window(
    tmp_path: Path,
) -> None:
    texts = (
        "Alpha selected the blue token.",
        "Alpha stored the token in the north cabinet.",
    )
    path = tmp_path / "memory.db"
    memberships = _write_store(path, {"history-a": list(texts)})
    history = _hydrate_all(path, memberships)["history-a"]
    em_plan = plan_source_history_hydration(
        _parent(),
        selections=(_selection("em-a", FactLane.EM, "history-a"),),
        histories=(history,),
        max_window_tokens=sum(
            row.token_count for row in history.chunks if not row.metadata_chunk
        ),
    )
    with pytest.raises(PostSelectionEMFactLaneError, match="partially covers"):
        _neighborhood(em_plan, texts[0])

    partition_plan = plan_source_history_hydration(
        _parent(),
        selections=(
            _selection("em-a", FactLane.PARTITION, "history-a"),
        ),
        histories=(history,),
    )
    with pytest.raises(PostSelectionEMFactLaneError, match="exact EM source"):
        _neighborhood(partition_plan, texts[0])


def test_compression_cannot_change_selected_order_or_source(
    tmp_path: Path,
) -> None:
    plan, text = _one_chunk_plan(tmp_path)
    neighborhood = _neighborhood(plan, text)
    compression = _compression(neighborhood)

    with pytest.raises(PostSelectionEMFactLaneError, match="evidence order"):
        map_post_selection_em_facts(
            plan,
            neighborhood,
            replace(
                compression,
                neighborhood_evidence_ids=(_sha("other-evidence"),),
                receipt_sha256="",
            ),
        )

    bad_citation = EMFactCitation(
        evidence_alias="E001",
        evidence_id=neighborhood.evidence_ids[0],
        source_id="history-b",
        quote="blue token",
        quote_sha256=quote_sha256("blue token"),
    )
    changed = EMFactCompression(
        question_id=neighborhood.question_id,
        source_stage_id=neighborhood.source_stage_id,
        neighborhood_evidence_ids=neighborhood.evidence_ids,
        facts=(EMFact("F1", "The token was blue.", (bad_citation,)),),
        response_sha256=_sha("bad-source-completion"),
    )
    with pytest.raises(PostSelectionEMFactLaneError, match="exact evidence"):
        map_post_selection_em_facts(plan, neighborhood, changed)


def test_neighborhood_receipt_hides_evidence_text_but_binds_bytes(
    tmp_path: Path,
) -> None:
    plan, text = _one_chunk_plan(tmp_path)
    neighborhood = _neighborhood(plan, text)

    assert text not in str(neighborhood.projection())
    assert neighborhood.evidence[0].text_sha256 == quote_sha256(text)
    assert neighborhood.receipt_sha256 == bind_post_selection_em_neighborhood(
        plan,
        question_id=neighborhood.question_id,
        question_sha256=neighborhood.question_sha256,
        dated_question_sha256=neighborhood.dated_question_sha256,
        source_stage_id=neighborhood.source_stage_id,
        upstream_selection_receipt_sha256=(
            neighborhood.upstream_selection_receipt_sha256
        ),
        evidence=(FastEvidence(neighborhood.evidence_ids[0], "history-a", text),),
        selection_ids=("em-a",),
    ).receipt_sha256
    assert neighborhood.evidence[0].namespace_id == _NAMESPACE_ID
