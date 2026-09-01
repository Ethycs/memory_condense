from __future__ import annotations

import json
from dataclasses import replace

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256

from tools._routed_repair_routing import route_question
from tools.matched_eval.contracts import ArtifactRef, assert_gold_blind, identity_sha256
from tools.matched_eval.source_gate_controller import (
    EligibleFrontierScope,
    LaneSourceBudget,
    ObligationKind,
    QuestionObligation,
    SourceGateActivationReceipt,
    SourceGateCandidate,
    SourceGatePlan,
    SourceGatePolicy,
    build_question_bound_mapping_plan,
    start_source_gate,
)
from tools.matched_eval.source_history_fact_union import (
    FactLane,
    FrozenHistoryChunk,
    HydratedSourceHistory,
    ParentIdentity,
    build_post_map_fact_union,
    direct_evidence_projection_sha256,
    plan_source_history_hydration,
)
from tools.matched_eval.source_history_mapper_live import (
    HARD_CONTEXT_TOKEN_CAP,
    MAPPER_CONTRACT_SHA256,
    MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    SOURCE_ALIAS,
    SourceHistoryMapperError,
    SourceMapperCachedCompletion,
    SourceMapperProviderJournal,
    WorkDisposition,
    build_source_history_mapper_preflight,
    materialize_source_history_mapper,
    render_source_history_mapper_messages,
    replay_source_history_mapper,
)


def _sha(value: str) -> str:
    return quote_sha256(value)


_NAMESPACE = _sha("namespace")


def _parent() -> ParentIdentity:
    return ParentIdentity(
        population_identity_sha256=_sha("population"),
        question_order_sha256=_sha("question-order"),
        snapshot_id=_sha("snapshot"),
        namespace_id=_NAMESPACE,
        parent_packet_id=_sha("parent-packet"),
        parent_stage_receipt_sha256=_sha("parent-stage"),
        direct_evidence_projection_sha256=direct_evidence_projection_sha256(()),
    )


def _membership_sha(source_id: str, chunk_id: str) -> str:
    return identity_sha256(
        {
            "content_chunk_ids": [chunk_id],
            "metadata_chunk_ids": [],
            "source_id": source_id,
            "stream_sha256": _sha(f"stream:{source_id}"),
        }
    )


def _history(text: str, source_id: str = "history-shared") -> HydratedSourceHistory:
    chunk_id = _sha(f"chunk:{source_id}")
    chunk = FrozenHistoryChunk(
        source_id=source_id,
        chunk_id=chunk_id,
        turn_id=_sha(f"turn:{source_id}"),
        turn_ordinal=1,
        role="user",
        created_at="2026-08-01T00:00:00+00:00",
        start_char=0,
        end_char=len(text),
        text=text,
        token_count=count_tokens(text),
        turn_text_sha256=quote_sha256(text),
        metadata_chunk=False,
    )
    return HydratedSourceHistory(
        namespace_id=_NAMESPACE,
        source_id=source_id,
        content_chunk_ids=(chunk_id,),
        metadata_chunk_ids=(),
        stream_sha256=_sha(f"stream:{source_id}"),
        membership_projection_sha256=_membership_sha(source_id, chunk_id),
        chunks=(chunk,),
        store_bytes_revalidated=True,
        receipt_sha256=_sha(f"history:{source_id}:{text}"),
    )


def _candidate(lane: FactLane, source_id: str, rank: int) -> SourceGateCandidate:
    chunk_id = _sha(f"chunk:{source_id}")
    return SourceGateCandidate(
        lane=lane,
        namespace_id=_NAMESPACE,
        source_id=source_id,
        rank=rank,
        membership_projection_sha256=_membership_sha(source_id, chunk_id),
        stream_sha256=_sha(f"stream:{source_id}"),
        source_stream_receipt_sha256=_sha(f"stream-receipt:{lane.value}"),
    )


def _gate(*, call_cap: int = 4) -> SourceGatePlan:
    question = "What color did Alpha choose?"
    obligation = QuestionObligation(
        kind=ObligationKind.SUPPORT,
        match_terms=("alpha", "blue"),
        required_match_term_count=2,
    )
    parent = _parent()
    candidates = (
        _candidate(FactLane.DIRECT, "history-shared", 0),
        _candidate(FactLane.GUIDED, "history-shared", 0),
    )
    policy = SourceGatePolicy(
        policy_id="mapper-test-policy-v1",
        lane_budgets=(
            LaneSourceBudget(FactLane.DIRECT, 1, 1, 1),
            LaneSourceBudget(FactLane.PARTITION, 0, 1, 1),
            LaneSourceBudget(FactLane.GUIDED, 1, 1, 1),
        ),
        global_unique_source_cap=2,
        max_physical_map_calls=call_cap,
        max_rounds=2,
    )
    activation = SourceGateActivationReceipt(
        question_id="question-1",
        question_sha256=_sha(question),
        dated_question_sha256=_sha(question),
        parent_packet_id=parent.parent_packet_id,
        upstream_question_plan_receipt_sha256=_sha("upstream-question-plan"),
        upstream_fact_frontier_receipt_sha256=_sha("upstream-fact-frontier"),
        obligation_ids=(obligation.obligation_id,),
        unresolved_obligation_ids=(obligation.obligation_id,),
    )
    return SourceGatePlan(
        parent=parent,
        question_id="question-1",
        question_sha256=_sha(question),
        dated_question=question,
        dated_question_sha256=_sha(question),
        as_of_turn=31,
        route=route_question(question),
        sealed_input_artifacts=(ArtifactRef("sealed-input", _sha("input")),),
        candidates=candidates,
        obligations=(obligation,),
        activation=activation,
        eligible_frontier=EligibleFrontierScope(
            eligible_candidate_ids=tuple(row.candidate_id for row in candidates),
            exhaustive=False,
            basis_receipt_sha256=_sha("eligible-frontier"),
        ),
        policy=policy,
    )


def _plans(
    text: str = "Alpha chose blue yesterday.",
    *,
    cached_work_ids: tuple[str, ...] = (),
    call_cap: int = 4,
):
    gate = _gate(call_cap=call_cap)
    gate_round = start_source_gate(gate)
    history = _history(text)
    hydration = plan_source_history_hydration(
        gate.parent,
        selections=gate_round.selections,
        histories=(history,),
        max_window_tokens=max(1, count_tokens(text)),
    )
    mapping = build_question_bound_mapping_plan(
        gate,
        gate_round,
        hydration,
        mapper_contract_sha256=MAPPER_CONTRACT_SHA256,
        cached_work_ids=cached_work_ids,
    )
    return history, hydration, mapping


def _completion(quote: str = "Alpha chose blue") -> str:
    return json.dumps(
        {
            "facts": [
                {
                    "chunk_alias": "C1",
                    "event_tuple": None,
                    "fact": "Alpha chose blue.",
                    "quote": quote,
                    "source_alias": SOURCE_ALIAS,
                }
            ]
        },
        sort_keys=True,
    )


def _representative_multi_chunk_work():
    _history_row, _hydration, mapping = _plans()
    base = mapping.work_items[0]
    source_id = _sha("representative-long-source-id")
    texts = tuple(
        (
            f"On day {index}, Alpha recorded preference {index} and explained "
            f"the supporting reason in source-history detail {index}."
        )
        for index in range(1, 9)
    )
    chunks = tuple(
        FrozenHistoryChunk(
            source_id=source_id,
            chunk_id=_sha(f"representative-chunk:{index}"),
            turn_id=_sha(f"representative-turn:{index}"),
            turn_ordinal=index,
            role="user" if index % 2 else "assistant",
            created_at=f"2026-08-{index:02d}T12:00:00+00:00",
            start_char=0,
            end_char=len(text),
            text=text,
            token_count=count_tokens(text),
            turn_text_sha256=quote_sha256(text),
            metadata_chunk=index == 1,
        )
        for index, text in enumerate(texts, 1)
    )
    content_tokens = sum(row.token_count for row in chunks)
    return replace(
        base,
        source_id=source_id,
        membership_projection_sha256=_sha("representative-membership"),
        stream_sha256=_sha("representative-stream"),
        source_history_receipt_sha256=_sha("representative-history"),
        history_window_token_cap=content_tokens,
        content_token_proxy=content_tokens,
        chunks=chunks,
    )


def _journal(preflight, completion: str, *, checkpoint_hit: bool = False):
    row = next(row for row in preflight.prompt_rows if row.submitted)
    return SourceMapperProviderJournal(
        physical_work_id=row.physical_work_id,
        prompt_id=row.prompt_id,
        messages_sha256=row.messages_sha256,
        call_key_sha256=_sha("call-key"),
        request_journal_sha256=_sha("request-journal"),
        response_journal_sha256=_sha("response-journal"),
        completion=completion,
        completion_sha256=quote_sha256(completion),
        physical_call=not checkpoint_hit,
        checkpoint_hit=checkpoint_hit,
    )


def test_preflight_is_one_physical_prompt_with_two_lane_aliases_and_hard_8k() -> None:
    _history_row, hydration, mapping = _plans()
    preflight = build_source_history_mapper_preflight(hydration, mapping)

    assert len(mapping.work_items) == 1
    assert len(mapping.aliases) == 2
    assert preflight.required_provider_calls == 1
    assert preflight.logical_alias_count == 2
    assert preflight.provider_population is not None
    assert preflight.provider_population.unique_prompt_count == 1
    row = preflight.prompt_rows[0]
    assert row.disposition is WorkDisposition.NEW_CALL
    assert row.prompt_token_proxy <= MAX_PROMPT_TOKENS
    assert row.output_token_reserve == OUTPUT_TOKEN_RESERVE
    assert row.combined_token_proxy <= HARD_CONTEXT_TOKEN_CAP
    prompt = "\n".join(message.content for message in row.messages)
    assert "What color did Alpha choose?" in prompt
    assert "alpha" in prompt and "blue" in prompt
    assert "Alpha chose blue yesterday." in prompt
    assert '"chunk_alias":"C1"' in prompt
    assert f'"source_alias":"{SOURCE_ALIAS}"' in prompt
    assert "reference_answer" not in prompt and "gold_answer" not in prompt
    user = row.messages[1].content
    source_window = user.split("SOURCE_HISTORY_WINDOW_JSON:\n", 1)[1].split(
        "\n\nFACT_MAP_JSON:", 1
    )[0]
    payload = json.loads(source_window)
    assert payload == {
        "chunks": [
            {
                "chunk_alias": "C1",
                "date": "2026-08-01T00:00:00+00:00",
                "kind": "content",
                "role": "user",
                "text": "Alpha chose blue yesterday.",
            }
        ],
        "frozen": True,
        "source_alias": SOURCE_ALIAS,
    }
    # Exact local provenance and cryptographic receipts stay outside the
    # model-visible history window.
    assert "history-shared" not in source_window
    assert _sha("chunk:history-shared") not in source_window
    assert _NAMESPACE not in source_window
    assert_gold_blind(preflight.projection())


def test_compact_alias_window_reduces_multi_chunk_prompt_token_proxy() -> None:
    work = _representative_multi_chunk_work()
    messages = render_source_history_mapper_messages(work)
    compact_json = messages[1].content.split(
        "SOURCE_HISTORY_WINDOW_JSON:\n", 1
    )[1].split("\n\nFACT_MAP_JSON:", 1)[0]
    compact_payload = json.loads(compact_json)
    legacy_payload = {
        "chunks": [row.projection(include_text=True) for row in work.chunks],
        "frozen_chunk_boundaries": True,
        "history_window_ordinal": work.history_window_ordinal,
        "namespace_id": work.namespace_id,
        "source_history_receipt_sha256": work.source_history_receipt_sha256,
        "source_id": work.source_id,
        "stream_sha256": work.stream_sha256,
    }
    compact_proxy = count_chat_prompt_token_proxy(
        (
            {
                "role": "user",
                "content": "SOURCE_HISTORY_WINDOW_JSON:\n" + compact_json,
            },
        )
    )
    legacy_proxy = count_chat_prompt_token_proxy(
        (
            {
                "role": "user",
                "content": "SOURCE_HISTORY_WINDOW_JSON:\n"
                + json.dumps(
                    legacy_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        )
    )

    assert [row["chunk_alias"] for row in compact_payload["chunks"]] == [
        f"C{index}" for index in range(1, 9)
    ]
    assert all("date" in row and "role" in row for row in compact_payload["chunks"])
    # Same eight texts and chat framing: removing repeated exact provenance
    # cuts the deterministic window proxy from 2,412 to 500 tokens (79.3%).
    assert (legacy_proxy, compact_proxy) == (2_412, 500)
    assert legacy_proxy > compact_proxy
    assert legacy_proxy - compact_proxy >= 1_000
    assert compact_proxy * 2 < legacy_proxy


def test_materialization_derives_quote_coordinates_then_preserves_lane_discovery() -> None:
    _history_row, hydration, mapping = _plans()
    preflight = build_source_history_mapper_preflight(hydration, mapping)
    raw = _completion()
    journal = _journal(preflight, raw)

    result = materialize_source_history_mapper(
        preflight,
        hydration,
        mapping,
        provider_journals=(journal,),
    )

    assert result.provider_calls_during_materialization == 0
    assert result.historical_physical_calls == 1
    assert result.journal_checkpoint_hits == 0
    assert len(result.work_results) == 1
    assert len(result.batches) == 2
    assert tuple(batch.accepted[0].lane for batch in result.batches) == (
        FactLane.DIRECT,
        FactLane.GUIDED,
    )
    for batch in result.batches:
        fact = batch.accepted[0]
        assert fact.source_id == "history-shared"
        assert fact.chunk_id == _sha("chunk:history-shared")
        assert fact.quote == "Alpha chose blue"
        assert fact.quote_start_char == 0
        assert fact.quote_end_char == len("Alpha chose blue")
        assert fact.quote_sha256 == quote_sha256("Alpha chose blue")

    # This layer never deduplicates. The existing downstream union does so only
    # after both lane aliases retain their discovery provenance.
    union = build_post_map_fact_union(hydration, batches=result.batches)
    assert len(union.retained_facts) == 1
    assert {origin.lane for origin in union.retained_facts[0].origins} == {
        FactLane.DIRECT,
        FactLane.GUIDED,
    }
    assert result.projection()["post_map_dedup_performed"] is False

    replay = replay_source_history_mapper(
        preflight,
        hydration,
        mapping,
        provider_journals=(journal,),
        expected_materialization_receipt_sha256=result.receipt_sha256,
    )
    assert replay.byte_identical is True
    assert replay.provider_calls_during_replay == 0


def test_nonexact_quote_is_salvaged_as_a_rejection_not_promoted_to_fact() -> None:
    _history_row, hydration, mapping = _plans()
    preflight = build_source_history_mapper_preflight(hydration, mapping)
    journal = _journal(preflight, _completion("Alpha selected blue"))

    result = materialize_source_history_mapper(
        preflight,
        hydration,
        mapping,
        provider_journals=(journal,),
    )

    assert len(result.batches) == 2
    assert all(not batch.accepted for batch in result.batches)
    assert all(len(batch.rejected) == 1 for batch in result.batches)
    assert all(
        "quote_not_exact" in batch.rejected[0].reason
        for batch in result.batches
    )


@pytest.mark.parametrize(
    "completion",
    (
        json.dumps(
            {
                "facts": [
                    {
                        "chunk_alias": "C999",
                        "event_tuple": None,
                        "fact": "Alpha chose blue.",
                        "quote": "Alpha chose blue",
                        "source_alias": SOURCE_ALIAS,
                    }
                ]
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "facts": [
                    {
                        "chunk_alias": "C1",
                        "event_tuple": None,
                        "fact": "Alpha chose blue.",
                        "quote": "Alpha chose blue",
                        "source_alias": "S999",
                    }
                ]
            },
            sort_keys=True,
        ),
        # The v1 exact-ID response schema must not enter a v2 alias-bound
        # cache or materialization under a changed contract.
        json.dumps(
            {
                "facts": [
                    {
                        "chunk_id": _sha("chunk:history-shared"),
                        "event_tuple": None,
                        "fact": "Alpha chose blue.",
                        "quote": "Alpha chose blue",
                        "source_id": "history-shared",
                    }
                ]
            },
            sort_keys=True,
        ),
    ),
)
def test_unknown_aliases_and_v1_exact_ids_fail_closed_per_item(completion: str) -> None:
    _history_row, hydration, mapping = _plans()
    preflight = build_source_history_mapper_preflight(hydration, mapping)
    result = materialize_source_history_mapper(
        preflight,
        hydration,
        mapping,
        provider_journals=(_journal(preflight, completion),),
    )

    assert all(not batch.accepted for batch in result.batches)
    assert all(len(batch.rejected) == 1 for batch in result.batches)
    assert all("item_schema" in batch.rejected[0].reason for batch in result.batches)


def test_cached_completion_is_not_submitted_and_replays_same_logical_aliases() -> None:
    _history_row, hydration, first_mapping = _plans()
    first_preflight = build_source_history_mapper_preflight(hydration, first_mapping)
    completion = _completion()
    first = materialize_source_history_mapper(
        first_preflight,
        hydration,
        first_mapping,
        provider_journals=(_journal(first_preflight, completion),),
    )

    _history_row, cached_hydration, cached_mapping = _plans(
        cached_work_ids=(first_mapping.work_items[0].work_id,)
    )
    cached_preflight = build_source_history_mapper_preflight(
        cached_hydration, cached_mapping
    )
    row = cached_preflight.prompt_rows[0]
    cached = SourceMapperCachedCompletion(
        physical_work_id=row.physical_work_id,
        prompt_id=row.prompt_id,
        messages_sha256=row.messages_sha256,
        completion=completion,
        completion_sha256=quote_sha256(completion),
        original_work_result_receipt_sha256=first.work_results[0].receipt_sha256,
    )

    assert row.disposition is WorkDisposition.REUSED
    assert cached_preflight.required_provider_calls == 0
    assert cached_preflight.provider_population is None
    result = materialize_source_history_mapper(
        cached_preflight,
        cached_hydration,
        cached_mapping,
        cached_completions=(cached,),
    )
    assert len(result.batches) == 2
    assert result.historical_physical_calls == 0
    assert result.work_results[0].completion_source == "sealed_cache"


def test_journal_binding_and_context_overflow_fail_closed() -> None:
    _history_row, hydration, mapping = _plans()
    preflight = build_source_history_mapper_preflight(hydration, mapping)
    journal = _journal(preflight, _completion())
    with pytest.raises(SourceHistoryMapperError, match="prompt binding"):
        materialize_source_history_mapper(
            preflight,
            hydration,
            mapping,
            provider_journals=(
                replace(journal, prompt_id=_sha("different-prompt")),
            ),
        )

    # The window itself can be legal under the hydration contract while its
    # dated question/schema/chat framing makes the live provider envelope too
    # large. Preflight, not execution, owns this hard failure.
    huge = " ".join(f"token-{index}" for index in range(9_000))
    _history_row, huge_hydration, huge_mapping = _plans(huge)
    with pytest.raises(SourceHistoryMapperError, match="envelope overflow"):
        build_source_history_mapper_preflight(huge_hydration, huge_mapping)


def test_deferred_work_renders_and_budgets_but_never_materializes() -> None:
    _history_row, hydration, mapping = _plans(call_cap=0)
    preflight = build_source_history_mapper_preflight(hydration, mapping)

    assert preflight.prompt_rows[0].disposition is WorkDisposition.DEFERRED
    assert preflight.required_provider_calls == 0
    result = materialize_source_history_mapper(preflight, hydration, mapping)
    assert result.work_results == ()
    assert result.batches == ()
    assert result.deferred_work_ids == mapping.deferred_work_ids
