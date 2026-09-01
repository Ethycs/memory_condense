from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_adaptive_source_map as source_cli
from tools import run_locked_adaptive_source_tail_wave as tail_cli
from tools._routed_repair_routing import route_question
from tools.matched_eval import adaptive_source_tail_typed as tail_typed
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.source_gate_controller import (
    MapWorkAlias,
    ObligationKind,
    QuestionBoundMappingPlan,
    QuestionBoundMapWork,
    QuestionObligation,
)
from tools.matched_eval.source_history_fact_union import (
    DirectEvidenceRef,
    FactLane,
    FrozenHistoryChunk,
    HydratedSourceHistory,
    ParentIdentity,
    SourceSelection,
    direct_evidence_projection_sha256,
    plan_source_history_hydration,
)
from tools.matched_eval.source_history_mapper_live import (
    MAPPER_CONTRACT_SHA256,
    SourceMapperCachedCompletion,
    build_source_history_mapper_preflight,
    materialize_source_history_mapper,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceStatus,
    FrontierMode,
    ProvenanceGrade,
    TypedItemKind,
    merge_typed_evidence_contributions,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


def _sha(value: str) -> str:
    return quote_sha256(value)


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("6 plants on 2023-03-01", 6.0),
        ("on 2023-03-01 there were 6 plants", 6.0),
        ("on 2023-03-01", None),
    ),
)
def test_tail_numeric_value_ignores_all_iso_date_fragments(
    text: str, expected: float | None
) -> None:
    assert tail_typed._numeric_value(text) == expected  # noqa: SLF001


def _tail_row(
    dated_question: str,
    mapped: tuple[tuple[str, str, dict[str, str] | None], ...],
    *,
    ordinal: int = 69,
    protect_direct: bool = False,
) -> tail_typed.TailFactUnionRow:
    question_key = _sha(dated_question)
    question_id = f"q-{question_key[:8]}"
    namespace = _sha("typed-tail-test-namespace")
    source_id = f"source-{question_key[:8]}"
    text = " ".join(dict.fromkeys(quote for _fact, quote, _event in mapped))
    chunk = FrozenHistoryChunk(
        source_id,
        _sha(f"chunk:{question_key}"),
        _sha(f"turn:{question_key}"),
        1,
        "user",
        "2023-03-10T12:00:00+00:00",
        0,
        len(text),
        text,
        count_tokens(text),
        quote_sha256(text),
        False,
    )
    membership = _sha(f"membership:{question_key}")
    history = HydratedSourceHistory(
        namespace,
        source_id,
        (chunk.chunk_id,),
        (),
        _sha(f"stream:{question_key}"),
        membership,
        (chunk,),
        True,
        _sha(f"history:{question_key}"),
    )
    direct_evidence = (
        tuple(
            DirectEvidenceRef(
                f"D{index:03d}",
                namespace,
                source_id,
                quote_sha256(quote),
                _sha(f"direct-evidence:{question_key}:{index}"),
                text=quote,
            )
            for index, (_fact, quote, _event) in enumerate(mapped, start=1)
        )
        if protect_direct
        else ()
    )
    parent = ParentIdentity(
        _sha("population"),
        _sha("question-order"),
        _sha("snapshot"),
        namespace,
        _sha(f"parent-packet:{question_key}"),
        _sha(f"parent-stage:{question_key}"),
        direct_evidence_projection_sha256(direct_evidence),
    )
    selection = SourceSelection(
        f"selection-{question_key[:12]}",
        FactLane.DIRECT,
        namespace,
        source_id,
        4,
        _sha(f"selector:{question_key}"),
    )
    hydration = plan_source_history_hydration(
        parent,
        selections=(selection,),
        histories=(history,),
        max_window_tokens=800,
    )
    window = hydration.windows[0]
    route = route_question(dated_question)
    obligations = (
        QuestionObligation(
            ObligationKind.SUPPORT,
            ("bike",),
            1,
            1,
            1,
            route.modifiers.requires_temporal_metadata,
            route.modifiers.requires_complete_frontier,
        ),
    )
    work = QuestionBoundMapWork(
        _sha(f"gate:{question_key}"),
        parent.identity_sha256,
        question_id,
        question_key,
        dated_question,
        question_key,
        route.receipt_sha256,
        obligations,
        namespace,
        source_id,
        membership,
        history.stream_sha256,
        history.receipt_sha256,
        window.window_ordinal,
        window.token_cap,
        window.content_token_proxy,
        window.chunks,
        MAPPER_CONTRACT_SHA256,
    )
    alias = MapWorkAlias(
        work.work_id,
        hydration.receipt_sha256,
        window.window_id,
        window.receipt_sha256,
        window.mapping_payload_sha256,
        selection.selection_id,
        selection.lane,
    )
    mapping = QuestionBoundMappingPlan(
        work.gate_plan_receipt_sha256,
        _sha(f"round:{question_key}"),
        hydration.receipt_sha256,
        (work,),
        (alias,),
        (work.work_id,),
        (),
        (),
        (),
    )
    preflight = build_source_history_mapper_preflight(hydration, mapping)
    prompt = preflight.prompt_rows[0]
    completion = json.dumps(
        {
            "facts": [
                {
                    "chunk_alias": "C1",
                    "event_tuple": event,
                    "fact": fact,
                    "quote": quote,
                    "source_alias": "S1",
                }
                for fact, quote, event in mapped
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    cache = SourceMapperCachedCompletion(
        work.work_id,
        prompt.prompt_id,
        prompt.messages_sha256,
        completion,
        quote_sha256(completion),
        _sha(f"prior-work-result:{question_key}"),
    )
    materialization = materialize_source_history_mapper(
        preflight,
        hydration,
        mapping,
        cached_completions=(cache,),
    )
    question = source_cli.FastMaterializationQuestionPlan(
        ordinal,
        question_id,
        direct_evidence,
        hydration,
        mapping,
        preflight,
    )
    return tail_typed.build_tail_post_map_fact_unions(
        (question,), (materialization,)
    )[0]


Q69_STYLE = """[Question asked at 2023/03/20 (Mon) 23:57]
How many bikes did I service or plan to service in March?"""


def test_q69_style_exclusions_become_two_semantic_pointers_not_source_duplicates() -> None:
    row = _tail_row(
        Q69_STYLE,
        (
            ("The user serviced 1 road bike in March.", "I serviced 1 road bike in March.", None),
            (
                "The user planned to service 1 mountain bike in March.",
                "I planned to service 1 mountain bike in March.",
                None,
            ),
        ),
        protect_direct=True,
    )
    spec = compile_typed_operator_spec(Q69_STYLE)
    source, pointers = tail_typed.adapt_tail_question_contributions(
        spec,
        row,
        materialization_artifact_sha256=_sha("sealed-tail-materialization"),
        parent_prompt_token_proxy=0,
        source_handle_start=401,
        source_group_start=601,
        pointer_handle_start=501,
        pointer_group_start=701,
    )

    assert row.fact_union.accepted_before_dedup_count == 2
    assert len(row.fact_union.retained_facts) == len(source.bindings) == 0
    assert len(row.fact_union.direct_exclusions) == len(pointers.bindings) == 2
    assert tuple(binding.handle_id for binding in pointers.bindings) == (
        "H501",
        "H502",
    )
    assert {binding.source_group_handle for binding in pointers.bindings} == {
        "G701"
    }
    assert all(
        binding.provenance_grade is ProvenanceGrade.DIRECT_POINTER
        and binding.citation_char_count == 0
        for binding in pointers.bindings
    )
    assert tuple(item.kind for item in pointers.parsed.accepted_items) == (
        TypedItemKind.OPERAND,
        TypedItemKind.OPERAND,
    )
    assert tuple(
        item.numeric_value for item in pointers.parsed.accepted_items
    ) == (1.0, 1.0)
    assert source.frontier_mode is pointers.frontier_mode is FrontierMode.BOUNDED
    assert source.provider_prompt_count == pointers.provider_prompt_count == 0
    assert tuple(
        binding.local_source_locator_sha256 for binding in pointers.bindings
    ) == tuple(
        identity_sha256(
            tail_typed._direct_pointer_equivalence_locator(fact, refs)
        )
        for exclusion, fact, refs in tail_typed._direct_pointer_rows(row)
    )
    packet = merge_typed_evidence_contributions(spec, (pointers,))
    provider_payload = packet.render_provider_payload()
    assert "direct_pointer" in provider_payload
    for _exclusion, fact, refs in tail_typed._direct_pointer_rows(row):
        assert all(variant in provider_payload for variant in fact.fact_variants)
        assert all(
            ref.text is None
            or ref.text in fact.fact_variants
            or ref.text not in provider_payload
            for ref in refs
        )


def test_eventless_direct_pointers_keep_distinct_mapped_actions_provider_visible() -> None:
    question = (
        "[Question asked at 2023/03/20 (Mon) 23:57]\n"
        "How many clothing items do I need to pick up or return?"
    )
    row = _tail_row(
        question,
        (
            (
                "The old small Zara boots still need to be returned.",
                "I need to return the old small boots to Zara.",
                None,
            ),
            (
                "The new larger Zara boots still need to be picked up.",
                "I have not picked up the new larger boots from Zara.",
                None,
            ),
        ),
        protect_direct=True,
    )
    _source, pointers = tail_typed.adapt_tail_question_contributions(
        compile_typed_operator_spec(question),
        row,
        materialization_artifact_sha256=_sha("sealed-q69-tail"),
        parent_prompt_token_proxy=0,
        source_handle_start=301,
        source_group_start=301,
        pointer_handle_start=401,
        pointer_group_start=401,
    )
    summaries = tuple(item.summary for item in pointers.parsed.accepted_items)
    assert len(summaries) == 2
    assert summaries[0] != summaries[1]
    assert "returned" in summaries[0]
    assert "picked up" in summaries[1]
    assert all(not row.content_conflict for row in pointers.parsed.accepted_items)
    assert all(
        row.status is EvidenceStatus.CURRENT
        for row in pointers.parsed.accepted_items
    )


def test_post_map_duplicate_variants_and_small_noise_are_not_silently_filtered() -> None:
    shared_quote = "I serviced 1 bike in March."
    row = _tail_row(
        Q69_STYLE,
        (
            ("The user serviced 1 bike in March.", shared_quote, None),
            ("One bike service was completed in March.", shared_quote, None),
            ("The bicycle tire pressure was 45 PSI.", "The tire pressure was 45 PSI.", None),
        ),
    )
    assert row.fact_union.accepted_before_dedup_count == 3
    assert len(row.fact_union.retained_facts) == 2
    assert len(row.fact_union.retained_facts[0].fact_variants) == 2
    assert len(row.fact_union.retained_facts[0].origins) == 2

    contribution = tail_typed.adapt_tail_fact_union_contribution(
        compile_typed_operator_spec(Q69_STYLE),
        row,
        materialization_artifact_sha256=_sha("sealed-tail-materialization"),
        parent_prompt_token_proxy=0,
        handle_start=11,
        group_start=21,
    )
    summaries = tuple(item.summary for item in contribution.parsed.accepted_items)
    assert len(contribution.bindings) == len(summaries) == 2
    assert "One bike service was completed in March." in summaries[0]
    assert "tire pressure" in summaries[1]
    assert {binding.source_group_handle for binding in contribution.bindings} == {
        "G021"
    }


@pytest.mark.parametrize(
    ("question", "fact", "quote", "expected_kind"),
    (
        (
            "[Question asked at 2023/03/20 (Mon) 23:57]\nWhat color was the bicycle?",
            "The bicycle was blue.",
            "My bicycle was blue.",
            TypedItemKind.DIRECT,
        ),
        (
            "[Question asked at 2023/03/20 (Mon) 23:57]\n"
            "Which happened first, I bought the bicycle or I serviced it?",
            "The user bought the bicycle on 2023-03-01.",
            "I bought the bicycle on 2023-03-01.",
            TypedItemKind.EVENT,
        ),
    ),
)
def test_direct_and_timeline_routes_use_the_canonical_enum_kind(
    question: str,
    fact: str,
    quote: str,
    expected_kind: TypedItemKind,
) -> None:
    row = _tail_row(question, ((fact, quote, None),))
    contribution = tail_typed.adapt_tail_fact_union_contribution(
        compile_typed_operator_spec(question),
        row,
        materialization_artifact_sha256=_sha("sealed-tail-materialization"),
        parent_prompt_token_proxy=0,
        handle_start=31,
        group_start=41,
    )
    assert contribution.parsed.accepted_items[0].kind is expected_kind


def test_question_route_and_ordered_materialization_lineage_fail_closed() -> None:
    first = _tail_row(
        Q69_STYLE,
        (("The user serviced 1 road bike.", "I serviced 1 road bike.", None),),
    )
    other_question = (
        "[Question asked at 2023/03/20 (Mon) 23:57]\nWhat color was the bicycle?"
    )
    second = _tail_row(
        other_question,
        (("The bicycle was blue.", "My bicycle was blue.", None),),
    )
    with pytest.raises(MatchedEvalContractError, match="ordered question plan"):
        tail_typed.build_tail_post_map_fact_unions(
            (first.question_plan,), (second.mapper_materialization,)
        )
    with pytest.raises(MatchedEvalContractError, match="question/route binding"):
        tail_typed.adapt_tail_fact_union_contribution(
            compile_typed_operator_spec(other_question),
            first,
            materialization_artifact_sha256=_sha("sealed-tail-materialization"),
            parent_prompt_token_proxy=0,
            handle_start=1,
            group_start=1,
        )


def test_public_checkpoint_loader_rejects_hash_drift_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed = SealedArtifact(
        tmp_path / tail_cli.PREFLIGHT_NAME,
        _sha("observed-preflight"),
        {},
    )
    monkeypatch.setattr(tail_cli, "read_sealed_json", lambda _path: observed)
    with pytest.raises(tail_cli.LockedAdaptiveSourceTailError, match="preflight changed"):
        tail_cli.load_typed_tail_materialization_root(
            tmp_path,
            expected_preflight_sha256=_sha("expected-preflight"),
            expected_work_manifest_sha256=_sha("work"),
            expected_materialization_sha256=_sha("materialization"),
            expected_replay_sha256=_sha("replay"),
            model="sealed-model",
            gateway_url="http://sealed-gateway",
            max_concurrency=1,
        )


def test_public_checkpoint_loader_never_accepts_the_wave1_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _unexpected_read(_path: object) -> SealedArtifact:
        raise AssertionError("invalid wave-1 root must fail before artifact I/O")

    monkeypatch.setattr(tail_cli, "read_sealed_json", _unexpected_read)
    with pytest.raises(tail_cli.LockedAdaptiveSourceTailError, match="wave-1 root"):
        tail_cli.load_typed_tail_materialization_root(
            tmp_path / tail_cli.INVALID_WAVE1_DIR_NAME,
            expected_preflight_sha256=_sha("preflight"),
            expected_work_manifest_sha256=_sha("work"),
            expected_materialization_sha256=_sha("materialization"),
            expected_replay_sha256=_sha("replay"),
            model="sealed-model",
            gateway_url="http://sealed-gateway",
            max_concurrency=1,
        )
