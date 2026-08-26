"""Acquisition, cumulative packing, and post-hoc scoring operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    EvidencePacket,
    QueryProgram,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval._identity import exact_int, sha256_digest
from memory_condense.eval._recall_guarded_cumulative_contracts import (
    _CUMULATIVE_STAGE_IDS,
    _atom_evidence_id,
    _freeze_messages,
    _nonempty,
    _numbered_context,
    _ordered_unique,
    _protected_evidence_id,
    causal_graph_context_budget,
    CausalCoveragePredecessor,
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    NovelClosureProjection,
    ProtectedExcerpt,
    RecallGuardedCumulativeReceipt,
)
from memory_condense.eval._recall_guarded_cumulative_result import (
    _addition_prompt_prefix,
    _novel_closure_projection,
    _stage_evidence_projection_sha256,
    RecallGuardedCumulativeMetrics,
    RecallGuardedCumulativeRetrieval,
    RecallGuardedCumulativeStageMetrics,
)
from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    best_f1,
    contains_answer,
)
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    QA_NO_CONTEXT,
    QA_SYSTEM_PROMPT,
    build_qa_prompt,
    cap_context_to_prompt_budget,
)
from memory_condense.eval.diffuse_longmemeval import (
    longmemeval_anchor_sequence_sha256,
)
from memory_condense.eval.schemas import RetrievalConfig
from memory_condense.eval.search_kwargs import graph_search_kwargs
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRepresentativeRetrievalPlan,
    EpisodeRetrievalPolicy,
    EpisodeRetrievalPlan,
    NestedEpisodeLinker,
)


def _require_causal_coverage(retrieval: RetrievalConfig) -> None:
    if not isinstance(retrieval, RetrievalConfig):
        raise TypeError("retrieval must be a RetrievalConfig")
    if retrieval.mode != "causal_graph" or not retrieval.coverage_selection:
        raise ValueError(
            "the cumulative predecessor requires causal_graph with coverage_selection"
        )


def _validate_coverage_runtime_binding(
    retrieval: RetrievalConfig,
    report: Mapping[str, Any],
    *,
    required: bool,
) -> bool:
    """Verify checkpoint identity when the caller requests a certified arm."""

    if type(required) is not bool:
        raise ValueError("require_certified_coverage_runtime must be boolean")
    if not required:
        return False
    backend = retrieval.coverage_selector_backend
    if backend == "local_ini":
        raise ValueError(
            "local_ini coverage has no checkpoint receipt and cannot certify "
            "the frozen production runtime"
        )
    expected: dict[str, str] = {}
    if backend in {
        "qwen_prefix",
        "qwen_prefix_choice",
        "cross_encoder_qwen_prefix",
    }:
        expected.update(
            {
                "prefix_model_id": retrieval.coverage_selector_prefix_model_id,
                "prefix_model_revision": (
                    retrieval.coverage_selector_prefix_revision
                ),
                "prefix_checkpoint_sha256": (
                    retrieval.coverage_selector_prefix_checkpoint_sha256
                ),
            }
        )
    if backend in {"cross_encoder", "cross_encoder_qwen_prefix"}:
        expected.update(
            {
                "semantic_model_id": (
                    retrieval.coverage_selector_cross_encoder_model_id
                ),
                "semantic_model_revision": (
                    retrieval.coverage_selector_cross_encoder_revision
                ),
                "semantic_checkpoint_sha256": (
                    retrieval.coverage_selector_cross_encoder_checkpoint_sha256
                ),
            }
        )
    if not expected or any(not str(value).strip() for value in expected.values()):
        raise ValueError("coverage runtime certification requires exact model identity")
    for name, value in expected.items():
        if str(report.get(name, "")) != str(value):
            raise ValueError(f"coverage runtime report changed {name}")
    if backend == "qwen_prefix_choice":
        choice_report = report.get("score_provider_report")
        if not isinstance(choice_report, Mapping):
            raise ValueError(
                "coverage runtime report is missing the choice-provider identity"
            )
        expected_choice = {
            "model_id": retrieval.coverage_selector_choice_model_id,
            "model_revision": retrieval.coverage_selector_choice_revision,
            "checkpoint_sha256": (
                retrieval.coverage_selector_choice_checkpoint_sha256
            ),
        }
        if any(not str(value).strip() for value in expected_choice.values()):
            raise ValueError(
                "coverage runtime certification requires exact choice-model identity"
            )
        for name, value in expected_choice.items():
            if str(choice_report.get(name, "")) != str(value):
                raise ValueError(f"coverage runtime choice report changed {name}")
    return True


def retrieve_causal_coverage_predecessor(
    condenser: Any,
    *,
    retrieval_query: str,
    prompt_question: str,
    retrieval: RetrievalConfig,
    matched_controls_sha256: str,
    max_prompt_tokens: int,
    responder_output_token_reserve: int = BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    require_certified_coverage_runtime: bool = True,
) -> CausalCoveragePredecessor:
    """Acquire the exact frozen-v3 rendered packet and rehydrate only its IDs."""

    _require_causal_coverage(retrieval)
    query = _nonempty(retrieval_query, "retrieval_query")
    prompt = _nonempty(prompt_question, "prompt_question")
    sha256_digest(matched_controls_sha256, "matched_controls_sha256")
    prompt_cap = exact_int(max_prompt_tokens, "max_prompt_tokens", minimum=1)
    reserve = exact_int(
        responder_output_token_reserve,
        "responder_output_token_reserve",
        minimum=0,
    )
    expected_budget = causal_graph_context_budget(retrieval)
    actual_budget = getattr(getattr(condenser, "_packer", None), "budget", None)
    if actual_budget != expected_budget:
        raise ValueError(
            "condenser ContextBudget does not match the frozen causal-graph arm"
        )

    graph_results = tuple(
        condenser.search_hybrid_graph(
            query,
            **graph_search_kwargs(retrieval, routing=True),
        )
    )
    if any(not isinstance(item, RetrievalResult) for item in graph_results):
        raise TypeError("causal graph retrieval returned a non-RetrievalResult row")
    packed = condenser.build_context(
        query,
        recent_turns=0,
        k_memories=0,
        k_expansions=0,
        hybrid=True,
        reheat_memories=False,
        use_consolidation=True,
        learn_consolidation=False,
        consolidation_memory_slots=0,
        consolidation_chunk_slots=retrieval.consolidation_chunk_slots,
        consolidation_min_count=retrieval.consolidation_min_count,
        consolidation_hops=retrieval.consolidation_hops,
        consolidation_candidates=retrieval.consolidation_candidates,
        consolidation_diffusion_width=retrieval.consolidation_diffusion_width,
        expansion_results=graph_results,
    )
    if packed.memory_header or packed.memory_ids:
        raise RuntimeError("causal_graph predecessor unexpectedly emitted memory items")
    packed_ids = tuple(packed.expansion_chunk_ids)
    packed_texts = tuple(packed.expansions)
    if len(packed_ids) != len(packed_texts) or len(set(packed_ids)) != len(packed_ids):
        raise RuntimeError("packed predecessor coordinates are not one-to-one")

    coverage_report = dict(
        getattr(condenser, "last_coverage_selection_report", {}) or {}
    )
    coverage_trace = tuple(
        dict(item)
        for item in (
            getattr(condenser, "last_coverage_candidate_trace", ()) or ()
        )
    )
    if not coverage_report:
        raise RuntimeError(
            "coverage_selection was configured but no selector report was produced"
        )
    retained = coverage_report.get("retained_transformer_state_bytes", 0)
    if type(retained) is not int or retained != 0:
        raise RuntimeError("coverage selector retained request transformer state")
    coverage_runtime_certified = _validate_coverage_runtime_binding(
        retrieval,
        coverage_report,
        required=require_certified_coverage_runtime,
    )

    protected_texts = tuple(
        cap_context_to_prompt_budget(prompt, list(packed_texts), prompt_cap)
    )
    protected_ids = packed_ids[: len(protected_texts)]
    raw_by_id = {item.chunk.chunk_id: item for item in graph_results}
    trace_by_id = {
        str(item.get("chunk_id")): item
        for item in coverage_trace
        if item.get("chunk_id") is not None
    }
    anchors: list[RetrievalResult] = []
    total = max(len(protected_ids), 1)
    for rank, chunk_id in enumerate(protected_ids, 1):
        item = raw_by_id.get(chunk_id)
        route = str(trace_by_id.get(chunk_id, {}).get("route") or "causal_coverage")
        score = (total - rank + 1) / total
        if item is None:
            item = condenser.retriever.hydrate_chunk(
                chunk_id,
                score=score,
                route=route,
            )
        else:
            item = item.model_copy(update={"score": score, "route": route})
        if item is None:
            raise RuntimeError("packed predecessor chunk could not be rehydrated")
        anchors.append(item)
    frozen_anchors = tuple(anchors)
    excerpts = tuple(
        ProtectedExcerpt(
            chunk_id=chunk_id,
            source_id=anchor.durable_source_id,
            text=text,
        )
        for chunk_id, anchor, text in zip(
            protected_ids,
            frozen_anchors,
            protected_texts,
            strict=True,
        )
    )
    messages = _freeze_messages(build_qa_prompt(prompt, list(protected_texts)))
    protected_context = _numbered_context(protected_texts)
    direct = tuple(
        chunk_id
        for chunk_id in packed.direct_expansion_chunk_ids
        if chunk_id in set(protected_ids)
    )
    receipt = CausalCoveragePredecessorReceipt(
        matched_controls_sha256=matched_controls_sha256,
        retrieval_query_sha256=identity_sha256({"query": query}),
        prompt_question_sha256=identity_sha256({"prompt_question": prompt}),
        retrieval_policy_sha256=identity_sha256(retrieval.model_dump(mode="json")),
        context_budget_sha256=identity_sha256(
            {
                name: getattr(expected_budget, name)
                for name in expected_budget.__dataclass_fields__
            }
        ),
        raw_graph_anchor_sequence_sha256=longmemeval_anchor_sequence_sha256(
            graph_results
        ),
        raw_graph_chunk_ids=tuple(item.chunk.chunk_id for item in graph_results),
        packed_chunk_ids=packed_ids,
        protected_chunk_ids=protected_ids,
        direct_protected_chunk_ids=direct,
        protected_excerpt_projection_sha256=identity_sha256(
            [item.identity_payload() for item in excerpts]
        ),
        protected_context_sha256=quote_sha256(protected_context),
        selected_anchor_sequence_sha256=longmemeval_anchor_sequence_sha256(
            frozen_anchors
        ),
        coverage_selector_report_sha256=identity_sha256(coverage_report),
        coverage_candidate_trace_sha256=identity_sha256(coverage_trace),
        coverage_runtime_certified=coverage_runtime_certified,
        packed_token_counts=tuple(sorted(packed.token_counts.items())),
        packed_dropped_counts=tuple(sorted(packed.dropped.items())),
        prompt_messages_sha256=identity_sha256(list(messages)),
        prompt_token_proxy=count_chat_prompt_token_proxy(messages),
        max_prompt_token_proxy=prompt_cap,
        responder_output_token_reserve=reserve,
        retained_request_token_state_bytes=0,
    )
    return CausalCoveragePredecessor(
        excerpts=excerpts,
        anchors=frozen_anchors,
        messages=messages,
        receipt=receipt,
    )


def _pack_additions(
    condenser: Any,
    plan: ClosurePlan,
    *,
    prompt_question: str,
    protected_context: str,
    protected_count: int,
    max_context_tokens: int,
    max_prompt_tokens: int,
    responder_output_token_reserve: int,
) -> EvidencePacket | None:
    if not plan.atoms or not plan.bundles:
        return None
    protected_tokens = count_tokens(protected_context)
    if protected_tokens > max_context_tokens:
        raise ValueError("protected predecessor exceeds the cumulative context cap")
    addition_cap = max_context_tokens - protected_tokens
    prefix, suffix = _addition_prompt_prefix(
        prompt_question,
        protected_context,
        protected_count,
    )
    empty_messages = (
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": prefix + suffix},
    )
    if count_chat_prompt_token_proxy(empty_messages) > max_prompt_tokens:
        return None
    workspace_cap = max_prompt_tokens + responder_output_token_reserve
    while addition_cap >= 0:
        packet = condenser.pack_discourse_evidence(
            plan,
            max_context_tokens=addition_cap,
            base_messages=({"role": "system", "content": QA_SYSTEM_PROMPT},),
            evidence_message_role="user",
            evidence_prefix=prefix,
            evidence_suffix=suffix,
            max_prompt_tokens=workspace_cap,
            output_token_reserve=responder_output_token_reserve,
        )
        if not packet.context:
            return packet
        combined = (
            f"{protected_context}\n[{protected_count + 1}] {packet.context}"
            if protected_context
            else f"[1] {packet.context}"
        )
        total = count_tokens(combined)
        if total <= max_context_tokens:
            return packet
        addition_cap -= max(1, total - max_context_tokens)
    return None  # pragma: no cover - loop returns at cap zero


def _episode_seed_payload(seed: Any) -> dict[str, object]:
    return {
        "episode_id": seed.episode_id,
        "anchor_chunk_id": seed.anchor_chunk_id,
        "score": seed.score,
        "route": seed.route,
        "path": list(seed.path),
    }


def _combine_episode_seeds(
    direct: Sequence[Any],
    representative: Sequence[Any],
) -> tuple[Any, ...]:
    selected: dict[str, Any] = {}
    for seed in (*direct, *representative):
        prior = selected.get(seed.episode_id)
        if prior is None or (
            -seed.score,
            seed.anchor_chunk_id,
            seed.route,
            seed.path,
        ) < (
            -prior.score,
            prior.anchor_chunk_id,
            prior.route,
            prior.path,
        ):
            selected[seed.episode_id] = seed
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (
                -item.score,
                item.episode_id,
                item.anchor_chunk_id,
                item.route,
                item.path,
            ),
        )
    )


def _widen_direct_episode_policy(
    policy: EpisodeRetrievalPolicy,
    anchor_count: int,
) -> EpisodeRetrievalPolicy:
    """Ensure every protected anchor can map without policy truncation."""

    anchors = max(anchor_count, 1)
    episode_bound = anchors * (
        1 + policy.previous_episodes + policy.next_episodes
    )
    return replace(
        policy,
        max_anchor_episodes=max(policy.max_anchor_episodes, anchors),
        max_episode_seeds=max(policy.max_episode_seeds, episode_bound),
        max_direct_fallbacks=max(policy.max_direct_fallbacks, anchors),
    )


def _close_cumulative_method_plan(
    condenser: Any,
    *,
    query: str,
    query_program: QueryProgram | None,
    seeds: Sequence[Any],
    direct_chunk_ids: Sequence[str],
    policy: ClosurePolicy,
    artifact_id: str,
    expansion_identity_sha256: str,
    expansion_exhaustive: bool,
    routing_scope: Literal["artifact_global", "seeded_graph"],
) -> ClosurePlan:
    plan = condenser.close_discourse_evidence(
        None if query_program is not None else query,
        query_program=query_program,
        seeds=tuple(seeds),
        direct_chunk_ids=tuple(direct_chunk_ids),
        policy=policy,
        artifact_id=artifact_id,
        expansion_receipt_sha256=expansion_identity_sha256,
        expansion_exhaustive=expansion_exhaustive,
        routing_scope=routing_scope,
    )
    return plan


def retrieve_recall_guarded_cumulative_packet(
    condenser: Any,
    *,
    query: str,
    prompt_question: str | None,
    retrieval: RetrievalConfig,
    artifact_id: str,
    max_context_tokens: int,
    max_prompt_tokens: int,
    responder_output_token_reserve: int = BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    episode_policy: EpisodeRetrievalPolicy | None = None,
    representative_linker: NestedEpisodeLinker,
    representative_policy: EpisodeRepresentativeRetrievalPolicy,
    source_router_max_sources: int = 64,
    source_router_rrf_constant: int = 60,
    closure_policy: ClosurePolicy | None = None,
    query_program: QueryProgram | None = None,
    require_certified_coverage_runtime: bool = True,
    require_owned_representative_runtime: bool = True,
) -> RecallGuardedCumulativeRetrieval:
    """Run four independently measurable, strictly cumulative retrieval stages.

    The predecessor search intentionally uses ``prompt_question`` because the
    frozen-v3 LongMemEval treatment used the dated question for retrieval.
    Direct episode mapping, representative episode selection, and global
    discourse closure then receive separate plans and provider-visible stages.
    Every child prompt preserves its complete parent evidence as an exact
    prefix and can spend only the remaining budget.
    """

    normalized_query = _nonempty(query, "query")
    prompt = _nonempty(
        normalized_query if prompt_question is None else prompt_question,
        "prompt_question",
    )
    artifact = _nonempty(artifact_id, "artifact_id")
    context_cap = exact_int(max_context_tokens, "max_context_tokens", minimum=1)
    prompt_cap = exact_int(max_prompt_tokens, "max_prompt_tokens", minimum=1)
    reserve = exact_int(
        responder_output_token_reserve,
        "responder_output_token_reserve",
        minimum=0,
    )
    max_sources = exact_int(
        source_router_max_sources,
        "source_router_max_sources",
        minimum=1,
    )
    rrf_constant = exact_int(
        source_router_rrf_constant,
        "source_router_rrf_constant",
        minimum=1,
    )
    if type(require_owned_representative_runtime) is not bool:
        raise ValueError("require_owned_representative_runtime must be boolean")
    _require_causal_coverage(retrieval)
    snapshot = condenser.discourse.snapshot()
    if artifact not in snapshot.artifact_ids:
        raise ValueError("artifact_id is absent from the active discourse snapshot")

    active_episode_policy = episode_policy or EpisodeRetrievalPolicy(
        artifact_id=artifact
    )
    if active_episode_policy.artifact_id is None:
        active_episode_policy = replace(active_episode_policy, artifact_id=artifact)
    if active_episode_policy.artifact_id != artifact:
        raise ValueError("episode policy belongs to another artifact")
    active_representative_policy = representative_policy
    if active_representative_policy.artifact_id != artifact:
        raise ValueError("representative policy belongs to another artifact")
    active_closure_policy = closure_policy or ClosurePolicy()

    matched_controls_sha256 = identity_sha256(
        {
            "artifact_id": artifact,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "retrieval_query_sha256": identity_sha256({"query": normalized_query}),
            "prompt_question_sha256": identity_sha256({"prompt_question": prompt}),
            "retrieval_policy_sha256": identity_sha256(
                retrieval.model_dump(mode="json")
            ),
            "episode_policy_sha256": active_episode_policy.policy_sha256,
            "representative_policy_sha256": (
                active_representative_policy.policy_sha256
            ),
            "closure_policy_sha256": active_closure_policy.policy_sha256,
            "query_program_sha256": (
                None if query_program is None else query_program.program_sha256
            ),
            "source_router_max_sources": max_sources,
            "source_router_rrf_constant": rrf_constant,
            "max_context_tokens": context_cap,
            "max_prompt_tokens": prompt_cap,
            "responder_output_token_reserve": reserve,
            "require_certified_coverage_runtime": (
                require_certified_coverage_runtime
            ),
            "require_owned_representative_runtime": (
                require_owned_representative_runtime
            ),
            "method_stages": list(_CUMULATIVE_STAGE_IDS),
        }
    )
    predecessor = retrieve_causal_coverage_predecessor(
        condenser,
        retrieval_query=prompt,
        prompt_question=prompt,
        retrieval=retrieval,
        matched_controls_sha256=matched_controls_sha256,
        max_prompt_tokens=prompt_cap,
        responder_output_token_reserve=reserve,
        require_certified_coverage_runtime=(
            require_certified_coverage_runtime
        ),
    )
    if count_tokens(predecessor.protected_context) > context_cap:
        raise ValueError("protected predecessor exceeds the cumulative context cap")

    source_scope = condenser.route_discourse_episode_sources(
        normalized_query,
        predecessor.anchors,
        artifact_id=artifact,
        max_sources=max_sources,
        rrf_constant=rrf_constant,
    )
    active_episode_policy = _widen_direct_episode_policy(
        active_episode_policy,
        len(predecessor.anchors),
    )
    expansion = condenser.expand_discourse_episode_seeds(
        predecessor.anchors,
        policy=active_episode_policy,
    )
    if not isinstance(expansion, EpisodeRetrievalPlan):
        raise TypeError("direct episode expansion returned an invalid plan")
    if expansion.truncated_episode_ids or expansion.truncated_direct_chunk_ids:
        raise RuntimeError("protected direct episode expansion was truncated")
    representative_expansion = (
        condenser.retrieve_discourse_episode_representatives(
            normalized_query,
            source_scope.candidates,
            representative_linker,
            policy=active_representative_policy,
            source_scope=source_scope,
        )
    )
    if not isinstance(
        representative_expansion,
        EpisodeRepresentativeRetrievalPlan,
    ):
        raise TypeError("representative episode retrieval returned an invalid plan")
    if representative_expansion.artifact_id != artifact:
        raise ValueError("representative expansion belongs to another artifact")
    if representative_expansion.query_sha256 != identity_sha256(
        {"query": normalized_query}
    ):
        raise ValueError("representative expansion belongs to another query")
    if representative_expansion.policy_sha256 != (
        active_representative_policy.policy_sha256
    ):
        raise ValueError("representative expansion changed its policy")
    if (
        require_owned_representative_runtime
        and not representative_expansion.runtime_binding_certified
    ):
        raise ValueError("representative linker runtime is not certified")
    if representative_expansion.returned_plan_transformer_state_bytes != 0:
        raise ValueError("representative expansion retained transformer state")

    direct_expansion_identity = identity_sha256(
        {
            "method": "direct_episode",
            "episode_expansion_receipt_sha256": expansion.receipt_sha256,
            "seeds": [_episode_seed_payload(item) for item in expansion.seeds],
            "direct_chunk_ids": list(expansion.direct_chunk_ids),
        }
    )
    direct_plan = _close_cumulative_method_plan(
        condenser,
        query=normalized_query,
        query_program=query_program,
        seeds=expansion.seeds,
        direct_chunk_ids=expansion.direct_chunk_ids,
        policy=active_closure_policy,
        artifact_id=artifact,
        expansion_identity_sha256=direct_expansion_identity,
        expansion_exhaustive=True,
        routing_scope="seeded_graph",
    )
    active_program = direct_plan.query_program
    representative_expansion_identity = identity_sha256(
        {
            "method": "representative_episode",
            "representative_expansion_receipt_sha256": (
                representative_expansion.receipt_sha256
            ),
            "seeds": [
                _episode_seed_payload(item)
                for item in representative_expansion.seeds
            ],
            "direct_chunk_ids": [],
        }
    )
    representative_plan = _close_cumulative_method_plan(
        condenser,
        query=normalized_query,
        query_program=active_program,
        seeds=representative_expansion.seeds,
        direct_chunk_ids=(),
        policy=active_closure_policy,
        artifact_id=artifact,
        expansion_identity_sha256=representative_expansion_identity,
        expansion_exhaustive=(
            representative_expansion.candidate_scope_exhaustive
        ),
        routing_scope="seeded_graph",
    )
    combined_seeds = _combine_episode_seeds(
        expansion.seeds,
        representative_expansion.seeds,
    )
    union_expansion_identity = identity_sha256(
        {
            "method": "artifact_global_closure",
            "direct_expansion_receipt_sha256": expansion.receipt_sha256,
            "representative_expansion_receipt_sha256": (
                representative_expansion.receipt_sha256
            ),
            "seeds": [_episode_seed_payload(item) for item in combined_seeds],
            "direct_chunk_ids": list(expansion.direct_chunk_ids),
        }
    )
    union_plan = _close_cumulative_method_plan(
        condenser,
        query=normalized_query,
        query_program=active_program,
        seeds=combined_seeds,
        direct_chunk_ids=expansion.direct_chunk_ids,
        policy=active_closure_policy,
        artifact_id=artifact,
        expansion_identity_sha256=union_expansion_identity,
        expansion_exhaustive=(
            representative_expansion.candidate_scope_exhaustive
        ),
        routing_scope="artifact_global",
    )
    plans = (direct_plan, representative_plan, union_plan)
    if any(item.snapshot.snapshot_sha256 != snapshot.snapshot_sha256 for item in plans):
        raise RuntimeError("cumulative closure read a different discourse snapshot")

    protected_evidence_ids = tuple(
        _protected_evidence_id(item) for item in predecessor.excerpts
    )
    current_evidence_ids = protected_evidence_ids
    current_evidence_context = predecessor.protected_context
    current_messages = predecessor.messages
    current_entry_count = len(predecessor.excerpts)
    admitted_atoms: list[Any] = []
    projections: list[NovelClosureProjection] = []
    packets: list[EvidencePacket | None] = []
    root_context = current_evidence_context or QA_NO_CONTEXT
    root_stage = CumulativeRetrievalStageReceipt(
        stage_id=_CUMULATIVE_STAGE_IDS[0],
        matched_controls_sha256=matched_controls_sha256,
        method_evidence_sha256=predecessor.receipt.receipt_sha256,
        parent_stage_receipt_sha256=None,
        parent_evidence_ids=(),
        selected_evidence_ids=protected_evidence_ids,
        added_evidence_ids=protected_evidence_ids,
        admission_status="root",
        evidence_projection_sha256=_stage_evidence_projection_sha256(
            predecessor,
            (),
        ),
        context_sha256=quote_sha256(root_context),
        prompt_messages_sha256=identity_sha256(list(current_messages)),
        context_token_proxy=count_tokens(root_context),
        max_context_token_proxy=context_cap,
        prompt_token_proxy=count_chat_prompt_token_proxy(current_messages),
        max_prompt_token_proxy=prompt_cap,
        responder_output_token_reserve=reserve,
    )
    stages = [root_stage]

    for stage_id, plan in zip(_CUMULATIVE_STAGE_IDS[1:], plans, strict=True):
        projection = _novel_closure_projection(
            plan,
            predecessor.excerpts,
            admitted_atoms,
        )
        packet = _pack_additions(
            condenser,
            projection.plan,
            prompt_question=prompt,
            protected_context=current_evidence_context,
            protected_count=current_entry_count,
            max_context_tokens=context_cap,
            max_prompt_tokens=prompt_cap,
            responder_output_token_reserve=reserve,
        )
        packet_atoms = () if packet is None else tuple(packet.atoms)
        added_evidence_ids = tuple(_atom_evidence_id(item) for item in packet_atoms)
        if set(added_evidence_ids) & set(current_evidence_ids):
            raise RuntimeError("cumulative stage attempted to duplicate evidence")
        if packet_atoms:
            prefix, suffix = _addition_prompt_prefix(
                prompt,
                current_evidence_context,
                current_entry_count,
            )
            next_messages = _freeze_messages(
                (
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prefix + packet.context + suffix,
                    },
                )
            )
            current_evidence_context = (
                f"{current_evidence_context}\n"
                f"[{current_entry_count + 1}] {packet.context}"
                if current_evidence_context
                else f"[1] {packet.context}"
            )
            current_messages = next_messages
            current_entry_count += 1
            current_evidence_ids = (*current_evidence_ids, *added_evidence_ids)
            admitted_atoms.extend(packet_atoms)
            admission_status: Literal[
                "added", "no_novel_evidence", "budget_exhausted"
            ] = "added"
        else:
            admission_status = (
                "no_novel_evidence"
                if not projection.plan.atoms or not projection.plan.bundles
                else "budget_exhausted"
            )
        stage_context = current_evidence_context or QA_NO_CONTEXT
        parent_stage = stages[-1]
        stage = CumulativeRetrievalStageReceipt(
            stage_id=stage_id,
            matched_controls_sha256=matched_controls_sha256,
            method_evidence_sha256=projection.receipt.receipt_sha256,
            parent_stage_receipt_sha256=parent_stage.receipt_sha256,
            parent_evidence_ids=parent_stage.selected_evidence_ids,
            selected_evidence_ids=current_evidence_ids,
            added_evidence_ids=added_evidence_ids,
            admission_status=admission_status,
            evidence_projection_sha256=_stage_evidence_projection_sha256(
                predecessor,
                admitted_atoms,
            ),
            context_sha256=quote_sha256(stage_context),
            prompt_messages_sha256=identity_sha256(list(current_messages)),
            context_token_proxy=count_tokens(stage_context),
            max_context_token_proxy=context_cap,
            prompt_token_proxy=count_chat_prompt_token_proxy(current_messages),
            max_prompt_token_proxy=prompt_cap,
            responder_output_token_reserve=reserve,
        )
        projections.append(projection)
        packets.append(packet)
        stages.append(stage)

    ladder = CumulativeRetrievalLadder(stages=tuple(stages))
    context = current_evidence_context or QA_NO_CONTEXT
    messages = _freeze_messages(current_messages)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    context_tokens = count_tokens(context)
    if context_tokens > context_cap or prompt_tokens > prompt_cap:
        raise RuntimeError("cumulative packer exceeded a hard budget")
    protected_chunk_ids = predecessor.receipt.protected_chunk_ids
    added_atom_ids = tuple(item.atom_id for item in admitted_atoms)
    added_chunk_ids = _ordered_unique(
        tuple(item.span.chunk_id for item in admitted_atoms)
    )
    final_chunk_ids = _ordered_unique((*protected_chunk_ids, *added_chunk_ids))
    receipt = RecallGuardedCumulativeReceipt(
        matched_controls_sha256=matched_controls_sha256,
        predecessor_receipt_sha256=predecessor.receipt.receipt_sha256,
        direct_expansion_receipt_sha256=expansion.receipt_sha256,
        representative_expansion_receipt_sha256=(
            representative_expansion.receipt_sha256
        ),
        closure_plan_sha256s=tuple(item.plan_sha256 for item in plans),
        novel_projection_receipt_sha256s=tuple(
            item.receipt.receipt_sha256 for item in projections
        ),
        addition_packet_receipt_sha256s=tuple(
            None if item is None else item.receipt.receipt_sha256
            for item in packets
        ),
        stage_admission_statuses=tuple(
            item.admission_status for item in stages[1:]
        ),
        ladder_receipt_sha256=ladder.receipt_sha256,
        representative_runtime_certified=(
            representative_expansion.runtime_binding_certified
        ),
        protected_chunk_ids=protected_chunk_ids,
        protected_evidence_ids=protected_evidence_ids,
        added_atom_ids=added_atom_ids,
        added_chunk_ids=added_chunk_ids,
        final_chunk_ids=final_chunk_ids,
        final_evidence_ids=current_evidence_ids,
        protected_excerpt_projection_sha256=(
            predecessor.receipt.protected_excerpt_projection_sha256
        ),
        addition_evidence_projection_sha256=identity_sha256(
            [item.identity_payload() for item in admitted_atoms]
        ),
        final_context_sha256=quote_sha256(context),
        prompt_messages_sha256=identity_sha256(list(messages)),
        context_token_proxy=context_tokens,
        max_context_token_proxy=context_cap,
        prompt_token_proxy=prompt_tokens,
        max_prompt_token_proxy=prompt_cap,
        responder_output_token_reserve=reserve,
        prompt_workspace_token_proxy=prompt_tokens + reserve,
        retained_request_token_state_bytes=0,
    )
    return RecallGuardedCumulativeRetrieval(
        predecessor=predecessor,
        episode_expansion=expansion,
        representative_expansion=representative_expansion,
        closure_plans=plans,
        novel_projections=tuple(projections),
        addition_packets=tuple(packets),
        ladder=ladder,
        prompt_question=prompt,
        context=context,
        messages=messages,
        receipt=receipt,
    )


def measure_recall_guarded_cumulative_packet(
    retrieval: RecallGuardedCumulativeRetrieval,
    *,
    question_id: str,
    gold_answer: str,
    evidence_source_ids: Sequence[str],
    hydrate_span: Any | None = None,
) -> RecallGuardedCumulativeMetrics:
    """Score one already-frozen cumulative prompt; never influence retrieval."""

    if type(retrieval) is not RecallGuardedCumulativeRetrieval:
        raise TypeError("retrieval must be a RecallGuardedCumulativeRetrieval")
    expected = tuple(
        dict.fromkeys(
            _nonempty(item, "evidence_source_id") for item in evidence_source_ids
        )
    )
    retrieved = retrieval.retrieved_source_ids
    expected_set = set(expected)
    retrieved_set = set(retrieved)
    coverage = (
        None
        if not expected_set
        else len(expected_set & retrieved_set) / len(expected_set)
    )
    evidence_texts = [item.text for item in retrieval.predecessor.excerpts]
    stage_sources = [item.source_id for item in retrieval.predecessor.excerpts]
    stage_rows: list[RecallGuardedCumulativeStageMetrics] = []

    def stage_metric(index: int) -> RecallGuardedCumulativeStageMetrics:
        stage_retrieved = tuple(dict.fromkeys(stage_sources))
        stage_retrieved_set = set(stage_retrieved)
        stage_coverage = (
            None
            if not expected_set
            else len(expected_set & stage_retrieved_set) / len(expected_set)
        )
        values = answer_value_component_coverage(
            gold_answer,
            len(expected),
            list(evidence_texts),
        )
        stage_receipt = retrieval.ladder.stages[index]
        return RecallGuardedCumulativeStageMetrics(
            stage_id=stage_receipt.stage_id,
            answer_present=contains_answer(evidence_texts, gold_answer),
            best_evidence_f1=best_f1(evidence_texts, gold_answer),
            retrieved_source_ids=stage_retrieved,
            evidence_source_recall=stage_coverage,
            answer_value_components_expected=(
                None if values is None else values.expected
            ),
            answer_value_components_found=(
                None if values is None else values.found
            ),
            answer_value_component_recall=(
                None if values is None else values.recall
            ),
            all_answer_value_components=(
                None if values is None else values.all_components
            ),
            answer_value_component_hit_mask=(
                () if values is None else values.hit_mask
            ),
            answer_value_metric_kind=(
                "" if values is None else values.metric_kind
            ),
            context_token_proxy=stage_receipt.context_token_proxy,
            prompt_token_proxy=stage_receipt.prompt_token_proxy,
        )

    stage_rows.append(stage_metric(0))
    for index, packet in enumerate(retrieval.addition_packets, 1):
        if packet is not None:
            evidence_texts.extend(atom.text for atom in packet.atoms)
            stage_sources.extend(
                atom.span.source_id
                for atom in packet.atoms
                if atom.span.source_id is not None
            )
        stage_rows.append(stage_metric(index))
    if hydrate_span is not None:
        for packet in retrieval.addition_packets:
            if packet is None:
                continue
            for atom in packet.atoms:
                if hydrate_span(atom.span) != atom.text:
                    raise ValueError("addition evidence span hydration changed")
    values = answer_value_component_coverage(
        gold_answer,
        len(expected),
        list(evidence_texts),
    )
    return RecallGuardedCumulativeMetrics(
        question_id=_nonempty(question_id, "question_id"),
        retrieval_receipt_sha256=retrieval.receipt.receipt_sha256,
        answer_present=contains_answer(evidence_texts, gold_answer),
        best_evidence_f1=best_f1(evidence_texts, gold_answer),
        expected_source_ids=expected,
        retrieved_source_ids=retrieved,
        evidence_source_recall=coverage,
        any_evidence_source=(
            None if coverage is None else bool(expected_set & retrieved_set)
        ),
        all_evidence_sources=(None if coverage is None else coverage == 1.0),
        answer_value_components_expected=(
            None if values is None else values.expected
        ),
        answer_value_components_found=(
            None if values is None else values.found
        ),
        answer_value_component_recall=(
            None if values is None else values.recall
        ),
        all_answer_value_components=(
            None if values is None else values.all_components
        ),
        answer_value_component_hit_mask=(
            () if values is None else values.hit_mask
        ),
        answer_value_metric_kind=("" if values is None else values.metric_kind),
        protected_excerpts=len(retrieval.predecessor.excerpts),
        added_atoms=sum(
            0 if packet is None else len(packet.atoms)
            for packet in retrieval.addition_packets
        ),
        hard_budget_compliant=(
            retrieval.receipt.context_token_proxy
            <= retrieval.receipt.max_context_token_proxy
            and retrieval.receipt.prompt_token_proxy
            <= retrieval.receipt.max_prompt_token_proxy
        ),
        context_token_proxy=retrieval.receipt.context_token_proxy,
        prompt_token_proxy=retrieval.receipt.prompt_token_proxy,
        prompt_workspace_token_proxy=(
            retrieval.receipt.prompt_workspace_token_proxy
        ),
        stages=tuple(stage_rows),
    )
