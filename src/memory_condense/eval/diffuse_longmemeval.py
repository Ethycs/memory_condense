"""Provider-free LongMemEval bridge for episodic discourse retrieval.

The live retrieval half of this module is deliberately gold-blind.  It accepts
the exact ranked :class:`~memory_condense.domain.schemas.RetrievalResult` rows
produced by a caller, expands those rows through a selected episode artifact,
closes the query's discourse obligations, and atomically packs the final
provider-visible evidence packet.  Gold answers and evidence-source labels
enter only through :func:`measure_longmemeval_diffuse_packet`, after retrieval
and packing have completed.

No provider, responder, judge, or transformer runtime is imported or called.
The optional Qwen episode signal belongs to the earlier annotation build, not
to this query-time evaluation seam.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    EvidencePacket,
    EvidenceSpan,
    EpisodeSeed,
    QueryProgram,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval.answer_value_coverage import best_f1, contains_answer
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    build_qa_prompt,
)
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPlan,
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRetrievalPlan,
    EpisodeRetrievalPolicy,
    EpisodeSourceCandidate,
    EpisodeSourceCandidateScope,
    NestedEpisodeLinker,
)


DIFFUSE_QUERY_RECEIPT_FORMAT = (
    "memory-condense-longmemeval-diffuse-query-receipt-v1"
)


class SupportsDiffuseEvidence(Protocol):
    """Existing application facade required by the gold-blind query path."""

    def expand_discourse_episode_seeds(
        self,
        results: Sequence[RetrievalResult],
        *,
        policy: EpisodeRetrievalPolicy | None = None,
    ) -> EpisodeRetrievalPlan: ...

    def retrieve_discourse_episode_representatives(
        self,
        query: str,
        source_candidates: Sequence[EpisodeSourceCandidate],
        linker: NestedEpisodeLinker,
        *,
        policy: EpisodeRepresentativeRetrievalPolicy,
        source_scope: EpisodeSourceCandidateScope | None = None,
    ) -> EpisodeRepresentativeRetrievalPlan: ...

    def close_discourse_evidence(
        self,
        query: str | QueryProgram | None = None,
        *,
        query_program: QueryProgram | None = None,
        seeds=(),
        direct_chunk_ids=(),
        policy: ClosurePolicy | None = None,
        artifact_id: str | None = None,
        expansion_receipt_sha256: str | None = None,
        expansion_exhaustive: bool | None = None,
    ) -> ClosurePlan: ...

    def pack_discourse_evidence(
        self,
        plan: ClosurePlan,
        *,
        max_context_tokens: int,
        encoding: str = "cl100k_base",
        base_messages: Sequence[Mapping[str, str]] | None = None,
        evidence_message_role: str = "user",
        evidence_prefix: str = "",
        evidence_suffix: str = "",
        max_prompt_tokens: int | None = None,
        output_token_reserve: int = 0,
    ) -> EvidencePacket: ...


def _exact_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        normalized = int(value)  # type: ignore[arg-type]
        exact = math.isfinite(float(value)) and float(value) == normalized  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if not exact or normalized < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return normalized


def _anchor_payload(result: RetrievalResult) -> dict[str, object]:
    optional_scores: dict[str, float | None] = {}
    for name in ("dense_score", "lexical_score", "association_score"):
        raw = getattr(result, name)
        value = None if raw is None else float(raw)
        if value is not None and not math.isfinite(value):
            raise ValueError(f"anchor {name} must be finite when present")
        optional_scores[name] = value
    score = float(result.score)
    if not math.isfinite(score):
        raise ValueError("anchor score must be finite")
    turn = result.turn
    if turn is not None and turn.turn_id != result.chunk.turn_id:
        raise ValueError("anchor chunk and hydrated turn disagree")
    memory_source = (
        None
        if result.memory_source_id is None
        else str(result.memory_source_id).strip()
    )
    turn_source = (
        None
        if turn is None or turn.source_id is None
        else str(turn.source_id).strip()
    )
    if memory_source and turn_source and memory_source != turn_source:
        raise ValueError("anchor source identities disagree")
    return {
        "chunk_id": result.chunk.chunk_id,
        "turn_id": result.chunk.turn_id,
        "source_id": (
            memory_source
            or turn_source
            or result.chunk.turn_id
        ),
        "start_char": result.chunk.start_char,
        "end_char": result.chunk.end_char,
        "token_count": result.chunk.token_count,
        "text_sha256": quote_sha256(result.chunk.text),
        "score": score,
        "route": result.route or "unspecified",
        **optional_scores,
    }


def _evidence_coordinates(packet: EvidencePacket) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "atom_id": atom.atom_id,
            **atom.span.identity_payload(),
            "label": atom.label,
        }
        for atom in packet.atoms
    )


def _qa_packet_framing(question: str) -> tuple[str, str]:
    """Split the authoritative QA template around one atomic packet."""

    if QA_USER_TEMPLATE.count("{context}") != 1:
        raise RuntimeError("QA_USER_TEMPLATE must contain exactly one context slot")
    prefix_template, suffix_template = QA_USER_TEMPLATE.split("{context}", 1)
    try:
        prefix = prefix_template.format(question=question) + "[1] "
        suffix = suffix_template.format(question=question)
    except (KeyError, ValueError) as exc:  # pragma: no cover - template guard
        raise RuntimeError("QA_USER_TEMPLATE has unsupported fields") from exc
    return prefix, suffix


@dataclass(frozen=True, slots=True)
class LongMemEvalDiffuseQueryReceipt:
    """Text-free binding from exact direct anchors to the final QA prompt."""

    artifact_id: str
    snapshot_sha256: str
    anchor_sequence_sha256: str
    input_anchor_chunk_ids: tuple[str, ...]
    episode_policy_sha256: str
    expansion_receipt_sha256: str
    representative_receipt_sha256: str | None
    representative_scope_exhaustive: bool | None
    representative_runtime_binding_certified: bool | None
    representative_returned_plan_transformer_state_bytes: int | None
    combined_expansion_sha256: str
    representative_seed_episode_ids: tuple[str, ...]
    truncated_episode_ids: tuple[str, ...]
    truncated_direct_chunk_ids: tuple[str, ...]
    expansion_exhaustive: bool
    query_program_sha256: str
    retrieval_query_sha256: str
    prompt_question_sha256: str
    closure_policy_sha256: str
    closure_plan_sha256: str
    closure_stopping_reason: str
    closure_complete_claimed: bool
    scope_witness_sha256s: tuple[str, ...]
    closure_scope_exhaustive: bool
    packet_receipt_sha256: str
    context_sha256: str
    evidence_coordinates_sha256: str
    prompt_messages_sha256: str
    prompt_token_proxy: int
    max_input_prompt_token_proxy: int
    responder_output_token_reserve: int
    prompt_workspace_token_proxy: int
    max_prompt_workspace_token_proxy: int
    packet_retained_request_token_state_bytes: int
    store_retained_request_token_state_bytes: int | None = None
    format: str = DIFFUSE_QUERY_RECEIPT_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != DIFFUSE_QUERY_RECEIPT_FORMAT:
            raise ValueError("unsupported diffuse query receipt format")
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        for name in (
            "snapshot_sha256",
            "anchor_sequence_sha256",
            "episode_policy_sha256",
            "expansion_receipt_sha256",
            "combined_expansion_sha256",
            "query_program_sha256",
            "retrieval_query_sha256",
            "prompt_question_sha256",
            "closure_policy_sha256",
            "closure_plan_sha256",
            "packet_receipt_sha256",
            "context_sha256",
            "evidence_coordinates_sha256",
            "prompt_messages_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.representative_receipt_sha256 is not None:
            value = str(self.representative_receipt_sha256)
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(
                    "representative_receipt_sha256 must be a SHA-256 digest"
                )
        if self.representative_scope_exhaustive is not None and type(
            self.representative_scope_exhaustive
        ) is not bool:
            raise ValueError("representative_scope_exhaustive must be boolean")
        if self.representative_runtime_binding_certified is not None and type(
            self.representative_runtime_binding_certified
        ) is not bool:
            raise ValueError(
                "representative_runtime_binding_certified must be boolean"
            )
        if self.representative_receipt_sha256 is None:
            if (
                self.representative_scope_exhaustive is not None
                or self.representative_runtime_binding_certified is not None
                or self.representative_returned_plan_transformer_state_bytes
                is not None
                or self.representative_seed_episode_ids
            ):
                raise ValueError(
                    "absent representative retrieval cannot carry its results"
                )
        elif self.representative_scope_exhaustive is None:
            raise ValueError(
                "representative retrieval requires an exhaustiveness claim"
            )
        elif self.representative_runtime_binding_certified is None:
            raise ValueError(
                "representative retrieval requires a runtime-binding claim"
            )
        if self.prompt_workspace_token_proxy != (
            self.prompt_token_proxy + self.responder_output_token_reserve
        ):
            raise ValueError("prompt workspace must equal prompt plus output reserve")
        if self.prompt_token_proxy > self.max_input_prompt_token_proxy:
            raise ValueError("final prompt exceeds the LongMemEval input cap")
        if (
            self.prompt_workspace_token_proxy
            > self.max_prompt_workspace_token_proxy
        ):
            raise ValueError("final request exceeds its workspace cap")
        if self.max_prompt_workspace_token_proxy != (
            self.max_input_prompt_token_proxy
            + self.responder_output_token_reserve
        ):
            raise ValueError("workspace cap must preserve the exact input cap")
        for name in (
            "packet_retained_request_token_state_bytes",
            "representative_returned_plan_transformer_state_bytes",
            "store_retained_request_token_state_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value != 0):
                raise ValueError(f"{name} must be zero or unattested")
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("diffuse query receipt does not match its contents")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "receipt_sha256"
        }
        payload.update(
            {
                "input_anchor_chunk_ids": list(self.input_anchor_chunk_ids),
                "representative_seed_episode_ids": list(
                    self.representative_seed_episode_ids
                ),
                "truncated_episode_ids": list(self.truncated_episode_ids),
                "truncated_direct_chunk_ids": list(
                    self.truncated_direct_chunk_ids
                ),
                "scope_witness_sha256s": list(self.scope_witness_sha256s),
            }
        )
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True, slots=True)
class LongMemEvalDiffuseRetrieval:
    """Gold-blind final packet plus every intermediate immutable receipt."""

    expansion: EpisodeRetrievalPlan
    representative_expansion: EpisodeRepresentativeRetrievalPlan | None
    plan: ClosurePlan
    packet: EvidencePacket
    messages: tuple[dict[str, str], ...]
    evidence_coordinates: tuple[dict[str, object], ...]
    receipt: LongMemEvalDiffuseQueryReceipt

    def provider_messages(self) -> list[dict[str, str]]:
        """Return a defensive provider-compatible copy; this method calls none."""

        return [dict(message) for message in self.messages]


@dataclass(frozen=True, slots=True)
class LongMemEvalDiffuseMetrics:
    """Post-packet, provider-free LongMemEval answer/source reachability."""

    question_id: str
    retrieval_receipt_sha256: str
    answer_present: bool
    best_evidence_f1: float
    expected_source_ids: tuple[str, ...]
    retrieved_source_ids: tuple[str, ...]
    evidence_source_recall: float | None
    any_evidence_source: bool | None
    all_evidence_sources: bool | None
    selected_atoms: int
    selected_bundles: int
    source_span_hash_valid: bool
    closure_complete_claimed: bool
    closure_scope_exhaustive: bool
    hard_budget_compliant: bool
    context_token_proxy: int
    prompt_token_proxy: int
    prompt_workspace_token_proxy: int


def retrieve_longmemeval_diffuse_packet(
    condenser: SupportsDiffuseEvidence,
    *,
    query: str,
    prompt_question: str | None = None,
    anchors: Sequence[RetrievalResult],
    artifact_id: str,
    max_context_tokens: int,
    max_prompt_tokens: int,
    responder_output_token_reserve: int = (
        BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
    ),
    episode_policy: EpisodeRetrievalPolicy | None = None,
    source_candidates: Sequence[EpisodeSourceCandidate] = (),
    source_candidate_scope: EpisodeSourceCandidateScope | None = None,
    representative_linker: NestedEpisodeLinker | None = None,
    representative_policy: EpisodeRepresentativeRetrievalPolicy | None = None,
    require_owned_representative_runtime: bool = False,
    closure_policy: ClosurePolicy | None = None,
    query_program: QueryProgram | None = None,
) -> LongMemEvalDiffuseRetrieval:
    """Close and pack one question without accepting any benchmark gold.

    ``max_prompt_tokens`` retains the benchmark's provider-input-cap meaning.
    The atomic packer instead accepts a total workspace ceiling, so this bridge
    passes ``max_prompt_tokens + responder_output_token_reserve`` and records
    both quantities.  No later truncation or repacking is permitted.
    """

    normalized_query = str(query).strip()
    normalized_prompt_question = str(
        normalized_query if prompt_question is None else prompt_question
    ).strip()
    normalized_artifact = str(artifact_id).strip()
    if not normalized_query or not normalized_prompt_question or not normalized_artifact:
        raise ValueError("query, prompt question, and artifact_id must be non-empty")
    context_cap = _exact_nonnegative_int(max_context_tokens, "max_context_tokens")
    prompt_cap = _exact_nonnegative_int(max_prompt_tokens, "max_prompt_tokens")
    reserve = _exact_nonnegative_int(
        responder_output_token_reserve,
        "responder_output_token_reserve",
    )
    if prompt_cap < 1:
        raise ValueError("max_prompt_tokens must be positive")

    exact_anchors = tuple(anchors)
    if any(not isinstance(item, RetrievalResult) for item in exact_anchors):
        raise TypeError("anchors must contain RetrievalResult values")
    anchor_payload = tuple(_anchor_payload(item) for item in exact_anchors)
    active_episode_policy = episode_policy or EpisodeRetrievalPolicy(
        artifact_id=normalized_artifact
    )
    if active_episode_policy.artifact_id not in (None, normalized_artifact):
        raise ValueError("episode policy belongs to another artifact")
    if active_episode_policy.artifact_id is None:
        active_episode_policy = replace(
            active_episode_policy,
            artifact_id=normalized_artifact,
        )

    expansion = condenser.expand_discourse_episode_seeds(
        exact_anchors,
        policy=active_episode_policy,
    )
    expansion_exhaustive = not (
        expansion.truncated_episode_ids
        or expansion.truncated_direct_chunk_ids
    )
    representative_expansion = None
    exact_source_candidates = tuple(source_candidates)
    if source_candidate_scope is not None:
        if source_candidate_scope.artifact_id != normalized_artifact:
            raise ValueError("source candidate scope belongs to another artifact")
        if source_candidate_scope.query_sha256 != identity_sha256(
            {"query": normalized_query}
        ):
            raise ValueError("source candidate scope belongs to another query")
        if exact_source_candidates and exact_source_candidates != (
            source_candidate_scope.candidates
        ):
            raise ValueError("source candidates disagree with their scope receipt")
        exact_source_candidates = source_candidate_scope.candidates
    if (representative_linker is None) != (representative_policy is None):
        raise ValueError(
            "representative linker and policy must be supplied together"
        )
    if source_candidate_scope is not None and representative_linker is None:
        raise ValueError("source candidate scope requires representative retrieval")
    if require_owned_representative_runtime and representative_linker is None:
        raise ValueError("owned representative runtime requires representative retrieval")
    if exact_source_candidates and representative_linker is None:
        raise ValueError(
            "source candidates require representative linker and policy"
        )
    if representative_linker is not None and representative_policy is not None:
        if representative_policy.artifact_id != normalized_artifact:
            raise ValueError("representative policy belongs to another artifact")
        representative_expansion = (
            condenser.retrieve_discourse_episode_representatives(
                normalized_query,
                exact_source_candidates,
                representative_linker,
                policy=representative_policy,
                source_scope=source_candidate_scope,
            )
        )
        if representative_expansion.artifact_id != normalized_artifact:
            raise ValueError("representative plan belongs to another artifact")
        if representative_expansion.policy_sha256 != (
            representative_policy.policy_sha256
        ):
            raise ValueError("representative plan policy identity changed")
        if representative_expansion.query_sha256 != identity_sha256(
            {"query": normalized_query}
        ):
            raise ValueError("representative plan belongs to another query")
        if (
            require_owned_representative_runtime
            and not representative_expansion.runtime_binding_certified
        ):
            raise ValueError("representative linker runtime is not certified")

    combined_seeds = _combine_episode_seeds(
        expansion.seeds,
        () if representative_expansion is None else representative_expansion.seeds,
    )
    combined_expansion_sha256 = identity_sha256(
        {
            "direct_expansion_receipt_sha256": expansion.receipt_sha256,
            "representative_expansion_receipt_sha256": (
                None
                if representative_expansion is None
                else representative_expansion.receipt_sha256
            ),
            "seeds": [_episode_seed_payload(seed) for seed in combined_seeds],
            "direct_chunk_ids": list(expansion.direct_chunk_ids),
        }
    )
    representative_exhaustive = (
        None
        if representative_expansion is None
        else representative_expansion.candidate_scope_exhaustive
    )
    combined_expansion_exhaustive = bool(
        expansion_exhaustive
        and (
            representative_exhaustive
            if representative_exhaustive is not None
            else True
        )
    )
    plan = condenser.close_discourse_evidence(
        normalized_query,
        query_program=query_program,
        seeds=combined_seeds,
        direct_chunk_ids=expansion.direct_chunk_ids,
        policy=closure_policy,
        artifact_id=normalized_artifact,
        expansion_receipt_sha256=combined_expansion_sha256,
        expansion_exhaustive=combined_expansion_exhaustive,
    )
    if plan.expansion_receipt_sha256 != combined_expansion_sha256:
        raise RuntimeError("closure plan does not bind the combined expansion")

    prefix, suffix = _qa_packet_framing(normalized_prompt_question)
    workspace_cap = prompt_cap + reserve
    packet = condenser.pack_discourse_evidence(
        plan,
        max_context_tokens=context_cap,
        base_messages=(
            {"role": "system", "content": QA_SYSTEM_PROMPT},
        ),
        evidence_message_role="user",
        evidence_prefix=prefix,
        evidence_suffix=suffix,
        max_prompt_tokens=workspace_cap,
        output_token_reserve=reserve,
    )
    # A packet is one atomic evidence excerpt in the existing QA protocol.
    messages = tuple(
        build_qa_prompt(normalized_prompt_question, [packet.context])
    )
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    if prompt_tokens != packet.receipt.prompt_token_proxy:
        raise RuntimeError("packet receipt does not bind the final QA prompt")
    if prompt_tokens > prompt_cap:
        raise RuntimeError("atomic packer exceeded the benchmark input cap")
    message_sha256 = identity_sha256(list(messages))
    if message_sha256 != packet.receipt.prompt_messages_sha256:
        raise RuntimeError("packet prompt hash does not match the final messages")

    coordinates = _evidence_coordinates(packet)
    discourse = getattr(condenser, "discourse", None)
    stats = getattr(discourse, "stats", None)
    store_retained = None
    if callable(stats):
        raw_retained = stats().get("retained_request_token_state_bytes")
        if raw_retained is not None:
            store_retained = raw_retained
    active_closure_policy = closure_policy or ClosurePolicy()
    receipt = LongMemEvalDiffuseQueryReceipt(
        artifact_id=normalized_artifact,
        snapshot_sha256=plan.snapshot.snapshot_sha256,
        anchor_sequence_sha256=identity_sha256(anchor_payload),
        input_anchor_chunk_ids=tuple(
            item.chunk.chunk_id for item in exact_anchors
        ),
        episode_policy_sha256=active_episode_policy.policy_sha256,
        expansion_receipt_sha256=expansion.receipt_sha256,
        representative_receipt_sha256=(
            None
            if representative_expansion is None
            else representative_expansion.receipt_sha256
        ),
        representative_scope_exhaustive=representative_exhaustive,
        representative_runtime_binding_certified=(
            None
            if representative_expansion is None
            else representative_expansion.runtime_binding_certified
        ),
        representative_returned_plan_transformer_state_bytes=(
            None
            if representative_expansion is None
            else representative_expansion.returned_plan_transformer_state_bytes
        ),
        combined_expansion_sha256=combined_expansion_sha256,
        representative_seed_episode_ids=(
            ()
            if representative_expansion is None
            else tuple(
                seed.episode_id for seed in representative_expansion.seeds
            )
        ),
        truncated_episode_ids=expansion.truncated_episode_ids,
        truncated_direct_chunk_ids=expansion.truncated_direct_chunk_ids,
        expansion_exhaustive=combined_expansion_exhaustive,
        query_program_sha256=plan.query_program.program_sha256,
        retrieval_query_sha256=identity_sha256({"query": normalized_query}),
        prompt_question_sha256=identity_sha256(
            {"prompt_question": normalized_prompt_question}
        ),
        closure_policy_sha256=active_closure_policy.policy_sha256,
        closure_plan_sha256=plan.plan_sha256,
        closure_stopping_reason=plan.stopping_reason,
        closure_complete_claimed=plan.complete_claimed,
        scope_witness_sha256s=tuple(
            witness.witness_sha256 for witness in plan.scope_witnesses
        ),
        closure_scope_exhaustive=bool(
            plan.scope_witnesses
            and all(witness.exhaustive for witness in plan.scope_witnesses)
        ),
        packet_receipt_sha256=packet.receipt.receipt_sha256,
        context_sha256=packet.receipt.context_sha256,
        evidence_coordinates_sha256=identity_sha256(coordinates),
        prompt_messages_sha256=message_sha256,
        prompt_token_proxy=prompt_tokens,
        max_input_prompt_token_proxy=prompt_cap,
        responder_output_token_reserve=reserve,
        prompt_workspace_token_proxy=prompt_tokens + reserve,
        max_prompt_workspace_token_proxy=workspace_cap,
        packet_retained_request_token_state_bytes=(
            packet.receipt.retained_request_token_state_bytes
        ),
        store_retained_request_token_state_bytes=store_retained,
    )
    return LongMemEvalDiffuseRetrieval(
        expansion=expansion,
        representative_expansion=representative_expansion,
        plan=plan,
        packet=packet,
        messages=messages,
        evidence_coordinates=coordinates,
        receipt=receipt,
    )


def _episode_seed_payload(seed: EpisodeSeed) -> dict[str, object]:
    return {
        "episode_id": seed.episode_id,
        "anchor_chunk_id": seed.anchor_chunk_id,
        "score": seed.score,
        "route": seed.route,
        "path": list(seed.path),
    }


def _combine_episode_seeds(
    direct: Sequence[EpisodeSeed],
    representative: Sequence[EpisodeSeed],
) -> tuple[EpisodeSeed, ...]:
    selected: dict[str, EpisodeSeed] = {}
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
            key=lambda seed: (
                -seed.score,
                seed.episode_id,
                seed.anchor_chunk_id,
                seed.route,
                seed.path,
            ),
        )
    )


def measure_longmemeval_diffuse_packet(
    retrieval: LongMemEvalDiffuseRetrieval,
    *,
    question_id: str,
    gold_answer: str,
    evidence_source_ids: Sequence[str] = (),
    hydrate_span: Callable[[EvidenceSpan], str],
) -> LongMemEvalDiffuseMetrics:
    """Score only the frozen final packet; benchmark gold enters here first."""

    normalized_question_id = str(question_id).strip()
    if not normalized_question_id:
        raise ValueError("question_id must be non-empty")
    expected = tuple(
        dict.fromkeys(
            source.strip()
            for source in map(str, evidence_source_ids)
            if source.strip()
        )
    )
    evidence_texts = tuple(atom.text for atom in retrieval.packet.atoms)
    retrieved_sources = tuple(
        dict.fromkeys(
            atom.span.source_id
            for atom in retrieval.packet.atoms
            if atom.span.source_id
        )
    )
    expected_set = set(expected)
    retrieved_set = set(retrieved_sources)
    coverage = (
        len(expected_set & retrieved_set) / len(expected_set)
        if expected_set
        else None
    )
    hash_valid = True
    for atom in retrieval.packet.atoms:
        try:
            authoritative = hydrate_span(atom.span)
        except Exception:  # noqa: BLE001 - a failed resolver invalidates provenance
            hash_valid = False
            break
        if authoritative != atom.text or quote_sha256(authoritative) != (
            atom.span.quote_sha256
        ):
            hash_valid = False
            break
    packet_receipt = retrieval.packet.receipt
    receipt = retrieval.receipt
    hard_budget_compliant = bool(
        packet_receipt.context_token_proxy
        <= packet_receipt.max_context_token_proxy
        and receipt.prompt_token_proxy <= receipt.max_input_prompt_token_proxy
        and receipt.prompt_workspace_token_proxy
        <= receipt.max_prompt_workspace_token_proxy
    )
    return LongMemEvalDiffuseMetrics(
        question_id=normalized_question_id,
        retrieval_receipt_sha256=receipt.receipt_sha256,
        answer_present=contains_answer(evidence_texts, str(gold_answer)),
        best_evidence_f1=best_f1(evidence_texts, str(gold_answer)),
        expected_source_ids=expected,
        retrieved_source_ids=retrieved_sources,
        evidence_source_recall=coverage,
        any_evidence_source=(
            bool(expected_set & retrieved_set) if expected_set else None
        ),
        all_evidence_sources=(coverage == 1.0 if coverage is not None else None),
        selected_atoms=len(retrieval.packet.atoms),
        selected_bundles=len(retrieval.packet.bundles),
        source_span_hash_valid=hash_valid,
        closure_complete_claimed=receipt.closure_complete_claimed,
        closure_scope_exhaustive=receipt.closure_scope_exhaustive,
        hard_budget_compliant=hard_budget_compliant,
        context_token_proxy=packet_receipt.context_token_proxy,
        prompt_token_proxy=receipt.prompt_token_proxy,
        prompt_workspace_token_proxy=receipt.prompt_workspace_token_proxy,
    )


__all__ = [
    "DIFFUSE_QUERY_RECEIPT_FORMAT",
    "LongMemEvalDiffuseMetrics",
    "LongMemEvalDiffuseQueryReceipt",
    "LongMemEvalDiffuseRetrieval",
    "SupportsDiffuseEvidence",
    "measure_longmemeval_diffuse_packet",
    "retrieve_longmemeval_diffuse_packet",
]
