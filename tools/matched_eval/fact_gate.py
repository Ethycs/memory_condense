"""Provider-free, parent-preserving fact gate for admitted evidence deltas.

The gate does not retrieve, compress, answer, judge, or expand construction
recall.  It validates an already selected evidence delta and an already
produced exact-cited :class:`EMFactCompression`, then reproduces the existing
facts-only EM representation with a question-only answer operator.  A caller
may submit the resulting prompt later; every non-admitted path exposes the
exact sealed parent prediction instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.fast_em_fact_memory import (
    DEFAULT_EM_STAGE_ID,
    EMFact,
    EMFactAnswerPrompt,
    EMFactCompression,
    EMFactMemoryError,
    parse_fact_compression,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastEvidence,
    FastProviderMessage,
)
from tools._routed_repair_prompts import (
    RoutedRepairPromptError,
    build_routed_answer_prompt,
    numeric_facts_are_quote_grounded,
)
from tools._routed_repair_routing import (
    ROUTED_REPAIR_ROUTING_FORMAT,
    RoutedRepairReceipt,
    RoutedRepairStyle,
    route_question,
)

from .contracts import MatchedEvalContractError, assert_gold_blind, require_text


FACT_GATE_FORMAT = "memory-condense-matched-admitted-fact-gate-v1"
FIXED_S1_ADAPTER_ID = "fixed_s1_post_selection_em_delta_v1"
CLOSURE_V9_ADAPTER_ID = "independent_closure_v9_admitted_delta_v1"
ROUTE_POLICY_PATH = Path(__file__).with_name("fact_gate_route_policy_v1.json")
ROUTE_POLICY_SHA256 = (
    "97d353def0d81183419e631b3227a8b7221c1e2d8acc4cb932486d95704a89c6"
)
MAX_FACTS = 24
MAX_PROMPT_TOKENS = 8_000
ANSWER_OUTPUT_TOKEN_RESERVE = 256

FactRoute = Literal[
    "direct_extract",
    "numeric_reduce",
    "set_join",
    "state_chain",
    "synthesize",
    "temporal_timeline",
]
GateDisposition = Literal["compiled", "parent_fallback"]


class FactGateError(MatchedEvalContractError):
    """Raised when the protected root or sealed route policy is invalid."""


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FactGateError(f"fact-gate policy repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FactGateError(f"fact-gate policy contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactGateError("fact-gate policy is not strict JSON") from exc
    if type(value) is not dict:
        raise FactGateError("fact-gate policy must be an exact object")
    return value


@dataclass(frozen=True, slots=True)
class FactRouteRule:
    route_id: FactRoute
    admitted: bool


@dataclass(frozen=True, slots=True)
class FactRoutePolicy:
    policy_id: str
    sha256: str
    classifier_format: str
    admitted_routes: frozenset[FactRoute]

    def route(self, dated_question: str) -> FactRouteRule:
        require_text(dated_question, "fact-gate dated question")
        receipt = route_question(dated_question)
        route_id: FactRoute = receipt.style.value
        return FactRouteRule(
            route_id=route_id,
            admitted=route_id in self.admitted_routes,
        )

    def classify(
        self,
        dated_question: str,
    ) -> tuple[FactRouteRule, RoutedRepairReceipt]:
        """Return the established question-only route and its sealed receipt."""

        require_text(dated_question, "fact-gate dated question")
        receipt = route_question(dated_question)
        route_id: FactRoute = receipt.style.value
        return (
            FactRouteRule(
                route_id=route_id,
                admitted=route_id in self.admitted_routes,
            ),
            receipt,
        )


def load_fact_route_policy(
    path: str | Path = ROUTE_POLICY_PATH,
    *,
    expected_sha256: str = ROUTE_POLICY_SHA256,
) -> FactRoutePolicy:
    """Load the separately pinned, question-only route policy."""

    source = Path(path)
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise FactGateError("fact-gate route policy SHA-256 changed")
    value = _strict_json_object(raw)
    expected_keys = {
        "admission_basis",
        "admitted_routes",
        "classification_input",
        "classifier_format",
        "construction_recall_claimed",
        "format",
        "policy_id",
        "source_target_expansion_claimed",
    }
    if set(value) != expected_keys:
        raise FactGateError("fact-gate route policy fields changed")
    if (
        value["format"]
        != "memory-condense-matched-fact-gate-route-policy-v1"
        or value["classification_input"] != "dated_question_text_only"
        or value["classifier_format"] != ROUTED_REPAIR_ROUTING_FORMAT
        or value["admission_basis"] != "isolated_positive_marginal_only"
        or value["construction_recall_claimed"] is not False
        or value["source_target_expansion_claimed"] is not False
    ):
        raise FactGateError("fact-gate route policy boundary changed")
    raw_admitted = value["admitted_routes"]
    if (
        type(raw_admitted) is not list
        or raw_admitted != ["numeric_reduce", "state_chain"]
    ):
        raise FactGateError("fact-gate admitted route set changed")
    assert_gold_blind(value, path="fact_gate_route_policy")
    return FactRoutePolicy(
        policy_id=require_text(value["policy_id"], "fact-gate policy ID"),
        sha256=observed,
        classifier_format=ROUTED_REPAIR_ROUTING_FORMAT,
        admitted_routes=frozenset(("numeric_reduce", "state_chain")),
    )


@dataclass(frozen=True, slots=True)
class FactGateResult:
    adapter_id: str
    question_id: str
    dated_question_sha256: str
    route_id: FactRoute
    route_admitted: bool
    route_policy_id: str
    route_policy_sha256: str
    route_reason: str
    route_receipt_sha256: str
    disposition: GateDisposition
    reason: str
    parent_prediction: str
    protected_evidence_ids: tuple[str, ...]
    selected_evidence_ids_before_dedup: tuple[str, ...]
    dedup_excluded_evidence_ids: tuple[str, ...]
    admitted_delta_evidence_ids: tuple[str, ...]
    facts: tuple[EMFact, ...]
    compression_receipt_sha256: str | None
    source_representation_messages_sha256: str | None
    prompt: EMFactAnswerPrompt | None

    @property
    def requires_provider_answer(self) -> bool:
        return self.disposition == "compiled"

    @property
    def fallback_prediction(self) -> str:
        """The exact sealed parent prediction for every closed gate."""

        return self.parent_prediction

    @property
    def raw_delta_rows_in_prompt(self) -> int:
        return 0

    def projection(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "admitted_delta_evidence_ids": list(
                self.admitted_delta_evidence_ids
            ),
            "compression_receipt_sha256": self.compression_receipt_sha256,
            "construction_recall_claimed": False,
            "dated_question_sha256": self.dated_question_sha256,
            "dedup_excluded_evidence_ids": list(
                self.dedup_excluded_evidence_ids
            ),
            "disposition": self.disposition,
            "fact_identities": [row.identity_payload() for row in self.facts],
            "format": FACT_GATE_FORMAT,
            "gold_loaded": False,
            "parent_prediction_sha256": quote_sha256(self.parent_prediction),
            "prompt_messages_sha256": (
                None if self.prompt is None else self.prompt.messages_sha256
            ),
            "provider_calls": 0,
            "question_id": self.question_id,
            "raw_delta_rows_in_prompt": 0,
            "reason": self.reason,
            "retained_request_token_state_bytes": 0,
            "route_admitted": self.route_admitted,
            "route_id": self.route_id,
            "route_policy_id": self.route_policy_id,
            "route_policy_sha256": self.route_policy_sha256,
            "route_reason": self.route_reason,
            "route_receipt_sha256": self.route_receipt_sha256,
            "selected_evidence_ids_before_dedup": list(
                self.selected_evidence_ids_before_dedup
            ),
            "source_representation_messages_sha256": (
                self.source_representation_messages_sha256
            ),
            "source_target_expansion_claimed": False,
            "protected_evidence_ids": list(self.protected_evidence_ids),
        }
        assert_gold_blind(body, path="fact_gate_result")
        body["receipt_sha256"] = identity_sha256(body)
        return body

    @property
    def receipt_sha256(self) -> str:
        return str(self.projection()["receipt_sha256"])


def _fallback(
    *,
    adapter_id: str,
    question_id: str,
    dated_question: str,
    route: FactRouteRule,
    route_receipt: RoutedRepairReceipt,
    policy: FactRoutePolicy,
    reason: str,
    parent_prediction: str,
    protected_ids: tuple[str, ...],
    selected_ids: tuple[str, ...],
    excluded_ids: tuple[str, ...],
    delta_ids: tuple[str, ...],
) -> FactGateResult:
    return FactGateResult(
        adapter_id=adapter_id,
        question_id=question_id,
        dated_question_sha256=quote_sha256(dated_question),
        route_id=route.route_id,
        route_admitted=route.admitted,
        route_policy_id=policy.policy_id,
        route_policy_sha256=policy.sha256,
        route_reason=route_receipt.reason.value,
        route_receipt_sha256=route_receipt.receipt_sha256,
        disposition="parent_fallback",
        reason=reason,
        parent_prediction=parent_prediction,
        protected_evidence_ids=protected_ids,
        selected_evidence_ids_before_dedup=selected_ids,
        dedup_excluded_evidence_ids=excluded_ids,
        admitted_delta_evidence_ids=delta_ids,
        facts=(),
        compression_receipt_sha256=None,
        source_representation_messages_sha256=None,
        prompt=None,
    )


def _protected_rows(value: Sequence[FastEvidence]) -> tuple[FastEvidence, ...]:
    rows = tuple(value)
    if any(type(row) is not FastEvidence for row in rows):
        raise FactGateError("fact gate requires exact protected FastEvidence rows")
    ids = tuple(row.evidence_id for row in rows)
    if len(set(ids)) != len(ids):
        raise FactGateError("verified protected evidence repeats an ID")
    if any(
        type(row.evidence_id) is not str
        or not row.evidence_id
        or type(row.source_id) is not str
        or not row.source_id
        or type(row.text) is not str
        for row in rows
    ):
        raise FactGateError("verified protected evidence changed shape")
    return rows


def _selected_rows(value: object) -> tuple[FastEvidence, ...] | None:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        rows = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if any(type(row) is not FastEvidence for row in rows):
        return None
    if any(
        type(row.evidence_id) is not str
        or not row.evidence_id
        or type(row.source_id) is not str
        or not row.source_id
        or type(row.text) is not str
        for row in rows
    ):
        return None
    ids = tuple(row.evidence_id for row in rows)
    if len(set(ids)) != len(ids):
        return None
    return rows


def _post_selection_delta(
    root: tuple[FastEvidence, ...],
    selected: tuple[FastEvidence, ...],
) -> tuple[tuple[FastEvidence, ...], tuple[str, ...]] | None:
    root_by_id = {row.evidence_id: row for row in root}
    root_coordinates = {(row.source_id, row.text) for row in root}
    seen_coordinates: set[tuple[str, str]] = set()
    novel: list[FastEvidence] = []
    excluded: list[str] = []
    for row in selected:
        protected = root_by_id.get(row.evidence_id)
        if protected is not None:
            if protected != row:
                return None
            excluded.append(row.evidence_id)
            continue
        coordinate = (row.source_id, row.text)
        if coordinate in root_coordinates or coordinate in seen_coordinates:
            excluded.append(row.evidence_id)
            continue
        seen_coordinates.add(coordinate)
        novel.append(row)
    return tuple(novel), tuple(excluded)


def _validated_facts(
    compression: EMFactCompression,
    delta: tuple[FastEvidence, ...],
    *,
    question_id: str,
) -> tuple[EMFact, ...] | None:
    delta_ids = tuple(row.evidence_id for row in delta)
    if (
        compression.question_id != question_id
        or compression.neighborhood_evidence_ids != delta_ids
        or not compression.facts
    ):
        return None
    by_id = {row.evidence_id: row for row in delta}
    result: list[EMFact] = []
    seen: set[tuple[Any, ...]] = set()
    for fact in compression.facts:
        if type(fact) is not EMFact:
            return None
        coordinates: list[tuple[str, str, str]] = []
        for citation in fact.citations:
            evidence = by_id.get(citation.evidence_id)
            if (
                evidence is None
                or evidence.source_id != citation.source_id
                or citation.quote not in evidence.text
                or quote_sha256(citation.quote) != citation.quote_sha256
            ):
                return None
            coordinates.append(
                (citation.evidence_id, citation.source_id, citation.quote)
            )
        identity = (fact.text, *coordinates)
        if identity not in seen:
            seen.add(identity)
            result.append(fact)
    return tuple(result)


def _question_view(
    question_id: str,
    dated_question: str,
    source_stage_id: str,
    root: tuple[FastEvidence, ...],
    delta: tuple[FastEvidence, ...],
) -> Any:
    return SimpleNamespace(
        question_id=question_id,
        dated_question=dated_question,
        stages=(
            SimpleNamespace(stage_id="fact_gate_protected_s0", evidence=root),
            SimpleNamespace(
                stage_id=source_stage_id,
                evidence=(*root, *delta),
            ),
        ),
    )


def _guarded_prompt(
    source: EMFactAnswerPrompt,
    *,
    parent_prediction: str,
) -> EMFactAnswerPrompt | None:
    messages = list(source.messages)
    if len(messages) != 3:
        return None
    messages[1] = FastProviderMessage(
        role=messages[1].role,
        content=(
            messages[1].content
            + "\n\nSealed parent answer (preserve by default; revise only "
            "when the exact-cited additions require it):\n"
            + parent_prediction
        ),
    )
    mappings = tuple(
        {"role": row.role, "content": row.content} for row in messages
    )
    tokens = count_chat_prompt_token_proxy(mappings)
    if tokens + source.responder_output_token_reserve > source.max_prompt_token_proxy:
        return None
    return replace(
        source,
        messages=tuple(messages),
        prompt_token_proxy=tokens,
        messages_sha256=identity_sha256(list(mappings)),
    )


def _compression_response(facts: Sequence[EMFact]) -> str:
    """Project validated typed facts into the existing public parser format."""

    return json.dumps(
        {
            "facts": [
                {
                    "citations": [
                        {
                            "evidence_alias": citation.evidence_alias,
                            "quote": citation.quote,
                        }
                        for citation in fact.citations
                    ],
                    "text": fact.text,
                }
                for fact in facts
            ]
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compile_admitted_fact_delta(
    *,
    adapter_id: str,
    question_id: str,
    dated_question: str,
    parent_prediction: str,
    protected_evidence: Sequence[FastEvidence],
    selected_evidence_before_dedup: object,
    compression: EMFactCompression | None,
    compression_invalid: bool = False,
    route_policy: FactRoutePolicy | None = None,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    responder_output_token_reserve: int = ANSWER_OUTPUT_TOKEN_RESERVE,
) -> FactGateResult:
    """Compile one selected delta, or close the gate to the exact parent."""

    require_text(adapter_id, "fact-gate adapter ID")
    require_text(question_id, "fact-gate question ID")
    require_text(dated_question, "fact-gate dated question")
    require_text(parent_prediction, "fact-gate parent prediction")
    policy = route_policy or load_fact_route_policy()
    route, route_receipt = policy.classify(dated_question)
    root = _protected_rows(protected_evidence)
    protected_ids = tuple(row.evidence_id for row in root)
    selected = _selected_rows(selected_evidence_before_dedup)
    if selected is None:
        return _fallback(
            adapter_id=adapter_id,
            question_id=question_id,
            dated_question=dated_question,
            route=route,
            route_receipt=route_receipt,
            policy=policy,
            reason="invalid_selected_delta",
            parent_prediction=parent_prediction,
            protected_ids=protected_ids,
            selected_ids=(),
            excluded_ids=(),
            delta_ids=(),
        )
    selected_ids = tuple(row.evidence_id for row in selected)
    projected = _post_selection_delta(root, selected)
    if projected is None:
        return _fallback(
            adapter_id=adapter_id,
            question_id=question_id,
            dated_question=dated_question,
            route=route,
            route_receipt=route_receipt,
            policy=policy,
            reason="invalid_selected_delta",
            parent_prediction=parent_prediction,
            protected_ids=protected_ids,
            selected_ids=selected_ids,
            excluded_ids=(),
            delta_ids=(),
        )
    delta, excluded_ids = projected
    delta_ids = tuple(row.evidence_id for row in delta)
    fallback_args = {
        "adapter_id": adapter_id,
        "question_id": question_id,
        "dated_question": dated_question,
        "route": route,
        "route_receipt": route_receipt,
        "policy": policy,
        "parent_prediction": parent_prediction,
        "protected_ids": protected_ids,
        "selected_ids": selected_ids,
        "excluded_ids": excluded_ids,
        "delta_ids": delta_ids,
    }
    if not delta:
        return _fallback(reason="empty_or_non_novel_delta", **fallback_args)
    if not route.admitted:
        return _fallback(reason="question_route_not_admitted", **fallback_args)
    if compression is None:
        return _fallback(
            reason=(
                "invalid_fact_compression"
                if compression_invalid
                else "empty_fact_compression"
            ),
            **fallback_args,
        )
    if type(compression) is not EMFactCompression:
        return _fallback(reason="invalid_fact_compression", **fallback_args)
    facts = _validated_facts(compression, delta, question_id=question_id)
    if not facts:
        return _fallback(reason="empty_or_invalid_cited_facts", **fallback_args)
    if (
        route_receipt.style is RoutedRepairStyle.NUMERIC_REDUCE
        and not numeric_facts_are_quote_grounded(facts)
    ):
        return _fallback(reason="unsupported_numeric_fact", **fallback_args)
    view = _question_view(
        question_id,
        dated_question,
        compression.source_stage_id,
        root,
        delta,
    )
    for count in range(len(facts), 0, -1):
        bounded = EMFactCompression(
            question_id=question_id,
            source_stage_id=compression.source_stage_id,
            neighborhood_evidence_ids=delta_ids,
            facts=facts[:count],
            response_sha256=compression.response_sha256,
        )
        try:
            routed = build_routed_answer_prompt(
                view,
                _compression_response(bounded.facts),
                route_receipt,
                stage_id=bounded.source_stage_id,
                measured_arm="facts",
                max_prompt_tokens=max_prompt_tokens,
                responder_output_token_reserve=responder_output_token_reserve,
                max_facts=MAX_FACTS,
            )
        except (EMFactMemoryError, RoutedRepairPromptError):
            continue
        if routed.fallback_reason is not None:
            return _fallback(reason="invalid_routed_fact_prompt", **fallback_args)
        source_prompt = routed.prompt
        if not source_prompt.fact_ids:
            continue
        prompt = _guarded_prompt(
            source_prompt,
            parent_prediction=parent_prediction,
        )
        if prompt is None:
            continue
        packed = tuple(
            fact for fact in facts if fact.fact_id in set(prompt.fact_ids)
        )
        return FactGateResult(
            adapter_id=adapter_id,
            question_id=question_id,
            dated_question_sha256=quote_sha256(dated_question),
            route_id=route.route_id,
            route_admitted=True,
            route_policy_id=policy.policy_id,
            route_policy_sha256=policy.sha256,
            route_reason=route_receipt.reason.value,
            route_receipt_sha256=route_receipt.receipt_sha256,
            disposition="compiled",
            reason="positive_cell_exact_cited_fact_delta",
            parent_prediction=parent_prediction,
            protected_evidence_ids=protected_ids,
            selected_evidence_ids_before_dedup=selected_ids,
            dedup_excluded_evidence_ids=excluded_ids,
            admitted_delta_evidence_ids=delta_ids,
            facts=packed,
            compression_receipt_sha256=bounded.receipt_sha256,
            source_representation_messages_sha256=(
                source_prompt.messages_sha256
            ),
            prompt=prompt,
        )
    return _fallback(reason="fact_prompt_overflow", **fallback_args)


def compile_fixed_s1_em_fact_gate(
    question: Any,
    *,
    parent_prediction: str,
    compression_response: object,
    route_policy: FactRoutePolicy | None = None,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    responder_output_token_reserve: int = ANSWER_OUTPUT_TOKEN_RESERVE,
) -> FactGateResult:
    """Adapt fixed S1 selection and its existing fact compression to the gate."""

    try:
        root = tuple(question.stages[0].evidence)
        stage = next(
            row for row in question.stages if row.stage_id == DEFAULT_EM_STAGE_ID
        )
        selected = tuple(stage.evidence)
    except (AttributeError, IndexError, StopIteration, TypeError) as exc:
        raise FactGateError("fixed-S1 adapter received an invalid question") from exc
    compression: EMFactCompression | None = None
    invalid = False
    if type(compression_response) is str:
        try:
            compression = parse_fact_compression(
                question,
                compression_response,
                stage_id=DEFAULT_EM_STAGE_ID,
                max_facts=MAX_FACTS,
            )
        except EMFactMemoryError:
            invalid = True
    else:
        invalid = True
    return compile_admitted_fact_delta(
        adapter_id=FIXED_S1_ADAPTER_ID,
        question_id=str(question.question_id),
        dated_question=str(question.dated_question),
        parent_prediction=parent_prediction,
        protected_evidence=root,
        selected_evidence_before_dedup=selected,
        compression=compression,
        compression_invalid=invalid,
        route_policy=route_policy,
        max_prompt_tokens=max_prompt_tokens,
        responder_output_token_reserve=responder_output_token_reserve,
    )


def compile_closure_v9_fact_gate(
    *,
    question: Any,
    arm_label: str,
    parent: Any,
    compression: EMFactCompression | None = None,
    route_policy: FactRoutePolicy | None = None,
) -> FactGateResult:
    """Adapt a verified closure question; absent cited facts close the gate."""

    from . import closure, live

    if type(question) is not closure.IndependentClosureQuestion or type(
        parent
    ) is not live.VerifiedS0V2AnswerRow:
        raise FactGateError("closure fact gate requires verified typed inputs")
    if (
        parent.question_id != question.question_id
        or parent.question_sha256 != question.question_sha256
        or parent.dated_question_sha256 != question.dated_question_sha256
        or arm_label not in closure.ARM_LABELS
    ):
        raise FactGateError("closure fact-gate parent binding changed")
    arm = question.arm(arm_label)
    root = tuple(
        FastEvidence(row.evidence_id, row.source_id, row.text)
        for row in question.root_protected_evidence
    )
    additions = () if arm is None else tuple(
        FastEvidence(row.evidence_id, row.source_id, row.text)
        for row in arm.admitted_atoms
    )
    return compile_admitted_fact_delta(
        adapter_id=CLOSURE_V9_ADAPTER_ID,
        question_id=question.question_id,
        dated_question=question.dated_question,
        parent_prediction=parent.prediction,
        protected_evidence=root,
        selected_evidence_before_dedup=(*root, *additions),
        compression=compression,
        route_policy=route_policy,
    )


__all__ = [
    "ANSWER_OUTPUT_TOKEN_RESERVE",
    "CLOSURE_V9_ADAPTER_ID",
    "FACT_GATE_FORMAT",
    "FIXED_S1_ADAPTER_ID",
    "FactGateError",
    "FactGateResult",
    "FactRoutePolicy",
    "FactRouteRule",
    "MAX_PROMPT_TOKENS",
    "ROUTE_POLICY_PATH",
    "ROUTE_POLICY_SHA256",
    "compile_admitted_fact_delta",
    "compile_closure_v9_fact_gate",
    "compile_fixed_s1_em_fact_gate",
    "load_fact_route_policy",
]
