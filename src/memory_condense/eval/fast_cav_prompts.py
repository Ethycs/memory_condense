"""Provider-free matched QA prompts from exact fast-retrieval evidence rows.

This module is the tensor-free seam between a bounded ordering readout and the
provider runtime.  It never imports a router or tensor library.  Each original,
base, and treatment arm renders the same exact ``FastEvidence`` rows with the
same stable aliases; only catalog row order may differ.

The original arm is a newly rendered canonical-catalog control.  It is not the
retrieval artifact's historical provider prompt, whose context framing may be
more verbose and therefore would not be matched to the reordered arms.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.benchmark import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    STAGE_IDS,
    FastEvidence,
    FastProviderMessage,
    FastRetrievalArtifact,
)


FAST_CAV_PROMPT_POPULATION_FORMAT = "memory-condense-fast-cav-prompts-v1"
FAST_CAV_STAGE_RECEIPT_FORMAT = "memory-condense-fast-cav-stage-prompt-v1"
FAST_CAV_ARM_RECEIPT_FORMAT = "memory-condense-fast-cav-arm-prompt-v1"
FAST_CAV_ORDER_INPUT_FORMAT = "memory-condense-tensor-free-stage-order-v1"
FAST_CAV_ORDER_FORMAT = "memory-condense-fast-cav-evidence-order-v1"
FAST_CAV_ALIAS_FORMAT = "memory-condense-fast-cav-alias-bindings-v1"
FAST_CAV_MEMBERSHIP_FORMAT = "memory-condense-fast-cav-evidence-membership-v1"
ORIGINAL_CONTROL_KIND = (
    "canonical_evidence_catalog_original_order_not_artifact_provider_prompt"
)
ARM_IDS = ("original", "base", "treatment")
ABSOLUTE_MAX_PROMPT_TOKENS = 8_000

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_HEADER = (
    "Canonical evidence catalog (matched arms; exact retrieval rows):"
)


def _nonempty(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be an exact non-empty string")
    return value


def _digest(value: Any, *, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _ids(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of evidence IDs")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence of evidence IDs") from exc
    if not result:
        raise ValueError(f"{label} must be non-empty")
    for item in result:
        _nonempty(item, label=f"{label} item")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique evidence IDs")
    return result


def _order_sha256(
    evidence_ids: tuple[str, ...],
    alias_order: tuple[str, ...],
) -> str:
    return identity_sha256(
        {
            "format": FAST_CAV_ORDER_FORMAT,
            "evidence_ids": list(evidence_ids),
            "alias_order": list(alias_order),
        }
    )


@dataclass(frozen=True, slots=True)
class TensorFreeStageOrder:
    """Minimal tensor-free input projected from one per-stage readout."""

    question_id: str
    stage_id: str
    original_evidence_ids: tuple[str, ...]
    base_evidence_ids: tuple[str, ...]
    treatment_evidence_ids: tuple[str, ...]
    upstream_receipt_sha256: str
    retained_tensor_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "question_id", _nonempty(self.question_id, label="question_id")
        )
        object.__setattr__(
            self, "stage_id", _nonempty(self.stage_id, label="stage_id")
        )
        original = _ids(
            self.original_evidence_ids,
            label="original_evidence_ids",
        )
        base = _ids(self.base_evidence_ids, label="base_evidence_ids")
        treatment = _ids(
            self.treatment_evidence_ids,
            label="treatment_evidence_ids",
        )
        object.__setattr__(self, "original_evidence_ids", original)
        object.__setattr__(self, "base_evidence_ids", base)
        object.__setattr__(self, "treatment_evidence_ids", treatment)
        if len(base) != len(original) or set(base) != set(original):
            raise ValueError("base_evidence_ids must preserve the exact evidence set")
        if len(treatment) != len(original) or set(treatment) != set(original):
            raise ValueError(
                "treatment_evidence_ids must preserve the exact evidence set"
            )
        object.__setattr__(
            self,
            "upstream_receipt_sha256",
            _digest(
                self.upstream_receipt_sha256,
                label="upstream_receipt_sha256",
            ),
        )
        if type(self.retained_tensor_bytes) is not int or (
            self.retained_tensor_bytes != 0
        ):
            raise ValueError("stage order input must retain zero tensor bytes")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "format": FAST_CAV_ORDER_INPUT_FORMAT,
            "question_id": self.question_id,
            "stage_id": self.stage_id,
            "original_evidence_ids": list(self.original_evidence_ids),
            "base_evidence_ids": list(self.base_evidence_ids),
            "treatment_evidence_ids": list(self.treatment_evidence_ids),
            "upstream_receipt_sha256": self.upstream_receipt_sha256,
            "retained_tensor_bytes": self.retained_tensor_bytes,
        }

    @property
    def order_input_sha256(self) -> str:
        return identity_sha256(self.identity_payload())


@dataclass(frozen=True, slots=True)
class FastCAVAliasBinding:
    alias: str
    evidence_id: str
    source_id: str
    text_sha256: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FastCAVUniquePrompt:
    unique_prompt_ordinal: int
    messages_sha256: str
    context_sha256: str
    prompt_token_proxy: int
    messages: tuple[FastProviderMessage, ...]
    context: str

    def as_mappings(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"role": message.role, "content": message.content}
            for message in self.messages
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "unique_prompt_ordinal": self.unique_prompt_ordinal,
            "messages_sha256": self.messages_sha256,
            "context_sha256": self.context_sha256,
            "prompt_token_proxy": self.prompt_token_proxy,
        }


@dataclass(frozen=True, slots=True)
class FastCAVArmPrompt:
    logical_ordinal: int
    question_ordinal: int
    question_id: str
    stage_id: str
    arm_id: str
    evidence_ids: tuple[str, ...]
    alias_order: tuple[str, ...]
    evidence_order_sha256: str
    context_sha256: str
    messages_sha256: str
    prompt_token_proxy: int
    hard_prompt_token_cap: int
    unique_prompt_ordinal: int
    arm_prompt_sha256: str

    def identity_payload(self, *, include_sha256: bool = True) -> dict[str, Any]:
        result = {
            "format": FAST_CAV_ARM_RECEIPT_FORMAT,
            "logical_ordinal": self.logical_ordinal,
            "question_ordinal": self.question_ordinal,
            "question_id": self.question_id,
            "stage_id": self.stage_id,
            "arm_id": self.arm_id,
            "evidence_ids": list(self.evidence_ids),
            "alias_order": list(self.alias_order),
            "evidence_order_sha256": self.evidence_order_sha256,
            "context_sha256": self.context_sha256,
            "messages_sha256": self.messages_sha256,
            "prompt_token_proxy": self.prompt_token_proxy,
            "hard_prompt_token_cap": self.hard_prompt_token_cap,
            "unique_prompt_ordinal": self.unique_prompt_ordinal,
        }
        if include_sha256:
            result["arm_prompt_sha256"] = self.arm_prompt_sha256
        return result


@dataclass(frozen=True, slots=True)
class FastCAVStagePromptReceipt:
    question_ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    stage_id: str
    stage_receipt_sha256: str
    artifact_sha256: str
    order_input_sha256: str
    upstream_order_receipt_sha256: str
    alias_bindings: tuple[FastCAVAliasBinding, ...]
    alias_bindings_sha256: str
    evidence_membership_sha256: str
    original_control_kind: str
    arm_prompt_sha256s: tuple[str, ...]
    retained_tensor_bytes: int
    receipt_sha256: str

    def identity_payload(self, *, include_sha256: bool = True) -> dict[str, Any]:
        result = {
            "format": FAST_CAV_STAGE_RECEIPT_FORMAT,
            "question_ordinal": self.question_ordinal,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "dated_question_sha256": self.dated_question_sha256,
            "stage_id": self.stage_id,
            "stage_receipt_sha256": self.stage_receipt_sha256,
            "artifact_sha256": self.artifact_sha256,
            "order_input_sha256": self.order_input_sha256,
            "upstream_order_receipt_sha256": (
                self.upstream_order_receipt_sha256
            ),
            "alias_bindings": [row.model_dump() for row in self.alias_bindings],
            "alias_bindings_sha256": self.alias_bindings_sha256,
            "evidence_membership_sha256": self.evidence_membership_sha256,
            "original_control_kind": self.original_control_kind,
            "arm_prompt_sha256s": list(self.arm_prompt_sha256s),
            "retained_tensor_bytes": self.retained_tensor_bytes,
        }
        if include_sha256:
            result["receipt_sha256"] = self.receipt_sha256
        return result


@dataclass(frozen=True, slots=True)
class FastCAVPromptPopulation:
    format: str
    artifact_sha256: str
    selected_stage_ids: tuple[str, ...]
    logical_prompt_count: int
    unique_prompt_count: int
    logical_prompts: tuple[FastCAVArmPrompt, ...]
    unique_prompts: tuple[FastCAVUniquePrompt, ...]
    stage_receipts: tuple[FastCAVStagePromptReceipt, ...]
    prompt_population_sha256: str
    retained_tensor_bytes: int = 0

    @property
    def logical_message_population(self) -> tuple[tuple[dict[str, str], ...], ...]:
        return tuple(
            self.unique_prompts[row.unique_prompt_ordinal].as_mappings()
            for row in self.logical_prompts
        )

    def identity_payload(self, *, include_sha256: bool = True) -> dict[str, Any]:
        result = {
            "format": self.format,
            "artifact_sha256": self.artifact_sha256,
            "selected_stage_ids": list(self.selected_stage_ids),
            "logical_prompt_count": self.logical_prompt_count,
            "unique_prompt_count": self.unique_prompt_count,
            "logical_prompts": [
                row.identity_payload() for row in self.logical_prompts
            ],
            "unique_prompts": [
                row.identity_payload() for row in self.unique_prompts
            ],
            "stage_receipt_sha256s": [
                row.receipt_sha256 for row in self.stage_receipts
            ],
            "retained_tensor_bytes": self.retained_tensor_bytes,
        }
        if include_sha256:
            result["prompt_population_sha256"] = self.prompt_population_sha256
        return result


def _selected_stages(
    artifact: FastRetrievalArtifact,
    stage_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if tuple(artifact.stage_ids) != STAGE_IDS:
        raise ValueError("fast retrieval artifact changed the canonical stage IDs")
    if stage_ids is None:
        return tuple(artifact.stage_ids)
    if isinstance(stage_ids, (str, bytes, bytearray)):
        raise TypeError("stage_ids must be a sequence")
    selected = tuple(stage_ids)
    if not selected or any(type(item) is not str for item in selected):
        raise ValueError("stage_ids must contain exact stage IDs")
    if len(selected) != len(set(selected)):
        raise ValueError("stage_ids must be unique")
    canonical = tuple(item for item in artifact.stage_ids if item in set(selected))
    if selected != canonical:
        raise ValueError("stage_ids must preserve canonical artifact order")
    return selected


def _catalog(
    evidence_ids: tuple[str, ...],
    *,
    evidence_by_id: Mapping[str, FastEvidence],
    alias_by_id: Mapping[str, str],
) -> str:
    rows: list[str] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id[evidence_id]
        source_id = json.dumps(
            evidence.source_id,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append(
            f"[{alias_by_id[evidence_id]}] source_id={source_id}\n{evidence.text}"
        )
    return _CATALOG_HEADER + "\n\n" + "\n\n".join(rows)


def _messages(context: str, dated_question: str) -> tuple[FastProviderMessage, ...]:
    return (
        FastProviderMessage(role="system", content=QA_SYSTEM_PROMPT),
        FastProviderMessage(
            role="user",
            content=QA_USER_TEMPLATE.format(
                context=context,
                question=dated_question,
            ),
        ),
    )


def _message_mappings(
    messages: tuple[FastProviderMessage, ...],
) -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in messages
    ]


def build_fast_cav_prompt_population(
    artifact: FastRetrievalArtifact,
    stage_orders: Sequence[TensorFreeStageOrder],
    *,
    stage_ids: Sequence[str] | None = None,
) -> FastCAVPromptPopulation:
    """Render matched original/base/treatment prompts without I/O or tensors."""

    if type(artifact) is not FastRetrievalArtifact:
        raise TypeError("artifact must be an exact FastRetrievalArtifact")
    _digest(artifact.raw_sha256, label="artifact.raw_sha256")
    if artifact.retained_request_token_state_bytes != 0:
        raise ValueError("artifact retained transformer request state")
    selected_stage_ids = _selected_stages(artifact, stage_ids)
    if isinstance(stage_orders, (str, bytes, bytearray)):
        raise TypeError("stage_orders must be a sequence")

    order_by_key: dict[tuple[str, str], TensorFreeStageOrder] = {}
    for index, order in enumerate(stage_orders):
        if type(order) is not TensorFreeStageOrder:
            raise TypeError(
                f"stage_orders[{index}] must be an exact TensorFreeStageOrder"
            )
        key = (order.question_id, order.stage_id)
        if key in order_by_key:
            raise ValueError(f"duplicate stage order input: {key!r}")
        order_by_key[key] = order

    expected_keys = {
        (question.question_id, stage_id)
        for question in artifact.questions
        for stage_id in selected_stage_ids
    }
    observed_keys = set(order_by_key)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            "stage order inputs do not exactly cover the selected population; "
            f"missing={missing!r} extra={extra!r}"
        )

    unique_prompts: list[FastCAVUniquePrompt] = []
    unique_by_messages_sha: dict[
        str, tuple[int, tuple[FastProviderMessage, ...]]
    ] = {}
    logical_prompts: list[FastCAVArmPrompt] = []
    stage_receipts: list[FastCAVStagePromptReceipt] = []
    observed_question_ids: set[str] = set()

    for question in artifact.questions:
        if question.question_id in observed_question_ids:
            raise ValueError("artifact contains duplicate question IDs")
        observed_question_ids.add(question.question_id)
        if question.retained_request_token_state_bytes != 0:
            raise ValueError("question retained transformer request state")
        if quote_sha256(question.question) != question.question_sha256:
            raise ValueError("question text no longer matches question_sha256")
        if quote_sha256(question.dated_question) != question.dated_question_sha256:
            raise ValueError(
                "dated question text no longer matches dated_question_sha256"
            )

        for stage_id in selected_stage_ids:
            stage = question.stage(stage_id)
            order = order_by_key[(question.question_id, stage_id)]
            if order.original_evidence_ids != stage.evidence_ids:
                raise ValueError(
                    f"{question.question_id}/{stage_id} original evidence order "
                    "does not match the exact artifact stage"
                )
            if stage.max_prompt_token_proxy < 1:
                raise ValueError("stage hard prompt-token cap must be positive")
            hard_cap = min(
                stage.max_prompt_token_proxy,
                ABSOLUTE_MAX_PROMPT_TOKENS,
            )
            evidence_by_id = {row.evidence_id: row for row in stage.evidence}
            if len(evidence_by_id) != len(stage.evidence):
                raise ValueError("artifact stage contains duplicate evidence IDs")
            aliases = tuple(
                FastCAVAliasBinding(
                    alias=f"E{index:03d}",
                    evidence_id=row.evidence_id,
                    source_id=row.source_id,
                    text_sha256=quote_sha256(row.text),
                )
                for index, row in enumerate(stage.evidence, start=1)
            )
            alias_by_id = {row.evidence_id: row.alias for row in aliases}
            alias_payload = [row.model_dump() for row in aliases]
            alias_bindings_sha = identity_sha256(
                {"format": FAST_CAV_ALIAS_FORMAT, "bindings": alias_payload}
            )
            membership_sha = identity_sha256(
                {
                    "format": FAST_CAV_MEMBERSHIP_FORMAT,
                    "evidence": sorted(
                        alias_payload,
                        key=lambda row: str(row["evidence_id"]),
                    ),
                }
            )

            arm_sha256s: list[str] = []
            for arm_id, evidence_ids in zip(
                ARM_IDS,
                (
                    order.original_evidence_ids,
                    order.base_evidence_ids,
                    order.treatment_evidence_ids,
                ),
                strict=True,
            ):
                alias_order = tuple(alias_by_id[item] for item in evidence_ids)
                order_sha = _order_sha256(evidence_ids, alias_order)
                context = _catalog(
                    evidence_ids,
                    evidence_by_id=evidence_by_id,
                    alias_by_id=alias_by_id,
                )
                messages = _messages(context, question.dated_question)
                message_mappings = _message_mappings(messages)
                context_sha = quote_sha256(context)
                messages_sha = identity_sha256(message_mappings)
                prompt_tokens = count_chat_prompt_token_proxy(message_mappings)
                if type(prompt_tokens) is not int or prompt_tokens < 1:
                    raise ValueError("prompt token counter must return a positive int")
                if prompt_tokens > hard_cap:
                    raise ValueError(
                        "canonical CAV prompt exceeds the hard prompt-token cap "
                        f"for {question.question_id}/{stage_id}/{arm_id}: "
                        f"{prompt_tokens} > {hard_cap}"
                    )

                existing = unique_by_messages_sha.get(messages_sha)
                if existing is None:
                    unique_ordinal = len(unique_prompts)
                    unique_by_messages_sha[messages_sha] = (
                        unique_ordinal,
                        messages,
                    )
                    unique_prompts.append(
                        FastCAVUniquePrompt(
                            unique_prompt_ordinal=unique_ordinal,
                            messages_sha256=messages_sha,
                            context_sha256=context_sha,
                            prompt_token_proxy=prompt_tokens,
                            messages=messages,
                            context=context,
                        )
                    )
                else:
                    unique_ordinal, previous_messages = existing
                    if previous_messages != messages:
                        raise RuntimeError("prompt message SHA-256 collision")
                    previous = unique_prompts[unique_ordinal]
                    if (
                        previous.context_sha256 != context_sha
                        or previous.prompt_token_proxy != prompt_tokens
                    ):
                        raise RuntimeError("identical prompt hash changed metadata")

                logical_ordinal = len(logical_prompts)
                arm_body = {
                    "format": FAST_CAV_ARM_RECEIPT_FORMAT,
                    "logical_ordinal": logical_ordinal,
                    "question_ordinal": question.ordinal,
                    "question_id": question.question_id,
                    "stage_id": stage_id,
                    "arm_id": arm_id,
                    "evidence_ids": list(evidence_ids),
                    "alias_order": list(alias_order),
                    "evidence_order_sha256": order_sha,
                    "context_sha256": context_sha,
                    "messages_sha256": messages_sha,
                    "prompt_token_proxy": prompt_tokens,
                    "hard_prompt_token_cap": hard_cap,
                    "unique_prompt_ordinal": unique_ordinal,
                }
                arm_sha = identity_sha256(arm_body)
                logical_prompts.append(
                    FastCAVArmPrompt(
                        logical_ordinal=logical_ordinal,
                        question_ordinal=question.ordinal,
                        question_id=question.question_id,
                        stage_id=stage_id,
                        arm_id=arm_id,
                        evidence_ids=evidence_ids,
                        alias_order=alias_order,
                        evidence_order_sha256=order_sha,
                        context_sha256=context_sha,
                        messages_sha256=messages_sha,
                        prompt_token_proxy=prompt_tokens,
                        hard_prompt_token_cap=hard_cap,
                        unique_prompt_ordinal=unique_ordinal,
                        arm_prompt_sha256=arm_sha,
                    )
                )
                arm_sha256s.append(arm_sha)

            stage_body = {
                "format": FAST_CAV_STAGE_RECEIPT_FORMAT,
                "question_ordinal": question.ordinal,
                "question_id": question.question_id,
                "question_sha256": question.question_sha256,
                "dated_question_sha256": question.dated_question_sha256,
                "stage_id": stage_id,
                "stage_receipt_sha256": stage.stage_receipt_sha256,
                "artifact_sha256": artifact.raw_sha256,
                "order_input_sha256": order.order_input_sha256,
                "upstream_order_receipt_sha256": order.upstream_receipt_sha256,
                "alias_bindings": alias_payload,
                "alias_bindings_sha256": alias_bindings_sha,
                "evidence_membership_sha256": membership_sha,
                "original_control_kind": ORIGINAL_CONTROL_KIND,
                "arm_prompt_sha256s": arm_sha256s,
                "retained_tensor_bytes": 0,
            }
            stage_receipts.append(
                FastCAVStagePromptReceipt(
                    question_ordinal=question.ordinal,
                    question_id=question.question_id,
                    question_sha256=question.question_sha256,
                    dated_question_sha256=question.dated_question_sha256,
                    stage_id=stage_id,
                    stage_receipt_sha256=stage.stage_receipt_sha256,
                    artifact_sha256=artifact.raw_sha256,
                    order_input_sha256=order.order_input_sha256,
                    upstream_order_receipt_sha256=(
                        order.upstream_receipt_sha256
                    ),
                    alias_bindings=aliases,
                    alias_bindings_sha256=alias_bindings_sha,
                    evidence_membership_sha256=membership_sha,
                    original_control_kind=ORIGINAL_CONTROL_KIND,
                    arm_prompt_sha256s=tuple(arm_sha256s),
                    retained_tensor_bytes=0,
                    receipt_sha256=identity_sha256(stage_body),
                )
            )

    population_body = {
        "format": FAST_CAV_PROMPT_POPULATION_FORMAT,
        "artifact_sha256": artifact.raw_sha256,
        "selected_stage_ids": list(selected_stage_ids),
        "logical_prompt_count": len(logical_prompts),
        "unique_prompt_count": len(unique_prompts),
        "logical_prompts": [row.identity_payload() for row in logical_prompts],
        "unique_prompts": [row.identity_payload() for row in unique_prompts],
        "stage_receipt_sha256s": [
            row.receipt_sha256 for row in stage_receipts
        ],
        "retained_tensor_bytes": 0,
    }
    return FastCAVPromptPopulation(
        format=FAST_CAV_PROMPT_POPULATION_FORMAT,
        artifact_sha256=artifact.raw_sha256,
        selected_stage_ids=selected_stage_ids,
        logical_prompt_count=len(logical_prompts),
        unique_prompt_count=len(unique_prompts),
        logical_prompts=tuple(logical_prompts),
        unique_prompts=tuple(unique_prompts),
        stage_receipts=tuple(stage_receipts),
        prompt_population_sha256=identity_sha256(population_body),
        retained_tensor_bytes=0,
    )


__all__ = [
    "ABSOLUTE_MAX_PROMPT_TOKENS",
    "ARM_IDS",
    "FAST_CAV_ALIAS_FORMAT",
    "FAST_CAV_ARM_RECEIPT_FORMAT",
    "FAST_CAV_MEMBERSHIP_FORMAT",
    "FAST_CAV_ORDER_FORMAT",
    "FAST_CAV_ORDER_INPUT_FORMAT",
    "FAST_CAV_PROMPT_POPULATION_FORMAT",
    "FAST_CAV_STAGE_RECEIPT_FORMAT",
    "ORIGINAL_CONTROL_KIND",
    "FastCAVAliasBinding",
    "FastCAVArmPrompt",
    "FastCAVPromptPopulation",
    "FastCAVStagePromptReceipt",
    "FastCAVUniquePrompt",
    "TensorFreeStageOrder",
    "build_fast_cav_prompt_population",
]
