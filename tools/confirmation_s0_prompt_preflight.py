#!/usr/bin/env python3
"""Population-neutral confirmation S0 adapter and inert Terra preflight.

This module consumes the authoritative sealed confirmation cumulative merge.
It accepts arbitrary ordered populations, authenticates namespace/source
isolation and cumulative S0--S3 receipt chains, reconstructs the protected S0
``MemoryPacket``, and verifies that the supplied Terra messages are exactly the
common V4 renderer output.

Only provider-free ``compile`` and ``replay`` operations exist.  Runtime model,
prompt/output budgets, concurrency, gateway, and retry count are read from the
sealed policy-v5-r3 treatment projection.  The preflight reports the exact
would-call population but cannot perform a call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools.confirmation_contracts import (  # noqa: E402
    RuntimePolicy,
    SealedJson,
    _decode_treatment,
    _verify_preflight,
    publish_sealed_json,
    read_runtime_policy,
    read_sealed_json,
)
from tools.confirmation_cumulative_retrieval import (  # noqa: E402
    EVIDENCE_FORMAT,
    MERGED_ROW_FORMAT,
    POPULATION_IDENTITY_FORMAT,
    QUESTION_FORMAT,
    SOURCE_STAGE_ID,
    STAGE_FORMAT,
    STAGE_IDS,
)
from memory_condense.eval.recall_guarded_cumulative import (  # noqa: E402
    CUMULATIVE_STAGE_FORMAT,
    CumulativeRetrievalStageReceipt,
)
from tools.matched_eval.contracts import (  # noqa: E402
    EvidenceItem,
    MatchedEvalContractError,
    MemoryPacket,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.renderer import (  # noqa: E402
    RenderedPrompt,
    V4_RENDERER_ID,
    render_memory_packet_for_id,
)
from tools.confirmation_canonical import (  # noqa: E402
    assert_snapshot_unchanged,
    exact_keys,
    require_int,
    require_list,
    require_mapping,
)


CUMULATIVE_RETRIEVAL_FORMAT = "memory-condense-confirmation-cumulative-merged-v1"
CUMULATIVE_ROW_FORMAT = MERGED_ROW_FORMAT
CUMULATIVE_STAGE_RECEIPT_FORMAT = CUMULATIVE_STAGE_FORMAT
PREFLIGHT_FORMAT = "memory-condense-confirmation-matched-s0-terra-preflight-v1"
PREFLIGHT_ROW_FORMAT = f"{PREFLIGHT_FORMAT}-row-v1"
PREFLIGHT_PROVIDER_INPUT_FORMAT = (
    "memory-condense-confirmation-terra-provider-input-v1"
)
CUMULATIVE_STAGE_IDS = STAGE_IDS

_CUMULATIVE_KEYS = {
    "backend_identity_sha256",
    "format",
    "freeze_sha256",
    "gold_loaded",
    "namespace_checkpoints",
    "namespace_count",
    "physical_provider_calls",
    "population_identity",
    "population_identity_sha256",
    "preflight_sha256",
    "question_count",
    "question_order_sha256",
    "question_receipt_sha256s",
    "questions",
    "stage_ids",
    "workset_identity_sha256",
    "merge_receipt_sha256",
}
_ROW_KEYS = {
    "format",
    "namespace_checkpoint_sha256",
    "namespace_id",
    "namespace_store_id",
    "question",
    "source_question_receipt_sha256",
}
_NAMESPACE_CHECKPOINT_KEYS = {
    "checkpoint_receipt_sha256",
    "checkpoint_sha256",
    "namespace_id",
    "namespace_store_id",
    "namespace_work_receipt_sha256",
}
_POPULATION_IDENTITY_KEYS = {
    "dataset_sha256",
    "format",
    "namespace_store_ids",
    "ordered_row_receipt_sha256s",
    "preflight_sha256",
    "sanitized_projection_sha256",
    "split_manifest_sha256",
    "workset_identity_sha256",
    "population_identity_sha256",
}
_QUESTION_KEYS = {
    "base_retrieval_receipt_sha256",
    "content_binding_sha256",
    "dated_question",
    "dated_question_sha256",
    "format",
    "physical_provider_calls",
    "predecessor_receipt",
    "question",
    "question_id",
    "question_id_sha256",
    "question_receipt_sha256",
    "question_sha256",
    "retrieval_receipt",
    "row_receipt_sha256",
    "stage_ids",
    "stages",
}
_EVIDENCE_KEYS = {"evidence_id", "format", "source_id", "text"}
_STAGE_KEYS = {
    "evidence",
    "format",
    "stage_id",
    "provider_messages",
    "stage_receipt",
}
_RUNTIME_KEYS = {
    "gateway_url",
    "hard_complete_chat_token_cap",
    "input_token_cap",
    "max_concurrency",
    "model",
    "output_token_reserve",
    "retry_count",
}


class ConfirmationS0PreflightError(ValueError):
    """The generic S0 boundary or inert execution plan failed closed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationS0PreflightError(message)


def _text(value: object, label: str) -> str:
    try:
        return require_text(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationS0PreflightError(str(exc)) from exc


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfirmationS0PreflightError(str(exc)) from exc


def _self_sealed(value: Mapping[str, Any], *, key: str, label: str) -> str:
    declared = _sha(value.get(key), f"{label} {key}")
    body = dict(value)
    body.pop(key, None)
    _require(identity_sha256(body) == declared, f"{label} self-seal differs")
    return declared


def _ordered_unique_text(value: object, label: str) -> tuple[str, ...]:
    rows = require_list(value, label)
    result = tuple(_text(item, f"{label} item") for item in rows)
    _require(len(result) == len(set(result)), f"{label} must be ordered and unique")
    return result


def _raw_question(dated_question: str) -> str:
    first, separator, remainder = dated_question.partition("\n")
    if (
        separator
        and first.startswith("[Question asked at ")
        and first.endswith("]")
        and remainder
    ):
        return remainder
    return dated_question


@dataclass(frozen=True, slots=True)
class FrozenTerraRuntime:
    gateway_url: str
    hard_complete_chat_token_cap: int
    input_token_cap: int
    max_concurrency: int
    model: str
    output_token_reserve: int
    retry_count: int

    def __post_init__(self) -> None:
        _text(self.gateway_url, "Terra gateway URL")
        _text(self.model, "Terra model")
        for value, label in (
            (self.hard_complete_chat_token_cap, "hard complete-chat cap"),
            (self.input_token_cap, "input-token cap"),
            (self.max_concurrency, "max concurrency"),
            (self.output_token_reserve, "output-token reserve"),
        ):
            _require(type(value) is int and value > 0, f"{label} must be positive")
        _require(self.retry_count == 0, "frozen Terra retries must equal zero")
        _require(
            self.input_token_cap + self.output_token_reserve
            == self.hard_complete_chat_token_cap,
            "frozen Terra input/output budgets do not sum to the hard cap",
        )

    def projection(self) -> dict[str, Any]:
        return {
            "gateway_url": self.gateway_url,
            "hard_complete_chat_token_cap": self.hard_complete_chat_token_cap,
            "input_token_cap": self.input_token_cap,
            "max_concurrency": self.max_concurrency,
            "model": self.model,
            "output_token_reserve": self.output_token_reserve,
            "retry_count": self.retry_count,
        }


@dataclass(frozen=True, slots=True)
class GenericMatchedS0Row:
    row_index: int
    question_id: str
    namespace_id: str
    namespace_receipt_sha256: str
    namespace_store_id: str
    namespace_checkpoint_sha256: str
    source_question_receipt_sha256: str
    s0_stage_receipt_sha256: str
    final_stage_receipt_sha256: str
    packet: MemoryPacket
    rendered_prompt: RenderedPrompt

    def __post_init__(self) -> None:
        _require(type(self.row_index) is int and self.row_index >= 0, "row index is invalid")
        _text(self.question_id, "S0 question ID")
        _text(self.namespace_id, "S0 namespace ID")
        for value, label in (
            (self.namespace_receipt_sha256, "namespace receipt"),
            (self.namespace_store_id, "namespace store"),
            (self.namespace_checkpoint_sha256, "namespace checkpoint"),
            (self.source_question_receipt_sha256, "source question receipt"),
            (self.s0_stage_receipt_sha256, "S0 stage receipt"),
            (self.final_stage_receipt_sha256, "final stage receipt"),
        ):
            _sha(value, f"{label} SHA-256")
        _require(self.packet.question_id == self.question_id, "S0 packet question changed")
        _require(
            self.rendered_prompt.packet_id == self.packet.packet_id,
            "S0 prompt packet binding changed",
        )
        _require(
            self.rendered_prompt.renderer_id == V4_RENDERER_ID,
            "confirmation S0 renderer changed",
        )

    def binding_projection(self) -> dict[str, Any]:
        provider_body = {
            "format": PREFLIGHT_PROVIDER_INPUT_FORMAT,
            "messages": [dict(message) for message in self.rendered_prompt.messages],
            "messages_sha256": self.rendered_prompt.messages_sha256,
        }
        provider_input = {
            **provider_body,
            "provider_input_receipt_sha256": identity_sha256(provider_body),
        }
        body = {
            "format": PREFLIGHT_ROW_FORMAT,
            "row_index": self.row_index,
            "question_id": self.question_id,
            "namespace_id": self.namespace_id,
            "namespace_receipt_sha256": self.namespace_receipt_sha256,
            "namespace_store_id": self.namespace_store_id,
            "namespace_checkpoint_sha256": self.namespace_checkpoint_sha256,
            "source_question_receipt_sha256": self.source_question_receipt_sha256,
            "s0_stage_receipt_sha256": self.s0_stage_receipt_sha256,
            "final_stage_receipt_sha256": self.final_stage_receipt_sha256,
            "packet_id": self.packet.packet_id,
            "prompt_id": self.rendered_prompt.prompt_id,
            "messages_sha256": self.rendered_prompt.messages_sha256,
            "prompt_token_proxy": self.rendered_prompt.total_prompt_token_proxy,
            "provider_input": provider_input,
        }
        return {**body, "row_receipt_sha256": identity_sha256(body)}


@dataclass(frozen=True, slots=True)
class GenericMatchedS0Population:
    cumulative_retrieval_sha256: str
    policy_manifest_sha256: str
    treatment_file_sha256: str
    treatment_preflight_sha256: str
    ordered_question_ids_sha256: str
    rows: tuple[GenericMatchedS0Row, ...]
    prompt_population: FastPromptPopulation
    runtime: FrozenTerraRuntime

    def __post_init__(self) -> None:
        for value, label in (
            (self.cumulative_retrieval_sha256, "cumulative retrieval"),
            (self.policy_manifest_sha256, "policy manifest"),
            (self.treatment_file_sha256, "treatment file"),
            (self.treatment_preflight_sha256, "treatment preflight"),
            (self.ordered_question_ids_sha256, "ordered question IDs"),
        ):
            _sha(value, f"{label} SHA-256")
        _require(bool(self.rows), "generic S0 population cannot be empty")
        _require(
            tuple(row.row_index for row in self.rows) == tuple(range(len(self.rows))),
            "generic S0 rows are incomplete or reordered",
        )
        ids = tuple(row.question_id for row in self.rows)
        _require(len(ids) == len(set(ids)), "generic S0 question IDs repeat")
        _require(
            identity_sha256(list(ids)) == self.ordered_question_ids_sha256,
            "generic S0 ordered question root differs",
        )
        _require(
            self.prompt_population.logical_prompt_count == len(self.rows)
            and self.prompt_population.unique_prompt_count == len(self.rows),
            "confirmation S0 requires exactly one unique prompt per row",
        )
        _require(
            tuple(row.rendered_prompt.messages_sha256 for row in self.rows)
            == tuple(row.messages_sha256 for row in self.prompt_population.ordered_rows),
            "generic S0 prompt population order differs",
        )
        _require(
            self.prompt_population.max_prompt_token_proxy == self.runtime.input_token_cap,
            "generic S0 prompt budget differs from frozen policy",
        )

    @property
    def question_count(self) -> int:
        return len(self.rows)

    @property
    def population_id(self) -> str:
        return identity_sha256(
            {
                "format": "memory-condense-confirmation-generic-matched-s0-population-v1",
                "cumulative_retrieval_sha256": self.cumulative_retrieval_sha256,
                "policy_manifest_sha256": self.policy_manifest_sha256,
                "treatment_file_sha256": self.treatment_file_sha256,
                "treatment_preflight_sha256": self.treatment_preflight_sha256,
                "ordered_question_ids_sha256": self.ordered_question_ids_sha256,
                "renderer_id": V4_RENDERER_ID,
                "rows": [row.binding_projection() for row in self.rows],
                "runtime": self.runtime.projection(),
            }
        )

    def preflight_projection(self) -> dict[str, Any]:
        rows = [row.binding_projection() for row in self.rows]
        body = {
            "format": PREFLIGHT_FORMAT,
            "status": "compiled",
            "gold_loaded": False,
            "bindings": {
                "cumulative_retrieval_sha256": self.cumulative_retrieval_sha256,
                "policy_manifest_sha256": self.policy_manifest_sha256,
                "treatment_file_sha256": self.treatment_file_sha256,
                "treatment_preflight_sha256": self.treatment_preflight_sha256,
            },
            "population": {
                "matched_population_id": self.population_id,
                "question_count": self.question_count,
                "ordered_question_ids_sha256": self.ordered_question_ids_sha256,
                "renderer_id": V4_RENDERER_ID,
            },
            "runtime": self.runtime.projection(),
            "execution": {
                "logical_prompt_count": self.question_count,
                "unique_prompt_count": self.question_count,
                "would_call_count": self.question_count,
                "would_call_count_status": "exact",
                "count_basis": "one-unique-terra-prompt-per-complete-s0-row",
                "physical_provider_calls": 0,
                "provider_execution_available": False,
                "authorization_released": False,
                "retained_request_token_state_bytes": 0,
            },
            "ordered_rows": rows,
            "prompt_population": self.prompt_population.model_dump(),
            "prompt_population_sha256": self.prompt_population.prompt_population_sha256,
        }
        assert_gold_blind(body, path="confirmation_s0_preflight")
        return {**body, "preflight_identity_sha256": identity_sha256(body)}


def _runtime_from_policy(policy: RuntimePolicy) -> FrozenTerraRuntime:
    treatment_policy = require_mapping(
        policy.payload["treatment_policy"],
        "frozen treatment policy",
    )
    runtime = require_mapping(
        treatment_policy["responder_runtime"],
        "frozen treatment responder runtime",
    )
    exact_keys(runtime, _RUNTIME_KEYS, "frozen treatment responder runtime")
    try:
        result = FrozenTerraRuntime(
            gateway_url=_text(runtime["gateway_url"], "Terra gateway URL"),
            hard_complete_chat_token_cap=require_int(
                runtime["hard_complete_chat_token_cap"],
                "hard complete-chat token cap",
                minimum=1,
            ),
            input_token_cap=require_int(
                runtime["input_token_cap"],
                "input token cap",
                minimum=1,
            ),
            max_concurrency=require_int(
                runtime["max_concurrency"],
                "max concurrency",
                minimum=1,
            ),
            model=_text(runtime["model"], "Terra model"),
            output_token_reserve=require_int(
                runtime["output_token_reserve"],
                "output token reserve",
                minimum=1,
            ),
            retry_count=require_int(runtime["retry_count"], "retry count"),
        )
    except ValueError as exc:
        raise ConfirmationS0PreflightError(str(exc)) from exc
    return result


def _preflight_namespace_map(preflight: SealedJson) -> dict[str, tuple[str, str]]:
    raw_namespaces = require_list(
        preflight.payload.get("namespaces"),
        "treatment preflight namespaces",
    )
    mapping: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(raw_namespaces):
        namespace = require_mapping(raw, f"treatment namespace {index}")
        namespace_id = _text(namespace.get("namespace_id"), f"namespace {index} ID")
        receipt = _sha(
            namespace.get("namespace_receipt_sha256"),
            f"namespace {index} receipt",
        )
        ids = _ordered_unique_text(
            namespace.get("question_ids"),
            f"namespace {index} question IDs",
        )
        for question_id in ids:
            _require(question_id not in mapping, "treatment namespaces overlap")
            mapping[question_id] = (namespace_id, receipt)
    return mapping


def _plain_messages(value: object, label: str) -> list[dict[str, str]]:
    raw_messages = require_list(value, label)
    _require(bool(raw_messages), f"{label} cannot be empty")
    messages: list[dict[str, str]] = []
    for index, raw in enumerate(raw_messages):
        message = require_mapping(raw, f"{label} {index}")
        exact_keys(message, {"role", "content"}, f"{label} {index}")
        messages.append(
            {
                "role": _text(message["role"], f"{label} {index} role"),
                "content": _text(message["content"], f"{label} {index} content"),
            }
        )
    return messages


def _decode_row(
    value: object,
    *,
    row_index: int,
    expected_sample: Any,
    expected_preflight_row: Mapping[str, Any],
    expected_namespace: tuple[str, str],
    expected_namespace_store_id: str,
    expected_namespace_checkpoint_sha256: str,
) -> GenericMatchedS0Row:
    label = f"cumulative merged row {row_index}"
    wrapper = require_mapping(value, label)
    exact_keys(wrapper, _ROW_KEYS, label)
    _require(wrapper["format"] == MERGED_ROW_FORMAT, f"{label} format changed")
    namespace_id = _text(wrapper["namespace_id"], f"{label} namespace ID")
    namespace_store_id = _sha(
        wrapper["namespace_store_id"], f"{label} namespace store"
    )
    namespace_checkpoint_sha = _sha(
        wrapper["namespace_checkpoint_sha256"], f"{label} namespace checkpoint"
    )
    _require(
        namespace_id == expected_namespace[0]
        and namespace_store_id == expected_namespace_store_id
        and namespace_checkpoint_sha == expected_namespace_checkpoint_sha256,
        f"{label} escapes its authenticated namespace",
    )

    row = require_mapping(wrapper["question"], f"{label} question")
    exact_keys(row, _QUESTION_KEYS, f"{label} question")
    _require(row["format"] == QUESTION_FORMAT, f"{label} question format changed")
    question_receipt = _self_sealed(
        row, key="question_receipt_sha256", label=f"{label} question"
    )
    _require(
        _sha(wrapper["source_question_receipt_sha256"], f"{label} source question")
        == question_receipt,
        f"{label} source question receipt differs",
    )
    question_id = _text(row["question_id"], f"{label} question ID")
    question = _text(row["question"], f"{label} question text")
    dated_question = _text(row["dated_question"], f"{label} dated question")
    expected_question = expected_sample.questions[0]
    _require(
        question_id == expected_sample.sample_id
        and question == expected_question.question
        and dated_question == expected_question.dated_question,
        f"{label} question order or content differs",
    )
    question_sha = _sha(row["question_sha256"], f"{label} question")
    dated_sha = _sha(row["dated_question_sha256"], f"{label} dated question")
    _require(
        question_sha == quote_sha256(question),
        f"{label} raw-question identity differs",
    )
    _require(dated_sha == quote_sha256(dated_question), f"{label} dated-question identity differs")
    _require(
        _sha(row["question_id_sha256"], f"{label} question ID")
        == quote_sha256(question_id),
        f"{label} question-ID identity differs",
    )
    _require(row["physical_provider_calls"] == 0, f"{label} reports provider calls")
    _require(
        row["row_receipt_sha256"] == expected_preflight_row["row_receipt_sha256"]
        and row["content_binding_sha256"]
        == expected_preflight_row["content_binding_sha256"],
        f"{label} treatment binding differs",
    )
    require_mapping(row["predecessor_receipt"], f"{label} predecessor receipt")
    require_mapping(row["retrieval_receipt"], f"{label} retrieval receipt")
    _require(tuple(row["stage_ids"]) == STAGE_IDS, f"{label} stage order changed")

    stages = require_list(row["stages"], f"{label} stages")
    _require(len(stages) == len(STAGE_IDS), f"{label} stage chain is incomplete")
    parent_receipt: str | None = None
    parent_ids: tuple[str, ...] = ()
    root_evidence: tuple[EvidenceItem, ...] | None = None
    root_messages: list[dict[str, str]] | None = None
    first_stage: str | None = None
    final_stage: str | None = None
    for position, (raw_stage, expected_stage_id) in enumerate(
        zip(stages, STAGE_IDS, strict=True)
    ):
        stage_label = f"{label} stage {expected_stage_id}"
        stage = require_mapping(raw_stage, stage_label)
        exact_keys(stage, _STAGE_KEYS, stage_label)
        _require(
            stage["format"] == STAGE_FORMAT
            and stage["stage_id"] == expected_stage_id,
            f"{stage_label} identity or order changed",
        )
        try:
            receipt = CumulativeRetrievalStageReceipt(
                **dict(require_mapping(stage["stage_receipt"], f"{stage_label} receipt"))
            )
        except (TypeError, ValueError) as exc:
            raise ConfirmationS0PreflightError(
                f"{stage_label} receipt failed authentication: {exc}"
            ) from exc
        _require(
            receipt.stage_id == expected_stage_id
            and receipt.parent_stage_receipt_sha256 == parent_receipt
            and receipt.parent_evidence_ids == parent_ids,
            f"{stage_label} parent binding differs",
        )
        evidence_rows = require_list(stage["evidence"], f"{stage_label} evidence")
        evidence: list[EvidenceItem] = []
        for evidence_index, raw_evidence in enumerate(evidence_rows):
            evidence_row = require_mapping(
                raw_evidence, f"{stage_label} evidence {evidence_index}"
            )
            exact_keys(
                evidence_row,
                _EVIDENCE_KEYS,
                f"{stage_label} evidence {evidence_index}",
            )
            _require(
                evidence_row["format"] == EVIDENCE_FORMAT,
                f"{stage_label} evidence format changed",
            )
            evidence_text = evidence_row["text"]
            _require(type(evidence_text) is str, f"{stage_label} evidence text changed")
            evidence.append(
                EvidenceItem(
                    evidence_id=_text(evidence_row["evidence_id"], f"{stage_label} evidence ID"),
                    source_id=_text(evidence_row["source_id"], f"{stage_label} source ID"),
                    text=evidence_text,
                    token_count=count_tokens(evidence_text),
                )
            )
        evidence_tuple = tuple(evidence)
        evidence_ids = tuple(item.evidence_id for item in evidence_tuple)
        _require(
            evidence_ids == receipt.selected_evidence_ids
            and evidence_ids[: len(parent_ids)] == parent_ids,
            f"{stage_label} evidence coordinates or cumulative prefix differ",
        )
        context = "\n".join(
            f"[{number}] {item.text}"
            for number, item in enumerate(evidence_tuple, start=1)
        )
        messages = _plain_messages(stage["provider_messages"], f"{stage_label} messages")
        _require(
            receipt.context_sha256 == quote_sha256(context)
            and receipt.context_token_proxy == count_tokens(context)
            and receipt.prompt_messages_sha256 == identity_sha256(messages)
            and receipt.prompt_token_proxy == count_chat_prompt_token_proxy(messages),
            f"{stage_label} evidence/prompt receipt differs",
        )
        if position == 0:
            root_evidence = evidence_tuple
            root_messages = messages
            first_stage = receipt.receipt_sha256
        parent_ids = evidence_ids
        parent_receipt = receipt.receipt_sha256
        final_stage = receipt.receipt_sha256

    assert root_evidence is not None and root_messages is not None
    assert first_stage is not None and final_stage is not None
    try:
        packet = MemoryPacket(
            question_id=question_id,
            question_sha256=question_sha,
            dated_question=dated_question,
            dated_question_sha256=dated_sha,
            stage_id=SOURCE_STAGE_ID,
            protected_evidence=root_evidence,
        )
        rendered = render_memory_packet_for_id(packet, renderer_id=V4_RENDERER_ID)
    except MatchedEvalContractError as exc:
        raise ConfirmationS0PreflightError(str(exc)) from exc
    _require(
        root_messages == [dict(message) for message in rendered.messages],
        f"{label} S0 messages differ from the V4 renderer",
    )
    return GenericMatchedS0Row(
        row_index=row_index,
        question_id=question_id,
        namespace_id=namespace_id,
        namespace_receipt_sha256=expected_namespace[1],
        namespace_store_id=namespace_store_id,
        namespace_checkpoint_sha256=namespace_checkpoint_sha,
        source_question_receipt_sha256=question_receipt,
        s0_stage_receipt_sha256=first_stage,
        final_stage_receipt_sha256=final_stage,
        packet=packet,
        rendered_prompt=rendered,
    )


def load_generic_matched_s0_population(
    *,
    runtime_policy_path: str | Path,
    expected_runtime_policy_sha256: str,
    treatment_input_path: str | Path,
    expected_treatment_input_sha256: str,
    treatment_preflight_path: str | Path,
    expected_treatment_preflight_sha256: str,
    cumulative_retrieval_path: str | Path,
    expected_cumulative_retrieval_sha256: str,
) -> GenericMatchedS0Population:
    """Load arbitrary-N confirmation S0 prompts without opening benchmark data."""

    treatment_artifact = read_sealed_json(
        treatment_input_path,
        expected_sha256=expected_treatment_input_sha256,
        label="label-free confirmation treatment",
    )
    treatment, _raw_samples = _decode_treatment(treatment_artifact)
    policy = read_runtime_policy(
        runtime_policy_path,
        expected_runtime_policy_sha256=expected_runtime_policy_sha256,
        treatment=treatment,
    )
    treatment_preflight = read_sealed_json(
        treatment_preflight_path,
        expected_sha256=expected_treatment_preflight_sha256,
        label="label-free confirmation preflight",
    )
    _verify_preflight(treatment_preflight, treatment)
    runtime = _runtime_from_policy(policy)
    namespace_map = _preflight_namespace_map(treatment_preflight)
    question_ids = tuple(sample.sample_id for sample in treatment.samples)
    _require(set(namespace_map) == set(question_ids), "treatment namespace coverage differs")

    cumulative = read_sealed_json(
        cumulative_retrieval_path,
        expected_sha256=expected_cumulative_retrieval_sha256,
        label="confirmation cumulative retrieval",
    )
    value = cumulative.payload
    assert_gold_blind(value, path="confirmation_cumulative_retrieval")
    exact_keys(value, _CUMULATIVE_KEYS, "confirmation cumulative retrieval")
    _require(value["format"] == CUMULATIVE_RETRIEVAL_FORMAT, "unsupported cumulative retrieval format")
    _require(value["gold_loaded"] is False, "cumulative retrieval crossed the gold firewall")
    _require(value["physical_provider_calls"] == 0, "cumulative retrieval reports provider calls")
    _require(value["freeze_sha256"] == policy.sha256, "cumulative retrieval binds another policy")
    _require(value["preflight_sha256"] == treatment_preflight.sha256, "cumulative retrieval binds another preflight")
    _sha(value["backend_identity_sha256"], "cumulative backend identity")
    count = require_int(value["question_count"], "cumulative question count", minimum=1)
    _require(count == len(question_ids), "cumulative retrieval population is incomplete")
    _require(tuple(value["stage_ids"]) == CUMULATIVE_STAGE_IDS, "cumulative stage order changed")
    _self_sealed(
        value,
        key="merge_receipt_sha256",
        label="confirmation cumulative retrieval",
    )

    preflight_rows = require_list(
        treatment_preflight.payload.get("rows"), "treatment preflight rows"
    )
    _require(len(preflight_rows) == count, "treatment preflight rows are incomplete")
    expected_row_receipts = [
        _sha(
            require_mapping(row, f"treatment preflight row {index}").get(
                "row_receipt_sha256"
            ),
            f"treatment preflight row {index}",
        )
        for index, row in enumerate(preflight_rows)
    ]

    population_identity = require_mapping(
        value["population_identity"], "cumulative population identity"
    )
    exact_keys(
        population_identity,
        _POPULATION_IDENTITY_KEYS,
        "cumulative population identity",
    )
    _require(
        population_identity["format"] == POPULATION_IDENTITY_FORMAT,
        "cumulative population identity format changed",
    )
    population_receipt = _self_sealed(
        population_identity,
        key="population_identity_sha256",
        label="cumulative population identity",
    )
    _require(
        population_receipt
        == _sha(value["population_identity_sha256"], "cumulative population identity")
        and population_identity["dataset_sha256"] == treatment.dataset_sha256
        and population_identity["split_manifest_sha256"]
        == treatment.split_manifest_sha256
        and population_identity["sanitized_projection_sha256"]
        == treatment.sanitized_projection_sha256
        and population_identity["preflight_sha256"] == treatment_preflight.sha256
        and population_identity["workset_identity_sha256"]
        == value["workset_identity_sha256"]
        and population_identity["ordered_row_receipt_sha256s"]
        == expected_row_receipts,
        "cumulative population binding differs from treatment/preflight",
    )

    preflight_namespaces = require_list(
        treatment_preflight.payload.get("namespaces"), "treatment namespaces"
    )
    checkpoint_rows = require_list(
        value["namespace_checkpoints"], "cumulative namespace checkpoints"
    )
    _require(
        require_int(value["namespace_count"], "cumulative namespace count", minimum=1)
        == len(checkpoint_rows)
        == len(preflight_namespaces),
        "cumulative namespace population differs",
    )
    checkpoints: dict[str, tuple[str, str]] = {}
    checkpoint_namespace_order: list[str] = []
    for index, (raw_checkpoint, raw_namespace) in enumerate(
        zip(checkpoint_rows, preflight_namespaces, strict=True)
    ):
        checkpoint = require_mapping(raw_checkpoint, f"cumulative checkpoint {index}")
        exact_keys(
            checkpoint,
            _NAMESPACE_CHECKPOINT_KEYS,
            f"cumulative checkpoint {index}",
        )
        namespace = require_mapping(raw_namespace, f"treatment namespace {index}")
        namespace_id = _text(checkpoint["namespace_id"], f"checkpoint {index} namespace")
        _require(
            namespace_id == namespace["namespace_id"] and namespace_id not in checkpoints,
            "cumulative namespace checkpoint order or identity differs",
        )
        store_id = _sha(checkpoint["namespace_store_id"], f"checkpoint {index} store")
        checkpoint_sha = _sha(
            checkpoint["checkpoint_sha256"], f"checkpoint {index} artifact"
        )
        _sha(checkpoint["checkpoint_receipt_sha256"], f"checkpoint {index} receipt")
        _sha(
            checkpoint["namespace_work_receipt_sha256"],
            f"checkpoint {index} namespace work receipt",
        )
        checkpoints[namespace_id] = (store_id, checkpoint_sha)
        checkpoint_namespace_order.append(namespace_id)
    _require(
        population_identity["namespace_store_ids"]
        == [checkpoints[key][0] for key in checkpoint_namespace_order],
        "cumulative namespace-store order differs",
    )

    raw_rows = require_list(value["questions"], "cumulative retrieval rows")
    _require(len(raw_rows) == count, "cumulative retrieval rows are incomplete")
    question_receipts = [
        _sha(
            require_mapping(row, f"cumulative merged row {index}").get(
                "source_question_receipt_sha256"
            ),
            f"cumulative merged row {index} source question receipt",
        )
        for index, row in enumerate(raw_rows)
    ]
    _require(
        value["question_receipt_sha256s"] == question_receipts
        and _sha(value["question_order_sha256"], "cumulative question order")
        == identity_sha256(question_receipts),
        "cumulative question receipt order differs",
    )
    rows: list[GenericMatchedS0Row] = []
    for index, (raw, sample, preflight_row) in enumerate(
        zip(raw_rows, treatment.samples, preflight_rows, strict=True)
    ):
        namespace = namespace_map[sample.sample_id]
        namespace_store_id, checkpoint_sha = checkpoints[namespace[0]]
        rows.append(
            _decode_row(
                raw,
                row_index=index,
                expected_sample=sample,
                expected_preflight_row=require_mapping(
                    preflight_row, f"treatment preflight row {index}"
                ),
                expected_namespace=namespace,
                expected_namespace_store_id=namespace_store_id,
                expected_namespace_checkpoint_sha256=checkpoint_sha,
            )
        )

    namespace_sources: dict[str, tuple[str, str]] = {}
    source_to_namespace: dict[str, str] = {}
    for row in rows:
        source = (row.namespace_store_id, row.namespace_checkpoint_sha256)
        previous = namespace_sources.setdefault(row.namespace_id, source)
        _require(previous == source, "one namespace binds multiple checkpoints")
        owner = source_to_namespace.setdefault(row.namespace_store_id, row.namespace_id)
        _require(owner == row.namespace_id, "a source store crosses namespace boundaries")
    _require(
        set(namespace_sources) == {value[0] for value in namespace_map.values()},
        "cumulative retrieval omits a treatment namespace",
    )
    prompts = tuple(row.rendered_prompt.messages for row in rows)
    try:
        prompt_population = preflight_fast_completion_prompts(
            prompts,
            max_prompt_tokens=runtime.input_token_cap,
        )
    except ValueError as exc:
        raise ConfirmationS0PreflightError(f"S0 prompt preflight failed: {exc}") from exc
    _require(
        prompt_population.logical_prompt_count
        == prompt_population.unique_prompt_count
        == count,
        "confirmation S0 prompts are not unique per row",
    )
    population = GenericMatchedS0Population(
        cumulative_retrieval_sha256=cumulative.sha256,
        policy_manifest_sha256=policy.sha256,
        treatment_file_sha256=treatment_artifact.sha256,
        treatment_preflight_sha256=treatment_preflight.sha256,
        ordered_question_ids_sha256=treatment.ordered_question_ids_sha256,
        rows=tuple(rows),
        prompt_population=prompt_population,
        runtime=runtime,
    )
    for artifact, label in (
        (policy, "frozen policy manifest"),
        (treatment_artifact, "label-free confirmation treatment"),
        (treatment_preflight, "label-free confirmation preflight"),
        (cumulative, "confirmation cumulative retrieval"),
    ):
        assert_snapshot_unchanged(artifact.snapshot, label)
        assert_snapshot_unchanged(artifact.sidecar, f"{label} digest sidecar")
    return population


def compile_confirmation_s0_preflight(**kwargs: Any) -> dict[str, Any]:
    return load_generic_matched_s0_population(**kwargs).preflight_projection()


def publish_confirmation_s0_preflight(
    output_path: str | Path,
    **kwargs: Any,
) -> tuple[SealedJson, bool]:
    return publish_sealed_json(output_path, compile_confirmation_s0_preflight(**kwargs))


def replay_confirmation_s0_preflight(
    *,
    source_preflight_path: str | Path,
    expected_source_preflight_sha256: str,
    replay_output_path: str | Path,
    **kwargs: Any,
) -> tuple[SealedJson, bool]:
    source = read_sealed_json(
        source_preflight_path,
        expected_sha256=expected_source_preflight_sha256,
        label="confirmation S0 preflight",
    )
    expected = compile_confirmation_s0_preflight(**kwargs)
    _require(source.payload == expected, "confirmation S0 preflight replay differs")
    replay, created = publish_sealed_json(replay_output_path, expected)
    _require(replay.sha256 == source.sha256, "confirmation S0 replay seal differs")
    assert_snapshot_unchanged(source.snapshot, "confirmation S0 preflight")
    assert_snapshot_unchanged(source.sidecar, "confirmation S0 preflight digest sidecar")
    return replay, created


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-policy", type=Path, required=True)
    parser.add_argument("--expected-runtime-policy-sha256", required=True)
    parser.add_argument("--treatment-input", type=Path, required=True)
    parser.add_argument("--expected-treatment-input-sha256", required=True)
    parser.add_argument("--treatment-preflight", type=Path, required=True)
    parser.add_argument("--expected-treatment-preflight-sha256", required=True)
    parser.add_argument("--cumulative-retrieval", type=Path, required=True)
    parser.add_argument("--expected-cumulative-retrieval-sha256", required=True)


def _input_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime_policy_path": args.runtime_policy,
        "expected_runtime_policy_sha256": args.expected_runtime_policy_sha256,
        "treatment_input_path": args.treatment_input,
        "expected_treatment_input_sha256": args.expected_treatment_input_sha256,
        "treatment_preflight_path": args.treatment_preflight,
        "expected_treatment_preflight_sha256": args.expected_treatment_preflight_sha256,
        "cumulative_retrieval_path": args.cumulative_retrieval,
        "expected_cumulative_retrieval_sha256": args.expected_cumulative_retrieval_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile", help="compile inert S0 Terra prompts")
    _add_inputs(compile_parser)
    compile_parser.add_argument("--output", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay", help="provider-free exact preflight replay")
    _add_inputs(replay_parser)
    replay_parser.add_argument("--source-preflight", type=Path, required=True)
    replay_parser.add_argument("--expected-source-preflight-sha256", required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "compile":
        artifact, created = publish_confirmation_s0_preflight(
            args.output,
            **_input_kwargs(args),
        )
    elif args.command == "replay":
        artifact, created = replay_confirmation_s0_preflight(
            source_preflight_path=args.source_preflight,
            expected_source_preflight_sha256=args.expected_source_preflight_sha256,
            replay_output_path=args.output,
            **_input_kwargs(args),
        )
    else:  # pragma: no cover - argparse owns the closed choices.
        raise ConfirmationS0PreflightError("unknown command")
    return {
        "created": created,
        "preflight_sha256": artifact.sha256,
        "question_count": artifact.payload["population"]["question_count"],
        "would_call_count": artifact.payload["execution"]["would_call_count"],
        "physical_provider_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (ConfirmationS0PreflightError, MatchedEvalContractError, ValueError) as exc:
        print(f"confirmation S0 preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CUMULATIVE_RETRIEVAL_FORMAT",
    "CUMULATIVE_ROW_FORMAT",
    "CUMULATIVE_STAGE_IDS",
    "CUMULATIVE_STAGE_RECEIPT_FORMAT",
    "ConfirmationS0PreflightError",
    "FrozenTerraRuntime",
    "GenericMatchedS0Population",
    "GenericMatchedS0Row",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_PROVIDER_INPUT_FORMAT",
    "build_parser",
    "compile_confirmation_s0_preflight",
    "load_generic_matched_s0_population",
    "main",
    "publish_confirmation_s0_preflight",
    "replay_confirmation_s0_preflight",
]
