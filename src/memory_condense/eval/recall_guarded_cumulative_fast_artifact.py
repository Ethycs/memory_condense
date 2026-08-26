"""Read-only, model-free access to a sealed cumulative retrieval artifact.

This module is deliberately independent of the corpus, stores, tokenizers, and
transformer runtimes.  It turns the already-published ``retrieval.json`` into
small immutable views and a per-question table of distinct feature inputs.
Repeated evidence in later cumulative stages maps back to that table instead
of requesting another tokenizer/model pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

RETRIEVAL_FORMAT = "memory-condense-recall-guarded-cumulative-1m-retrieval-v1"
CAMPAIGN_FORMAT = "memory-condense-recall-guarded-cumulative-1m-campaign-v1"
QUESTION_FORMAT = "memory-condense-recall-guarded-cumulative-1m-query-v1"
POPULATION_FORMAT = "memory-condense-original-1m-development-population-v1"
CUMULATIVE_STAGE_FORMAT = "memory-condense-cumulative-retrieval-stage-v2"
STAGE_IDS = (
    "causal_graph_coverage_predecessor",
    "direct_episode_additions",
    "representative_episode_additions",
    "artifact_global_closure_additions",
)

# The sealed development artifact reused by the fast benchmark path.
ORIGINAL_1M_RETRIEVAL_SHA256 = (
    "aa22f7c18470d9a7c931fd16f8f58bf67d8566e2298a45371ee2815c11a9bd97"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QA_CONTEXT_PREFIX = "Retrieved excerpts from the conversation history:\n"
_QA_QUESTION_MARKER = "\n\nQuestion: "
_QA_ANSWER_SUFFIX = "\nShort answer:"
_DATED_QUESTION_PREFIX = "[Question asked at "


class FastArtifactValidationError(ValueError):
    """Raised when an artifact cannot prove its published identity or shape."""


@dataclass(frozen=True, slots=True)
class FastProviderMessage:
    """One exact provider message, preserving its original order and text."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class FastEvidence:
    """One exact evidence coordinate from the published retrieval artifact."""

    evidence_id: str
    source_id: str
    text: str


@dataclass(frozen=True, slots=True)
class FastFeatureRow:
    """One distinct content-level feature computation within a question.

    The existing Qwen feature row depends on the raw question and evidence
    text, not on a stage or provenance label.  Stage evidence coordinates are
    retained separately and map to these rows by integer index.
    """

    question: str
    evidence_text: str
    row_sha256: str


@dataclass(frozen=True, slots=True)
class FastQuestionParseReceipt:
    """Proof that the exact question came from the sealed final user prompt."""

    framing: str
    source_stage_id: str
    provider_message_index: int
    provider_message_sha256: str
    question_marker_occurrences: int
    matching_framing_candidates: int
    dated_question_sha256: str
    question_sha256: str
    question_form: str


@dataclass(frozen=True, slots=True)
class FastRetrievalStage:
    """One ordered cumulative stage and its feature-table projection."""

    stage_id: str
    stage_receipt_sha256: str
    matched_controls_sha256: str
    evidence_projection_sha256: str
    context_sha256: str
    prompt_messages_sha256: str
    context_token_proxy: int
    max_context_token_proxy: int
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    responder_output_token_reserve: int
    admission_status: str
    added_evidence_ids: tuple[str, ...]
    context: str
    evidence: tuple[FastEvidence, ...]
    provider_messages: tuple[FastProviderMessage, ...]
    feature_row_indices: tuple[int, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(row.evidence_id for row in self.evidence)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(row.source_id for row in self.evidence)

    @property
    def exact_texts(self) -> tuple[str, ...]:
        return tuple(row.text for row in self.evidence)


@dataclass(frozen=True, slots=True)
class FastRetrievalQuestion:
    """One ordered question with immutable stages and deduplicated row work."""

    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    predecessor_receipt_sha256: str
    retrieval_receipt_sha256: str
    protected_chunk_ids: tuple[str, ...]
    retained_request_token_state_bytes: int
    question: str
    dated_question: str
    final_user_message: FastProviderMessage
    question_parse_receipt: FastQuestionParseReceipt
    feature_rows: tuple[FastFeatureRow, ...]
    stages: tuple[FastRetrievalStage, ...]

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)

    @property
    def logical_feature_row_count(self) -> int:
        return sum(len(stage.feature_row_indices) for stage in self.stages)

    @property
    def unique_feature_row_count(self) -> int:
        return len(self.feature_rows)

    def stage(self, stage_id: str) -> FastRetrievalStage:
        """Return one named stage without changing its published order."""

        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)


@dataclass(frozen=True, slots=True)
class FastRetrievalArtifact:
    """Immutable, read-only projection of the sealed retrieval JSON."""

    source_path: str
    raw_sha256: str
    format: str
    campaign_format: str
    population_identity_sha256: str
    source_store_receipt_sha256: str
    combined_store_receipt_sha256: str
    retrieval_implementation_sha256: str
    retrieval_policy_sha256: str
    transcript_tokens: int
    turn_count: int
    retained_request_token_state_bytes: int
    stage_ids: tuple[str, ...]
    questions: tuple[FastRetrievalQuestion, ...]

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def logical_feature_row_count(self) -> int:
        return sum(row.logical_feature_row_count for row in self.questions)

    @property
    def unique_feature_row_count(self) -> int:
        return sum(row.unique_feature_row_count for row in self.questions)

    def question(self, question_id: str) -> FastRetrievalQuestion:
        """Return one question by its exact published identifier."""

        for question in self.questions:
            if question.question_id == question_id:
                return question
        raise KeyError(question_id)


@dataclass(frozen=True, slots=True)
class _ParsedUserPrompt:
    context: str
    dated_question: str
    message_index: int
    message_sha256: str
    marker_occurrences: int
    matching_candidates: int


@dataclass(frozen=True, slots=True)
class _ParsedStage:
    stage_id: str
    receipt_sha256: str
    matched_controls_sha256: str
    evidence_projection_sha256: str
    context_sha256: str
    prompt_messages_sha256: str
    context_token_proxy: int
    max_context_token_proxy: int
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    responder_output_token_reserve: int
    admission_status: str
    added_evidence_ids: tuple[str, ...]
    context: str
    evidence: tuple[FastEvidence, ...]
    messages: tuple[FastProviderMessage, ...]
    parsed_prompt: _ParsedUserPrompt


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _identity_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quote_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_nonfinite_json(value: str) -> None:
    raise FastArtifactValidationError(
        f"artifact contains non-standard JSON numeric value {value}"
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise FastArtifactValidationError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise FastArtifactValidationError(f"{label} must be a JSON array")
    return value


def _require_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise FastArtifactValidationError(f"{label} must be a {qualifier}string")
    return value


def _require_digest(value: Any, label: str) -> str:
    digest = _require_string(value, label)
    if not _SHA256_RE.fullmatch(digest):
        raise FastArtifactValidationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return digest


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FastArtifactValidationError(
            f"{label} must be an integer of at least {minimum}"
        )
    return value


def _require_string_list(value: Any, label: str) -> tuple[str, ...]:
    rows = _require_list(value, label)
    result = tuple(
        _require_string(item, f"{label}[{index}]")
        for index, item in enumerate(rows)
    )
    return result


def _validate_sidecar(path: Path, digest: str, sidecar_path: Path) -> None:
    if not sidecar_path.is_file():
        raise FastArtifactValidationError(
            f"artifact digest sidecar is missing: {sidecar_path}"
        )
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar_path.read_bytes() != expected:
        raise FastArtifactValidationError(
            f"artifact digest sidecar is invalid: {sidecar_path}"
        )


def _parse_messages(value: Any, label: str) -> tuple[FastProviderMessage, ...]:
    raw_messages = _require_list(value, label)
    if len(raw_messages) != 2:
        raise FastArtifactValidationError(
            f"{label} must contain exactly one system and one user message"
        )
    messages: list[FastProviderMessage] = []
    for index, raw in enumerate(raw_messages):
        item = _require_mapping(raw, f"{label}[{index}]")
        if set(item) != {"role", "content"}:
            raise FastArtifactValidationError(
                f"{label}[{index}] has a noncanonical shape"
            )
        messages.append(
            FastProviderMessage(
                role=_require_string(item.get("role"), f"{label}[{index}].role"),
                content=_require_string(
                    item.get("content"),
                    f"{label}[{index}].content",
                ),
            )
        )
    if tuple(item.role for item in messages) != ("system", "user"):
        raise FastArtifactValidationError(
            f"{label} changed the sealed system/user message order"
        )
    return tuple(messages)


def _parse_user_prompt(
    messages: tuple[FastProviderMessage, ...],
    *,
    expected_dated_question_sha256: str,
    expected_context_sha256: str,
    label: str,
) -> _ParsedUserPrompt:
    message_index = len(messages) - 1
    content = messages[message_index].content
    if not content.startswith(_QA_CONTEXT_PREFIX) or not content.endswith(
        _QA_ANSWER_SUFFIX
    ):
        raise FastArtifactValidationError(
            f"{label} changed the sealed QA user-message framing"
        )
    body = content[len(_QA_CONTEXT_PREFIX) : -len(_QA_ANSWER_SUFFIX)]
    starts: list[int] = []
    offset = 0
    while True:
        index = body.find(_QA_QUESTION_MARKER, offset)
        if index < 0:
            break
        starts.append(index)
        offset = index + 1
    candidates: list[tuple[str, str]] = []
    for index in starts:
        context = body[:index]
        dated_question = body[index + len(_QA_QUESTION_MARKER) :]
        if (
            context
            and dated_question
            and _quote_sha256(context) == expected_context_sha256
            and _quote_sha256(dated_question)
            == expected_dated_question_sha256
        ):
            candidates.append((context, dated_question))
    if len(candidates) != 1:
        raise FastArtifactValidationError(
            f"{label} has {len(candidates)} hash-matched QA framing candidates; "
            "refusing to guess"
        )
    context, dated_question = candidates[0]
    reconstructed = (
        _QA_CONTEXT_PREFIX
        + context
        + _QA_QUESTION_MARKER
        + dated_question
        + _QA_ANSWER_SUFFIX
    )
    if reconstructed != content:
        raise FastArtifactValidationError(
            f"{label} could not be reconstructed byte-for-byte"
        )
    return _ParsedUserPrompt(
        context=context,
        dated_question=dated_question,
        message_index=message_index,
        message_sha256=_quote_sha256(content),
        marker_occurrences=len(starts),
        matching_candidates=len(candidates),
    )


def _recover_raw_question(
    dated_question: str,
    expected_question_sha256: str,
    *,
    label: str,
) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = [(dated_question, "undated")]
    if dated_question.startswith(_DATED_QUESTION_PREFIX):
        end = dated_question.find("]\n", len(_DATED_QUESTION_PREFIX))
        if end >= 0 and end + 2 < len(dated_question):
            candidates.append((dated_question[end + 2 :], "dated_header"))
    matching = tuple(
        (text, form)
        for text, form in candidates
        if _quote_sha256(text) == expected_question_sha256
    )
    if len(matching) != 1:
        raise FastArtifactValidationError(
            f"{label} has {len(matching)} raw-question hash matches; refusing to guess"
        )
    return matching[0]


def _validate_receipt_sha256(receipt: Mapping[str, Any], label: str) -> str:
    declared = _require_digest(receipt.get("receipt_sha256"), f"{label}.receipt_sha256")
    body = dict(receipt)
    body.pop("receipt_sha256")
    if _identity_sha256(body) != declared:
        raise FastArtifactValidationError(f"{label} receipt SHA-256 does not match")
    return declared


def _parse_evidence(value: Any, label: str) -> tuple[FastEvidence, ...]:
    rows = _require_list(value, label)
    evidence: list[FastEvidence] = []
    for index, raw in enumerate(rows):
        item = _require_mapping(raw, f"{label}[{index}]")
        if set(item) != {"evidence_id", "source_id", "text"}:
            raise FastArtifactValidationError(
                f"{label}[{index}] has a noncanonical shape"
            )
        evidence.append(
            FastEvidence(
                evidence_id=_require_string(
                    item.get("evidence_id"), f"{label}[{index}].evidence_id"
                ),
                source_id=_require_string(
                    item.get("source_id"), f"{label}[{index}].source_id"
                ),
                text=_require_string(
                    item.get("text"), f"{label}[{index}].text"
                ),
            )
        )
    ids = tuple(item.evidence_id for item in evidence)
    if len(set(ids)) != len(ids):
        raise FastArtifactValidationError(f"{label} contains duplicate evidence IDs")
    return tuple(evidence)


def _parse_question(
    raw: Mapping[str, Any],
    *,
    ordinal: int,
    population_identity_sha256: str,
    combined_store_receipt_sha256: str,
    retrieval_implementation_sha256: str,
    retrieval_policy_sha256: str,
) -> FastRetrievalQuestion:
    label = f"questions[{ordinal}]"
    if raw.get("format") != QUESTION_FORMAT:
        raise FastArtifactValidationError(f"{label} has an unsupported format")
    if _require_int(raw.get("ordinal"), f"{label}.ordinal") != ordinal:
        raise FastArtifactValidationError(f"{label} changed its ordered ordinal")
    question_id = _require_string(raw.get("question_id"), f"{label}.question_id")
    question_sha = _require_digest(
        raw.get("question_sha256"), f"{label}.question_sha256"
    )
    dated_sha = _require_digest(
        raw.get("dated_question_sha256"), f"{label}.dated_question_sha256"
    )
    if raw.get("population_identity_sha256") != population_identity_sha256:
        raise FastArtifactValidationError(f"{label} changed population identity")
    if raw.get("retrieval_implementation_sha256") != retrieval_implementation_sha256:
        raise FastArtifactValidationError(f"{label} changed retrieval implementation")
    if raw.get("combined_store_receipt_sha256") != combined_store_receipt_sha256:
        raise FastArtifactValidationError(f"{label} changed combined-store identity")
    if raw.get("provider_calls") != 0:
        raise FastArtifactValidationError(f"{label} unexpectedly contains provider calls")
    if tuple(_require_string_list(raw.get("stage_ids"), f"{label}.stage_ids")) != STAGE_IDS:
        raise FastArtifactValidationError(f"{label} changed its cumulative stages")

    predecessor_receipt = _require_mapping(
        raw.get("predecessor_receipt"), f"{label}.predecessor_receipt"
    )
    predecessor_receipt_sha = _validate_receipt_sha256(
        predecessor_receipt, f"{label}.predecessor_receipt"
    )
    packed_chunk_ids = _require_string_list(
        predecessor_receipt.get("packed_chunk_ids"),
        f"{label}.predecessor_receipt.packed_chunk_ids",
    )
    protected_chunk_ids = _require_string_list(
        predecessor_receipt.get("protected_chunk_ids"),
        f"{label}.predecessor_receipt.protected_chunk_ids",
    )
    if (
        not protected_chunk_ids
        or len(set(protected_chunk_ids)) != len(protected_chunk_ids)
    ):
        raise FastArtifactValidationError(
            f"{label}.predecessor_receipt protected chunk IDs must be non-empty "
            "and unique"
        )
    if packed_chunk_ids != protected_chunk_ids:
        raise FastArtifactValidationError(
            f"{label}.predecessor_receipt protected chunk IDs must exactly match "
            "the packed S0 coordinates"
        )
    retrieval_receipt = _require_mapping(
        raw.get("retrieval_receipt"), f"{label}.retrieval_receipt"
    )
    retrieval_receipt_sha = _validate_receipt_sha256(
        retrieval_receipt, f"{label}.retrieval_receipt"
    )
    for receipt_name, receipt in (
        ("predecessor_receipt", predecessor_receipt),
        ("retrieval_receipt", retrieval_receipt),
    ):
        retained = _require_int(
            receipt.get("retained_request_token_state_bytes"),
            f"{label}.{receipt_name}.retained_request_token_state_bytes",
        )
        if retained != 0:
            raise FastArtifactValidationError(
                f"{label}.{receipt_name} persisted transformer request state"
            )
    if predecessor_receipt.get("retrieval_policy_sha256") != retrieval_policy_sha256:
        raise FastArtifactValidationError(
            f"{label}.predecessor_receipt changed retrieval policy"
        )

    raw_stages = _require_list(raw.get("stages"), f"{label}.stages")
    if len(raw_stages) != len(STAGE_IDS):
        raise FastArtifactValidationError(f"{label} changed its cumulative stage count")

    intermediate: list[_ParsedStage] = []
    parent_ids: tuple[str, ...] = ()
    parent_receipt_sha: str | None = None
    dated_question: str | None = None
    evidence_by_id: dict[str, FastEvidence] = {}
    ladder_controls_sha: str | None = None
    ladder_budgets: tuple[int, int, int] | None = None
    for stage_index, expected_stage_id in enumerate(STAGE_IDS):
        stage_label = f"{label}.stages[{stage_index}]"
        stage = _require_mapping(raw_stages[stage_index], stage_label)
        if set(stage) != {
            "stage_id",
            "stage_receipt",
            "provider_messages",
            "evidence",
        }:
            raise FastArtifactValidationError(f"{stage_label} has a noncanonical shape")
        stage_id = _require_string(stage.get("stage_id"), f"{stage_label}.stage_id")
        if stage_id != expected_stage_id:
            raise FastArtifactValidationError(f"{stage_label} changed the stage order")
        receipt = _require_mapping(stage.get("stage_receipt"), f"{stage_label}.receipt")
        receipt_sha = _validate_receipt_sha256(receipt, f"{stage_label}.receipt")
        if receipt.get("format") != CUMULATIVE_STAGE_FORMAT:
            raise FastArtifactValidationError(
                f"{stage_label} receipt has an unsupported format"
            )
        if receipt.get("stage_id") != stage_id:
            raise FastArtifactValidationError(f"{stage_label} receipt changed stage ID")
        matched_controls_sha = _require_digest(
            receipt.get("matched_controls_sha256"),
            f"{stage_label}.receipt.matched_controls_sha256",
        )
        _require_digest(
            receipt.get("method_evidence_sha256"),
            f"{stage_label}.receipt.method_evidence_sha256",
        )

        evidence = _parse_evidence(stage.get("evidence"), f"{stage_label}.evidence")
        evidence_ids = tuple(item.evidence_id for item in evidence)
        for item in evidence:
            previous = evidence_by_id.setdefault(item.evidence_id, item)
            if previous != item:
                raise FastArtifactValidationError(
                    f"{stage_label} changed evidence payload {item.evidence_id!r}"
                )
        selected_ids = _require_string_list(
            receipt.get("selected_evidence_ids"),
            f"{stage_label}.receipt.selected_evidence_ids",
        )
        declared_parent_ids = _require_string_list(
            receipt.get("parent_evidence_ids"),
            f"{stage_label}.receipt.parent_evidence_ids",
        )
        added_ids = _require_string_list(
            receipt.get("added_evidence_ids"),
            f"{stage_label}.receipt.added_evidence_ids",
        )
        if selected_ids != evidence_ids:
            raise FastArtifactValidationError(
                f"{stage_label} evidence IDs differ from its receipt"
            )
        if evidence_ids[: len(parent_ids)] != parent_ids:
            raise FastArtifactValidationError(
                f"{stage_label} is not an ordered-prefix cumulative extension"
            )
        if declared_parent_ids != parent_ids:
            raise FastArtifactValidationError(
                f"{stage_label} changed its parent evidence IDs"
            )
        if added_ids != evidence_ids[len(parent_ids) :]:
            raise FastArtifactValidationError(
                f"{stage_label} changed its added-evidence suffix"
            )
        declared_parent_receipt = receipt.get("parent_stage_receipt_sha256")
        if declared_parent_receipt != parent_receipt_sha:
            raise FastArtifactValidationError(
                f"{stage_label} changed its parent receipt binding"
            )
        admission_status = _require_string(
            receipt.get("admission_status"),
            f"{stage_label}.receipt.admission_status",
        )
        if stage_index == 0:
            if admission_status != "root" or added_ids != evidence_ids:
                raise FastArtifactValidationError(
                    f"{stage_label} is not a valid cumulative root"
                )
        elif added_ids:
            if admission_status != "added":
                raise FastArtifactValidationError(
                    f"{stage_label} added evidence without an added status"
                )
        elif admission_status not in {"no_novel_evidence", "budget_exhausted"}:
            raise FastArtifactValidationError(
                f"{stage_label} has no additions without an explicit no-op status"
            )

        messages = _parse_messages(
            stage.get("provider_messages"), f"{stage_label}.provider_messages"
        )
        prompt_messages_sha = _require_digest(
            receipt.get("prompt_messages_sha256"),
            f"{stage_label}.receipt.prompt_messages_sha256",
        )
        message_payload = [
            {"role": item.role, "content": item.content} for item in messages
        ]
        if _identity_sha256(message_payload) != prompt_messages_sha:
            raise FastArtifactValidationError(
                f"{stage_label} provider messages differ from their receipt"
            )
        context_sha = _require_digest(
            receipt.get("context_sha256"), f"{stage_label}.receipt.context_sha256"
        )
        parsed = _parse_user_prompt(
            messages,
            expected_dated_question_sha256=dated_sha,
            expected_context_sha256=context_sha,
            label=f"{stage_label}.provider_messages[-1]",
        )
        if dated_question is None:
            dated_question = parsed.dated_question
        elif parsed.dated_question != dated_question:
            raise FastArtifactValidationError(
                f"{stage_label} changed the exact dated question"
            )

        context_tokens = _require_int(
            receipt.get("context_token_proxy"),
            f"{stage_label}.receipt.context_token_proxy",
        )
        max_context_tokens = _require_int(
            receipt.get("max_context_token_proxy"),
            f"{stage_label}.receipt.max_context_token_proxy",
        )
        prompt_tokens = _require_int(
            receipt.get("prompt_token_proxy"),
            f"{stage_label}.receipt.prompt_token_proxy",
        )
        max_prompt_tokens = _require_int(
            receipt.get("max_prompt_token_proxy"),
            f"{stage_label}.receipt.max_prompt_token_proxy",
        )
        output_reserve = _require_int(
            receipt.get("responder_output_token_reserve"),
            f"{stage_label}.receipt.responder_output_token_reserve",
        )
        if context_tokens > max_context_tokens or prompt_tokens > max_prompt_tokens:
            raise FastArtifactValidationError(f"{stage_label} exceeds its hard token cap")
        stage_budgets = (max_context_tokens, max_prompt_tokens, output_reserve)
        if ladder_controls_sha is None:
            ladder_controls_sha = matched_controls_sha
            ladder_budgets = stage_budgets
        elif (
            matched_controls_sha != ladder_controls_sha
            or stage_budgets != ladder_budgets
        ):
            raise FastArtifactValidationError(
                f"{stage_label} changed cumulative controls or hard budgets"
            )
        evidence_projection_sha = _require_digest(
            receipt.get("evidence_projection_sha256"),
            f"{stage_label}.receipt.evidence_projection_sha256",
        )
        intermediate.append(
            _ParsedStage(
                stage_id=stage_id,
                receipt_sha256=receipt_sha,
                matched_controls_sha256=matched_controls_sha,
                evidence_projection_sha256=evidence_projection_sha,
                context_sha256=context_sha,
                prompt_messages_sha256=prompt_messages_sha,
                context_token_proxy=context_tokens,
                max_context_token_proxy=max_context_tokens,
                prompt_token_proxy=prompt_tokens,
                max_prompt_token_proxy=max_prompt_tokens,
                responder_output_token_reserve=output_reserve,
                admission_status=admission_status,
                added_evidence_ids=added_ids,
                context=parsed.context,
                evidence=evidence,
                messages=messages,
                parsed_prompt=parsed,
            )
        )
        parent_ids = evidence_ids
        parent_receipt_sha = receipt_sha

    assert dated_question is not None
    question, question_form = _recover_raw_question(
        dated_question, question_sha, label=label
    )

    feature_rows: list[FastFeatureRow] = []
    feature_index_by_input: dict[tuple[str, str], int] = {}
    stages: list[FastRetrievalStage] = []
    for row in intermediate:
        evidence = row.evidence
        feature_indices: list[int] = []
        for item in evidence:
            key = (question, item.text)
            index = feature_index_by_input.get(key)
            if index is None:
                index = len(feature_rows)
                feature_index_by_input[key] = index
                feature_rows.append(
                    FastFeatureRow(
                        question=question,
                        evidence_text=item.text,
                        row_sha256=_identity_sha256(
                            {
                                "format": "memory-condense-fast-feature-row-v1",
                                "question": question,
                                "evidence_text": item.text,
                            }
                        ),
                    )
                )
            feature_indices.append(index)
        stages.append(
            FastRetrievalStage(
                stage_id=row.stage_id,
                stage_receipt_sha256=row.receipt_sha256,
                matched_controls_sha256=row.matched_controls_sha256,
                evidence_projection_sha256=row.evidence_projection_sha256,
                context_sha256=row.context_sha256,
                prompt_messages_sha256=row.prompt_messages_sha256,
                context_token_proxy=row.context_token_proxy,
                max_context_token_proxy=row.max_context_token_proxy,
                prompt_token_proxy=row.prompt_token_proxy,
                max_prompt_token_proxy=row.max_prompt_token_proxy,
                responder_output_token_reserve=row.responder_output_token_reserve,
                admission_status=row.admission_status,
                added_evidence_ids=row.added_evidence_ids,
                context=row.context,
                evidence=evidence,
                provider_messages=row.messages,
                feature_row_indices=tuple(feature_indices),
            )
        )

    first_stage = stages[0]
    final_stage = stages[-1]
    if len(protected_chunk_ids) != len(first_stage.evidence):
        raise FastArtifactValidationError(
            f"{label}.predecessor_receipt protected chunk coordinates do not "
            "match the S0 evidence cardinality"
        )
    if (
        predecessor_receipt.get("prompt_messages_sha256")
        != first_stage.prompt_messages_sha256
        or predecessor_receipt.get("prompt_token_proxy")
        != first_stage.prompt_token_proxy
        or predecessor_receipt.get("max_prompt_token_proxy")
        != first_stage.max_prompt_token_proxy
        or predecessor_receipt.get("responder_output_token_reserve")
        != first_stage.responder_output_token_reserve
        or predecessor_receipt.get("protected_context_sha256")
        != first_stage.context_sha256
    ):
        raise FastArtifactValidationError(
            f"{label}.predecessor_receipt does not bind the S0 prompt"
        )
    expected_retrieval_values: tuple[tuple[str, object], ...] = (
        ("predecessor_receipt_sha256", predecessor_receipt_sha),
        ("protected_chunk_ids", protected_chunk_ids),
        ("protected_evidence_ids", first_stage.evidence_ids),
        ("final_evidence_ids", final_stage.evidence_ids),
        ("final_context_sha256", final_stage.context_sha256),
        ("prompt_messages_sha256", final_stage.prompt_messages_sha256),
        ("context_token_proxy", final_stage.context_token_proxy),
        ("max_context_token_proxy", final_stage.max_context_token_proxy),
        ("prompt_token_proxy", final_stage.prompt_token_proxy),
        ("max_prompt_token_proxy", final_stage.max_prompt_token_proxy),
        ("responder_output_token_reserve", final_stage.responder_output_token_reserve),
        ("matched_controls_sha256", final_stage.matched_controls_sha256),
        (
            "stage_admission_statuses",
            tuple(stage.admission_status for stage in stages[1:]),
        ),
    )
    for field_name, expected_value in expected_retrieval_values:
        observed_value = retrieval_receipt.get(field_name)
        if isinstance(expected_value, tuple):
            try:
                observed_value = tuple(observed_value)
            except TypeError as exc:
                raise FastArtifactValidationError(
                    f"{label}.retrieval_receipt.{field_name} is not a sequence"
                ) from exc
        if observed_value != expected_value:
            raise FastArtifactValidationError(
                f"{label}.retrieval_receipt does not bind final field {field_name}"
            )

    final_parsed = intermediate[-1].parsed_prompt
    final_messages = intermediate[-1].messages
    parse_receipt = FastQuestionParseReceipt(
        framing="memory-condense-qa-user-template-v1",
        source_stage_id=STAGE_IDS[-1],
        provider_message_index=final_parsed.message_index,
        provider_message_sha256=final_parsed.message_sha256,
        question_marker_occurrences=final_parsed.marker_occurrences,
        matching_framing_candidates=final_parsed.matching_candidates,
        dated_question_sha256=dated_sha,
        question_sha256=question_sha,
        question_form=question_form,
    )
    return FastRetrievalQuestion(
        ordinal=ordinal,
        question_id=question_id,
        question_sha256=question_sha,
        dated_question_sha256=dated_sha,
        predecessor_receipt_sha256=predecessor_receipt_sha,
        retrieval_receipt_sha256=retrieval_receipt_sha,
        protected_chunk_ids=protected_chunk_ids,
        retained_request_token_state_bytes=0,
        question=question,
        dated_question=dated_question,
        final_user_message=final_messages[final_parsed.message_index],
        question_parse_receipt=parse_receipt,
        feature_rows=tuple(feature_rows),
        stages=tuple(stages),
    )


def load_fast_retrieval_artifact(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    verify_sidecar: bool = True,
    sidecar_path: str | Path | None = None,
) -> FastRetrievalArtifact:
    """Load and validate a sealed retrieval artifact without other I/O.

    The artifact bytes are read once and hashed before JSON parsing.  By
    default the sibling ``retrieval.json.sha256`` is required and checked.
    Callers may additionally pin ``expected_sha256``.  Disabling the sidecar
    requires an explicit expected digest, so an unanchored JSON document can
    never be mistaken for a sealed artifact.
    """

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if expected_sha256 is not None:
        expected_sha256 = _require_digest(expected_sha256, "expected_sha256")
    if not verify_sidecar and expected_sha256 is None:
        raise FastArtifactValidationError(
            "an expected SHA-256 is required when sidecar verification is disabled"
        )
    if not verify_sidecar and sidecar_path is not None:
        raise ValueError("sidecar_path requires verify_sidecar=True")

    raw_bytes = artifact_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and raw_sha256 != expected_sha256:
        raise FastArtifactValidationError(
            f"artifact SHA-256 mismatch ({raw_sha256} != {expected_sha256})"
        )
    if verify_sidecar:
        selected_sidecar = (
            Path(sidecar_path)
            if sidecar_path is not None
            else artifact_path.with_name(artifact_path.name + ".sha256")
        )
        _validate_sidecar(artifact_path, raw_sha256, selected_sidecar)

    try:
        payload = json.loads(raw_bytes, parse_constant=_reject_nonfinite_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FastArtifactValidationError("artifact is not valid UTF-8 JSON") from exc
    root = _require_mapping(payload, "artifact")
    if raw_bytes != _canonical_json_bytes(root):
        raise FastArtifactValidationError("artifact is not canonical JSON")
    if root.get("format") != RETRIEVAL_FORMAT:
        raise FastArtifactValidationError("artifact has an unsupported format")
    if root.get("campaign_format") != CAMPAIGN_FORMAT:
        raise FastArtifactValidationError("artifact has an unsupported campaign format")
    if root.get("gold_fields_present") is not False:
        raise FastArtifactValidationError("fast artifact must remain gold-blind")
    if root.get("provider_calls") != 0:
        raise FastArtifactValidationError("retrieval artifact contains provider calls")

    stage_ids = _require_string_list(root.get("stage_ids"), "artifact.stage_ids")
    if stage_ids != STAGE_IDS:
        raise FastArtifactValidationError("artifact changed the ordered S0-S3 stages")
    population_sha = _require_digest(
        root.get("population_identity_sha256"),
        "artifact.population_identity_sha256",
    )
    population = _require_mapping(
        root.get("population_identity"), "artifact.population_identity"
    )
    if population.get("format") != POPULATION_FORMAT:
        raise FastArtifactValidationError("artifact has an unsupported population")
    if _identity_sha256(population) != population_sha:
        raise FastArtifactValidationError("population identity SHA-256 does not match")
    retrieval_implementation_sha = _require_digest(
        root.get("retrieval_implementation_sha256"),
        "artifact.retrieval_implementation_sha256",
    )
    retrieval_policy_sha = _require_digest(
        root.get("retrieval_policy_sha256"),
        "artifact.retrieval_policy_sha256",
    )
    transcript_tokens = _require_int(
        root.get("transcript_tokens"), "artifact.transcript_tokens", minimum=1
    )
    turn_count = _require_int(root.get("turn_count"), "artifact.turn_count", minimum=1)

    source_store_receipt = _require_mapping(
        root.get("source_store_receipt"), "artifact.source_store_receipt"
    )
    source_store_receipt_sha = _validate_receipt_sha256(
        source_store_receipt, "artifact.source_store_receipt"
    )
    combined_store_receipt = _require_mapping(
        root.get("combined_store_receipt"), "artifact.combined_store_receipt"
    )
    combined_store_receipt_sha = _validate_receipt_sha256(
        combined_store_receipt, "artifact.combined_store_receipt"
    )
    retained_state_bytes = _require_int(
        combined_store_receipt.get("retained_request_token_state_bytes"),
        "artifact.combined_store_receipt.retained_request_token_state_bytes",
    )
    if retained_state_bytes != 0:
        raise FastArtifactValidationError(
            "combined-store receipt persisted transformer request state"
        )
    if (
        source_store_receipt.get("turn_count") != turn_count
        or combined_store_receipt.get("turn_count") != turn_count
        or combined_store_receipt.get("retrieval_policy_sha256")
        != retrieval_policy_sha
        or combined_store_receipt.get("compilation_receipt_sha256")
        != root.get("compilation_receipt_sha256")
    ):
        raise FastArtifactValidationError(
            "artifact store receipts do not bind its population or policy"
        )

    raw_questions = _require_list(root.get("questions"), "artifact.questions")
    question_count = _require_int(
        root.get("question_count"), "artifact.question_count", minimum=1
    )
    if question_count != len(raw_questions):
        raise FastArtifactValidationError("artifact question count changed")
    part_hashes = _require_string_list(
        root.get("question_part_sha256s"), "artifact.question_part_sha256s"
    )
    if len(part_hashes) != question_count:
        raise FastArtifactValidationError("artifact question-part count changed")

    questions: list[FastRetrievalQuestion] = []
    observed_question_ids: set[str] = set()
    for ordinal, (raw_question, expected_part_sha) in enumerate(
        zip(raw_questions, part_hashes, strict=True)
    ):
        question_mapping = _require_mapping(raw_question, f"questions[{ordinal}]")
        observed_part_sha = hashlib.sha256(
            _canonical_json_bytes(question_mapping)
        ).hexdigest()
        expected_part_sha = _require_digest(
            expected_part_sha, f"artifact.question_part_sha256s[{ordinal}]"
        )
        if observed_part_sha != expected_part_sha:
            raise FastArtifactValidationError(
                f"questions[{ordinal}] no longer matches its part SHA-256"
            )
        question = _parse_question(
            question_mapping,
            ordinal=ordinal,
            population_identity_sha256=population_sha,
            combined_store_receipt_sha256=combined_store_receipt_sha,
            retrieval_implementation_sha256=retrieval_implementation_sha,
            retrieval_policy_sha256=retrieval_policy_sha,
        )
        if question.question_id in observed_question_ids:
            raise FastArtifactValidationError("artifact contains duplicate question IDs")
        observed_question_ids.add(question.question_id)
        questions.append(question)

    if (
        population.get("transcript_tokens") != transcript_tokens
        or population.get("turn_count") != turn_count
        or population.get("question_count") != question_count
        or population.get("archived_compiled_sample_sha256")
        != root.get("archived_compiled_sample_sha256")
    ):
        raise FastArtifactValidationError(
            "population identity does not bind artifact dimensions"
        )
    expected_id_hashes = tuple(
        _identity_sha256({"question_id": question.question_id})
        for question in questions
    )
    observed_id_hashes = _require_string_list(
        population.get("ordered_question_id_sha256s"),
        "artifact.population_identity.ordered_question_id_sha256s",
    )
    expected_probe_hashes = tuple(
        _identity_sha256(
            {
                "question_id": question.question_id,
                "question_sha256": question.question_sha256,
                "dated_question_sha256": question.dated_question_sha256,
            }
        )
        for question in questions
    )
    observed_probe_hashes = _require_string_list(
        population.get("ordered_question_probe_sha256s"),
        "artifact.population_identity.ordered_question_probe_sha256s",
    )
    if (
        observed_id_hashes != expected_id_hashes
        or observed_probe_hashes != expected_probe_hashes
    ):
        raise FastArtifactValidationError(
            "population identity changed ordered question provenance"
        )

    return FastRetrievalArtifact(
        source_path=str(artifact_path.resolve()),
        raw_sha256=raw_sha256,
        format=RETRIEVAL_FORMAT,
        campaign_format=CAMPAIGN_FORMAT,
        population_identity_sha256=population_sha,
        source_store_receipt_sha256=source_store_receipt_sha,
        combined_store_receipt_sha256=combined_store_receipt_sha,
        retrieval_implementation_sha256=retrieval_implementation_sha,
        retrieval_policy_sha256=retrieval_policy_sha,
        transcript_tokens=transcript_tokens,
        turn_count=turn_count,
        retained_request_token_state_bytes=retained_state_bytes,
        stage_ids=stage_ids,
        questions=tuple(questions),
    )


__all__ = [
    "CAMPAIGN_FORMAT",
    "FastArtifactValidationError",
    "FastEvidence",
    "FastFeatureRow",
    "FastProviderMessage",
    "FastQuestionParseReceipt",
    "FastRetrievalArtifact",
    "FastRetrievalQuestion",
    "FastRetrievalStage",
    "ORIGINAL_1M_RETRIEVAL_SHA256",
    "QUESTION_FORMAT",
    "RETRIEVAL_FORMAT",
    "STAGE_IDS",
    "load_fast_retrieval_artifact",
]
