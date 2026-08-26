#!/usr/bin/env python3
"""Run the isolated locked S0-plus-genuine-CAV-links mechanism arm.

The feature phase mirrors the exact S0 packet into the canonical four-stage
feature-session surface, encodes the globally deduplicated question/evidence
texts in one Qwen call, and persists only tensor-free genuine CAV-v2 receipts.
The mirrored stages never add evidence and the X/X1 ordering readout is not
consumed by this arm.  The answer phase exposes only the bounded CAV link guide
over exact S0 membership and preserves the sealed S0 prediction whenever that
guide is unavailable or invalid.

No phase loads benchmark gold, categories, source topology, or oracle labels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # support ``python tools/run_...py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._artifact_json import canonical_json_bytes
from memory_condense.eval.fast_cav_feature_artifact import (
    FAST_CAV_FEATURE_ARTIFACT_FORMAT,
    load_fast_cav_feature_artifact,
)
from memory_condense.eval.fast_cav_feature_session import (
    FAST_CAV_MAX_ENCODER_ROWS,
    FastCAVFeatureSessionReceipt,
    run_fast_cav_feature_session,
)
from memory_condense.eval.fast_cav_link_synthesis import (
    FAST_CAV_LINK_SYNTHESIS_MAX_COMPLETION_TOKENS,
    FAST_CAV_LINK_SYNTHESIS_MAX_PROMPT_TOKENS,
    FastCAVLinkSynthesisArmPrompt,
    FastCAVLinkSynthesisStageReceipt,
    _render_linked_guide,
    build_fast_cav_link_synthesis_population,
    parse_fast_cav_link_synthesis,
)
from memory_condense.eval.fast_cav_links import (
    FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT,
    FastCAVLinkReceipt,
)
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    CAMPAIGN_FORMAT,
    RETRIEVAL_FORMAT,
    STAGE_IDS,
    FastEvidence,
    FastFeatureRow,
    FastProviderMessage,
    FastQuestionParseReceipt,
    FastRetrievalArtifact,
    FastRetrievalQuestion,
    FastRetrievalStage,
)
from memory_condense.eval.run_fast_1m_cav import (
    FEATURE_MANIFEST_FORMAT,
    ZERO_STATE_CONTRACT,
    _orders_from_session,
    _orders_payload,
)
from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    expected_prefix_checkpoint_sha256,
)
from memory_condense.search.fusion.tensor_identity import (
    canonical_float32_tensor,
)
from tools import run_locked_retrieval_mechanism_arm as s0_runner
from tools.run_routed_full_source_repair import (
    _distribution,
    _make_provider_client,
    _publish,
    _read,
    _record_by_messages,
    _stable_batch,
)


ARM_LABEL = "S0_PLUS_CAV_LINKS"
PARENT_ARM_LABEL = "S0_CONTROL"
PREFLIGHT_FORMAT = "memory-condense-locked-s0-cav-links-preflight-v1"
ANSWER_PREFLIGHT_FORMAT = (
    "memory-condense-locked-s0-cav-links-answer-preflight-v1"
)
RUN_FORMAT = "memory-condense-locked-retrieval-mechanism-arm-run-v1"
ADAPTER_FORMAT = "memory-condense-locked-s0-cav-fast-artifact-adapter-v1"
FEATURE_BINDING_FORMAT = "memory-condense-locked-s0-cav-feature-plan-v1"
ENCODER_EXECUTION_FORMAT = (
    "memory-condense-locked-s0-cav-encoder-execution-v1"
)

DEFAULT_RETRIEVAL = s0_runner.DEFAULT_RETRIEVAL
DEFAULT_BASELINE_ANSWERS = s0_runner.DEFAULT_BASELINE_ANSWERS
DEFAULT_S0_RUN = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-control-v1/run.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826"
    "/s0-plus-cav-links-v1"
)
DEFAULT_MODEL_DIR = Path(".cache/models/Qwen3-8B")
DEFAULT_EVENT_CAV = Path(
    "eval_results/qwen3_event_membership_cav_probe.safetensors"
)
DEFAULT_PREFIX_CAV = Path("eval_results/qwen3_prefix_cav_probe.safetensors")
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"

EXPECTED_RETRIEVAL_SHA256 = s0_runner.EXPECTED_RETRIEVAL_SHA256
EXPECTED_BASELINE_ANSWERS_SHA256 = s0_runner.EXPECTED_BASELINE_ANSWERS_SHA256
EXPECTED_QUESTION_COUNT = 100
FEATURE_LAYER = 0
FEATURE_PREFIX_LAYERS = 1
CONCEPT_COUNT = 3
EXTRACTION_LINKS_PER_CONCEPT = 4
MAX_GUIDE_TOKENS = 256
MAX_PROMPT_TOKENS = 8_000
MAX_ANSWER_OUTPUT_TOKENS = 256
_CAV_SELECTION_KEYS = (
    "autobiographical_completed_event.layer_0",
    "context_dependency.layer_0",
    "binding_constraint.layer_0",
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer_session_ids",
        "category",
        "evidence_sources",
        "gold",
        "gold_answer",
        "reference",
        "reference_answer",
        "source_completeness",
        "source_topology",
    }
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_KEYS
            or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an exact lowercase SHA-256 digest")
    return value


def _validate_common(args: argparse.Namespace) -> None:
    if args.gateway_url != DEFAULT_GATEWAY_URL or args.model != DEFAULT_MODEL:
        raise ValueError("CAV arm requires the locked central-dev Terra route")
    if args.expected_question_count != EXPECTED_QUESTION_COUNT:
        raise ValueError("CAV arm requires the exact locked-100 population")
    if type(args.max_concurrency) is not int or args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be positive")
    if type(args.batch_size) is not int or args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.device != "cuda" or args.dtype != "bfloat16":
        raise ValueError("CAV feature runtime requires cuda/bfloat16 Qwen")
    if (
        args.extraction_temperature != 0.05
        or args.reinjection_temperature != 0.05
        or args.alpha != 1.0
    ):
        raise ValueError("CAV router parameters changed from the locked runtime")


def _verified_s0(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], str]:
    """Run the S0 historical validator once and replay its exact journals."""

    plan = s0_runner._prepare(
        retrieval_path=Path(args.retrieval),
        baseline_answers_path=Path(args.baseline_answers),
        expected_retrieval_sha256=str(args.expected_retrieval_sha256),
        expected_baseline_answers_sha256=str(
            args.expected_baseline_answers_sha256
        ),
        expected_question_count=int(args.expected_question_count),
    )
    run_path = Path(args.s0_run)
    source, source_sha = s0_runner._read(run_path)
    if source_sha != _require_sha256(
        args.expected_s0_run_sha256,
        "S0 run expected SHA-256",
    ):
        raise ValueError("S0 run artifact SHA-256 changed")
    batch = s0_runner._runtime(
        plan,
        checkpoint_dir=Path(
            args.s0_checkpoint_dir or run_path.parent / "terra-answer-calls"
        ),
        client=None,
        max_concurrency=int(args.max_concurrency),
    ).run()
    if batch.usage.physical_calls or batch.usage.checkpoint_hits != plan.exact_calls:
        raise RuntimeError("S0 replay did not consume its complete journal set")
    expected = s0_runner._run_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("S0 run differs from immutable runtime journals")
    return plan, source, source_sha


def _raw_question(dated: str, expected_sha256: str) -> tuple[str, str]:
    candidates = [(dated, "undated")]
    if dated.startswith("[Question asked at "):
        end = dated.find("]\n")
        if end >= 0:
            candidates.append((dated[end + 2 :], "dated_header"))
    matches = [
        (text, form)
        for text, form in candidates
        if quote_sha256(text) == expected_sha256
    ]
    if len(matches) != 1:
        raise ValueError("dated question cannot recover one sealed raw question")
    return matches[0]


def _feature_row_sha256(question: str, evidence_text: str) -> str:
    return identity_sha256(
        {
            "format": "memory-condense-fast-feature-row-v1",
            "question": question,
            "evidence_text": evidence_text,
        }
    )


def _adapter_artifact(
    plan: Any,
    s0_run: Mapping[str, Any],
    s0_run_sha256: str,
) -> tuple[FastRetrievalArtifact, dict[str, Any]]:
    s0_rows = s0_run.get("questions")
    if not isinstance(s0_rows, list) or len(s0_rows) != len(plan.rows):
        raise ValueError("S0 run question population changed")
    question_bindings: list[dict[str, Any]] = []
    for row, locked, prediction in zip(
        plan.rows,
        plan.population.rows,
        s0_rows,
        strict=True,
    ):
        evidence = locked.question.stages[0].evidence
        prediction_body = prediction.get("prediction")
        if (
            not isinstance(prediction_body, Mapping)
            or prediction.get("source_binding_sha256") != row.binding_sha256
        ):
            raise ValueError("S0 prediction/source binding changed")
        coordinates = [
            {
                "evidence_id": item.evidence_id,
                "source_id": item.source_id,
                "text_sha256": quote_sha256(item.text),
            }
            for item in evidence
        ]
        question_bindings.append(
            {
                "ordinal": row.ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "dated_question_sha256": row.dated_question_sha256,
                "retrieval_question_part_sha256": (
                    row.retrieval_question_part_sha256
                ),
                "s0_source_binding_sha256": row.binding_sha256,
                "s0_stage_receipt_sha256": row.stage_receipt_sha256,
                "s0_evidence_projection_sha256": (
                    row.evidence_projection_sha256
                ),
                "s0_provider_messages_sha256": (
                    row.provider_messages_sha256
                ),
                "s0_prompt_token_proxy": row.prompt_token_proxy,
                "s0_prediction_sha256": prediction_body["sha256"],
                "ordered_s0_coordinates_sha256": identity_sha256(coordinates),
                "ordered_s0_evidence_ids": [item.evidence_id for item in evidence],
            }
        )
    binding: dict[str, Any] = {
        "format": ADAPTER_FORMAT,
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "retrieval_sha256": plan.population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            plan.population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": plan.population.population_identity_sha256,
        "historical_validator_binding_sha256": plan.population.binding_sha256,
        "s0_control_run_sha256": s0_run_sha256,
        "adapter_policy": {
            "canonical_fast_stage_count": len(STAGE_IDS),
            "all_fast_stages_mirror_exact_s0_membership_and_order": True,
            "evidence_additions_after_s0": 0,
            "answer_consumes_genuine_link_receipts_only": True,
            "x_x1_ordering_proxy_consumed": False,
        },
        "question_bindings": question_bindings,
    }
    binding["adapter_artifact_sha256"] = identity_sha256(binding)
    artifact_sha = binding["adapter_artifact_sha256"]
    questions: list[FastRetrievalQuestion] = []
    for row, locked in zip(plan.rows, plan.population.rows, strict=True):
        question = locked.question
        raw_question, question_form = _raw_question(
            question.dated_question,
            question.question_sha256,
        )
        s0_evidence = tuple(question.stages[0].evidence)
        text_to_index: dict[str, int] = {}
        feature_rows: list[FastFeatureRow] = []
        feature_indices: list[int] = []
        for evidence in s0_evidence:
            index = text_to_index.get(evidence.text)
            if index is None:
                index = len(feature_rows)
                text_to_index[evidence.text] = index
                feature_rows.append(
                    FastFeatureRow(
                        question=raw_question,
                        evidence_text=evidence.text,
                        row_sha256=_feature_row_sha256(
                            raw_question,
                            evidence.text,
                        ),
                    )
                )
            feature_indices.append(index)
        messages = tuple(
            FastProviderMessage(role=item["role"], content=item["content"])
            for item in row.messages
        )
        synthetic_stages: list[FastRetrievalStage] = []
        for stage_ordinal, stage_id in enumerate(STAGE_IDS):
            stage_receipt = (
                row.stage_receipt_sha256
                if stage_ordinal == 0
                else identity_sha256(
                    {
                        "format": ADAPTER_FORMAT + "-stage-mirror-v1",
                        "adapter_artifact_sha256": artifact_sha,
                        "question_id": row.question_id,
                        "stage_id": stage_id,
                        "source_s0_stage_receipt_sha256": (
                            row.stage_receipt_sha256
                        ),
                    }
                )
            )
            synthetic_stages.append(
                FastRetrievalStage(
                    stage_id=stage_id,
                    stage_receipt_sha256=stage_receipt,
                    matched_controls_sha256=identity_sha256(
                        {
                            "format": ADAPTER_FORMAT + "-controls-v1",
                            "source_s0_stage_receipt_sha256": (
                                row.stage_receipt_sha256
                            ),
                        }
                    ),
                    evidence_projection_sha256=(
                        row.evidence_projection_sha256
                    ),
                    context_sha256=quote_sha256(messages[-1].content),
                    prompt_messages_sha256=row.provider_messages_sha256,
                    context_token_proxy=0,
                    max_context_token_proxy=MAX_PROMPT_TOKENS,
                    prompt_token_proxy=row.prompt_token_proxy,
                    max_prompt_token_proxy=MAX_PROMPT_TOKENS,
                    responder_output_token_reserve=MAX_ANSWER_OUTPUT_TOKENS,
                    admission_status=(
                        "root" if stage_ordinal == 0 else "no_novel_evidence"
                    ),
                    added_evidence_ids=(
                        tuple(item.evidence_id for item in s0_evidence)
                        if stage_ordinal == 0
                        else ()
                    ),
                    context="exact sealed S0 packet",
                    evidence=s0_evidence,
                    provider_messages=messages,
                    feature_row_indices=tuple(feature_indices),
                )
            )
        final_user = messages[-1]
        questions.append(
            FastRetrievalQuestion(
                ordinal=row.ordinal,
                question_id=row.question_id,
                question_sha256=row.question_sha256,
                dated_question_sha256=row.dated_question_sha256,
                predecessor_receipt_sha256=row.stage_receipt_sha256,
                retrieval_receipt_sha256=identity_sha256(
                    {
                        "format": ADAPTER_FORMAT + "-question-v1",
                        "adapter_artifact_sha256": artifact_sha,
                        "question_id": row.question_id,
                        "s0_source_binding_sha256": row.binding_sha256,
                    }
                ),
                protected_chunk_ids=tuple(
                    item.evidence_id for item in s0_evidence
                ),
                retained_request_token_state_bytes=0,
                question=raw_question,
                dated_question=question.dated_question,
                final_user_message=final_user,
                question_parse_receipt=FastQuestionParseReceipt(
                    framing="locked_s0_control_fast_adapter_v1",
                    source_stage_id=STAGE_IDS[-1],
                    provider_message_index=len(messages) - 1,
                    provider_message_sha256=quote_sha256(final_user.content),
                    question_marker_occurrences=final_user.content.count(
                        "\n\nQuestion: "
                    ),
                    matching_framing_candidates=1,
                    dated_question_sha256=row.dated_question_sha256,
                    question_sha256=row.question_sha256,
                    question_form=question_form,
                ),
                feature_rows=tuple(feature_rows),
                stages=tuple(synthetic_stages),
            )
        )
    artifact = FastRetrievalArtifact(
        source_path=str(DEFAULT_S0_RUN),
        raw_sha256=artifact_sha,
        format=RETRIEVAL_FORMAT,
        campaign_format=CAMPAIGN_FORMAT,
        population_identity_sha256=plan.population.population_identity_sha256,
        source_store_receipt_sha256=identity_sha256(
            {"format": ADAPTER_FORMAT, "source": "locked-validation-shards"}
        ),
        combined_store_receipt_sha256=identity_sha256(
            {"format": ADAPTER_FORMAT, "source": "locked-merged-retrieval"}
        ),
        retrieval_implementation_sha256=identity_sha256(
            {"format": ADAPTER_FORMAT, "source": "tool-only-adapter"}
        ),
        retrieval_policy_sha256=identity_sha256(binding["adapter_policy"]),
        transcript_tokens=1,
        turn_count=1,
        retained_request_token_state_bytes=0,
        stage_ids=STAGE_IDS,
        questions=tuple(questions),
    )
    if _contains_forbidden_key(binding):
        raise RuntimeError("CAV adapter crossed the gold firewall")
    return artifact, binding


def _feature_plan(
    args: argparse.Namespace,
    artifact: FastRetrievalArtifact,
    adapter_binding: Mapping[str, Any],
) -> dict[str, Any]:
    event_path = Path(args.event_cav)
    prefix_path = Path(args.prefix_cav)
    if not event_path.is_file() or not prefix_path.is_file():
        raise FileNotFoundError("the three fixed CAV selections are unavailable")
    evidence_texts = {
        row.evidence_text
        for question in artifact.questions
        for row in question.feature_rows
    }
    question_texts = {question.question for question in artifact.questions}
    all_texts = tuple(
        sorted(evidence_texts | question_texts, key=lambda value: (len(value), value))
    )
    if not all_texts:
        raise ValueError("locked S0 global feature table is empty")
    execution_chunk_count = (
        len(all_texts) + FAST_CAV_MAX_ENCODER_ROWS - 1
    ) // FAST_CAV_MAX_ENCODER_ROWS
    event_sha = file_sha256(event_path)
    prefix_sha = file_sha256(prefix_path)
    expected_checkpoint = expected_prefix_checkpoint_sha256(
        FEATURE_PREFIX_LAYERS
    )
    command_argv = [
        "pixi",
        "run",
        "-e",
        "dev",
        "python",
        "tools/run_locked_s0_cav_links_arm.py",
        "--phase",
        "features",
        "--expected-s0-run-sha256",
        str(args.expected_s0_run_sha256),
        "--enable-feature-model",
        "--authorized-feature-encoder-calls",
        str(execution_chunk_count),
    ]
    plan: dict[str, Any] = {
        "format": FEATURE_BINDING_FORMAT,
        "arm_label": ARM_LABEL,
        "adapter_artifact_sha256": artifact.raw_sha256,
        "adapter_binding_sha256": adapter_binding["adapter_artifact_sha256"],
        "encoder": {
            "model_dir": str(Path(args.model_dir)),
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_MODEL_REVISION,
            "expected_prefix_checkpoint_sha256": expected_checkpoint,
            "prefix_layers": FEATURE_PREFIX_LAYERS,
            "feature_layer": FEATURE_LAYER,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "model_init_count": 1,
            "feature_session_wrapper_call_count": 1,
            "qwen_encode_layers_call_count": execution_chunk_count,
            "execution_chunk_row_cap": FAST_CAV_MAX_ENCODER_ROWS,
            "execution_chunk_count": execution_chunk_count,
            "transformer_forward_batch_count": (
                len(all_texts) + args.batch_size - 1
            )
            // args.batch_size,
            "global_deduplication": (
                "exact-text-before-bounded-encoder-execution-chunks-v1"
            ),
            "row_ceiling_scope": (
                "tool-only-exact-population-session-with-bounded-"
                "encoder-execution-chunks-v1"
            ),
            "rows_truncated": 0,
        },
        "router": {
            "algorithm": "genuine-fixed-cav-two-rectangular-pass-v1",
            "device": "cpu",
            "dtype": "float32",
            "concept_count": CONCEPT_COUNT,
            "selections": [
                {
                    "artifact": str(event_path),
                    "artifact_sha256": event_sha,
                    "tensor_key": _CAV_SELECTION_KEYS[0],
                },
                {
                    "artifact": str(prefix_path),
                    "artifact_sha256": prefix_sha,
                    "tensor_key": _CAV_SELECTION_KEYS[1],
                },
                {
                    "artifact": str(prefix_path),
                    "artifact_sha256": prefix_sha,
                    "tensor_key": _CAV_SELECTION_KEYS[2],
                },
            ],
            "extraction_temperature": args.extraction_temperature,
            "reinjection_temperature": args.reinjection_temperature,
            "alpha": args.alpha,
            "top_extraction_links_per_concept": (
                EXTRACTION_LINKS_PER_CONCEPT
            ),
            "evidence_pair_matrix_constructed": False,
        },
        "input": {
            "question_count": artifact.question_count,
            "global_unique_evidence_text_count": len(evidence_texts),
            "global_unique_question_text_count": len(question_texts),
            "global_unique_text_count": len(all_texts),
            "encoder_input_projection_sha256": identity_sha256(
                {
                    "format": (
                        "memory-condense-fast-cav-encoder-input-projection-v1"
                    ),
                    "ordered_text_sha256s": [
                        quote_sha256(value) for value in all_texts
                    ],
                }
            ),
            "canonical_stage_placements": (
                artifact.question_count * len(STAGE_IDS)
            ),
            "unique_s0_packet_router_calls": artifact.question_count,
        },
        "answer_budget": {
            "guide_token_cap": MAX_GUIDE_TOKENS,
            "prompt_token_cap": MAX_PROMPT_TOKENS,
            "answer_output_token_cap": MAX_ANSWER_OUTPUT_TOKENS,
            "evidence_additions": 0,
        },
        "exact_feature_command_argv": command_argv,
    }
    plan["feature_plan_sha256"] = identity_sha256(plan)
    return plan


@dataclass(frozen=True, slots=True)
class _Source:
    s0_plan: Any
    s0_run: Mapping[str, Any]
    s0_run_sha256: str
    artifact: FastRetrievalArtifact
    adapter_binding: Mapping[str, Any]
    feature_plan: Mapping[str, Any]
    binding: Mapping[str, Any]


def _source_from_verified(
    args: argparse.Namespace,
    plan: Any,
    s0_run: Mapping[str, Any],
    s0_sha: str,
) -> _Source:
    artifact, adapter = _adapter_artifact(plan, s0_run, s0_sha)
    feature = _feature_plan(args, artifact, adapter)
    binding: dict[str, Any] = {
        "format": "memory-condense-locked-s0-cav-links-binding-v1",
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "retrieval_sha256": plan.population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            plan.population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": plan.population.population_identity_sha256,
        "historical_validator_binding_sha256": plan.population.binding_sha256,
        "s0_control_run_sha256": s0_sha,
        "adapter_artifact_sha256": artifact.raw_sha256,
        "adapter_binding_sha256": adapter["adapter_artifact_sha256"],
        "feature_plan_sha256": feature["feature_plan_sha256"],
        "question_binding_sha256s": [
            row.binding_sha256 for row in plan.rows
        ],
    }
    binding["binding_sha256"] = identity_sha256(binding)
    return _Source(
        s0_plan=plan,
        s0_run=s0_run,
        s0_run_sha256=s0_sha,
        artifact=artifact,
        adapter_binding=adapter,
        feature_plan=feature,
        binding=binding,
    )


def _load_source(args: argparse.Namespace) -> _Source:
    _validate_common(args)
    plan, s0_run, s0_sha = _verified_s0(args)
    return _source_from_verified(args, plan, s0_run, s0_sha)


def _features_path(args: argparse.Namespace) -> Path:
    return Path(args.features or Path(args.output_root) / "features.json")


def _run_path(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "run.json"


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.enable_provider
        or args.authorized_provider_calls
        or args.enable_feature_model
        or args.authorized_feature_encoder_calls
    ):
        raise ValueError("preflight forbids model/provider access and authorization")
    source = _load_source(args)
    s0_counts = [
        len(question.stages[0].evidence)
        for question in source.artifact.questions
    ]
    result = {
        "format": PREFLIGHT_FORMAT,
        "arm_label": ARM_LABEL,
        "source_binding": dict(source.binding),
        "adapter_binding": dict(source.adapter_binding),
        "feature_plan": dict(source.feature_plan),
        "question_count": source.artifact.question_count,
        "s0_evidence_rows": _distribution(s0_counts),
        "feature_encoder_calls_required": source.feature_plan["encoder"][
            "qwen_encode_layers_call_count"
        ],
        "feature_session_wrapper_calls_required": 1,
        "feature_model_loads_required": 1,
        "dependent_answer_preflight_requires_features": True,
        "concept_count": CONCEPT_COUNT,
        "top_extraction_links_per_concept": EXTRACTION_LINKS_PER_CONCEPT,
        "guide_token_cap": MAX_GUIDE_TOKENS,
        "evidence_additions": 0,
        "x_x1_ordering_proxy_consumed": False,
        "feature_model_loads": 0,
        "feature_encoder_calls": 0,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
    }
    if _contains_forbidden_key(result):
        raise RuntimeError("CAV preflight crossed the gold firewall")
    return result


def _load_feature_runtime(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load the exact three-CAV router and one-layer Qwen encoder once."""

    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
    from memory_condense.search.fusion.fixed_cav_router import FixedCAVRouter

    selections = (
        (Path(args.event_cav), _CAV_SELECTION_KEYS[0]),
        (Path(args.prefix_cav), _CAV_SELECTION_KEYS[1]),
        (Path(args.prefix_cav), _CAV_SELECTION_KEYS[2]),
    )
    router = FixedCAVRouter.load(
        selections,
        layer=FEATURE_LAYER,
        device="cpu",
        dtype="float32",
        extraction_temperature=args.extraction_temperature,
        reinjection_temperature=args.reinjection_temperature,
        alpha=args.alpha,
    )
    encoder = Qwen3PrefixEncoder(
        Path(args.model_dir),
        layers=FEATURE_PREFIX_LAYERS,
        device=args.device,
        dtype=args.dtype,
        expected_checkpoint_sha256=expected_prefix_checkpoint_sha256(
            FEATURE_PREFIX_LAYERS
        ),
    )
    return encoder, router


class _ChunkedFeatureEncoder:
    """One resident Qwen, exact global rows, bounded execution calls."""

    def __init__(self, encoder: Any, feature_plan: Mapping[str, Any]) -> None:
        self._encoder = encoder
        self._plan = feature_plan
        self.checkpoint_sha256 = encoder.checkpoint_sha256
        self.feature_backend_identity_sha256 = getattr(
            encoder,
            "feature_backend_identity_sha256",
            None,
        )
        self.dtype_name = encoder.dtype_name
        self.device = encoder.device
        self.layers = encoder.layers
        self.model_id = getattr(encoder, "model_id", None)
        self.model_revision = getattr(encoder, "model_revision", None)
        self.execution_receipt: Mapping[str, Any] | None = None

    def _identity(self) -> tuple[object, ...]:
        return (
            self._encoder.checkpoint_sha256,
            getattr(self._encoder, "feature_backend_identity_sha256", None),
            self._encoder.dtype_name,
            str(self._encoder.device),
            self._encoder.layers,
            getattr(self._encoder, "model_id", None),
            getattr(self._encoder, "model_revision", None),
        )

    def encode_layers(
        self,
        texts: Sequence[str],
        *,
        layers: Sequence[int],
        batch_size: int,
    ) -> dict[int, Any]:
        if self.execution_receipt is not None:
            raise RuntimeError("feature-session wrapper may be called only once")
        ordered = tuple(texts)
        selected_layers = tuple(layers)
        encoder_plan = self._plan["encoder"]
        input_plan = self._plan["input"]
        observed_projection = identity_sha256(
            {
                "format": (
                    "memory-condense-fast-cav-encoder-input-projection-v1"
                ),
                "ordered_text_sha256s": [
                    quote_sha256(value) for value in ordered
                ],
            }
        )
        if (
            len(ordered) != input_plan["global_unique_text_count"]
            or observed_projection
            != input_plan["encoder_input_projection_sha256"]
            or selected_layers != (FEATURE_LAYER,)
            or batch_size != encoder_plan["batch_size"]
        ):
            raise ValueError("chunked encoder received a changed global row table")
        chunk_cap = int(encoder_plan["execution_chunk_row_cap"])
        expected_calls = int(encoder_plan["qwen_encode_layers_call_count"])
        identity_before = self._identity()
        chunks: list[Any] = []
        receipts: list[dict[str, Any]] = []
        torch: Any = None
        try:
            import torch as torch_module

            torch = torch_module
            for chunk_ordinal, start in enumerate(
                range(0, len(ordered), chunk_cap)
            ):
                stop = min(start + chunk_cap, len(ordered))
                chunk_texts = ordered[start:stop]
                encoded = self._encoder.encode_layers(
                    chunk_texts,
                    layers=selected_layers,
                    batch_size=batch_size,
                )
                if (
                    type(encoded) is not dict
                    or tuple(encoded) != selected_layers
                ):
                    raise TypeError("Qwen chunk returned a changed layer mapping")
                tensor = encoded[FEATURE_LAYER]
                if (
                    type(tensor) is not torch.Tensor
                    or tensor.ndim != 2
                    or int(tensor.shape[0]) != len(chunk_texts)
                    or bool(tensor.requires_grad)
                    or tensor.grad_fn is not None
                    or not bool(torch.isfinite(tensor).all().item())
                ):
                    raise ValueError("Qwen chunk returned an invalid feature tensor")
                canonical = canonical_float32_tensor(
                    tensor,
                    label=f"Qwen feature chunk {chunk_ordinal}",
                    retain_values=False,
                )
                chunk_receipt: dict[str, Any] = {
                    "chunk_ordinal": chunk_ordinal,
                    "row_start_inclusive": start,
                    "row_stop_exclusive": stop,
                    "row_count": len(chunk_texts),
                    "ordered_text_sha256s_projection_sha256": identity_sha256(
                        [quote_sha256(value) for value in chunk_texts]
                    ),
                    "output_shape": list(canonical.shape),
                    "output_dtype": canonical.dtype,
                    "output_tensor_sha256": canonical.tensor_sha256,
                    "transformer_forward_batch_count": (
                        len(chunk_texts) + batch_size - 1
                    )
                    // batch_size,
                }
                chunk_receipt["chunk_receipt_sha256"] = identity_sha256(
                    chunk_receipt
                )
                receipts.append(chunk_receipt)
                chunks.append(tensor)
            if len(chunks) != expected_calls:
                raise RuntimeError("Qwen execution chunk population changed")
            combined = torch.cat(chunks, dim=0).contiguous()
            combined_canonical = canonical_float32_tensor(
                combined,
                label="global Qwen feature table",
                retain_values=False,
            )
            receipt: dict[str, Any] = {
                "format": ENCODER_EXECUTION_FORMAT,
                "encoder_input_projection_sha256": observed_projection,
                "global_row_count": len(ordered),
                "rows_truncated": 0,
                "qwen_model_load_count": 1,
                "feature_session_wrapper_call_count": 1,
                "qwen_encode_layers_call_count": len(receipts),
                "execution_chunk_row_cap": chunk_cap,
                "transformer_forward_batch_count": sum(
                    int(row["transformer_forward_batch_count"])
                    for row in receipts
                ),
                "chunks": receipts,
                "global_output_shape": list(combined_canonical.shape),
                "global_output_dtype": combined_canonical.dtype,
                "global_output_tensor_sha256": (
                    combined_canonical.tensor_sha256
                ),
            }
            receipt["execution_receipt_sha256"] = identity_sha256(receipt)
            self.execution_receipt = receipt
            if self._identity() != identity_before:
                raise RuntimeError("resident Qwen identity changed across chunks")
            return {FEATURE_LAYER: combined}
        finally:
            chunks.clear()


def _validate_encoder_execution_receipt(
    source: _Source,
    value: object,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("feature artifact has no encoder execution receipt")
    encoder = source.feature_plan["encoder"]
    feature_input = source.feature_plan["input"]
    chunks = value.get("chunks")
    if not isinstance(chunks, list):
        raise TypeError("encoder execution chunks must be an exact list")
    expected_count = encoder["qwen_encode_layers_call_count"]
    if (
        value.get("format") != ENCODER_EXECUTION_FORMAT
        or value.get("encoder_input_projection_sha256")
        != feature_input["encoder_input_projection_sha256"]
        or value.get("global_row_count")
        != feature_input["global_unique_text_count"]
        or value.get("rows_truncated") != 0
        or value.get("qwen_model_load_count") != 1
        or value.get("feature_session_wrapper_call_count") != 1
        or value.get("qwen_encode_layers_call_count") != expected_count
        or len(chunks) != expected_count
        or value.get("execution_chunk_row_cap")
        != encoder["execution_chunk_row_cap"]
        or value.get("transformer_forward_batch_count")
        != encoder["transformer_forward_batch_count"]
    ):
        raise ValueError("encoder execution receipt changed its locked plan")
    all_texts = tuple(
        sorted(
            {
                row.evidence_text
                for question in source.artifact.questions
                for row in question.feature_rows
            }
            | {question.question for question in source.artifact.questions},
            key=lambda item: (len(item), item),
        )
    )
    expected_start = 0
    hidden_dim: int | None = None
    for ordinal, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise TypeError("encoder execution chunk must be an object")
        start = chunk.get("row_start_inclusive")
        stop = chunk.get("row_stop_exclusive")
        if (
            chunk.get("chunk_ordinal") != ordinal
            or type(start) is not int
            or start != expected_start
            or type(stop) is not int
            or stop <= start
            or stop > len(all_texts)
            or chunk.get("row_count") != stop - start
            or stop - start > encoder["execution_chunk_row_cap"]
            or chunk.get("ordered_text_sha256s_projection_sha256")
            != identity_sha256(
                [quote_sha256(value) for value in all_texts[start:stop]]
            )
            or chunk.get("output_dtype") != "float32-le"
            or chunk.get("transformer_forward_batch_count")
            != (stop - start + encoder["batch_size"] - 1)
            // encoder["batch_size"]
        ):
            raise ValueError("encoder execution chunk changed its row projection")
        shape = chunk.get("output_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != stop - start
            or type(shape[1]) is not int
            or shape[1] < 1
        ):
            raise ValueError("encoder execution chunk shape changed")
        hidden_dim = shape[1] if hidden_dim is None else hidden_dim
        if shape[1] != hidden_dim:
            raise ValueError("encoder execution chunks changed hidden width")
        _require_sha256(chunk.get("output_tensor_sha256"), "chunk output")
        supplied_chunk_sha = _require_sha256(
            chunk.get("chunk_receipt_sha256"),
            "chunk receipt",
        )
        if identity_sha256(
            {
                key: child
                for key, child in chunk.items()
                if key != "chunk_receipt_sha256"
            }
        ) != supplied_chunk_sha:
            raise ValueError("encoder execution chunk receipt seal changed")
        expected_start = stop
    if expected_start != len(all_texts) or hidden_dim is None:
        raise ValueError("encoder execution chunks do not cover all global rows")
    if (
        value.get("global_output_shape") != [len(all_texts), hidden_dim]
        or value.get("global_output_dtype") != "float32-le"
    ):
        raise ValueError("global encoder output shape changed")
    _require_sha256(value.get("global_output_tensor_sha256"), "global output")
    supplied_receipt_sha = _require_sha256(
        value.get("execution_receipt_sha256"),
        "encoder execution receipt",
    )
    if identity_sha256(
        {
            key: child
            for key, child in value.items()
            if key != "execution_receipt_sha256"
        }
    ) != supplied_receipt_sha:
        raise ValueError("encoder execution receipt seal changed")


def _validate_feature_session(
    source: _Source,
    session: FastCAVFeatureSessionReceipt,
    router_receipt: Any,
) -> None:
    plan = source.feature_plan
    feature_input = plan["input"]
    selections = plan["router"]["selections"]
    expected_file_hashes = tuple(row["artifact_sha256"] for row in selections)
    expected_keys = tuple(row["tensor_key"] for row in selections)
    if (
        session.artifact_sha256 != source.artifact.raw_sha256
        or session.question_count != source.artifact.question_count
        or session.encoder_api_call_count != 1
        or session.feature_checkpoint_sha256
        != plan["encoder"]["expected_prefix_checkpoint_sha256"]
        or session.router_num_cavs != CONCEPT_COUNT
        or session.feature_layer != FEATURE_LAYER
        or session.feature_encoder_prefix_layers != FEATURE_PREFIX_LAYERS
        or session.feature_encoder_runtime_dtype
        != plan["encoder"]["dtype"]
        or session.feature_encoder_runtime_device
        != plan["encoder"]["device"]
        or session.batch_size != plan["encoder"]["batch_size"]
        or session.global_unique_evidence_text_count
        != feature_input["global_unique_evidence_text_count"]
        or session.global_unique_question_text_count
        != feature_input["global_unique_question_text_count"]
        or session.global_unique_text_count
        != feature_input["global_unique_text_count"]
        or session.encoder_input_projection_sha256
        != feature_input["encoder_input_projection_sha256"]
        or session.unique_router_call_count != source.artifact.question_count
        or session.result_retained_tensor_bytes
        or session.retained_token_id_count
        or session.persisted_token_state_bytes
    ):
        raise ValueError("CAV feature session changed the locked runtime/input")
    if (
        router_receipt.num_cavs != CONCEPT_COUNT
        or router_receipt.layer != FEATURE_LAYER
        or router_receipt.artifact_file_sha256s != expected_file_hashes
        or router_receipt.ordered_tensor_keys != expected_keys
        or router_receipt.runtime_sha256
        != session.router_runtime_identity_sha256
        or router_receipt.bank_identity_sha256
        != session.router_bank_identity_sha256
        or router_receipt.device != plan["router"]["device"]
        or router_receipt.execution_dtype != "torch.float32"
        or router_receipt.extraction_temperature
        != plan["router"]["extraction_temperature"]
        or router_receipt.reinjection_temperature
        != plan["router"]["reinjection_temperature"]
        or router_receipt.alpha != plan["router"]["alpha"]
    ):
        raise ValueError("CAV router receipt changed its three fixed concepts")
    for question in source.artifact.questions:
        s0_ids = question.stages[0].evidence_ids
        for stage in question.stages:
            if stage.evidence_ids != s0_ids or (
                stage.stage_id != STAGE_IDS[0] and stage.added_evidence_ids
            ):
                raise ValueError("CAV adapter changed exact S0 membership/order")
        feature_stage = session.stage(question.question_id, STAGE_IDS[-1])
        links = feature_stage.links
        if (
            type(links) is not FastCAVLinkReceipt
            or len(links.concepts) != CONCEPT_COUNT
            or links.evidence_ids != s0_ids
            or links.max_evidence_links_per_concept
            != EXTRACTION_LINKS_PER_CONCEPT
            or links.evidence_pair_matrix_constructed is not False
            or links.retained_token_id_count
            or links.retained_tensor_bytes
            or links.persisted_token_state_bytes
        ):
            raise ValueError("CAV link receipt violated the protected link budget")
        counts = Counter(row.concept_ordinal for row in links.extraction_links)
        if set(counts) != set(range(CONCEPT_COUNT)) or any(
            count > EXTRACTION_LINKS_PER_CONCEPT for count in counts.values()
        ):
            raise ValueError("CAV extraction links changed their top-four cap")


def _feature_manifest(
    source: _Source,
    session: FastCAVFeatureSessionReceipt,
    router_receipt: Any,
    encoder_execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_feature_session(source, session, router_receipt)
    _validate_encoder_execution_receipt(source, encoder_execution_receipt)
    orders = _orders_from_session(session)
    manifest = {
        "format": FEATURE_MANIFEST_FORMAT,
        "arm_label": ARM_LABEL,
        "source_binding": dict(source.binding),
        "feature_plan": dict(source.feature_plan),
        "retrieval_sha256": source.artifact.raw_sha256,
        "transcript_tokens": source.artifact.transcript_tokens,
        "turn_count": source.artifact.turn_count,
        "question_count": source.artifact.question_count,
        "feature_session": asdict(session),
        "encoder_execution_receipt": dict(encoder_execution_receipt),
        "router_runtime_receipt": asdict(router_receipt),
        "stage_orders": _orders_payload(orders),
        "mechanism": {
            "source": "genuine_cav_v2_two_pass_links",
            "concept_count": CONCEPT_COUNT,
            "top_extraction_links_per_concept": (
                EXTRACTION_LINKS_PER_CONCEPT
            ),
            "evidence_additions": 0,
            "x_x1_ordering_proxy_consumed": False,
        },
        "zero_state": {
            "contract": ZERO_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
        },
        "feature_model_loads": 1,
        "feature_session_wrapper_calls": 1,
        "feature_encoder_calls": source.feature_plan["encoder"][
            "qwen_encode_layers_call_count"
        ],
        "provider_calls": 0,
        "gold_loaded": False,
    }
    if _contains_forbidden_key(manifest):
        raise RuntimeError("CAV feature artifact crossed the gold firewall")
    return manifest


def _guard_existing(path: Path, source: _Source, format_name: str) -> None:
    if not path.exists():
        return
    existing, _digest_value = _read(path)
    if (
        existing.get("format") != format_name
        or existing.get("arm_label") != ARM_LABEL
        or existing.get("source_binding") != dict(source.binding)
    ):
        raise FileExistsError(f"output belongs to another experiment: {path}")


def run_features(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider or args.authorized_provider_calls:
        raise ValueError("feature phase forbids provider access")
    source = _load_source(args)
    if not args.enable_feature_model:
        raise ValueError("feature phase requires --enable-feature-model")
    exact_encoder_calls = source.feature_plan["encoder"][
        "qwen_encode_layers_call_count"
    ]
    if args.authorized_feature_encoder_calls != exact_encoder_calls:
        raise ValueError(
            "--authorized-feature-encoder-calls must exactly equal the "
            "bounded Qwen execution population "
            f"({args.authorized_feature_encoder_calls} != "
            f"{exact_encoder_calls})"
        )
    path = _features_path(args)
    _guard_existing(path, source, FEATURE_MANIFEST_FORMAT)
    encoder: Any = None
    wrapped_encoder: _ChunkedFeatureEncoder | None = None
    router: Any = None
    try:
        encoder, router = _load_feature_runtime(args)
        wrapped_encoder = _ChunkedFeatureEncoder(
            encoder,
            source.feature_plan,
        )
        session_globals = run_fast_cav_feature_session.__globals__
        prior_row_ceiling = session_globals.get("FAST_CAV_MAX_ENCODER_ROWS")
        if prior_row_ceiling != FAST_CAV_MAX_ENCODER_ROWS:
            raise RuntimeError("feature-session row ceiling changed")
        session_globals["FAST_CAV_MAX_ENCODER_ROWS"] = source.feature_plan[
            "input"
        ]["global_unique_text_count"]
        try:
            session = run_fast_cav_feature_session(
                source.artifact,
                encoder=wrapped_encoder,
                router=router,
                layer=FEATURE_LAYER,
                batch_size=args.batch_size,
            )
        finally:
            session_globals["FAST_CAV_MAX_ENCODER_ROWS"] = prior_row_ceiling
        execution_receipt = wrapped_encoder.execution_receipt
        if not isinstance(execution_receipt, Mapping):
            raise RuntimeError("chunked Qwen execution emitted no receipt")
        router_receipt = router.runtime_receipt
        manifest = _feature_manifest(
            source,
            session,
            router_receipt,
            execution_receipt,
        )
    finally:
        encoder = None
        wrapped_encoder = None
        router = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - optional feature runtime
            pass
    digest = _publish(path, manifest)
    sealed, sealed_sha = _read(path, expected_sha256=digest)
    return sealed, sealed_sha


@dataclass(frozen=True, slots=True)
class _Features:
    source: _Source
    manifest: Mapping[str, Any]
    manifest_sha256: str
    session: FastCAVFeatureSessionReceipt
    router_receipt: Any


def _load_verified_features(args: argparse.Namespace) -> _Features:
    source = _load_source(args)
    expected_sha = _require_sha256(
        args.expected_features_sha256,
        "feature artifact expected SHA-256",
    )
    path = _features_path(args)
    manifest, digest = _read(path, expected_sha256=expected_sha)
    if (
        manifest.get("format") != FEATURE_MANIFEST_FORMAT
        or manifest.get("arm_label") != ARM_LABEL
        or manifest.get("source_binding") != dict(source.binding)
        or manifest.get("feature_plan") != dict(source.feature_plan)
        or manifest.get("mechanism", {}).get("x_x1_ordering_proxy_consumed")
        is not False
    ):
        raise ValueError("feature artifact changed its locked arm binding")
    _validate_encoder_execution_receipt(
        source,
        manifest.get("encoder_execution_receipt"),
    )
    typed = load_fast_cav_feature_artifact(
        path,
        retrieval_artifact=source.artifact,
        expected_sha256=digest,
        verify_sidecar=True,
        require_links=True,
    )
    _validate_feature_session(
        source,
        typed.feature_session,
        typed.router_runtime_receipt,
    )
    return _Features(
        source=source,
        manifest=manifest,
        manifest_sha256=digest,
        session=typed.feature_session,
        router_receipt=typed.router_runtime_receipt,
    )


def run_feature_replay(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    if (
        args.enable_provider
        or args.authorized_provider_calls
        or args.enable_feature_model
        or args.authorized_feature_encoder_calls
    ):
        raise ValueError("feature replay forbids model/provider access")
    features = _load_verified_features(args)
    return dict(features.manifest), features.manifest_sha256


@dataclass(frozen=True, slots=True)
class _Candidate:
    prompt: FastCAVLinkSynthesisArmPrompt
    messages: tuple[dict[str, str], ...]
    receipt: FastCAVLinkSynthesisStageReceipt
    links: FastCAVLinkReceipt
    guide_token_proxy: int
    candidate_relation_targets: tuple[Mapping[str, Any], ...]


def _relation_targets(
    links: FastCAVLinkReceipt,
    *,
    disposition: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for link in links.extraction_links:
        rows.append(
            {
                "target_id": link.link_sha256,
                "target_id_encoding": "sha256",
                "target_kind": "cav_extraction_relation",
                "discovering_method": "genuine_cav_v2_two_pass",
                "disposition": disposition,
                "route_local_receipt_sha256": link.link_sha256,
                "concept_target_id": link.concept_id,
                "evidence_target_id": link.evidence_id,
                "source_target_id": link.source_id,
                "rank": link.rank,
            }
        )
    for link in links.reinjection_links:
        if link.rank != 1:
            continue
        rows.append(
            {
                "target_id": link.link_sha256,
                "target_id_encoding": "sha256",
                "target_kind": "cav_reinjection_relation",
                "discovering_method": "genuine_cav_v2_two_pass",
                "disposition": disposition,
                "route_local_receipt_sha256": link.link_sha256,
                "concept_target_id": link.concept_id,
                "evidence_target_id": link.evidence_id,
                "source_target_id": link.source_id,
                "rank": link.rank,
            }
        )
    return tuple(rows)


def _s0_evidence_targets(question: FastRetrievalQuestion) -> list[dict[str, Any]]:
    return [
        {
            "target_id": evidence.evidence_id,
            "target_id_encoding": "raw_sealed_evidence_id",
            "target_kind": "evidence",
            "discovering_method": STAGE_IDS[0],
            "source_target_id": evidence.source_id,
            "disposition": "protected_s0_unchanged",
            "route_local_receipt_sha256": (
                question.stages[0].stage_receipt_sha256
            ),
        }
        for evidence in question.stages[0].evidence
    ]


@dataclass(frozen=True, slots=True)
class _AnswerPlan:
    features: _Features
    population: Any
    candidates: tuple[_Candidate | None, ...]
    statuses: tuple[str, ...]
    prompts: tuple[tuple[dict[str, str], ...], ...]
    valid_ordinals: tuple[int, ...]
    preflight: FastPromptPopulation | None

    @property
    def unique_calls(self) -> int:
        return 0 if self.preflight is None else self.preflight.unique_prompt_count


def _build_answer_plan(args: argparse.Namespace) -> _AnswerPlan:
    features = _load_verified_features(args)
    population = build_fast_cav_link_synthesis_population(
        features.source.artifact,
        features.session,
    )
    logical_messages = population.logical_message_population
    candidates: list[_Candidate | None] = []
    statuses: list[str] = []
    for ordinal, (question, receipt) in enumerate(
        zip(
            features.source.artifact.questions,
            population.stage_receipts,
            strict=True,
        )
    ):
        prompt_index = ordinal * 2 + 1
        prompt = population.prompts[prompt_index]
        messages = logical_messages[prompt_index]
        feature_stage = features.session.stage(question.question_id, STAGE_IDS[-1])
        links = feature_stage.links
        status = "valid"
        candidate: _Candidate | None = None
        if type(links) is not FastCAVLinkReceipt:
            status = "invalid_link_receipt"
        elif (
            prompt.arm_id != "linked"
            or prompt.link_exposed is not True
            or prompt.evidence_ids != question.stages[0].evidence_ids
            or question.stages[-1].evidence_ids
            != question.stages[0].evidence_ids
            or question.stages[-1].added_evidence_ids
            or len(receipt.link_guide_groups) != CONCEPT_COUNT
            or any(
                len(group.extraction_evidence_aliases)
                > EXTRACTION_LINKS_PER_CONCEPT
                for group in receipt.link_guide_groups
            )
        ):
            status = "invalid_link_projection"
        else:
            guide = _render_linked_guide(receipt.link_guide_groups)
            guide_tokens = count_tokens(guide)
            if guide_tokens > MAX_GUIDE_TOKENS:
                status = "guide_overflow"
            elif (
                prompt.prompt_token_proxy > MAX_PROMPT_TOKENS
                or prompt.max_completion_tokens != MAX_ANSWER_OUTPUT_TOKENS
                or identity_sha256(list(messages)) != prompt.messages_sha256
            ):
                status = "prompt_overflow_or_binding_failure"
            else:
                candidate = _Candidate(
                    prompt=prompt,
                    messages=messages,
                    receipt=receipt,
                    links=links,
                    guide_token_proxy=guide_tokens,
                    candidate_relation_targets=_relation_targets(
                        links,
                        disposition="candidate_before_budget",
                    ),
                )
        candidates.append(candidate)
        statuses.append(status)
    valid_ordinals = tuple(
        ordinal for ordinal, row in enumerate(candidates) if row is not None
    )
    prompts = tuple(
        row.messages for row in candidates if row is not None
    )
    preflight = None
    if prompts:
        preflight = preflight_fast_completion_prompts(
            prompts,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
        )
        if (
            preflight.logical_prompt_count != len(valid_ordinals)
            or preflight.unique_prompt_count != len(valid_ordinals)
        ):
            raise ValueError("CAV answer prompts are not unique per valid question")
    return _AnswerPlan(
        features=features,
        population=population,
        candidates=tuple(candidates),
        statuses=tuple(statuses),
        prompts=prompts,
        valid_ordinals=valid_ordinals,
        preflight=preflight,
    )


def run_answer_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.enable_provider
        or args.authorized_provider_calls
        or args.enable_feature_model
        or args.authorized_feature_encoder_calls
    ):
        raise ValueError("answer preflight forbids model/provider access")
    plan = _build_answer_plan(args)
    guide_tokens = [
        row.guide_token_proxy for row in plan.candidates if row is not None
    ]
    result = {
        "format": ANSWER_PREFLIGHT_FORMAT,
        "arm_label": ARM_LABEL,
        "source_binding": dict(plan.features.source.binding),
        "feature_artifact_sha256": plan.features.manifest_sha256,
        "feature_session_receipt_sha256": (
            plan.features.session.session_receipt_sha256
        ),
        "link_population_sha256": plan.population.population_sha256,
        "question_count": plan.features.source.artifact.question_count,
        "valid_link_guide_count": len(plan.valid_ordinals),
        "s0_fallback_count": (
            plan.features.source.artifact.question_count
            - len(plan.valid_ordinals)
        ),
        "status_counts": dict(sorted(Counter(plan.statuses).items())),
        "guide_tokens": _distribution(guide_tokens),
        "guide_token_cap": MAX_GUIDE_TOKENS,
        "answer_prompt_population": (
            None if plan.preflight is None else plan.preflight.model_dump()
        ),
        "required_authorized_provider_calls": plan.unique_calls,
        "authorized_call_kind": "terra_s0_plus_genuine_cav_links_answer",
        "concept_count": CONCEPT_COUNT,
        "top_extraction_links_per_concept": EXTRACTION_LINKS_PER_CONCEPT,
        "evidence_additions": 0,
        "x_x1_ordering_proxy_consumed": False,
        "provider_calls": 0,
        "writes": 0,
        "gold_loaded": False,
    }
    if _contains_forbidden_key(result):
        raise RuntimeError("CAV answer preflight crossed the gold firewall")
    return result


def _runtime(
    plan: _AnswerPlan,
    args: argparse.Namespace,
    *,
    client: Any | None,
) -> FastCompletionRuntime | None:
    if not plan.prompts:
        return None
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / "terra-answer-calls",
        prompt_population=plan.prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=MAX_ANSWER_OUTPUT_TOKENS,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance={
            "experiment_format": RUN_FORMAT,
            "arm_label": ARM_LABEL,
            "source_binding_sha256": plan.features.source.binding[
                "binding_sha256"
            ],
            "s0_control_run_sha256": plan.features.source.s0_run_sha256,
            "feature_artifact_sha256": plan.features.manifest_sha256,
            "feature_session_receipt_sha256": (
                plan.features.session.session_receipt_sha256
            ),
            "link_population_sha256": plan.population.population_sha256,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
        },
    )


def _structural_target_ledger(
    plan: _AnswerPlan,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal, (question, candidate) in enumerate(
        zip(
            plan.features.source.artifact.questions,
            plan.candidates,
            strict=True,
        )
    ):
        feature_stage = plan.features.session.stage(
            question.question_id,
            STAGE_IDS[-1],
        )
        links = feature_stage.links
        candidate_relations = (
            ()
            if type(links) is not FastCAVLinkReceipt
            else _relation_targets(
                links,
                disposition="candidate_before_budget",
            )
        )
        admitted_relations = (
            ()
            if candidate is None
            else _relation_targets(
                links,
                disposition="admitted_after_budget",
            )
        )
        evidence_targets = _s0_evidence_targets(question)
        body: dict[str, Any] = {
            "ordinal": ordinal,
            "question_id": question.question_id,
            "evidence_targets": evidence_targets,
            "candidate_relation_targets_before_budget": list(
                candidate_relations
            ),
            "admitted_relation_targets_after_budget": list(
                admitted_relations
            ),
            "candidate_relation_target_ids_sha256": identity_sha256(
                [row["target_id"] for row in candidate_relations]
            ),
            "admitted_relation_target_ids_sha256": identity_sha256(
                [row["target_id"] for row in admitted_relations]
            ),
            "candidate_relation_target_count": len(candidate_relations),
            "admitted_relation_target_count": len(admitted_relations),
            "primary_evidence_owner_unchanged": True,
        }
        body["ledger_row_sha256"] = identity_sha256(body)
        rows.append(body)
    result = {
        "format": "memory-condense-structural-target-ledger-v1",
        "arm_label": ARM_LABEL,
        "source_s0_run_sha256": plan.features.source.s0_run_sha256,
        "source_feature_artifact_sha256": plan.features.manifest_sha256,
        "source_binding_sha256": plan.features.source.binding[
            "binding_sha256"
        ],
        "population_identity_sha256": (
            plan.features.source.s0_plan.population.population_identity_sha256
        ),
        "question_count": len(rows),
        "target_id_policy": {
            "evidence_targets": "raw_sealed_evidence_id",
            "relation_targets": "sealed_link_sha256",
        },
        "ownership_policy": (
            "join-primary-owner-from-posthoc-desired-target-registry"
        ),
        "discovery_projection": "candidate_relation_targets_before_budget",
        "admission_projection": "admitted_relation_targets_after_budget",
        "questions": rows,
    }
    result["ledger_sha256"] = identity_sha256(result)
    return result


def _run_artifact(
    plan: _AnswerPlan,
    batch: FastCompletionBatch | None,
) -> dict[str, Any]:
    completions: dict[int, str] = {}
    records: dict[str, Mapping[str, Any]] = {}
    if batch is not None:
        completions = dict(
            zip(plan.valid_ordinals, batch.logical_completions, strict=True)
        )
        records = _record_by_messages(batch)
    target_ledger = _structural_target_ledger(plan)
    ledger_rows = target_ledger["questions"]
    s0_rows = plan.features.source.s0_run["questions"]
    questions: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for ordinal, (question, s0, candidate, status, ledger_row) in enumerate(
        zip(
            plan.features.source.artifact.questions,
            s0_rows,
            plan.candidates,
            plan.statuses,
            ledger_rows,
            strict=True,
        )
    ):
        s0_prediction = s0["prediction"]
        prediction = str(s0_prediction["text"])
        prediction_kind = "sealed_s0_control_fallback"
        fallback_reason: str | None = status if candidate is None else None
        parsed_response_sha: str | None = None
        prompt_sha: str | None = None
        prompt_tokens: int | None = None
        call_key: str | None = None
        request_journal: str | None = None
        response_journal: str | None = None
        guide_tokens: int | None = None
        if candidate is not None:
            prompt_sha = candidate.prompt.messages_sha256
            prompt_tokens = candidate.prompt.prompt_token_proxy
            guide_tokens = candidate.guide_token_proxy
            record = records[prompt_sha]
            call_key = record["call_key_sha256"]
            request_journal = record["request_journal_sha256"]
            response_journal = record["response_journal_sha256"]
            try:
                parsed = parse_fast_cav_link_synthesis(
                    completions[ordinal],
                    stage=question.stages[-1],
                    receipt=candidate.receipt,
                )
            except (TypeError, ValueError):
                fallback_reason = "invalid_or_ungrounded_answer"
            else:
                prediction = parsed.answer
                prediction_kind = "terra_s0_plus_genuine_cav_links"
                fallback_reason = None
                parsed_response_sha = parsed.response_sha256
        feature_stage = plan.features.session.stage(
            question.question_id,
            STAGE_IDS[-1],
        )
        links = feature_stage.links
        questions.append(
            {
                "ordinal": ordinal,
                "question_id": question.question_id,
                "question_sha256": question.question_sha256,
                "dated_question_sha256": question.dated_question_sha256,
                "retrieval_question_part_sha256": s0[
                    "retrieval_question_part_sha256"
                ],
                "arm_label": ARM_LABEL,
                "parent_arm_label": PARENT_ARM_LABEL,
                "source_stage_id": STAGE_IDS[0],
                "s0_source_binding_sha256": s0["source_binding_sha256"],
                "s0_stage_receipt_sha256": s0["stage_receipt_sha256"],
                "s0_evidence_projection_sha256": s0[
                    "evidence_projection_sha256"
                ],
                "s0_provider_messages_sha256": s0[
                    "provider_messages_sha256"
                ],
                "s0_control_prediction_sha256": s0_prediction["sha256"],
                "ordered_s0_evidence_ids_sha256": identity_sha256(
                    list(question.stages[0].evidence_ids)
                ),
                "s0_evidence_row_count": len(question.stages[0].evidence),
                "evidence_additions": 0,
                "feature_stage_output_sha256": (
                    feature_stage.stage_output_sha256
                ),
                "packet_identity_sha256": feature_stage.packet_identity_sha256,
                "router_runtime_identity_sha256": (
                    feature_stage.router_runtime_identity_sha256
                ),
                "router_bank_identity_sha256": (
                    feature_stage.router_bank_identity_sha256
                ),
                "source_link_receipt_sha256": (
                    None
                    if type(links) is not FastCAVLinkReceipt
                    else links.link_receipt_sha256
                ),
                "link_guide_projection_sha256": (
                    None
                    if candidate is None
                    else candidate.receipt.link_guide_projection_sha256
                ),
                "link_status": status,
                "concept_count": CONCEPT_COUNT,
                "top_extraction_links_per_concept": (
                    EXTRACTION_LINKS_PER_CONCEPT
                ),
                "guide_token_proxy": guide_tokens,
                "guide_token_cap": MAX_GUIDE_TOKENS,
                "x_x1_ordering_proxy_consumed": False,
                "prediction_kind": prediction_kind,
                "s0_fallback_reason": fallback_reason,
                "prediction": {
                    "text": prediction,
                    "sha256": quote_sha256(prediction),
                },
                "changed_from_s0": (
                    quote_sha256(prediction) != s0_prediction["sha256"]
                ),
                "parsed_response_sha256": parsed_response_sha,
                "answer_prompt_messages_sha256": prompt_sha,
                "answer_call_key_sha256": call_key,
                "answer_request_journal_sha256": request_journal,
                "answer_response_journal_sha256": response_journal,
                "structural_target_ledger_row_sha256": ledger_row[
                    "ledger_row_sha256"
                ],
            }
        )
        budget_rows.append(
            {
                "ordinal": ordinal,
                "s0_evidence_rows": len(question.stages[0].evidence),
                "evidence_additions": 0,
                "candidate_relation_target_count": ledger_row[
                    "candidate_relation_target_count"
                ],
                "admitted_relation_target_count": ledger_row[
                    "admitted_relation_target_count"
                ],
                "guide_token_proxy": guide_tokens,
                "guide_token_cap": MAX_GUIDE_TOKENS,
                "answer_prompt_token_proxy": prompt_tokens,
                "answer_prompt_token_cap": MAX_PROMPT_TOKENS,
                "answer_output_token_cap": MAX_ANSWER_OUTPUT_TOKENS,
                "s0_fallback": prediction_kind == "sealed_s0_control_fallback",
                "s0_fallback_reason": fallback_reason,
            }
        )
    answer_tokens = [
        row["answer_prompt_token_proxy"]
        for row in budget_rows
        if row["answer_prompt_token_proxy"] is not None
    ]
    guide_tokens = [
        row["guide_token_proxy"]
        for row in budget_rows
        if row["guide_token_proxy"] is not None
    ]
    artifact = {
        "format": RUN_FORMAT,
        "arm_label": ARM_LABEL,
        "parent_arm_label": PARENT_ARM_LABEL,
        "source_binding": dict(plan.features.source.binding),
        "retrieval_sha256": (
            plan.features.source.s0_plan.population.retrieval_sha256
        ),
        "baseline_final_answers_sha256": (
            plan.features.source.s0_plan.population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": (
            plan.features.source.s0_plan.population.population_identity_sha256
        ),
        "historical_validator_binding_sha256": (
            plan.features.source.s0_plan.population.binding_sha256
        ),
        "s0_control_run_sha256": plan.features.source.s0_run_sha256,
        "feature_artifact_sha256": plan.features.manifest_sha256,
        "feature_session_receipt_sha256": (
            plan.features.session.session_receipt_sha256
        ),
        "link_population_sha256": plan.population.population_sha256,
        "question_count": len(questions),
        "required_answer_calls": plan.unique_calls,
        "settings": {
            "model": DEFAULT_MODEL,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "answer_output_tokens": MAX_ANSWER_OUTPUT_TOKENS,
            "concept_count": CONCEPT_COUNT,
            "top_extraction_links_per_concept": (
                EXTRACTION_LINKS_PER_CONCEPT
            ),
            "guide_token_cap": MAX_GUIDE_TOKENS,
            "evidence_additions": 0,
            "retries": 0,
            "x_x1_ordering_proxy_consumed": False,
        },
        "answer_completion_batch": (
            None if batch is None else _stable_batch(batch)
        ),
        "budget": {
            "s0_non_borrowable": True,
            "exact_s0_membership_and_order": True,
            "evidence_additions": 0,
            "shared_residual_tokens": 0,
            "guide_tokens": _distribution(guide_tokens),
            "answer_prompt_tokens": _distribution(answer_tokens),
            "questions": budget_rows,
        },
        "structural_target_ledger": target_ledger,
        "questions": questions,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }
    if _contains_forbidden_key(artifact):
        raise RuntimeError("CAV run crossed the gold firewall")
    return artifact


def _answer_batch(
    plan: _AnswerPlan,
    args: argparse.Namespace,
    *,
    client: Any | None,
) -> FastCompletionBatch | None:
    runtime = _runtime(plan, args, client=client)
    return None if runtime is None else runtime.run()


def run_treatment(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_feature_model or args.authorized_feature_encoder_calls:
        raise ValueError("answer run forbids the feature model")
    plan = _build_answer_plan(args)
    if not args.enable_provider:
        raise ValueError("run requires --enable-provider")
    if args.authorized_provider_calls != plan.unique_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal the dependent "
            f"answer population ({args.authorized_provider_calls} != "
            f"{plan.unique_calls})"
        )
    path = _run_path(args)
    _guard_existing(path, plan.features.source, RUN_FORMAT)
    client = None
    if plan.unique_calls:
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        if not api_key:
            raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
    try:
        batch = _answer_batch(plan, args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    if batch is not None and (
        batch.prompt_population.unique_prompt_count != plan.unique_calls
        or batch.usage.physical_calls + batch.usage.checkpoint_hits
        != plan.unique_calls
    ):
        raise RuntimeError("CAV answer journal population changed")
    artifact = _run_artifact(plan, batch)
    return artifact, _publish(path, artifact)


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if (
        args.enable_provider
        or args.authorized_provider_calls
        or args.enable_feature_model
        or args.authorized_feature_encoder_calls
    ):
        raise ValueError("run replay forbids model/provider access")
    plan = _build_answer_plan(args)
    source, digest = _read(_run_path(args))
    batch = _answer_batch(plan, args, client=None)
    if batch is not None and batch.usage.physical_calls:
        raise RuntimeError("CAV answer replay unexpectedly made provider calls")
    expected = _run_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("CAV run differs from immutable answer journals")
    replay_path = Path(args.output_root) / "run-replay.json"
    replay_digest = _publish(replay_path, source)
    if replay_digest != digest:
        raise RuntimeError("CAV replay publication changed the sealed run digest")
    return source, replay_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "features",
            "feature-replay",
            "answer-preflight",
            "run",
            "replay",
        ),
        default="preflight",
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--baseline-answers",
        type=Path,
        default=DEFAULT_BASELINE_ANSWERS,
    )
    parser.add_argument(
        "--expected-baseline-answers-sha256",
        default=EXPECTED_BASELINE_ANSWERS_SHA256,
    )
    parser.add_argument("--s0-run", type=Path, default=DEFAULT_S0_RUN)
    parser.add_argument("--s0-checkpoint-dir", type=Path)
    parser.add_argument("--expected-s0-run-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--expected-features-sha256")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--event-cav", type=Path, default=DEFAULT_EVENT_CAV)
    parser.add_argument("--prefix-cav", type=Path, default=DEFAULT_PREFIX_CAV)
    parser.add_argument("--extraction-temperature", type=float, default=0.05)
    parser.add_argument("--reinjection-temperature", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--expected-question-count",
        type=int,
        default=EXPECTED_QUESTION_COUNT,
    )
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--enable-feature-model", action="store_true")
    parser.add_argument(
        "--authorized-feature-encoder-calls",
        type=int,
        default=0,
    )
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result: Any = run_preflight(args)
        command = " ".join(result["feature_plan"]["exact_feature_command_argv"])
        print(
            "S0_PLUS_CAV_LINKS preflight passed: "
            f"questions={result['question_count']}; "
            "feature_model_loads=0; feature_encoder_calls=0; "
            "provider_calls=0; writes=0\n"
            f"Exact feature command: {command}",
            flush=True,
        )
        return 0
    if args.phase == "features":
        result, digest = run_features(args)
        print(
            "S0_PLUS_CAV_LINKS features published: "
            f"sha256={digest}; encoder_calls="
            f"{result['feature_encoder_calls']}; provider_calls=0",
            flush=True,
        )
        return 0
    if args.phase == "feature-replay":
        result, digest = run_feature_replay(args)
    elif args.phase == "answer-preflight":
        result = run_answer_preflight(args)
        print(
            "S0_PLUS_CAV_LINKS answer preflight passed: "
            f"valid={result['valid_link_guide_count']}; "
            f"fallback={result['s0_fallback_count']}; "
            f"authorized_terra_calls="
            f"{result['required_authorized_provider_calls']}; "
            "provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    elif args.phase == "run":
        result, digest = run_treatment(args)
    else:
        result, digest = run_replay(args)
    print(
        f"{args.phase} verified {ARM_LABEL} artifact {digest}; "
        f"questions={result['question_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_PREFLIGHT_FORMAT",
    "ARM_LABEL",
    "PREFLIGHT_FORMAT",
    "RUN_FORMAT",
    "build_parser",
    "main",
    "run_answer_preflight",
    "run_feature_replay",
    "run_features",
    "run_preflight",
    "run_replay",
    "run_treatment",
]
