"""Policy contracts and deterministic identity/manifest validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.eval.benchmark import BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
    project_root,
)
from memory_condense.eval.schemas import EvalConfig
from memory_condense.eval.validation_profile import (
    claimed_validation_profile,
    validate_longmemeval_claim_profile,
)

def _apply_locked_split(
    args: argparse.Namespace,
    samples,
    *,
    verbose: bool = True,
):
    manifest_path = args.benchmark_split_manifest
    split = args.benchmark_split
    if bool(manifest_path) != bool(split):
        raise ValueError(
            "--benchmark-split-manifest and --benchmark-split must be used together"
        )
    if not manifest_path:
        return samples
    dataset_path = (
        args.answer_recall or args.sufficiency_audit or args.benchmark_file
    )
    manifest = load_split_manifest(manifest_path)
    selected = select_locked_split(
        samples,
        dataset_path=dataset_path,
        manifest=manifest,
        split=split,
    )
    if verbose:
        print(
            f"Locked split {split!r}: {len(selected)} / {len(samples)} samples "
            f"(dataset sha256 {manifest.dataset_sha256[:12]}...)"
        )
    return selected


def _apply_sample_offset(
    args: argparse.Namespace,
    samples,
    *,
    verbose: bool = True,
):
    offset = int(args.sample_offset)
    if offset < 0:
        raise ValueError("--sample-offset must be non-negative")
    if offset >= len(samples) and offset:
        raise ValueError(
            f"--sample-offset {offset} is outside the {len(samples)} samples"
        )
    if offset and verbose:
        print(f"Sample shard starts at locked-split offset {offset}")
    if offset:
        return samples[offset:]
    return samples


def _planned_provider_calls(
    samples,
    *,
    max_samples: int | None,
    local_answerer: bool,
    use_judge: bool,
    provider_retries: int = 0,
) -> int:
    if provider_retries < 0:
        raise ValueError("provider_retries must be non-negative")
    selected = samples[:max_samples] if max_samples is not None else samples
    questions = sum(len(sample.questions) for sample in selected)
    logical_calls = (0 if local_answerer else questions) + (
        questions if use_judge else 0
    )
    return logical_calls * (provider_retries + 1)


def _benchmark_evaluation_identity(
    args: argparse.Namespace,
    config: EvalConfig,
) -> dict[str, object]:
    """Execution controls that a frozen validation policy must precommit.

    Retrieval and chunking already live in the policy's ``retrieval`` object.
    These values cover the answer/judge protocol and the exact resumable stress
    shard, none of which are represented by :class:`EvalConfig.retrieval`.
    """

    return {
        "responder_model": config.responder_model,
        "judge_model": config.judge_model,
        "embedding_device": config.embedding_device,
        "benchmark_format": str(args.benchmark_format),
        "use_judge": bool(args.use_judge),
        "provider_retries": int(args.provider_retries),
        "max_provider_calls": int(args.max_provider_calls),
        "max_prompt_tokens": config.max_prompt_tokens,
        "prompt_cap_semantics": (
            "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
        ),
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": (
            int(args.local_qwen_max_new_tokens)
            if args.local_qwen_model_dir
            else BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
        "recent_window": config.recent_window,
        "accuracy_target": config.accuracy_target,
        "min_target_questions": config.min_target_questions,
        "stress_context_tokens": getattr(args, "stress_context_tokens", None),
        "stress_questions": int(getattr(args, "stress_questions", 10)),
        "stress_question_offset": int(
            getattr(args, "stress_question_offset", 0)
        ),
        "max_samples": args.max_samples,
        "sample_offset": int(args.sample_offset),
    }


_SOURCE_MODES = frozenset({"hybrid_source", "hybrid_graph", "causal_graph"})
_GRAPH_MODES = frozenset({"hybrid_graph", "causal_graph"})
_CONSOLIDATION_MODES = frozenset({"causal_consolidation", "causal_graph"})
_PREFIX_COVERAGE_BACKENDS = frozenset(
    {"qwen_prefix", "qwen_prefix_choice", "cross_encoder_qwen_prefix"}
)
_CROSS_ENCODER_COVERAGE_BACKENDS = frozenset(
    {"cross_encoder", "cross_encoder_qwen_prefix"}
)

_COVERAGE_PREFIX_FIELDS = (
    "coverage_selector_prefix_model_id",
    "coverage_selector_prefix_revision",
    "coverage_selector_prefix_checkpoint_sha256",
    "coverage_selector_prefix_device",
    "coverage_selector_prefix_dtype",
)

#: Qwen-rerank runtime keys shared verbatim by the rerank and feedback arms.
#: ``qwen_rerank_score_weight`` sits between the two runs in the rerank arm
#: only, so the shared keys are kept as an ordered head and tail.
_QWEN_RERANK_SHARED_HEAD = (
    "qwen_rerank_group_size",
    "qwen_rerank_beam_per_group",
    "qwen_rerank_candidate_tokens",
    "qwen_rerank_query_tokens",
)
_QWEN_RERANK_SHARED_TAIL = (
    "qwen_rerank_model",
    "qwen_rerank_prefix_layers",
    "qwen_rerank_attention_layer",
    "qwen_rerank_use_cav",
    "qwen_rerank_cav_layer",
    "qwen_rerank_max_workspace_tokens",
)


def _coverage_prefix_policy_identity(config: EvalConfig) -> dict[str, str]:
    """Exact runtime/checkpoint identity for prefix-backed coverage arms."""

    if config.retrieval.coverage_selector_backend not in (
        _PREFIX_COVERAGE_BACKENDS
    ):
        return {}
    return {
        field: getattr(config.retrieval, field)
        for field in _COVERAGE_PREFIX_FIELDS
    }


def _enabled(name: str) -> tuple[str, object]:
    """Table entry that records an active gate flag as literal ``True``."""

    return (name, lambda config: True)


#: Declarative form of the conditional retrieval identity. Each row is
#: ``(modes, flags, backends, entries)``: the row applies when the retrieval
#: mode is in ``modes`` (``None`` = always), every named boolean flag is set,
#: and — for coverage sub-blocks — the selector backend is in ``backends``.
#: Entries are either a retrieval field name copied verbatim into the
#: identity, or an explicit ``(key, fn(config))`` override for the few values
#: that are not pure ``config.retrieval`` attributes. Row and entry order is
#: the serialized key order and must not change.
_RETRIEVAL_IDENTITY_TABLE: tuple = (
    (
        None,
        (),
        None,
        (
            "mode",
            "k",
            "ef_search",
            "alpha",
            "candidates",
            "neighbor_radius",
            "neighbor_slots",
            "neighbor_replacement_slots",
            ("max_prompt_tokens", lambda config: config.max_prompt_tokens),
            ("chunker_min_tokens", lambda config: config.chunker.min_tokens),
            ("chunker_max_tokens", lambda config: config.chunker.max_tokens),
        ),
    ),
    (
        _SOURCE_MODES,
        (),
        None,
        (
            "source_slots",
            (
                "source_activation_k",
                lambda config: (
                    config.retrieval.source_activation_k or config.retrieval.k
                ),
            ),
            "source_candidate_pool",
        ),
    ),
    (_SOURCE_MODES, ("source_local_search",), None, (
        _enabled("source_local_search"),
    )),
    (_SOURCE_MODES, ("source_tfisf_activation",), None, (
        _enabled("source_tfisf_activation"),
        "source_tfisf_slots",
    )),
    (_SOURCE_MODES, ("source_hsc_activation",), None, (
        _enabled("source_hsc_activation"),
        "source_hsc_slots",
        "source_hsc_hops",
        "source_hsc_chunk_slots",
    )),
    (_SOURCE_MODES, ("source_partition_routing",), None, (
        _enabled("source_partition_routing"),
        "source_partition_slots",
        "source_partition_separator",
    )),
    (_SOURCE_MODES, ("qwen_rerank",), None, (
        _enabled("qwen_rerank"),
        "qwen_rerank_candidate_pool",
        "qwen_rerank_slots",
        *_QWEN_RERANK_SHARED_HEAD,
        "qwen_rerank_score_weight",
        *_QWEN_RERANK_SHARED_TAIL,
    )),
    (_SOURCE_MODES, ("qwen_feedback",), None, (
        _enabled("qwen_feedback"),
        "qwen_feedback_candidate_pool",
        "qwen_feedback_seed_slots",
        "qwen_feedback_slots",
        "qwen_feedback_evidence_tokens",
        "qwen_feedback_query_tokens",
        *_QWEN_RERANK_SHARED_HEAD,
        *_QWEN_RERANK_SHARED_TAIL,
    )),
    (_GRAPH_MODES, (), None, ("neighbor_direction",)),
    (_GRAPH_MODES, ("query_facet_retrieval",), None, (
        _enabled("query_facet_retrieval"),
        "query_facet_slots",
        "query_facet_max",
    )),
    (_GRAPH_MODES, ("role_aware_retrieval",), None, (
        _enabled("role_aware_retrieval"),
        "role_user_weight",
        "role_assistant_weight",
        "role_system_weight",
    )),
    (_GRAPH_MODES, ("multi_fact_source_diversity",), None, (
        _enabled("multi_fact_source_diversity"),
    )),
    (
        _CONSOLIDATION_MODES,
        (),
        None,
        (
            "consolidation_chunk_slots",
            "consolidation_hops",
            "consolidation_candidates",
            "consolidation_diffusion_width",
            "consolidation_min_count",
            "consolidation_expansion_tokens",
            "consolidation_training_expansion_tokens",
            "consolidation_budget_aware_packing",
            "consolidation_training_k",
            "consolidation_max_event_nodes",
            "consolidation_new_event_nodes",
            "consolidation_max_training_prompt_tokens",
        ),
    ),
    (_CONSOLIDATION_MODES, ("consolidation_source_diverse_packing",), None, (
        _enabled("consolidation_source_diverse_packing"),
    )),
    (
        _CONSOLIDATION_MODES,
        ("consolidation_query_aware_sentence_packing",),
        None,
        (
            _enabled("consolidation_query_aware_sentence_packing"),
            "consolidation_max_sentences_per_expansion",
        ),
    ),
    (
        _CONSOLIDATION_MODES,
        ("consolidation_information_gain_packing",),
        None,
        (
            _enabled("consolidation_information_gain_packing"),
            "consolidation_min_information_gain_per_token",
        ),
    ),
    (_CONSOLIDATION_MODES, ("consolidation_source_metadata_packing",), None, (
        _enabled("consolidation_source_metadata_packing"),
    )),
    (
        _CONSOLIDATION_MODES,
        ("coverage_selection",),
        None,
        (
            _enabled("coverage_selection"),
            "coverage_selector_backend",
            "coverage_selector_model",
            "coverage_selector_dtype",
            "coverage_selector_candidate_pool",
            "coverage_selector_candidate_tokens",
            "coverage_selector_query_tokens",
            "coverage_selector_max_workspace_tokens",
            "coverage_selector_max_new_tokens",
            "coverage_selector_null_threshold",
            "coverage_selector_uncertainty_entropy",
            "coverage_selector_prefix_layers",
            "coverage_selector_attention_layer",
            "coverage_selector_merge_similarity",
            "coverage_selector_same_source_merge_similarity",
            "coverage_selector_strict",
        ),
    ),
    (
        _CONSOLIDATION_MODES,
        ("coverage_selection", "allow_selected_scope_fixed_k_closure"),
        None,
        (_enabled("allow_selected_scope_fixed_k_closure"),),
    ),
    (
        _CONSOLIDATION_MODES,
        ("coverage_selection",),
        _CROSS_ENCODER_COVERAGE_BACKENDS,
        (
            "coverage_selector_cross_encoder_model_id",
            "coverage_selector_cross_encoder_revision",
            "coverage_selector_cross_encoder_checkpoint_sha256",
            "coverage_selector_cross_encoder_device",
            "coverage_selector_cross_encoder_candidate_pool",
            "coverage_selector_cross_encoder_semantic_rerank",
            "coverage_selector_cross_encoder_score_only",
            "coverage_selector_cross_encoder_batch_size",
            "coverage_selector_cross_encoder_max_length",
        ),
    ),
    (
        _CONSOLIDATION_MODES,
        ("coverage_selection",),
        _PREFIX_COVERAGE_BACKENDS,
        _COVERAGE_PREFIX_FIELDS,
    ),
    (
        _CONSOLIDATION_MODES,
        ("coverage_selection",),
        frozenset({"qwen_prefix_choice"}),
        (
            "coverage_selector_choice_model_id",
            "coverage_selector_choice_revision",
            "coverage_selector_choice_checkpoint_sha256",
            "coverage_selector_choice_device",
            "coverage_selector_choice_dtype",
            "coverage_selector_choice_batch_size",
            "coverage_selector_choice_max_candidates",
            "coverage_selector_choice_query_tokens",
            "coverage_selector_choice_candidate_tokens",
            "coverage_selector_choice_max_prompt_tokens",
            "coverage_selector_choice_max_workspace_tokens",
        ),
    ),
)


def _policy_retrieval_identity(config: EvalConfig) -> dict[str, object]:
    """Return the exact conditional retrieval identity enforced by policy files.

    Keeping construction separate from file verification lets a selected
    development command generate its manifest through the same code path that
    later rejects drift. Disabled arms stay out of the identity; every active
    arm includes both its explicit values and runtime defaults.
    """

    retrieval = config.retrieval
    expected: dict[str, object] = {}
    for modes, flags, backends, entries in _RETRIEVAL_IDENTITY_TABLE:
        if modes is not None and retrieval.mode not in modes:
            continue
        if not all(getattr(retrieval, flag) for flag in flags):
            continue
        if (
            backends is not None
            and retrieval.coverage_selector_backend not in backends
        ):
            continue
        for entry in entries:
            if isinstance(entry, str):
                expected[entry] = getattr(retrieval, entry)
            else:
                key, value_fn = entry
                expected[key] = value_fn(config)
    return expected


def _verified_policy_sha256(
    path: Path | None,
    *,
    config: EvalConfig,
    dataset_sha256: str,
    split_manifest: str | None,
    active_split: str | None = None,
    active_implementation_sha256: str | None = None,
    active_environment_lock_sha256: str | None = None,
    repository_root: str | Path | None = None,
    evaluation_identity: dict[str, object] | None = None,
    prepare_only: bool = False,
    implementation_sha256_fn=implementation_sha256,
    environment_lock_sha256_fn=environment_lock_sha256,
) -> str:
    if path is None:
        return ""
    policy_bytes = path.read_bytes()
    payload = json.loads(policy_bytes)
    if not isinstance(payload, dict):
        raise ValueError("policy manifest must be a JSON object")
    raw_status = payload.get("status")
    status = raw_status if isinstance(raw_status, str) else ""
    if not status or status.startswith("superseded"):
        raise ValueError(f"policy manifest is not active: {path}")
    locked_validation = active_split == "validation"
    if payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError("policy manifest dataset SHA-256 mismatch")
    if split_manifest is None or payload.get("split_manifest") != Path(
        split_manifest
    ).name:
        raise ValueError("policy manifest locked-split identity mismatch")
    if locked_validation and "split" not in payload:
        raise ValueError("validation policy manifest must bind the active split")
    if "split" in payload:
        expected_split = payload["split"]
        if (
            not isinstance(expected_split, str)
            or not expected_split
            or expected_split != active_split
        ):
            raise ValueError("policy manifest active split mismatch")
    if locked_validation:
        if payload.get("format") != "memory-condense-retrieval-policy-v1":
            raise ValueError("validation policy manifest format mismatch")
        if status != "validation_frozen":
            raise ValueError(
                "validation requires a policy with status 'validation_frozen'"
            )
    expected_split_sha256 = _optional_policy_sha256(
        payload,
        "split_manifest_sha256",
    )
    if locked_validation and expected_split_sha256 is None:
        raise ValueError("validation policy must bind split_manifest_sha256")
    if expected_split_sha256 is not None:
        actual_split_sha256 = file_sha256(split_manifest)
        if actual_split_sha256 != expected_split_sha256:
            raise ValueError("policy manifest locked-split SHA-256 mismatch")

    expected_implementation_sha256 = _optional_policy_sha256(
        payload,
        "implementation_sha256",
    )
    if locked_validation and expected_implementation_sha256 is None:
        raise ValueError("validation policy must bind implementation_sha256")
    if expected_implementation_sha256 is not None:
        actual_implementation_sha256 = (
            active_implementation_sha256 or implementation_sha256_fn()
        ).casefold()
        if actual_implementation_sha256 != expected_implementation_sha256:
            raise ValueError("policy manifest implementation SHA-256 mismatch")

    expected_environment_sha256 = _optional_policy_sha256(
        payload,
        "environment_lock_sha256",
    )
    if locked_validation and expected_environment_sha256 is None:
        raise ValueError("validation policy must bind environment_lock_sha256")
    if expected_environment_sha256 is not None:
        actual_environment_sha256 = (
            active_environment_lock_sha256 or environment_lock_sha256_fn()
        ).casefold()
        if actual_environment_sha256 != expected_environment_sha256:
            raise ValueError("policy manifest environment-lock SHA-256 mismatch")

    selection_artifact_required = payload.get(
        "selection_artifact_required",
        False,
    )
    if not isinstance(selection_artifact_required, bool):
        raise ValueError(
            "policy manifest selection_artifact_required must be boolean"
        )
    if locked_validation and not selection_artifact_required:
        raise ValueError(
            "validation policy must require its development selection artifact"
        )
    if selection_artifact_required:
        has_selection_artifact = "selection_artifact" in payload
        has_selection_sha256 = "selection_artifact_sha256" in payload
        if not has_selection_artifact or not has_selection_sha256:
            raise ValueError(
                "policy manifest selection artifact and SHA-256 must be "
                "provided together when required"
            )
        selection_sha256 = _optional_policy_sha256(
            payload,
            "selection_artifact_sha256",
        )
        assert selection_sha256 is not None
        selection_path = _policy_repository_file(
            payload["selection_artifact"],
            field="selection_artifact",
            repository_root=repository_root,
        )
        if file_sha256(selection_path) != selection_sha256:
            raise ValueError("policy manifest selection artifact SHA-256 mismatch")

    if locked_validation:
        frozen_evaluation = payload.get("evaluation")
        if not isinstance(frozen_evaluation, dict):
            raise ValueError("validation policy must contain an evaluation object")
        frozen_evaluation = dict(frozen_evaluation)
        claim_profile = claimed_validation_profile(payload)
        if claim_profile:
            validate_longmemeval_claim_profile(payload, frozen_evaluation)
        sample_offsets = frozen_evaluation.pop("sample_offsets", None)
        if (
            not isinstance(sample_offsets, list)
            or not sample_offsets
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in sample_offsets
            )
            or len(set(sample_offsets)) != len(sample_offsets)
        ):
            raise ValueError(
                "validation evaluation.sample_offsets must be unique "
                "non-negative integers"
            )
        if frozen_evaluation.get("use_judge") is not True:
            raise ValueError("validation evaluation must enable the judge")
        if frozen_evaluation.get("provider_retries") != 0:
            raise ValueError("validation evaluation must freeze provider_retries=0")
        stress_target = frozen_evaluation.get("stress_context_tokens")
        stress_questions = frozen_evaluation.get("stress_questions")
        if (
            isinstance(stress_target, bool)
            or not isinstance(stress_target, int)
            or stress_target < 1
        ):
            raise ValueError(
                "validation evaluation must set a positive stress_context_tokens"
            )
        if (
            isinstance(stress_questions, bool)
            or not isinstance(stress_questions, int)
            or stress_questions < 1
        ):
            raise ValueError(
                "validation evaluation must set a positive stress_questions"
            )
        if frozen_evaluation.get("stress_question_offset") != 0:
            raise ValueError(
                "validation evaluation must freeze stress_question_offset=0"
            )
        if frozen_evaluation.get("max_samples") != 1:
            raise ValueError("validation evaluation must freeze max_samples=1")
        for field in ("responder_model", "judge_model", "embedding_device"):
            value = frozen_evaluation.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"validation evaluation must bind a non-empty {field}"
                )
        if frozen_evaluation.get("benchmark_format") != "longmemeval":
            raise ValueError(
                "validation evaluation must freeze "
                "benchmark_format='longmemeval'"
            )
        max_prompt_tokens = frozen_evaluation.get("max_prompt_tokens")
        if (
            isinstance(max_prompt_tokens, bool)
            or not isinstance(max_prompt_tokens, int)
            or max_prompt_tokens < 1
        ):
            raise ValueError(
                "validation evaluation must set a positive max_prompt_tokens"
            )
        accuracy_target = frozen_evaluation.get("accuracy_target")
        if (
            isinstance(accuracy_target, bool)
            or not isinstance(accuracy_target, (int, float))
            or not math.isfinite(float(accuracy_target))
            or not 0.0 <= float(accuracy_target) <= 1.0
        ):
            raise ValueError(
                "validation evaluation must set accuracy_target in [0, 1]"
            )
        min_target_questions = frozen_evaluation.get("min_target_questions")
        if (
            isinstance(min_target_questions, bool)
            or not isinstance(min_target_questions, int)
            or min_target_questions < 1
        ):
            raise ValueError(
                "validation evaluation must set a positive min_target_questions"
            )
        recent_window = frozen_evaluation.get("recent_window")
        if (
            isinstance(recent_window, bool)
            or not isinstance(recent_window, int)
            or recent_window < 0
        ):
            raise ValueError(
                "validation evaluation must set a non-negative recent_window"
            )
        authorization = frozen_evaluation.get("max_provider_calls")
        if (
            isinstance(authorization, bool)
            or not isinstance(authorization, int)
            or authorization != 2 * stress_questions
        ):
            raise ValueError(
                "validation evaluation must authorize exactly one responder "
                "and one judge call per question"
            )
        if evaluation_identity is None:
            raise ValueError("validation policy requires active evaluation identity")
        active_evaluation = dict(evaluation_identity)
        active_offset = active_evaluation.pop("sample_offset", None)
        if (
            isinstance(active_offset, bool)
            or not isinstance(active_offset, int)
            or active_offset not in sample_offsets
        ):
            raise ValueError("validation shard sample_offset is not in the policy")
        if prepare_only:
            cache_shaping_fields = (
                "embedding_device",
                "benchmark_format",
                "stress_context_tokens",
                "stress_questions",
                "stress_question_offset",
                "max_samples",
            )
            expected_prepare = {
                field: frozen_evaluation.get(field)
                for field in cache_shaping_fields
            }
            actual_prepare = {
                field: active_evaluation.get(field)
                for field in cache_shaping_fields
            }
            if actual_prepare != expected_prepare:
                raise ValueError(
                    "policy manifest cache-preparation config mismatch: expected "
                    f"{expected_prepare}, got {actual_prepare}"
                )
        else:
            if active_evaluation != frozen_evaluation:
                raise ValueError(
                    "policy manifest evaluation config mismatch: expected "
                    f"{frozen_evaluation}, got {active_evaluation}"
                )

    retrieval = payload.get("retrieval", {})
    expected = _policy_retrieval_identity(config)
    if retrieval != expected:
        raise ValueError(
            f"policy manifest retrieval config mismatch: expected {retrieval}, "
            f"got {expected}"
        )
    return hashlib.sha256(policy_bytes).hexdigest()


def _optional_policy_sha256(payload: dict, field: str) -> str | None:
    if field not in payload:
        return None
    value = payload[field]
    if not isinstance(value, str):
        raise ValueError(f"policy manifest {field} must be a SHA-256 digest")
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"policy manifest {field} must be a SHA-256 digest")
    return normalized


def _policy_repository_file(
    value: object,
    *,
    field: str,
    repository_root: str | Path | None,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"policy manifest {field} must be a repository-relative path"
        )
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"policy manifest {field} must be a safe repository-relative path"
        )
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else project_root().resolve()
    )
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"policy manifest {field} must stay within the repository"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"policy manifest {field} does not name an existing file")
    return candidate
