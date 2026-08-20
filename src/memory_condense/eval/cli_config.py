"""Pure translation from CLI argument contracts to immutable config."""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig


def _identity_kwargs(
    model_cls: type, args: argparse.Namespace, derived: dict[str, object]
) -> dict[str, object]:
    """Kwargs for every model field whose parser dest passes through unchanged.

    Fields with derived values are excluded by name, and fields without a
    matching parser dest (e.g. ``candidates``) keep their model defaults.
    """
    return {
        name: getattr(args, name)
        for name in model_cls.model_fields
        if name not in derived and hasattr(args, name)
    }


def _resolve_choice_identity(
    args: argparse.Namespace, choice_dir: Path | None
) -> tuple[str, str, str]:
    """Pin the forced-choice checkpoint to an exact id/revision/SHA-256."""
    if not choice_dir:
        return "", "", ""
    from memory_condense.search.selectors.causal_choice_scorer import (
        QWEN_CHOICE_MODEL_ID,
        QWEN_CHOICE_MODEL_REVISION,
        QWEN_CHOICE_WEIGHTS_SHA256,
        SMOLLM_CHOICE_MODEL_ID,
        SMOLLM_CHOICE_MODEL_REVISION,
        SMOLLM_CHOICE_WEIGHTS_SHA256,
    )

    explicit_identity = (
        args.coverage_selector_choice_model_id,
        args.coverage_selector_choice_revision,
        args.coverage_selector_choice_checkpoint_sha256,
    )
    if any(explicit_identity):
        if not all(explicit_identity):
            raise ValueError(
                "explicit choice identity requires model id, revision, "
                "and checkpoint SHA-256"
            )
        return explicit_identity
    if choice_dir.name.casefold() == "qwen3-0.6b".casefold():
        return (
            QWEN_CHOICE_MODEL_ID,
            QWEN_CHOICE_MODEL_REVISION,
            QWEN_CHOICE_WEIGHTS_SHA256,
        )
    if choice_dir.name.casefold() == "smollm2-360m-instruct".casefold():
        return (
            SMOLLM_CHOICE_MODEL_ID,
            SMOLLM_CHOICE_MODEL_REVISION,
            SMOLLM_CHOICE_WEIGHTS_SHA256,
        )
    raise ValueError(
        "unknown choice checkpoint directory; provide exact "
        "--coverage-selector-choice-model-id, "
        "--coverage-selector-choice-model-revision, and "
        "--coverage-selector-choice-checkpoint-sha256"
    )


def _resolve_prefix_identity(
    args: argparse.Namespace, qwen_prefix_dir: Path | None
) -> tuple[str, str, str, str, str]:
    """Pin the Qwen prefix checkpoint identity plus its device and dtype."""
    if not qwen_prefix_dir:
        return "", "", "", "", ""
    import torch

    from memory_condense.eval.local_qwen import resolve_local_qwen_dtype
    from memory_condense.modeling.qwen_prefix import (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
        expected_prefix_checkpoint_sha256,
    )

    prefix_device = str(args.coverage_selector_prefix_device)
    _prefix_torch_dtype, prefix_dtype = resolve_local_qwen_dtype(
        torch,
        args.coverage_selector_dtype,
        device=prefix_device,
    )
    return (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
        expected_prefix_checkpoint_sha256(args.coverage_selector_prefix_layers),
        prefix_device,
        prefix_dtype,
    )


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    # --hybrid predates --mode and is kept so the commands in
    # `docs/02 - Implementation/01` keep working.
    if args.qwen_feedback and not args.qwen_rerank_model_dir:
        raise ValueError("--qwen-feedback requires --qwen-rerank-model-dir")
    choice_dir = args.coverage_selector_choice_model_dir
    cross_encoder_dir = args.coverage_selector_cross_encoder_model_dir
    qwen_prefix_dir = args.coverage_selector_qwen_prefix_model_dir
    local_ini_dir = args.coverage_selector_local_model_dir
    if choice_dir and not qwen_prefix_dir:
        raise ValueError(
            "--coverage-selector-choice-model-dir requires "
            "--coverage-selector-qwen-prefix-model-dir"
        )
    if local_ini_dir and (qwen_prefix_dir or cross_encoder_dir or choice_dir):
        raise ValueError(
            "choose either the Qwen prefix/MS MARCO coverage path or the local "
            "INI classifier, not both"
        )
    if choice_dir and cross_encoder_dir:
        raise ValueError(
            "the forced-choice and MS MARCO score providers are separate "
            "coverage arms"
        )

    (
        choice_model_id,
        choice_revision,
        choice_checkpoint_sha256,
    ) = _resolve_choice_identity(args, choice_dir)
    (
        prefix_model_id,
        prefix_revision,
        prefix_checkpoint_sha256,
        prefix_device,
        prefix_dtype,
    ) = _resolve_prefix_identity(args, qwen_prefix_dir)

    coverage_selection = bool(
        cross_encoder_dir or qwen_prefix_dir or local_ini_dir or choice_dir
    )
    if choice_dir and qwen_prefix_dir:
        coverage_backend = "qwen_prefix_choice"
        coverage_model = f"{qwen_prefix_dir.name}+{choice_dir.name}"
    elif cross_encoder_dir and qwen_prefix_dir:
        coverage_backend = "cross_encoder_qwen_prefix"
        coverage_model = f"{cross_encoder_dir.name}+{qwen_prefix_dir.name}"
    elif cross_encoder_dir:
        coverage_backend = "cross_encoder"
        coverage_model = cross_encoder_dir.name
    elif qwen_prefix_dir:
        coverage_backend = "qwen_prefix"
        coverage_model = qwen_prefix_dir.name
    else:
        coverage_backend = "local_ini"
        coverage_model = local_ini_dir.name if local_ini_dir else ""
    if cross_encoder_dir:
        from memory_condense.search.selectors.cross_encoder_selector import (
            MS_MARCO_MODEL_ID,
            MS_MARCO_MODEL_REVISION,
            MS_MARCO_WEIGHTS_SHA256,
        )
    else:
        MS_MARCO_MODEL_ID = ""
        MS_MARCO_MODEL_REVISION = ""
        MS_MARCO_WEIGHTS_SHA256 = ""
    mode = "hybrid" if args.hybrid and args.mode == "dense" else args.mode
    routed_expansion_union = args.k + args.consolidation_chunk_slots
    if mode == "causal_graph":
        routed_expansion_union += args.neighbor_slots + args.source_slots
    cross_encoder_candidate_pool = (
        args.coverage_selector_cross_encoder_candidate_pool
        if args.coverage_selector_cross_encoder_candidate_pool is not None
        else max(
            128,
            routed_expansion_union,
            args.coverage_selector_candidate_pool,
        )
    )
    cross_encoder_semantic_rerank = bool(
        args.coverage_selector_cross_encoder_semantic_rerank
        and not args.coverage_selector_cross_encoder_score_only
    )

    retrieval_derived: dict[str, object] = dict(
        mode=mode,
        span_levels=tuple(
            int(x) for x in str(args.span_levels).split(",") if x.strip()
        ),
        qwen_rerank=(bool(args.qwen_rerank_model_dir) and not args.qwen_feedback),
        qwen_rerank_model=(
            args.qwen_rerank_model_dir.name if args.qwen_rerank_model_dir else ""
        ),
        coverage_selection=coverage_selection,
        coverage_selector_backend=coverage_backend,
        coverage_selector_model=coverage_model,
        coverage_selector_prefix_model_id=prefix_model_id,
        coverage_selector_prefix_revision=prefix_revision,
        coverage_selector_prefix_checkpoint_sha256=prefix_checkpoint_sha256,
        coverage_selector_prefix_device=prefix_device,
        coverage_selector_prefix_dtype=prefix_dtype,
        coverage_selector_cross_encoder_model_id=MS_MARCO_MODEL_ID,
        coverage_selector_cross_encoder_revision=MS_MARCO_MODEL_REVISION,
        coverage_selector_cross_encoder_checkpoint_sha256=MS_MARCO_WEIGHTS_SHA256,
        coverage_selector_cross_encoder_candidate_pool=cross_encoder_candidate_pool,
        coverage_selector_cross_encoder_semantic_rerank=cross_encoder_semantic_rerank,
        coverage_selector_choice_model_id=choice_model_id,
        coverage_selector_choice_revision=choice_revision,
        coverage_selector_choice_checkpoint_sha256=choice_checkpoint_sha256,
    )
    eval_derived: dict[str, object] = dict(
        chunker=ChunkerConfig(min_tokens=args.min_tokens, max_tokens=args.max_tokens),
        retrieval=RetrievalConfig(
            **retrieval_derived,
            **_identity_kwargs(RetrievalConfig, args, retrieval_derived),
        ),
        conversation_dir=(
            args.conversation_dir
            or args.benchmark_file
            or args.answer_recall
            or args.sufficiency_audit
            or ""
        ),
    )
    return EvalConfig(
        **eval_derived,
        **_identity_kwargs(EvalConfig, args, eval_derived),
    )
