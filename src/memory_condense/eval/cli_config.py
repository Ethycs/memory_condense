"""Pure translation from CLI argument contracts to immutable config."""

from __future__ import annotations

import argparse

from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig

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

    choice_model_id = ""
    choice_revision = ""
    choice_checkpoint_sha256 = ""
    if choice_dir:
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
            (
                choice_model_id,
                choice_revision,
                choice_checkpoint_sha256,
            ) = explicit_identity
        elif choice_dir.name.casefold() == "qwen3-0.6b".casefold():
            choice_model_id = QWEN_CHOICE_MODEL_ID
            choice_revision = QWEN_CHOICE_MODEL_REVISION
            choice_checkpoint_sha256 = QWEN_CHOICE_WEIGHTS_SHA256
        elif choice_dir.name.casefold() == "smollm2-360m-instruct".casefold():
            choice_model_id = SMOLLM_CHOICE_MODEL_ID
            choice_revision = SMOLLM_CHOICE_MODEL_REVISION
            choice_checkpoint_sha256 = SMOLLM_CHOICE_WEIGHTS_SHA256
        else:
            raise ValueError(
                "unknown choice checkpoint directory; provide exact "
                "--coverage-selector-choice-model-id, "
                "--coverage-selector-choice-model-revision, and "
                "--coverage-selector-choice-checkpoint-sha256"
            )

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
    prefix_model_id = ""
    prefix_revision = ""
    prefix_checkpoint_sha256 = ""
    prefix_device = ""
    prefix_dtype = ""
    if qwen_prefix_dir:
        import torch

        from memory_condense.eval.local_qwen import resolve_local_qwen_dtype
        from memory_condense.modeling.qwen_prefix import (
            DEFAULT_MODEL_ID,
            DEFAULT_MODEL_REVISION,
            expected_prefix_checkpoint_sha256,
        )

        prefix_model_id = DEFAULT_MODEL_ID
        prefix_revision = DEFAULT_MODEL_REVISION
        prefix_checkpoint_sha256 = expected_prefix_checkpoint_sha256(
            args.coverage_selector_prefix_layers
        )
        prefix_device = str(args.coverage_selector_prefix_device)
        _prefix_torch_dtype, prefix_dtype = resolve_local_qwen_dtype(
            torch,
            args.coverage_selector_dtype,
            device=prefix_device,
        )
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
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=args.min_tokens, max_tokens=args.max_tokens),
        retrieval=RetrievalConfig(
            k=args.k,
            ef_search=args.ef_search,
            mode=mode,
            hybrid=args.hybrid,
            alpha=args.alpha,
            k_memories=args.k_memories,
            span_levels=tuple(
                int(x) for x in str(args.span_levels).split(",") if x.strip()
            ),
            k_per_level=args.k_per_level,
            k_sources=args.k_sources,
            source_slots=args.source_slots,
            source_candidate_pool=args.source_candidate_pool,
            source_activation_k=args.source_activation_k,
            query_facet_retrieval=args.query_facet_retrieval,
            query_facet_slots=args.query_facet_slots,
            query_facet_max=args.query_facet_max,
            role_aware_retrieval=args.role_aware_retrieval,
            role_user_weight=args.role_user_weight,
            role_assistant_weight=args.role_assistant_weight,
            role_system_weight=args.role_system_weight,
            multi_fact_source_diversity=args.multi_fact_source_diversity,
            source_tfisf_activation=args.source_tfisf_activation,
            source_tfisf_slots=args.source_tfisf_slots,
            source_hsc_activation=args.source_hsc_activation,
            source_hsc_slots=args.source_hsc_slots,
            source_hsc_hops=args.source_hsc_hops,
            source_hsc_chunk_slots=args.source_hsc_chunk_slots,
            source_local_search=args.source_local_search,
            source_partition_routing=args.source_partition_routing,
            source_partition_slots=args.source_partition_slots,
            source_partition_separator=args.source_partition_separator,
            qwen_rerank=(
                bool(args.qwen_rerank_model_dir) and not args.qwen_feedback
            ),
            qwen_rerank_candidate_pool=args.qwen_rerank_candidate_pool,
            qwen_rerank_slots=args.qwen_rerank_slots,
            qwen_rerank_group_size=args.qwen_rerank_group_size,
            qwen_rerank_beam_per_group=args.qwen_rerank_beam_per_group,
            qwen_rerank_candidate_tokens=args.qwen_rerank_candidate_tokens,
            qwen_rerank_query_tokens=args.qwen_rerank_query_tokens,
            qwen_rerank_score_weight=args.qwen_rerank_score_weight,
            qwen_rerank_model=(
                args.qwen_rerank_model_dir.name
                if args.qwen_rerank_model_dir
                else ""
            ),
            qwen_rerank_prefix_layers=args.qwen_rerank_prefix_layers,
            qwen_rerank_attention_layer=args.qwen_rerank_attention_layer,
            qwen_rerank_use_cav=args.qwen_rerank_use_cav,
            qwen_rerank_cav_layer=args.qwen_rerank_cav_layer,
            qwen_rerank_max_workspace_tokens=(
                args.qwen_rerank_max_workspace_tokens
            ),
            qwen_feedback=args.qwen_feedback,
            qwen_feedback_candidate_pool=args.qwen_feedback_candidate_pool,
            qwen_feedback_seed_slots=args.qwen_feedback_seed_slots,
            qwen_feedback_slots=args.qwen_feedback_slots,
            qwen_feedback_evidence_tokens=args.qwen_feedback_evidence_tokens,
            qwen_feedback_query_tokens=args.qwen_feedback_query_tokens,
            coverage_selection=coverage_selection,
            coverage_selector_backend=coverage_backend,
            coverage_selector_model=coverage_model,
            coverage_selector_dtype=args.coverage_selector_dtype,
            coverage_selector_prefix_model_id=prefix_model_id,
            coverage_selector_prefix_revision=prefix_revision,
            coverage_selector_prefix_checkpoint_sha256=(
                prefix_checkpoint_sha256
            ),
            coverage_selector_prefix_device=prefix_device,
            coverage_selector_prefix_dtype=prefix_dtype,
            coverage_selector_candidate_pool=(
                args.coverage_selector_candidate_pool
            ),
            coverage_selector_candidate_tokens=(
                args.coverage_selector_candidate_tokens
            ),
            coverage_selector_query_tokens=args.coverage_selector_query_tokens,
            coverage_selector_max_workspace_tokens=(
                args.coverage_selector_max_workspace_tokens
            ),
            coverage_selector_max_new_tokens=(
                args.coverage_selector_max_new_tokens
            ),
            coverage_selector_cross_encoder_model_id=MS_MARCO_MODEL_ID,
            coverage_selector_cross_encoder_revision=MS_MARCO_MODEL_REVISION,
            coverage_selector_cross_encoder_checkpoint_sha256=(
                MS_MARCO_WEIGHTS_SHA256
            ),
            coverage_selector_cross_encoder_device=(
                args.coverage_selector_cross_encoder_device
            ),
            coverage_selector_cross_encoder_candidate_pool=(
                cross_encoder_candidate_pool
            ),
            coverage_selector_cross_encoder_semantic_rerank=(
                cross_encoder_semantic_rerank
            ),
            coverage_selector_cross_encoder_score_only=(
                args.coverage_selector_cross_encoder_score_only
            ),
            coverage_selector_cross_encoder_batch_size=(
                args.coverage_selector_cross_encoder_batch_size
            ),
            coverage_selector_cross_encoder_max_length=(
                args.coverage_selector_cross_encoder_max_length
            ),
            coverage_selector_choice_model_id=choice_model_id,
            coverage_selector_choice_revision=choice_revision,
            coverage_selector_choice_checkpoint_sha256=(
                choice_checkpoint_sha256
            ),
            coverage_selector_choice_device=(
                args.coverage_selector_choice_device
            ),
            coverage_selector_choice_dtype=args.coverage_selector_choice_dtype,
            coverage_selector_choice_batch_size=(
                args.coverage_selector_choice_batch_size
            ),
            coverage_selector_choice_max_candidates=(
                args.coverage_selector_choice_max_candidates
            ),
            coverage_selector_choice_query_tokens=(
                args.coverage_selector_choice_query_tokens
            ),
            coverage_selector_choice_candidate_tokens=(
                args.coverage_selector_choice_candidate_tokens
            ),
            coverage_selector_choice_max_prompt_tokens=(
                args.coverage_selector_choice_max_prompt_tokens
            ),
            coverage_selector_choice_max_workspace_tokens=(
                args.coverage_selector_choice_max_workspace_tokens
            ),
            coverage_selector_null_threshold=(
                args.coverage_selector_null_threshold
            ),
            coverage_selector_uncertainty_entropy=(
                args.coverage_selector_uncertainty_entropy
            ),
            coverage_selector_prefix_layers=args.coverage_selector_prefix_layers,
            coverage_selector_attention_layer=(
                args.coverage_selector_attention_layer
            ),
            coverage_selector_merge_similarity=(
                args.coverage_selector_merge_similarity
            ),
            coverage_selector_same_source_merge_similarity=(
                args.coverage_selector_same_source_merge_similarity
            ),
            allow_selected_scope_fixed_k_closure=(
                args.allow_selected_scope_fixed_k_closure
            ),
            coverage_selector_strict=args.coverage_selector_strict,
            neighbor_radius=args.neighbor_radius,
            neighbor_slots=args.neighbor_slots,
            neighbor_replacement_slots=args.neighbor_replacement_slots,
            neighbor_direction=args.neighbor_direction,
            consolidation_chunk_slots=args.consolidation_chunk_slots,
            consolidation_hops=args.consolidation_hops,
            consolidation_candidates=args.consolidation_candidates,
            consolidation_diffusion_width=args.consolidation_diffusion_width,
            consolidation_min_count=args.consolidation_min_count,
            consolidation_expansion_tokens=args.consolidation_expansion_tokens,
            consolidation_training_expansion_tokens=(
                args.consolidation_training_expansion_tokens
            ),
            consolidation_budget_aware_packing=(
                args.consolidation_budget_aware_packing
            ),
            consolidation_source_diverse_packing=(
                args.consolidation_source_diverse_packing
            ),
            consolidation_query_aware_sentence_packing=(
                args.consolidation_query_aware_sentence_packing
            ),
            consolidation_max_sentences_per_expansion=(
                args.consolidation_max_sentences_per_expansion
            ),
            consolidation_information_gain_packing=(
                args.consolidation_information_gain_packing
            ),
            consolidation_min_information_gain_per_token=(
                args.consolidation_min_information_gain_per_token
            ),
            consolidation_source_metadata_packing=(
                args.consolidation_source_metadata_packing
            ),
            consolidation_training_k=args.consolidation_training_k,
            consolidation_max_event_nodes=args.consolidation_max_event_nodes,
            consolidation_new_event_nodes=args.consolidation_new_event_nodes,
            consolidation_max_training_prompt_tokens=(
                args.consolidation_max_training_prompt_tokens
            ),
        ),
        judge_model=args.judge_model,
        responder_model=args.responder_model,
        embedding_device=args.embedding_device,
        conversation_dir=(
            args.conversation_dir
            or args.benchmark_file
            or args.answer_recall
            or args.sufficiency_audit
            or ""
        ),
        results_dir=args.results_dir,
        max_conversations=args.max_conversations,
        recent_window=args.recent_window,
        accuracy_target=args.accuracy_target,
        min_target_questions=args.min_target_questions,
        max_prompt_tokens=args.max_prompt_tokens,
    )
