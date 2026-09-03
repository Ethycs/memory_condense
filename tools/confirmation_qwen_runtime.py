"""Prediction-only construction of the shared local Qwen selector/linker.

This is the model-construction seam formerly nested in the 1M evaluation CLI.
It depends only on the supplied frozen retrieval configuration and local model
directories; it has no dataset, split, benchmark, scorer, or provider route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.search.selectors.causal_choice_scorer import CausalChoiceScorer
from memory_condense.search.selectors.prefix_selector import QwenPrefixCoverageSelector


def load_shared_qwen(
    config: Any,
    prefix_dir: Path,
    choice_dir: Path,
) -> tuple[QwenPrefixCoverageSelector, QwenMemoryLinker]:
    """Load one prefix linker shared by S0 coverage and S2 discovery."""

    retrieval = config.retrieval
    if not prefix_dir.is_dir() or not choice_dir.is_dir():
        raise FileNotFoundError(
            "one or both local Qwen checkpoint directories are missing"
        )
    encoder = Qwen3PrefixEncoder(
        prefix_dir,
        layers=retrieval.coverage_selector_prefix_layers,
        device=retrieval.coverage_selector_prefix_device,
        dtype=retrieval.coverage_selector_prefix_dtype,
        model_id=retrieval.coverage_selector_prefix_model_id,
        model_revision=retrieval.coverage_selector_prefix_revision,
        expected_checkpoint_sha256=(
            retrieval.coverage_selector_prefix_checkpoint_sha256
        ),
    )
    linker = QwenMemoryLinker(
        encoder,
        layer=retrieval.coverage_selector_attention_layer,
        max_candidates=retrieval.coverage_selector_candidate_pool,
        max_workspace_tokens=retrieval.coverage_selector_max_workspace_tokens,
    )
    scorer = CausalChoiceScorer.from_local_checkpoint(
        choice_dir,
        model_id=retrieval.coverage_selector_choice_model_id,
        model_revision=retrieval.coverage_selector_choice_revision,
        expected_weights_sha256=(
            retrieval.coverage_selector_choice_checkpoint_sha256
        ),
        device=retrieval.coverage_selector_choice_device,
        dtype=retrieval.coverage_selector_choice_dtype,
        batch_size=retrieval.coverage_selector_choice_batch_size,
        max_candidates=retrieval.coverage_selector_choice_max_candidates,
        query_tokens=retrieval.coverage_selector_choice_query_tokens,
        candidate_tokens=retrieval.coverage_selector_choice_candidate_tokens,
        max_prompt_tokens=retrieval.coverage_selector_choice_max_prompt_tokens,
        max_workspace_tokens=(
            retrieval.coverage_selector_choice_max_workspace_tokens
        ),
        require_single_token_labels=True,
        strict=retrieval.coverage_selector_strict,
    )
    selector = QwenPrefixCoverageSelector(
        linker,
        score_provider=scorer,
        candidate_pool=retrieval.coverage_selector_candidate_pool,
        candidate_tokens=retrieval.coverage_selector_candidate_tokens,
        query_tokens=retrieval.coverage_selector_query_tokens,
        merge_similarity=retrieval.coverage_selector_merge_similarity,
        same_source_merge_similarity=(
            retrieval.coverage_selector_same_source_merge_similarity
        ),
        null_threshold=retrieval.coverage_selector_null_threshold,
        uncertainty_entropy=retrieval.coverage_selector_uncertainty_entropy,
        allow_selected_scope_fixed_k_closure=(
            retrieval.allow_selected_scope_fixed_k_closure
        ),
        strict=retrieval.coverage_selector_strict,
    )
    return selector, linker


__all__ = ["load_shared_qwen"]
