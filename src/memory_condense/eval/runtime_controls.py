"""Stateful benchmark-ingest and transient-model adapters."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from memory_condense.eval.benchmark import ingest_sample
from memory_condense.eval.compiled_cache import compiled_store_ingest_fn
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import EvalConfig

def _benchmark_ingest_fn(
    args: argparse.Namespace,
    config: EvalConfig,
    *,
    prepare_only: bool = False,
):
    """Select an isolated benchmark writer without mutating compiled stores."""

    require_cache_hit = bool(
        not prepare_only and getattr(args, "benchmark_split", None) == "validation"
    )
    if require_cache_hit and config.retrieval.mode not in {
        "causal_consolidation",
        "causal_graph",
    }:
        raise ValueError(
            "locked validation requires causal compiled+learned cache receipts"
        )
    if config.retrieval.mode in {"causal_consolidation", "causal_graph"}:
        from memory_condense.eval.causal_benchmark import (
            causal_consolidation_ingest_fn,
        )

        return causal_consolidation_ingest_fn(
            args.compiled_store_cache,
            causal_cache_root=args.causal_store_cache,
            device=config.embedding_device,
            prepare_only=prepare_only,
            require_cache_hit=require_cache_hit,
        )
    if args.compiled_store_cache:
        return compiled_store_ingest_fn(
            args.compiled_store_cache,
            device=config.embedding_device,
            require_cache_hit=require_cache_hit,
        )
    return ingest_sample


def _load_candidate_reranker(args: argparse.Namespace, config: EvalConfig):
    """Load one shared, bounded Qwen control plane for a benchmark run."""

    if not (config.retrieval.qwen_rerank or config.retrieval.qwen_feedback):
        return None
    if args.qwen_rerank_model_dir is None:
        raise ValueError("Qwen attention requires --qwen-rerank-model-dir")
    from memory_condense.associations.association_store import AssociationArtifact
    from memory_condense.tooling.qwen_consolidation import load_qwen_linker
    from memory_condense.search.selectors.qwen_rerank import QwenCandidateReranker

    print(f"Loading bounded Qwen reranker from {args.qwen_rerank_model_dir}...")
    linker = load_qwen_linker(
        args.qwen_rerank_model_dir,
        prefix_layers=config.retrieval.qwen_rerank_prefix_layers,
        attention_layer=config.retrieval.qwen_rerank_attention_layer,
        cav_report=(
            args.qwen_rerank_cav_report
            if config.retrieval.qwen_rerank_use_cav
            else None
        ),
        cav_vectors=(
            args.qwen_rerank_cav_vectors
            if config.retrieval.qwen_rerank_use_cav
            else None
        ),
        cav_layer=config.retrieval.qwen_rerank_cav_layer,
        device=args.qwen_rerank_device,
        dtype=args.qwen_rerank_dtype,
        max_candidates=8,
        max_workspace_tokens=config.retrieval.qwen_rerank_max_workspace_tokens,
    )
    artifact = None
    if linker.cav_bank is not None:
        report_payload = json.loads(
            Path(args.qwen_rerank_cav_report).read_text(encoding="utf-8")
        )
        index_path = Path(args.qwen_rerank_model_dir) / "model.safetensors.index.json"
        artifact = AssociationArtifact.create(
            model_id=str(report_payload.get("model", "Qwen/Qwen3-8B")),
            checkpoint_id=f"safetensors-index:{file_sha256(index_path)}",
            prefix_layers=config.retrieval.qwen_rerank_prefix_layers,
            head_layer=config.retrieval.qwen_rerank_attention_layer,
            cav_layer=config.retrieval.qwen_rerank_cav_layer,
            concept_names=linker.cav_bank.names,
            head_count=int(linker.encoder.config.num_attention_heads),
            metadata={
                "cav_dataset_sha256": report_payload.get("dataset_sha256"),
                "cav_vectors_sha256": file_sha256(args.qwen_rerank_cav_vectors),
                "pooling": "conceptual-span-max-v1",
            },
        )
    return QwenCandidateReranker(
        linker,
        candidate_pool=(
            config.retrieval.qwen_feedback_candidate_pool
            if config.retrieval.qwen_feedback
            else config.retrieval.qwen_rerank_candidate_pool
        ),
        qwen_slots=(
            config.retrieval.qwen_feedback_seed_slots
            if config.retrieval.qwen_feedback
            else config.retrieval.qwen_rerank_slots
        ),
        group_size=config.retrieval.qwen_rerank_group_size,
        beam_per_group=config.retrieval.qwen_rerank_beam_per_group,
        candidate_tokens=config.retrieval.qwen_rerank_candidate_tokens,
        query_tokens=(
            config.retrieval.qwen_feedback_query_tokens
            if config.retrieval.qwen_feedback
            else config.retrieval.qwen_rerank_query_tokens
        ),
        score_weight=config.retrieval.qwen_rerank_score_weight,
        association_artifact=artifact,
    )


def _attach_candidate_reranker(ingest_fn, reranker):
    if reranker is None:
        return ingest_fn

    def attached(sample, config, data_dir):
        condenser = ingest_fn(sample, config, data_dir)
        condenser.set_source_candidate_reranker(reranker)
        artifact = getattr(reranker, "association_artifact", None)
        if artifact is not None:
            report = condenser.compile_indexed_cav_signatures(
                reranker.linker,
                artifact,
                batch_size=32,
                roles=("user",),
            )
            print(
                "  CAV concept index: "
                f"{report['compiled']} compiled, {report['reused']} reused, "
                f"{report['compiled_spans']} spans, "
                f"width={report['signature_width']}"
            )
        return condenser

    return attached


class _LazyQwenPrefixCoverageSelector:
    """Load the full-width prefix only after BGE has left the shared GPU."""

    requires_staged_gpu = True
    requires_baseline_ranking = True
    requires_complete_frontier = True

    def __init__(
        self,
        load,
        *,
        strict=False,
        allow_selected_scope_fixed_k_closure=False,
    ):
        self._load = load
        self._selector = None
        self.strict = bool(strict)
        self.allow_selected_scope_fixed_k_closure = bool(
            allow_selected_scope_fixed_k_closure
        )
        self.last_report = None
        self.last_source_companion_report = None
        self.last_candidate_trace = []
        self.load_elapsed_s = 0.0

    @property
    def loaded(self) -> bool:
        return self._selector is not None

    @staticmethod
    def requires_complete_frontier_for(query: str) -> bool:
        """Resolve query shape without loading either staged checkpoint."""

        from memory_condense.search.selectors.coverage_selector import compile_set_program

        return bool(compile_set_program(query).requires_completeness)

    def _ensure_loaded(self):
        if self._selector is None:
            started = time.perf_counter()
            self._selector = self._load()
            self.load_elapsed_s += time.perf_counter() - started
        return self._selector

    def select(self, query, candidates, **kwargs):
        self.last_candidate_trace = []
        selector = self._ensure_loaded()
        selected = selector.select(query, candidates, **kwargs)
        self.last_report = selector.last_report
        self.last_candidate_trace = list(
            getattr(selector, "last_candidate_trace", ())
        )
        return selected

    def select_source_companions(
        self,
        query,
        candidates_by_source,
        *,
        source_timestamps=None,
    ):
        self.last_source_companion_report = None
        selector = self._ensure_loaded()
        choose = getattr(selector, "select_source_companions", None)
        report_owner = selector
        if not callable(choose):
            score_provider = getattr(selector, "score_provider", None)
            choose = getattr(score_provider, "select_source_companions", None)
            report_owner = score_provider
        if not callable(choose):
            return {
                str(source_id): candidates[0]
                for source_id, candidates in candidates_by_source.items()
                if candidates
            }
        if source_timestamps is None:
            selected = choose(query, candidates_by_source)
        else:
            selected = choose(
                query,
                candidates_by_source,
                source_timestamps=source_timestamps,
            )
        self.last_source_companion_report = getattr(
            report_owner,
            "last_source_companion_report",
            None,
        )
        return selected

    def close(self) -> None:
        selector = self._selector
        self._selector = None
        self.last_report = None
        self.last_source_companion_report = None
        if selector is not None:
            selector.close()


class _LazyCrossEncoderCoverageSelector(_LazyQwenPrefixCoverageSelector):
    """Load semantic reranking only after BGE leaves the shared GPU."""

    requires_baseline_ranking = False

    def __init__(
        self,
        load,
        *,
        strict=False,
        semantic_rerank=True,
        semantic_score_only=False,
        allow_selected_scope_fixed_k_closure=False,
    ):
        super().__init__(
            load,
            strict=strict,
            allow_selected_scope_fixed_k_closure=(
                allow_selected_scope_fixed_k_closure
            ),
        )
        self.strict = bool(strict)
        self.semantic_rerank = bool(semantic_rerank)
        self.semantic_score_only = bool(semantic_score_only)
        self.requires_baseline_ranking = not self.semantic_rerank

    def select_source_companions(self, query, candidates_by_source):
        self.last_source_companion_report = None
        selector = self._ensure_loaded()
        selected = selector.select_source_companions(
            query,
            candidates_by_source,
        )
        self.last_source_companion_report = getattr(
            selector,
            "last_source_companion_report",
            None,
        )
        return selected


def _load_coverage_selector(args: argparse.Namespace, config: EvalConfig):
    """Load one shared transient coverage backend for a benchmark run."""

    if not config.retrieval.coverage_selection:
        return None
    if args.qwen_rerank_model_dir is not None:
        raise ValueError(
            "coverage selection and the source Qwen reranker are separate arms; "
            "measure them in separate processes"
        )

    def load_prefix_selector(*, score_provider=None):
        if args.coverage_selector_qwen_prefix_model_dir is None:
            raise ValueError(
                "Qwen prefix coverage selection requires "
                "--coverage-selector-qwen-prefix-model-dir"
            )

        import torch

        from memory_condense.search.selectors.coverage_selector import QwenPrefixCoverageSelector
        from memory_condense.eval.local_qwen import resolve_local_qwen_dtype
        from memory_condense.associations.head_memory import QwenMemoryLinker
        from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder

        _torch_dtype, dtype_name = resolve_local_qwen_dtype(
            torch,
            config.retrieval.coverage_selector_prefix_dtype
            or config.retrieval.coverage_selector_dtype,
            device=config.retrieval.coverage_selector_prefix_device
            or args.coverage_selector_prefix_device,
        )
        print(
            "Loading staged Qwen3-8B prefix coverage selector from "
            f"{args.coverage_selector_qwen_prefix_model_dir}...",
            flush=True,
        )
        encoder = Qwen3PrefixEncoder(
            args.coverage_selector_qwen_prefix_model_dir,
            layers=config.retrieval.coverage_selector_prefix_layers,
            device=config.retrieval.coverage_selector_prefix_device,
            dtype=dtype_name,
            model_id=config.retrieval.coverage_selector_prefix_model_id,
            model_revision=config.retrieval.coverage_selector_prefix_revision,
            expected_checkpoint_sha256=(
                config.retrieval.coverage_selector_prefix_checkpoint_sha256
            ),
        )
        linker = QwenMemoryLinker(
            encoder,
            layer=config.retrieval.coverage_selector_attention_layer,
            max_candidates=config.retrieval.coverage_selector_candidate_pool,
            max_workspace_tokens=(
                config.retrieval.coverage_selector_max_workspace_tokens
            ),
        )
        print(
            "  loaded layers: 0.."
            f"{config.retrieval.coverage_selector_prefix_layers - 1}; "
            "QK/OV readout layer: "
            f"{config.retrieval.coverage_selector_attention_layer}; "
            f"dtype: {dtype_name}; checkpoint: "
            f"{config.retrieval.coverage_selector_prefix_checkpoint_sha256[:12]}...; "
            "LM head: absent",
            flush=True,
        )
        return QwenPrefixCoverageSelector(
            linker,
            score_provider=score_provider,
            candidate_pool=config.retrieval.coverage_selector_candidate_pool,
            candidate_tokens=config.retrieval.coverage_selector_candidate_tokens,
            query_tokens=config.retrieval.coverage_selector_query_tokens,
            merge_similarity=config.retrieval.coverage_selector_merge_similarity,
            same_source_merge_similarity=(
                config.retrieval.coverage_selector_same_source_merge_similarity
            ),
            null_threshold=config.retrieval.coverage_selector_null_threshold,
            uncertainty_entropy=(
                config.retrieval.coverage_selector_uncertainty_entropy
            ),
            allow_selected_scope_fixed_k_closure=(
                config.retrieval.allow_selected_scope_fixed_k_closure
            ),
            strict=config.retrieval.coverage_selector_strict,
        )

    def load_choice_selector():
        from memory_condense.search.selectors.causal_choice_scorer import CausalChoiceScorer

        print(
            "Loading staged generation-free choice scorer from "
            f"{args.coverage_selector_choice_model_dir} "
            f"({config.retrieval.coverage_selector_choice_model_id}@"
            f"{config.retrieval.coverage_selector_choice_revision[:12]}, "
            "K/V cache disabled)...",
            flush=True,
        )
        scorer = CausalChoiceScorer.from_local_checkpoint(
            args.coverage_selector_choice_model_dir,
            model_id=config.retrieval.coverage_selector_choice_model_id,
            model_revision=(
                config.retrieval.coverage_selector_choice_revision
            ),
            expected_weights_sha256=(
                config.retrieval.coverage_selector_choice_checkpoint_sha256
            ),
            device=config.retrieval.coverage_selector_choice_device,
            dtype=config.retrieval.coverage_selector_choice_dtype,
            batch_size=config.retrieval.coverage_selector_choice_batch_size,
            max_candidates=(
                config.retrieval.coverage_selector_choice_max_candidates
            ),
            query_tokens=(
                config.retrieval.coverage_selector_choice_query_tokens
            ),
            candidate_tokens=(
                config.retrieval.coverage_selector_choice_candidate_tokens
            ),
            max_prompt_tokens=(
                config.retrieval.coverage_selector_choice_max_prompt_tokens
            ),
            max_workspace_tokens=(
                config.retrieval.coverage_selector_choice_max_workspace_tokens
            ),
            require_single_token_labels=True,
            strict=config.retrieval.coverage_selector_strict,
        )
        try:
            return load_prefix_selector(score_provider=scorer)
        except BaseException:
            scorer.close()
            raise

    def load_cross_encoder_selector():
        import gc

        import torch
        from sentence_transformers import CrossEncoder

        from memory_condense.search.selectors.cross_encoder_selector import (
            MS_MARCO_MODEL_ID,
            MS_MARCO_MODEL_REVISION,
            MSMarcoCrossEncoderSelector,
            verify_ms_marco_checkpoint,
        )

        checkpoint_sha256 = verify_ms_marco_checkpoint(
            args.coverage_selector_cross_encoder_model_dir
        )
        print(
            "Loading staged MS MARCO cross-encoder from "
            f"{args.coverage_selector_cross_encoder_model_dir} "
            f"({MS_MARCO_MODEL_ID}@{MS_MARCO_MODEL_REVISION[:12]}, "
            f"sha256={checkpoint_sha256[:12]}...)...",
            flush=True,
        )
        encoder = CrossEncoder(
            str(args.coverage_selector_cross_encoder_model_dir),
            device=config.retrieval.coverage_selector_cross_encoder_device,
            local_files_only=True,
            trust_remote_code=False,
            max_length=(
                config.retrieval.coverage_selector_cross_encoder_max_length
            ),
            model_kwargs={"use_safetensors": True},
        )
        duplicate_grouper = None
        try:
            if (
                config.retrieval.coverage_selector_backend
                == "cross_encoder_qwen_prefix"
            ):
                duplicate_grouper = load_prefix_selector()
            return MSMarcoCrossEncoderSelector(
                encoder,
                candidate_pool=(
                    config.retrieval.coverage_selector_cross_encoder_candidate_pool
                ),
                candidate_tokens=(
                    config.retrieval.coverage_selector_candidate_tokens
                ),
                query_tokens=config.retrieval.coverage_selector_query_tokens,
                batch_size=(
                    config.retrieval.coverage_selector_cross_encoder_batch_size
                ),
                max_length=(
                    config.retrieval.coverage_selector_cross_encoder_max_length
                ),
                max_workspace_tokens=(
                    config.retrieval.coverage_selector_max_workspace_tokens
                ),
                duplicate_grouper=duplicate_grouper,
                checkpoint_sha256=checkpoint_sha256,
                semantic_rerank=(
                    config.retrieval.coverage_selector_cross_encoder_semantic_rerank
                ),
                semantic_score_only=(
                    config.retrieval.coverage_selector_cross_encoder_score_only
                ),
                strict=config.retrieval.coverage_selector_strict,
            )
        except BaseException:
            if duplicate_grouper is not None:
                duplicate_grouper.close()
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def load_local_ini_selector():
        from memory_condense.search.selectors.coverage_selector import QueryConditionedCoverageSelector
        from memory_condense.eval.local_qwen import LocalQwenAnswerer

        print(
            "Loading staged local INI coverage selector from "
            f"{args.coverage_selector_local_model_dir}...",
            flush=True,
        )
        answerer = LocalQwenAnswerer(
            args.coverage_selector_local_model_dir,
            max_new_tokens=config.retrieval.coverage_selector_max_new_tokens,
            gpu_memory=args.coverage_selector_gpu_memory,
            cpu_memory=args.coverage_selector_cpu_memory,
            dtype=config.retrieval.coverage_selector_dtype,
            stop_strings=("[end]",),
        )
        print(
            f"  selector generation dtype: {answerer.dtype_name}",
            flush=True,
        )
        return QueryConditionedCoverageSelector(
            answerer,
            candidate_pool=config.retrieval.coverage_selector_candidate_pool,
            candidate_tokens=config.retrieval.coverage_selector_candidate_tokens,
            query_tokens=config.retrieval.coverage_selector_query_tokens,
            max_workspace_tokens=(
                config.retrieval.coverage_selector_max_workspace_tokens
            ),
            null_threshold=config.retrieval.coverage_selector_null_threshold,
            uncertainty_entropy=(
                config.retrieval.coverage_selector_uncertainty_entropy
            ),
            strict=config.retrieval.coverage_selector_strict,
        )

    # Backend registry: each spec keeps one backend's genuine differences —
    # lazy wrapper class, loader, eagerly-guarded model-dir argument (labeled
    # for its error message), and extra wrapper controls.  The guard and
    # construction tail below are shared.  Two deliberate asymmetries: the
    # qwen_prefix model-dir guard stays inside load_prefix_selector so it
    # still fails at load time rather than eagerly, and the local INI wrapper
    # is never strict.
    retrieval = config.retrieval
    strict = {"strict": retrieval.coverage_selector_strict}
    cross_encoder = (
        _LazyCrossEncoderCoverageSelector,
        load_cross_encoder_selector,
        ("MS MARCO", "coverage_selector_cross_encoder_model_dir"),
        {
            **strict,
            "semantic_rerank": (
                retrieval.coverage_selector_cross_encoder_semantic_rerank
            ),
            "semantic_score_only": (
                retrieval.coverage_selector_cross_encoder_score_only
            ),
        },
    )
    backends = {
        "qwen_prefix": (
            _LazyQwenPrefixCoverageSelector,
            load_prefix_selector,
            None,
            strict,
        ),
        "qwen_prefix_choice": (
            _LazyQwenPrefixCoverageSelector,
            load_choice_selector,
            ("forced-choice", "coverage_selector_choice_model_dir"),
            strict,
        ),
        "cross_encoder": cross_encoder,
        "cross_encoder_qwen_prefix": cross_encoder,
        "local_ini": (
            _LazyQwenPrefixCoverageSelector,
            load_local_ini_selector,
            ("local INI", "coverage_selector_local_model_dir"),
            {},
        ),
    }
    spec = backends.get(retrieval.coverage_selector_backend)
    if spec is None:
        raise ValueError(
            "unsupported coverage selector backend: "
            f"{retrieval.coverage_selector_backend}"
        )
    wrapper, loader, guard, extra = spec
    if guard is not None:
        label, required_arg = guard
        if getattr(args, required_arg) is None:
            raise ValueError(
                f"{label} coverage selection requires "
                f"--{required_arg.replace('_', '-')}"
            )
    return wrapper(
        loader,
        allow_selected_scope_fixed_k_closure=(
            retrieval.allow_selected_scope_fixed_k_closure
        ),
        **extra,
    )


def _attach_coverage_selector(ingest_fn, selector):
    if selector is None:
        return ingest_fn

    def attached(sample, config, data_dir):
        staged = bool(getattr(selector, "requires_staged_gpu", False))
        if staged and getattr(selector, "loaded", False):
            # A later sample needs BGE first; shed the prior sample's transient
            # selector before its frozen query-vector batch is prepared.
            selector.close()
        condenser = ingest_fn(sample, config, data_dir)
        if staged:
            release_embedder = getattr(ingest_fn, "release_embedder", None)
            if callable(release_embedder):
                print(
                    "Staging GPU: frozen query vectors ready; releasing BGE "
                    "before coverage-selector load.",
                    flush=True,
                )
                release_embedder()
        condenser.set_context_candidate_selector(selector)
        return condenser

    return attached


def _attach_runtime_controls(ingest_fn, *, reranker=None, selector=None):
    """Compose independent transient controls around one benchmark writer."""

    return _attach_coverage_selector(
        _attach_candidate_reranker(ingest_fn, reranker),
        selector,
    )


def _reserve_embedding_device_for_transient_models(
    args: argparse.Namespace,
) -> None:
    """Keep BGE on GPU when a causal run can release it before selection."""

    coverage_selector = bool(
        args.coverage_selector_local_model_dir
        or args.coverage_selector_qwen_prefix_model_dir
        or args.coverage_selector_cross_encoder_model_dir
    )
    staged_coverage = coverage_selector and args.mode in {
        "causal_consolidation",
        "causal_graph",
    }
    if (
        args.embedding_device is None
        and (args.qwen_rerank_model_dir or (coverage_selector and not staged_coverage))
    ):
        # Non-causal stores still need their live BGE embedder during search.
        args.embedding_device = "cpu"
