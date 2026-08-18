from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Default models for the eval harness.
#
# anthropic/claude-3-5-haiku-20241022 (the previous default for both roles) was
# retired on 2026-02-19 and now 404s, so every run failed. Replacements:
#   * responder -> anthropic/claude-haiku-4-5   (documented replacement for 3.5 Haiku)
#   * judge     -> anthropic/claude-sonnet-5    (stronger, different tier than the
#                                                responder, which also removes the
#                                                judge==responder validity problem)
DEFAULT_RESPONDER_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"


def _coerce_int(value: Any) -> int:
    """Best-effort int coercion.

    Provider usage objects vary a lot (and tests hand us mocks), so anything
    that is not already an int/float is treated as 0 rather than exploding.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


class UsageStats(BaseModel):
    """Token + latency accounting for one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    elapsed_s: float = 0.0
    calls: int = 0

    model_config = {"frozen": True}

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: UsageStats) -> UsageStats:
        if not isinstance(other, UsageStats):
            return NotImplemented
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            elapsed_s=self.elapsed_s + other.elapsed_s,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other: Any) -> UsageStats:
        # Lets sum([...]) work without an explicit start value.
        if other == 0:
            return self
        return self.__add__(other)

    @classmethod
    def from_litellm(cls, response: Any, elapsed_s: float) -> UsageStats:
        """Extract usage from a litellm completion response.

        Defensive on purpose: field names vary by provider and some providers
        omit usage entirely.
        """
        usage = getattr(response, "usage", None)

        cache_read = _coerce_int(getattr(usage, "cache_read_input_tokens", 0))
        if not cache_read:
            details = getattr(usage, "prompt_tokens_details", None)
            cache_read = _coerce_int(getattr(details, "cached_tokens", 0))

        return cls(
            input_tokens=_coerce_int(getattr(usage, "prompt_tokens", 0)),
            output_tokens=_coerce_int(getattr(usage, "completion_tokens", 0)),
            cache_read_input_tokens=cache_read,
            elapsed_s=elapsed_s,
            calls=1,
        )


class ChunkerConfig(BaseModel):
    min_tokens: int = 120
    max_tokens: int = 250

    model_config = {"frozen": True}


#: What the responder is given.
#:
#: * ``dense``  — top-k chunks by cosine (the historical baseline)
#: * ``hybrid`` — the same, with BM25 blended in
#: * ``memory`` — ``MemoryCondenser.build_context``: the memory-item header,
#:   verbatim expansions, and the recent window, all token-budgeted
#:
#: One field rather than a second boolean because ``hybrid=False,
#: memory=True`` is not a meaningful cell — the memory arm decides internally
#: whether its expansions are hybrid.
#:
#: **This is what makes the memory layer measurable at all.** Until it existed
#: both eval paths called ``mc.search``/``mc.search_hybrid`` directly, so
#: ``ContextPacker``, ``MemoryStore.retrieve``, ``rank_score`` and ``decay``
#: were exercised by no run.
#: * ``span``   — pools contiguous chunks up to a token target and matches the
#:   pooled vector, returning member chunks. The arm that matters on short-turn
#:   dialogue, where a single chunk is too small to carry retrievable signal.
RetrievalMode = Literal[
    "dense",
    "hybrid",
    "memory",
    "span",
    "source",
    "anchored_source",
    "hybrid_source",
    "hybrid_graph",
    "hybrid_neighbor",
    "causal_consolidation",
    "causal_graph",
]


class RetrievalConfig(BaseModel):
    k: int = 10
    ef_search: int = 50
    mode: RetrievalMode = "dense"
    #: Blend BM25 lexical candidates with the dense ones. Off by default so the
    #: k=0/k=N ablation keeps measuring the same dense baseline as before.
    #: Kept alongside ``mode`` for wire compatibility with runs saved before it
    #: existed; ``effective_hybrid`` is what code should read.
    hybrid: bool = False
    #: Dense weight when hybrid blending is on (1.0 == pure dense).
    alpha: float = 0.65
    #: Candidate pool size per side before reranking.
    candidates: int = 100
    #: Memory items requested for the header in ``memory`` mode.
    k_memories: int = 8
    #: Token target per pooled span, per level, in ``span`` mode. Tokens rather
    #: than chunk counts so one setting holds across corpora whose turns differ
    #: by an order of magnitude in length.
    span_levels: tuple[int, ...] = (110, 220)
    #: Spans taken from each level before merging. Stratified deliberately —
    #: a single mixed-granularity pool lets short chunks crowd out every span.
    k_per_level: int = 2
    #: Complete conversation/document sources selected in ``source`` mode.
    k_sources: int = Field(default=4, ge=1)
    #: Lower-ranked hybrid candidates admitted from sources activated by top-k.
    source_slots: int = Field(default=24, ge=0)
    #: Bounded global pool searched before source-conditioned admission.
    source_candidate_pool: int = Field(default=200, ge=1)
    #: Pool prefix whose source identities may admit second-stage candidates.
    source_activation_k: int | None = Field(default=None, ge=1)
    #: Split explicit colon-delimited lists into bounded retrieval queries and
    #: reserve source slots round-robin so every stated facet can contribute.
    query_facet_retrieval: bool = False
    query_facet_slots: int = Field(default=6, ge=0)
    query_facet_max: int = Field(default=4, ge=1)
    #: For explicitly first-person questions, apply a transient role prior so
    #: user statements outrank assistant suggestions with similar wording.
    role_aware_retrieval: bool = False
    role_user_weight: float = Field(default=1.25, ge=0.0)
    role_assistant_weight: float = Field(default=0.75, ge=0.0)
    role_system_weight: float = Field(default=0.50, ge=0.0)
    #: For explicit order/set questions, round-robin chunks by source so one
    #: session cannot monopolize anchors or the source-local candidate reserve.
    multi_fact_source_diversity: bool = False
    #: Add a bounded source-level TF-ISF activation channel derived from the
    #: live chunk-term index. Raw chunks remain the authoritative payload.
    source_tfisf_activation: bool = False
    source_tfisf_slots: int = Field(default=8, ge=1)
    #: Expand activated source leaves through bounded pairwise-contraction
    #: parents, then reserve a few source slots for their original chunks.
    source_hsc_activation: bool = False
    source_hsc_slots: int = Field(default=8, ge=1)
    source_hsc_hops: int = Field(default=2, ge=1)
    source_hsc_chunk_slots: int = Field(default=8, ge=1)
    #: Scan and rerank candidates inside activated sources instead of merely
    #: filtering the global candidate pool. False preserves historical arms.
    source_local_search: bool = False
    #: Route through hierarchical ``partition::source`` provenance before
    #: chunk competition. Off by default for historical-arm reproducibility.
    source_partition_routing: bool = False
    source_partition_slots: int = Field(default=3, ge=1)
    source_partition_separator: str = Field(default="::", min_length=1)
    #: Use a transient Qwen prefix to choose a bounded reserve of candidates
    #: from the source-local pool. The CLI enables this only when it can load
    #: an explicit local checkpoint; no transformer state enters the store.
    qwen_rerank: bool = False
    qwen_rerank_candidate_pool: int = Field(default=64, ge=1)
    qwen_rerank_slots: int = Field(default=6, ge=1)
    qwen_rerank_group_size: int = Field(default=8, ge=2)
    qwen_rerank_beam_per_group: int = Field(default=2, ge=1)
    qwen_rerank_candidate_tokens: int = Field(default=64, ge=1)
    qwen_rerank_query_tokens: int = Field(default=96, ge=1)
    qwen_rerank_score_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    qwen_rerank_model: str = ""
    qwen_rerank_prefix_layers: int = Field(default=2, ge=1)
    qwen_rerank_attention_layer: int = Field(default=1, ge=0)
    qwen_rerank_use_cav: bool = False
    qwen_rerank_cav_layer: int = Field(default=5, ge=0)
    qwen_rerank_max_workspace_tokens: int = Field(default=1024, ge=1)
    #: Two-hop treatment: attend over first-round evidence, then spend a fixed
    #: reserve on another source-local retrieval instead of directly reranking.
    qwen_feedback: bool = False
    qwen_feedback_candidate_pool: int = Field(default=32, ge=1)
    qwen_feedback_seed_slots: int = Field(default=6, ge=1)
    qwen_feedback_slots: int = Field(default=12, ge=0)
    qwen_feedback_evidence_tokens: int = Field(default=48, ge=1)
    qwen_feedback_query_tokens: int = Field(default=384, ge=1)
    #: Run a transient query-conditioned event selector over the complete
    #: bounded expansion set, then place one representative before duplicate
    #: support. Raw chunks remain the payload and all model state is shed.
    coverage_selection: bool = False
    coverage_selector_backend: Literal[
        "local_ini",
        "qwen_prefix",
        "qwen_prefix_choice",
        "cross_encoder",
        "cross_encoder_qwen_prefix",
    ] = "local_ini"
    coverage_selector_model: str = ""
    coverage_selector_dtype: Literal[
        "auto", "bfloat16", "float16", "float32"
    ] = "auto"
    coverage_selector_prefix_model_id: str = ""
    coverage_selector_prefix_revision: str = ""
    coverage_selector_prefix_checkpoint_sha256: str = ""
    coverage_selector_prefix_device: str = ""
    coverage_selector_prefix_dtype: str = ""
    coverage_selector_candidate_pool: int = Field(default=64, ge=1)
    coverage_selector_candidate_tokens: int = Field(default=96, ge=1)
    coverage_selector_query_tokens: int = Field(default=192, ge=1)
    coverage_selector_max_workspace_tokens: int = Field(default=8192, ge=1)
    coverage_selector_max_new_tokens: int = Field(default=4096, ge=1)
    coverage_selector_cross_encoder_model_id: str = ""
    coverage_selector_cross_encoder_revision: str = ""
    coverage_selector_cross_encoder_checkpoint_sha256: str = ""
    coverage_selector_cross_encoder_device: str = "cuda"
    #: Semantic reranking has its own frontier.  In the composite arm this is
    #: deliberately wider than the bounded Qwen duplicate-grouping prefix.
    coverage_selector_cross_encoder_candidate_pool: int = Field(
        default=128,
        ge=1,
    )
    coverage_selector_cross_encoder_semantic_rerank: bool = True
    coverage_selector_cross_encoder_score_only: bool = False
    coverage_selector_cross_encoder_batch_size: int = Field(default=32, ge=1)
    coverage_selector_cross_encoder_max_length: int = Field(default=256, ge=1)
    coverage_selector_choice_model_id: str = ""
    coverage_selector_choice_revision: str = ""
    coverage_selector_choice_checkpoint_sha256: str = ""
    coverage_selector_choice_device: str = "cuda"
    coverage_selector_choice_dtype: Literal[
        "auto", "bfloat16", "float16", "float32"
    ] = "auto"
    coverage_selector_choice_batch_size: int = Field(default=8, ge=1)
    coverage_selector_choice_max_candidates: int = Field(default=128, ge=1)
    coverage_selector_choice_query_tokens: int = Field(default=192, ge=1)
    coverage_selector_choice_candidate_tokens: int = Field(default=128, ge=1)
    coverage_selector_choice_max_prompt_tokens: int = Field(default=512, ge=1)
    coverage_selector_choice_max_workspace_tokens: int = Field(default=8192, ge=1)
    coverage_selector_null_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
    )
    coverage_selector_uncertainty_entropy: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
    )
    coverage_selector_prefix_layers: int = Field(default=6, ge=1)
    coverage_selector_attention_layer: int = Field(default=5, ge=0)
    coverage_selector_merge_similarity: float = Field(
        default=0.985,
        ge=0.0,
        le=1.0,
    )
    coverage_selector_same_source_merge_similarity: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
    )
    #: Frozen policy exception: permit a fully audited FIXED-K set from the
    #: selected partition scope to close the prompt tail even though global
    #: recall is not proven. Off by default; reports must label the weaker
    #: closure scope explicitly.
    allow_selected_scope_fixed_k_closure: bool = False
    coverage_selector_strict: bool = False
    #: Source-local chunk shells exposed around hybrid anchors.
    neighbor_radius: int = Field(default=1, ge=0)
    #: Hard count of additional neighbor chunks; direct anchors never compete.
    neighbor_slots: int = Field(default=5, ge=0)
    #: When positive, transition candidates replace this many weakest anchors.
    neighbor_replacement_slots: int = Field(default=0, ge=0)
    #: Restrict transition expansion to the useful temporal direction.
    neighbor_direction: Literal["both", "previous", "next"] = "both"
    #: Live Hebbian/QK graph candidates appended to the direct hybrid result.
    consolidation_chunk_slots: int = Field(default=3, ge=0)
    #: Degree-two diffusion was the first policy to recover a unique held-out
    #: build-session probe; one hop remains available as an ablation.
    consolidation_hops: int = Field(default=2, ge=1)
    consolidation_candidates: int = Field(default=128, ge=1)
    consolidation_diffusion_width: int = Field(default=32, ge=1)
    consolidation_min_count: int = Field(default=2, ge=1)
    #: Independent evidence budget used by ContextPacker before the benchmark's
    #: complete 8k responder-prompt cap is applied.
    consolidation_expansion_tokens: int = Field(default=1600, ge=1)
    consolidation_training_expansion_tokens: int = Field(default=1600, ge=1)
    consolidation_budget_aware_packing: bool = True
    consolidation_source_diverse_packing: bool = False
    consolidation_query_aware_sentence_packing: bool = False
    consolidation_max_sentences_per_expansion: int = Field(default=2, ge=1)
    consolidation_information_gain_packing: bool = False
    consolidation_min_information_gain_per_token: float = Field(
        default=0.0,
        ge=0.0,
    )
    consolidation_source_metadata_packing: bool = False
    #: Write-side causal replay bounds. They limit compute/workspace, not the
    #: amount of durable conversation evidence stored in SQLite/HNSW.
    consolidation_training_k: int = Field(default=10, ge=1)
    consolidation_max_event_nodes: int = Field(default=9, ge=2)
    consolidation_new_event_nodes: int = Field(default=5, ge=1)
    consolidation_max_training_prompt_tokens: int = Field(default=128, ge=1)

    model_config = {"frozen": True}

    def model_post_init(self, __context) -> None:
        if self.consolidation_new_event_nodes >= self.consolidation_max_event_nodes:
            raise ValueError(
                "consolidation_new_event_nodes must be smaller than "
                "consolidation_max_event_nodes"
            )
        if self.qwen_rerank and self.qwen_feedback:
            raise ValueError("qwen_rerank and qwen_feedback are separate arms")
        if self.coverage_selection and self.mode not in {
            "memory",
            "causal_consolidation",
            "causal_graph",
        }:
            raise ValueError(
                "coverage_selection requires a packed memory or causal mode"
            )
        if self.allow_selected_scope_fixed_k_closure:
            if not self.coverage_selection:
                raise ValueError(
                    "selected-scope closure requires coverage_selection"
                )
            if not self.source_partition_routing:
                raise ValueError(
                    "selected-scope closure requires source_partition_routing"
                )
            if self.coverage_selector_backend not in {
                "qwen_prefix",
                "qwen_prefix_choice",
            }:
                raise ValueError(
                    "selected-scope closure currently requires a Qwen prefix "
                    "coverage backend"
                )
        if self.coverage_selection and self.coverage_selector_backend in {
            "qwen_prefix",
            "qwen_prefix_choice",
            "cross_encoder_qwen_prefix",
        }:
            if self.coverage_selector_attention_layer < 1:
                raise ValueError(
                    "coverage selector attention layer must be at least 1 so "
                    "the readout state is query-conditioned"
                )
            if self.coverage_selector_attention_layer >= self.coverage_selector_prefix_layers:
                raise ValueError(
                    "coverage selector attention layer must be inside its prefix"
                )
            if (
                self.coverage_selector_same_source_merge_similarity
                > self.coverage_selector_merge_similarity
            ):
                raise ValueError(
                    "same-source merge threshold cannot exceed cross-source threshold"
                )
        if (
            self.coverage_selection
            and self.coverage_selector_backend
            in {"cross_encoder", "cross_encoder_qwen_prefix"}
            and self.coverage_selector_cross_encoder_max_length
            > self.coverage_selector_max_workspace_tokens
        ):
            raise ValueError(
                "cross-encoder max length cannot exceed selector workspace"
            )
        if (
            self.coverage_selector_cross_encoder_semantic_rerank
            and self.coverage_selector_cross_encoder_score_only
        ):
            raise ValueError(
                "cross-encoder semantic rerank and score-only modes are "
                "mutually exclusive"
            )
        if (
            self.coverage_selection
            and self.coverage_selector_backend == "qwen_prefix_choice"
        ):
            if not all(
                (
                    self.coverage_selector_choice_model_id,
                    self.coverage_selector_choice_revision,
                    self.coverage_selector_choice_checkpoint_sha256,
                )
            ):
                raise ValueError(
                    "choice coverage requires exact model identity, revision, "
                    "and checkpoint SHA-256"
                )
            digest = self.coverage_selector_choice_checkpoint_sha256.casefold()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("choice checkpoint SHA-256 must be 64 hex digits")
            # The CLI loader requires single-token A/B labels. Both logits
            # come from one shared final prompt state, so no paired prompt or
            # appended label occupies the token workspace.
            if (
                self.coverage_selector_choice_max_prompt_tokens
                > self.coverage_selector_choice_max_workspace_tokens
            ):
                raise ValueError(
                    "choice workspace cannot hold one candidate prompt"
                )
        if self.coverage_selector_prefix_checkpoint_sha256:
            digest = self.coverage_selector_prefix_checkpoint_sha256.casefold()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "prefix checkpoint SHA-256 must be 64 hex digits"
                )
        if self.qwen_rerank or self.qwen_feedback:
            if not self.source_local_search:
                raise ValueError("Qwen attention requires source_local_search")
            if self.qwen_rerank_beam_per_group >= 8:
                raise ValueError(
                    "qwen_rerank_beam_per_group must be smaller than the "
                    "eight-candidate Qwen workspace"
                )
            if self.qwen_rerank_group_size > 8:
                raise ValueError(
                    "qwen_rerank_group_size cannot exceed the eight-candidate "
                    "Qwen workspace"
                )
            if self.qwen_rerank_attention_layer >= self.qwen_rerank_prefix_layers:
                raise ValueError(
                    "qwen_rerank_attention_layer must be inside the loaded prefix"
                )
            if (
                self.qwen_rerank_use_cav
                and self.qwen_rerank_cav_layer >= self.qwen_rerank_prefix_layers
            ):
                raise ValueError(
                    "qwen_rerank_cav_layer must be inside the loaded prefix"
                )
        if self.source_partition_routing and self.mode not in {
            "hybrid_graph",
            "causal_graph",
        }:
            raise ValueError(
                "source_partition_routing requires hybrid_graph or causal_graph"
            )
        if self.source_hsc_activation:
            if self.mode not in {"hybrid_graph", "causal_graph"}:
                raise ValueError("source_hsc_activation requires a graph mode")
            if self.source_hsc_chunk_slots > self.source_slots:
                raise ValueError("source_hsc_chunk_slots cannot exceed source_slots")
        if self.query_facet_retrieval:
            if self.mode not in {"hybrid_graph", "causal_graph"}:
                raise ValueError("query_facet_retrieval requires a graph mode")
            if self.query_facet_slots > self.source_slots:
                raise ValueError("query_facet_slots cannot exceed source_slots")
            if (
                self.source_hsc_activation
                and self.query_facet_slots + self.source_hsc_chunk_slots
                > self.source_slots
            ):
                raise ValueError(
                    "facet and HSC reserves cannot exceed source_slots"
                )
        if self.role_aware_retrieval and self.mode not in {
            "hybrid_graph",
            "causal_graph",
        }:
            raise ValueError("role_aware_retrieval requires a graph mode")
        if self.multi_fact_source_diversity and self.mode not in {
            "hybrid_graph",
            "causal_graph",
        }:
            raise ValueError("multi_fact_source_diversity requires a graph mode")
        if self.qwen_rerank:
            if self.mode not in {
                "hybrid_source",
                "hybrid_graph",
                "causal_graph",
            }:
                raise ValueError(
                    "qwen_rerank requires hybrid_source, hybrid_graph, or "
                    "causal_graph mode"
                )
            if self.qwen_rerank_slots > self.source_slots:
                raise ValueError("qwen_rerank_slots cannot exceed source_slots")
            if self.qwen_rerank_candidate_pool < self.source_slots:
                raise ValueError(
                    "qwen_rerank_candidate_pool cannot be smaller than source_slots"
                )
        if self.qwen_feedback:
            if self.mode not in {"hybrid_graph", "causal_graph"}:
                raise ValueError(
                    "qwen_feedback requires hybrid_graph or causal_graph mode"
                )
            if self.qwen_feedback_slots > self.source_slots:
                raise ValueError("qwen_feedback_slots cannot exceed source_slots")
            if self.qwen_feedback_seed_slots > self.qwen_feedback_candidate_pool:
                raise ValueError(
                    "qwen_feedback_seed_slots cannot exceed its candidate pool"
                )

    @property
    def effective_hybrid(self) -> bool:
        return self.mode == "hybrid" or self.hybrid

    @property
    def coverage_label(self) -> str:
        if not self.coverage_selection:
            return ""
        backend = self.coverage_selector_backend.replace("_", "-")
        if self.coverage_selector_backend == "qwen_prefix_choice":
            model_name = self.coverage_selector_choice_model_id.rsplit("/", 1)[-1]
            slug = "".join(
                character.casefold() if character.isalnum() else "-"
                for character in model_name
            ).strip("-")
            while "--" in slug:
                slug = slug.replace("--", "-")
            backend = f"qwen-prefix-choice-{slug}"
        semantic_mode = (
            "-score-only"
            if self.coverage_selector_backend
            in {"cross_encoder", "cross_encoder_qwen_prefix"}
            and self.coverage_selector_cross_encoder_score_only
            else "-companion-only"
            if self.coverage_selector_backend
            in {"cross_encoder", "cross_encoder_qwen_prefix"}
            and not self.coverage_selector_cross_encoder_semantic_rerank
            else ""
        )
        selected_closure = (
            "-selected-scope-closure"
            if self.allow_selected_scope_fixed_k_closure
            else ""
        )
        return f"-coverage-{backend}{semantic_mode}{selected_closure}"

    @property
    def label(self) -> str:
        """Short tag for filenames and run tables."""
        if self.mode == "span":
            levels = "-".join(str(x) for x in self.span_levels)
            return f"span{levels}x{self.k_per_level}"
        if self.mode == "source":
            return f"source{self.k_sources}"
        if self.mode == "anchored_source":
            return f"anchored-source-k{self.k}"
        if self.mode == "hybrid_source":
            activation = self.source_activation_k or self.k
            local = "-local" if self.source_local_search else ""
            qwen = f"-qwen{self.qwen_rerank_slots}" if self.qwen_rerank else ""
            return (
                f"hybrid-source-k{self.k}-s{self.source_slots}"
                f"-a{activation}-p{self.source_candidate_pool}{local}{qwen}"
            )
        if self.mode == "hybrid_graph":
            activation = self.source_activation_k or self.k
            local = "-local" if self.source_local_search else ""
            facets = (
                f"-facet{self.query_facet_slots}"
                if self.query_facet_retrieval
                else ""
            )
            roles = "-role" if self.role_aware_retrieval else ""
            diversity = "-diverse" if self.multi_fact_source_diversity else ""
            qwen = (
                f"-qwenfb{self.qwen_feedback_slots}"
                if self.qwen_feedback
                else f"-qwen{self.qwen_rerank_slots}"
                if self.qwen_rerank
                else ""
            )
            partitions = (
                f"-part{self.source_partition_slots}"
                if self.source_partition_routing
                else ""
            )
            return (
                f"hybrid-graph-k{self.k}-r{self.neighbor_radius}"
                f"-n{self.neighbor_slots}-{self.neighbor_direction}"
                f"-s{self.source_slots}-a{activation}"
                f"-p{self.source_candidate_pool}{partitions}{local}"
                f"{facets}{roles}{diversity}{qwen}"
            )
        if self.mode == "hybrid_neighbor":
            replacement = (
                f"-replace{self.neighbor_replacement_slots}"
                if self.neighbor_replacement_slots
                else ""
            )
            return (
                f"hybrid-neighbor-k{self.k}-r{self.neighbor_radius}"
                f"-s{self.neighbor_slots}{replacement}"
            )
        if self.mode == "memory":
            return f"memory{self.k_memories}{self.coverage_label}"
        if self.mode in {"causal_consolidation", "causal_graph"}:
            base = "causal-graph" if self.mode == "causal_graph" else "causal"
            local = "-local" if self.source_local_search else ""
            facets = (
                f"-facet{self.query_facet_slots}"
                if self.query_facet_retrieval
                else ""
            )
            roles = "-role" if self.role_aware_retrieval else ""
            diversity = "-diverse" if self.multi_fact_source_diversity else ""
            qwen = (
                f"-qwenfb{self.qwen_feedback_slots}"
                if self.qwen_feedback
                else f"-qwen{self.qwen_rerank_slots}"
                if self.qwen_rerank
                else ""
            )
            partitions = (
                f"-part{self.source_partition_slots}"
                if self.source_partition_routing
                else ""
            )
            return (
                f"{base}-k{self.k}-s{self.consolidation_chunk_slots}"
                f"-h{self.consolidation_hops}{partitions}{local}"
                f"{facets}{roles}{diversity}{qwen}{self.coverage_label}"
            )
        if self.effective_hybrid:
            return f"hybrid{self.alpha:g}"
        return "dense"


class EvalConfig(BaseModel):
    """Full configuration for one eval run."""

    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    judge_model: str = DEFAULT_JUDGE_MODEL
    responder_model: str = DEFAULT_RESPONDER_MODEL
    embedding_device: str | None = None
    conversation_dir: str = ""
    results_dir: str = "./eval_results"
    max_conversations: int | None = None
    recent_window: int = 4  # number of recent turns to include in context
    #: Accuracy-first long-chat gate. Judge accuracy is the headline metric;
    #: F1/EM and retrieval containment remain diagnostics.
    accuracy_target: float = Field(default=0.95, ge=0.0, le=1.0)
    #: A small smoke cannot certify a 95% target even if it happens to be
    #: perfect. Paid/public runs must grade at least this many questions.
    min_target_questions: int = Field(default=100, ge=1)
    #: Hard cap over the deterministic local prompt-token proxy sent to the
    #: responder. The proxy uses cl100k_base plus an explicit chat-framing
    #: reserve; it is not an exact provider-token count. Provider input usage
    #: is checked after the call whenever the provider reports it. ``None``
    #: preserves historical uncapped behavior; the CLI defaults to 8k.
    max_prompt_tokens: int | None = Field(default=None, ge=1)


class TurnResult(BaseModel):
    """Result of evaluating one user turn."""

    turn_index: int
    user_text: str
    actual_response: str
    generated_response: str
    retrieved_chunks: list[str]
    score: int  # 1-5
    judge_reasoning: str
    responder_usage: UsageStats = Field(default_factory=UsageStats)
    judge_usage: UsageStats = Field(default_factory=UsageStats)
    retrieval_s: float = 0.0  # time spent inside mc.search
    context_tokens: int = 0  # tiktoken count of the assembled responder prompt

    # Memory-mode instrumentation. Zero in dense/hybrid mode, where no memory
    # items are consulted. `memories_dropped` is the per-turn measurement
    # behind `08 - Analysis/01`'s header-budget finding — without it that
    # number stays an offline estimate rather than a run artifact.
    memory_items_packed: int = 0
    memories_dropped: int = 0
    heat_counts: dict[str, int] = Field(default_factory=dict)


class ConversationResult(BaseModel):
    """Eval results for one conversation."""

    filename: str
    num_turns: int
    turn_results: list[TurnResult]
    mean_score: float
    scores_by_position: list[float] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)


class EvalRunResult(BaseModel):
    """Results from one config run."""

    config: EvalConfig
    conversations: list[ConversationResult]
    aggregate_mean_score: float
    aggregate_recall_at_4: float  # fraction of scores >= 4
    run_timestamp: str
    usage: UsageStats = Field(default_factory=UsageStats)
    total_elapsed_s: float = 0.0
    mean_context_tokens: float = 0.0
    tokens_per_scored_turn: float = 0.0


class SweepReport(BaseModel):
    """Results across all parameter configurations."""

    runs: list[EvalRunResult]
    best_config: EvalConfig | None = None
    generated_at: str
