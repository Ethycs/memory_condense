"""Resumable provider-free 1M LongMemEval cumulative retrieval campaign.

The campaign deliberately reconstructs the original 1,039,203-token
development concatenation.  Retrieval is gold-blind and publishes its exact
S0--S3 prompts/evidence/receipts before a separate post-hoc scoring pass reads
the benchmark answers and evidence labels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memory_condense.domain.discourse import ClosurePolicy, identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    best_f1,
    contains_answer,
)
from memory_condense.eval.context_stress import (
    compose_context_stress_sample,
    transcript_tokens,
)
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.recall_guarded_cumulative import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    RecallGuardedCumulativeReceipt,
    RecallGuardedCumulativeRetrieval,
    retrieve_recall_guarded_cumulative_packet,
)
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    PreparedRecallGuardedCumulativeStore,
    build_recall_guarded_cumulative_store,
    open_recall_guarded_cumulative_store,
)
from memory_condense.eval.recall_guarded_cumulative_1m_source import (
    CURRENT_SOURCE_FORMAT,
    CURRENT_SOURCE_SCOPE,
    CURRENT_SOURCE_SELECTION_NAME,
    CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
    current_source_binding,
    prepare_current_source_store,
    source_treatment_identity,
    validate_current_source_receipt,
)
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample, load_benchmark
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPolicy,
    EpisodeRetrievalPolicy,
)


CAMPAIGN_FORMAT = "memory-condense-recall-guarded-cumulative-1m-campaign-v1"
QUESTION_FORMAT = "memory-condense-recall-guarded-cumulative-1m-query-v1"
RETRIEVAL_FORMAT = "memory-condense-recall-guarded-cumulative-1m-retrieval-v1"
SCORE_FORMAT = "memory-condense-recall-guarded-cumulative-1m-score-v1"

ORIGINAL_DATASET_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)
ORIGINAL_SPLIT_SHA256 = (
    "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
)
ORIGINAL_POLICY_SHA256 = (
    "5ea9352372414a34805d5dd5c406aaad7f457a56b8d978cc87cf7dbbc6b15c54"
)
ORIGINAL_SAMPLE_SHA256 = (
    "591b370952b362a21ca40491d813262f759b922cf1717815dedd91f2b61adc00"
)
ORIGINAL_TRANSCRIPT_TOKENS = 1_039_203
ORIGINAL_TURNS = 5_400
ORIGINAL_QUESTIONS = 10
ORIGINAL_SOURCE_CHUNKS = 7_930
ORIGINAL_SOURCE_DATABASE_SHA256 = (
    "36cf80c03347703cbe72e7a2bfed99c7b841154864a218d01c60eac28dc4c575"
)
ORIGINAL_SOURCE_INDEX_SHA256 = (
    "7952c357bd94bd1c7ab309aefeebe95e4365129e934f0e546a91947cf9f494a3"
)
ORIGINAL_SOURCE_CACHE_KEY = (
    "d5e87dc6c94909299d73dbc2b0bedf7bbb1d7e30153ee57e3b62bbf8cda92452"
)
ORIGINAL_BGE_IDENTITY = (
    "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181"
    "#a3d5c49f064ab58d7cf5bba1c2085918f529778e88535aca7de674c9094af0b7"
)
# The archived store used by the original 1M pilot recorded ingestion
# wall-clock instants in ``turns.created_at``.  Its dated synthetic boundary
# turns still carry the LongMemEval session dates in their source text.  Keep
# that behavior explicit: this campaign is a byte-matched comparison to the
# original store, not a claim that the archived column contains source time.
ORIGINAL_SOURCE_TIMESTAMP_SEMANTICS = (
    "archived_v3_ingestion_wall_clock_preserved_for_matched_baseline"
)
ORIGINAL_ORDERED_QUESTION_IDS = (
    "c4f10528",
    "6ade9755",
    "gpt4_d6585ce8",
    "bbf86515",
    "6e984301",
    "gpt4_f49edff3",
    "e01b8e2f",
    "gpt4_7abb270c",
    "2311e44b",
    "a2f3aa27",
)
STAGE_IDS = (
    "causal_graph_coverage_predecessor",
    "direct_episode_additions",
    "representative_episode_additions",
    "artifact_global_closure_additions",
)

DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-development-20260821"
)
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_POLICY = Path(
    "docs/10 - Research Log/data/longmemeval-qwen-choice-coverage-operational-development-v3.json"
)
DEFAULT_QWEN_PREFIX = Path(".cache/models/Qwen3-8B")
DEFAULT_QWEN_CHOICE = Path(".cache/models/Qwen3-0.6B")


class _UnboundCoverageSelector:
    """Construction-only sentinel replaced before the first retrieval."""

    strict = True
    requires_baseline_ranking = True
    requires_complete_frontier = True

    def select(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("coverage selector was not bound before retrieval")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: object) -> str:
    """Publish canonical JSON once and return its SHA-256."""

    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace another artifact: {path}")
    else:
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.exists():
        if sidecar.read_bytes() != sidecar_payload:
            raise FileExistsError(f"refusing to replace another digest: {sidecar}")
    else:
        sidecar.write_bytes(sidecar_payload)
    return digest


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise ValueError(f"artifact is not a canonical JSON object: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
        raise ValueError(f"artifact digest sidecar is missing or invalid: {path}")
    return payload, digest


def _require_file_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} is not the frozen original ({observed} != {expected})"
        )


def load_original_population(
    dataset_path: str | Path,
    split_path: str | Path,
) -> BenchmarkSample:
    """Reconstruct and certify the original development concatenation."""

    dataset = Path(dataset_path)
    split = Path(split_path)
    _require_file_digest(dataset, ORIGINAL_DATASET_SHA256, "LongMemEval dataset")
    _require_file_digest(split, ORIGINAL_SPLIT_SHA256, "locked split manifest")
    samples = load_benchmark(dataset, "longmemeval")
    development = select_locked_split(
        samples,
        dataset_path=dataset,
        manifest=load_split_manifest(split),
        split="development",
    )
    composed = compose_context_stress_sample(
        development,
        target_tokens=1_000_000,
        max_questions=ORIGINAL_QUESTIONS,
    )
    observed = {
        "transcript_tokens": transcript_tokens(composed),
        "turns": len(composed.turns),
        "questions": len(composed.questions),
        "ordered_question_ids": tuple(
            question.question_id for question in composed.questions
        ),
    }
    expected = {
        "transcript_tokens": ORIGINAL_TRANSCRIPT_TOKENS,
        "turns": ORIGINAL_TURNS,
        "questions": ORIGINAL_QUESTIONS,
        "ordered_question_ids": ORIGINAL_ORDERED_QUESTION_IDS,
    }
    if observed != expected:
        raise RuntimeError(
            f"original concatenated-memory population changed: {observed!r}"
        )
    return composed


def population_identity_payload(sample: BenchmarkSample) -> dict[str, object]:
    """Current-parser identity for the already certified original population."""

    source_ids = sample.turn_source_ids or [None] * len(sample.turns)
    timestamps = sample.turn_created_at or [None] * len(sample.turns)
    if len(source_ids) != len(sample.turns) or len(timestamps) != len(sample.turns):
        raise ValueError("population turn provenance is misaligned")
    corpus_sha256 = identity_sha256(
        [
            {
                "ordinal": ordinal,
                "role": role,
                "text_sha256": quote_sha256(text),
                "source_id": source_id,
                "created_at": (
                    None if created_at is None else created_at.isoformat()
                ),
            }
            for ordinal, ((role, text), source_id, created_at) in enumerate(
                zip(sample.turns, source_ids, timestamps, strict=True),
                1,
            )
        ]
    )
    return {
        "format": "memory-condense-original-1m-development-population-v1",
        "dataset_sha256": ORIGINAL_DATASET_SHA256,
        "split_manifest_sha256": ORIGINAL_SPLIT_SHA256,
        "split": "development",
        "construction": {
            "target_tokens": 1_000_000,
            "max_questions": ORIGINAL_QUESTIONS,
            "question_offset": 0,
        },
        # Gold-blind current-loader identity. Answers and labeled evidence
        # sources are deliberately absent from this projection.
        "current_corpus_sha256": corpus_sha256,
        "archived_compiled_sample_sha256": ORIGINAL_SAMPLE_SHA256,
        "transcript_tokens": transcript_tokens(sample),
        "turn_count": len(sample.turns),
        "question_count": len(sample.questions),
        "ordered_question_id_sha256s": [
            identity_sha256({"question_id": question.question_id})
            for question in sample.questions
        ],
        "ordered_question_probe_sha256s": [
            identity_sha256(
                {
                    "question_id": question.question_id,
                    "question_sha256": quote_sha256(question.question),
                    "dated_question_sha256": quote_sha256(
                        question.dated_question
                    ),
                }
            )
            for question in sample.questions
        ],
    }


def population_identity_sha256(sample: BenchmarkSample) -> str:
    return identity_sha256(population_identity_payload(sample))


def load_frozen_config(policy_path: str | Path, *, device: str) -> EvalConfig:
    """Load only the exact frozen-v3 retrieval controls from its manifest."""

    policy_file = Path(policy_path)
    _require_file_digest(policy_file, ORIGINAL_POLICY_SHA256, "retrieval policy")
    payload = json.loads(policy_file.read_bytes())
    retrieval_body = dict(payload["retrieval"])
    min_tokens = int(retrieval_body.pop("chunker_min_tokens"))
    max_tokens = int(retrieval_body.pop("chunker_max_tokens"))
    max_prompt_tokens = int(retrieval_body.pop("max_prompt_tokens"))
    retrieval = RetrievalConfig(**retrieval_body)
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=min_tokens, max_tokens=max_tokens),
        retrieval=retrieval,
        embedding_device=device,
        max_prompt_tokens=max_prompt_tokens,
        min_target_questions=ORIGINAL_QUESTIONS,
    )


def verify_original_source_store(
    source_dir: str | Path,
) -> Path:
    """Bind the copied compiled store to the exact original sample receipt."""

    source = Path(source_dir)
    manifest_path = source / "compiled-store.json"
    database_path = source / "memory.db"
    index_path = source / "hnsw_index.bin"
    manifest = json.loads(manifest_path.read_bytes())
    required = {
        "cache_revision": 3,
        "cache_key": ORIGINAL_SOURCE_CACHE_KEY,
        "sample_sha256": ORIGINAL_SAMPLE_SHA256,
        "turn_count": ORIGINAL_TURNS,
        "chunk_count": ORIGINAL_SOURCE_CHUNKS,
        "chunker_min_tokens": 120,
        "chunker_max_tokens": 250,
        "embedding_model": ORIGINAL_BGE_IDENTITY,
        "database_sha256": ORIGINAL_SOURCE_DATABASE_SHA256,
        "index_sha256": ORIGINAL_SOURCE_INDEX_SHA256,
    }
    if any(manifest.get(name) != value for name, value in required.items()):
        raise ValueError("compiled source manifest is not the original v3 store")
    if file_sha256(database_path) != ORIGINAL_SOURCE_DATABASE_SHA256:
        raise RuntimeError("compiled source database differs from its receipt")
    if file_sha256(index_path) != ORIGINAL_SOURCE_INDEX_SHA256:
        raise RuntimeError("compiled source index differs from its receipt")
    return database_path


def archived_source_provenance_payload() -> dict[str, object]:
    """Describe the frozen legacy source without claiming it is current-safe."""

    return {
        "compiled_sample_sha256": ORIGINAL_SAMPLE_SHA256,
        "cache_key": ORIGINAL_SOURCE_CACHE_KEY,
        "database_sha256": ORIGINAL_SOURCE_DATABASE_SHA256,
        "index_sha256": ORIGINAL_SOURCE_INDEX_SHA256,
        "chunk_count": ORIGINAL_SOURCE_CHUNKS,
        "embedding_model": ORIGINAL_BGE_IDENTITY,
        "timestamp_semantics": ORIGINAL_SOURCE_TIMESTAMP_SEMANTICS,
        "used_for_current_build": False,
    }


def _held_out_queries(sample: BenchmarkSample) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            text
            for question in sample.questions
            for text in (question.question, question.dated_question)
        )
    )


def prepare_store(
    *,
    sample: BenchmarkSample,
    config: EvalConfig,
    source_dir: Path,
    combined_dir: Path,
    qwen_prefix_model_dir: Path,
) -> tuple[
    PreparedRecallGuardedCumulativeStore,
    Any,
    str,
    dict[str, Any],
    str,
]:
    """Build once or verify/reopen, retaining a frozen held-out query batch."""

    source_config, binding = current_source_binding(
        config,
        qwen_model_dir=qwen_prefix_model_dir,
    )
    embedder = binding.embedder
    queries = _held_out_queries(sample)
    sentinel = _UnboundCoverageSelector()
    try:
        source_database, source_receipt, source_mode = (
            prepare_current_source_store(
                sample=sample,
                config=source_config,
                treatment_identity=source_treatment_identity(
                    sample,
                    dataset_sha256=ORIGINAL_DATASET_SHA256,
                    split_manifest_sha256=ORIGINAL_SPLIT_SHA256,
                    sanitized_projection_sha256=(
                        population_identity_sha256(sample)
                    ),
                ),
                binding=binding,
                source_root=source_dir,
                selection_path=(
                    source_dir.parent / CURRENT_SOURCE_SELECTION_NAME
                ),
            )
        )
        if combined_dir.exists():
            prepared = open_recall_guarded_cumulative_store(
                combined_dir,
                config=config,
                embedder=embedder,
                held_out_queries=queries,
                coverage_selector=sentinel,
            )
            mode = "verified_cache_hit"
        else:
            prepared = build_recall_guarded_cumulative_store(
                source_database,
                combined_dir,
                config=config,
                embedder=embedder,
                held_out_queries=queries,
                compilation_policy=DiffuseCompilationPolicy(
                    boundary_mode="fixed_interval"
                ),
                coverage_selector=sentinel,
                embedding_identity={
                    "backend": "sentence-transformers.encode-v1",
                    "model_id": "BAAI/bge-m3",
                    "dimension": 1024,
                },
            )
            mode = "fresh_atomic_build"
        if (
            prepared.receipt.source_database_sha256
            != source_receipt["database_sha256"]
            or prepared.receipt.turn_count != source_receipt["turn_count"]
            or prepared.receipt.chunk_count != source_receipt["chunk_count"]
        ):
            prepared.close()
            raise RuntimeError(
                "combined store does not bind the selected current source"
            )
    except BaseException:
        embedder.close()
        raise
    return prepared, embedder, mode, source_receipt, source_mode


def _load_shared_qwen(config: EvalConfig, prefix_dir: Path, choice_dir: Path):
    """Load one 8B prefix linker shared by S0 coverage and S2 discovery."""

    retrieval = config.retrieval
    if not prefix_dir.is_dir() or not choice_dir.is_dir():
        raise FileNotFoundError("one or both local Qwen checkpoint directories are missing")
    from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
    from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
    from memory_condense.search.selectors.causal_choice_scorer import CausalChoiceScorer
    from memory_condense.search.selectors.prefix_selector import QwenPrefixCoverageSelector

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


def _episode_policy(artifact_id: str) -> EpisodeRetrievalPolicy:
    return EpisodeRetrievalPolicy(
        artifact_id=artifact_id,
        max_anchor_episodes=96,
        previous_episodes=1,
        next_episodes=1,
        max_episode_seeds=256,
        max_direct_fallbacks=96,
    )


def _representative_policy(artifact_id: str) -> EpisodeRepresentativeRetrievalPolicy:
    return EpisodeRepresentativeRetrievalPolicy(
        artifact_id=artifact_id,
        max_input_sources=64,
        max_source_groups=64,
        max_episodes_per_source=64,
        max_total_episodes=256,
        max_representatives_per_episode=2,
        group_size=8,
        beam_per_group=2,
        top_k=8,
        representative_tokens=96,
        query_tokens=96,
        score_mode="qk_ov",
    )


def _closure_policy() -> ClosurePolicy:
    return ClosurePolicy(
        max_hops=3,
        max_units=1024,
        max_relations=2048,
        max_degree=32,
        max_episode_neighbors=2,
        max_frontier=1024,
        max_bundles=256,
        beam_width=128,
        min_relation_confidence=0.5,
    )


def _question_part(
    result: RecallGuardedCumulativeRetrieval,
    *,
    question: BenchmarkQuestion,
    ordinal: int,
    population_sha: str,
    store_receipt_sha256: str,
    retrieval_implementation_sha256: str,
) -> dict[str, object]:
    messages = result.provider_messages_by_stage()
    evidence: list[dict[str, object]] = [
        {
            "evidence_id": stage_id,
            "source_id": excerpt.source_id,
            "text": excerpt.text,
        }
        for stage_id, excerpt in zip(
            result.ladder.stages[0].selected_evidence_ids,
            result.predecessor.excerpts,
            strict=True,
        )
    ]
    stages: list[dict[str, object]] = []
    for index, stage in enumerate(result.ladder.stages):
        if index:
            packet = result.addition_packets[index - 1]
            if packet is not None:
                evidence.extend(
                    {
                        "evidence_id": evidence_id,
                        "source_id": atom.span.source_id,
                        "text": atom.text,
                    }
                    for evidence_id, atom in zip(
                        stage.added_evidence_ids,
                        packet.atoms,
                        strict=True,
                    )
                )
        if tuple(item["evidence_id"] for item in evidence) != (
            stage.selected_evidence_ids
        ):
            raise RuntimeError("stage evidence rows changed their sealed coordinates")
        stages.append(
            {
                "stage_id": stage.stage_id,
                "stage_receipt": asdict(stage),
                "provider_messages": messages[stage.stage_id],
                "evidence": [dict(item) for item in evidence],
            }
        )
    return {
        "format": QUESTION_FORMAT,
        "population_identity_sha256": population_sha,
        "ordinal": ordinal,
        "question_id": question.question_id,
        "question_sha256": quote_sha256(question.question),
        "dated_question_sha256": quote_sha256(question.dated_question),
        "combined_store_receipt_sha256": store_receipt_sha256,
        "retrieval_implementation_sha256": retrieval_implementation_sha256,
        "retrieval_receipt": asdict(result.receipt),
        "predecessor_receipt": asdict(result.predecessor.receipt),
        "stage_ids": list(STAGE_IDS),
        "stages": stages,
        "provider_calls": 0,
    }


def _validate_question_part(
    part: Mapping[str, Any],
    *,
    question: BenchmarkQuestion,
    ordinal: int,
    population_sha: str,
    store_receipt_sha256: str,
    retrieval_implementation_sha256: str,
) -> None:
    if (
        part.get("format") != QUESTION_FORMAT
        or part.get("population_identity_sha256") != population_sha
        or part.get("ordinal") != ordinal
        or part.get("question_id") != question.question_id
        or part.get("question_sha256") != quote_sha256(question.question)
        or part.get("dated_question_sha256") != quote_sha256(question.dated_question)
        or part.get("combined_store_receipt_sha256") != store_receipt_sha256
        or part.get("retrieval_implementation_sha256")
        != retrieval_implementation_sha256
        or tuple(part.get("stage_ids", ())) != STAGE_IDS
        or part.get("provider_calls") != 0
    ):
        raise ValueError("retrieval question part belongs to another campaign")
    stages = part.get("stages")
    if not isinstance(stages, list) or tuple(
        item.get("stage_id") for item in stages if isinstance(item, Mapping)
    ) != STAGE_IDS:
        raise ValueError("retrieval question part changed its cumulative stages")
    typed_stages: list[CumulativeRetrievalStageReceipt] = []
    for stage in stages:
        receipt = stage.get("stage_receipt", {})
        evidence = stage.get("evidence", ())
        typed = CumulativeRetrievalStageReceipt(**receipt)
        typed_stages.append(typed)
        if tuple(item.get("evidence_id") for item in evidence) != tuple(
            typed.selected_evidence_ids
        ):
            raise ValueError("retrieval question part changed stage evidence IDs")
        if identity_sha256(stage.get("provider_messages")) != (
            typed.prompt_messages_sha256
        ):
            raise ValueError("retrieval question part changed a provider prompt")
    ladder = CumulativeRetrievalLadder(stages=tuple(typed_stages))
    final_receipt = RecallGuardedCumulativeReceipt(
        **part.get("retrieval_receipt", {})
    )
    predecessor = CausalCoveragePredecessorReceipt(
        **part.get("predecessor_receipt", {})
    )
    if (
        final_receipt.ladder_receipt_sha256 != ladder.receipt_sha256
        or final_receipt.predecessor_receipt_sha256
        != predecessor.receipt_sha256
        or final_receipt.prompt_messages_sha256
        != typed_stages[-1].prompt_messages_sha256
    ):
        raise ValueError("retrieval question receipts no longer cross-bind")


def run_gold_blind_retrieval(
    *,
    prepared: PreparedRecallGuardedCumulativeStore,
    sample: BenchmarkSample,
    config: EvalConfig,
    selector: Any,
    representative_linker: Any,
    output_root: Path,
    source_store_receipt: Mapping[str, Any],
    source_embedding_device: str,
) -> tuple[dict[str, Any], str]:
    """Run missing questions, atomically checkpoint each, then publish a merge."""

    selected_source = validate_current_source_receipt(
        source_store_receipt,
        sample=sample,
        expected_device=source_embedding_device,
    )
    if prepared.receipt.source_database_sha256 != selected_source[
        "database_sha256"
    ]:
        raise RuntimeError("retrieval store changed its selected current source")
    condenser = prepared.condenser
    condenser.set_context_candidate_selector(selector)
    artifact_id = prepared.compilation.artifact.artifact_id
    population_sha = population_identity_sha256(sample)
    retrieval_implementation = implementation_sha256()
    parts_dir = output_root / "retrieval-parts"
    part_rows: list[dict[str, Any]] = []
    part_hashes: list[str] = []
    for ordinal, question in enumerate(sample.questions):
        path = parts_dir / f"q{ordinal:03d}.json"
        if path.exists():
            part, digest = _read_canonical_json(path)
            _validate_question_part(
                part,
                question=question,
                ordinal=ordinal,
                population_sha=population_sha,
                store_receipt_sha256=prepared.receipt.receipt_sha256,
                retrieval_implementation_sha256=retrieval_implementation,
            )
            print(
                f"Question {ordinal + 1}/{len(sample.questions)} "
                f"{question.question_id}: verified checkpoint hit",
                flush=True,
            )
        else:
            print(
                f"Question {ordinal + 1}/{len(sample.questions)} "
                f"{question.question_id}: running S0-S3",
                flush=True,
            )
            started = time.perf_counter()
            result = retrieve_recall_guarded_cumulative_packet(
                condenser,
                query=question.question,
                prompt_question=question.dated_question,
                retrieval=config.retrieval,
                artifact_id=artifact_id,
                max_context_tokens=7000,
                max_prompt_tokens=int(config.max_prompt_tokens or 8000),
                responder_output_token_reserve=256,
                episode_policy=_episode_policy(artifact_id),
                representative_linker=representative_linker,
                representative_policy=_representative_policy(artifact_id),
                source_router_max_sources=64,
                source_router_rrf_constant=60,
                closure_policy=_closure_policy(),
                require_certified_coverage_runtime=True,
                require_owned_representative_runtime=True,
            )
            part = _question_part(
                result,
                question=question,
                ordinal=ordinal,
                population_sha=population_sha,
                store_receipt_sha256=prepared.receipt.receipt_sha256,
                retrieval_implementation_sha256=retrieval_implementation,
            )
            part["elapsed_seconds"] = time.perf_counter() - started
            digest = _atomic_write_json(path, part)
            print(
                "  published; statuses="
                + ",".join(
                    stage["stage_receipt"]["admission_status"]
                    for stage in part["stages"]
                )
                + f"; elapsed={part['elapsed_seconds']:.1f}s",
                flush=True,
            )
        part_rows.append(part)
        part_hashes.append(digest)
    retrieval = {
        "format": RETRIEVAL_FORMAT,
        "campaign_format": CAMPAIGN_FORMAT,
        "archived_compiled_sample_sha256": ORIGINAL_SAMPLE_SHA256,
        "archived_source_provenance": archived_source_provenance_payload(),
        "source_timestamp_semantics": CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "source_store_receipt": selected_source,
        "population_identity": population_identity_payload(sample),
        "population_identity_sha256": population_identity_sha256(sample),
        "transcript_tokens": transcript_tokens(sample),
        "turn_count": len(sample.turns),
        "question_count": len(sample.questions),
        "stage_ids": list(STAGE_IDS),
        "retrieval_policy_sha256": identity_sha256(
            config.retrieval.model_dump(mode="json")
        ),
        "retrieval_implementation_sha256": retrieval_implementation,
        "combined_store_receipt": asdict(prepared.receipt),
        "compilation_receipt_sha256": prepared.compilation.receipt_sha256,
        "question_part_sha256s": part_hashes,
        "questions": part_rows,
        "provider_calls": 0,
        "gold_fields_present": False,
    }
    if implementation_sha256() != retrieval_implementation:
        raise RuntimeError("retrieval implementation changed during the run")
    digest = _atomic_write_json(output_root / "retrieval.json", retrieval)
    print(
        f"Gold-blind retrieval published: {output_root / 'retrieval.json'} "
        f"({digest})",
        flush=True,
    )
    return retrieval, digest


def _score_stage(
    stage: Mapping[str, Any],
    *,
    question: BenchmarkQuestion,
) -> dict[str, object]:
    evidence = list(stage["evidence"])
    texts = [str(item["text"]) for item in evidence]
    retrieved_sources = tuple(
        dict.fromkeys(
            str(item["source_id"])
            for item in evidence
            if item.get("source_id") is not None
        )
    )
    expected_sources = tuple(dict.fromkeys(question.evidence_sources))
    expected_set = set(expected_sources)
    retrieved_set = set(retrieved_sources)
    recall = (
        None
        if not expected_set
        else len(expected_set & retrieved_set) / len(expected_set)
    )
    components = answer_value_component_coverage(
        question.answer,
        len(expected_sources),
        texts,
    )
    receipt = stage["stage_receipt"]
    return {
        "stage_id": stage["stage_id"],
        "stage_receipt_sha256": receipt["receipt_sha256"],
        "answer_present": contains_answer(texts, question.answer),
        "best_evidence_f1": best_f1(texts, question.answer),
        "expected_source_ids": list(expected_sources),
        "retrieved_source_ids": list(retrieved_sources),
        "evidence_source_recall": recall,
        "any_evidence_source": (
            None if recall is None else bool(expected_set & retrieved_set)
        ),
        "all_evidence_sources": None if recall is None else recall == 1.0,
        "answer_value_components_expected": (
            None if components is None else components.expected
        ),
        "answer_value_components_found": (
            None if components is None else components.found
        ),
        "answer_value_component_recall": (
            None if components is None else components.recall
        ),
        "all_answer_value_components": (
            None if components is None else components.all_components
        ),
        "answer_value_component_hit_mask": (
            [] if components is None else list(components.hit_mask)
        ),
        "answer_value_metric_kind": (
            "" if components is None else components.metric_kind
        ),
        "context_token_proxy": receipt["context_token_proxy"],
        "prompt_token_proxy": receipt["prompt_token_proxy"],
        "admission_status": receipt["admission_status"],
    }


def score_published_retrieval(
    *,
    sample: BenchmarkSample,
    retrieval_path: Path,
    output_path: Path,
    source_embedding_device: str,
) -> tuple[dict[str, Any], str]:
    """Post-hoc score an already durable gold-blind retrieval artifact."""

    retrieval, retrieval_sha = _read_canonical_json(retrieval_path)
    retrieval_implementation = str(
        retrieval.get("retrieval_implementation_sha256", "")
    )
    if len(retrieval_implementation) != 64 or any(
        character not in "0123456789abcdef"
        for character in retrieval_implementation
    ):
        raise ValueError("retrieval artifact has no valid implementation digest")
    source_receipt = validate_current_source_receipt(
        retrieval.get("source_store_receipt"),
        sample=sample,
        expected_device=source_embedding_device,
    )
    if (
        retrieval.get("format") != RETRIEVAL_FORMAT
        or retrieval.get("archived_compiled_sample_sha256")
        != ORIGINAL_SAMPLE_SHA256
        or retrieval.get("archived_source_provenance")
        != archived_source_provenance_payload()
        or retrieval.get("source_timestamp_semantics")
        != CURRENT_SOURCE_TIMESTAMP_SEMANTICS
        or retrieval.get("population_identity")
        != population_identity_payload(sample)
        or retrieval.get("population_identity_sha256")
        != population_identity_sha256(sample)
        or retrieval.get("question_count") != len(sample.questions)
        or tuple(retrieval.get("stage_ids", ())) != STAGE_IDS
        or retrieval.get("provider_calls") != 0
        or retrieval.get("gold_fields_present") is not False
    ):
        raise ValueError("retrieval artifact belongs to another population or route")
    ordered_parts = list(retrieval["questions"])
    if [item.get("question_id") for item in ordered_parts] != [
        question.question_id for question in sample.questions
    ]:
        raise ValueError("retrieval artifact changed ordered question membership")
    observed_part_hashes = [
        hashlib.sha256(_canonical_json_bytes(item)).hexdigest()
        for item in ordered_parts
    ]
    if observed_part_hashes != retrieval.get("question_part_sha256s"):
        raise ValueError("retrieval artifact changed its embedded question parts")
    store_receipt = retrieval.get("combined_store_receipt")
    if not isinstance(store_receipt, Mapping):
        raise ValueError("retrieval artifact omitted its combined-store receipt")
    if store_receipt.get("source_database_sha256") != source_receipt[
        "database_sha256"
    ]:
        raise ValueError("retrieval artifact changed its selected source binding")
    store_receipt_sha256 = str(store_receipt.get("receipt_sha256", ""))
    population_sha = population_identity_sha256(sample)
    for ordinal, (question, part) in enumerate(
        zip(sample.questions, ordered_parts, strict=True)
    ):
        _validate_question_part(
            part,
            question=question,
            ordinal=ordinal,
            population_sha=population_sha,
            store_receipt_sha256=store_receipt_sha256,
            retrieval_implementation_sha256=retrieval_implementation,
        )
    rows_by_id = {item["question_id"]: item for item in ordered_parts}
    question_rows: list[dict[str, object]] = []
    for question in sample.questions:
        part = rows_by_id.get(question.question_id)
        if part is None:
            raise ValueError("retrieval artifact omitted a benchmark question")
        stages = [
            _score_stage(stage, question=question) for stage in part["stages"]
        ]
        question_rows.append(
            {
                "question_id": question.question_id,
                "category": question.category,
                "retrieval_receipt_sha256": part["retrieval_receipt"][
                    "receipt_sha256"
                ],
                "stages": stages,
            }
        )
    aggregates: list[dict[str, object]] = []
    for index, stage_id in enumerate(STAGE_IDS):
        stage_rows = [item["stages"][index] for item in question_rows]
        recalls = [
            float(item["evidence_source_recall"])
            for item in stage_rows
            if item["evidence_source_recall"] is not None
        ]
        component_recalls = [
            float(item["answer_value_component_recall"])
            for item in stage_rows
            if item["answer_value_component_recall"] is not None
        ]
        aggregates.append(
            {
                "stage_id": stage_id,
                "questions": len(stage_rows),
                "literal_answer_hits": sum(
                    bool(item["answer_present"]) for item in stage_rows
                ),
                "mean_best_evidence_f1": sum(
                    float(item["best_evidence_f1"]) for item in stage_rows
                )
                / len(stage_rows),
                "mean_evidence_source_recall": (
                    None if not recalls else sum(recalls) / len(recalls)
                ),
                "all_evidence_source_hits": sum(
                    item["all_evidence_sources"] is True for item in stage_rows
                ),
                "mean_answer_value_component_recall": (
                    None
                    if not component_recalls
                    else sum(component_recalls) / len(component_recalls)
                ),
                "mean_context_token_proxy": sum(
                    int(item["context_token_proxy"]) for item in stage_rows
                )
                / len(stage_rows),
                "max_context_token_proxy": max(
                    int(item["context_token_proxy"]) for item in stage_rows
                ),
                "max_prompt_token_proxy": max(
                    int(item["prompt_token_proxy"]) for item in stage_rows
                ),
                "hard_context_cap_compliant": all(
                    int(item["context_token_proxy"]) <= 7000
                    for item in stage_rows
                ),
                "hard_prompt_cap_compliant": all(
                    int(item["prompt_token_proxy"]) <= 8000
                    for item in stage_rows
                ),
            }
        )
    scores = {
        "format": SCORE_FORMAT,
        "campaign_format": CAMPAIGN_FORMAT,
        "status": "provider_free_retrieval_metrics_not_answer_accuracy",
        "retrieval_artifact_sha256": retrieval_sha,
        "archived_compiled_sample_sha256": ORIGINAL_SAMPLE_SHA256,
        "archived_source_provenance": archived_source_provenance_payload(),
        "source_timestamp_semantics": CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "source_store_receipt": source_receipt,
        "population_identity": population_identity_payload(sample),
        "population_identity_sha256": population_identity_sha256(sample),
        "transcript_tokens": transcript_tokens(sample),
        "turn_count": len(sample.turns),
        "question_count": len(sample.questions),
        "stage_ids": list(STAGE_IDS),
        "retrieval_implementation_sha256": retrieval_implementation,
        "aggregates": aggregates,
        "questions": question_rows,
        "responder_calls": 0,
        "judge_calls": 0,
    }
    digest = _atomic_write_json(output_path, scores)
    if file_sha256(retrieval_path) != retrieval_sha:
        raise RuntimeError("retrieval artifact changed during post-hoc scoring")
    print(f"Post-hoc scores published: {output_path} ({digest})", flush=True)
    return scores, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the provider-free S0-S3 cumulative route on the exact original "
            "1,039,203-token LongMemEval development concatenation"
        )
    )
    parser.add_argument(
        "--phase",
        choices=("all", "source", "build", "retrieve", "score"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--qwen-prefix-model-dir", type=Path, default=DEFAULT_QWEN_PREFIX)
    parser.add_argument("--qwen-choice-model-dir", type=Path, default=DEFAULT_QWEN_CHOICE)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sample = load_original_population(args.dataset, args.split_manifest)
    config = load_frozen_config(args.policy_manifest, device=args.device)
    retrieval_path = root / "retrieval.json"
    score_path = root / "scores.json"

    if args.phase == "score":
        score_published_retrieval(
            sample=sample,
            retrieval_path=retrieval_path,
            output_path=score_path,
            source_embedding_device=args.device,
        )
        return 0
    if retrieval_path.exists() and args.phase in {"all", "retrieve"}:
        score_published_retrieval(
            sample=sample,
            retrieval_path=retrieval_path,
            output_path=score_path,
            source_embedding_device=args.device,
        )
        return 0

    if args.phase == "source":
        source_config, binding = current_source_binding(
            config,
            qwen_model_dir=args.qwen_prefix_model_dir,
        )
        try:
            _database, source_receipt, source_mode = (
                prepare_current_source_store(
                    sample=sample,
                    config=source_config,
                    treatment_identity=source_treatment_identity(
                        sample,
                        dataset_sha256=ORIGINAL_DATASET_SHA256,
                        split_manifest_sha256=ORIGINAL_SPLIT_SHA256,
                        sanitized_projection_sha256=(
                            population_identity_sha256(sample)
                        ),
                    ),
                    binding=binding,
                    source_root=root / "source-current",
                    selection_path=root / CURRENT_SOURCE_SELECTION_NAME,
                )
            )
            print(
                f"Current exact-span source: {source_mode}; receipt "
                f"{source_receipt['receipt_sha256']}",
                flush=True,
            )
            return 0
        finally:
            binding.embedder.close()

    prepared, embedder, build_mode, source_receipt, source_mode = prepare_store(
        sample=sample,
        config=config,
        source_dir=root / "source-current",
        combined_dir=root / "combined-store",
        qwen_prefix_model_dir=args.qwen_prefix_model_dir,
    )
    try:
        print(
            f"Current exact-span source: {source_mode}; receipt "
            f"{source_receipt['receipt_sha256']}\n"
            f"Combined store: {build_mode}; receipt "
            f"{prepared.receipt.receipt_sha256}",
            flush=True,
        )
        if args.phase == "build":
            return 0
        # All query vectors now live in the FrozenQueryEmbedder owned by the
        # read-only condenser.  Releasing BGE makes room for the two local
        # provider-free Qwen runtimes.
        embedder.close()
        selector, representative_linker = _load_shared_qwen(
            config,
            args.qwen_prefix_model_dir.resolve(),
            args.qwen_choice_model_dir.resolve(),
        )
        try:
            run_gold_blind_retrieval(
                prepared=prepared,
                sample=sample,
                config=config,
                selector=selector,
                representative_linker=representative_linker,
                output_root=root,
                source_store_receipt=source_receipt,
                source_embedding_device=args.device,
            )
        finally:
            selector.close()
        del representative_linker, selector
        gc.collect()
        score_published_retrieval(
            sample=sample,
            retrieval_path=retrieval_path,
            output_path=score_path,
            source_embedding_device=args.device,
        )
        return 0
    finally:
        prepared.close()
        close = getattr(embedder, "close", None)
        if callable(close):
            close()


__all__ = [
    "CAMPAIGN_FORMAT",
    "ORIGINAL_SAMPLE_SHA256",
    "ORIGINAL_TRANSCRIPT_TOKENS",
    "QUESTION_FORMAT",
    "RETRIEVAL_FORMAT",
    "SCORE_FORMAT",
    "STAGE_IDS",
    "load_frozen_config",
    "load_original_population",
    "main",
    "population_identity_payload",
    "population_identity_sha256",
    "run_gold_blind_retrieval",
    "score_published_retrieval",
    "verify_original_source_store",
]
