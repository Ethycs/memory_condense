"""Shared orchestration helpers for the eval mode workflows.

Each mode module receives the executable facade (``eval.__main__``) as its
``runtime`` argument so the evaluation safety tests can keep patching one
namespace.  These helpers take that same ``runtime`` object and route every
patch-sensitive call through it.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreparedSamples:
    """Benchmark samples after locked split, shard offset, and stress."""

    samples: list = field(default_factory=list)
    #: Whether a context-stress memory was composed from the samples.
    stress_composed: bool = False
    #: Question shard to apply after the immutable stress store is opened.
    #: Zero/None when the shard was already applied to the composed sample.
    stress_question_offset: int = 0
    stress_question_count: int | None = None
    #: Actual transcript tokens of the composed stress memory (0 without one).
    stress_tokens: int = 0


def prepare_samples(
    args: argparse.Namespace,
    dataset_path,
    *,
    runtime,
    verbose: bool = True,
    stress: bool = False,
    shard_stress_questions: bool = False,
) -> PreparedSamples | None:
    """Load -> locked split -> sample shard -> optional stress composition.

    Returns ``None`` when the dataset parsed to zero samples so each mode can
    keep its own empty-input contract.  When ``stress`` is requested and
    ``--stress-context-tokens`` is set, the stress memory is always composed
    over the full canonical question pool (``max(10, offset + count)``): the
    composed transcript -- and therefore the immutable causal-store cache
    identity -- never depends on which question shard this process asks.
    Callers that can shard questions after the store is opened (answer
    recall) receive the shard as ``stress_question_offset``/``count``;
    callers whose downstream runner asks every sample question (benchmark,
    blind cache preparation) pass ``shard_stress_questions=True`` and get the
    shard applied to the composed sample's question list instead.
    """

    samples = runtime.load_benchmark(dataset_path, args.benchmark_format)
    if not samples:
        return None
    samples = runtime._apply_sample_offset(
        args,
        runtime._apply_locked_split(args, samples, verbose=verbose),
        verbose=verbose,
    )

    stress_target = (
        getattr(args, "stress_context_tokens", None) if stress else None
    )
    if stress_target is None:
        return PreparedSamples(samples=samples)

    from memory_condense.eval.context_stress import (
        compose_context_stress_sample,
        transcript_tokens,
    )

    question_offset = int(getattr(args, "stress_question_offset", 0))
    question_count = int(getattr(args, "stress_questions", 10))
    if question_offset < 0:
        raise ValueError("--stress-question-offset must be non-negative")
    if question_count < 1:
        raise ValueError("--stress-questions must be positive")
    # Keep the canonical ten-question stress sample as the causal-store
    # cache identity. Question sharding happens only after that immutable
    # store is opened; held-out questions do not change its learned graph.
    question_pool = max(10, question_offset + question_count)
    composed = compose_context_stress_sample(
        samples,
        target_tokens=stress_target,
        max_questions=question_pool,
    )
    if shard_stress_questions:
        questions = composed.questions[
            question_offset : question_offset + question_count
        ]
        if not questions:
            raise ValueError(
                "question_offset is outside the available stress questions"
            )
        composed = composed.model_copy(update={"questions": questions})
        question_offset = 0
        question_count = None
    actual_tokens = transcript_tokens(composed)
    if verbose:
        print(
            f"Context stress memory: {actual_tokens:,} tokens, "
            f"{len(composed.turns):,} turns, "
            f"{len(composed.questions)} questions"
        )
    return PreparedSamples(
        samples=[composed],
        stress_composed=True,
        stress_question_offset=question_offset,
        stress_question_count=question_count,
        stress_tokens=actual_tokens,
    )


@contextlib.contextmanager
def transient_runtime_controls(args: argparse.Namespace, config, *, runtime):
    """Load the optional reranker and coverage selector; always release both.

    Loading happens inside the guarded block, so a selector that fails to
    load still releases an already-loaded reranker.
    """

    reranker = None
    selector = None
    try:
        reranker = runtime._load_candidate_reranker(args, config)
        selector = runtime._load_coverage_selector(args, config)
        yield reranker, selector
    finally:
        if reranker is not None:
            reranker.close()
        if selector is not None:
            selector.close()


@dataclass(frozen=True)
class RunProvenance:
    """One-way hashes binding a run to its dataset, code, and frozen policy."""

    dataset_sha256: str
    split_manifest_sha256: str
    implementation_sha256: str
    environment_lock_sha256: str
    policy_manifest_sha256: str
    evaluation_protocol: dict[str, object]


def run_provenance(
    args: argparse.Namespace,
    dataset_path,
    config,
    *,
    runtime,
    prepare_only: bool = False,
) -> RunProvenance:
    """Hash the dataset/split/code/environment and verify the frozen policy."""

    dataset_hash = runtime.file_sha256(dataset_path)
    implementation_hash = runtime.implementation_sha256()
    environment_lock_hash = runtime.environment_lock_sha256()
    evaluation_identity = runtime._benchmark_evaluation_identity(args, config)
    policy_hash = runtime._verified_policy_sha256(
        args.policy_manifest,
        config=config,
        dataset_sha256=dataset_hash,
        split_manifest=args.benchmark_split_manifest,
        active_split=args.benchmark_split,
        active_implementation_sha256=implementation_hash,
        active_environment_lock_sha256=environment_lock_hash,
        evaluation_identity=evaluation_identity,
        prepare_only=prepare_only,
    )
    return RunProvenance(
        dataset_sha256=dataset_hash,
        split_manifest_sha256=(
            runtime.file_sha256(args.benchmark_split_manifest)
            if args.benchmark_split_manifest
            else ""
        ),
        implementation_sha256=implementation_hash,
        environment_lock_sha256=environment_lock_hash,
        policy_manifest_sha256=policy_hash,
        evaluation_protocol=evaluation_identity,
    )
