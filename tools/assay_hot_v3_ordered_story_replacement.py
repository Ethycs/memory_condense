#!/usr/bin/env python3
"""Gold-blind hot-v3 assay with graph-backed ordered-story replacement.

This is a thin artifact-identity and CLI layer over
``assay_hot_v3_provider_free_witness``.  Every baseline lane, dedup rule,
binary pack, fallback, replay, and post-hoc scorer remains shared.  The only
alternative is the typed-result resolver: on a query-computable ambiguous
ordered-list frontier it may select a strictly validated chronological subset
from a graph built once per sealed namespace.  In every other semantic case it
returns the original typed result object.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from tools import assay_hot_v3_provider_free_witness as base
from memory_condense.domain._discourse_identity import quote_sha256
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.full_store_slot_closure import FullStoreWindowIndex
from tools.matched_eval.hot_incremental_graph_lane import (
    HotIncrementalGraphIndex,
    build_hot_incremental_graph_index,
)
from tools.matched_eval.hot_incremental_ordered_story import (
    replace_ambiguous_ordered_typed_witnesses_from_graph,
)
from tools.matched_eval.hot_typed_witness import HotTypedWitnessResult
from tools.matched_eval.typed_operator_spec import (
    AnswerShape,
    TemporalMode,
    compile_typed_operator_spec,
)


FORMAT = "memory-condense-hot-v3-ordered-story-construction-v4"
RUNTIME_FORMAT = "memory-condense-hot-v3-ordered-story-runtime-v4"
SCORE_FORMAT = "memory-condense-hot-v3-ordered-story-score-v4"
ROW_FORMAT = "memory-condense-hot-v3-ordered-story-row-v4"
COMPOSITION_FORMAT = "memory-condense-hot-v3-ordered-story-composition-v4"
TIMING_FORMAT = "memory-condense-hot-v3-ordered-story-timing-v4"
REPLAY_FORMAT = "memory-condense-hot-v3-ordered-story-replay-v4"
POLICY_ID = "hot-v3-ambiguous-ordered-story-graph-replacement-v4"

DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v3-ordered-story-graph-20260907"
)

VARIANT = base.AssayVariant(
    construction_format=FORMAT,
    runtime_format=RUNTIME_FORMAT,
    score_format=SCORE_FORMAT,
    row_format=ROW_FORMAT,
    composition_format=COMPOSITION_FORMAT,
    timing_format=TIMING_FORMAT,
    replay_format=REPLAY_FORMAT,
    policy_id=POLICY_ID,
    construction_status=(
        "sealed_gold_blind_hot_v3_graph_ordered_story_replacement"
    ),
    implementation_format=(
        "memory-condense-hot-v3-graph-ordered-story-implementation-v3"
    ),
    implementation_extra_paths=(
        "src/memory_condense/search/incremental_conversation_graph.py",
        "tools/assay_hot_v3_ordered_story_replacement.py",
        "tools/matched_eval/hot_incremental_graph_lane.py",
        "tools/matched_eval/hot_incremental_ordered_story.py",
        "tools/matched_eval/typed_operator_spec.py",
    ),
    repair_legacy_parent_projection_collisions=True,
)


@dataclass(frozen=True, slots=True)
class OrderedStoryNamespaceIndex:
    """One namespace plus its question-gated optional story graph."""

    full: FullStoreWindowIndex
    eligible_dated_question_sha256s: tuple[str, ...]
    graph_build_reason: Literal[
        "eligible_ordered_temporal_questions",
        "no_eligible_ordered_temporal_questions",
    ]
    graph: HotIncrementalGraphIndex | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.full) is not FullStoreWindowIndex:
            raise ValueError("ordered-story namespace lost its full-store index")
        eligible = self.eligible_dated_question_sha256s
        if (
            type(eligible) is not tuple
            or len(eligible) != len(set(eligible))
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in eligible
            )
        ):
            raise ValueError("ordered-story eligibility hashes changed")
        expected_reason = (
            "eligible_ordered_temporal_questions"
            if eligible
            else "no_eligible_ordered_temporal_questions"
        )
        if self.graph_build_reason != expected_reason:
            raise ValueError("ordered-story graph-build reason changed")
        if not eligible and self.graph is not None:
            raise ValueError("ineligible namespace unexpectedly built a graph")
        if eligible and type(self.graph) is not HotIncrementalGraphIndex:
            raise ValueError("eligible namespace is missing its story graph")
        if self.graph is not None and (
            self.graph.parent is not self.full
            or self.graph.parent.receipt_sha256 != self.full.receipt_sha256
            or self.graph.graph.stats() != self.graph.graph_stats
        ):
            raise ValueError("ordered-story namespace graph lost its sealed parent")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("ordered-story namespace index receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(
            self.projection(),
            path="ordered_story_namespace_index",
        )

    @property
    def rows(self) -> tuple[Any, ...]:
        """Expose the physical inventory expected by the shared assay loop."""

        return tuple(self.full.rows)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        graph = self.graph
        value = {
            "cache_receipt_sha256": self.full.cache.cache_receipt_sha256,
            "eligible_dated_question_sha256s": list(
                self.eligible_dated_question_sha256s
            ),
            "extraction_policy_sha256": (
                None
                if graph is None
                else graph.graph.extraction_policy.policy_sha256
            ),
            "format": "memory-condense-ordered-story-namespace-index-v3",
            "full_store_index_receipt_sha256": self.full.receipt_sha256,
            "gold_loaded": False,
            "graph_build_count": int(graph is not None),
            "graph_build_reason": self.graph_build_reason,
            "graph_index_receipt_sha256": (
                None if graph is None else graph.receipt_sha256
            ),
            "graph_stats": None if graph is None else asdict(graph.graph_stats),
            "model_calls": 0,
            "new_provider_calls": 0,
            "story_index_policy_sha256": (
                None
                if graph is None
                else graph.graph.story_index_policy.policy_sha256
            ),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _eligible_dated_question_sha256s(
    context: object,
    namespace: object,
) -> tuple[str, ...]:
    """Compile one sealed question-only eligibility inventory."""

    namespace_id = str(getattr(namespace, "namespace_id"))
    population = getattr(context, "population", None)
    rows = tuple(getattr(population, "rows", ()))
    if not namespace_id or not rows:
        raise ValueError("ordered-story eligibility population changed")
    matched = 0
    eligible: list[str] = []
    for row in rows:
        row_namespace = getattr(row, "namespace", None)
        if str(getattr(row_namespace, "namespace_id", "")) != namespace_id:
            continue
        matched += 1
        source = getattr(row, "source", None)
        packet = getattr(source, "packet", None)
        dated_question = getattr(packet, "dated_question", None)
        dated_sha256 = getattr(packet, "dated_question_sha256", None)
        if (
            type(dated_question) is not str
            or not dated_question
            or dated_question.strip() != dated_question
            or type(dated_sha256) is not str
            or quote_sha256(dated_question) != dated_sha256
        ):
            raise ValueError("ordered-story dated-question seal changed")
        spec = compile_typed_operator_spec(dated_question)
        if (
            spec.answer_shape is AnswerShape.ORDERED_LIST
            and spec.temporal_mode is TemporalMode.ORDER
            and spec.requires_complete_frontier
            and type(spec.cardinality) is int
            and spec.cardinality > 0
        ):
            eligible.append(dated_sha256)
    if matched == 0:
        raise ValueError("namespace is absent from the sealed question population")
    return tuple(dict.fromkeys(eligible))


def _build_ordered_story_namespace_index(
    context: object,
    namespace: object,
) -> tuple[OrderedStoryNamespaceIndex, Mapping[str, Any]]:
    """Build the graph once in the existing per-namespace cold lifecycle."""

    full, timing = base._build_resident_index(  # noqa: SLF001
        context,
        namespace,
    )
    eligible = _eligible_dated_question_sha256s(context, namespace)
    graph: HotIncrementalGraphIndex | None = None
    graph_build_ns = 0
    if eligible:
        started = time.perf_counter_ns()
        graph = build_hot_incremental_graph_index(full)
        graph_build_ns = time.perf_counter_ns() - started
    reason = (
        "eligible_ordered_temporal_questions"
        if eligible
        else "no_eligible_ordered_temporal_questions"
    )
    combined = OrderedStoryNamespaceIndex(
        full=full,
        eligible_dated_question_sha256s=eligible,
        graph_build_reason=reason,
        graph=graph,
    )
    return combined, {
        **dict(timing),
        "graph_eligible_dated_question_sha256s": list(eligible),
        "graph_index_build_count": int(graph is not None),
        "graph_index_build_ns": graph_build_ns,
        "graph_index_build_reason": reason,
        "graph_index_receipt_sha256": (
            None if graph is None else graph.receipt_sha256
        ),
        "graph_index_stats": (
            None if graph is None else asdict(graph.graph_stats)
        ),
        "graph_namespace_index_receipt_sha256": combined.receipt_sha256,
        "graph_story_index_policy_sha256": (
            None
            if graph is None
            else graph.graph.story_index_policy.policy_sha256
        ),
    }


def _ordered_namespace(value: object) -> OrderedStoryNamespaceIndex:
    if type(value) is not OrderedStoryNamespaceIndex:
        raise TypeError("ordered-story hook requires its namespace index")
    return value


def _select_profile(index: object, dated_question: str) -> object:
    return base.select_profile_preference_evidence(
        _ordered_namespace(index).full,
        dated_question,
    )


def _build_typed_index(index: object) -> object:
    return base.build_hot_typed_witness_index(_ordered_namespace(index).full)


def _build_link_index(index: object) -> object:
    return base.build_hot_v3_activated_turn_link_index(
        _ordered_namespace(index).full
    )


def _resolve_typed(
    index: object,
    dated_question: str,
    baseline: object,
) -> object:
    namespace = _ordered_namespace(index)
    if (
        type(baseline) is not HotTypedWitnessResult
        or baseline.dated_question != dated_question
    ):
        raise ValueError("ordered-story baseline question binding changed")
    if quote_sha256(dated_question) not in (
        namespace.eligible_dated_question_sha256s
    ):
        return baseline
    if namespace.graph is None:
        raise RuntimeError("eligible ordered-story question has no graph")
    return replace_ambiguous_ordered_typed_witnesses_from_graph(
        namespace.graph,
        dated_question,
        baseline,
    )


def _runtime_hooks() -> base.RuntimeHooks:
    return replace(
        base._default_hooks(),  # noqa: SLF001 - shared assay extension seam
        build_full_index=_build_ordered_story_namespace_index,
        select_profile=_select_profile,
        build_typed_index=_build_typed_index,
        build_link_index=_build_link_index,
        resolve_typed=_resolve_typed,
    )


def construct(
    *,
    v3_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
    ordinals: Sequence[int] = tuple(range(base.EXPECTED_QUESTION_COUNT)),
) -> tuple[str, str]:
    return base.construct(
        v3_root=v3_root,
        retrieval_path=retrieval_path,
        store_root=store_root,
        output_root=output_root,
        ordinals=ordinals,
        hooks=_runtime_hooks(),
        variant=VARIANT,
    )


def replay(*, v3_root: Path, output_root: Path) -> str:
    return base.replay(
        v3_root=v3_root,
        output_root=output_root,
        variant=VARIANT,
    )


def score(
    *,
    dataset: Path,
    split_manifest: Path,
    v3_root: Path,
    output_root: Path,
) -> str:
    return base.score(
        dataset=dataset,
        split_manifest=split_manifest,
        v3_root=v3_root,
        output_root=output_root,
        variant=VARIANT,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=base.DEFAULT_V3_ROOT)
    parser.add_argument("--retrieval", type=Path, default=base.DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=base.DEFAULT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    construct_parser = commands.add_parser("construct")
    construct_parser.add_argument(
        "--ordinals",
        default="all",
        help="'all'/'full100' or comma-separated locked ordinals",
    )
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--split-manifest", type=Path, required=True)
    commands.add_parser("replay")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.command == "construct":
        construct(
            v3_root=args.v3_root.resolve(),
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
            output_root=output_root,
            ordinals=base._parse_ordinals(args.ordinals),  # noqa: SLF001
        )
    elif args.command == "score":
        score(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            v3_root=args.v3_root.resolve(),
            output_root=output_root,
        )
    elif args.command == "replay":
        replay(v3_root=args.v3_root.resolve(), output_root=output_root)
    else:  # pragma: no cover
        raise AssertionError(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "POLICY_ID",
    "VARIANT",
    "construct",
    "replay",
    "score",
]
