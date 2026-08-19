"""Run one delayed, bounded Qwen consolidation pass over a live memory store.

The command reconstructs the context that direct retrieval would expose for a
prompt, lets a frozen Qwen prefix inspect only those packed direct members,
and commits scalar co-activation statistics to schema v9.  Candidate text,
token activations, attention matrices, residuals, and K/V state die with the
process and are never written to the memory database.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Sequence

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.associations.head_memory import CAVBank, MemoryLinkResult, QwenMemoryLinker
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.domain.schemas import PackedContext


@dataclass(frozen=True, slots=True)
class QwenConsolidationReport:
    """Text-free diagnostics from one completed consolidation event."""

    event_id: str
    created: bool
    packed_direct_memories: int
    packed_direct_chunks: int
    workspace_candidates: int
    workspace_tokens: int
    workspace_passes: int
    total_candidate_inspections: int
    nodes_observed: int
    edges_reinforced: int
    edges_pruned: int
    qk_mean: float
    qk_max: float
    ov_mean: float
    ov_max: float
    cav_dimensions: int
    cav_active_dimensions: int
    elapsed_s: float
    retrieval_s: float = 0.0
    linker_load_s: float = 0.0
    retained_prompt_state_bytes: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _finite_nonnegative(values: Sequence[float]) -> list[float]:
    return [
        max(0.0, float(value))
        for value in values
        if math.isfinite(float(value))
    ]


def consolidate_packed_context(
    condenser: MemoryCondenser,
    linker: QwenMemoryLinker,
    user_text: str,
    packed: PackedContext,
    *,
    access_event_id: str | None = None,
    now_turn: int | None = None,
) -> QwenConsolidationReport:
    """Apply Qwen activation weighting to an already packed direct context."""

    started = time.perf_counter()
    result, update = condenser.consolidate_context_with_qwen(
        user_text,
        packed,
        linker,
        access_event_id=access_event_id,
        now_turn=now_turn,
    )
    if not isinstance(result, MemoryLinkResult):
        raise TypeError("Qwen linker must return MemoryLinkResult")
    qk = _finite_nonnegative([hit.qk_score for hit in result.hits])
    ov = _finite_nonnegative([hit.ov_transport for hit in result.hits])
    cav = tuple(float(value) for value in result.source_cav_signature)
    return QwenConsolidationReport(
        event_id=update.event_id,
        created=update.created,
        packed_direct_memories=len(packed.direct_memory_ids),
        packed_direct_chunks=len(packed.direct_expansion_chunk_ids),
        workspace_candidates=result.workspace_candidates,
        workspace_tokens=result.workspace_tokens,
        workspace_passes=result.passes,
        total_candidate_inspections=(
            result.total_candidate_inspections or result.workspace_candidates
        ),
        nodes_observed=update.nodes_observed,
        edges_reinforced=update.edges_reinforced,
        edges_pruned=update.edges_pruned,
        qk_mean=fmean(qk) if qk else 0.0,
        qk_max=max(qk, default=0.0),
        ov_mean=fmean(ov) if ov else 0.0,
        ov_max=max(ov, default=0.0),
        cav_dimensions=len(cav),
        cav_active_dimensions=sum(value > 0.0 for value in cav),
        elapsed_s=time.perf_counter() - started,
    )


def load_qwen_linker(
    model_dir: str | Path,
    *,
    prefix_layers: int = 7,
    attention_layer: int = 1,
    cav_report: str | Path | None = None,
    cav_vectors: str | Path | None = None,
    cav_layer: int = 5,
    concepts: Sequence[str] | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_candidates: int = 8,
    max_workspace_tokens: int = 1024,
) -> QwenMemoryLinker:
    """Load the frozen prefix and optional CAV bank used by the live teacher."""

    if (cav_report is None) != (cav_vectors is None):
        raise ValueError("cav_report and cav_vectors must be supplied together")
    encoder = Qwen3PrefixEncoder(
        model_dir,
        layers=prefix_layers,
        device=device,
        dtype=dtype,
    )
    bank = None
    if cav_report is not None and cav_vectors is not None:
        bank = CAVBank.load(
            cav_report,
            cav_vectors,
            layer=cav_layer,
            concepts=concepts,
            device=encoder.device,
        )
    return QwenMemoryLinker(
        encoder,
        layer=attention_layer,
        cav_bank=bank,
        max_candidates=max_candidates,
        max_workspace_tokens=max_workspace_tokens,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path(".cache/models/Qwen3-8B")
    )
    parser.add_argument(
        "--cav-report",
        type=Path,
        default=Path("eval_results/qwen3_prefix_cav_probe.json"),
    )
    parser.add_argument(
        "--cav-vectors",
        type=Path,
        default=Path("eval_results/qwen3_prefix_cav_probe.safetensors"),
    )
    parser.add_argument("--prefix-layers", type=int, default=7)
    parser.add_argument("--attention-layer", type=int, default=1)
    parser.add_argument("--cav-layer", type=int, default=5)
    parser.add_argument("--concept", action="append", dest="concepts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--retrieval-device", default="cpu")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-workspace-tokens", type=int, default=1024)
    parser.add_argument("--recent-turns", type=int, default=8)
    parser.add_argument("--k-memories", type=int, default=8)
    parser.add_argument("--k-expansions", type=int, default=10)
    parser.add_argument("--event-id")
    parser.add_argument(
        "--memory-id",
        action="append",
        default=[],
        help="Already packed direct memory ID; repeat to bypass retrieval.",
    )
    parser.add_argument(
        "--chunk-id",
        action="append",
        default=[],
        help="Already packed direct chunk ID; repeat to bypass retrieval.",
    )
    parser.add_argument(
        "--no-association-read",
        action="store_true",
        help="Do not use established consolidation links while rebuilding context.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with MemoryCondenser(
        data_dir=args.data_dir,
        device=args.retrieval_device,
    ) as condenser:
        retrieval_started = time.perf_counter()
        if args.memory_id or args.chunk_id:
            # Production should normally call ``consolidate_packed_context``
            # in-process with the PackedContext already used for generation.
            # Explicit IDs provide the same no-reretrieval path to operators.
            packed = PackedContext(
                direct_memory_ids=list(dict.fromkeys(args.memory_id)),
                direct_expansion_chunk_ids=list(dict.fromkeys(args.chunk_id)),
                consolidation_event_id=args.event_id,
            )
        else:
            # Retrieval runs first. Keeping it on CPU by default prevents
            # bge-m3 and the Qwen prefix from competing for a small GPU.
            packed = condenser.build_context(
                args.prompt,
                recent_turns=args.recent_turns,
                k_memories=args.k_memories,
                k_expansions=args.k_expansions,
                reheat_memories=False,
                use_consolidation=not args.no_association_read,
                learn_consolidation=False,
                access_event_id=args.event_id,
            )
        retrieval_s = time.perf_counter() - retrieval_started
        member_count = len(packed.direct_memory_ids) + len(
            packed.direct_expansion_chunk_ids
        )
        if member_count == 0:
            raise RuntimeError("direct retrieval packed no members to consolidate")

        linker_started = time.perf_counter()
        linker = load_qwen_linker(
            args.model_dir,
            prefix_layers=args.prefix_layers,
            attention_layer=args.attention_layer,
            cav_report=args.cav_report,
            cav_vectors=args.cav_vectors,
            cav_layer=args.cav_layer,
            concepts=args.concepts,
            device=args.device,
            dtype=args.dtype,
            max_candidates=args.max_candidates,
            max_workspace_tokens=args.max_workspace_tokens,
        )
        linker_load_s = time.perf_counter() - linker_started
        report = consolidate_packed_context(
            condenser,
            linker,
            args.prompt,
            packed,
            access_event_id=args.event_id,
        )
        report = replace(
            report,
            retrieval_s=retrieval_s,
            linker_load_s=linker_load_s,
        )
        print(report.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
