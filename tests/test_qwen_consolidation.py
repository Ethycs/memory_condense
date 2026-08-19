from __future__ import annotations

import json
from types import SimpleNamespace

from memory_condense.associations.consolidation import ConsolidationUpdate
from memory_condense.associations.head_memory import MemoryLinkHit, MemoryLinkResult
from memory_condense.tooling.qwen_consolidation import (
    build_parser,
    consolidate_packed_context,
)
from memory_condense.domain.schemas import PackedContext


class FakeQwenLinker:
    def link(self, source_text, candidates, *, top_k=None):
        bounded = list(candidates[:top_k])
        return MemoryLinkResult(
            hits=tuple(
                MemoryLinkHit(
                    episode_id=candidate.episode_id,
                    qk_score=0.8 - index * 0.1,
                    ov_transport=0.4 + index * 0.1,
                    head_weights=(0.25, 0.75),
                )
                for index, candidate in enumerate(bounded)
            ),
            source_cav_signature=(-0.2, 0.7),
            workspace_candidates=len(bounded),
            workspace_tokens=48,
        )


class FakeCondenser:
    def consolidate_context_with_qwen(
        self,
        user_text,
        packed,
        linker,
        *,
        access_event_id=None,
        now_turn=None,
    ):
        del now_turn
        candidates = [
            SimpleNamespace(episode_id=f"m:{item_id}")
            for item_id in packed.direct_memory_ids
        ]
        candidates.extend(
            SimpleNamespace(episode_id=f"c:{item_id}")
            for item_id in packed.direct_expansion_chunk_ids
        )
        return linker.link(user_text, candidates, top_k=len(candidates)), (
            ConsolidationUpdate(
                event_id=access_event_id or "generated-event",
                created=True,
                members_observed=len(candidates),
                edges_reinforced=1,
                edges_pruned=0,
            )
        )


def test_delayed_qwen_report_contains_only_bounded_scalar_diagnostics():
    packed = PackedContext(
        direct_memory_ids=["memory-1"],
        direct_expansion_chunk_ids=["chunk-1"],
    )

    report = consolidate_packed_context(
        FakeCondenser(),
        FakeQwenLinker(),
        "What storage constraints matter?",
        packed,
        access_event_id="completed-turn-7",
    )

    payload = json.loads(report.to_json())
    assert payload["event_id"] == "completed-turn-7"
    assert payload["created"] is True
    assert payload["workspace_tokens"] == 48
    assert payload["cav_dimensions"] == 2
    assert payload["cav_active_dimensions"] == 1
    assert payload["retained_prompt_state_bytes"] == 0
    assert "prompt" not in payload
    assert "hits" not in payload


def test_qwen_consolidation_command_defaults_to_bounded_prefix_and_cpu_retrieval():
    args = build_parser().parse_args(
        ["--data-dir", "memory", "--prompt", "What did we decide?"]
    )
    assert args.prefix_layers == 7
    assert args.attention_layer == 1
    assert args.cav_layer == 5
    assert args.max_candidates == 8
    assert args.max_workspace_tokens == 1024
    assert args.retrieval_device == "cpu"
    assert args.memory_id == []
    assert args.chunk_id == []


def test_explicit_packed_ids_are_repeatable_cli_inputs():
    args = build_parser().parse_args(
        [
            "--data-dir",
            "memory",
            "--prompt",
            "What did we decide?",
            "--memory-id",
            "m1",
            "--memory-id",
            "m2",
            "--chunk-id",
            "c1",
        ]
    )
    assert args.memory_id == ["m1", "m2"]
    assert args.chunk_id == ["c1"]
