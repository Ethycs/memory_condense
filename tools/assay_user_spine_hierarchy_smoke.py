"""Live Qwen user-spine smoke with precomputed summaries and raw canaries.

This tests actual attention, summary compilation, routing, and exact hydration.
The summaries are authored fixture data, not a measured raw-summary compiler.
"""

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain.schemas import Turn
from memory_condense.domain.integrity import file_sha256
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges, build_user_spine_hierarchy
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.summary_reasoning import reason_over_summary_hierarchy
from tools.assay_user_spine_hierarchy import Journal, REPO, implementation
from tools.matched_eval.artifacts import publish_sealed_json


SUMMARIES = (
    "User requests advice on harvesting apples from their orchard.",
    "Assistant suggests checking fruit ripeness and using padded baskets.",
    "User asks how to irrigate and prune orchard trees during summer.",
    "Assistant proposes a watering schedule and pruning practices; no user acceptance is recorded.",
    "User asks how to reserve a telescope session at the observatory.",
    "Assistant describes advance telescope reservations and appointment availability.",
    "User asks about observatory opening hours and stargazing appointments.",
    "Assistant describes evening observatory visits and the appointment process.",
)


class SummaryOnlyJournal(Journal):
    def complete_messages(self, messages, **kwargs):
        if "RAW_CANARY" in json.dumps(messages) or "private-source" in json.dumps(messages):
            raise ValueError("raw content or source identity escaped into a Qwen request")
        return super().complete_messages(messages, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--enable-provider", action="store_true")
    args = parser.parse_args()
    root = args.output_root
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "format": "qwen-user-spine-canary-smoke-v1", "implementation": implementation(),
        "smoke_sha256": file_sha256(Path(__file__)), "summaries": list(SUMMARIES),
        "qwen_model": "qwen3-8b", "gateway_url": "https://central-dev.zt:4000/v1",
        "max_qwen_completion_calls": 24, "precomputed_fixture_summaries": True,
        "private_transcript_payload": False, "accuracy_claim": None,
    })
    turns = [Turn(turn_id=f"private-turn-{i}", source_id="private-source", role="user" if i % 2 == 0 else "assistant",
                  text=f"  RAW_CANARY_{i} τ\r\n", created_at=datetime(2026, 9, 9, tzinfo=timezone.utc)) for i in range(8)]
    atoms = [SectionSummary(f"atom-{i}", turn.source_id, summary, (RawSectionSpan.from_turn(turn),), "authored-smoke-summary-v1")
             for i, (turn, summary) in enumerate(zip(turns, SUMMARIES, strict=True))]
    journal = SummaryOnlyJournal(root, preflight, args.enable_provider)
    started = time.perf_counter()
    exchanges = compile_user_spine_exchanges(atoms, summarize=journal.summarize, summarizer_identity=preflight.sha256)
    print("Loading Qwen attention prefix...", flush=True)
    encoder = Qwen3PrefixEncoder(REPO / ".cache/models/Qwen3-8B", layers=6, device="cuda", dtype="float16")
    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=4, max_workspace_tokens=1024)
    hierarchy = build_user_spine_hierarchy(exchanges,
        scorer=QwenAttentionHeadSurpriseScorer(linker, max_spans=3, span_token_cap=64),
        summarize=journal.summarize, summarizer_identity=preflight.sha256,
        max_leaf_exchanges=1, window_exchange_cap=3)
    index = SectionSummaryIndex.from_json(hierarchy.summary_index().to_json())
    query = "How can I reserve a telescope session at the observatory?"
    plans = {"conventional_bm25": index.route(query, max_sections=1),
             "qwen_summary_reasoning": reason_over_summary_hierarchy(query, index, reasoner=journal,
                                         max_sections=1, group_size=4, max_calls=8, max_prompt_tokens=2048)}
    by_id = {turn.turn_id: turn for turn in turns}
    results = {name: hydrate_section_plan(plan, load_turn=by_id.get) for name, plan in plans.items()}
    assert all(r.sections and not r.requires_raw_fallback for r in results.values())
    for result in results.values():
        for section in result.sections:
            assert [r.span.role for r in section.evidence] == ["user", "assistant"]
            assert all(r.text == by_id[r.span.turn_id].text for r in section.evidence)
    artifact, _ = publish_sealed_json(root / "result.json", {
        "preflight_sha256": preflight.sha256, "hierarchy": hierarchy.identity_payload(),
        "qwen_request_artifact_shas": journal.requests,
        "arms": {name: result.identity_payload() for name, result in results.items()},
        "raw_qwen_inputs": 0, "precomputed_fixture_summaries": True, "accuracy_claim": None,
        "summary_routing_target_selected": {name: "private-turn-4" in [r.span.turn_id for s in result.sections for r in s.evidence]
                                            for name, result in results.items()},
    })
    print(json.dumps({"result_sha256": artifact.sha256, "new_provider_calls": journal.calls,
        "checkpoint_hits": journal.hits, "elapsed_seconds": time.perf_counter() - started,
        "sections": len(hierarchy.sections), "attention_windows": len(hierarchy.windows),
        "targets": artifact.payload["summary_routing_target_selected"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
