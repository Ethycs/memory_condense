"""Reproducible local Qwen smoke: summary attention, hierarchy, exact hydration.

The fixture uses precomputed summaries, not a fake attention signal. It tests
the real pinned Qwen prefix without sending any raw content to that model.
This is a mechanism smoke, not a benchmark accuracy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain.schemas import Turn
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.search.episodes.attention_hierarchy import build_attention_section_hierarchy
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.section_attention import route_summary_hierarchy
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=6)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw = [f"  RAW_SECRET_{i} τ\r\n" for i in range(4)]
    turns = [Turn(turn_id=f"turn-{i}", source_id="synthetic-session", role="user", text=text,
                  created_at=datetime(2026, 9, 8, tzinfo=timezone.utc)) for i, text in enumerate(raw)]
    summaries = ["Apple orchard harvest and seasonal fruit storage.",
                 "Orchard irrigation and pruning fruit trees.",
                 "Observatory telescope reservations and astronomy visits.",
                 "Observatory opening hours and stargazing appointments."]
    atoms = [SectionSummary(f"atom-{i}", turn.source_id, summary,
                            (RawSectionSpan.from_turn(turn),), "precomputed-smoke-summary-v1")
             for i, (turn, summary) in enumerate(zip(turns, summaries))]
    started = time.perf_counter()
    print("Loading local Qwen prefix...", flush=True)
    encoder = Qwen3PrefixEncoder(args.qwen_model_dir.resolve(), layers=args.layers,
                                 device=args.device, dtype="float16" if args.device != "cpu" else "float32")
    linker = QwenMemoryLinker(encoder, layer=args.layers - 1, max_candidates=4, max_workspace_tokens=1024)
    loaded_seconds = time.perf_counter() - started
    observed = []
    codes = {QwenMemoryLinker.link.__code__, QwenMemoryLinker.inspect_coverage.__code__}
    def observe(frame, event, _arg):
        if event == "call" and frame.f_code in codes:
            values = frame.f_locals
            observed.append({"method": frame.f_code.co_name,
                             "probe": values.get("source_text"),
                             "summaries": [candidate.text for candidate in values["candidates"]]})
    previous_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        print("Scoring summary boundaries and building hierarchy...", flush=True)
        hierarchy = build_attention_section_hierarchy(atoms,
            scorer=QwenAttentionHeadSurpriseScorer(linker, max_spans=3, span_token_cap=64),
            summarize_summaries=lambda text: " ".join(text.split()),
            summarizer_identity="bounded-summary-concatenation-smoke-v1",
            atom_token_cap=16, leaf_token_cap=16, window_atom_cap=3)
        index = SectionSummaryIndex.from_json(hierarchy.summary_index().to_json())
        print("Routing through summaries with Qwen...", flush=True)
        plan = route_summary_hierarchy("When can I visit the observatory?", index,
                                       linker=linker, max_sections=1)
    finally:
        sys.setprofile(previous_profile)
    allowed = set(summaries) | {section.summary for section in index.sections}
    assert observed and all(text in allowed for call in observed for text in call["summaries"])
    assert "RAW_SECRET" not in json.dumps(observed)
    assert all(window.signal.owned_runtime_binding for window in hierarchy.windows)
    assert len(plan.attention_receipt.rounds) > 1
    records = {turn.turn_id: turn for turn in turns}
    reads = []
    def load(identity):
        reads.append(identity)
        return records.get(identity)
    result = hydrate_section_plan(plan, load_turn=load)
    assert result.sections and not result.diagnostics
    for section in result.sections:
        for evidence in section.evidence:
            assert evidence.text == records[evidence.span.turn_id].text[evidence.span.start_char:evidence.span.end_char]
    assert len(observed) == sum(window.signal.forward_passes for window in hierarchy.windows) + sum(
        row.model_passes for row in plan.attention_receipt.rounds)
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), *(root / "src/memory_condense" / name for name in (
        "search/section_summary.py", "search/section_routing.py", "search/section_attention.py",
        "search/episodes/attention_hierarchy.py", "application/section_retrieval.py",
        "application/retrieval_workflow.py"))]
    report = {
        "format": "attention-summary-sections-local-smoke-v1",
        "synthetic_fixture": True, "precomputed_summaries": True, "accuracy_claim": None,
        "external_provider_calls": 0, "qwen_raw_content_inputs": 0,
        "qwen_input_audit": observed, "qwen_forward_workspaces": len(observed),
        "raw_turn_reads_after_routing": reads, "load_seconds": loaded_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "implementation": {str(path.relative_to(root)).replace("\\", "/"):
                           hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "hierarchy": hierarchy.identity_payload(), "result": result.identity_payload(),
    }
    (args.output_dir / "summary-index.json").write_text(index.to_json(), encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "qwen_forward_workspaces", "qwen_raw_content_inputs", "external_provider_calls",
        "raw_turn_reads_after_routing", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
