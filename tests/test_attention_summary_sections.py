from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from memory_condense.application.retrieval_workflow import RetrievalWorkflowMixin
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.attention_hierarchy import (
    build_attention_section_hierarchy, compile_attention_atoms,
)
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.section_attention import route_summary_hierarchy
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary


class SummaryLinker:
    """Transparent test double: real scorer reduces these synthetic head vectors."""
    max_candidates = 4
    max_workspace_tokens = 512
    layer = 1
    head_vote_k = 2
    cav_bank = None

    def __init__(self):
        self.encoder = SimpleNamespace(
            checkpoint_identity=SimpleNamespace(model_id="Qwen/Qwen3-8B",
                model_revision="test", checkpoint_sha256="a" * 64),
            device="cpu", dtype_name="float32", layers=2,
        )
        self.inputs = []
        self.fail = None

    def inspect_coverage(self, probe, candidates):
        self.inputs.append((probe, tuple(candidate.text for candidate in candidates)))
        return SimpleNamespace(
            hits=tuple(SimpleNamespace(episode_id=c.episode_id, qk_score=0.5, ov_transport=1.0,
                transport_signature=np.array((1.0 if "orchard" in c.text else -1.0, 0.0))) for c in candidates),
            workspace_candidates=len(candidates), workspace_tokens=20 + len(candidates),
            passes=1, total_candidate_inspections=len(candidates), retained_transformer_state_bytes=0,
        )

    def inspect_nested(self, probe, groups, *, beam_per_group, top_k, score_mode):
        candidates = [candidate for group in groups for candidate in group]
        self.inputs.append((probe, tuple(candidate.text for candidate in candidates)))
        hits = [SimpleNamespace(episode_id=c.episode_id,
                qk_score=0.9 if "observatory" in c.text else 0.1, ov_transport=0.5) for c in candidates]
        hits.sort(key=lambda hit: (-hit.qk_score, hit.episode_id))
        hits = hits[:top_k]
        if self.fail == "foreign":
            hits[0].episode_id = "foreign"
        elif self.fail == "nan":
            hits[0].qk_score = float("nan")
        elif self.fail == "empty":
            hits = []
        return SimpleNamespace(hits=tuple(hits), passes=len(groups),
            max_workspace_candidates=max(map(len, groups)), max_workspace_tokens=40,
            total_candidate_inspections=len(candidates))


def turn(i, source="source"):
    return Turn(turn_id=str(i), source_id=source, role="user",
                text=f"  RAW_SECRET_{i} τ\r\n", created_at=datetime(2026, 9, 8, tzinfo=timezone.utc))


def hierarchy(linker, n=4, window=3):
    turns = [turn(i) for i in range(n)]
    atoms = compile_attention_atoms(turns, summarize_raw=lambda raw:
        "orchard harvest" if any(f"SECRET_{i}" in raw for i in range(2)) else "observatory telescope",
        summarizer_identity="separate-test-summarizer", atom_token_cap=16)
    material = []
    def combine(text):
        material.append(text)
        return " / ".join(dict.fromkeys(text.split("\n\n")))
    result = build_attention_section_hierarchy(atoms,
        scorer=QwenAttentionHeadSurpriseScorer(linker, span_token_cap=64, max_spans=window),
        summarize_summaries=combine, summarizer_identity="test-summary-combiner",
        atom_token_cap=16, leaf_token_cap=16, window_atom_cap=window)
    return turns, atoms, result, material


def test_attention_guides_boundaries_and_never_observes_raw_content():
    linker = SummaryLinker()
    turns, atoms, built, material = hierarchy(linker)
    root = next(s for s in built.sections if s.section_id == built.root_section_ids[0])
    assert len(root.child_section_ids) == 2
    split = next(s for s in built.splits if s.section_id == root.section_id)
    assert split.split_atom == 2 and split.attention_change == pytest.approx(1.0)
    assert root.spans == tuple(atom.spans[0] for atom in atoms)
    leaves = sorted((s for s in built.sections if not s.child_section_ids), key=lambda s: s.spans[0].turn_id)
    assert tuple(span for leaf in leaves for span in leaf.spans) == root.spans
    assert [(w.atom_start, w.atom_end) for w in built.windows] == [(0, 3), (2, 4)]
    assert all(w.signal.retained_signal_transformer_state_bytes == 0 for w in built.windows)
    assert built.retained_transformer_token_state_bytes == 0
    assert "RAW_SECRET" not in repr(linker.inputs) + repr(material) + built.summary_index().to_json()
    assert {text for _, batch in linker.inputs for text in batch} == {atom.summary for atom in atoms}


def test_qwen_routes_summary_hierarchy_before_any_raw_read():
    linker = SummaryLinker()
    turns, _, built, _ = hierarchy(linker)
    # Persist/reload only summaries and authenticated raw references.
    index = SectionSummaryIndex.from_json(built.summary_index().to_json())
    linker.inputs.clear()
    reads = []
    def load(identity):
        assert len(linker.inputs) == 2  # root and selected children finished
        reads.append(identity)
        return next(t for t in turns if t.turn_id == identity)
    facade = RetrievalWorkflowMixin()
    facade._transcript = SimpleNamespace(get_turn=load)
    result = facade.search_attention_summary_sections("observatory", index, linker=linker, max_sections=1)
    assert reads == ["2", "3"]
    assert [r.text for r in result.sections[0].evidence] == [turns[2].text, turns[3].text]
    assert "RAW_SECRET" not in repr(linker.inputs)
    assert "observatory" not in result.render_context()
    audit = result.plan.attention_receipt
    assert audit.raw_content_inspections == audit.retained_transformer_token_state_bytes == 0
    assert len(audit.rounds) == 2
    assert not result.plan.frontier_closed


def test_depth_cap_hydrates_whole_parent_and_budget_rejects_atomically():
    linker = SummaryLinker()
    turns, _, built, _ = hierarchy(linker)
    plan = route_summary_hierarchy("observatory", built.summary_index(), linker=linker,
                                   max_sections=1, max_depth=1)
    assert plan.routes[0].section.section_id == built.root_section_ids[0]
    records = {t.turn_id: t for t in turns}
    full = hydrate_section_plan(plan, load_turn=records.get)
    assert len(full.sections[0].evidence) == 4
    rejected = hydrate_section_plan(plan, load_turn=records.get, max_context_tokens=1)
    assert not rejected.sections and rejected.raw_turn_read_count == 0


def test_source_scope_precedes_qwen_and_empty_scope_never_calls_it():
    linker = SummaryLinker()
    sections = [SectionSummary(str(i), t.source_id, "observatory", (RawSectionSpan.from_turn(t),), "test")
                for i, t in enumerate([turn(0, "prefix:a"), turn(1, "prefix:b")])]
    index = SectionSummaryIndex(sections)
    assert not route_summary_hierarchy("query", index, linker=linker, eligible_source_ids=["prefix"]).routes
    assert linker.inputs == []
    scoped = route_summary_hierarchy("query", index, linker=linker, eligible_source_ids=["prefix:b"])
    assert [r.section.source_id for r in scoped.routes] == ["prefix:b"]


@pytest.mark.parametrize("mode", ["foreign", "nan", "empty"])
def test_invalid_qwen_selections_fail_before_hydration(mode):
    linker = SummaryLinker()
    _, _, built, _ = hierarchy(linker)
    linker.fail = mode
    facade = RetrievalWorkflowMixin()
    facade._transcript = SimpleNamespace(get_turn=lambda _: pytest.fail("raw I/O before valid routing"))
    with pytest.raises(ValueError, match="Qwen returned"):
        facade.search_attention_summary_sections("query", built.summary_index(), linker=linker)


def test_unicode_atom_partition_and_reassembly_are_lossless():
    original = turn(0).model_copy(update={"text": "  🙂猫 café\r\n" * 35})
    atoms = compile_attention_atoms([original], summarize_raw=lambda _: "compact topic",
                                    summarizer_identity="test", atom_token_cap=8, max_summary_tokens=8)
    assert len(atoms) > 1
    assert "".join(original.text[a.spans[0].start_char:a.spans[0].end_char] for a in atoms) == original.text
    section = SectionSummary("whole", original.source_id, "topic", tuple(a.spans[0] for a in atoms), "test")
    plan = SectionSummaryIndex([section]).route("topic")
    result = hydrate_section_plan(plan, load_turn=lambda _: original, max_raw_spans=len(atoms))
    assert result.raw_turn_read_count == 1
    assert original.text in result.render_context()


def test_hierarchy_rejects_raw_inputs_and_summary_truncation_before_qwen():
    linker = SummaryLinker()
    scorer = QwenAttentionHeadSurpriseScorer(linker, span_token_cap=8)
    kwargs = dict(scorer=scorer, summarize_summaries=lambda _: "topic", summarizer_identity="test")
    with pytest.raises(ValueError, match="truncated"):
        build_attention_section_hierarchy([], **kwargs, max_summary_tokens=9)
    with pytest.raises(ValueError, match="summary descriptors"):
        build_attention_section_hierarchy([turn(0)], **kwargs, max_summary_tokens=8)
    assert linker.inputs == []


def test_children_cannot_change_their_parent_raw_partition():
    first, second = turn(0), turn(1)
    leaf = SectionSummary("leaf", first.source_id, "topic", (RawSectionSpan.from_turn(first),), "test")
    parent = SectionSummary("parent", first.source_id, "topic", (RawSectionSpan.from_turn(second),),
                            "test", child_section_ids=("leaf",))
    with pytest.raises(ValueError, match="partition"):
        SectionSummaryIndex([parent, leaf])
