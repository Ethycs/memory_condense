from dataclasses import replace

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.user_spine_hierarchy import build_user_spine_hierarchy, compile_user_spine_exchanges
from memory_condense.search.native_hierarchy_occurrence import bind_native_hierarchy
from memory_condense.search.native_spine_memory import materialize_history
from memory_condense.search.native_spine_summary import body_identity, fragment_body
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_parent_hierarchy import summary_channels
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tests.test_attention_summary_sections import SummaryLinker


def fixture():
    body = {"turns": [
        {"role": "user", "text": "I plan to visit the café tomorrow. I have not booked it."},
        {"role": "assistant", "text": "You could visit the local museum too."},
        {"role": "user", "text": "I visited the gardens on May 4. 🌍"},
        {"role": "assistant", "text": "You might like the nearby trails."},
        {"role": "user", "text": "I prefer quiet parks and short walks."},
    ]}
    atoms = [{"pointer": f.pointer(), "summary": "User discusses visits tomorrow and on May 4."
              if f.role == "user" else "Assistant suggests places."} for f in fragment_body(body, token_cap=8)]
    def occurrence(day):
        source = {"original_session_ordinal": 0, "session_id": "repeated-body", "created_at": day + "T00:00:00+00:00",
                  "metadata_text": "source boundary", "body_sha256": body_identity(body), "dataset_origin": "M"}
        source["occurrence_id"] = identity_sha256(source)
        history = materialize_history([source], load_body=lambda _: body, load_summaries=lambda _: atoms,
                                      compiler_identity="fixture")
        return source, history
    source, history = occurrence("2026-01-02")
    def forbidden(_):
        raise AssertionError("the fixture requires no model generation")
    summarize = ReusingSpineSummarizer(forbidden)
    exchanges = compile_user_spine_exchanges(history.atoms, summarize=summarize,
                                            summarizer_identity="fixture", max_channel_tokens=128)
    tree = build_user_spine_hierarchy(exchanges,
        scorer=QwenAttentionHeadSurpriseScorer(SummaryLinker(), max_spans=8, span_token_cap=128),
        summarize=summarize, summarizer_identity="fixture", max_channel_tokens=128,
        window_exchange_cap=8, max_leaf_exchanges=2)
    template = {"body_sha256": source["body_sha256"], "index_json": tree.summary_index().to_json(),
                "atomic_index_json": SectionSummaryIndex(history.atoms).to_json(),
                "root_section_ids": list(tree.root_section_ids), "original_atomic_addresses_preserved": True,
                "raw_span_population_sha256": identity_sha256([s.receipt_sha256 for a in history.atoms for s in a.spans])}
    return template, occurrence


def test_repeated_body_keeps_content_and_topology_with_distinct_real_occurrences():
    template, occurrence = fixture()
    original = SectionSummaryIndex.from_json(template["index_json"])
    original_root = next(s for s in original.sections if s.section_id == template["root_section_ids"][0])
    indices, roots = [], []
    for day in ("2026-01-02", "2026-09-12"):
        source, history = occurrence(day)
        index, atoms, binding = bind_native_hierarchy(template, source, history.atoms)
        indices.append(index)
        roots.extend(binding.root_section_ids)
        root = next(s for s in index.sections if s.section_id == binding.root_section_ids[0])
        assert summary_channels(root) == summary_channels(original_root)
        assert root.spans == tuple(a.spans[0] for a in history.atoms)
        assert atoms.sections == SectionSummaryIndex(history.atoms).sections
        assert binding.model_calls == binding.raw_text_reads == 0
        packet = hydrate_section_plan(index.route("visits tomorrow May 4 suggests", max_sections=len(index.sections)),
                                      load_turn=history.get_turn, max_raw_spans=128, max_context_tokens=4096)
        assert not packet.diagnostics
        assert {e.span.receipt_sha256 for s in packet.sections for e in s.evidence} == {
            a.spans[0].receipt_sha256 for a in history.atoms}
        for section in packet.sections:
            for e in section.evidence:
                turn = history.get_turn(e.span.turn_id)
                assert e.text == turn.text[e.span.start_char:e.span.end_char]
                assert e.span.created_at == source["created_at"]
    assert roots[0] != roots[1]
    forest = SectionSummaryIndex(tuple(s for index in indices for s in index.sections))
    assert len(forest.sections) == sum(len(i.sections) for i in indices)


@pytest.mark.parametrize("defect", ["date", "body", "summary", "partition", "extra_source_field"])
def test_rebinding_cannot_relabel_changed_content_or_source(defect):
    template, occurrence = fixture()
    source, history = occurrence("2026-09-12")
    atoms = history.atoms
    if defect == "date":
        source["created_at"] = "2026-09-13T00:00:00+00:00"
        source["occurrence_id"] = identity_sha256({k: v for k, v in source.items() if k != "occurrence_id"})
    elif defect == "body":
        template = dict(template, body_sha256="different")
    elif defect == "summary":
        atoms = (replace(atoms[0], summary="A changed cached summary.", receipt_sha256=""), *atoms[1:])
    elif defect == "partition":
        atoms = atoms[1:]
    else:
        source["question"] = "A future benchmark question"
    with pytest.raises(ValueError):
        bind_native_hierarchy(template, source, atoms)
