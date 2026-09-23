from dataclasses import replace

import pytest

from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.episodes.user_spine_hierarchy import _render_channels
from memory_condense.search.native_hierarchy_occurrence import bind_native_hierarchy
from memory_condense.search.parent_budget_hierarchy_occurrence import bind_parent_budget_hierarchy, parent_summary_channels
from memory_condense.search.section_routing import SectionSummaryIndex
from tests.test_native_hierarchy_occurrence import fixture
from tests.test_native_spine_namespace import inputs
from tools.parent_budget_native_spine_namespace import materialize_parent_budget_namespace, validate_parent_policy


def larger_parent(template, *, oversized=False, wrong_role=False):
    index = SectionSummaryIndex.from_json(template["index_json"])
    sections = []
    for section in index.sections:
        if section.section_id == template["root_section_ids"][0]:
            user, attached = parent_summary_channels(section)
            user = (user + " Preserved visit detail." * (200 if oversized else 55)) if not wrong_role else None
            if not oversized and not wrong_role:
                assert 128 < count_tokens(user) <= 512
            section = replace(section, summary=_render_channels(user, attached, section.spans), receipt_sha256="")
        sections.append(section)
    return dict(template, index_json=SectionSummaryIndex(sections).to_json())


def test_original_small_templates_rebind_identically_under_both_contracts():
    template, occurrence = fixture()
    source, history = occurrence("2026-09-12")
    new_tree, new_atoms, new_binding = bind_parent_budget_hierarchy(template, source, history.atoms)
    old_tree, old_atoms, old_binding = bind_native_hierarchy(template, source, history.atoms)
    assert (new_tree.to_json(), new_atoms.to_json(), new_binding) == (old_tree.to_json(), old_atoms.to_json(), old_binding)


def test_larger_parent_preserves_literal_channels_and_exact_hydration_at_distinct_occurrences():
    template, occurrence = fixture()
    template = larger_parent(template)
    old_index = SectionSummaryIndex.from_json(template["index_json"])
    old_root = next(s for s in old_index.sections if s.section_id == template["root_section_ids"][0])
    roots = []
    for day in ("2026-01-02", "2026-09-12"):
        source, history = occurrence(day)
        with pytest.raises(ValueError, match="channel contract"):
            bind_native_hierarchy(template, source, history.atoms)
        index, atoms, binding = bind_parent_budget_hierarchy(template, source, history.atoms)
        roots.extend(binding.root_section_ids)
        root = next(s for s in index.sections if s.section_id == binding.root_section_ids[0])
        assert parent_summary_channels(root) == parent_summary_channels(old_root)
        assert root.spans == tuple(a.spans[0] for a in history.atoms)
        assert atoms.to_json() == SectionSummaryIndex(history.atoms).to_json()
        assert binding.raw_text_reads == binding.model_calls == 0
        packet = hydrate_section_plan(index.route("visit detail tomorrow", max_sections=len(index.sections)),
            load_turn=history.get_turn, max_raw_spans=128, max_context_tokens=4096)
        assert {e.span.receipt_sha256 for s in packet.sections for e in s.evidence} == {
            a.spans[0].receipt_sha256 for a in history.atoms}
        for s in packet.sections:
            for evidence in s.evidence:
                span = evidence.span
                assert evidence.text == history.get_turn(span.turn_id).text[span.start_char:span.end_char]
                assert span.created_at == source["created_at"]
    assert roots[0] != roots[1]


@pytest.mark.parametrize("defect", ["oversized", "wrong_role"])
def test_new_contract_rejects_oversized_or_misattributed_parent_channels(defect):
    template, occurrence = fixture()
    source, history = occurrence("2026-09-12")
    with pytest.raises(ValueError):
        bind_parent_budget_hierarchy(larger_parent(template, **{defect: True}), source, history.atoms)


def test_larger_parent_does_not_permit_changed_original_atomic_summary():
    template, occurrence = fixture()
    source, history = occurrence("2026-09-12")
    atoms = list(history.atoms)
    atoms[0] = replace(atoms[0], summary="Different user fact.", receipt_sha256="")
    with pytest.raises(ValueError, match="raw content, turn ownership, summary"):
        bind_parent_budget_hierarchy(larger_parent(template), source, atoms)


def test_partial_parent_namespace_preserves_all_original_atoms_and_rejects_full_admission():
    _, source, body, summaries = inputs()
    kwargs = dict(body_ids={source["body_sha256"]}, templates={}, load_body=lambda _: body,
                  load_summaries=lambda _: summaries, compiler_identity="fixture")
    with pytest.raises(ValueError, match="missing summaries or hierarchies"):
        materialize_parent_budget_namespace([source], **kwargs)
    namespace = materialize_parent_budget_namespace([source], allow_partial=True, **kwargs)
    assert len(namespace.atomic_index.sections) == len(summaries)
    assert not namespace.hierarchy.sections and not namespace.audit["complete_namespace"]
    with pytest.raises(ValueError):
        namespace.require_complete(minimum_body_tokens=1)


@pytest.mark.parametrize("changed", [dict(max_exchange_channel_tokens=512),
                                    dict(format="native-spine-original"), dict(raw_inputs_to_qwen=True)])
def test_admission_rejects_unrecognized_producer_or_changed_attention_input_policy(changed):
    payload = dict(format="native-spine-parent-budgeted-hierarchy-v1", max_exchange_channel_tokens=128,
        max_parent_channel_tokens=512, leaf_token_cap=512, max_leaf_exchanges=2, window_exchange_cap=8,
        max_prompt_tokens=2048, raw_inputs_to_qwen=False, timestamp_metadata_in_model_inputs=False,
        original_atomic_addresses_preserved=True)
    validate_parent_policy(payload)
    payload.update(changed)
    with pytest.raises(ValueError, match="producer or input policy"):
        validate_parent_policy(payload)
