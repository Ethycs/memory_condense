from __future__ import annotations

from memory_condense.domain.discourse import (
    EvidenceAtom,
    EvidenceSpan,
    make_atom_id,
    quote_sha256,
)
from memory_condense.ingest.discourse_linker import (
    LinkerInput,
    RuleBasedDiscourseLinker,
)


def _input(ordinal: int, text: str, *, source: str = "thread") -> LinkerInput:
    span = EvidenceSpan(
        chunk_id=f"{source}-chunk-{ordinal}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=ordinal,
        source_id=source,
    )
    return LinkerInput(
        EvidenceAtom(
            atom_id=make_atom_id(span),
            span=span,
            text=text,
            label=f"turn-{ordinal}",
        ),
        episode_id=f"episode-{source}-{ordinal // 3}",
    )


def test_rule_linker_connects_distant_explicit_engineering_evidence() -> None:
    inputs = (
        _input(1, "Our goal is 95 percent judged accuracy."),
        _input(2, "We implemented a larger cross encoder."),
        _input(3, "The weather report was unrelated noise."),
        _input(4, "The measured result showed excessive latency."),
        _input(5, "We decided to use the smaller selector."),
        _input(6, "A lunch discussion added unrelated noise."),
        _input(7, "We revised that decision; instead use the hybrid selector."),
        _input(8, "A regression remains an unresolved blocker."),
        _input(9, "The cache-key change resolved the blocker."),
    )

    result = RuleBasedDiscourseLinker().link("artifact-rules", inputs)

    kinds = [item.kind for item in result.units]
    assert "goal" in kinds
    assert "action" in kinds
    assert "observation" in kinds
    assert "decision" in kinds
    assert "issue" in kinds
    semantic = [
        item for item in result.relations if item.relation_type != "sequence"
    ]
    assert {item.relation_type for item in semantic} == {
        "evaluates",
        "revises",
        "resolves",
    }
    by_type = {item.relation_type: item for item in semantic}
    unit_by_id = {item.unit_id: item for item in result.units}
    revised_members = by_type["revises"].members
    assert unit_by_id[revised_members[0].unit_id].asserted_ordinal == 5
    assert unit_by_id[revised_members[1].unit_id].asserted_ordinal == 7
    resolved_members = by_type["resolves"].members
    assert unit_by_id[resolved_members[0].unit_id].asserted_ordinal == 8
    assert unit_by_id[resolved_members[1].unit_id].asserted_ordinal == 9
    assert result.retained_request_token_state_bytes == 0


def test_rule_linker_is_domain_neutral_and_evaluates_a_medical_action() -> None:
    inputs = (
        _input(1, "We observed a recurring fever.", source="history"),
        _input(2, "The clinician tried a new intervention.", source="history"),
        _input(3, "The measured result showed the fever declined.", source="history"),
    )

    result = RuleBasedDiscourseLinker().link("artifact-medical", inputs)

    evaluates = [
        item for item in result.relations if item.relation_type == "evaluates"
    ]
    assert len(evaluates) == 1
    units = {item.unit_id: item for item in result.units}
    assert units[evaluates[0].members[0].unit_id].kind == "action"
    assert units[evaluates[0].members[1].unit_id].kind == "observation"


def test_rule_linker_never_creates_implicit_cross_source_edges() -> None:
    inputs = (
        _input(1, "We decided to use option A.", source="alpha"),
        _input(2, "However option A conflicts with the measurement.", source="beta"),
    )

    result = RuleBasedDiscourseLinker().link("artifact-isolation", inputs)

    assert result.relations == ()


def test_rule_linker_sequence_uses_nearest_prior_in_each_interleaved_source() -> None:
    inputs = (
        _input(1, "Alpha began with option A.", source="alpha"),
        _input(2, "Beta began with option B.", source="beta"),
        _input(3, "Alpha continued with option C.", source="alpha"),
        _input(4, "Beta continued with option D.", source="beta"),
    )

    result = RuleBasedDiscourseLinker().link("artifact-interleaved", inputs)

    units = {item.unit_id: item for item in result.units}
    sequence_pairs = {
        (
            units[relation.members[0].unit_id].evidence[0].source_id,
            units[relation.members[0].unit_id].asserted_ordinal,
            units[relation.members[1].unit_id].asserted_ordinal,
        )
        for relation in result.relations
        if relation.relation_type == "sequence"
    }
    assert sequence_pairs == {("alpha", 1, 3), ("beta", 2, 4)}


def test_rule_linker_output_is_deterministic_and_contains_no_generated_evidence() -> None:
    inputs = (
        _input(2, "The hard budget must remain fixed."),
        _input(1, "Our objective is accurate recall."),
    )
    linker = RuleBasedDiscourseLinker()

    first = linker.link("artifact-deterministic", inputs)
    second = linker.link("artifact-deterministic", tuple(reversed(inputs)))

    assert first == second
    assert all("text" not in item.metadata for item in first.units)
    assert all("text" not in item.metadata for item in first.relations)
    assert all(item.evidence for item in first.units)
    assert all(item.evidence for item in first.relations)


def test_rule_linker_kinds_align_with_generic_recommendation_obligations() -> None:
    inputs = (
        _input(1, "The current state of retrieval is a 70 percent baseline."),
        _input(2, "Retrieval failed on diffuse evidence."),
        _input(3, "An unresolved retrieval issue remains open."),
        _input(4, "Retrieval depends on exact source provenance."),
    )

    result = RuleBasedDiscourseLinker().link("artifact-recommend", inputs)

    assert [item.kind for item in result.units] == [
        "current_state",
        "failure",
        "issue",
        "dependency",
    ]
