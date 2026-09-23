from dataclasses import replace

import pytest

from memory_condense.application.atomic_section_fallback import AtomicSectionFallback
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.search.native_spine_memory import materialize_history
from memory_condense.search.native_spine_summary import body_identity, fragment_body
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary


def fixture(repetitions=2000, *, day="2026-01-02", short=True):
    turns = [{"role": "user", "text": "Background details. " * repetitions + "Needle END."}]
    if short:
        turns.insert(0, {"role": "user", "text": "Keep this baseline evidence."})
    body = {"turns": turns}
    fs = fragment_body(body, token_cap=128)
    cached = [{"pointer": f.pointer(), "summary": "Needle END detail." if f.text.endswith("END.")
               else "Background material."} for f in fs]
    source = {"original_session_ordinal": 0, "session_id": "fixture", "created_at": day + "T00:00:00+00:00",
              "metadata_text": "source boundary", "body_sha256": body_identity(body), "dataset_origin": "M"}
    source["occurrence_id"] = identity_sha256(source)
    history = materialize_history([source], load_body=lambda _: body, load_summaries=lambda _: cached,
                                  compiler_identity="fixture")
    big = tuple(a for a, f in zip(history.atoms, fs, strict=True) if f.turn_ordinal == len(turns)-1)
    section = SectionSummary("whole-exchange", big[0].source_id, "Whole exchange about Needle END.",
                             tuple(s for a in big for s in a.spans), "fixture")
    selected = (history.atoms[0], section) if short else (section,)
    query = "Needle END"
    primary_index = SectionSummaryIndex(selected)
    primary = SectionRoutePlan(primary_index.receipt_sha256, quote_sha256(query),
        tuple(SectionRoute(s, 1/(i+1), ()) for i, s in enumerate(selected)), len(selected), None, len(selected))
    atomic_index = SectionSummaryIndex(history.atoms)
    atom_plan = atomic_index.route(query, max_sections=32)
    assert atom_plan.routes[0].section == big[-1]
    return history, primary, atomic_index, atom_plan, big[-1]


def test_oversized_exchange_recovers_its_exact_tail_without_displacing_baseline():
    history, primary, index, atoms, tail = fixture()
    reads = []
    def load(turn_id):
        reads.append(turn_id)
        return history.get_turn(turn_id)
    result = AtomicSectionFallback(index).hydrate(primary, atoms, load_turn=load)
    assert len(result.baseline.sections) == 1
    assert result.hydration.sections[0] == result.baseline.sections[0]
    assert result.added_atomic_ids == (tail.section_id,)
    recovered = result.hydration.sections[-1].evidence[0]
    turn = history.get_turn(recovered.span.turn_id)
    assert recovered.text == turn.text[recovered.span.start_char:recovered.span.end_char]
    assert recovered.text.endswith("Needle END.")
    assert result.hydration.context_token_count <= 3072 and len(reads) == 2
    # Recovering one atom does not erase the rejected whole-section diagnostic.
    assert result.hydration.requires_raw_fallback
    assert result.atomic_owners == ((tail.section_id, ("whole-exchange",)),)


@pytest.mark.parametrize("budget,added", [(1, 0), (2, 1)])
def test_primary_and_fallback_share_the_raw_inspection_budget(budget, added):
    history, primary, index, atoms, _ = fixture()
    reads = []
    def load(turn_id):
        reads.append(turn_id)
        return history.get_turn(turn_id)
    result = AtomicSectionFallback(index).hydrate(primary, atoms, load_turn=load, max_raw_spans=budget)
    assert len(result.added_atomic_ids) == added
    assert result.attempted_raw_spans == budget
    assert len(reads) == 1 + added
    assert result.hydration.sections[:1] == result.baseline.sections


def test_framing_overflow_reuses_the_already_read_raw_turn():
    history, primary, index, atoms, _ = fixture(70, short=False)
    raw_tokens = sum(s.token_count for s in primary.routes[0].section.spans)
    reads = []
    def load(turn_id):
        reads.append(turn_id)
        return history.get_turn(turn_id)
    result = AtomicSectionFallback(index).hydrate(primary, atoms, load_turn=load, max_context_tokens=raw_tokens)
    assert not result.baseline.sections and result.baseline.diagnostics[0].reason == "context_budget"
    assert result.added_atomic_ids
    assert len(reads) == result.hydration.raw_turn_read_count == 1
    assert result.hydration.context_token_count <= raw_tokens


def test_a_changed_raw_source_is_not_bypassed_by_atomic_fallback():
    history, primary, index, atoms, _ = fixture(70)
    big_id = primary.routes[-1].section.spans[0].turn_id
    reads = []
    def load(turn_id):
        reads.append(turn_id)
        turn = history.get_turn(turn_id)
        return turn.model_copy(update={"text": turn.text + " changed"}) if turn_id == big_id else turn
    result = AtomicSectionFallback(index).hydrate(primary, atoms, load_turn=load)
    assert not result.added_atomic_ids
    assert result.hydration == result.baseline
    assert result.baseline.diagnostics[0].reason == "raw_turn_identity_changed"
    assert len(reads) == 2


def test_query_mismatch_rejected_before_raw_reads_and_foreign_occurrences_are_ignored():
    history, primary, index, atoms, _ = fixture()
    def forbidden(_):
        raise AssertionError("invalid candidates must not read raw text")
    with pytest.raises(ValueError, match="candidate query"):
        AtomicSectionFallback(index).hydrate(primary, replace(atoms, query_sha256=quote_sha256("other"), receipt_sha256=""),
                                             load_turn=forbidden)
    foreign, _, _, _, foreign_tail = fixture(day="2026-02-02")
    mixed = SectionSummaryIndex((*history.atoms, *foreign.atoms))
    foreign_plan = SectionRoutePlan(mixed.receipt_sha256, primary.query_sha256,
                                    (SectionRoute(foreign_tail, 1, ()),), 1, None, 1)
    reads = []
    def load(turn_id):
        reads.append(turn_id)
        return history.get_turn(turn_id)
    result = AtomicSectionFallback(mixed).hydrate(primary, foreign_plan, load_turn=load)
    assert not result.added_atomic_ids and len(reads) == 1
