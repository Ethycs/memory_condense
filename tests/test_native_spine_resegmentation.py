import copy

import pytest

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.search.native_spine_memory import validate_body_summaries
from memory_condense.search.native_spine_resegmentation import reconcile_sections, subdivide
from memory_condense.search.native_spine_summary import fragment_body
from tools.resegment_native_spine_repairs import collect


def body():
    return {"turns": [{"role": "user", "text": "I visited the café 🌍 yesterday. " * 80},
                      {"role": "assistant", "text": "Consider a return visit."}]}


def response(fragments):
    return canonical_json({"atoms": [{"label": f"T{i}", "summary": "word " * 150 if i == 0 else "A valid unchanged summary."}
                                     for i in range(len(fragments))]})


def atoms(pieces):
    return [{"pointer": p.pointer(), "summary": "User reports visiting a café yesterday."} for p in pieces]


def test_unicode_subdivision_preserves_nonzero_absolute_turn_offsets():
    original = fragment_body(body(), token_cap=300)[1]
    assert original.start_char > 0
    parts = subdivide(original)
    assert len(parts) >= 2
    assert parts[0].start_char == original.start_char
    assert parts[-1].end_char == original.end_char
    assert "".join(p.text for p in parts) == original.text
    assert all(p.pointer()["token_count"] <= 512 for p in parts)
    assert all((p.body_sha256, p.turn_ordinal, p.role, p.turn_text_sha256) ==
               (original.body_sha256, original.turn_ordinal, original.role, original.turn_text_sha256) for p in parts)


def test_resegmented_body_retains_full_raw_coverage_and_valid_summaries():
    raw = body()
    fs = fragment_body(raw)
    parts = subdivide(fs[0])
    repaired = reconcile_sections(response(fs), fs, {0: atoms(parts)})
    assert len(repaired) == len(fs) + len(parts) - 1
    assert repaired[-1] == {"pointer": fs[-1].pointer(), "summary": "A valid unchanged summary."}
    validate_body_summaries(raw, repaired)


@pytest.mark.parametrize("defect", ["missing", "reordered", "wrong_hash"])
def test_subdivision_cannot_hide_a_hole_or_change_evidence(defect):
    fs = fragment_body(body())
    replacements = atoms(subdivide(fs[0]))
    if defect == "missing":
        replacements.pop()
    elif defect == "reordered":
        replacements.reverse()
    else:
        replacements[0]["pointer"]["span_text_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        reconcile_sections(response(fs), fs, {0: replacements})


def test_one_pending_repair_does_not_hide_other_complete_original_batches():
    first = fragment_body(body())[:1]
    second = fragment_body({"turns": [{"role": "user", "text": "A separate numeric list. " * 80}]})
    one, two = subdivide(first[0]), subdivide(second[0])
    pieces = (*one, *two)
    owners = ((0, 0),) * len(one) + ((1, 0),) * len(two)
    originals = {0: (None, None, first, response(first), {}, (0,)),
                 1: (None, None, second, response(second), {}, (0,))}
    accepted = dict(enumerate(atoms(one)))
    complete, pending = collect(originals, {}, pieces, owners, accepted)
    assert set(complete) == {0}
    assert pending == [{"ordinal": 1, "unresolved_original_atom_indices": [0]}]
    with pytest.raises(ValueError):
        collect(originals, {}, pieces, owners, {len(pieces): atoms(one)[0]})
