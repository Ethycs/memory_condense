import pytest

from memory_condense.search.native_spine_summary import fragment_body
from tools.finish_native_spine_section_repairs import expand


def test_refinement_sends_only_unresolved_text_and_keeps_accepted_sections_exact():
    pieces = fragment_body({"turns": [{"role": "assistant", "text": "Prices in rupees: ₹1,500–₹11,000. " * 30}]}, token_cap=160)
    owners = ((136, 15),) * len(pieces)
    original = {i: {"pointer": p.pointer(), "summary": "Accepted price summary."}
                for i, p in enumerate(pieces) if i != 1}
    result, next_owners, ready, needed = expand(pieces, owners, original)
    assert "".join(p.text for p in result) == "".join(p.text for p in pieces)
    assert "".join(result[i].text for i in needed) == pieces[1].text
    assert len(needed) >= 2
    assert list(ready.values()) == list(original.values())
    assert next_owners == ((136, 15),) * len(result)
    assert all(ready[i]["pointer"] == result[i].pointer() for i in ready)


@pytest.mark.parametrize("defect", ["foreign_position", "changed_pointer"])
def test_refinement_cannot_relabel_a_saved_source_section(defect):
    pieces = fragment_body({"turns": [{"role": "user", "text": "A source passage. " * 50}]})
    accepted = {0: {"pointer": pieces[0].pointer(), "summary": "Accepted summary."}}
    if defect == "foreign_position":
        accepted[1] = accepted[0]
    else:
        accepted[0]["pointer"]["start_char"] += 1
    with pytest.raises(ValueError):
        expand(pieces, ((0, 0),), accepted)
