from datetime import datetime, timezone
import json

import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.spine_batch_summary import RawSummaryFragment
from memory_condense.search.spine_source_admission import admit_source_bound_summaries


def fragment(i):
    turn = Turn(turn_id=str(i), source_id="source", role="user" if i == 0 else "assistant",
                text=f"  original raw {i} τ\r\n", created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
    return RawSummaryFragment(RawSectionSpan.from_turn(turn), turn.text)


def test_bad_generated_quote_is_diagnostic_with_complete_exact_source_preserved():
    fragments = [fragment(0), fragment(1)]
    rows = [{"label": f"T{i}", "summary": f"summary {i}", "support": ["a nonverbatim model quote"]}
            for i in range(2)]
    batch = admit_source_bound_summaries(json.dumps({"atoms": rows}), fragments, compiler_identity="terra")
    assert [a.summary for a in batch.atoms] == [r["summary"] for r in rows]
    assert [a.spans for a in batch.atoms] == [(f.span,) for f in fragments]
    assert all(r["failures"] == [{"quote_index": 0, "reason": "quote_not_exact"}] for r in batch.quote_diagnostics)
    assert not batch.summary_entailment_verified
    assert "a nonverbatim model quote" not in repr([a.summary for a in batch.atoms])


@pytest.mark.parametrize("change", ["missing", "foreign", "role_field", "over_budget"])
def test_admission_still_rejects_missing_or_changed_attribution_and_invalid_summary(change):
    fragments = [fragment(0), fragment(1)]
    rows = [{"label": f"T{i}", "summary": "summary", "support": []} for i in range(2)]
    if change == "missing":
        rows.pop()
    elif change == "foreign":
        rows[1]["label"] = "T0"
    elif change == "role_field":
        rows[1]["role"] = "user"
    else:
        rows[1]["summary"] = " word" * 200
    with pytest.raises(ValueError):
        admit_source_bound_summaries(json.dumps({"atoms": rows}), fragments, compiler_identity="terra")
