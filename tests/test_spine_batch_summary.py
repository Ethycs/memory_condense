from datetime import datetime, timezone
import json

import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan
from memory_condense.search.spine_batch_summary import (
    RawSummaryFragment, batch_messages, pack_summary_batches, parse_batch_summaries,
)


def fragment(i, source="source", role="user"):
    turn = Turn(turn_id=str(i), source_id=source, role=role,
                text=f"  raw fragment {i} τ\r\n", created_at=datetime(2026, 9, 9, tzinfo=timezone.utc))
    return RawSummaryFragment(RawSectionSpan.from_turn(turn), turn.text)


def test_batching_preserves_every_byte_and_source_order():
    raw = (fragment(0), fragment(1, role="assistant"), fragment(2), fragment(3, "another"))
    batches = pack_summary_batches(raw, max_atoms=2)
    assert tuple(f for batch in batches for f in batch) == raw
    assert [len(batch) for batch in batches] == [2, 1, 1]
    messages = batch_messages(batches[0])
    items = json.loads(messages[1]["content"])["fragments"]
    assert [i["fragment"] for i in items] == [f.text for f in raw[:2]]
    assert [i["speaker"] for i in items] == ["user", "assistant"]
    with pytest.raises(ValueError, match="exceeds"):
        pack_summary_batches(raw, max_prompt_tokens=1)
    with pytest.raises(ValueError, match="unique"):
        pack_summary_batches([raw[0], raw[0]])


def test_summary_support_cannot_cross_speaker_or_fragment():
    raw = [fragment(0), fragment(1, role="assistant")]
    rows = [{"label": f"T{i}", "summary": f"summary {i}", "support": [f.text]} for i, f in enumerate(raw)]
    atoms = parse_batch_summaries(json.dumps({"atoms": rows}), raw, compiler_identity="terra")
    assert [a.spans[0] for a in atoms] == [f.span for f in raw]
    assert [a.summary for a in atoms] == ["summary 0", "summary 1"]
    assert all("raw fragment" not in a.summary for a in atoms)
    rows[1]["support"] = [raw[0].text]
    with pytest.raises(ValueError, match="own fragment"):
        parse_batch_summaries(json.dumps({"atoms": rows}), raw, compiler_identity="terra")


@pytest.mark.parametrize("change", ["drop", "reorder", "duplicate", "oversize"])
def test_partial_or_misattributed_batch_is_not_published(change):
    raw = [fragment(0), fragment(1)]
    rows = [{"label": f"T{i}", "summary": "summary", "support": [f.text]} for i, f in enumerate(raw)]
    if change == "drop":
        rows.pop()
    elif change == "reorder":
        rows.reverse()
    elif change == "duplicate":
        rows[1] = rows[0]
    else:
        rows[0]["summary"] = " long" * 200
    with pytest.raises(ValueError):
        parse_batch_summaries(json.dumps({"atoms": rows}), raw, compiler_identity="terra")


def test_mutated_raw_fragment_is_rejected():
    raw = fragment(0)
    with pytest.raises(ValueError, match="digest"):
        RawSummaryFragment(raw.span, raw.text.strip())
