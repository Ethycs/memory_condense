from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import Turn
from memory_condense.search.section_summary import RawSectionSpan
from tools.compile_spine_semantic_index_v2 import restore_with_token_accounting


def fixture():
    turn = Turn(turn_id="opaque-turn", source_id="opaque-source", role="user",
        text="Hello world.", created_at=datetime(2026, 9, 10, tzinfo=timezone.utc))
    fragments = [SimpleNamespace(span=RawSectionSpan.from_turn(turn, start_char=start, end_char=end),
        text=turn.text[start:end]) for start, end in ((0, 2), (2, len(turn.text)))]
    namespace = {"turn_count": 1, "source_count": 1,
                 "raw_token_proxy": sum(count_tokens(f.text) for f in fragments)}
    assert namespace["raw_token_proxy"] != count_tokens(turn.text)
    return turn, fragments, namespace


def test_nonadditive_token_counts_do_not_reject_exact_original_bytes():
    turn, fragments, namespace = fixture()
    turns, accounting = restore_with_token_accounting(fragments, namespace)
    assert turns == (turn,)
    assert accounting["fragment_raw_token_proxy"] == namespace["raw_token_proxy"]
    assert accounting["whole_turn_raw_token_proxy"] == count_tokens(turn.text)
    assert accounting["whole_minus_fragment_tokens"] != 0
    assert accounting["nonadditive_turns"][0]["fragment_count"] == 2
    assert accounting["token_difference_tolerance_used"] is False


@pytest.mark.parametrize("field", ["turn_count", "source_count", "raw_token_proxy"])
def test_real_population_changes_are_still_rejected(field):
    _, fragments, namespace = fixture()
    namespace[field] += 1
    with pytest.raises(ValueError, match="population changed"):
        restore_with_token_accounting(fragments, namespace)


@pytest.mark.parametrize("change", ["bytes", "gap", "duplicate", "order"])
def test_token_accounting_does_not_relax_exact_reconstruction(change):
    _, fragments, namespace = fixture()
    if change == "bytes":
        fragments[1].text += " changed"
    elif change == "gap":
        fragments.pop(0)
    elif change == "duplicate":
        fragments.append(fragments[1])
    else:
        fragments.reverse()
    with pytest.raises(ValueError):
        restore_with_token_accounting(fragments, namespace)
