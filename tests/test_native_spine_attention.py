from datetime import datetime, timezone

import pytest

from memory_condense.domain.schemas import Turn
from memory_condense.search.episodes.attention_hierarchy import compile_attention_atoms
from memory_condense.search.episodes.user_spine_hierarchy import compile_user_spine_exchanges
from memory_condense.search.spine_summary_reuse import ReusingSpineSummarizer
from tests.test_user_spine_hierarchy import make_exchanges
from tools.compile_native_spine_attention import cache_method, user_windows


def rows(count, *, summary="A user plans a furniture purchase."):
    turns = [Turn(turn_id=f"turn-{i}", source_id="actual-occurrence", role="user",
                  created_at=datetime(2026, 1, 2, tzinfo=timezone.utc), text=f"RAW_CANARY_{i}")
             for i in range(count)]
    atoms = compile_attention_atoms(turns, summarize_raw=lambda _: summary,
                                    summarizer_identity="fixture", max_summary_tokens=256)
    return compile_user_spine_exchanges(atoms, summarize=ReusingSpineSummarizer(lambda _: None),
                                         summarizer_identity="fixture", max_channel_tokens=256)


@pytest.mark.parametrize("population", [1, 7, 8, 9, 19])
def test_bounded_windows_preserve_every_adjacent_user_pair(population):
    exchanges = rows(population)
    windows = user_windows(exchanges)
    assert windows[0]["start_exchange"] == 0 and windows[-1]["end_exchange"] == population
    assert all(1 <= len(w["texts"]) <= 8 for w in windows)
    pairs = [(i-1, i) for w in windows for i in range(w["start_exchange"]+1, w["end_exchange"])]
    assert pairs == list(zip(range(population-1), range(1, population)))
    for window in windows:
        assert window["texts"] == [e.user_spine for e in exchanges[window["start_exchange"]:window["end_exchange"]]]
        assert "RAW_CANARY" not in str(window) and "2026-01-02" not in str(window)


def test_attached_machine_text_cannot_control_the_attention_signal():
    _, _, _, exchanges = make_exchanges()
    observed = [text for w in user_windows(exchanges) for text in w["texts"]]
    assert observed == ["Unowned prelude.", "orchard harvest", "orchard irrigation", "observatory reservations"]
    assert "machine" not in str(observed) and "RAW_CANARY" not in str(observed)
    with pytest.raises(ValueError, match="cross source occurrences"):
        user_windows((*exchanges, *rows(1)))


def test_attention_cannot_silently_clip_an_oversized_summary():
    with pytest.raises(ValueError, match="truncated"):
        user_windows(rows(1, summary="Long summary. " * 55))
    with pytest.raises(TypeError):
        user_windows(["raw text"])


def test_attention_method_is_reusable_independently_of_source_snapshot(tmp_path):
    one = cache_method(tmp_path)
    two = cache_method(tmp_path)
    assert one.sha256 == two.sha256
    assert not {"sources_sha256", "occurrence_id", "transcript_date"} & set(one.payload)
    assert one.payload["prefix_layers"] == 6 and one.payload["attention_layer"] == 5
    assert one.payload["span_token_cap"] == 128 and one.payload["raw_inputs_to_qwen"] is False
