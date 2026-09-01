from __future__ import annotations

from tools import run_reduced_oracle_source_assay as assay


def _source(turns: list[tuple[str, str]], *, date: str = "2023/04/01") -> dict[str, object]:
    return {
        "date": date,
        "source_id": "hidden-source-id",
        "turns": [{"role": role, "content": content} for role, content in turns],
    }


def test_fit_turns_preserves_complete_small_oracle_source() -> None:
    selected, audit = assay._fit_turns(  # noqa: SLF001
        "[Question asked at 2023/04/05]\nWhich bike did I service?",
        [_source([("user", "I serviced my road bike."), ("assistant", "Great.")])],
    )

    assert audit["full_source_retained"] is True
    assert audit["selected_turn_count"] == audit["total_turn_count"] == 2
    assert audit["dropped_turn_count"] == 0
    assert audit["fitted_prompt_token_proxy"] + assay.ANSWER_OUTPUT_RESERVE <= 8_000
    assert "hidden-source-id" not in "\n".join(row["rendered"] for row in selected)


def test_fit_turns_uses_question_only_whole_turn_packing(monkeypatch) -> None:
    monkeypatch.setattr(assay, "ANSWER_PROMPT_CAP", 400)
    sources = [
        _source(
            [
                ("user", "Unrelated filler " * 60),
                ("assistant", "More filler " * 60),
                ("user", "The exact museum was Art Cube Gallery."),
            ]
        )
    ]

    selected, audit = assay._fit_turns(  # noqa: SLF001
        "[Question asked at 2023/04/05]\nWhich museum did I visit?",
        sources,
    )

    rendered = "\n".join(row["rendered"] for row in selected)
    assert audit["full_source_retained"] is False
    assert audit["whole_turns_only"] is True
    assert "Art Cube Gallery" in rendered
    assert audit["fitted_prompt_token_proxy"] <= 400


def test_remaining_population_is_exact_miss27_minus_three_rescues() -> None:
    assert len(assay.REMAINING_ORDINALS) == 24
    assert set(assay.REMAINING_ORDINALS).isdisjoint({17, 74, 87})
    assert tuple(sorted(assay.REMAINING_ORDINALS)) == assay.REMAINING_ORDINALS
