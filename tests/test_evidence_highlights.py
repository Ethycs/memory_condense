import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.packing.evidence_highlights import HighlightCandidate, render_evidence_highlights


def test_highlights_are_bounded_verbatim_prefixes_with_original_citations():
    rows = [HighlightCandidate(f"G{i}", "user", "2024-01-01", "A long exact raw statement. " * 30) for i in range(1, 10)]
    result = render_evidence_highlights(rows, max_rows=3, max_tokens=180)
    assert 0 < len(result.bindings) <= 3
    assert count_tokens(result.text) <= 180
    assert "partial" in result.text
    for binding in result.bindings:
        raw = next(r for r in rows if r.citation == binding["citation"])
        assert raw.text.startswith(binding["snippet"])
        assert binding["snippet"] in result.text


def test_no_partial_reference_is_emitted_when_the_budget_cannot_fit_one():
    row = HighlightCandidate("G1", "user", "2024-01-01", "Exact raw.")
    result = render_evidence_highlights([row], max_tokens=1)
    assert result.text == ""
    assert result.bindings == ()


def test_invalid_citations_and_roles_fail_before_presentation():
    with pytest.raises(ValueError, match="G labels"):
        render_evidence_highlights([HighlightCandidate("G1>bad", "user", "2024-01-01", "raw")])
    with pytest.raises(ValueError, match="speaker"):
        render_evidence_highlights([HighlightCandidate("G1", "unknown", "2024-01-01", "raw")])


def test_unicode_boundaries_never_create_nonverbatim_evidence():
    raw = "👋 café 漢字 " * 10
    result = render_evidence_highlights([HighlightCandidate("G1", "user", "2024-01-01", raw)], snippet_tokens=3)
    assert all(raw.startswith(b["snippet"]) for b in result.bindings)
