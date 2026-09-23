import pytest

from tools.repair_user_spine_atoms import normalize_support


def test_only_exact_wrapper_removal_and_lossless_quote_splitting():
    raw = "A literal statement about a user request. " + "Unicode τ wording " * 20
    quote = raw[:raw.index("Unicode")].strip()
    result, operations = normalize_support({"summary": "Immutable summary.", "support": ['"' + quote + '"']}, raw)
    assert result == {"summary": "Immutable summary.", "support": [quote]}
    assert operations[0]["original"] != operations[0]["unwrapped"]
    exact_long = "Unicode τ wording " * 8
    result, operations = normalize_support({"summary": "Immutable summary.", "support": [exact_long]}, raw)
    assert "".join(result["support"]) == exact_long


@pytest.mark.parametrize("quote", ['"Invented statement."', "A paraphrase", '“A literal ... user request.”'])
def test_normalization_does_not_repair_semantic_or_noncontiguous_quotes(quote):
    with pytest.raises(ValueError, match="exact source substring"):
        normalize_support({"summary": "Same", "support": [quote]}, "A literal statement about a user request.")
