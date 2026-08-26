from __future__ import annotations

import json

import pytest

from memory_condense.eval._artifact_json import canonical_json_bytes


def test_canonical_json_bytes_is_exact_sorted_utf8() -> None:
    value = {"z": [3, {"\u03b2": True}], "a": "caf\u00e9", "n": 1.25}

    assert canonical_json_bytes(value) == (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_bytes_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"value": value})
