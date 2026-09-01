from __future__ import annotations

import copy

import pytest

from tools.matched_eval.artifacts import read_sealed_json
from tools.run_locked_numeric_evidence_assay import (
    DEFAULT_SOURCE,
    EXPECTED_SOURCE_SHA256,
    LockedNumericEvidenceAssayError,
    _load_exact_json,
    build_frozen_population,
    freeze_locked_numeric_population,
)
from tools.run_locked_numeric_evidence_assay_v2 import (
    build_frozen_population_v2,
    freeze_locked_numeric_population_v2,
)


V1_BYTE_SHA256 = (
    "1199b1b0ce9880136a4033b0709ecc45bc32d53f30cfecf06581a9f9790bb69c"
)
V1_POPULATION_SHA256 = (
    "9a53d5c6f576be3298b469ba8e9128a1020ec4b8433d0b513693e44199e4d957"
)
V2_BYTE_SHA256 = (
    "108cdaf00488ecb5fdf205d45ec5ea452312369975e11556ec9361d36b5a6952"
)
V2_POPULATION_SHA256 = (
    "87bb0862e00727be3387fdab39854500fbe74165b84290c3147d83b355b17038"
)


def test_v1_population_publishes_a_readable_canonical_artifact(tmp_path) -> None:
    target = tmp_path / "numeric-v1.json"

    result = freeze_locked_numeric_population(DEFAULT_SOURCE, target)
    sealed = read_sealed_json(target)

    assert sealed.sha256 == V1_BYTE_SHA256
    assert sealed.payload == result
    assert sealed.payload["population_sha256"] == V1_POPULATION_SHA256
    assert sealed.payload["status_counts"] == {
        "conflicted": 6,
        "insufficient": 64,
        "supported": 2,
    }
    assert target.with_name(f"{target.name}.sha256").is_file()


def test_v2_population_is_sealed_and_adds_only_two_supported_rows(tmp_path) -> None:
    target = tmp_path / "numeric-v2.json"

    result = freeze_locked_numeric_population_v2(DEFAULT_SOURCE, target)
    sealed = read_sealed_json(target)

    assert sealed.sha256 == V2_BYTE_SHA256
    assert sealed.payload == result
    assert sealed.payload["population_sha256"] == V2_POPULATION_SHA256
    assert sealed.payload["status_counts"] == {
        "conflicted": 6,
        "insufficient": 62,
        "supported": 4,
    }
    supported = {
        row["ordinal"]: row["reconciliation"]
        for row in sealed.payload["ordered_rows"]
        if row["reconciliation"]["status"] == "supported"
    }
    assert set(supported) == {34, 44, 87, 90}
    assert supported[34]["numeric_result"] == 3
    assert supported[34]["used_handle_ids"] == ["H001", "H002", "H003"]
    assert len(supported[34]["contributions"]) == 3
    assert supported[87]["numeric_result"] == 5
    assert supported[87]["used_handle_ids"] == ["H001", "H002", "H003", "H004"]
    assert len(supported[87]["contributions"]) == 3
    assert supported[87]["deduplicated_item_count"] == 1


@pytest.mark.parametrize(
    "builder",
    [build_frozen_population, build_frozen_population_v2],
)
def test_ordinary_provider_input_must_match_adapter_transform_seal(builder) -> None:
    source, source_sha = _load_exact_json(DEFAULT_SOURCE)
    assert source_sha == EXPECTED_SOURCE_SHA256
    changed = copy.deepcopy(source)
    row = next(
        value
        for value in changed["physical_prompt_rows"]
        if value.get("adapter_prompt_transform", {}).get("provider_input_sha256")
    )
    row["adapter_prompt_transform"]["provider_input_sha256"] = "0" * 64

    with pytest.raises(
        LockedNumericEvidenceAssayError,
        match="ordinary typed transform provider-input SHA-256 mismatch",
    ):
        builder(changed, source_artifact_sha256=source_sha)
