from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass, replace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.fast_cav_links import (
    FAST_CAV_LINK_COMPLEXITY,
    FAST_CAV_MAX_CONCEPT_LINKS_PER_EVIDENCE,
    FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT,
    FastCAVLinkError,
    build_fast_cav_concepts,
    build_fast_cav_link_receipt,
)
from memory_condense.search.fusion.tensor_identity import canonical_float32_tensor


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _concepts(count: int = 2):
    return build_fast_cav_concepts(
        bank_identity_sha256=_digest("bank"),
        artifact_file_sha256s=tuple(_digest(f"artifact-{index}") for index in range(count)),
        tensor_keys=tuple(f"concept_{index}.layer_2" for index in range(count)),
    )


def _receipt():
    return build_fast_cav_link_receipt(
        packet_identity_sha256=_digest("packet"),
        router_runtime_identity_sha256=_digest("runtime"),
        router_bank_identity_sha256=_digest("bank"),
        concepts=_concepts(),
        evidence_ids=("e-0", "e-1", "e-2"),
        source_ids=("source-0", "source-1", "source-2"),
        evidence_text_sha256s=tuple(_digest(f"text-{index}") for index in range(3)),
        extraction_attention=((0.1, 0.7, 0.2), (0.6, 0.3, 0.1)),
        reinjection_attention=((0.8, 0.2), (0.4, 0.6), (0.5, 0.5)),
    )


def _contains_tensor_like(value: object) -> bool:
    if hasattr(value, "detach") or hasattr(value, "numpy"):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_tensor_like(getattr(value, row.name)) for row in fields(value))
    if isinstance(value, (tuple, list, dict)):
        children = value.values() if isinstance(value, dict) else value
        return any(_contains_tensor_like(child) for child in children)
    return False


def test_two_pass_receipt_preserves_exact_ranked_links_and_canonical_hashes():
    receipt = _receipt()

    assert receipt.complexity_contract == FAST_CAV_LINK_COMPLEXITY
    assert receipt.extraction_shape == (2, 3)
    assert receipt.reinjection_shape == (3, 2)
    assert receipt.rectangular_route_cell_count == 12
    assert receipt.evidence_pair_matrix_constructed is False
    assert receipt.evidence_pair_matrix_cell_count == 0
    assert receipt.extraction_matrix_sha256 == canonical_float32_tensor(
        ((0.1, 0.7, 0.2), (0.6, 0.3, 0.1)),
        label="expected extraction",
    ).tensor_sha256
    assert receipt.reinjection_matrix_sha256 == canonical_float32_tensor(
        ((0.8, 0.2), (0.4, 0.6), (0.5, 0.5)),
        label="expected reinjection",
    ).tensor_sha256

    extraction = receipt.extraction_links
    assert [(row.concept_ordinal, row.evidence_ordinal) for row in extraction] == [
        (0, 1),
        (0, 2),
        (0, 0),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert extraction[0].evidence_id == "e-1"
    assert extraction[0].source_id == "source-1"
    assert extraction[0].weight_sha256 == canonical_float32_tensor(
        (extraction[0].weight,),
        label="expected scalar",
        retain_values=False,
    ).tensor_sha256

    reinjection = receipt.reinjection_links
    assert [(row.evidence_ordinal, row.concept_ordinal) for row in reinjection] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (2, 0),
        (2, 1),
    ]
    assert reinjection[4].rank == 1  # exact ties retain lower concept ordinal
    assert receipt.retained_token_id_count == 0
    assert receipt.retained_tensor_bytes == 0
    assert receipt.persisted_token_state_bytes == 0
    assert not _contains_tensor_like(receipt)


def test_link_populations_are_bounded_while_route_work_remains_two_k_n():
    concept_count = 6
    evidence_count = 7
    receipt = build_fast_cav_link_receipt(
        packet_identity_sha256=_digest("wide-packet"),
        router_runtime_identity_sha256=_digest("wide-runtime"),
        router_bank_identity_sha256=_digest("bank"),
        concepts=_concepts(concept_count),
        evidence_ids=tuple(f"e-{index}" for index in range(evidence_count)),
        source_ids=tuple(f"source-{index}" for index in range(evidence_count)),
        evidence_text_sha256s=tuple(
            _digest(f"wide-text-{index}") for index in range(evidence_count)
        ),
        extraction_attention=tuple(
            tuple(1.0 / evidence_count for _ in range(evidence_count))
            for _ in range(concept_count)
        ),
        reinjection_attention=tuple(
            tuple(1.0 / concept_count for _ in range(concept_count))
            for _ in range(evidence_count)
        ),
    )

    assert len(receipt.extraction_links) == (
        concept_count * FAST_CAV_MAX_EVIDENCE_LINKS_PER_CONCEPT
    )
    assert len(receipt.reinjection_links) == (
        evidence_count * FAST_CAV_MAX_CONCEPT_LINKS_PER_EVIDENCE
    )
    assert receipt.rectangular_route_cell_count == 2 * concept_count * evidence_count
    assert [row.evidence_ordinal for row in receipt.extraction_links[:4]] == [0, 1, 2, 3]
    assert [row.concept_ordinal for row in receipt.reinjection_links[:4]] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("extraction", "reinjection", "match"),
    [
        (
            ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
            ((0.5, 0.5),) * 3,
            "KxN/NxK",
        ),
        (
            ((0.1, 0.1, 0.1), (0.6, 0.3, 0.1)),
            ((0.5, 0.5),) * 3,
            "softmax-normalized",
        ),
    ],
)
def test_rejects_non_rectangular_contract_or_non_probability_rows(
    extraction,
    reinjection,
    match: str,
):
    with pytest.raises(FastCAVLinkError, match=match):
        build_fast_cav_link_receipt(
            packet_identity_sha256=_digest("packet"),
            router_runtime_identity_sha256=_digest("runtime"),
            router_bank_identity_sha256=_digest("bank"),
            concepts=_concepts(),
            evidence_ids=("e-0", "e-1", "e-2"),
            source_ids=("source-0", "source-1", "source-2"),
            evidence_text_sha256s=tuple(_digest(f"text-{index}") for index in range(3)),
            extraction_attention=extraction,
            reinjection_attention=reinjection,
        )


def test_rejects_tampered_scalar_provenance_and_n_by_n_claims():
    receipt = _receipt()
    first = receipt.extraction_links[0]

    with pytest.raises(FastCAVLinkError, match="canonical bytes"):
        replace(first, weight=0.25, link_sha256="")
    tampered = (replace(first, source_id="wrong-source", link_sha256=""), *receipt.extraction_links[1:])
    with pytest.raises(FastCAVLinkError, match="exact provenance"):
        replace(
            receipt,
            extraction_links=tampered,
            extraction_links_sha256=identity_sha256(
                [row.identity_payload() for row in tampered]
            ),
            link_receipt_sha256="",
        )
    with pytest.raises(FastCAVLinkError, match="N x N"):
        replace(receipt, evidence_pair_matrix_constructed=True, link_receipt_sha256="")
    with pytest.raises(FastCAVLinkError, match="exactly zero"):
        replace(receipt, evidence_pair_matrix_cell_count=False, link_receipt_sha256="")
