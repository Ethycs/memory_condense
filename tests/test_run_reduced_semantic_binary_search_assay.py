from __future__ import annotations

import argparse
import base64
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)
from tools import run_reduced_semantic_binary_search_assay as arm
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.full_store_slot_closure import LocalCitationBinding


def _sha(label: str) -> str:
    return quote_sha256(label)


def _composition_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{} for _ in range(max(arm.TARGET_ORDINALS) + 1)]
    bodies = {
        42: "Which university did I attend for the Graph Memory conference?",
        65: "Which online communities did I join for photography and cooking?",
        74: "Which posture reset video did I save on YouTube?",
        79: "What were the two quoted prices for the replacement part?",
    }
    for ordinal, question in bodies.items():
        dated = f"[Question asked at 2026/08/27 12:00]\n{question}"
        rows[ordinal] = {
            "dated_question_sha256": quote_sha256(dated),
            "parent_prediction": f"parent {ordinal}",
            "provider_projection": {"provider_input": {"dated_question": dated}},
            "question_id": f"q{ordinal}",
            "question_sha256": quote_sha256(question),
        }
    return rows


class _FakeEmbedder:
    model_name = DEFAULT_MODEL_NAME
    model_revision = DEFAULT_MODEL_REVISION
    checkpoint_sha256 = BGE_M3_CHECKPOINT_SHA256

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    @property
    def execution_identity(self) -> dict[str, object]:
        return {
            "backend": "sentence-transformers.encode-v1",
            "batch_size": 32,
            "device": "cpu",
            "normalize_embeddings": False,
            "output_dtype": "float32",
        }

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        self.calls.append(tuple(texts))
        values = np.zeros((len(texts), DEFAULT_MODEL_DIM), dtype=np.float32)
        for ordinal in range(len(texts)):
            values[ordinal, ordinal % DEFAULT_MODEL_DIM] = 1.0
        return values


def test_query_vectors_flatten_all_real_facets_into_one_batch_and_round_trip(
    tmp_path: Path,
) -> None:
    rows = _composition_rows()
    embedder = _FakeEmbedder()

    payload = arm.build_query_vector_payload(rows, embedder)

    expected_facets = tuple(
        facet
        for ordinal in arm.TARGET_ORDINALS
        for facet in arm.residual.semantic_residual_query_facets(
            rows[ordinal]["provider_projection"]["provider_input"]["dated_question"]
        )
    )
    assert embedder.calls == [expected_facets]
    assert payload["embedding"]["offline_environment"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert payload["facet_count"] == len(expected_facets)
    assert payload["facet_count"] > arm.QUESTION_COUNT
    assert all(
        row["facet_count"] == len(row["facets"]) for row in payload["rows"]
    )

    artifact, _ = publish_sealed_json(tmp_path / arm.VECTOR_NAME, payload)
    loaded, decoded = arm.load_query_vectors(
        artifact.path,
        expected_sha256=artifact.sha256,
    )

    assert loaded.sha256 == artifact.sha256
    assert tuple(row[0]["ordinal"] for row in decoded) == arm.TARGET_ORDINALS
    assert sum(len(row[1]) for row in decoded) == len(expected_facets)


def test_query_vector_validator_rejects_tampered_facet_bytes() -> None:
    payload = arm.build_query_vector_payload(_composition_rows(), _FakeEmbedder())
    tampered = copy.deepcopy(payload)
    raw = base64.b64decode(tampered["rows"][0]["facets"][0]["vector_base64"])
    tampered["rows"][0]["facets"][0]["vector_base64"] = base64.b64encode(
        b"x" + raw[1:]
    ).decode("ascii")
    facet_body = dict(tampered["rows"][0]["facets"][0])
    facet_body.pop("facet_receipt_sha256")
    tampered["rows"][0]["facets"][0]["facet_receipt_sha256"] = identity_sha256(
        facet_body
    )
    row_body = dict(tampered["rows"][0])
    row_body.pop("row_receipt_sha256")
    tampered["rows"][0]["row_receipt_sha256"] = identity_sha256(row_body)
    artifact_body = dict(tampered)
    artifact_body.pop("vector_artifact_identity_sha256")
    tampered["vector_artifact_identity_sha256"] = identity_sha256(artifact_body)

    with pytest.raises(
        arm.ReducedSemanticBinarySearchAssayError,
        match="query-vector facet changed",
    ):
        arm._validate_query_vectors(  # noqa: SLF001
            SealedArtifact(Path("unused.json"), _sha("artifact"), tampered)
        )


def test_verified_vector_rows_rejects_replicated_question_vector_as_fake_facets() -> None:
    composition = _composition_rows()
    payload = arm.build_query_vector_payload(composition, _FakeEmbedder())
    artifact = SealedArtifact(Path("unused.json"), _sha("artifact"), payload)
    decoded = list(arm._validate_query_vectors(artifact))  # noqa: SLF001
    row, vectors = next(value for value in decoded if len(value[0]["facets"]) > 1)
    forged = copy.deepcopy(row)
    forged["facets"][1]["facet_text"] = forged["facets"][0]["facet_text"]
    decoded[0] = (forged, tuple(vectors[0] for _ in vectors))

    with pytest.raises(
        arm.ReducedSemanticBinarySearchAssayError,
        match="question/facet binding changed",
    ):
        arm._verified_vector_rows(decoded, composition)  # noqa: SLF001


def _closure_fixture(*, duplicate: bool):
    segment = _sha("segment")
    evidence_receipt = _sha("evidence")
    local_receipt = _sha("local")
    residual_binding_receipt = _sha("residual-binding")
    residual_item_receipt = _sha("residual-item")
    owner_item_receipt = _sha("owner-item")
    owner_binding_receipt = _sha("owner-binding")
    summary = "An exact retained memory sentence."
    evidence = SimpleNamespace(
        candidate_id=_sha("candidate"),
        cell_id="cell-0",
        receipt_sha256=evidence_receipt,
        segment_receipt_sha256=segment,
    )
    local = SimpleNamespace(candidate_id=evidence.candidate_id, receipt_sha256=local_receipt)
    residual_binding = SimpleNamespace(
        evidence_receipt_sha256=evidence_receipt,
        handle_id="H950001",
        local_source_locator_sha256=local_receipt,
        receipt_sha256=residual_binding_receipt,
    )
    residual_item = SimpleNamespace(
        handle_ids=("H950001",),
        receipt_sha256=residual_item_receipt,
        summary=summary,
    )
    residual_contribution = SimpleNamespace(
        bindings=(residual_binding,),
        parsed=SimpleNamespace(accepted_items=(residual_item,)),
    )
    if duplicate:
        visible_item = SimpleNamespace(
            handle_ids=("H100001",),
            receipt_sha256=owner_item_receipt,
            summary=summary,
        )
        visible_binding = SimpleNamespace(
            handle_id="H100001", receipt_sha256=owner_binding_receipt
        )
        exclusions = [
            {
                "duplicate_item_receipt_sha256": residual_item_receipt,
                "duplicate_mechanism_id": arm.residual.TYPED_ADAPTER_MECHANISM_ID,
                "operation_position": (
                    "after_independent_lane_admission_and_shared_surplus_fill"
                ),
                "owner_item_receipt_sha256": owner_item_receipt,
            }
        ]
    else:
        visible_item = residual_item
        visible_binding = residual_binding
        exclusions = []
    dedup_body = {"exclusions": exclusions, "format": "dedup-test"}
    dedup = {**dedup_body, "receipt_sha256": identity_sha256(dedup_body)}
    composition = SimpleNamespace(
        packet=SimpleNamespace(
            items=(visible_item,), local_bindings=(visible_binding,)
        ),
        post_selection_dedup_audit=dedup,
    )
    search = SimpleNamespace(
        classified_frontier=SimpleNamespace(
            closed=True,
            packed_segment_receipt_sha256s=(segment,),
            protected_duplicate_segment_receipt_sha256s=(),
            receipt_sha256=_sha("frontier"),
            retained_segment_receipt_sha256s=(segment,),
            unresolved_segment_receipt_sha256s=(),
        ),
        evidence=(evidence,),
        local_bindings=(local,),
        protected_duplicates=(),
        receipt_sha256=_sha("search"),
    )
    return search, residual_contribution, composition


@pytest.mark.parametrize(
    ("duplicate", "expected_disposition"),
    [
        (False, "residual_visible"),
        (True, "protected_visible_exact_duplicate"),
    ],
)
def test_closure_plan_covers_each_retained_segment_with_visible_typed_owner(
    duplicate: bool,
    expected_disposition: str,
) -> None:
    search, contribution, composition = _closure_fixture(duplicate=duplicate)

    planned = arm._closure_plan(  # noqa: SLF001
        search, contribution, composition, protected_owners={}
    )

    assert planned is not None
    rows, protected_items, protection_receipt = planned
    assert [row["segment_receipt_sha256"] for row in rows] == list(
        search.classified_frontier.retained_segment_receipt_sha256s
    )
    assert rows[0]["disposition"] == expected_disposition
    assert protected_items == (rows[0]["visible_item_receipt_sha256"],)
    assert len(protection_receipt) == 64


def test_classified_closure_fails_closed_when_visible_owner_was_not_fitted() -> None:
    search, contribution, composition = _closure_fixture(duplicate=True)
    planned = arm._closure_plan(  # noqa: SLF001
        search, contribution, composition, protected_owners={}
    )
    assert planned is not None
    rows, protected, protection = planned
    fitted = SimpleNamespace(
        allowed_handle_ids=(),
        protected_item_receipt_sha256s=protected,
        receipt_sha256=_sha("fit"),
    )

    with pytest.raises(
        arm.ReducedSemanticBinarySearchAssayError,
        match="omitted a classified MAY survivor",
    ):
        arm._classified_closure(  # noqa: SLF001
            search,
            composition,
            fitted,
            rows,
            protection,
        )


def test_fallback_row_has_no_terminal_or_composition_state() -> None:
    common = {
        "dated_question_sha256": _sha("dated"),
        "namespace_id": _sha("namespace"),
        "new_provider_calls": 0,
        "ordinal": 42,
        "parent_source": {"prediction": "parent"},
        "query_vector_artifact_sha256": _sha("vectors"),
        "query_vector_row_receipt_sha256": _sha("vector-row"),
        "question_id": "q42",
        "question_sha256": _sha("question"),
        "retained_transformer_token_state_bytes": 0,
        "semantic_query": {},
        "semantic_residual_index_receipt_sha256": _sha("index"),
        "semantic_residual_local_audit": {},
        "semantic_residual_search": {},
    }

    row = arm._fallback_question(  # noqa: SLF001
        common,
        fallback_reason="retained_unknowns_exceed_payload_cap",
    )

    assert row["mode"] == "parent_passthrough"
    assert row["terminal_prompt"] is None
    assert row["classified_closure"] is None
    assert row["fitted_typed_prompt"] is None
    assert row["additive_composition"] is None
    assert row["new_provider_calls"] == 0


def test_policy_knobs_are_exposed_without_code_edits() -> None:
    args = arm.build_parser().parse_args(
        [
            "construct",
            "--expected-vector-sha256",
            _sha("vector"),
            "--max-cell-tokens",
            "777",
            "--payload-token-cap",
            "3333",
            "--cosine-upper-bound-floor",
            "0.27",
            "--specificity-upper-bound-ratio",
            "0.9",
            "--no-dual-gate-enabled",
        ]
    )

    policy = arm._policy(args)  # noqa: SLF001

    assert policy.max_cell_tokens == 777
    assert policy.payload_token_cap == 3333
    assert policy.cosine_upper_bound_floor == 0.27
    assert policy.specificity_upper_bound_ratio == 0.9
    assert policy.dual_gate_enabled is False


def test_typed_composer_successor_has_explicit_mode_and_isolated_artifact_path() -> None:
    legacy = arm.build_parser().parse_args(
        ["construct", "--expected-vector-sha256", _sha("legacy-vector")]
    )
    successor = arm.build_parser().parse_args(
        [
            "construct",
            "--expected-vector-sha256",
            _sha("successor-vector"),
            "--typed-composition-mode",
            arm.POST_DEDUP_BACKFILL_COMPOSITION_MODE,
        ]
    )

    assert arm.construction_path_for_args(legacy) == (
        arm.DEFAULT_OUTPUT_ROOT / arm.CONSTRUCTION_NAME
    )
    assert arm.construction_path_for_args(successor) == (
        arm.DEFAULT_SUCCESSOR_OUTPUT_ROOT / arm.SUCCESSOR_CONSTRUCTION_NAME
    )
    assert arm.construction_path_for_args(successor) != (
        arm.DEFAULT_OUTPUT_ROOT / arm.CONSTRUCTION_NAME
    )


def test_typed_composer_successor_audit_defaults_are_isolated() -> None:
    legacy = arm.build_parser().parse_args(
        ["audit", "--expected-construction-sha256", _sha("legacy")]
    )
    successor = arm.build_parser().parse_args(
        [
            "audit",
            "--expected-construction-sha256",
            _sha("successor"),
            "--typed-composition-mode",
            arm.POST_DEDUP_BACKFILL_COMPOSITION_MODE,
        ]
    )

    assert arm.audit_construction_path_for_args(legacy) == (
        arm.DEFAULT_OUTPUT_ROOT / arm.CONSTRUCTION_NAME
    )
    assert arm.audit_output_path_for_args(legacy) == (
        arm.DEFAULT_OUTPUT_ROOT / arm.AUDIT_NAME
    )
    assert arm.audit_construction_path_for_args(successor) == (
        arm.DEFAULT_SUCCESSOR_OUTPUT_ROOT / arm.SUCCESSOR_CONSTRUCTION_NAME
    )
    assert arm.audit_output_path_for_args(successor) == (
        arm.DEFAULT_SUCCESSOR_OUTPUT_ROOT / arm.SUCCESSOR_AUDIT_NAME
    )


def test_successor_audit_format_binds_typed_composer_v2(monkeypatch) -> None:
    target_ids = {
        42: ("answer_42_a", "answer_42_b"),
        65: ("answer_65_a", "answer_65_b"),
        74: ("answer_74",),
        79: ("answer_79",),
    }
    rows = []
    desired = []
    for ordinal, expected in target_ids.items():
        question_id = f"q{ordinal}"
        attempted_rows = []
        closure_rows = []
        for index, target_id in enumerate(expected):
            binding_receipt = _sha(f"successor-binding-{ordinal}-{index}")
            segment = _sha(f"successor-segment-{ordinal}-{index}")
            attempted_rows.append(
                {
                    "attempted_selection": {
                        "local_binding_receipt_sha256": binding_receipt,
                        "segment_receipt_sha256": segment,
                        "source_id": f"{question_id}::{target_id}",
                    }
                }
            )
            closure_rows.append({"segment_receipt_sha256": segment})
            desired.append(
                {
                    "ordinal": ordinal,
                    "question_id": question_id,
                    "target_id": target_id,
                    "target_kind": "source_id",
                }
            )
        rows.append(
            {
                "classified_closure": {"rows": closure_rows},
                "mode": "semantic_residual",
                "ordinal": ordinal,
                "question_id": question_id,
                "semantic_residual_local_audit": {
                    "attempted_selection_manifest": {"rows": attempted_rows}
                },
            }
        )
    plan = {"desired_targets": desired, "plan_sha256": _sha("successor-plan")}
    construction = SealedArtifact(
        Path("successor-construction.json"),
        _sha("successor-construction"),
        {"typed_composition_format": arm.SUCCESSOR_TYPED_COMPOSITION_FORMAT},
    )
    monkeypatch.setattr(arm, "validate_construction", lambda _artifact: tuple(rows))

    audit = arm.build_target_audit(
        construction,
        plan,
        target_plan_file_sha256=_sha("successor-plan-file"),
    )

    assert audit["format"] == arm.SUCCESSOR_AUDIT_FORMAT
    assert (
        audit["typed_composition_format"]
        == arm.SUCCESSOR_TYPED_COMPOSITION_FORMAT
    )


def test_posthoc_audit_reports_exact_six_source_targets(monkeypatch) -> None:
    target_ids = {
        42: ("answer_42_a", "answer_42_b"),
        65: ("answer_65_a", "answer_65_b"),
        74: ("answer_74",),
        79: ("answer_79",),
    }
    rows = []
    desired = []
    for ordinal, expected in target_ids.items():
        question_id = f"q{ordinal}"
        attempted_rows = []
        closure_rows = []
        for index, target_id in enumerate(expected):
            binding_receipt = _sha(f"binding-{ordinal}-{index}")
            segment = _sha(f"segment-{ordinal}-{index}")
            attempted_rows.append(
                {
                    "attempted_selection": {
                        "local_binding_receipt_sha256": binding_receipt,
                        "segment_receipt_sha256": segment,
                        "source_id": f"{question_id}::{target_id}",
                    }
                }
            )
            closure_rows.append({"segment_receipt_sha256": segment})
            desired.append(
                {
                    "ordinal": ordinal,
                    "question_id": question_id,
                    "target_id": target_id,
                    "target_kind": "source_id",
                }
            )
        rows.append(
            {
                "classified_closure": {"rows": closure_rows},
                "mode": "semantic_residual",
                "ordinal": ordinal,
                "question_id": question_id,
                "semantic_residual_local_audit": {
                    "attempted_selection_manifest": {"rows": attempted_rows}
                },
            }
        )
    plan = {"desired_targets": desired, "plan_sha256": _sha("plan")}
    artifact = SealedArtifact(Path("construction.json"), _sha("construction"), {})
    monkeypatch.setattr(arm, "validate_construction", lambda _artifact: tuple(rows))

    audit = arm.build_target_audit(
        artifact,
        plan,
        target_plan_file_sha256=_sha("plan-file"),
    )

    assert audit["construction_verified_before_target_plan_load"] is True
    assert audit["runtime_use_forbidden"] is True
    assert audit["question_count"] == 4
    assert audit["selected_source_target_hits"] == 6
    assert audit["selected_source_target_count"] == 6
    assert audit["terminal_source_target_hits"] == 6
    assert audit["terminal_source_target_count"] == 6
    assert audit["new_provider_calls"] == 0


def test_identity_projection_rejects_resealed_inner_tamper() -> None:
    body = {"format": "test", "value": 1}
    projection = {**body, "receipt_sha256": identity_sha256(body)}
    projection["value"] = 2

    with pytest.raises(
        arm.ReducedSemanticBinarySearchAssayError,
        match="receipt changed",
    ):
        arm._identity_projection(projection, label="test projection")  # noqa: SLF001


def _local_binding(*, quote: str, namespace_id: str | None = None) -> LocalCitationBinding:
    source_id = "source-1"
    span = EvidenceSpan(
        chunk_id="chunk-1",
        start_char=0,
        end_char=len(quote),
        quote_sha256=quote_sha256(quote),
        ordinal=0,
        source_id=source_id,
        role="user",
        created_at="2026-01-01T00:00:00Z",
    )
    return LocalCitationBinding(
        candidate_id=_sha("local-candidate"),
        source_group_handle="G0001",
        namespace_id=namespace_id or _sha("namespace"),
        cache_receipt_sha256=_sha("cache"),
        source_database_sha256=_sha("database"),
        source_store_receipt_sha256=_sha("store"),
        source_id=source_id,
        partition_id="partition-1",
        span=span,
        quote_sha256=quote_sha256(quote),
    )


def test_protected_parent_local_evidence_requires_exact_visible_summary() -> None:
    quote = "Exact parent-visible bytes."
    local = _local_binding(quote=quote)
    item = SimpleNamespace(
        handle_ids=("H100001",), receipt_sha256=_sha("parent-item"), summary=quote
    )
    provenance = SimpleNamespace(
        cloned_binding=SimpleNamespace(receipt_sha256=_sha("cloned-binding")),
        handle_id="H100001",
        original_binding=SimpleNamespace(
            citation_char_count=len(quote),
            citation_sha256=quote_sha256(quote),
            local_source_locator_sha256=local.receipt_sha256,
        ),
    )
    parent = SimpleNamespace(
        audit=SimpleNamespace(
            compact_item_receipt_order=(item.receipt_sha256,),
            receipt_sha256=_sha("parent-audit"),
            source_provenance=(provenance,),
        ),
        contributions=(
            SimpleNamespace(parsed=SimpleNamespace(accepted_items=(item,))),
        ),
    )
    composition_row = {"local_audit": {"nested": [local.projection()]}}

    protected, owners, inventory = arm._protected_parent_local_evidence(  # noqa: SLF001
        composition_row, parent, namespace_id=local.namespace_id
    )

    assert protected == (local,)
    assert owners[local.receipt_sha256]["parent_item_receipt_sha256"] == (
        item.receipt_sha256
    )
    assert inventory["provider_visible_exact_owner_count"] == 1

    item.summary = "different bytes"
    protected, owners, _inventory = arm._protected_parent_local_evidence(  # noqa: SLF001
        composition_row, parent, namespace_id=local.namespace_id
    )
    assert protected == ()
    assert owners == {}


def test_closure_plan_accepts_search_level_protected_duplicate() -> None:
    segment = _sha("protected-segment")
    owner_item = SimpleNamespace(
        handle_ids=("H100001",),
        receipt_sha256=_sha("protected-owner-item"),
        summary="Exact protected sentence.",
    )
    owner_binding = SimpleNamespace(
        handle_id="H100001", receipt_sha256=_sha("protected-owner-binding")
    )
    duplicate = SimpleNamespace(
        cell_id="cell-protected",
        protected_binding_receipt_sha256=_sha("protected-local"),
        receipt_sha256=_sha("protected-duplicate"),
        segment_receipt_sha256=segment,
    )
    frontier = SimpleNamespace(
        closed=True,
        packed_segment_receipt_sha256s=(),
        protected_duplicate_segment_receipt_sha256s=(segment,),
        receipt_sha256=_sha("protected-frontier"),
        retained_segment_receipt_sha256s=(segment,),
        unresolved_segment_receipt_sha256s=(),
    )
    search = SimpleNamespace(
        classified_frontier=frontier,
        evidence=(),
        local_bindings=(),
        protected_duplicates=(duplicate,),
        receipt_sha256=_sha("protected-search"),
    )
    contribution = SimpleNamespace(
        bindings=(), parsed=SimpleNamespace(accepted_items=())
    )
    dedup_body = {"exclusions": [], "format": "dedup-test"}
    composition = SimpleNamespace(
        packet=SimpleNamespace(
            items=(owner_item,), local_bindings=(owner_binding,)
        ),
        post_selection_dedup_audit={
            **dedup_body,
            "receipt_sha256": identity_sha256(dedup_body),
        },
    )
    owners = {
        duplicate.protected_binding_receipt_sha256: {
            "exact_text_sha256": quote_sha256(owner_item.summary),
            "local_binding_receipt_sha256": (
                duplicate.protected_binding_receipt_sha256
            ),
            "parent_binding_receipt_sha256s": [owner_binding.receipt_sha256],
            "parent_handle_ids": [owner_binding.handle_id],
            "parent_item_receipt_sha256": owner_item.receipt_sha256,
        }
    }

    planned = arm._closure_plan(  # noqa: SLF001
        search, contribution, composition, protected_owners=owners
    )

    assert planned is not None
    rows, protected_items, _receipt = planned
    assert rows[0]["segment_receipt_sha256"] == segment
    assert rows[0]["residual_evidence_receipt_sha256"] == duplicate.receipt_sha256
    assert rows[0]["dedup_exclusion_sha256"] == duplicate.receipt_sha256
    assert protected_items == (owner_item.receipt_sha256,)


def test_attempted_selection_preserves_source_and_binding_on_overcap() -> None:
    quote = "Selected but too large for the provider packet."
    local = _local_binding(quote=quote)
    segment = SimpleNamespace(
        partition_id=local.partition_id,
        quote_sha256=local.quote_sha256,
        receipt_sha256=_sha("attempted-segment"),
        source_id=local.source_id,
        span=local.span,
    )
    cell = SimpleNamespace(
        cell_id="cell-0",
        receipt_sha256=_sha("attempted-cell"),
        segments=(segment,),
        source_id=local.source_id,
    )
    index = SimpleNamespace(
        cache_receipt_sha256=local.cache_receipt_sha256,
        cell_by_id={cell.cell_id: cell},
        namespace_id=local.namespace_id,
        receipt_sha256=_sha("attempted-index"),
        source_database_sha256=local.source_database_sha256,
        source_store_receipt_sha256=local.source_store_receipt_sha256,
    )
    frontier = SimpleNamespace(
        retained_segment_receipt_sha256s=(segment.receipt_sha256,),
    )
    search = SimpleNamespace(
        attempted_evidence_count=1,
        attempted_selection=(
            SimpleNamespace(
                candidate_id=_sha("attempted-candidate"),
                cell_id=cell.cell_id,
                disposition="novel",
                evidence_receipt_sha256=_sha("attempted-evidence"),
                local_binding_receipt_sha256=_sha("attempted-binding"),
                projection=lambda: {
                    "candidate_id": _sha("attempted-candidate"),
                    "cell_id": cell.cell_id,
                    "disposition": "novel",
                    "evidence_receipt_sha256": _sha("attempted-evidence"),
                    "format": f"{arm.residual.RESULT_FORMAT}-attempted-selection-v1",
                    "local_binding_receipt_sha256": _sha("attempted-binding"),
                    "protected_duplicate_receipt_sha256": None,
                    "receipt_sha256": _sha("attempted-row"),
                    "segment_receipt_sha256": segment.receipt_sha256,
                    "source_id": cell.source_id,
                },
                protected_duplicate_receipt_sha256=None,
                receipt_sha256=_sha("attempted-row"),
                segment_receipt_sha256=segment.receipt_sha256,
                source_id=cell.source_id,
            ),
        ),
        classified_frontier=frontier,
        core_result=SimpleNamespace(retained_leaf_cell_ids=(cell.cell_id,)),
        evidence=(),
        local_bindings=(),
        protected_duplicates=(),
        projection=lambda: {
            "attempted_selection_receipt_sha256": identity_sha256(
                {
                    "format": (
                        f"{arm.residual.RESULT_FORMAT}-attempted-selection-population-v1"
                    ),
                    "row_receipt_sha256s": [_sha("attempted-row")],
                }
            )
        },
        receipt_sha256=_sha("attempted-search"),
    )

    attempted = arm._attempted_selection_projection(index, search)  # noqa: SLF001

    assert attempted["exact_text_included"] is False
    assert attempted["novel_attempted_count"] == 1
    canonical = attempted["rows"][0]["attempted_selection"]
    assert canonical["source_id"] == local.source_id
    assert len(canonical["local_binding_receipt_sha256"]) == 64


def test_stored_core_projection_replaces_quadratic_visit_payload_with_receipts() -> None:
    decision = SimpleNamespace(
        projection=lambda: {"receipt_sha256": _sha("decision")},
        receipt_sha256=_sha("decision"),
    )
    visit = SimpleNamespace(receipt_sha256=_sha("visit"))
    outcome = SimpleNamespace(receipt_sha256=_sha("outcome"))
    full = {
        "visits": [{"covered_leaf_cell_ids": ["cell"] * 100}],
        "leaf_outcomes": [{"cell_id": "cell"}],
    }
    core = SimpleNamespace(
        classified_node_token_count=10,
        classifier_calls=1,
        classifier_id="classifier",
        decisions=(decision,),
        fit_policy_id="fit",
        leaf_outcomes=(outcome,),
        projection=lambda: full,
        pruned_leaf_cell_ids=(),
        pruned_token_count=0,
        question_sha256=_sha("question"),
        question_token_count=3,
        receipt_sha256=_sha("core"),
        retained_leaf_cell_ids=("cell",),
        retained_token_count=10,
        tree_receipt_sha256=_sha("tree"),
        visits=(visit,),
    )

    compact = arm._stored_core_projection(core)  # noqa: SLF001

    assert "visits" not in compact
    assert "leaf_outcomes" not in compact
    assert compact["visit_receipt_sha256s"] == [visit.receipt_sha256]
    assert compact["leaf_outcome_receipt_sha256s"] == [outcome.receipt_sha256]
    arm._stored_projection(  # noqa: SLF001
        compact, label="compact core", expected_format=arm.STORED_CORE_FORMAT
    )
    compact["retained_token_count"] += 1
    with pytest.raises(
        arm.ReducedSemanticBinarySearchAssayError,
        match="compact receipt changed",
    ):
        arm._stored_projection(  # noqa: SLF001
            compact, label="compact core", expected_format=arm.STORED_CORE_FORMAT
        )


def test_local_audit_keeps_canonical_attempted_rows_separate_from_manifest() -> None:
    canonical_attempted = [{"format": "canonical-attempted", "value": 1}]
    search = SimpleNamespace(
        local_audit_projection=lambda: {
            "attempted_selection": canonical_attempted,
            "classified_frontier": {},
            "compact_result_receipt_sha256": _sha("result"),
            "local_bindings": [],
            "protected_duplicates": [],
            "query": {},
        }
    )
    attempted_body = {"format": arm.ATTEMPTED_SELECTION_FORMAT, "rows": []}
    attempted = {
        **attempted_body,
        "receipt_sha256": identity_sha256(attempted_body),
    }
    capacity_body = {"format": arm.CAPACITY_CERTIFICATE_FORMAT}
    capacity = {
        **capacity_body,
        "receipt_sha256": identity_sha256(capacity_body),
    }
    inventory_body = {"format": arm.PROTECTED_PARENT_INVENTORY_FORMAT}
    inventory = {
        **inventory_body,
        "receipt_sha256": identity_sha256(inventory_body),
    }

    local = arm._semantic_local_audit(  # noqa: SLF001
        search,
        attempted_selection=attempted,
        capacity_certificate=capacity,
        protected_parent_inventory=inventory,
    )

    assert local["attempted_selection"] == canonical_attempted
    assert local["attempted_selection_manifest"] == attempted
    arm._identity_projection(local, label="semantic local audit")  # noqa: SLF001


def test_run_construct_validates_before_publish_and_leaves_no_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arm, "build_construction", lambda _args: {"invalid": True})

    def reject(_artifact: SealedArtifact):
        raise arm.ReducedSemanticBinarySearchAssayError("invalid candidate")

    monkeypatch.setattr(arm, "validate_construction", reject)

    with pytest.raises(
        arm.ReducedSemanticBinarySearchAssayError, match="invalid candidate"
    ):
        arm.run_construct(argparse.Namespace(output_root=tmp_path))

    target = tmp_path / arm.CONSTRUCTION_NAME
    assert not target.exists()
    assert not target.with_name(target.name + ".sha256").exists()
