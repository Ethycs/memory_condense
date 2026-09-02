"""Regression coverage for one-body projection and tamper-evident resealing."""

from __future__ import annotations

from dataclasses import fields

from memory_condense.domain.discourse import quote_sha256

from tools.matched_eval import source_gate_controller
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.source_gate_controller import EligibleFrontierScope


def _sha(value: str) -> str:
    return quote_sha256(value)


def _scope() -> EligibleFrontierScope:
    return EligibleFrontierScope(
        eligible_candidate_ids=(_sha("candidate"),),
        exhaustive=False,
        basis_receipt_sha256=_sha("basis"),
    )


def test_projection_builds_and_seals_one_shared_body(monkeypatch) -> None:
    body_calls = 0
    seal_calls = 0
    original_body = EligibleFrontierScope._body
    original_seal = source_gate_controller._seal

    def counted_body(self: EligibleFrontierScope):
        nonlocal body_calls
        body_calls += 1
        return original_body(self)

    def counted_seal(kind, body):
        nonlocal seal_calls
        seal_calls += 1
        return original_seal(kind, body)

    monkeypatch.setattr(EligibleFrontierScope, "_body", counted_body)
    monkeypatch.setattr(source_gate_controller, "_seal", counted_seal)

    scope = _scope()
    first_projection = scope.projection()

    assert body_calls == 1
    assert seal_calls == 1
    first_receipt = scope.receipt_sha256
    assert body_calls == 2
    assert seal_calls == 2
    assert first_projection["receipt_sha256"] == first_receipt

    assert scope.receipt_sha256 == first_receipt
    assert scope.projection() == first_projection
    assert body_calls == 4
    assert seal_calls == 4


def test_projection_reseals_after_forced_tamper() -> None:
    scope = _scope()
    original_receipt = scope.receipt_sha256

    object.__setattr__(scope, "exhaustive", True)
    projection = scope.projection()

    assert projection["exhaustive"] is True
    assert projection["receipt_sha256"] != original_receipt
    assert projection["receipt_sha256"] == scope.receipt_sha256


def test_receipt_cache_preserves_constructor_projection_and_sealed_identity() -> None:
    scope = _scope()
    body = {
        "basis_receipt_sha256": _sha("basis"),
        "eligible_candidate_ids": [_sha("candidate")],
        "exhaustive": False,
    }
    expected_receipt = identity_sha256(
        {
            "format": "memory-condense-source-gate-controller-v1-eligible-frontier",
            **body,
        }
    )
    expected_projection = {
        "format": "memory-condense-source-gate-controller-v1-eligible-frontier",
        **body,
        "receipt_sha256": expected_receipt,
    }

    assert tuple(field.name for field in fields(scope)) == (
        "eligible_candidate_ids",
        "exhaustive",
        "basis_receipt_sha256",
    )
    assert scope.receipt_sha256 == expected_receipt
    assert scope.receipt_sha256 == expected_receipt
    assert scope.projection() == expected_projection
    assert canonical_json_bytes(scope.projection()) == canonical_json_bytes(
        expected_projection
    )
