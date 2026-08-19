from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from memory_condense.eval import diffuse_longmemeval_runtime_matched as runtime_matched
from memory_condense.eval.diffuse_longmemeval_matched import (
    MATCHED_BOUNDARY_MODES,
    DiffuseLongMemEvalMatchedProbeReceipt,
    DiffuseLongMemEvalMatchedSuiteReceipt,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalRuntimeResult,
    ResidencyPreflightObservation,
)
from memory_condense.eval.diffuse_longmemeval_runtime_matched import (
    MINIMUM_RESIDENT_FREE_BYTES,
    RESIDENT_PREFLIGHT_POLICY,
    STAGED_PREFLIGHT_POLICY,
    validate_matched_diffuse_runtime_results,
)


def _sha(character: str) -> str:
    return character * 64


def _phase(mode: str, receipt_sha256: str):
    return SimpleNamespace(
        receipt_sha256=receipt_sha256,
        arm=SimpleNamespace(
            compilation=SimpleNamespace(boundary_mode=mode),
        ),
    )


def _probe() -> DiffuseLongMemEvalMatchedProbeReceipt:
    return DiffuseLongMemEvalMatchedProbeReceipt(
        question_id="q-1",
        question_probe_sha256=_sha("1"),
        retrieval_query_sha256=_sha("2"),
        retrieval_policy_sha256=_sha("3"),
        anchor_sequence_sha256=_sha("4"),
        anchor_chunk_ids=("chunk-a",),
        source_candidate_sequence_sha256=_sha("5"),
        source_candidate_ids=("source-a",),
        source_scope_identity_sha256=_sha("6"),
        legacy_input_provider_identity_sha256=_sha("7"),
        representative_linker_identity_sha256=_sha("8"),
        representative_policy_factory_identity_sha256=_sha("9"),
        representative_policy_controls_sha256=_sha("a"),
        episode_policy_sha256=_sha("b"),
        closure_policy_sha256=_sha("c"),
    )


def _matched_receipt(
    phase_receipts: tuple[str, str, str],
) -> DiffuseLongMemEvalMatchedSuiteReceipt:
    return DiffuseLongMemEvalMatchedSuiteReceipt(
        sample_id="sample-v4",
        corpus_sha256=_sha("d"),
        deterministic_turn_ids_sha256=_sha("e"),
        evaluation_policy_sha256=_sha("f"),
        matched_controls_sha256=_sha("0"),
        pipeline_modes=MATCHED_BOUNDARY_MODES,
        pipeline_arm_sha256s=(_sha("1"), _sha("2"), _sha("3")),
        compilation_receipt_sha256s=(_sha("4"), _sha("5"), _sha("6")),
        retrieval_phase_receipt_sha256s=phase_receipts,
        probes=(_probe(),),
        qwen_source_signal_receipt_sha256s=(_sha("7"),),
        qwen_owned_representative_runtime=True,
        zero_returned_transformer_state=True,
        zero_persisted_transformer_state=True,
    )


def _resident_observation(free_bytes: int) -> ResidencyPreflightObservation:
    return ResidencyPreflightObservation(
        policy=RESIDENT_PREFLIGHT_POLICY,
        device="cuda:0",
        required_free_bytes=MINIMUM_RESIDENT_FREE_BYTES,
        observed_free_bytes=free_bytes,
        observed_total_bytes=8 * 1024**3,
        embedding_released_before_qwen_load=False,
    )


def _staged_observation() -> ResidencyPreflightObservation:
    return ResidencyPreflightObservation(
        policy=STAGED_PREFLIGHT_POLICY,
        device="cuda:0",
        required_free_bytes=0,
        observed_free_bytes=None,
        observed_total_bytes=None,
        embedding_released_before_qwen_load=True,
    )


def _runtime_results(*, staged: bool = False):
    phase_receipts = (_sha("8"), _sha("9"), _sha("a"))
    rows = []
    for index, (mode, phase_receipt) in enumerate(
        zip(MATCHED_BOUNDARY_MODES, phase_receipts, strict=True)
    ):
        observation = (
            _staged_observation()
            if staged
            else _resident_observation(
                MINIMUM_RESIDENT_FREE_BYTES + (index + 1) * 1024**2
            )
        )
        rows.append(
            DiffuseLongMemEvalRuntimeResult(
                phase=_phase(mode, phase_receipt),
                runtime_binding_sha256=_sha("b"),
                runtime_binding_certified=True,
                residency_preflight=observation,
            )
        )
    return tuple(rows), _matched_receipt(phase_receipts)


def _install_delegate(monkeypatch, matched, calls=None):
    def validate(phases):
        if calls is not None:
            calls.append(tuple(phases))
        return matched

    monkeypatch.setattr(
        runtime_matched,
        "validate_matched_diffuse_retrieval_phases",
        validate,
    )


def test_runtime_suite_delegates_phases_and_binds_full_preflights(
    monkeypatch,
) -> None:
    results, matched = _runtime_results()
    calls = []
    _install_delegate(monkeypatch, matched, calls)

    receipt = validate_matched_diffuse_runtime_results(results[::-1])

    assert calls == [tuple(item.phase for item in results[::-1])]
    assert receipt.runtime_binding_certified
    assert receipt.runtime_binding_sha256 == _sha("b")
    assert receipt.residency_policy == RESIDENT_PREFLIGHT_POLICY
    assert receipt.residency_device == "cuda:0"
    assert receipt.required_free_bytes == MINIMUM_RESIDENT_FREE_BYTES
    assert receipt.runtime_result_receipt_sha256s == tuple(
        item.receipt_sha256 for item in results
    )
    assert receipt.matched_suite is matched
    payload = receipt.identity_payload()
    assert payload["matched_suite_receipt_sha256"] == matched.receipt_sha256
    assert [
        item["observed_free_bytes"]
        for item in payload["preflight_observations"]
    ] == [
        item.residency_preflight.observed_free_bytes for item in results
    ]
    assert [
        item["receipt_sha256"]
        for item in payload["preflight_observations"]
    ] == [item.residency_preflight.receipt_sha256 for item in results]


def test_runtime_suite_accepts_exact_staged_release_semantics(monkeypatch) -> None:
    results, matched = _runtime_results(staged=True)
    _install_delegate(monkeypatch, matched)

    receipt = validate_matched_diffuse_runtime_results(results)

    assert receipt.residency_policy == STAGED_PREFLIGHT_POLICY
    assert receipt.required_free_bytes == 0
    assert all(
        item.embedding_released_before_qwen_load
        for item in receipt.preflight_observations
    )


def test_runtime_suite_requires_exactly_three_results() -> None:
    results, _matched = _runtime_results()

    with pytest.raises(ValueError, match="exactly three"):
        validate_matched_diffuse_runtime_results(results[:2])


@pytest.mark.parametrize("mutation", ["uncertified", "binding", "device"])
def test_runtime_suite_rejects_changed_runtime_controls(
    mutation,
    monkeypatch,
) -> None:
    results, matched = _runtime_results()
    _install_delegate(monkeypatch, matched)
    rows = list(results)
    if mutation == "uncertified":
        rows[1] = replace(
            rows[1],
            runtime_binding_certified=False,
            receipt_sha256="",
        )
    elif mutation == "binding":
        rows[1] = replace(
            rows[1],
            runtime_binding_sha256=_sha("c"),
            receipt_sha256="",
        )
    else:
        changed = replace(rows[1].residency_preflight, device="cuda:1")
        rows[1] = replace(
            rows[1],
            residency_preflight=changed,
            receipt_sha256="",
        )

    with pytest.raises(ValueError):
        validate_matched_diffuse_runtime_results(rows)


def test_runtime_suite_rejects_resident_memory_below_bound_threshold(
    monkeypatch,
) -> None:
    results, matched = _runtime_results()
    _install_delegate(monkeypatch, matched)
    below = replace(
        results[0].residency_preflight,
        observed_free_bytes=MINIMUM_RESIDENT_FREE_BYTES - 1,
    )
    rows = (
        replace(results[0], residency_preflight=below, receipt_sha256=""),
        *results[1:],
    )

    with pytest.raises(ValueError, match="below its threshold"):
        validate_matched_diffuse_runtime_results(rows)


def test_runtime_suite_rejects_a_lowered_resident_gate(monkeypatch) -> None:
    results, matched = _runtime_results()
    _install_delegate(monkeypatch, matched)
    rows = tuple(
        replace(
            item,
            residency_preflight=replace(
                item.residency_preflight,
                required_free_bytes=MINIMUM_RESIDENT_FREE_BYTES - 1,
            ),
            receipt_sha256="",
        )
        for item in results
    )

    with pytest.raises(ValueError, match="at least"):
        validate_matched_diffuse_runtime_results(rows)


def test_runtime_suite_rejects_forged_result_receipt(monkeypatch) -> None:
    results, matched = _runtime_results()
    _install_delegate(monkeypatch, matched)
    object.__setattr__(results[0], "receipt_sha256", _sha("f"))

    with pytest.raises(ValueError, match="receipt does not match"):
        validate_matched_diffuse_runtime_results(results)


def test_runtime_suite_rejects_staged_observation_that_claims_gpu_memory(
    monkeypatch,
) -> None:
    results, matched = _runtime_results(staged=True)
    _install_delegate(monkeypatch, matched)
    invalid = replace(
        results[2].residency_preflight,
        observed_free_bytes=MINIMUM_RESIDENT_FREE_BYTES,
    )
    rows = (
        *results[:2],
        replace(results[2], residency_preflight=invalid, receipt_sha256=""),
    )

    with pytest.raises(ValueError, match="cannot claim"):
        validate_matched_diffuse_runtime_results(rows)


def test_runtime_suite_rejects_delegate_receipt_for_other_phases(monkeypatch) -> None:
    results, matched = _runtime_results()
    wrong = replace(
        matched,
        retrieval_phase_receipt_sha256s=(
            _sha("c"),
            *matched.retrieval_phase_receipt_sha256s[1:],
        ),
        receipt_sha256="",
    )
    _install_delegate(monkeypatch, wrong)

    with pytest.raises(ValueError, match="does not bind"):
        validate_matched_diffuse_runtime_results(results)


def test_runtime_suite_rejects_forged_matched_receipt(monkeypatch) -> None:
    results, matched = _runtime_results()
    object.__setattr__(matched, "receipt_sha256", _sha("f"))
    _install_delegate(monkeypatch, matched)

    with pytest.raises(ValueError, match="matched-phase suite receipt"):
        validate_matched_diffuse_runtime_results(results)
