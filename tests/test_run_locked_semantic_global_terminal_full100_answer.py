from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_semantic_global_terminal_full100_answer as answer
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import VALIDATION_CONTRACT_FORMAT


def _sha(label: str) -> str:
    return quote_sha256(label)


def _artifact(path: Path, digest_label: str, payload: dict[str, Any]) -> SealedArtifact:
    return SealedArtifact(path, _sha(digest_label), payload)


def _with_receipt(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: identity_sha256(body)}


def _promotion_audit(
    root: Path,
    promotion: SealedArtifact,
    *,
    atom_count: int | None = None,
) -> SealedArtifact:
    count = answer.postseal_cli.SEMANTIC_ATOM_COUNT if atom_count is None else atom_count
    return _artifact(
        root / "promotion-audit.json",
        "promotion audit",
        {
            "audit_identity_sha256": _sha("promotion identity"),
            "promotion_gate_passed": True,
            "semantic_atom_manifest_artifact_sha256": (
                answer.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
            ),
            "semantic_atom_manifest_identity_sha256": (
                answer.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256
            ),
            "semantic_atom_population_sha256": (
                answer.postseal_cli.DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256
            ),
            "target_plan_artifact_sha256": (
                answer.postseal_cli.DEFAULT_TARGET_PLAN_SHA256
            ),
            "target_plan_identity_sha256": (
                answer.postseal_cli.DEFAULT_TARGET_PLAN_IDENTITY_SHA256
            ),
            "terminal_construction_sha256": promotion.sha256,
            "terminal_replay_sha256": promotion.sha256,
            "totals": {
                "fact_final_usable_count": 29,
                "positive_witness_count": answer.postseal_cli.POSITIVE_WITNESS_COUNT,
                "raw_witness_final_usable_count": 29,
                "semantic_atom_count": count,
                "semantic_atom_final_usable_count": count,
                "source_final_usable_count": 24,
                "source_target_count": answer.postseal_cli.SOURCE_TARGET_COUNT,
            },
            "witness_manifest_artifact_sha256": (
                answer.postseal_cli.DEFAULT_WITNESS_MANIFEST_SHA256
            ),
            "witness_manifest_identity_sha256": (
                answer.postseal_cli.DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256
            ),
        },
    )


def _full_plan(ordinal: int) -> dict[str, Any]:
    question = f"What is memory {ordinal}?"
    dated = f"[Question asked at 2026/08/30 12:00] {question}"
    parent = f"Parent prediction {ordinal}."
    provider_input = {
        "dated_question": dated,
        "format": "synthetic-full100-provider-input-v1",
        "protected_parent_fallback": {"prediction": parent},
    }
    messages = answer.render_final_messages(provider_input)
    compilation_receipt = _sha(f"compilation {ordinal}")
    body = {
        "allowed_handle_ids": [],
        "dated_question": dated,
        "dated_question_sha256": quote_sha256(dated),
        "format": "synthetic-full100-answer-plan-v1",
        "handle_group_by_id": {},
        "hard_prompt_token_cap": answer.HARD_PROMPT_TOKEN_CAP,
        "messages_sha256": identity_sha256(list(messages)),
        "ordinal": ordinal,
        "output_token_reserve": answer.OUTPUT_TOKEN_RESERVE,
        "parent_prediction": parent,
        "parent_prediction_sha256": quote_sha256(parent),
        "preservation_requirements": {},
        "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
        "provider_input": provider_input,
        "provider_input_sha256": identity_sha256(provider_input),
        "question_id": f"question-{ordinal:03d}",
        "question_sha256": quote_sha256(question),
        "route_id": answer.TERMINAL_ROUTE_ID,
        "source_artifact_bindings": {
            "protected_owner": _sha("protected owner"),
            "residual": _sha("residual"),
        },
        "story_coherence": {},
        "terminal_compilation": {
            "format": "synthetic-terminal-compilation-v1",
            "receipt_sha256": compilation_receipt,
        },
        "terminal_compilation_receipt_sha256": compilation_receipt,
        "validation_contract": {
            "answer_shape": "direct",
            "by_handle": {},
            "cardinality": None,
            "comparison_mode": "none",
            "deterministic_execution_advisory": None,
            "format": VALIDATION_CONTRACT_FORMAT,
            "include_proposed": False,
            "operation": "single_supported_fact",
            "operator_spec_receipt_sha256": _sha(f"operator {ordinal}"),
            "packet_receipt_sha256": _sha(f"packet {ordinal}"),
            "question_action_concepts": [],
            "question_terms": ["memory"],
            "required_slot_ids": [],
            "required_slots": [],
            "requires_all_slots": False,
            "scalar_validation_advisory": None,
            "temporal_mode": "none",
        },
    }
    return {**body, "answer_plan_receipt_sha256": identity_sha256(body)}


@dataclass(frozen=True)
class _Fixture:
    sources: answer._VerifiedSources
    construction: SealedArtifact
    replay: SealedArtifact
    provider_plans: tuple[dict[str, Any], ...]
    passthroughs: tuple[dict[str, Any], ...]
    promotion: SealedArtifact
    promotion_plans: tuple[dict[str, Any], ...]
    audit: SealedArtifact


def _fixture(root: Path) -> _Fixture:
    eligible = set(answer.EXACT_ORDINALS)
    eligible.update(
        ordinal
        for ordinal in answer.ALL_ORDINALS
        if len(eligible) < answer.ELIGIBLE_COUNT
    )
    assert len(eligible) == answer.ELIGIBLE_COUNT
    full_plans: dict[int, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    passthroughs: list[dict[str, Any]] = []
    for ordinal in answer.ALL_ORDINALS:
        full_plan = _full_plan(ordinal)
        parent = full_plan["parent_prediction"]
        base = {
            "dated_question_sha256": full_plan["dated_question_sha256"],
            "eligibility_receipt_sha256": _sha(f"eligibility {ordinal}"),
            "format": answer.full100_cli.ROW_FORMAT,
            "gate_row_receipt_sha256": _sha(f"gate {ordinal}"),
            "mode": (
                answer.TERMINAL_MODE
                if ordinal in eligible
                else answer.PASSTHROUGH_MODE
            ),
            "namespace_id": _sha(f"namespace {ordinal % 10}"),
            "new_provider_calls": 0,
            "ordinal": ordinal,
            "parent_answer_row_sha256": _sha(f"parent row {ordinal}"),
            "parent_prediction": parent,
            "parent_prediction_sha256": quote_sha256(parent),
            "passthrough_prediction": None if ordinal in eligible else parent,
            "question_id": full_plan["question_id"],
            "question_sha256": full_plan["question_sha256"],
            "retained_transformer_token_state_bytes": 0,
            "terminal_answer_plan": None,
            "terminal_question_receipt_sha256": None,
            "terminal_sidecar_sha256": None,
        }
        if ordinal in eligible:
            provider_plan = {
                key: value
                for key, value in full_plan.items()
                if key != "terminal_compilation"
            }
            compact_body = {
                "format": answer.full100_cli.COMPACT_PLAN_FORMAT,
                "full_answer_plan_receipt_sha256": full_plan[
                    "answer_plan_receipt_sha256"
                ],
                "provider_plan": provider_plan,
                "provider_plan_sha256": identity_sha256(provider_plan),
                "terminal_compilation_receipt_sha256": full_plan[
                    "terminal_compilation_receipt_sha256"
                ],
            }
            base["terminal_answer_plan"] = {
                **compact_body,
                "compact_plan_receipt_sha256": identity_sha256(compact_body),
            }
            base["terminal_question_receipt_sha256"] = _sha(
                f"terminal question {ordinal}"
            )
            base["terminal_sidecar_sha256"] = _sha(f"sidecar {ordinal % 10}")
            full_plans[ordinal] = full_plan
        row = _with_receipt(base, "question_construction_receipt_sha256")
        questions.append(row)
        if ordinal not in eligible:
            passthroughs.append(row)
    payload = {
        "eligible_count": answer.ELIGIBLE_COUNT,
        "format": answer.full100_cli.FORMAT,
        "gold_loaded": False,
        "new_provider_calls": 0,
        "ordinal_cli_routing_available": False,
        "passthrough_count": answer.PASSTHROUGH_COUNT,
        "production_ordinal_routing_enabled": False,
        "question_count": answer.QUESTION_COUNT,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
    }
    construction = _artifact(root / "full100.json", "full100", payload)
    replay = _artifact(root / "full100-replay.json", "full100", payload)
    promotion_payload = {
        "format": answer.terminal_cli.FORMAT,
        "question_count": len(answer.EXACT_ORDINALS),
    }
    promotion = _artifact(
        root / "promotion-terminal.json", "promotion terminal", promotion_payload
    )
    promotion_replay = _artifact(
        root / "promotion-terminal-replay.json",
        "promotion terminal",
        promotion_payload,
    )
    provider_plans = tuple(
        {
            key: value
            for key, value in full_plans[ordinal].items()
            if key != "terminal_compilation"
        }
        for ordinal in sorted(eligible)
    )
    promotion_plans = tuple(full_plans[ordinal] for ordinal in answer.EXACT_ORDINALS)
    audit = _promotion_audit(root, promotion)
    sources = answer._VerifiedSources(  # noqa: SLF001
        full100_construction=construction,
        full100_replay=replay,
        provider_plans=provider_plans,
        passthroughs=tuple(passthroughs),
        promotion_construction=promotion,
        promotion_replay=promotion_replay,
        promotion_plans=promotion_plans,
        promotion_audit=audit,
    )
    return _Fixture(
        sources=sources,
        construction=construction,
        replay=replay,
        provider_plans=provider_plans,
        passthroughs=tuple(passthroughs),
        promotion=promotion,
        promotion_plans=promotion_plans,
        audit=audit,
    )


def _strict_source_fixture(root: Path):
    import test_run_locked_semantic_global_terminal_full100_construction as support

    original, _ = support._source_fixture(root / "original")  # noqa: SLF001
    eligible = set(answer.EXACT_ORDINALS)
    eligible.update(
        ordinal
        for ordinal in answer.ALL_ORDINALS
        if len(eligible) < answer.ELIGIBLE_COUNT
    )
    eligible_ordinals = tuple(sorted(eligible))
    parent = original.parent
    gate_rows: list[dict[str, Any]] = []
    for ordinal, parent_row in enumerate(original.parent_rows):
        is_eligible = ordinal in eligible
        eligibility = _with_receipt(
            {
                "eligible": is_eligible,
                "format": "synthetic-eligibility-v1",
                "reasons": ["synthetic_open_frontier"] if is_eligible else [],
            },
            "receipt_sha256",
        )
        gate_rows.append(
            _with_receipt(
                {
                    "current_prediction": parent_row["prediction"],
                    "current_prediction_sha256": parent_row["prediction_sha256"],
                    "dated_question_sha256": parent_row["dated_question_sha256"],
                    "eligibility": eligibility,
                    "namespace_id": identity_sha256({"namespace": ordinal % 10}),
                    "ordinal": ordinal,
                    "question_id": parent_row["question_id"],
                    "question_sha256": parent_row["question_sha256"],
                    "source_answer_row_sha256": identity_sha256(parent_row),
                },
                "gate_row_receipt_sha256",
            )
        )
    gate_body = {
        "bindings": {"answer_artifact_sha256": parent.sha256},
        "eligibility_policy": original.gate.payload["eligibility_policy"],
        "eligible_count": answer.ELIGIBLE_COUNT,
        "eligible_ordinals": list(eligible_ordinals),
        "format": answer.full100_cli.r7_cli.GATE_FORMAT,
        "question_count": answer.QUESTION_COUNT,
        "questions": gate_rows,
    }
    gate = support._publish(  # noqa: SLF001
        root / "gate.json",
        support._artifact_payload(gate_body, "gate_identity_sha256"),  # noqa: SLF001
    )
    vector_body = {
        "format": answer.full100_cli.r7_cli.VECTOR_FORMAT,
        "gate_artifact_sha256": gate.sha256,
        "question_count": answer.ELIGIBLE_COUNT,
        "rows": [{"ordinal": ordinal} for ordinal in eligible_ordinals],
    }
    vector_payload = support._artifact_payload(  # noqa: SLF001
        vector_body, "vector_identity_sha256"
    )
    vectors = support._publish(root / "vectors.json", vector_payload)  # noqa: SLF001
    vector_replay = support._publish(  # noqa: SLF001
        root / "vector-replay.json", vector_payload
    )
    r7_rows = [
        _with_receipt(
            {
                "dated_question_sha256": original.parent_rows[ordinal][
                    "dated_question_sha256"
                ],
                "mode": (
                    "residual_synthesis" if ordinal in eligible else "not_eligible"
                ),
                "ordinal": ordinal,
                "question_id": original.parent_rows[ordinal]["question_id"],
                "question_sha256": original.parent_rows[ordinal]["question_sha256"],
            },
            "question_receipt_sha256",
        )
        for ordinal in answer.ALL_ORDINALS
    ]
    r7_body = {
        "bindings": {
            "gate_artifact_sha256": gate.sha256,
            "query_vector_artifact_sha256": vectors.sha256,
            "query_vector_replay_artifact_sha256": vector_replay.sha256,
        },
        "format": answer.full100_cli.r7_cli.CONSTRUCTION_FORMAT,
        "question_count": answer.QUESTION_COUNT,
        "questions": r7_rows,
        "residual_search_policy": original.r7.payload["residual_search_policy"],
    }
    r7 = support._publish(  # noqa: SLF001
        root / "r7.json",
        support._artifact_payload(r7_body, "construction_identity_sha256"),  # noqa: SLF001
    )
    sources = answer.full100_cli._validate_source_artifacts(  # noqa: SLF001
        gate, r7, vectors, vector_replay, parent
    )
    return sources, eligible_ordinals


def _strict_plan(
    ordinal: int,
    gate_row: Mapping[str, Any],
    *,
    sealed_sources: Mapping[str, Any],
    audit_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    question = f"Question {ordinal}?"
    dated = f"[Question asked at 2026/08/29 12:00] {question}"
    parent = gate_row["current_prediction"]
    if audit_plan is None:
        provider_input = {
            "dated_question": dated,
            "format": "synthetic-strict-full100-provider-input-v1",
            "protected_parent_fallback": {"prediction": parent},
        }
        allowed: list[str] = []
        compilation: dict[str, Any] = {
            "format": answer.terminal_cli.TERMINAL_COMPILATION_FORMAT,
            "new_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    else:
        provider_input = deepcopy(audit_plan["provider_input"])
        provider_input["dated_question"] = dated
        provider_input["protected_parent_fallback"] = {"prediction": parent}
        allowed = list(audit_plan["allowed_handle_ids"])
        compilation = deepcopy(audit_plan["terminal_compilation"])
        compilation.setdefault(
            "format", answer.terminal_cli.TERMINAL_COMPILATION_FORMAT
        )
    compilation.update(
        {
            "policy": answer.full100_cli.SemanticGlobalTerminalPolicy().projection(),
            "sealed_sources": dict(sealed_sources),
        }
    )
    compilation_body = {
        key: value for key, value in compilation.items() if key != "receipt_sha256"
    }
    compilation["receipt_sha256"] = identity_sha256(compilation_body)
    messages = answer.render_final_messages(provider_input)
    body = {
        "allowed_handle_ids": allowed,
        "dated_question": dated,
        "dated_question_sha256": gate_row["dated_question_sha256"],
        "format": answer.terminal_cli.ANSWER_PLAN_FORMAT,
        "handle_group_by_id": {handle: f"G-{handle}" for handle in allowed},
        "hard_prompt_token_cap": answer.HARD_PROMPT_TOKEN_CAP,
        "messages_sha256": identity_sha256(list(messages)),
        "ordinal": ordinal,
        "output_token_reserve": answer.OUTPUT_TOKEN_RESERVE,
        "parent_prediction": parent,
        "parent_prediction_sha256": quote_sha256(parent),
        "preservation_requirements": {},
        "prompt_token_proxy": count_chat_prompt_token_proxy(messages),
        "provider_input": provider_input,
        "provider_input_sha256": identity_sha256(provider_input),
        "question_id": gate_row["question_id"],
        "question_sha256": gate_row["question_sha256"],
        "route_id": answer.TERMINAL_ROUTE_ID,
        "source_artifact_bindings": dict(sealed_sources),
        "story_coherence": {},
        "terminal_compilation": compilation,
        "terminal_compilation_receipt_sha256": compilation["receipt_sha256"],
        "validation_contract": {},
    }
    return {**body, "answer_plan_receipt_sha256": identity_sha256(body)}


def _strict_terminalized(
    sources,
    eligible_ordinals: tuple[int, ...],
    audit_plans: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    sealed_sources = answer.full100_cli.TerminalSealedSources(
        protected_owner_artifact_sha256=sources.r7.sha256,
        residual_artifact_sha256=sources.r7.sha256,
        parent_artifact_sha256=sources.gate.sha256,
    ).projection()
    questions: list[dict[str, Any]] = []
    plans: dict[int, dict[str, Any]] = {}
    by_namespace: dict[str, list[str]] = {}
    for ordinal in eligible_ordinals:
        gate = sources.gate_rows[ordinal]
        plan = _strict_plan(
            ordinal,
            gate,
            sealed_sources=sealed_sources,
            audit_plan=audit_plans.get(ordinal),
        )
        plans[ordinal] = plan
        question = _with_receipt(
            {
                "dated_question_sha256": gate["dated_question_sha256"],
                "namespace_id": gate["namespace_id"],
                "new_provider_calls": 0,
                "ordinal": ordinal,
                "question_id": gate["question_id"],
                "question_sha256": gate["question_sha256"],
                "r7_exact_question_rebuilt": True,
                "r7_question_receipt_sha256": sources.r7_rows[ordinal][
                    "question_receipt_sha256"
                ],
                "retained_transformer_token_state_bytes": 0,
                "terminal_answer_plan": plan,
            },
            "question_assay_receipt_sha256",
        )
        questions.append(question)
        by_namespace.setdefault(gate["namespace_id"], []).append(
            question["question_assay_receipt_sha256"]
        )
    namespaces = [
        _with_receipt(
            {
                "namespace_id": namespace,
                "question_assay_receipt_sha256s": by_namespace[namespace],
            },
            "namespace_assay_receipt_sha256",
        )
        for namespace in sorted(by_namespace)
    ]
    body = {
        "diagnostic_population_explicitly_supplied": True,
        "format": answer.full100_cli.v7_cli.FORMAT,
        "global_policy": answer.full100_cli.SemanticGlobalCompletionPolicy().projection(),
        "gold_loaded": False,
        "local_policy": answer.full100_cli.SourceGroupReinjectionPolicy().projection(),
        "namespace_receipts": namespaces,
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": len(questions),
        "questions": questions,
        "r7_bindings": {
            "construction_artifact_sha256": sources.r7.sha256,
            "gate_artifact_sha256": sources.gate.sha256,
            "query_vector_artifact_sha256": sources.vectors.sha256,
            "query_vector_replay_artifact_sha256": sources.vector_replay.sha256,
        },
        "retained_transformer_token_state_bytes": 0,
        "source_indexes_rebuilt_not_serialized": True,
        "v6_v7_single_resident_index_pass": True,
        "v7_replay_count": len(questions),
    }
    return {**body, "construction_identity_sha256": identity_sha256(body)}, plans


def _build(fixture: _Fixture):
    return answer.build_preflight_payload(
        fixture.construction,
        fixture.replay,
        fixture.provider_plans,
        fixture.passthroughs,
        promotion_construction=fixture.sources.promotion_construction,
        promotion_replay=fixture.sources.promotion_replay,
        promotion_plans=fixture.promotion_plans,
        promotion_audit=fixture.audit,
        model=answer.DEFAULT_MODEL,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
    )


def _args(tmp_path: Path, fixture: _Fixture) -> SimpleNamespace:
    return SimpleNamespace(
        approve_provider_release=False,
        expected_full100_construction_sha256=fixture.construction.sha256,
        expected_full100_replay_sha256=fixture.replay.sha256,
        expected_postseal_audit_sha256=fixture.audit.sha256,
        expected_promotion_terminal_construction_sha256=fixture.promotion.sha256,
        expected_promotion_terminal_replay_sha256=fixture.promotion.sha256,
        full100_terminal_root=tmp_path / "full100-root",
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=tmp_path / "answer-root",
        postseal_audit=fixture.audit.path,
        promotion_from_full100=False,
        promotion_terminal_root=tmp_path / "promotion-root",
        r7_construction=None,
        expected_r7_construction_sha256=None,
    )


def _install_sources(
    monkeypatch: pytest.MonkeyPatch, fixture: _Fixture
) -> None:
    monkeypatch.setattr(answer, "_load_verified_sources", lambda _args: fixture.sources)

    def audit_reader(
        _path,
        expected_sha256,
        *,
        expected_terminal_construction_sha256,
        expected_terminal_replay_sha256,
        **_kwargs,
    ):
        assert expected_sha256 == fixture.audit.sha256
        assert expected_terminal_construction_sha256 == fixture.promotion.sha256
        assert expected_terminal_replay_sha256 == fixture.promotion.sha256
        return fixture.audit

    monkeypatch.setattr(
        answer.postseal_cli, "load_verified_promotion_audit", audit_reader
    )


def test_direct_full100_promotion_uses_one_canonical_source_and_explicit_r7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    args = _args(tmp_path, fixture)
    shared_root = tmp_path / "shared-full100"
    args.promotion_from_full100 = True
    args.full100_terminal_root = shared_root
    args.promotion_terminal_root = shared_root / "."
    args.expected_promotion_terminal_construction_sha256 = fixture.construction.sha256
    args.expected_promotion_terminal_replay_sha256 = fixture.replay.sha256
    args.r7_construction = tmp_path / "successor-r7.json"
    args.expected_r7_construction_sha256 = _sha("successor R7")
    captured: dict[str, Any] = {}
    detailed = SimpleNamespace(
        construction=fixture.construction,
        replay=fixture.replay,
        provider_plans=fixture.provider_plans,
        passthroughs=fixture.passthroughs,
        exact11_terminal_plans=fixture.promotion_plans,
        residual_policy=SimpleNamespace(
            classifier_mode=answer.EVIDENCE_CONSERVING_RESIDUAL_CLASSIFIER_MODE
        ),
    )

    def load_detailed(root, construction_sha, replay_sha, **kwargs):
        captured.update(
            root=root,
            construction_sha=construction_sha,
            replay_sha=replay_sha,
            **kwargs,
        )
        return detailed

    def read_audit(_path, _sha256, *, construction_sha256, replay_sha256):
        captured["audit_construction_sha"] = construction_sha256
        captured["audit_replay_sha"] = replay_sha256
        return fixture.audit

    monkeypatch.setattr(
        answer.full100_cli,
        "load_verified_full100_construction_detailed",
        load_detailed,
    )
    monkeypatch.setattr(
        answer.full100_cli,
        "load_verified_full100_construction",
        lambda *_args, **_kwargs: pytest.fail("direct mode used the legacy accessor"),
    )
    monkeypatch.setattr(
        answer.terminal_cli,
        "load_verified_terminal_assay",
        lambda *_args, **_kwargs: pytest.fail("direct mode opened a reduced assay"),
    )
    monkeypatch.setattr(answer, "_read_promotion_audit", read_audit)

    verified = answer._load_verified_sources(args)  # noqa: SLF001

    assert verified.full100_construction is verified.promotion_construction
    assert verified.full100_replay is verified.promotion_replay
    assert verified.promotion_plans == fixture.promotion_plans
    assert captured["root"] == shared_root
    assert captured["construction_sha"] == fixture.construction.sha256
    assert captured["replay_sha"] == fixture.replay.sha256
    assert captured["r7_path"] == args.r7_construction
    assert captured["expected_r7_sha256"] == args.expected_r7_construction_sha256
    assert captured["audit_construction_sha"] == fixture.construction.sha256
    assert captured["audit_replay_sha"] == fixture.replay.sha256


def test_direct_full100_promotion_rejects_different_sha_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    args = _args(tmp_path, fixture)
    args.promotion_from_full100 = True
    args.promotion_terminal_root = args.full100_terminal_root
    args.expected_promotion_terminal_construction_sha256 = _sha("foreign full100")
    args.r7_construction = tmp_path / "successor-r7.json"
    args.expected_r7_construction_sha256 = _sha("successor R7")
    monkeypatch.setattr(
        answer.full100_cli,
        "load_verified_full100_construction_detailed",
        lambda *_args, **_kwargs: pytest.fail("mismatched roots reached the loader"),
    )

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="canonical same-root/same-SHA",
    ):
        answer._load_verified_sources(args)  # noqa: SLF001


def test_preflight_seals_fixed_68_32_population_and_hard_budget(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload, prompts = _build(fixture)
    artifact = _artifact(tmp_path / "preflight.json", "preflight", payload)
    verified_prompts, prompt_rows, passthrough_rows = answer._validate_preflight(  # noqa: SLF001
        artifact
    )

    assert len(prompts) == len(verified_prompts) == len(prompt_rows) == 68
    assert len(passthrough_rows) == 32
    assert payload["question_count"] == 100
    assert payload["required_authorized_provider_calls"] == 68
    assert payload["ordinal_cli_routing_available"] is False
    assert payload["production_ordinal_routing_enabled"] is False
    assert set(payload["eligible_ordinals"]).isdisjoint(
        payload["passthrough_ordinals"]
    )
    assert sorted(payload["eligible_ordinals"] + payload["passthrough_ordinals"]) == list(
        range(100)
    )
    assert all(
        row["prompt_token_proxy"] + answer.OUTPUT_TOKEN_RESERVE
        <= answer.HARD_PROMPT_TOKEN_CAP
        for row in prompt_rows
    )
    assert payload["retained_transformer_token_state_bytes"] == 0


def test_strict_construction_and_promotion_readers_compose_before_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_root = str(Path(__file__).resolve().parent)
    sys.path.insert(0, test_root)
    try:
        import test_audit_locked_semantic_global_terminal_postseal as audit_support
        import test_run_locked_semantic_global_terminal_full100_construction as construction_support
    finally:
        sys.path.remove(test_root)

    sources, eligible_ordinals = _strict_source_fixture(tmp_path / "sources")
    (
        _unused_construction,
        _unused_replay,
        audit_base_plans,
        target_plan,
        targets,
    ) = audit_support._fixture(tmp_path / "audit")  # noqa: SLF001
    audit_by_ordinal = {row["ordinal"]: row for row in audit_base_plans}
    terminalized, full_plans = _strict_terminalized(
        sources, eligible_ordinals, audit_by_ordinal
    )
    monkeypatch.setattr(
        answer.full100_cli,
        "_validate_terminal_answer_plan",
        lambda raw, _question: raw,
    )
    bundle = answer.full100_cli._compose_payload(  # noqa: SLF001
        sources=sources,
        terminalized=terminalized,
        terminal_policy=answer.full100_cli.SemanticGlobalTerminalPolicy(),
    )
    full100_root = tmp_path / "strict-full100"
    full100_sha = construction_support._publish_pair(  # noqa: SLF001
        full100_root, bundle.manifest, bundle.sidecars
    )
    construction, replay, provider_plans, passthroughs = (
        answer.full100_cli.load_verified_full100_construction(
            full100_root,
            full100_sha,
            full100_sha,
            gate_path=sources.gate.path,
            expected_gate_sha256=sources.gate.sha256,
            r7_path=sources.r7.path,
            expected_r7_sha256=sources.r7.sha256,
            vectors_path=sources.vectors.path,
            vector_replay_path=sources.vector_replay.path,
            expected_vector_sha256=sources.vectors.sha256,
            parent_path=sources.parent.path,
            expected_parent_sha256=sources.parent.sha256,
        )
    )
    assert len(provider_plans) == 68
    assert len(passthroughs) == 32

    promotion_plans = tuple(full_plans[ordinal] for ordinal in answer.EXACT_ORDINALS)
    promotion_payload = {
        "format": answer.terminal_cli.FORMAT,
        "gold_loaded": False,
        "plan_population_sha256": identity_sha256(
            [row["answer_plan_receipt_sha256"] for row in promotion_plans]
        ),
        "question_count": len(promotion_plans),
        "retained_transformer_token_state_bytes": 0,
    }
    promotion_root = tmp_path / "strict-promotion"
    promotion_construction, _ = publish_sealed_json(
        promotion_root / answer.terminal_cli.CONSTRUCTION_NAME,
        promotion_payload,
    )
    promotion_replay, _ = publish_sealed_json(
        promotion_root / answer.terminal_cli.REPLAY_NAME,
        promotion_payload,
    )
    assert promotion_construction.sha256 == promotion_replay.sha256

    witness_manifest, witness_index = audit_support._publish_manifest(  # noqa: SLF001
        tmp_path / "audit",
        target_plan,
        targets,
        promotion_plans,
    )
    atom_manifest, semantic_atoms = audit_support._publish_semantic_atom_manifest(  # noqa: SLF001
        tmp_path / "audit",
        target_plan,
        targets,
        promotion_plans,
        witness_manifest,
    )
    audit_payload = answer.postseal_cli.build_audit(
        construction=promotion_construction,
        replay=promotion_replay,
        plans=promotion_plans,
        target_plan=target_plan,
        source_targets=targets,
        witness_manifest=witness_manifest,
        witness_index=witness_index,
        semantic_atom_manifest=atom_manifest,
        semantic_atoms=semantic_atoms,
    )
    promotion_audit, _ = publish_sealed_json(
        tmp_path / "audit" / "strict-promotion-audit.json", audit_payload
    )
    verified_audit = answer.postseal_cli.load_verified_promotion_audit(
        promotion_audit.path,
        promotion_audit.sha256,
        expected_terminal_construction_sha256=promotion_construction.sha256,
        expected_terminal_replay_sha256=promotion_replay.sha256,
        expected_target_plan_sha256=target_plan.sha256,
        expected_target_plan_identity_sha256=target_plan.payload["plan_sha256"],
        expected_witness_manifest_sha256=witness_manifest.sha256,
        expected_witness_manifest_identity_sha256=witness_manifest.payload[
            "manifest_identity_sha256"
        ],
        expected_semantic_atom_manifest_sha256=atom_manifest.sha256,
        expected_semantic_atom_manifest_identity_sha256=atom_manifest.payload[
            "manifest_identity_sha256"
        ],
        expected_semantic_atom_population_sha256=atom_manifest.payload[
            "atom_population_sha256"
        ],
    )
    for name, value in (
        ("DEFAULT_TARGET_PLAN_SHA256", target_plan.sha256),
        ("DEFAULT_TARGET_PLAN_IDENTITY_SHA256", target_plan.payload["plan_sha256"]),
        ("DEFAULT_WITNESS_MANIFEST_SHA256", witness_manifest.sha256),
        (
            "DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256",
            witness_manifest.payload["manifest_identity_sha256"],
        ),
        ("DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256", atom_manifest.sha256),
        (
            "DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256",
            atom_manifest.payload["manifest_identity_sha256"],
        ),
        (
            "DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256",
            atom_manifest.payload["atom_population_sha256"],
        ),
    ):
        monkeypatch.setattr(answer.postseal_cli, name, value)

    payload, prompts = answer.build_preflight_payload(
        construction,
        replay,
        provider_plans,
        passthroughs,
        promotion_construction=promotion_construction,
        promotion_replay=promotion_replay,
        promotion_plans=promotion_plans,
        promotion_audit=verified_audit,
        model=answer.DEFAULT_MODEL,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
    )
    assert len(prompts) == 68
    assert payload["postseal_semantic_atom_final_usable_count"] == 26
    assert payload["full100_construction_artifact_sha256"] == full100_sha
    assert payload["promotion_terminal_construction_artifact_sha256"] == (
        promotion_construction.sha256
    )


def test_promotion_requires_byte_identical_exact11_projection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    promotion = [deepcopy(row) for row in fixture.promotion_plans]
    promotion[0]["source_artifact_bindings"]["residual"] = _sha("other residual")
    unsigned = dict(promotion[0])
    unsigned.pop("answer_plan_receipt_sha256")
    promotion[0]["answer_plan_receipt_sha256"] = identity_sha256(unsigned)

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="not byte-identical",
    ):
        answer.build_preflight_payload(
            fixture.construction,
            fixture.replay,
            fixture.provider_plans,
            fixture.passthroughs,
            promotion_construction=fixture.sources.promotion_construction,
            promotion_replay=fixture.sources.promotion_replay,
            promotion_plans=promotion,
            promotion_audit=fixture.audit,
            model=answer.DEFAULT_MODEL,
            gateway_url=answer.DEFAULT_GATEWAY_URL,
            max_concurrency=3,
        )


def test_preflight_rejects_passthrough_drift_and_incomplete_atom_gate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    passthroughs = [deepcopy(row) for row in fixture.passthroughs]
    passthroughs[0]["passthrough_prediction"] = "changed"
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="passthrough",
    ):
        answer.build_preflight_payload(
            fixture.construction,
            fixture.replay,
            fixture.provider_plans,
            passthroughs,
            promotion_construction=fixture.sources.promotion_construction,
            promotion_replay=fixture.sources.promotion_replay,
            promotion_plans=fixture.promotion_plans,
            promotion_audit=fixture.audit,
            model=answer.DEFAULT_MODEL,
            gateway_url=answer.DEFAULT_GATEWAY_URL,
            max_concurrency=3,
        )

    bad_audit = _promotion_audit(tmp_path, fixture.promotion, atom_count=25)
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="promotion binding changed",
    ):
        answer.build_preflight_payload(
            fixture.construction,
            fixture.replay,
            fixture.provider_plans,
            fixture.passthroughs,
            promotion_construction=fixture.sources.promotion_construction,
            promotion_replay=fixture.sources.promotion_replay,
            promotion_plans=fixture.promotion_plans,
            promotion_audit=bad_audit,
            model=answer.DEFAULT_MODEL,
            gateway_url=answer.DEFAULT_GATEWAY_URL,
            max_concurrency=3,
        )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda payload: payload.update({"source_question_population_sha256": _sha("other population")}),
        lambda payload: payload.update({"caller_selected_ordinals": [14]}),
        lambda payload: payload["physical_prompt_rows"][0].update(
            {"prompt_token_proxy": answer.MAX_CHAT_PROMPT_TOKENS + 1}
        ),
    ),
    ids=("source-population", "injected-routing-field", "prompt-budget"),
)
def test_sealed_preflight_rejects_resealed_population_control_and_budget_drift(
    tmp_path: Path, mutator
) -> None:
    fixture = _fixture(tmp_path)
    payload, _ = _build(fixture)
    mutated = deepcopy(payload)
    mutator(mutated)

    with pytest.raises(
        (answer.LockedSemanticGlobalTerminalFull100AnswerError, ValueError),
    ):
        answer._validate_preflight(  # noqa: SLF001
            _artifact(tmp_path / "mutated-preflight.json", "mutated", mutated)
        )


def test_release_is_explicit_and_provider_authorization_is_exact_68(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_sources(monkeypatch, fixture)
    args = _args(tmp_path, fixture)
    preflight = answer.run_preflight(args)
    args.expected_preflight_sha256 = preflight["preflight_sha256"]

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="explicit provider-release approval",
    ):
        answer.run_approve_release(args)
    args.approve_provider_release = True
    released = answer.run_approve_release(args)
    args.expected_release_sha256 = released["release_sha256"]
    args.enable_provider = True
    args.authorized_provider_calls = 67
    args.api_key_env = "SEALED_KEY"
    monkeypatch.setattr(
        answer,
        "load_dotenv",
        lambda: pytest.fail("environment opened before exact authorization check"),
    )

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="exactly equal remaining",
    ):
        answer.run_provider(args)
    assert not (Path(args.output_root) / answer.CHECKPOINT_DIR_NAME).exists()


class _FakeCompletions:
    def __init__(self, parents_by_messages: Mapping[str, str]) -> None:
        self.parents_by_messages = dict(parents_by_messages)
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls.append(dict(request))
        messages_sha = identity_sha256(request["messages"])
        parent = self.parents_by_messages[messages_sha]
        completion = json.dumps(
            {
                "decision": "keep_parent",
                "prediction": parent,
                "used_handle_ids": [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=completion),
                ),
            ),
            id=f"fake-full100-{len(self.calls):03d}",
            model=answer.DEFAULT_MODEL,
            usage=None,
        )


class _FakeClient:
    max_retries = 0

    def __init__(self, parents_by_messages: Mapping[str, str]) -> None:
        self.completions = _FakeCompletions(parents_by_messages)
        self.chat = SimpleNamespace(completions=self.completions)

    def close(self) -> None:
        return None


def test_materialize_replay_merges_ordered_100_row_judge_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _install_sources(monkeypatch, fixture)
    args = _args(tmp_path, fixture)
    preflight_result = answer.run_preflight(args)
    args.expected_preflight_sha256 = preflight_result["preflight_sha256"]
    args.approve_provider_release = True
    release_result = answer.run_approve_release(args)
    args.expected_release_sha256 = release_result["release_sha256"]
    preflight = read_sealed_json(Path(args.output_root) / answer.PREFLIGHT_NAME)
    prompts, prompt_rows, _ = answer._validate_preflight(preflight)  # noqa: SLF001
    release = read_sealed_json(Path(args.output_root) / answer.RELEASE_NAME)
    client = _FakeClient(
        {row["messages_sha256"]: row["parent_prediction"] for row in prompt_rows}
    )
    runtime = answer._runtime(  # noqa: SLF001
        preflight, release, prompts, args=args, client=client
    )
    try:
        seeded = runtime.run()
    finally:
        runtime.close()
    assert seeded.usage.physical_calls == 68
    assert len(client.completions.calls) == 68

    materialized = answer.run_materialize(args)
    run = read_sealed_json(Path(args.output_root) / answer.RUN_NAME)
    args.expected_run_sha256 = run.sha256
    replayed = answer.run_replay(args)
    replay = read_sealed_json(Path(args.output_root) / answer.REPLAY_NAME)
    verified_run, verified_replay, judge_rows = answer.load_verified_answer_run(
        args.output_root,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=run.sha256,
        expected_replay_sha256=replay.sha256,
        postseal_audit=args.postseal_audit,
        expected_postseal_audit_sha256=args.expected_postseal_audit_sha256,
    )

    assert materialized["checkpoint_hits"] == 68
    assert materialized["passthrough_count"] == 32
    assert replayed["byte_identical"] is True
    assert replayed["physical_provider_calls"] == 0
    assert verified_run.sha256 == run.sha256
    assert verified_replay.sha256 == replay.sha256
    assert len(judge_rows) == 100
    assert tuple(row["ordinal"] for row in judge_rows) == tuple(range(100))
    assert sum(
        row["answer_mode"] == answer.TERMINAL_MODE
        for row in run.payload["questions"]
    ) == 68
    passthrough_results = [
        row
        for row in run.payload["questions"]
        if row["answer_mode"] == answer.PASSTHROUGH_MODE
    ]
    assert len(passthrough_results) == 32
    assert all(row["call_key_sha256"] is None for row in passthrough_results)
    assert run.payload["retained_transformer_token_state_bytes"] == 0

    checkpoint_batch = answer._checkpoint_batch(  # noqa: SLF001
        preflight, release, prompts, args=args, client=None
    )
    bad_batch = deepcopy(run.payload)
    bad_batch["completion_batch"]["provenance"]["retries"] = 1
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="completion batch",
    ):
        answer._validate_run(  # noqa: SLF001
            _artifact(tmp_path / "bad-batch.json", "bad batch", bad_batch),
            preflight=preflight,
            expected_release_sha256=release.sha256,
            expected_batch=checkpoint_batch,
        )

    bad_aggregate = deepcopy(run.payload)
    bad_aggregate["changed_prediction_count"] = 1
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="aggregate counts",
    ):
        answer._validate_run(  # noqa: SLF001
            _artifact(tmp_path / "bad-aggregate.json", "bad aggregate", bad_aggregate),
            preflight=preflight,
            expected_release_sha256=release.sha256,
            expected_batch=checkpoint_batch,
        )

    bad_semantics = deepcopy(run.payload)
    terminal_ordinal = bad_semantics["eligible_ordinals"][0]
    terminal_row = bad_semantics["questions"][terminal_ordinal]
    terminal_row["decision"] = "replace"
    terminal_row["prediction_source"] = (
        "typed_final_model_attested_replacement_v1"
    )
    unsigned = {
        key: value
        for key, value in terminal_row.items()
        if key != "source_row_sha256"
    }
    terminal_row["source_row_sha256"] = identity_sha256(unsigned)
    bad_semantics["judge_rows"][terminal_ordinal] = answer.judge_row_projection(
        terminal_row
    )
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalFull100AnswerError,
        match="terminal result provenance",
    ):
        answer._validate_run(  # noqa: SLF001
            _artifact(tmp_path / "bad-semantics.json", "bad semantics", bad_semantics),
            preflight=preflight,
            expected_release_sha256=release.sha256,
            expected_batch=checkpoint_batch,
        )


def test_cli_exposes_no_caller_controlled_ordinal_selection() -> None:
    parser = answer.build_parser()
    preflight = parser.parse_args(
        [
            "preflight",
            "--expected-full100-construction-sha256",
            "a" * 64,
            "--expected-full100-replay-sha256",
            "b" * 64,
            "--expected-promotion-terminal-construction-sha256",
            "c" * 64,
            "--expected-promotion-terminal-replay-sha256",
            "d" * 64,
            "--postseal-audit",
            "audit.json",
            "--expected-postseal-audit-sha256",
            "e" * 64,
        ]
    )
    provider = parser.parse_args(
        [
            "provider-run",
            "--postseal-audit",
            "audit.json",
            "--expected-postseal-audit-sha256",
            "a" * 64,
            "--expected-preflight-sha256",
            "b" * 64,
            "--expected-release-sha256",
            "c" * 64,
            "--authorized-provider-calls",
            "68",
        ]
    )
    assert not hasattr(preflight, "ordinals")
    assert not hasattr(provider, "ordinals")
    assert not hasattr(provider, "full100_terminal_root")
    assert not hasattr(provider, "promotion_terminal_root")
