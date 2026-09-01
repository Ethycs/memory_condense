from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tracemalloc
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_semantic_global_terminal_full100_construction as full100
from tools import (
    run_locked_semantic_global_terminal_full100_compact_resumable as compact_resumable,
)
from tools import run_locked_semantic_global_terminal_full100_resumable as resumable
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.semantic_global_completion import (
    SemanticGlobalCompletionPolicy,
)
from tools.matched_eval.semantic_global_terminal_adapter import (
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
)
from tools.matched_eval.semantic_residual_eligibility import (
    SemanticResidualEligibilityPolicy,
)
from tools.matched_eval.semantic_residual_search import SemanticResidualPolicy
from tools.matched_eval.source_group_reinjection import (
    SourceGroupReinjectionPolicy,
)


def _with_receipt(body: dict, key: str = "receipt_sha256") -> dict:
    return {**body, key: identity_sha256(body)}


def _artifact_payload(body: dict, key: str) -> dict:
    return {**body, key: identity_sha256(body)}


def _publish(path: Path, payload: dict) -> SealedArtifact:
    artifact, created = publish_sealed_json(path, payload)
    assert created is True
    return artifact


@dataclass(frozen=True)
class _Fixture:
    sources: full100._SourceArtifacts
    terminalized: dict
    bundle: full100.Full100ConstructionBundle
    payload: dict
    root: Path


def _fake_plan_validator(raw: dict, _question: dict) -> dict:
    assert type(raw) is dict
    return raw


def _source_fixture(tmp_path: Path) -> tuple[full100._SourceArtifacts, tuple[int, ...]]:
    source_root = tmp_path / "sources"
    excluded = set(range(0, 96, 3))
    eligible_ordinals = tuple(
        ordinal for ordinal in range(full100.QUESTION_COUNT) if ordinal not in excluded
    )
    assert len(eligible_ordinals) == full100.ELIGIBLE_COUNT

    parent_rows: list[dict] = []
    for ordinal in range(full100.QUESTION_COUNT):
        question = f"Question {ordinal}?"
        dated = f"[Question asked at 2026/08/29 12:00] {question}"
        prediction = f"Parent prediction {ordinal}."
        parent_rows.append(
            {
                "dated_question_sha256": quote_sha256(dated),
                "ordinal": ordinal,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "question_id": f"q-{ordinal}",
                "question_sha256": quote_sha256(question),
            }
        )
    parent = _publish(
        source_root / "parent.json",
        {
            "format": full100.v3_cli.FORMAT,
            "question_count": full100.QUESTION_COUNT,
            "questions": parent_rows,
        },
    )

    eligibility_policy = SemanticResidualEligibilityPolicy().projection()
    gate_rows: list[dict] = []
    for ordinal, parent_row in enumerate(parent_rows):
        eligible = ordinal in eligible_ordinals
        eligibility = _with_receipt(
            {
                "eligible": eligible,
                "format": "synthetic-eligibility-v1",
                "reasons": ["synthetic_open_frontier"] if eligible else [],
            }
        )
        gate_rows.append(
            _with_receipt(
                {
                    "current_prediction": parent_row["prediction"],
                    "current_prediction_sha256": parent_row["prediction_sha256"],
                    "dated_question_sha256": parent_row["dated_question_sha256"],
                    "eligibility": eligibility,
                    "namespace_id": identity_sha256(
                        {"namespace": ordinal % 10}
                    ),
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
        "eligibility_policy": eligibility_policy,
        "eligible_count": len(eligible_ordinals),
        "eligible_ordinals": list(eligible_ordinals),
        "format": full100.r7_cli.GATE_FORMAT,
        "question_count": full100.QUESTION_COUNT,
        "questions": gate_rows,
    }
    gate = _publish(
        source_root / "gate.json",
        _artifact_payload(gate_body, "gate_identity_sha256"),
    )

    vector_body = {
        "format": full100.r7_cli.VECTOR_FORMAT,
        "gate_artifact_sha256": gate.sha256,
        "question_count": len(eligible_ordinals),
        "rows": [{"ordinal": ordinal} for ordinal in eligible_ordinals],
    }
    vectors_payload = _artifact_payload(vector_body, "vector_identity_sha256")
    vectors = _publish(source_root / "vectors.json", vectors_payload)
    vector_replay = _publish(source_root / "vector-replay.json", vectors_payload)
    assert vectors.sha256 == vector_replay.sha256

    r7_rows = [
        {
            "dated_question_sha256": parent_rows[ordinal]["dated_question_sha256"],
            "mode": (
                "residual_synthesis"
                if ordinal in eligible_ordinals
                else "not_eligible"
            ),
            "ordinal": ordinal,
            "question_id": parent_rows[ordinal]["question_id"],
            "question_sha256": parent_rows[ordinal]["question_sha256"],
        }
        for ordinal in range(full100.QUESTION_COUNT)
    ]
    r7_body = {
        "bindings": {
            "gate_artifact_sha256": gate.sha256,
            "query_vector_artifact_sha256": vectors.sha256,
            "query_vector_replay_artifact_sha256": vector_replay.sha256,
        },
        "format": full100.r7_cli.CONSTRUCTION_FORMAT,
        "question_count": full100.QUESTION_COUNT,
        "questions": r7_rows,
        "residual_search_policy": SemanticResidualPolicy().projection(),
    }
    r7 = _publish(
        source_root / "r7.json",
        _artifact_payload(r7_body, "construction_identity_sha256"),
    )
    sources = full100._validate_source_artifacts(  # noqa: SLF001
        gate, r7, vectors, vector_replay, parent
    )
    return sources, eligible_ordinals


def _terminalized(
    sources: full100._SourceArtifacts, eligible_ordinals: tuple[int, ...]
) -> dict:
    terminal_policy = SemanticGlobalTerminalPolicy().projection()
    sealed_sources = TerminalSealedSources(
        protected_owner_artifact_sha256=sources.r7.sha256,
        residual_artifact_sha256=sources.r7.sha256,
        parent_artifact_sha256=sources.gate.sha256,
    ).projection()
    questions: list[dict] = []
    by_namespace: dict[str, list[str]] = {}
    for ordinal in eligible_ordinals:
        gate_row = sources.gate_rows[ordinal]
        compilation_body = {"policy": terminal_policy}
        compilation = {
            **compilation_body,
            "receipt_sha256": identity_sha256(compilation_body),
        }
        plan_body = {
            "parent_prediction": gate_row["current_prediction"],
            "parent_prediction_sha256": gate_row["current_prediction_sha256"],
            "source_artifact_bindings": sealed_sources,
            "terminal_compilation": compilation,
            "terminal_compilation_receipt_sha256": compilation["receipt_sha256"],
        }
        plan = {
            **plan_body,
            "answer_plan_receipt_sha256": identity_sha256(plan_body),
        }
        question = _with_receipt(
            {
                "dated_question_sha256": gate_row["dated_question_sha256"],
                "namespace_id": gate_row["namespace_id"],
                "new_provider_calls": 0,
                "ordinal": ordinal,
                "question_id": gate_row["question_id"],
                "question_sha256": gate_row["question_sha256"],
                "retained_transformer_token_state_bytes": 0,
                "terminal_answer_plan": plan,
            },
            "question_assay_receipt_sha256",
        )
        questions.append(question)
        by_namespace.setdefault(gate_row["namespace_id"], []).append(
            question["question_assay_receipt_sha256"]
        )
    namespace_receipts = [
        _with_receipt(
            {
                "namespace_id": namespace_id,
                "question_assay_receipt_sha256s": by_namespace[namespace_id],
            },
            "namespace_assay_receipt_sha256",
        )
        for namespace_id in sorted(by_namespace)
    ]
    body = {
        "diagnostic_population_explicitly_supplied": True,
        "format": full100.v7_cli.FORMAT,
        "global_policy": SemanticGlobalCompletionPolicy().projection(),
        "gold_loaded": False,
        "local_policy": SourceGroupReinjectionPolicy().projection(),
        "namespace_receipts": namespace_receipts,
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
    return {**body, "construction_identity_sha256": identity_sha256(body)}


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    monkeypatch.setattr(
        full100, "_validate_terminal_answer_plan", _fake_plan_validator
    )
    sources, eligible_ordinals = _source_fixture(tmp_path)
    terminalized = _terminalized(sources, eligible_ordinals)
    bundle = full100._compose_payload(  # noqa: SLF001
        sources=sources,
        terminalized=terminalized,
        terminal_policy=SemanticGlobalTerminalPolicy(),
    )
    return _Fixture(
        sources=sources,
        terminalized=terminalized,
        bundle=bundle,
        payload=bundle.manifest,
        root=tmp_path,
    )


def _publish_sidecars(
    root: Path, sidecars: tuple[dict, ...]
) -> None:
    for payload in sidecars:
        digest = full100._sidecar_artifact_sha256(payload)  # noqa: SLF001
        artifact = _publish(
            root / full100.SIDECAR_DIR_NAME / f"{digest}.json", payload
        )
        assert artifact.sha256 == digest


def _publish_pair(
    root: Path, payload: dict, sidecars: tuple[dict, ...]
) -> str:
    _publish_sidecars(root, sidecars)
    construction = _publish(root / full100.CONSTRUCTION_NAME, payload)
    replay = _publish(root / full100.REPLAY_NAME, payload)
    assert construction.sha256 == replay.sha256
    return construction.sha256


def _reader(fixture: _Fixture, root: Path, digest: str):
    sources = fixture.sources
    return full100.load_verified_full100_construction(
        root,
        digest,
        digest,
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


def _reseal_top(payload: dict) -> None:
    body = {
        key: value
        for key, value in payload.items()
        if key != "construction_identity_sha256"
    }
    payload["construction_identity_sha256"] = identity_sha256(body)


def _reseal_question(row: dict) -> None:
    body = {
        key: value
        for key, value in row.items()
        if key != "question_construction_receipt_sha256"
    }
    row["question_construction_receipt_sha256"] = identity_sha256(body)


def test_cli_has_no_ordinal_selector() -> None:
    parser = full100.build_parser()
    actions = list(parser._actions)
    subparsers = next(action for action in actions if action.dest == "command")
    for child in subparsers.choices.values():
        assert "ordinals" not in {action.dest for action in child._actions}
        assert "--ordinals" not in child._option_string_actions


def test_build_derives_workset_and_invokes_existing_resident_callback(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(full100, "_load_build_sources", lambda _args: fixture.sources)

    def fake_build(args, *, terminal_compiler):
        captured["ordinals"] = args.ordinals
        captured["terminal_compiler"] = terminal_compiler
        return fixture.terminalized

    monkeypatch.setattr(full100.v7_cli, "build_assay", fake_build)
    payload = full100.build_construction(argparse.Namespace())

    assert captured["ordinals"] == tuple(
        payload["gate_derived_population"]["eligible_ordinals"]
    )
    assert callable(captured["terminal_compiler"])
    assert payload["question_count"] == 100
    assert payload["eligible_count"] == 68
    assert payload["passthrough_count"] == 32
    assert sum(row["mode"] == full100.TERMINAL_MODE for row in payload["questions"]) == 68
    assert sum(
        row["mode"] == full100.PASSTHROUGH_MODE for row in payload["questions"]
    ) == 32


def test_public_reader_returns_exact_partition_and_replay(fixture: _Fixture) -> None:
    root = fixture.root / "valid"
    digest = _publish_pair(root, fixture.payload, fixture.bundle.sidecars)
    construction, replay, plans, passthroughs = _reader(fixture, root, digest)

    assert construction.sha256 == replay.sha256 == digest
    assert len(plans) == 68
    assert len(passthroughs) == 32
    assert all(row["passthrough_prediction"] == row["parent_prediction"] for row in passthroughs)


def test_manifest_is_compact_and_full_audits_exist_once_in_sidecars(
    fixture: _Fixture,
) -> None:
    encoded = json.dumps(fixture.payload, ensure_ascii=False, sort_keys=True)
    assert "local_audit" not in encoded
    assert sum(sidecar["question_count"] for sidecar in fixture.bundle.sidecars) == 68
    assert len(fixture.bundle.sidecars) == fixture.payload[
        "terminal_namespace_sidecar_count"
    ]
    sidecar_shas = {
        full100._sidecar_artifact_sha256(sidecar)  # noqa: SLF001
        for sidecar in fixture.bundle.sidecars
    }
    terminal_rows = [
        row
        for row in fixture.payload["questions"]
        if row["mode"] == full100.TERMINAL_MODE
    ]
    assert {row["terminal_sidecar_sha256"] for row in terminal_rows} == sidecar_shas
    assert all(
        "terminal_compilation"
        not in row["terminal_answer_plan"]["provider_plan"]
        for row in terminal_rows
    )


def test_reader_requires_content_addressed_namespace_sidecars(
    fixture: _Fixture,
) -> None:
    root = fixture.root / "missing-sidecars"
    construction = _publish(root / full100.CONSTRUCTION_NAME, fixture.payload)
    replay = _publish(root / full100.REPLAY_NAME, fixture.payload)

    with pytest.raises(
        full100.LockedSemanticGlobalTerminalFull100Error,
        match="unavailable or unauthenticated",
    ):
        _reader(fixture, root, construction.sha256)
    assert construction.sha256 == replay.sha256


def test_construct_and_replay_publish_identical_manifest_and_sidecars(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fixture.root / "run-lifecycle"
    monkeypatch.setattr(
        full100, "build_construction_bundle", lambda _args: fixture.bundle
    )
    constructed = full100.run_construct(argparse.Namespace(output_root=root))
    replayed = full100.run_replay(
        argparse.Namespace(
            output_root=root,
            expected_construction_output_sha256=constructed[
                "construction_sha256"
            ],
        )
    )

    assert constructed["sidecar_created_count"] == len(fixture.bundle.sidecars)
    assert replayed["byte_identical"] is True
    assert replayed["replay_sha256"] == constructed["construction_sha256"]


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("population", "question population"),
        ("reorder", "identity/mode"),
        ("mode", "identity/mode"),
        ("policy", "roots, policies"),
        ("source", "roots, policies"),
        ("passthrough", "passthrough prediction"),
    ),
)
def test_reader_rejects_resealed_population_mode_source_and_passthrough_mutations(
    fixture: _Fixture, mutation: str, match: str
) -> None:
    payload = deepcopy(fixture.payload)
    if mutation == "population":
        payload["questions"].pop()
    elif mutation == "reorder":
        payload["questions"][0], payload["questions"][1] = (
            payload["questions"][1],
            payload["questions"][0],
        )
    elif mutation == "mode":
        row = payload["questions"][0]
        row["mode"] = (
            full100.PASSTHROUGH_MODE
            if row["mode"] == full100.TERMINAL_MODE
            else full100.TERMINAL_MODE
        )
        _reseal_question(row)
    elif mutation == "policy":
        policy = payload["policy_bindings"]
        policy["terminal_policy"] = {
            **policy["terminal_policy"],
            "receipt_sha256": identity_sha256({"shifted": "terminal-policy"}),
        }
        policy_body = {
            key: value for key, value in policy.items() if key != "receipt_sha256"
        }
        policy["receipt_sha256"] = identity_sha256(policy_body)
    elif mutation == "source":
        source = payload["source_artifact_bindings"]
        source["r7_construction_artifact_sha256"] = identity_sha256(
            {"shifted": "r7"}
        )
        source_body = {
            key: value for key, value in source.items() if key != "receipt_sha256"
        }
        source["receipt_sha256"] = identity_sha256(source_body)
    else:
        row = next(
            row
            for row in payload["questions"]
            if row["mode"] == full100.PASSTHROUGH_MODE
        )
        row["passthrough_prediction"] = "coherently resealed but wrong"
        _reseal_question(row)
    _reseal_top(payload)
    root = fixture.root / f"mutated-{mutation}"
    digest = _publish_pair(root, payload, fixture.bundle.sidecars)

    with pytest.raises(full100.LockedSemanticGlobalTerminalFull100Error, match=match):
        _reader(fixture, root, digest)


def test_reader_rejects_nonidentical_replay(fixture: _Fixture) -> None:
    root = fixture.root / "bad-replay"
    _publish_sidecars(root, fixture.bundle.sidecars)
    construction = _publish(root / full100.CONSTRUCTION_NAME, fixture.payload)
    changed = deepcopy(fixture.payload)
    changed["namespace_count"] += 1
    _reseal_top(changed)
    replay = _publish(root / full100.REPLAY_NAME, changed)

    with pytest.raises(
        full100.LockedSemanticGlobalTerminalFull100Error,
        match="not byte-identical",
    ):
        full100.load_verified_full100_construction(
            root,
            construction.sha256,
            replay.sha256,
            gate_path=fixture.sources.gate.path,
            expected_gate_sha256=fixture.sources.gate.sha256,
            r7_path=fixture.sources.r7.path,
            expected_r7_sha256=fixture.sources.r7.sha256,
            vectors_path=fixture.sources.vectors.path,
            vector_replay_path=fixture.sources.vector_replay.path,
            expected_vector_sha256=fixture.sources.vectors.sha256,
            parent_path=fixture.sources.parent.path,
            expected_parent_sha256=fixture.sources.parent.sha256,
        )


def _terminalized_subset(payload: dict, ordinals: tuple[int, ...]) -> dict:
    selected = set(ordinals)
    questions = [
        deepcopy(row) for row in payload["questions"] if row["ordinal"] in selected
    ]
    namespace_ids = {row["namespace_id"] for row in questions}
    namespaces = [
        deepcopy(row)
        for row in payload["namespace_receipts"]
        if row["namespace_id"] in namespace_ids
    ]
    body = {
        key: deepcopy(value)
        for key, value in payload.items()
        if key != "construction_identity_sha256"
    }
    body["namespace_receipts"] = namespaces
    body["question_count"] = len(questions)
    body["questions"] = questions
    body["v7_replay_count"] = len(questions)
    return {**body, "construction_identity_sha256": identity_sha256(body)}


def _resumable_args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(output_root=root)


def test_resumable_crash_skips_only_complete_checkpoints_and_matches_resident(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fixture.root / "resumable-crash"
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    first_calls: list[tuple[int, ...]] = []

    def crashing_build(args, *, terminal_compiler):
        assert callable(terminal_compiler)
        ordinals = tuple(args.ordinals)
        first_calls.append(ordinals)
        if len(first_calls) == 3:
            raise RuntimeError("synthetic namespace crash")
        return _terminalized_subset(fixture.terminalized, ordinals)

    monkeypatch.setattr(full100.v7_cli, "build_assay", crashing_build)
    with pytest.raises(RuntimeError, match="synthetic namespace crash"):
        resumable.run_construct(_resumable_args(root))

    checkpoint_root = root / resumable.CHECKPOINT_DIR_NAME
    assert len(tuple(checkpoint_root.glob("*.json"))) == 2
    assert not (root / full100.CONSTRUCTION_NAME).exists()

    resumed_calls: list[tuple[int, ...]] = []

    def resumed_build(args, *, terminal_compiler):
        assert callable(terminal_compiler)
        ordinals = tuple(args.ordinals)
        resumed_calls.append(ordinals)
        return _terminalized_subset(fixture.terminalized, ordinals)

    monkeypatch.setattr(full100.v7_cli, "build_assay", resumed_build)
    result = resumable.run_construct(_resumable_args(root))
    construction = read_sealed_json(root / full100.CONSTRUCTION_NAME)
    resident_root = fixture.root / "resident-equivalence"
    resident_sha = _publish_pair(
        resident_root, fixture.bundle.manifest, fixture.bundle.sidecars
    )

    assert result["checkpoint_reused_count"] == 2
    assert result["checkpoint_created_count"] == len(resumed_calls)
    assert len(first_calls[:2]) + len(resumed_calls) == result[
        "namespace_checkpoint_count"
    ]
    assert construction.payload == fixture.bundle.manifest
    assert construction.sha256 == resident_sha
    assert all(
        set(left).isdisjoint(right)
        for left in first_calls[:2]
        for right in resumed_calls
    )

    monkeypatch.setattr(
        full100.v7_cli,
        "build_assay",
        lambda *_args, **_kwargs: pytest.fail("replay rebuilt a namespace"),
    )
    replayed = resumable.run_replay(
        argparse.Namespace(
            output_root=root,
            expected_construction_output_sha256=construction.sha256,
        )
    )
    assert replayed["byte_identical"] is True
    assert replayed["replay_sha256"] == construction.sha256
    _reader(fixture, root, construction.sha256)


@pytest.mark.parametrize("mutation", ("tamper", "partial"))
def test_resumable_refuses_tampered_or_partial_checkpoint(
    fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = fixture.root / f"resumable-{mutation}"
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    calls = 0

    def one_then_crash(args, *, terminal_compiler):
        nonlocal calls
        assert callable(terminal_compiler)
        calls += 1
        if calls == 2:
            raise RuntimeError("stop after one checkpoint")
        return _terminalized_subset(fixture.terminalized, tuple(args.ordinals))

    monkeypatch.setattr(full100.v7_cli, "build_assay", one_then_crash)
    with pytest.raises(RuntimeError, match="stop after one checkpoint"):
        resumable.run_construct(_resumable_args(root))
    checkpoint = next((root / resumable.CHECKPOINT_DIR_NAME).glob("*.json"))
    if mutation == "tamper":
        checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
        match = "tampered or incomplete"
    else:
        checkpoint.with_name(checkpoint.name + ".sha256").unlink()
        match = "partial"
    monkeypatch.setattr(
        full100.v7_cli,
        "build_assay",
        lambda *_args, **_kwargs: pytest.fail("invalid checkpoint was skipped"),
    )
    with pytest.raises(
        resumable.LockedSemanticGlobalTerminalFull100ResumableError,
        match=match,
    ):
        resumable.run_construct(_resumable_args(root))


def test_resumable_cli_has_no_ordinal_and_requires_nonlegacy_output_root() -> None:
    parser = resumable.build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )
    for child in subparsers.choices.values():
        assert "ordinals" not in {action.dest for action in child._actions}
        assert "--ordinals" not in child._option_string_actions
    parsed = parser.parse_args(["construct"])
    assert parsed.output_root is None
    with pytest.raises(
        resumable.LockedSemanticGlobalTerminalFull100ResumableError,
        match="requires --output-root",
    ):
        resumable.run_construct(parsed)
    with pytest.raises(
        resumable.LockedSemanticGlobalTerminalFull100ResumableError,
        match="refuses the legacy default",
    ):
        resumable._safe_output_root(  # noqa: SLF001
            argparse.Namespace(output_root=full100.DEFAULT_OUTPUT_ROOT)
        )


def _legacy_import_args(
    successor_root: Path, legacy_root: Path, legacy_sha256: str
) -> argparse.Namespace:
    return argparse.Namespace(
        expected_legacy_construction_sha256=legacy_sha256,
        legacy_root=legacy_root,
        output_root=successor_root,
    )


def test_import_legacy_without_replay_seeds_exact_checkpoints_and_fast_replay(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "legacy-import-source"
    successor_root = fixture.root / "legacy-import-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    assert not (legacy_root / full100.REPLAY_NAME).exists()
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    monkeypatch.setattr(
        full100.v7_cli,
        "build_assay",
        lambda *_args, **_kwargs: pytest.fail("legacy import scanned V7/database"),
    )
    args = _legacy_import_args(successor_root, legacy_root, legacy.sha256)

    imported = resumable.run_import_legacy(args)
    successor = read_sealed_json(
        successor_root / full100.CONSTRUCTION_NAME
    )

    assert imported["legacy_construction_sha256"] == legacy.sha256
    assert imported["construction_sha256"] == successor.sha256 == legacy.sha256
    assert imported["checkpoint_created_count"] == imported[
        "namespace_checkpoint_count"
    ]
    assert imported["checkpoint_reused_count"] == 0
    assert successor.payload == fixture.bundle.manifest
    assert len(
        tuple(
            (successor_root / resumable.CHECKPOINT_DIR_NAME).glob("*.json")
        )
    ) == imported["namespace_checkpoint_count"]
    assert not (legacy_root / full100.REPLAY_NAME).exists()

    # Import is write-once/idempotent and skips only exact existing checkpoints.
    imported_again = resumable.run_import_legacy(args)
    assert imported_again["checkpoint_created_count"] == 0
    assert imported_again["checkpoint_reused_count"] == imported[
        "namespace_checkpoint_count"
    ]
    assert imported_again["construction_created"] is False

    replayed = resumable.run_replay(
        argparse.Namespace(
            output_root=successor_root,
            expected_construction_output_sha256=successor.sha256,
        )
    )
    assert replayed["byte_identical"] is True
    assert replayed["replay_sha256"] == legacy.sha256
    _reader(fixture, successor_root, legacy.sha256)


def test_import_legacy_authenticates_all_sidecars_before_successor_write(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "legacy-import-tampered"
    successor_root = fixture.root / "legacy-import-tampered-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    sidecar = next((legacy_root / full100.SIDECAR_DIR_NAME).glob("*.json"))
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    monkeypatch.setattr(
        full100.v7_cli,
        "build_assay",
        lambda *_args, **_kwargs: pytest.fail("tampered import scanned V7"),
    )

    with pytest.raises(MatchedEvalContractError):
        resumable.run_import_legacy(
            _legacy_import_args(successor_root, legacy_root, legacy.sha256)
        )
    assert not successor_root.exists()


def test_import_legacy_rejects_foreign_preexisting_successor_state(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "legacy-import-foreign-source"
    successor_root = fixture.root / "legacy-import-foreign-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    successor_root.mkdir(parents=True)
    (successor_root / "foreign.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )

    with pytest.raises(
        resumable.LockedSemanticGlobalTerminalFull100ResumableError,
        match="foreign state",
    ):
        resumable.run_import_legacy(
            _legacy_import_args(successor_root, legacy_root, legacy.sha256)
        )
    assert not (successor_root / resumable.PREFLIGHT_NAME).exists()


def _compact_import_args(
    successor_root: Path,
    legacy_root: Path,
    legacy_sha256: str,
    *,
    expected_attestation_sha256: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        expected_attestation_sha256=expected_attestation_sha256,
        expected_legacy_construction_sha256=legacy_sha256,
        legacy_root=legacy_root,
        output_root=successor_root,
    )


def test_compact_spooled_identity_matches_canonical_identity(tmp_path: Path) -> None:
    questions = [
        {"ordinal": 2, "text": "café"},
        {"ordinal": 7, "nested": {"z": 1, "a": [True, None]}},
    ]
    body = {"format": "synthetic", "questions": questions, "zero": 0}
    without = {key: value for key, value in body.items() if key != "questions"}
    with (tmp_path / "spool.bin").open("w+b") as spool:
        fragments = [
            compact_resumable._write_canonical_value(spool, row)  # noqa: SLF001
            for row in questions
        ]
        observed = compact_resumable._identity_with_spooled_questions(  # noqa: SLF001
            without, fragments, spool
        )
    assert observed == identity_sha256(body)


def test_compact_cli_is_opt_in_and_has_no_ordinal_selector() -> None:
    parser = compact_resumable.build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert set(subparsers.choices) == {"import-legacy", "replay"}
    for child in subparsers.choices.values():
        assert "ordinals" not in {action.dest for action in child._actions}
        assert "--ordinals" not in child._option_string_actions


def test_compact_import_and_bounded_replay_preserve_exact_bytes_without_v7(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-legacy"
    successor_root = fixture.root / "compact-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    monkeypatch.setattr(
        full100.v7_cli,
        "build_assay",
        lambda *_args, **_kwargs: pytest.fail("compact import reopened V7/store"),
    )

    imported = compact_resumable.run_import_legacy(
        _compact_import_args(successor_root, legacy_root, legacy.sha256)
    )
    assert imported["construction_sha256"] == legacy.sha256
    assert imported["checkpoint_created_count"] == len(fixture.bundle.sidecars)
    assert imported["sidecar_created_count"] == len(fixture.bundle.sidecars)
    assert (
        successor_root / full100.CONSTRUCTION_NAME
    ).read_bytes() == legacy.path.read_bytes()
    for sidecar in fixture.bundle.sidecars:
        digest = full100._sidecar_artifact_sha256(sidecar)  # noqa: SLF001
        source = legacy_root / full100.SIDECAR_DIR_NAME / f"{digest}.json"
        target = successor_root / full100.SIDECAR_DIR_NAME / f"{digest}.json"
        assert target.read_bytes() == source.read_bytes()
    checkpoints = tuple(
        (successor_root / compact_resumable.CHECKPOINT_DIR_NAME).glob("*.json")
    )
    assert max(path.stat().st_size for path in checkpoints) < min(
        (
            legacy_root / full100.SIDECAR_DIR_NAME / f"{digest}.json"
        ).stat().st_size
        for digest in {
            full100._sidecar_artifact_sha256(row)  # noqa: SLF001
            for row in fixture.bundle.sidecars
        }
    )

    replayed = compact_resumable.run_replay(
        argparse.Namespace(
            expected_attestation_sha256=imported["attestation_sha256"],
            output_root=successor_root,
            expected_construction_output_sha256=legacy.sha256,
        )
    )
    assert replayed["byte_identical"] is True
    assert replayed["replay_sha256"] == legacy.sha256
    assert (
        successor_root / full100.REPLAY_NAME
    ).read_bytes() == legacy.path.read_bytes()
    _reader(fixture, successor_root, legacy.sha256)


def test_compact_import_crash_after_sidecar_resumes_from_attestation(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-crash-legacy"
    successor_root = fixture.root / "compact-crash-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    original_ensure = compact_resumable._ensure_small_sealed_json  # noqa: SLF001
    crashed = False

    def crash_before_first_checkpoint(path: Path, payload: dict, **kwargs):
        nonlocal crashed
        if (
            compact_resumable.CHECKPOINT_DIR_NAME in path.parts
            and not crashed
        ):
            crashed = True
            raise RuntimeError("synthetic checkpoint crash")
        return original_ensure(path, payload, **kwargs)

    monkeypatch.setattr(
        compact_resumable,
        "_ensure_small_sealed_json",
        crash_before_first_checkpoint,
    )
    args = _compact_import_args(successor_root, legacy_root, legacy.sha256)
    with pytest.raises(RuntimeError, match="synthetic checkpoint crash"):
        compact_resumable.run_import_legacy(args)
    assert (successor_root / compact_resumable.ATTESTATION_NAME).exists()
    assert len(
        tuple((successor_root / full100.SIDECAR_DIR_NAME).glob("*.json"))
    ) == 1
    assert not (
        successor_root / compact_resumable.CHECKPOINT_DIR_NAME
    ).exists()
    assert not (successor_root / full100.CONSTRUCTION_NAME).exists()

    monkeypatch.setattr(
        compact_resumable, "_ensure_small_sealed_json", original_ensure
    )
    monkeypatch.setattr(
        compact_resumable,
        "_authenticate_new_import",
        lambda **_kwargs: pytest.fail("resume repeated deep authentication"),
    )
    with pytest.raises(MatchedEvalContractError, match="unpinned attestation"):
        compact_resumable.run_import_legacy(args)
    pinned_args = _compact_import_args(
        successor_root,
        legacy_root,
        legacy.sha256,
        expected_attestation_sha256=read_sealed_json(
            successor_root / compact_resumable.ATTESTATION_NAME
        ).sha256,
    )
    resumed = compact_resumable.run_import_legacy(pinned_args)
    assert resumed["attestation_created"] is False
    assert resumed["sidecar_created_count"] == len(fixture.bundle.sidecars) - 1
    assert resumed["construction_sha256"] == legacy.sha256


def test_compact_import_authenticates_late_tamper_before_successor_write(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-tampered-legacy"
    successor_root = fixture.root / "compact-tampered-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    sidecars = sorted(
        (legacy_root / full100.SIDECAR_DIR_NAME).glob("*.json")
    )
    sidecars[-1].write_bytes(sidecars[-1].read_bytes() + b" ")
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )

    with pytest.raises(MatchedEvalContractError):
        compact_resumable.run_import_legacy(
            _compact_import_args(successor_root, legacy_root, legacy.sha256)
        )
    assert not successor_root.exists()


def test_compact_import_rejects_foreign_root_before_source_loading(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-foreign-legacy"
    successor_root = fixture.root / "compact-foreign-successor"
    successor_root.mkdir(parents=True)
    (successor_root / "foreign.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        compact_resumable.v1_cli,
        "_build_context",
        lambda _args: pytest.fail("foreign root loaded sources"),
    )
    with pytest.raises(
        compact_resumable.LockedSemanticGlobalTerminalFull100CompactResumableError,
        match="foreign state",
    ):
        compact_resumable.run_import_legacy(
            _compact_import_args(successor_root, legacy_root, "0" * 64)
        )


def test_compact_import_rejects_wrong_attestation_type_before_source_loading(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-wrong-type-legacy"
    successor_root = fixture.root / "compact-wrong-type-successor"
    (successor_root / compact_resumable.ATTESTATION_NAME).mkdir(parents=True)
    monkeypatch.setattr(
        compact_resumable.v1_cli,
        "_build_context",
        lambda _args: pytest.fail("wrong output type loaded sources"),
    )

    with pytest.raises(MatchedEvalContractError, match="regular file"):
        compact_resumable.run_import_legacy(
            _compact_import_args(successor_root, legacy_root, "0" * 64)
        )


def test_compact_import_rejects_wrong_attestation_pin_before_source_loading(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-wrong-pin-legacy"
    successor_root = fixture.root / "compact-wrong-pin-successor"
    attestation, _created = publish_sealed_json(
        successor_root / compact_resumable.ATTESTATION_NAME, {}
    )
    assert attestation.sha256 != "0" * 64
    monkeypatch.setattr(
        compact_resumable.v1_cli,
        "_build_context",
        lambda _args: pytest.fail("wrong attestation pin loaded sources"),
    )

    with pytest.raises(MatchedEvalContractError, match="external pin"):
        compact_resumable.run_import_legacy(
            _compact_import_args(
                successor_root,
                legacy_root,
                "0" * 64,
                expected_attestation_sha256="0" * 64,
            )
        )


@pytest.mark.parametrize(
    "reserved_name",
    [full100.SIDECAR_DIR_NAME, compact_resumable.CHECKPOINT_DIR_NAME],
)
def test_compact_import_rejects_reserved_output_root_basename_before_sources(
    fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    reserved_name: str,
) -> None:
    successor_root = fixture.root / reserved_name
    monkeypatch.setattr(
        compact_resumable.v1_cli,
        "_build_context",
        lambda _args: pytest.fail("reserved output root loaded sources"),
    )

    with pytest.raises(MatchedEvalContractError, match="reserved basename"):
        compact_resumable.run_import_legacy(
            _compact_import_args(
                successor_root,
                fixture.root / "unused-legacy",
                "0" * 64,
            )
        )


def test_compact_replay_rejects_streamed_sidecar_tamper(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-replay-tamper-legacy"
    successor_root = fixture.root / "compact-replay-tamper-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    imported = compact_resumable.run_import_legacy(
        _compact_import_args(successor_root, legacy_root, legacy.sha256)
    )
    target = next(
        (successor_root / full100.SIDECAR_DIR_NAME).glob("*.json")
    )
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(MatchedEvalContractError, match="streamed sealed artifact"):
        compact_resumable.run_replay(
            argparse.Namespace(
                expected_attestation_sha256=imported["attestation_sha256"],
                output_root=successor_root,
                expected_construction_output_sha256=legacy.sha256,
            )
        )


def test_compact_replay_validates_all_checkpoints_before_sidecar_scans(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-checkpoint-first-legacy"
    successor_root = fixture.root / "compact-checkpoint-first-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    imported = compact_resumable.run_import_legacy(
        _compact_import_args(successor_root, legacy_root, legacy.sha256)
    )
    checkpoint = sorted(
        (successor_root / compact_resumable.CHECKPOINT_DIR_NAME).glob("*.json")
    )[-1]
    checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
    monkeypatch.setattr(
        compact_resumable,
        "_verify_sealed_bytes",
        lambda *_args, **_kwargs: pytest.fail(
            "replay scanned a sidecar before validating every checkpoint"
        ),
    )

    with pytest.raises(MatchedEvalContractError):
        compact_resumable.run_replay(
            argparse.Namespace(
                expected_attestation_sha256=imported["attestation_sha256"],
                output_root=successor_root,
                expected_construction_output_sha256=legacy.sha256,
            )
        )


def test_compact_replay_rejects_rehashed_attestation_sidecar_rebinding(
    fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = fixture.root / "compact-attestation-tamper-legacy"
    successor_root = fixture.root / "compact-attestation-tamper-successor"
    _publish_sidecars(legacy_root, fixture.bundle.sidecars)
    legacy = _publish(
        legacy_root / full100.CONSTRUCTION_NAME, fixture.bundle.manifest
    )
    monkeypatch.setattr(
        full100, "_load_build_sources", lambda _args: fixture.sources
    )
    imported = compact_resumable.run_import_legacy(
        _compact_import_args(successor_root, legacy_root, legacy.sha256)
    )
    path = successor_root / compact_resumable.ATTESTATION_NAME
    payload = deepcopy(read_sealed_json(path).payload)
    rows = payload["namespaces"]
    rows[0]["legacy_sidecar_sha256"] = rows[1]["legacy_sidecar_sha256"]
    row_body = {
        key: value
        for key, value in rows[0].items()
        if key != "namespace_attestation_receipt_sha256"
    }
    rows[0]["namespace_attestation_receipt_sha256"] = identity_sha256(row_body)
    body = {
        key: value
        for key, value in payload.items()
        if key != "attestation_identity_sha256"
    }
    payload["attestation_identity_sha256"] = identity_sha256(body)
    raw = compact_resumable.canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )

    with pytest.raises(MatchedEvalContractError, match="external pin"):
        compact_resumable.run_replay(
            argparse.Namespace(
                expected_attestation_sha256=imported["attestation_sha256"],
                output_root=successor_root,
                expected_construction_output_sha256=legacy.sha256,
            )
        )


def test_compact_streaming_copy_peak_is_chunk_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    payload = b"x" * (4 * 1024 * 1024)
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    source.with_name(source.name + ".sha256").write_bytes(
        f"{digest}  {source.name}\n".encode("ascii")
    )
    del payload
    monkeypatch.setattr(compact_resumable, "STREAM_CHUNK_BYTES", 64 * 1024)

    tracemalloc.start()
    created = compact_resumable._publish_verified_copy(  # noqa: SLF001
        source, target, digest, output_root=tmp_path
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert created is True
    assert target.stat().st_size == source.stat().st_size
    assert peak < 8 * 64 * 1024


def test_compact_streaming_copy_rejects_existing_target_hardlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"authenticated bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_name(source.name + ".sha256").write_bytes(
        f"{digest}  {source.name}\n".encode("ascii")
    )
    os.link(source, target)

    with pytest.raises(MatchedEvalContractError, match="hard-linked"):
        compact_resumable._publish_verified_copy(  # noqa: SLF001
            source, target, digest, output_root=tmp_path
        )


def test_compact_streaming_copy_safely_replaces_stranded_staging_hardlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "successor" / "target.bin"
    outside = tmp_path / "outside.bin"
    source.write_bytes(b"authenticated bytes")
    outside.write_bytes(b"must remain unchanged")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_name(source.name + ".sha256").write_bytes(
        f"{digest}  {source.name}\n".encode("ascii")
    )
    output_root = tmp_path / "successor"
    pending = compact_resumable._publication_staging_path(  # noqa: SLF001
        target, output_root=output_root
    )
    pending.parent.mkdir(parents=True)
    os.link(outside, pending)

    assert compact_resumable._publish_verified_copy(  # noqa: SLF001
        source, target, digest, output_root=output_root
    )
    assert outside.read_bytes() == b"must remain unchanged"
    assert target.read_bytes() == source.read_bytes()
    assert not pending.exists()


def test_compact_lifecycle_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    output_root = tmp_path / "successor"
    with compact_resumable._exclusive_output_lock(output_root):  # noqa: SLF001
        with pytest.raises(MatchedEvalContractError, match="already locked"):
            with compact_resumable._exclusive_output_lock(  # noqa: SLF001
                output_root
            ):
                pytest.fail("a concurrent compact writer acquired the lock")


def test_compact_staging_is_bound_to_explicit_lifecycle_root(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "successor"
    misleading_target = (
        output_root
        / "nested"
        / compact_resumable.CHECKPOINT_DIR_NAME
        / "artifact.json"
    )

    staging = compact_resumable._publication_staging_path(  # noqa: SLF001
        misleading_target, output_root=output_root
    )

    assert staging.parent == compact_resumable._control_root(  # noqa: SLF001
        output_root
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction reproduction")
def test_compact_rejects_windows_junctions_across_lifecycle_paths(
    tmp_path: Path,
) -> None:
    def junction(link: Path, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"junction creation unavailable: {completed.stderr}")
        assert link.is_junction()

    real_output = tmp_path / "real-output"
    output_junction = tmp_path / "output-junction"
    junction(output_junction, real_output)
    try:
        with pytest.raises(MatchedEvalContractError, match="junction"):
            compact_resumable._safe_output_root(  # noqa: SLF001
                argparse.Namespace(output_root=output_junction)
            )
    finally:
        output_junction.rmdir()

    ancestor_junction = tmp_path / "ancestor-junction"
    junction(ancestor_junction, tmp_path / "real-ancestor")
    try:
        with pytest.raises(MatchedEvalContractError, match="junction"):
            compact_resumable._safe_output_root(  # noqa: SLF001
                argparse.Namespace(
                    output_root=ancestor_junction / "nested" / "successor"
                )
            )
    finally:
        ancestor_junction.rmdir()

    output_root = tmp_path / "successor"
    output_root.mkdir()
    for directory_name in (
        full100.SIDECAR_DIR_NAME,
        compact_resumable.CHECKPOINT_DIR_NAME,
    ):
        redirected = output_root / directory_name
        junction(redirected, tmp_path / f"real-{directory_name}")
        try:
            with pytest.raises(MatchedEvalContractError, match="junction"):
                compact_resumable._require_owned_target(  # noqa: SLF001
                    output_root,
                    redirected / "artifact.json",
                    "test lifecycle target",
                )
        finally:
            redirected.rmdir()

    control = compact_resumable._control_root(output_root)  # noqa: SLF001
    junction(control, tmp_path / "real-control")
    try:
        with pytest.raises(MatchedEvalContractError, match="junction"):
            compact_resumable._ensure_control_root(output_root)  # noqa: SLF001
    finally:
        control.rmdir()

    target = output_root / "artifact.json"
    control.mkdir()
    staging = compact_resumable._publication_staging_path(  # noqa: SLF001
        target, output_root=output_root
    )
    junction(staging, tmp_path / "real-staging")
    try:
        with pytest.raises(MatchedEvalContractError, match="junction"):
            compact_resumable._open_fresh_staging(  # noqa: SLF001
                target, output_root=output_root
            )
    finally:
        staging.rmdir()
