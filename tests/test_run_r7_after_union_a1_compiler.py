from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_r7_after_union_a1 as a1_cli
from tools import run_r7_after_union_a1_compiler as compiler
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.r7_after_union_a1 import (
    COMPILER_OUTPUTS_FORMAT,
    DISPOSITIONS_FORMAT,
    build_r7_after_union_a1_payload,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


def _sha(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _question(
    question_id: str,
    dated_question: str,
    summaries: tuple[str, ...],
) -> dict[str, Any]:
    spec = compile_typed_operator_spec(dated_question)
    handles: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    groups: list[str] = []
    for index, summary in enumerate(summaries):
        handle = f"H{index + 1:03d}"
        group = f"G{index + 1:03d}"
        groups.append(group)
        handles.append(
            {
                "group_handle": group,
                "handle_id": handle,
                "origin": "map",
                "provenance_grade": "exact_citation",
            }
        )
        items.append(
            {
                "date": f"2025-01-{index + 1:02d}",
                "entity_key": f"Topic {index + 1}",
                "handle_ids": [handle],
                "included": True,
                "kind": "event",
                "relation": "completed event",
                "status": "completed",
                "summary": summary,
                "supported_slot_ids": [],
            }
        )
    typed = {
        "conflict_policy": "quarantine",
        "format": "synthetic-r7-typed-evidence-v1",
        "frontier": {
            "available_handle_ids": [row["handle_id"] for row in handles],
            "closed": False,
            "mode": "bounded",
            "omitted_handle_ids": [],
            "represented_handle_ids": [row["handle_id"] for row in handles],
            "truncated": False,
        },
        "handles": handles,
        "items": items,
        "operator_spec": spec.projection(),
    }
    question_sha = quote_sha256(dated_question)
    return {
        "ordinal": 7001,
        "question_id": question_id,
        "dated_question_sha256": question_sha,
        "terminal_answer_plan": {
            "dated_question_sha256": question_sha,
            "parent_prediction": "SOURCE-ONLY-PARENT-SENTINEL",
            "provider_input": {
                "dated_question": dated_question,
                "protected_parent_fallback": "SOURCE-ONLY-PARENT-SENTINEL",
                "story_coherence": {
                    "group_links": (
                        [
                            {
                                "group_handles": groups[:2],
                                "relation": "same event boundary",
                            }
                        ]
                        if len(groups) >= 2
                        else []
                    ),
                    "incompatible_group_pairs": [],
                    "link_overlays": [],
                },
                "typed_evidence": typed,
            },
            "reference_answer": "SOURCE-ONLY-GOLD-SENTINEL",
        },
    }


def _source(*questions: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "memory-condense-reduced-semantic-global-terminal-assay-v2",
        "gold_loaded": False,
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": len(questions),
        "questions": list(questions),
        "retained_transformer_token_state_bytes": 0,
        "terminal_answer_plan_count": len(questions),
    }


def _dispositions(
    source: dict[str, Any], classifier_preflight: dict[str, Any]
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    for row in classifier_preflight["questions"]:
        leaves = row["semantic_selection"]["leaves"]
        questions.append(
            {
                "classifier_request_sha256s": [
                    request["request_sha256"]
                    for request in row["classifier_requests"]
                ],
                "dispositions": [
                    {
                        "disposition": "relevant",
                        "handle_id": leaf["handle_id"],
                        "leaf_receipt_sha256": leaf["receipt_sha256"],
                    }
                    for leaf in leaves
                ],
                "question_sha256": row["question_sha256"],
                "selected_union_population_sha256": row[
                    "selected_population_sha256"
                ],
            }
        )
    return {
        "classifier_id": "synthetic-all-relevant-v1",
        "format": DISPOSITIONS_FORMAT,
        "provider_calls_performed_by_core": 0,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": {
            "gold_loaded": False,
            "ordinal_routing_enabled": False,
            "protected_parent_loaded": False,
            "reference_loaded": False,
            "semantic_atom_manifest_loaded": False,
            "source_allowlist_loaded": False,
        },
        "source_artifact_sha256": _sha(source),
    }


def _classified(
    source: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_sha = _sha(source)
    classifier_preflight = build_r7_after_union_a1_payload(
        source,
        source_sha,
        source_sha,
        expected_question_count=len(source["questions"]),
        max_leaves_per_classifier_shard=2,
        max_leaves_per_shard=2,
    )
    dispositions = _dispositions(source, classifier_preflight)
    classified = build_r7_after_union_a1_payload(
        source,
        source_sha,
        source_sha,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
        expected_question_count=len(source["questions"]),
        max_leaves_per_classifier_shard=2,
        max_leaves_per_shard=2,
    )
    return classified, dispositions


def _seal_classified(root: Path, payload: dict[str, Any]) -> str:
    construction, _ = publish_sealed_json(
        root / a1_cli.CONSTRUCTION_NAME, payload
    )
    replay, _ = publish_sealed_json(root / a1_cli.REPLAY_NAME, payload)
    assert construction.sha256 == replay.sha256
    return construction.sha256


def _args(
    output_root: Path,
    classified_root: Path,
    classified_sha: str,
    disposition_sha: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        api_key_env="TEST_TERRA_KEY",
        approve_provider_release=False,
        authorized_provider_calls=0,
        classified_root=classified_root,
        enable_provider=False,
        expected_classified_construction_sha256=classified_sha,
        expected_classified_replay_sha256=classified_sha,
        expected_disposition_artifact_sha256=disposition_sha,
        expected_compiler_outputs_sha256=None,
        expected_preflight_sha256=None,
        expected_release_sha256=None,
        gateway_url=compiler.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=compiler.DEFAULT_MODEL,
        output_root=output_root,
    )


def _fixture(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    argparse.Namespace,
]:
    source = _source(
        _question(
            "q-one",
            "[Question asked at 2025-01-09] What did I buy and where?",
            (
                "I bought the cobalt kettle.",
                "I traveled to Kyoto.",
                "The spare receipt concerned a notebook.",
            ),
        ),
        _question(
            "q-two",
            "[Question asked at 2025-02-10] What class interests me?",
            ("I am interested in a Korean cooking class.",),
        ),
    )
    classified, dispositions = _classified(source)
    classified_root = tmp_path / "classified"
    classified_sha = _seal_classified(classified_root, classified)
    args = _args(
        tmp_path / "compiler",
        classified_root,
        classified_sha,
        _sha(dispositions),
    )
    return source, classified, dispositions, args


def _valid_completion(request: Mapping[str, Any]) -> str:
    provider_input = json.loads(request["messages"][1]["content"])
    return json.dumps(
        {
            "facts": [
                {
                    "citations": [
                        {
                            "handle_id": evidence["handle_ids"][0],
                            "quote": evidence["summary"],
                        }
                    ],
                    "date": None,
                    "entity": None,
                    "kind": "event",
                    "numeric_value": None,
                    "slot_ids": [],
                    "status": None,
                    "text": evidence["summary"],
                    "unit": None,
                }
                for evidence in provider_input["evidence"]
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class _FakeCompletions:
    def __init__(self, completion_builder: Any = _valid_completion) -> None:
        self.completion_builder = completion_builder
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls.append(dict(request))
        completion = self.completion_builder(request)
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=completion),
                ),
            ),
            id=f"fake-a1b-compiler-{len(self.calls):03d}",
            model=compiler.DEFAULT_MODEL,
            usage=None,
        )


class _FakeClient:
    max_retries = 0

    def __init__(self, completion_builder: Any = _valid_completion) -> None:
        self.completions = _FakeCompletions(completion_builder)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _preflight_release(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight = compiler.run_preflight(args)
    args.expected_preflight_sha256 = preflight["preflight_sha256"]
    args.approve_provider_release = True
    release = compiler.run_approve_release(args)
    args.expected_release_sha256 = release["release_sha256"]
    return preflight, release


def _run_fake_provider(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    fake: _FakeClient,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    args.enable_provider = True
    args.authorized_provider_calls = preflight["derived_provider_call_count"]
    monkeypatch.setattr(compiler, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_TERRA_KEY", "fake-key")
    monkeypatch.setattr(
        compiler, "_make_provider_client", lambda *_args: fake
    )
    return compiler.run_provider(args)


def test_fake_concurrent_lifecycle_is_dynamic_and_adapter_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, classified, dispositions, args = _fixture(tmp_path)
    expected_disposition_sha = args.expected_disposition_artifact_sha256
    args.expected_disposition_artifact_sha256 = "0" * 64
    with pytest.raises(
        compiler.R7AfterUnionA1CompilerError,
        match="disposition artifact changed",
    ):
        compiler.run_preflight(args)
    assert not (Path(args.output_root) / compiler.PREFLIGHT_NAME).exists()
    args.expected_disposition_artifact_sha256 = expected_disposition_sha
    preflight = compiler.run_preflight(args)
    args.expected_preflight_sha256 = preflight["preflight_sha256"]
    assert preflight["derived_provider_call_count"] == 3
    with pytest.raises(
        compiler.R7AfterUnionA1CompilerError,
        match="explicit provider approval",
    ):
        compiler.run_approve_release(args)
    args.approve_provider_release = True
    release = compiler.run_approve_release(args)
    args.expected_release_sha256 = release["release_sha256"]

    args.enable_provider = True
    args.authorized_provider_calls = 2
    monkeypatch.setattr(
        compiler,
        "load_dotenv",
        lambda: pytest.fail("provider environment opened before authorization"),
    )
    with pytest.raises(
        compiler.R7AfterUnionA1CompilerError,
        match="exactly equal remaining",
    ):
        compiler.run_provider(args)

    fake = _FakeClient()
    args.authorized_provider_calls = 3
    monkeypatch.setattr(compiler, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_TERRA_KEY", "fake-key")
    monkeypatch.setattr(
        compiler, "_make_provider_client", lambda *_args: fake
    )
    provider = compiler.run_provider(args)
    assert provider["physical_provider_calls"] == 3
    assert len(fake.completions.calls) == 3

    sealed_preflight = read_sealed_json(
        Path(args.output_root) / compiler.PREFLIGHT_NAME
    )
    expected_messages = [
        row["compiler_request"]["messages"]
        for row in sealed_preflight.payload["request_rows"]
    ]
    assert sorted(
        identity_sha256(call["messages"]) for call in fake.completions.calls
    ) == sorted(identity_sha256(messages) for messages in expected_messages)
    assert all(
        set(call) == {"max_tokens", "messages", "model"}
        and call["max_tokens"] == compiler.MAX_NEW_TOKENS
        and call["model"] == compiler.DEFAULT_MODEL
        for call in fake.completions.calls
    )
    serialized = json.dumps(expected_messages, sort_keys=True)
    for forbidden in (
        "SOURCE-ONLY-GOLD-SENTINEL",
        "SOURCE-ONLY-PARENT-SENTINEL",
        "semantic_atom_manifest",
        "source_allowlist",
        '"ordinal"',
        '"targets"',
    ):
        assert forbidden not in serialized

    monkeypatch.setattr(
        compiler,
        "_make_provider_client",
        lambda *_args: pytest.fail("materialization opened provider"),
    )
    materialized = compiler.run_materialize(args)
    outputs = read_sealed_json(Path(args.output_root) / compiler.OUTPUTS_NAME)
    assert outputs.sha256 == materialized["compiler_outputs_sha256"]
    assert outputs.payload["format"] == COMPILER_OUTPUTS_FORMAT
    assert outputs.payload["response_count"] == 3
    assert all(
        binding["rejected_fact_count"] == 0
        and binding["resolved_leaf_handle_ids"]
        and not binding["unresolved_leaf_handle_ids"]
        for binding in outputs.payload["response_bindings"]
    )

    source_sha = _sha(source)
    accepted = build_r7_after_union_a1_payload(
        source,
        source_sha,
        source_sha,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
        compiler_output_payload=outputs.payload,
        compiler_output_artifact_sha256=outputs.sha256,
        expected_question_count=len(source["questions"]),
        max_leaves_per_classifier_shard=2,
        max_leaves_per_shard=2,
    )
    assert accepted["missing_compiler_call_count"] == 0
    assert accepted["compiler_request_count"] == 3

    args.expected_compiler_outputs_sha256 = outputs.sha256
    replayed = compiler.run_replay(args)
    assert replayed["byte_identical"] is True
    assert replayed["replay_sha256"] == outputs.sha256

    args.authorized_provider_calls = 0
    checkpoint_replay = compiler.run_provider(args)
    assert checkpoint_replay["physical_provider_calls"] == 0
    assert checkpoint_replay["checkpoint_hits"] == 3
    assert classified["compiler_request_count"] == 3


def _malformed_citation(kind: str):
    def build(request: Mapping[str, Any]) -> str:
        response = json.loads(_valid_completion(request))
        citation = response["facts"][0]["citations"][0]
        if kind == "missing-field":
            response["facts"][0].pop("citations")
        elif kind == "empty":
            response["facts"][0]["citations"] = []
        elif kind == "unknown-handle":
            citation["handle_id"] = "H999"
        elif kind == "wrong-quote":
            citation["quote"] = "not an exact source substring"
        return json.dumps(response)

    return build


@pytest.mark.parametrize(
    "kind", ("missing-field", "empty", "unknown-handle", "wrong-quote")
)
def test_malformed_or_omitted_citation_fails_without_resolving_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _source_payload, _classified_payload, _dispositions_payload, args = _fixture(
        tmp_path
    )
    preflight, _release = _preflight_release(args)
    _run_fake_provider(
        args, preflight, _FakeClient(_malformed_citation(kind)), monkeypatch
    )
    with pytest.raises(
        compiler.R7AfterUnionA1CompilerError,
        match="fact|citation|rejected|malformed",
    ):
        compiler.run_materialize(args)
    assert not (Path(args.output_root) / compiler.OUTPUTS_NAME).exists()


def test_tampered_response_journal_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_payload, _classified_payload, _dispositions_payload, args = _fixture(
        tmp_path
    )
    preflight, _release = _preflight_release(args)
    _run_fake_provider(args, preflight, _FakeClient(), monkeypatch)
    response = next(
        (Path(args.output_root) / compiler.CHECKPOINT_DIR_NAME).glob(
            "*.response.json"
        )
    )
    response.write_bytes(response.read_bytes() + b" ")
    with pytest.raises((ValueError, compiler.R7AfterUnionA1CompilerError)):
        compiler.run_materialize(args)


def test_incomplete_pair_and_unknown_request_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_payload, _classified_payload, _dispositions_payload, args = _fixture(
        tmp_path
    )
    preflight, _release = _preflight_release(args)
    checkpoint_root = Path(args.output_root) / compiler.CHECKPOINT_DIR_NAME
    checkpoint_root.mkdir(parents=True)
    unknown = "a" * 64
    (checkpoint_root / f"{unknown}.request.json").write_text("{}", encoding="utf-8")
    args.enable_provider = True
    args.authorized_provider_calls = preflight["derived_provider_call_count"]
    monkeypatch.setattr(
        compiler,
        "load_dotenv",
        lambda: pytest.fail("unsafe journal opened provider environment"),
    )
    with pytest.raises(
        compiler.R7AfterUnionA1CompilerError,
        match="incomplete; unsafe retry forbidden",
    ):
        compiler.run_provider(args)

    (checkpoint_root / f"{unknown}.response.json").write_text(
        "{}", encoding="utf-8"
    )
    args.authorized_provider_calls = preflight["derived_provider_call_count"] - 1
    with pytest.raises(ValueError, match="outside the preflighted population"):
        compiler.run_provider(args)


def test_cli_has_no_ordinal_routing() -> None:
    parser = compiler.build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )
    for child in subparsers.choices.values():
        assert "ordinal" not in {action.dest for action in child._actions}
        assert "ordinals" not in {action.dest for action in child._actions}
        assert "--ordinal" not in child._option_string_actions
        assert "--ordinals" not in child._option_string_actions


def test_temporal_effective_successor_is_the_default_authenticated_source() -> None:
    assert compiler.DEFAULT_CLASSIFIED_ROOT == Path(
        "eval_results/matched_eval_100/"
        "locked-r7-after-union-a1-classified-temporal-effective-v1"
    )
    assert compiler.EXPECTED_CLASSIFIED_SHA256 == (
        "d9071196d57fedf96516aae38dfe5ed0adb5218858bee32d7f7904353c9c4da1"
    )
    assert compiler.EXPECTED_DISPOSITION_ARTIFACT_SHA256 == (
        "40a584d6499f3682a89cab1aa272c34a8ccf7ead825d2451192bc2b49114a278"
    )
