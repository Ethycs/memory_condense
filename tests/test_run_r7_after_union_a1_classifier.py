from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tools import run_r7_after_union_a1 as a1_cli
from tools import run_r7_after_union_a1_classifier as classifier
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.r7_after_union_a1 import (
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
        "ordinal": 9001,
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
                                "relation": "same entity across boundary",
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


def _a1_payload(source: dict[str, Any]) -> dict[str, Any]:
    source_sha = _sha(source)
    return build_r7_after_union_a1_payload(
        source,
        source_sha,
        source_sha,
        expected_question_count=len(source["questions"]),
        max_leaves_per_classifier_shard=2,
    )


def _seal_a1(root: Path, payload: dict[str, Any]) -> str:
    construction, _ = publish_sealed_json(
        root / a1_cli.CONSTRUCTION_NAME, payload
    )
    replay, _ = publish_sealed_json(root / a1_cli.REPLAY_NAME, payload)
    assert construction.sha256 == replay.sha256
    return construction.sha256


def _args(output_root: Path, a1_root: Path, a1_sha: str) -> argparse.Namespace:
    return argparse.Namespace(
        a1_root=a1_root,
        api_key_env="TEST_TERRA_KEY",
        approve_provider_release=False,
        authorized_provider_calls=0,
        enable_provider=False,
        expected_a1_construction_sha256=a1_sha,
        expected_a1_replay_sha256=a1_sha,
        expected_dispositions_sha256=None,
        expected_preflight_sha256=None,
        expected_release_sha256=None,
        gateway_url=classifier.DEFAULT_GATEWAY_URL,
        max_concurrency=2,
        model=classifier.DEFAULT_MODEL,
        output_root=output_root,
    )


class _FakeCompletions:
    def __init__(
        self,
        *,
        completion_builder: Any | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.completion_builder = completion_builder or self._valid_completion

    @staticmethod
    def _valid_completion(request: Mapping[str, Any]) -> str:
        provider_input = json.loads(request["messages"][1]["content"])
        labels = ("relevant", "unresolved", "definitely_irrelevant")
        return json.dumps(
            {
                "leaf_dispositions": [
                    {
                        "disposition": labels[index % len(labels)],
                        "handle_id": leaf["handle_id"],
                    }
                    for index, leaf in enumerate(
                        provider_input["leaf_population"]
                    )
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

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
            id=f"fake-a1-classifier-{len(self.calls):03d}",
            model=classifier.DEFAULT_MODEL,
            usage=None,
        )


class _FakeClient:
    max_retries = 0

    def __init__(self, completion_builder: Any | None = None) -> None:
        self.completions = _FakeCompletions(
            completion_builder=completion_builder
        )
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], argparse.Namespace]:
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
    a1_payload = _a1_payload(source)
    a1_root = tmp_path / "a1"
    a1_sha = _seal_a1(a1_root, a1_payload)
    return source, a1_payload, _args(tmp_path / "classifier", a1_root, a1_sha)


def test_fake_client_full_lifecycle_is_dynamic_exact_and_adapter_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, a1_payload, args = _fixture(tmp_path)
    preflight_result = classifier.run_preflight(args)
    args.expected_preflight_sha256 = preflight_result["preflight_sha256"]

    assert preflight_result["derived_provider_call_count"] == 3
    with pytest.raises(
        classifier.R7AfterUnionA1ClassifierError,
        match="explicit provider approval",
    ):
        classifier.run_approve_release(args)
    args.approve_provider_release = True
    release_result = classifier.run_approve_release(args)
    args.expected_release_sha256 = release_result["release_sha256"]
    args.enable_provider = True
    args.authorized_provider_calls = 2
    monkeypatch.setattr(
        classifier,
        "load_dotenv",
        lambda: pytest.fail("provider environment opened before authorization"),
    )
    with pytest.raises(
        classifier.R7AfterUnionA1ClassifierError,
        match="exactly equal remaining",
    ):
        classifier.run_provider(args)

    fake = _FakeClient()
    monkeypatch.setattr(classifier, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_TERRA_KEY", "fake-key")
    monkeypatch.setattr(
        classifier, "_make_provider_client", lambda *_args: fake
    )
    args.authorized_provider_calls = 3
    provider_result = classifier.run_provider(args)
    assert provider_result["physical_provider_calls"] == 3
    assert len(fake.completions.calls) == 3

    preflight = read_sealed_json(
        Path(args.output_root) / classifier.PREFLIGHT_NAME
    )
    expected_messages = [
        row["classifier_request"]["messages"]
        for row in preflight.payload["request_rows"]
    ]
    assert sorted(
        identity_sha256(call["messages"]) for call in fake.completions.calls
    ) == sorted(identity_sha256(messages) for messages in expected_messages)
    assert all(
        set(call) == {"max_tokens", "messages", "model"}
        and call["model"] == classifier.DEFAULT_MODEL
        and call["max_tokens"] == classifier.MAX_NEW_TOKENS
        for call in fake.completions.calls
    )
    serialized_messages = json.dumps(expected_messages, sort_keys=True)
    for forbidden in (
        "SOURCE-ONLY-GOLD-SENTINEL",
        "SOURCE-ONLY-PARENT-SENTINEL",
        "semantic_atom_manifest",
        "source_allowlist",
        '"ordinal"',
    ):
        assert forbidden not in serialized_messages

    materialized = classifier.run_materialize(args)
    dispositions = read_sealed_json(
        Path(args.output_root) / classifier.DISPOSITIONS_NAME
    )
    assert dispositions.sha256 == materialized["dispositions_sha256"]
    assert dispositions.payload["format"] == DISPOSITIONS_FORMAT
    assert dispositions.payload["derived_provider_call_count"] == 3
    assert any(
        row["disposition"] == "unresolved"
        for question in dispositions.payload["questions"]
        for row in question["dispositions"]
    )

    source_sha = _sha(source)
    accepted = build_r7_after_union_a1_payload(
        source,
        source_sha,
        source_sha,
        disposition_payload=dispositions.payload,
        disposition_artifact_sha256=dispositions.sha256,
        expected_question_count=len(source["questions"]),
        max_leaves_per_classifier_shard=2,
    )
    assert accepted["missing_classifier_call_count"] == 0
    assert sum(
        question["disposition_counts"]["uncertain"]
        for question in accepted["questions"]
    ) >= 1

    args.expected_dispositions_sha256 = dispositions.sha256
    replayed = classifier.run_replay(args)
    assert replayed["byte_identical"] is True
    assert replayed["replay_sha256"] == dispositions.sha256

    args.authorized_provider_calls = 0
    monkeypatch.setattr(
        classifier,
        "_make_provider_client",
        lambda *_args: pytest.fail("complete checkpoint replay opened provider"),
    )
    replay_provider = classifier.run_provider(args)
    assert replay_provider["physical_provider_calls"] == 0
    assert replay_provider["checkpoint_hits"] == 3
    assert a1_payload["classifier_request_count"] == 3


def _malformed_builder(kind: str):
    def build(request: Mapping[str, Any]) -> str:
        provider_input = json.loads(request["messages"][1]["content"])
        rows = [
            {"disposition": "relevant", "handle_id": leaf["handle_id"]}
            for leaf in provider_input["leaf_population"]
        ]
        if kind == "omitted":
            rows = rows[:-1]
        elif kind == "reordered":
            rows = list(reversed(rows))
        elif kind == "invalid":
            rows[0]["disposition"] = "irrelevant"
        elif kind == "extra-row":
            rows.append(
                {"disposition": "definitely_irrelevant", "handle_id": "HX"}
            )
        elif kind == "extra-key":
            return json.dumps(
                {"leaf_dispositions": rows, "default": "definitely_irrelevant"}
            )
        return json.dumps({"leaf_dispositions": rows})

    return build


@pytest.mark.parametrize(
    "kind", ("omitted", "reordered", "invalid", "extra-row", "extra-key")
)
def test_malformed_or_incomplete_response_never_infers_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _source_payload, _a1, args = _fixture(tmp_path)
    preflight = classifier.run_preflight(args)
    args.expected_preflight_sha256 = preflight["preflight_sha256"]
    args.approve_provider_release = True
    release = classifier.run_approve_release(args)
    args.expected_release_sha256 = release["release_sha256"]
    args.enable_provider = True
    args.authorized_provider_calls = preflight["derived_provider_call_count"]
    fake = _FakeClient(_malformed_builder(kind))
    monkeypatch.setattr(classifier, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_TERRA_KEY", "fake-key")
    monkeypatch.setattr(
        classifier, "_make_provider_client", lambda *_args: fake
    )
    classifier.run_provider(args)

    with pytest.raises(
        classifier.R7AfterUnionA1ClassifierError,
        match="response|disposition|cover every handle",
    ):
        classifier.run_materialize(args)
    assert not (Path(args.output_root) / classifier.DISPOSITIONS_NAME).exists()


def test_incomplete_checkpoint_pair_is_terminal_and_opens_no_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_payload, _a1, args = _fixture(tmp_path)
    preflight = classifier.run_preflight(args)
    args.expected_preflight_sha256 = preflight["preflight_sha256"]
    args.approve_provider_release = True
    release = classifier.run_approve_release(args)
    args.expected_release_sha256 = release["release_sha256"]
    checkpoint_root = Path(args.output_root) / classifier.CHECKPOINT_DIR_NAME
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / f"{'a' * 64}.request.json").write_text(
        "{}", encoding="utf-8"
    )
    args.enable_provider = True
    args.authorized_provider_calls = preflight["derived_provider_call_count"]
    monkeypatch.setattr(
        classifier,
        "load_dotenv",
        lambda: pytest.fail("incomplete journal opened provider environment"),
    )

    with pytest.raises(
        classifier.R7AfterUnionA1ClassifierError,
        match="incomplete; unsafe retry forbidden",
    ):
        classifier.run_provider(args)


def test_provider_message_contamination_and_ordinal_cli_are_rejected(
    tmp_path: Path,
) -> None:
    _source_payload, a1_payload, _args_value = _fixture(tmp_path)
    request = deepcopy(a1_payload["questions"][0]["classifier_requests"][0])
    provider_input = json.loads(request["messages"][1]["content"])
    provider_input["ordinal"] = 0
    request["messages"][1]["content"] = json.dumps(
        provider_input,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    request["prompt_token_proxy"] = count_chat_prompt_token_proxy(
        request["messages"]
    )
    request["request_sha256"] = identity_sha256(
        {
            key: value
            for key, value in request.items()
            if key != "request_sha256"
        }
    )
    with pytest.raises(
        classifier.R7AfterUnionA1ClassifierError,
        match="provider-message contract",
    ):
        classifier._request_projection(request)  # noqa: SLF001

    parser = classifier.build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )
    for child in subparsers.choices.values():
        assert "ordinals" not in {action.dest for action in child._actions}
        assert "ordinal" not in {action.dest for action in child._actions}
        assert "--ordinals" not in child._option_string_actions
        assert "--ordinal" not in child._option_string_actions
