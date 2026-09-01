from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_r7_a1_terminal_answer as answer
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.r7_after_union_a1 import COMPILER_OUTPUTS_FORMAT


def _sha(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _signed(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: identity_sha256(body)}


def _artifact(path: Path, payload: dict[str, Any]) -> SealedArtifact:
    return SealedArtifact(path, _sha(payload), payload)


def _leaf(handle: str, group: str, text: str) -> dict[str, Any]:
    body = {
        "boundary_labels": ["date:2025-01-01"],
        "cross_boundary_edge_ids": [],
        "format": "memory-condense-after-union-selected-h-leaf-v1",
        "group_handle": group,
        "handle_id": handle,
        "source_receipt_sha256": _sha({"source": handle}),
        "text": text,
        "token_count": 8,
        "topic_labels": ["kind:event"],
    }
    return _signed(body, "receipt_sha256")


def _outcome(leaf: dict[str, Any], disposition: str) -> dict[str, Any]:
    body = {
        "disposition": disposition,
        "facts": [],
        "format": "memory-condense-after-union-leaf-fact-outcome-v1",
        "handle_id": leaf["handle_id"],
        "leaf_disposition_receipt_sha256": _sha(
            {"disposition": disposition, "handle": leaf["handle_id"]}
        ),
        "leaf_receipt_sha256": leaf["receipt_sha256"],
        "unresolved_obligation_ids": [],
    }
    return _signed(body, "receipt_sha256")


def _merged_fact(
    leaf: dict[str, Any], fact_index: int, *, alternate: bool = False
) -> dict[str, Any]:
    quote = leaf["text"]
    citation = {
        "group_handle": leaf["group_handle"],
        "handle_id": leaf["handle_id"],
        "quote": quote,
        "quote_sha256": quote_sha256(quote),
        "source_summary_sha256": quote_sha256(quote),
    }
    compiled = {
        "citations": [citation],
        "date": "2025-01-01",
        "entity": f"entity-{fact_index}",
        "kind": "event",
        "numeric_value": None,
        "slot_ids": [],
        "status": "completed",
        "text": quote,
        "unit": None,
    }
    structured = {
        "compiled_fact": compiled,
        "event_time": "2025-01-01",
        "format": "memory-condense-after-union-structured-atomic-fact-v1",
        "leaf_handle_id": leaf["handle_id"],
        "member_key": f"member-{fact_index}",
        "obligation_ids": [],
        "predicate": "supports" if not alternate else "supports-detail",
        "qualifiers": [],
        "receipt_sha256": _sha(
            {"structured": leaf["handle_id"], "alternate": alternate}
        ),
        "source_time": "2025-01-01",
    }
    body = {
        "citations": [citation],
        "facts": [structured],
        "fingerprint": {"alternate": alternate, "fact_index": fact_index},
        "format": "memory-condense-after-union-merged-fact-v1",
        "leaf_handle_ids": [leaf["handle_id"]],
    }
    return _signed(body, "receipt_sha256")


def _question(
    question_index: int,
    *,
    selected_count: int,
    retained_count: int,
    fact_count: int,
    add_extra_fact: bool,
) -> dict[str, Any]:
    leaves = [
        _leaf(
            f"H{question_index + 1}{index + 1:05d}",
            f"G{question_index + 1}{index + 1:05d}",
            f"Memory {question_index + 1}-{index + 1} supplies exact detail.",
        )
        for index in range(selected_count)
    ]
    retained = [row["handle_id"] for row in leaves[:retained_count]]
    outcomes = [
        _outcome(
            leaf,
            (
                "facts"
                if index < fact_count
                else "unresolved"
                if index < retained_count
                else "definitely_irrelevant"
            ),
        )
        for index, leaf in enumerate(leaves)
    ]
    merged = [
        _merged_fact(leaves[index], index) for index in range(fact_count)
    ]
    if add_extra_fact:
        merged.append(_merged_fact(leaves[0], fact_count, alternate=True))
    selection_body = {
        "cross_boundary_edges": [],
        "format": "memory-condense-after-union-selection-v1",
        "leaves": leaves,
        "semantic_result": {"retained_leaf_cell_ids": retained},
    }
    selection = _signed(selection_body, "receipt_sha256")
    closure_body = {
        "format": "memory-condense-after-union-fact-closure-v1",
        "full_store_support_closure_available": False,
        "gold_loaded": False,
        "leaf_outcomes": outcomes,
        "merged_facts": merged,
        "operator_obligation_coverage": {},
        "provider_calls_performed_by_core": 0,
        "question_sha256": _sha({"question": question_index}),
        "retained_transformer_token_state_bytes": 0,
        "selected_population_coverage": {},
        "selection_receipt_sha256": selection["receipt_sha256"],
        "shard_receipt_sha256s": [],
    }
    closure = _signed(closure_body, "receipt_sha256")
    dated = f"[Question asked at 2025-01-{question_index + 1:02d}] What happened?"
    question_sha = quote_sha256(dated)
    closure["question_sha256"] = question_sha
    closure["receipt_sha256"] = identity_sha256(
        {key: value for key, value in closure.items() if key != "receipt_sha256"}
    )
    spec = _signed(
        {
            "absence_decision_requires_closed_frontier": True,
            "answer_shape": "concise",
            "cardinality": "single",
            "comparison_mode": "none",
            "format": "synthetic-operator-spec-v1",
            "include_proposed": False,
            "operation": "retrieve",
            "ordering": "chronological",
            "personalization_required": False,
            "query_timestamp": f"2025-01-{question_index + 1:02d}",
            "question_sha256": question_sha,
            "required_evidence_role": "support",
            "required_slots": [],
            "requires_all_slots": False,
            "requires_complete_frontier": False,
            "retained_transformer_token_state_bytes": 0,
            "route_receipt_sha256": _sha({"route": question_index}),
            "specificity_required": True,
            "style": "direct",
            "temporal_mode": "as_of",
            "temporal_window_days": None,
        },
        "receipt_sha256",
    )
    frontier = _signed(
        {
            "closed": False,
            "format": "synthetic-operator-frontier-v1",
            "question_sha256": question_sha,
        },
        "receipt_sha256",
    )
    packet = _signed(
        {
            "conflict_policy": "prefer_dated_support",
            "format": "synthetic-operator-packet-v1",
            "frontier": frontier,
            "hard_prompt_token_cap": 8_000,
            "output_token_reserve": 768,
            "provider_payload_mode": "evidence_only",
        },
        "receipt_sha256",
    )
    execution = _signed(
        {
            "format": "synthetic-operator-execution-v1",
            "prediction": "must not leak",
            "status": "partial",
        },
        "receipt_sha256",
    )
    body = {
        "dated_question": dated,
        "dated_question_sha256": question_sha,
        "fact_closure": closure,
        "format": "memory-condense-r7-after-union-a1-preflight-v2-question-v1",
        "operator_execution": execution,
        "operator_packet": packet,
        "operator_spec": spec,
        "question_id": f"q-{question_index + 1}",
        "question_sha256": question_sha,
        "semantic_selection": selection,
    }
    return _signed(body, "question_receipt_sha256")


@pytest.fixture(scope="module")
def sealed_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
]:
    root = tmp_path_factory.mktemp("r7-a1-answer-inputs")
    questions = [
        _question(
            index,
            selected_count=35 if index < 7 else 34,
            retained_count=12 if index < 2 else 11,
            fact_count=5 if index == 0 else 4,
            add_extra_fact=index < 9,
        )
        for index in range(answer.QUESTION_COUNT)
    ]
    compiler_payload = {
        "format": COMPILER_OUTPUTS_FORMAT,
        "physical_provider_calls_during_materialization": 0,
        "provider_calls_performed_by_core": 0,
        "response_bindings": [{}],
        "response_count": 1,
        "responses": [{}],
        "retained_transformer_token_state_bytes": 0,
    }
    compiler = _artifact(root / "compiler.json", compiler_payload)
    source_body = {
        "compiler_output_artifact_sha256": compiler.sha256,
        "compiler_request_count": 1,
        "construction_status": "materialized_with_unresolved_closure",
        "expected_question_count": answer.QUESTION_COUNT,
        "format": "memory-condense-r7-after-union-a1-preflight-v2",
        "gold_loaded": False,
        "missing_classifier_call_count": 0,
        "missing_compiler_call_count": 0,
        "missing_external_call_count": 0,
        "missing_external_request_sha256s": [],
        "provider_calls_performed_by_core": 0,
        "question_count": answer.QUESTION_COUNT,
        "question_population_sha256": identity_sha256(
            [row["question_receipt_sha256"] for row in questions]
        ),
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": dict(answer._A1_RUNTIME_FIREWALL),  # noqa: SLF001
        "union_before_exclusion": True,
    }
    source_payload = _signed(source_body, "construction_identity_sha256")
    source = _artifact(root / "source.json", source_payload)
    return source, source, compiler, compiler


def test_preflight_is_exact_cover_and_paired(
    sealed_inputs: tuple[SealedArtifact, SealedArtifact, SealedArtifact, SealedArtifact],
) -> None:
    payload, prompts = answer.build_preflight_payload(*sealed_inputs)
    artifact = _artifact(Path("preflight.json"), payload)
    validated_prompts, rows, questions = answer.validate_preflight_artifact(artifact)

    assert prompts == validated_prompts
    assert len(rows) == answer.REQUEST_COUNT == 33
    assert len(questions) == answer.QUESTION_COUNT == 11
    assert payload["selected_union_leaf_count"] == 381
    assert payload["retained_leaf_count"] == 123
    assert payload["fact_bearing_leaf_count"] == 45
    assert payload["unresolved_raw_leaf_count"] == 78
    assert payload["merged_fact_count"] == 54
    assert payload["exact_retained_cover"] is True
    assert payload["observed_max_complete_envelope_tokens"] <= 8_000
    assert payload["physical_provider_calls"] == 0
    for offset in range(0, len(rows), 3):
        raw, operator, hybrid = rows[offset : offset + 3]
        assert [row["arm"] for row in (raw, operator, hybrid)] == list(answer.ARM_LABELS)
        assert raw["allowed_handle_ids"] == operator["allowed_handle_ids"] == hybrid["allowed_handle_ids"]
        raw_payload = json.loads(raw["messages"][1]["content"])
        operator_payload = json.loads(operator["messages"][1]["content"])
        hybrid_payload = json.loads(hybrid["messages"][1]["content"])
        assert raw_payload["memory"] == operator_payload["memory"]
        assert raw_payload["operator_projection"] is None
        assert operator_payload["operator_projection"] == hybrid_payload["operator_projection"]


def test_exact_cover_rejects_fact_unresolved_overlap(
    sealed_inputs: tuple[SealedArtifact, SealedArtifact, SealedArtifact, SealedArtifact],
) -> None:
    question = copy.deepcopy(sealed_inputs[0].payload["questions"][0])
    closure = question["fact_closure"]
    fact_handle = closure["merged_facts"][0]["leaf_handle_ids"][0]
    outcome = next(
        row for row in closure["leaf_outcomes"] if row["handle_id"] == fact_handle
    )
    outcome["disposition"] = "unresolved"
    outcome["receipt_sha256"] = identity_sha256(
        {key: value for key, value in outcome.items() if key != "receipt_sha256"}
    )
    closure["receipt_sha256"] = identity_sha256(
        {key: value for key, value in closure.items() if key != "receipt_sha256"}
    )
    question["question_receipt_sha256"] = identity_sha256(
        {
            key: value
            for key, value in question.items()
            if key != "question_receipt_sha256"
        }
    )
    with pytest.raises(answer.R7A1TerminalAnswerError, match="merged fact escaped"):
        answer._question_prompt_rows(question)  # noqa: SLF001


def test_provider_firewall_and_completion_contract() -> None:
    assert answer._forbidden_provider_keys(  # noqa: SLF001
        {"nested": {"reference_answer": "forbidden"}}
    ) == {"reference_answer"}
    prediction, used, receipt = answer._parse_completion(  # noqa: SLF001
        '{"response_text":"The cobalt kettle.","used_handle_ids":["H1"]}',
        ("H1", "H2"),
    )
    assert prediction == "The cobalt kettle."
    assert used == ("H1",)
    assert len(receipt) == 64
    with pytest.raises(answer.R7A1TerminalAnswerError, match="invalid evidence"):
        answer._parse_completion(  # noqa: SLF001
            '{"response_text":"No.","used_handle_ids":["H9"]}',
            ("H1",),
        )


def _reseal_preflight(payload: dict[str, Any]) -> SealedArtifact:
    payload["prompt_population"] = [
        row["prompt_row_receipt_sha256"] for row in payload["prompt_rows"]
    ]
    payload["prompt_population_sha256"] = identity_sha256(payload["prompt_population"])
    payload["model_prompt_population_sha256"] = identity_sha256(
        [row["messages_sha256"] for row in payload["prompt_rows"]]
    )
    payload["construction_identity_sha256"] = identity_sha256(
        {key: value for key, value in payload.items() if key != "construction_identity_sha256"}
    )
    return _artifact(Path("coherently-resealed-preflight.json"), payload)


def test_coherent_reseal_rejects_foreign_handle(
    sealed_inputs: tuple[SealedArtifact, SealedArtifact, SealedArtifact, SealedArtifact],
) -> None:
    payload, _ = answer.build_preflight_payload(*sealed_inputs)
    changed = copy.deepcopy(payload)
    row = changed["prompt_rows"][0]
    provider = json.loads(row["messages"][1]["content"])
    provider["memory"]["raw_summaries"].append(
        {"group_handle": "G999999", "handle_id": "H999999", "summary": "foreign"}
    )
    row["messages"][1]["content"] = json.dumps(provider, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    row["allowed_handle_ids"].append("H999999")
    row["messages_sha256"] = identity_sha256(row["messages"])
    row["request_sha256"] = identity_sha256(
        {"max_tokens": answer.OUTPUT_TOKEN_RESERVE, "messages": row["messages"], "model": answer.DEFAULT_MODEL}
    )
    row["prompt_row_receipt_sha256"] = identity_sha256(
        {key: value for key, value in row.items() if key != "prompt_row_receipt_sha256"}
    )
    with pytest.raises(answer.R7A1TerminalAnswerError):
        answer.validate_preflight_artifact(_reseal_preflight(changed))


def test_coherent_reseal_rejects_upstream_digest_swap(
    sealed_inputs: tuple[SealedArtifact, SealedArtifact, SealedArtifact, SealedArtifact],
) -> None:
    payload, _ = answer.build_preflight_payload(*sealed_inputs)
    changed = copy.deepcopy(payload)
    changed["source_a1_construction_artifact_sha256"] = "a" * 64
    with pytest.raises(answer.R7A1TerminalAnswerError):
        answer.validate_preflight_artifact(_reseal_preflight(changed))


def test_release_is_separate_and_provider_free(
    sealed_inputs: tuple[SealedArtifact, SealedArtifact, SealedArtifact, SealedArtifact],
    tmp_path: Path,
) -> None:
    payload, _ = answer.build_preflight_payload(*sealed_inputs)
    preflight = _artifact(tmp_path / answer.PREFLIGHT_NAME, payload)
    replay = _artifact(tmp_path / answer.PREFLIGHT_REPLAY_NAME, payload)
    release_payload = answer._release_payload(preflight, replay, output_root=tmp_path)  # noqa: SLF001
    release = _artifact(tmp_path / answer.RELEASE_NAME, release_payload)
    answer._validate_release(  # noqa: SLF001
        release, preflight=preflight, preflight_replay=replay, output_root=tmp_path
    )
    assert release.payload["provider_calls_during_release"] == 0
    assert release.payload["retry_count"] == 0
    assert release.payload["request_count"] == 33


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls.append(dict(request))
        provider = json.loads(request["messages"][1]["content"])
        memory = provider["memory"]
        if memory["typed_facts"]:
                handle = memory["typed_facts"][0]["handle_ids"][0]
        else:
            handle = memory["raw_summaries"][0]["handle_id"]
        completion = json.dumps(
            {
                "response_text": "A concise supported response.",
                "used_handle_ids": [handle],
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
            id=f"fake-a1-answer-{len(self.calls):03d}",
            model=answer.DEFAULT_MODEL,
            usage=None,
        )


class _FakeClient:
    max_retries = 0

    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def close(self) -> None:
        return None


def test_fake_lifecycle_is_zero_retry_and_byte_replays(
    sealed_inputs: tuple[SealedArtifact, SealedArtifact, SealedArtifact, SealedArtifact],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _ = answer.build_preflight_payload(*sealed_inputs, max_concurrency=3)
    preflight, _ = publish_sealed_json(tmp_path / answer.PREFLIGHT_NAME, payload)
    preflight_replay, _ = publish_sealed_json(tmp_path / answer.PREFLIGHT_REPLAY_NAME, payload)
    args = SimpleNamespace(
        api_key_env="TEST_TERRA_KEY",
        approve_provider_release=True,
        authorized_provider_calls=answer.REQUEST_COUNT,
        enable_provider=True,
        expected_preflight_construction_sha256=preflight.sha256,
        expected_preflight_replay_sha256=preflight_replay.sha256,
        expected_release_sha256=None,
        expected_run_sha256=None,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=tmp_path,
    )
    release_result = answer.run_approve_release(args)
    args.expected_release_sha256 = release_result["release_sha256"]
    fake = _FakeClient()
    monkeypatch.setattr(answer, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_TERRA_KEY", "fake-key")
    monkeypatch.setattr(answer.live, "_make_provider_client", lambda *_args: fake)
    provider = answer.run_provider(args)
    assert provider["physical_provider_calls"] == answer.REQUEST_COUNT
    assert len(fake.completions.calls) == answer.REQUEST_COUNT
    assert all(
        set(call) == {"max_tokens", "messages", "model"}
        and call["max_tokens"] == answer.OUTPUT_TOKEN_RESERVE
        and call["model"] == answer.DEFAULT_MODEL
        for call in fake.completions.calls
    )

    materialized = answer.run_materialize(args)
    args.expected_run_sha256 = materialized["run_sha256"]
    replayed = answer.run_replay(args)
    assert replayed["byte_identical"] is True
    assert replayed["replay_sha256"] == materialized["run_sha256"]
    run = read_sealed_json(tmp_path / answer.RUN_NAME)
    replay = read_sealed_json(tmp_path / answer.REPLAY_NAME)
    loaded_run, loaded_replay, judge_rows = answer.load_verified_answer_run(
        tmp_path,
        expected_preflight_construction_sha256=preflight.sha256,
        expected_preflight_replay_sha256=preflight_replay.sha256,
        expected_release_sha256=release_result["release_sha256"],
        expected_run_sha256=run.sha256,
        expected_replay_sha256=replay.sha256,
    )
    assert loaded_run.sha256 == loaded_replay.sha256
    assert len(judge_rows) == answer.REQUEST_COUNT
    assert {arm: sum(row["arm"] == arm for row in judge_rows) for arm in answer.ARM_LABELS} == {arm: 11 for arm in answer.ARM_LABELS}

    with pytest.raises(answer.R7A1TerminalAnswerError):
        answer.load_verified_answer_run(
            tmp_path,
            expected_preflight_construction_sha256="a" * 64,
            expected_preflight_replay_sha256=preflight_replay.sha256,
            expected_release_sha256=release_result["release_sha256"],
            expected_run_sha256=run.sha256,
            expected_replay_sha256=replay.sha256,
        )
    with pytest.raises(answer.R7A1TerminalAnswerError):
        answer.load_verified_answer_run(
            tmp_path,
            expected_preflight_construction_sha256=preflight.sha256,
            expected_preflight_replay_sha256=preflight_replay.sha256,
            expected_release_sha256="b" * 64,
            expected_run_sha256=run.sha256,
            expected_replay_sha256=replay.sha256,
        )

    copied_root = tmp_path.parent / f"{tmp_path.name}-wrong-root"
    shutil.copytree(tmp_path, copied_root)
    with pytest.raises(answer.R7A1TerminalAnswerError, match="root|release"):
        answer.load_verified_answer_run(
            copied_root,
            expected_preflight_construction_sha256=preflight.sha256,
            expected_preflight_replay_sha256=preflight_replay.sha256,
            expected_release_sha256=release_result["release_sha256"],
            expected_run_sha256=run.sha256,
            expected_replay_sha256=replay.sha256,
        )

    response_journals = list((tmp_path / answer.CHECKPOINT_DIR_NAME).glob("*.response.json"))
    assert len(response_journals) == answer.REQUEST_COUNT
    response_journals[0].unlink()
    with pytest.raises(answer.R7A1TerminalAnswerError, match="checkpoint"):
        answer.run_materialize(args)
    with pytest.raises(answer.R7A1TerminalAnswerError, match="journal|checkpoint"):
        answer.load_verified_answer_run(
            tmp_path,
            expected_preflight_construction_sha256=preflight.sha256,
            expected_preflight_replay_sha256=preflight_replay.sha256,
            expected_release_sha256=release_result["release_sha256"],
            expected_run_sha256=run.sha256,
            expected_replay_sha256=replay.sha256,
        )


def test_cli_has_no_ordinal_or_target_routing_switch() -> None:
    parser = answer.build_parser()
    option_strings: set[str] = set()

    def collect(current: Any) -> None:
        for action in current._actions:  # noqa: SLF001 - parser contract assay
            option_strings.update(action.option_strings)
            for child in (getattr(action, "choices", None) or {}).values():
                collect(child)

    collect(parser)
    rendered = " ".join(sorted(option_strings)).casefold()
    assert "ordinal" not in rendered
    assert "target" not in rendered
