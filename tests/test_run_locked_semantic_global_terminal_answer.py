from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256


TERMINAL_MODULE = "tools.run_reduced_semantic_global_terminal_assay"
TERMINAL_ORDINALS = (14, 28, 40, 49, 53, 54, 67, 69, 82, 94, 97)


def _load_answer_module():
    stubbed = importlib.util.find_spec(TERMINAL_MODULE) is None
    if stubbed:
        terminal = ModuleType(TERMINAL_MODULE)
        terminal.FORMAT = "memory-condense-reduced-semantic-global-terminal-assay-v1"
        terminal.CONSTRUCTION_NAME = "reduced-semantic-global-terminal-assay-v1.json"
        terminal.REPLAY_NAME = (
            "reduced-semantic-global-terminal-assay-replay-v1.json"
        )
        terminal.DEFAULT_OUTPUT_ROOT = Path("synthetic-terminal-root")
        terminal.EXACT_ORDINALS = TERMINAL_ORDINALS

        def unavailable(*_args: Any, **_kwargs: Any):
            raise AssertionError("terminal reader was not replaced by the test")

        terminal.load_verified_terminal_assay = unavailable
        sys.modules[TERMINAL_MODULE] = terminal
    try:
        return importlib.import_module(
            "tools.run_locked_semantic_global_terminal_answer"
        )
    finally:
        if stubbed:
            sys.modules.pop(TERMINAL_MODULE, None)


answer = _load_answer_module()


def _sha(label: str) -> str:
    return quote_sha256(label)


def _artifact(path: Path, label: str, payload: dict[str, Any]) -> SealedArtifact:
    return SealedArtifact(path, _sha(label), payload)


def _terminal_sources(tmp_path: Path):
    payload = {
        "format": "memory-condense-reduced-semantic-global-terminal-assay-v1",
        "question_count": answer.QUESTION_COUNT,
    }
    construction = _artifact(tmp_path / "terminal.json", "terminal", payload)
    replay = _artifact(tmp_path / "terminal-replay.json", "terminal replay", payload)
    return construction, replay


def _promotion_audit(
    tmp_path: Path,
    construction: SealedArtifact,
    replay: SealedArtifact,
    *,
    semantic_atom_count: int | None = None,
    semantic_atom_final_usable_count: int | None = None,
    semantic_atom_manifest_artifact_sha256: str | None = None,
    source_final_usable_count: int | None = None,
    witness_final_usable_count: int | None = None,
) -> SealedArtifact:
    atom_count = (
        answer.postseal_cli.SEMANTIC_ATOM_COUNT
        if semantic_atom_count is None
        else semantic_atom_count
    )
    atom_usable = (
        answer.postseal_cli.SEMANTIC_ATOM_COUNT
        if semantic_atom_final_usable_count is None
        else semantic_atom_final_usable_count
    )
    source_usable = (
        answer.postseal_cli.SOURCE_TARGET_COUNT
        if source_final_usable_count is None
        else source_final_usable_count
    )
    witness_usable = (
        answer.postseal_cli.POSITIVE_WITNESS_COUNT
        if witness_final_usable_count is None
        else witness_final_usable_count
    )
    body = {
        "audit_identity_sha256": _sha("promotion identity"),
        "promotion_gate_passed": True,
        "semantic_atom_manifest_artifact_sha256": (
            answer.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
            if semantic_atom_manifest_artifact_sha256 is None
            else semantic_atom_manifest_artifact_sha256
        ),
        "semantic_atom_manifest_identity_sha256": (
            answer.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256
        ),
        "semantic_atom_population_sha256": (
            answer.postseal_cli.DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256
        ),
        "target_plan_artifact_sha256": answer.postseal_cli.DEFAULT_TARGET_PLAN_SHA256,
        "target_plan_identity_sha256": (
            answer.postseal_cli.DEFAULT_TARGET_PLAN_IDENTITY_SHA256
        ),
        "terminal_construction_sha256": construction.sha256,
        "terminal_replay_sha256": replay.sha256,
        "totals": {
            "fact_final_usable_count": witness_usable,
            "positive_witness_count": answer.postseal_cli.POSITIVE_WITNESS_COUNT,
            "raw_witness_final_usable_count": witness_usable,
            "semantic_atom_count": atom_count,
            "semantic_atom_final_usable_count": atom_usable,
            "source_final_usable_count": source_usable,
            "source_target_count": answer.postseal_cli.SOURCE_TARGET_COUNT,
        },
        "witness_manifest_artifact_sha256": (
            answer.postseal_cli.DEFAULT_WITNESS_MANIFEST_SHA256
        ),
        "witness_manifest_identity_sha256": (
            answer.postseal_cli.DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256
        ),
    }
    return _artifact(tmp_path / "postseal-promotion.json", "promotion audit", body)


def _answer_plans(
    construction: SealedArtifact,
    replay: SealedArtifact,
) -> tuple[dict[str, Any], ...]:
    plans: list[dict[str, Any]] = []
    for ordinal in answer.EXACT_ORDINALS:
        question_id = f"terminal-question-{ordinal:03d}"
        dated_question = (
            f"[Question asked at 2024/02/20 23:40]\n"
            f"What is the sealed memory for {ordinal}?"
        )
        parent = f"parent prediction {ordinal}"
        provider_input = {
            "dated_question": dated_question,
            "format": "synthetic-terminal-provider-input-v1",
            "protected_parent_fallback": {"prediction": parent},
        }
        messages = answer.render_final_messages(provider_input)
        compilation_receipt = _sha(f"terminal compilation {ordinal}")
        body = {
            "allowed_handle_ids": [],
            "dated_question": dated_question,
            "dated_question_sha256": quote_sha256(dated_question),
            "format": "synthetic-terminal-answer-plan-v1",
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
            "question_id": question_id,
            "question_sha256": _sha(f"question {ordinal}"),
            "route_id": answer.ROUTE_ID,
            "source_artifact_bindings": {
                "terminal_construction": construction.sha256,
                "terminal_replay": replay.sha256,
            },
            "story_coherence": {},
            "terminal_compilation": {
                "format": "synthetic-terminal-compilation-v1",
                "receipt_sha256": compilation_receipt,
            },
            "terminal_compilation_receipt_sha256": compilation_receipt,
            "validation_contract": {},
        }
        plans.append(
            {
                **body,
                "answer_plan_receipt_sha256": identity_sha256(body),
            }
        )
    return tuple(plans)


def _preflight_args(tmp_path: Path, construction: SealedArtifact, replay: SealedArtifact):
    return SimpleNamespace(
        expected_terminal_construction_sha256=construction.sha256,
        expected_terminal_replay_sha256=replay.sha256,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=tmp_path / "answer",
        postseal_audit=tmp_path / "postseal-promotion.json",
        expected_postseal_audit_sha256=_sha("promotion audit"),
        terminal_root=tmp_path / "terminal-source",
    )


def _install_source_reader(
    monkeypatch: pytest.MonkeyPatch,
    construction: SealedArtifact,
    replay: SealedArtifact,
    plans: tuple[dict[str, Any], ...],
    events: list[str] | None = None,
) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def reader(root, expected_construct, expected_replay):
        if events is not None:
            events.append("terminal")
        calls.append((Path(root), expected_construct, expected_replay))
        return construction, replay, plans

    promotion = _promotion_audit(construction.path.parent, construction, replay)

    def promotion_reader(
        path,
        expected_sha256,
        *,
        expected_terminal_construction_sha256,
        expected_terminal_replay_sha256,
        **_kwargs,
    ):
        if events is not None:
            events.append("postseal")
        assert Path(path).name == "postseal-promotion.json"
        assert expected_sha256 == promotion.sha256
        assert expected_terminal_construction_sha256 == construction.sha256
        assert expected_terminal_replay_sha256 == replay.sha256
        return promotion

    monkeypatch.setattr(
        answer.terminal_cli, "load_verified_terminal_assay", reader
    )
    monkeypatch.setattr(
        answer.postseal_cli, "load_verified_promotion_audit", promotion_reader
    )
    return calls


def _approve_release(
    preflight_args: SimpleNamespace,
    preflight_sha256: str,
) -> tuple[dict[str, Any], SealedArtifact]:
    release_args = SimpleNamespace(
        **vars(preflight_args),
        approve_provider_release=True,
        expected_preflight_sha256=preflight_sha256,
    )
    result = answer.run_approve_release(release_args)
    artifact = read_sealed_json(
        Path(preflight_args.output_root) / answer.RELEASE_NAME
    )
    assert result["release_sha256"] == artifact.sha256
    return result, artifact


def test_preflight_is_exact_11_gold_free_and_delegates_terminal_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, replay)
    events: list[str] = []
    calls = _install_source_reader(
        monkeypatch, construction, replay, plans, events=events
    )
    args = _preflight_args(tmp_path, construction, replay)

    first = answer.run_preflight(args)
    second = answer.run_preflight(args)
    artifact = read_sealed_json(Path(args.output_root) / answer.PREFLIGHT_NAME)
    prompts, rows = answer._validate_preflight(artifact)  # noqa: SLF001

    assert len(calls) == 2
    assert events == ["terminal", "postseal", "terminal", "postseal"]
    assert calls[0] == (
        Path(args.terminal_root),
        construction.sha256,
        replay.sha256,
    )
    assert first["created"] is True
    assert second["created"] is False
    assert first["preflight_sha256"] == second["preflight_sha256"] == artifact.sha256
    assert first["physical_provider_calls"] == 0
    assert artifact.payload["gold_loaded"] is False
    assert artifact.payload["question_count"] == answer.QUESTION_COUNT == 11
    assert artifact.payload["required_authorized_provider_calls"] == 11
    assert artifact.payload["exact_ordinals"] == list(answer.EXACT_ORDINALS)
    assert len(prompts) == len(rows) == 11
    assert tuple(row["ordinal"] for row in rows) == answer.EXACT_ORDINALS
    assert all(
        "postseal" not in "\n".join(message["content"] for message in prompt)
        and answer.postseal_cli.DEFAULT_WITNESS_MANIFEST_SHA256
        not in "\n".join(message["content"] for message in prompt)
        for prompt in prompts
    )
    assert all(
        row["prompt_token_proxy"] + answer.OUTPUT_TOKEN_RESERVE
        <= answer.HARD_PROMPT_TOKEN_CAP
        for row in rows
    )
    assert not (Path(args.output_root) / answer.CHECKPOINT_DIR_NAME).exists()


def test_preflight_rejects_fitted_prompt_mirror_drift() -> None:
    construction, replay = _terminal_sources(Path("synthetic"))
    plans = list(_answer_plans(construction, replay))
    drifted = dict(plans[0])
    drifted["messages_sha256"] = _sha("different messages")
    unsigned = dict(drifted)
    unsigned.pop("answer_plan_receipt_sha256")
    drifted["answer_plan_receipt_sha256"] = identity_sha256(unsigned)
    plans[0] = drifted

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="authenticated mirrors",
    ):
        answer.build_preflight_payload(
            construction,
            replay,
            tuple(plans),
            promotion_audit=_promotion_audit(Path("synthetic"), construction, replay),
            model=answer.DEFAULT_MODEL,
            gateway_url=answer.DEFAULT_GATEWAY_URL,
            max_concurrency=2,
        )


def test_atom_complete_promotion_accepts_incomplete_raw_and_source_diagnostics() -> None:
    construction, replay = _terminal_sources(Path("synthetic"))
    payload, _ = answer.build_preflight_payload(
        construction,
        replay,
        _answer_plans(construction, replay),
        promotion_audit=_promotion_audit(
            Path("synthetic"),
            construction,
            replay,
            source_final_usable_count=7,
            witness_final_usable_count=9,
        ),
        model=answer.DEFAULT_MODEL,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=2,
    )
    answer._validate_preflight(  # noqa: SLF001
        _artifact(Path("synthetic/preflight.json"), "preflight", payload)
    )

    assert payload["postseal_semantic_atom_count"] == 26
    assert payload["postseal_semantic_atom_final_usable_count"] == 26
    assert payload["postseal_source_final_usable_count"] == 7
    assert payload["postseal_witness_final_usable_count"] == 9
    assert payload["postseal_semantic_atom_manifest_artifact_sha256"] == (
        answer.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
    )


@pytest.mark.parametrize(
    "audit_overrides",
    (
        {"semantic_atom_count": 25},
        {"semantic_atom_final_usable_count": 25},
        {"semantic_atom_manifest_artifact_sha256": _sha("wrong atom manifest")},
    ),
    ids=("atom-count", "atom-usable", "atom-manifest-binding"),
)
def test_preflight_rejects_incomplete_or_rebound_semantic_atoms(
    audit_overrides: dict[str, Any],
) -> None:
    construction, replay = _terminal_sources(Path("synthetic"))

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="promotion binding changed",
    ):
        answer.build_preflight_payload(
            construction,
            replay,
            _answer_plans(construction, replay),
            promotion_audit=_promotion_audit(
                Path("synthetic"),
                construction,
                replay,
                **audit_overrides,
            ),
            model=answer.DEFAULT_MODEL,
            gateway_url=answer.DEFAULT_GATEWAY_URL,
            max_concurrency=2,
        )


def test_release_requires_explicit_opt_in_and_seals_exact_gold_free_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, replay)
    calls = _install_source_reader(monkeypatch, construction, replay, plans)
    preflight_args = _preflight_args(tmp_path, construction, replay)
    preflight = answer.run_preflight(preflight_args)
    release_args = SimpleNamespace(
        **vars(preflight_args),
        approve_provider_release=False,
        expected_preflight_sha256=preflight["preflight_sha256"],
    )

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="explicit provider-release approval",
    ):
        answer.run_approve_release(release_args)
    assert len(calls) == 1
    assert not (Path(preflight_args.output_root) / answer.RELEASE_NAME).exists()

    release_args.approve_provider_release = True
    first = answer.run_approve_release(release_args)
    second = answer.run_approve_release(release_args)
    artifact = read_sealed_json(
        Path(preflight_args.output_root) / answer.RELEASE_NAME
    )

    assert first["created"] is True
    assert second["created"] is False
    assert first["release_sha256"] == second["release_sha256"] == artifact.sha256
    assert artifact.payload["gold_loaded"] is False
    assert artifact.payload["approval_opt_in"] is True
    assert artifact.payload["release_status"] == "approved_for_provider_execution"
    assert artifact.payload["preflight_artifact_sha256"] == preflight[
        "preflight_sha256"
    ]
    assert artifact.payload["terminal_construction_artifact_sha256"] == construction.sha256
    assert artifact.payload["terminal_replay_artifact_sha256"] == replay.sha256
    assert artifact.payload["prompt_population_sha256"]
    assert artifact.payload["answer_plan_population_sha256"]
    assert artifact.payload["provider_calls_during_release"] == 0
    assert not (Path(preflight_args.output_root) / answer.CHECKPOINT_DIR_NAME).exists()


def test_provider_rejects_absent_or_mismatched_release_before_checkpoint_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, replay)
    _install_source_reader(monkeypatch, construction, replay, plans)
    preflight_args = _preflight_args(tmp_path, construction, replay)
    preflight = answer.run_preflight(preflight_args)
    args = SimpleNamespace(
        api_key_env="SEALED_KEY",
        authorized_provider_calls=11,
        enable_provider=True,
        expected_preflight_sha256=preflight["preflight_sha256"],
        expected_release_sha256=_sha("missing release"),
        expected_postseal_audit_sha256=preflight_args.expected_postseal_audit_sha256,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=preflight_args.output_root,
        postseal_audit=preflight_args.postseal_audit,
    )
    monkeypatch.setattr(
        answer,
        "load_dotenv",
        lambda: pytest.fail("environment opened before release validation"),
    )

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="absent or invalid",
    ):
        answer.run_provider(args)
    assert not (Path(args.output_root) / answer.CHECKPOINT_DIR_NAME).exists()

    release, _artifact = _approve_release(
        preflight_args, preflight["preflight_sha256"]
    )
    args.expected_release_sha256 = _sha("different release")
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="release artifact changed",
    ):
        answer.run_provider(args)
    assert release["release_sha256"] != args.expected_release_sha256
    assert not (Path(args.output_root) / answer.CHECKPOINT_DIR_NAME).exists()


def test_provider_reauthenticates_promotion_before_checkpoint_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, replay)
    _install_source_reader(monkeypatch, construction, replay, plans)
    preflight_args = _preflight_args(tmp_path, construction, replay)
    preflight = answer.run_preflight(preflight_args)
    release, _artifact = _approve_release(
        preflight_args, preflight["preflight_sha256"]
    )
    monkeypatch.setattr(
        answer.postseal_cli,
        "load_verified_promotion_audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            answer.postseal_cli.SemanticGlobalTerminalPostSealAuditError(
                "resealed promotion forgery"
            )
        ),
    )
    monkeypatch.setattr(
        answer,
        "load_dotenv",
        lambda: pytest.fail("environment opened before promotion authentication"),
    )
    args = SimpleNamespace(
        api_key_env="SEALED_KEY",
        authorized_provider_calls=11,
        enable_provider=True,
        expected_postseal_audit_sha256=(
            preflight_args.expected_postseal_audit_sha256
        ),
        expected_preflight_sha256=preflight["preflight_sha256"],
        expected_release_sha256=release["release_sha256"],
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=preflight_args.output_root,
        postseal_audit=preflight_args.postseal_audit,
    )

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="absent, invalid, or not promoted",
    ):
        answer.run_provider(args)
    assert not (Path(args.output_root) / answer.CHECKPOINT_DIR_NAME).exists()


def test_supersession_marker_and_known_stale_shas_fail_closed_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, replay)
    _install_source_reader(monkeypatch, construction, replay, plans)
    preflight_args = _preflight_args(tmp_path, construction, replay)
    preflight = answer.run_preflight(preflight_args)
    release, _artifact = _approve_release(
        preflight_args, preflight["preflight_sha256"]
    )
    marker, _ = publish_sealed_json(
        Path(preflight_args.output_root) / answer.SUPERSESSION_NAME,
        answer.supersession_marker_payload(),
    )
    args = SimpleNamespace(
        api_key_env="SEALED_KEY",
        authorized_provider_calls=11,
        enable_provider=True,
        expected_preflight_sha256=preflight["preflight_sha256"],
        expected_release_sha256=release["release_sha256"],
        expected_postseal_audit_sha256=preflight_args.expected_postseal_audit_sha256,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=preflight_args.output_root,
        postseal_audit=preflight_args.postseal_audit,
    )
    monkeypatch.setattr(
        answer,
        "load_dotenv",
        lambda: pytest.fail("environment opened for superseded release"),
    )

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="explicitly superseded",
    ):
        answer.run_provider(args)
    assert marker.payload == answer.supersession_marker_payload()
    assert not (Path(args.output_root) / answer.CHECKPOINT_DIR_NAME).exists()

    stale = SealedArtifact(
        Path("copied-stale-preflight.json"),
        answer.SUPERSEDED_PREFLIGHT_SHA256,
        {
            "terminal_construction_artifact_sha256": answer.SUPERSEDED_TERMINAL_SHA256,
            "terminal_replay_artifact_sha256": answer.SUPERSEDED_TERMINAL_SHA256,
        },
    )
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="preflight/terminal is superseded",
    ):
        answer._assert_preflight_not_superseded(stale)  # noqa: SLF001

    copied_args = _preflight_args(tmp_path / "copy", construction, replay)
    copied_args.expected_terminal_construction_sha256 = (
        answer.SUPERSEDED_TERMINAL_SHA256
    )
    monkeypatch.setattr(
        answer.terminal_cli,
        "load_verified_terminal_assay",
        lambda *_a, **_k: pytest.fail("stale terminal reader was opened"),
    )
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="non-superseded source",
    ):
        answer.run_preflight(copied_args)


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls.append(dict(request))
        completion = (
            '{"decision":"keep_parent","prediction":"sealed",'
            '"used_handle_ids":[]}'
        )
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=completion),
                ),
            ),
            id=f"fake-response-{len(self.calls)}",
            model=answer.DEFAULT_MODEL,
            usage=None,
        )


class _FakeClient:
    max_retries = 0

    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_provider_authenticates_and_resumes_four_of_eleven_completed_journals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, replay)
    _install_source_reader(monkeypatch, construction, replay, plans)
    preflight_args = _preflight_args(tmp_path, construction, replay)
    preflight_result = answer.run_preflight(preflight_args)
    release_result, release = _approve_release(
        preflight_args, preflight_result["preflight_sha256"]
    )
    preflight, prompts, _rows = answer._read_preflight(  # noqa: SLF001
        Path(preflight_args.output_root), preflight_result["preflight_sha256"]
    )
    args = SimpleNamespace(
        api_key_env="SEALED_KEY",
        authorized_provider_calls=7,
        enable_provider=True,
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        expected_release_sha256=release_result["release_sha256"],
        expected_postseal_audit_sha256=preflight_args.expected_postseal_audit_sha256,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=preflight_args.output_root,
        postseal_audit=preflight_args.postseal_audit,
    )
    seed_client = _FakeClient()
    runtime = answer._runtime(  # noqa: SLF001
        preflight, release, prompts, args=args, client=seed_client
    )
    assert runtime.provenance.benchmark_provenance[
        "postseal_semantic_atom_manifest_artifact_sha256"
    ] == answer.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
    assert runtime.provenance.benchmark_provenance[
        "postseal_semantic_atom_final_usable_count"
    ] == 26
    try:
        for messages_sha in runtime._unique_order[:4]:  # noqa: SLF001
            runtime._provider_call(messages_sha)  # noqa: SLF001
    finally:
        runtime.close()

    args.authorized_provider_calls = 8
    monkeypatch.setattr(
        answer,
        "load_dotenv",
        lambda: pytest.fail("environment opened before remaining-call check"),
    )
    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="exactly equal remaining",
    ):
        answer.run_provider(args)

    resume_client = _FakeClient()
    args.authorized_provider_calls = 7
    monkeypatch.setattr(answer, "load_dotenv", lambda: None)
    monkeypatch.setenv("SEALED_KEY", "sealed-test-key")
    monkeypatch.setattr(
        answer.live,
        "_make_provider_client",
        lambda *_args, **_kwargs: resume_client,
    )
    resumed = answer.run_provider(args)

    assert resumed["authorized_remaining_provider_calls"] == 7
    assert resumed["physical_provider_calls"] == 7
    assert resumed["checkpoint_hits"] == 4
    assert len(resume_client.completions.calls) == 7
    assert all(call["model"] == answer.DEFAULT_MODEL for call in resume_client.completions.calls)

    args.authorized_provider_calls = 0
    monkeypatch.setattr(
        answer,
        "load_dotenv",
        lambda: pytest.fail("environment opened for complete checkpoint replay"),
    )
    complete = answer.run_provider(args)
    assert complete["authorized_remaining_provider_calls"] == 0
    assert complete["physical_provider_calls"] == 0
    assert complete["checkpoint_hits"] == 11

    provider_args = answer.build_parser().parse_args(
        [
            "provider-run",
            "--expected-preflight-sha256",
            "a" * 64,
            "--expected-release-sha256",
            "b" * 64,
            "--postseal-audit",
            "promotion.json",
            "--expected-postseal-audit-sha256",
            "c" * 64,
            "--authorized-provider-calls",
            "11",
        ]
    )
    assert not hasattr(provider_args, "terminal_root")
    assert not hasattr(provider_args, "expected_terminal_construction_sha256")


def test_provider_rejects_foreign_checkpoint_json_before_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, replay)
    _install_source_reader(monkeypatch, construction, replay, plans)
    preflight_args = _preflight_args(tmp_path, construction, replay)
    preflight = answer.run_preflight(preflight_args)
    release, _artifact = _approve_release(
        preflight_args, preflight["preflight_sha256"]
    )
    checkpoint = Path(preflight_args.output_root) / answer.CHECKPOINT_DIR_NAME
    checkpoint.mkdir(parents=True)
    (checkpoint / "foreign.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        answer,
        "load_dotenv",
        lambda: pytest.fail("environment opened before journal validation"),
    )
    args = SimpleNamespace(
        api_key_env="SEALED_KEY",
        authorized_provider_calls=11,
        enable_provider=True,
        expected_preflight_sha256=preflight["preflight_sha256"],
        expected_release_sha256=release["release_sha256"],
        expected_postseal_audit_sha256=preflight_args.expected_postseal_audit_sha256,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=preflight_args.output_root,
        postseal_audit=preflight_args.postseal_audit,
    )

    with pytest.raises(ValueError, match="unexpected JSON completion journal"):
        answer.run_provider(args)


def _checkpoint_batch(rows: tuple[dict[str, Any], ...]):
    completions = tuple(f"completion {row['ordinal']}" for row in rows)
    records = tuple(
        SimpleNamespace(
            call_key_sha256=_sha(f"call {row['ordinal']}"),
            checkpoint_hit=True,
            completion=completion,
            completion_sha256=quote_sha256(completion),
            messages_sha256=row["messages_sha256"],
            physical_call=False,
            request_journal_sha256=_sha(f"request {row['ordinal']}"),
            response_journal_sha256=_sha(f"response {row['ordinal']}"),
        )
        for row, completion in zip(rows, completions, strict=True)
    )
    return SimpleNamespace(
        logical_completions=completions,
        model_dump=lambda: {
            "format": "synthetic-completion-batch",
            "logical_count": len(rows),
        },
        unique_records=records,
        usage=SimpleNamespace(
            checkpoint_hits=len(rows),
            logical_calls=len(rows),
            physical_calls=0,
            unique_calls=len(rows),
        ),
    )


def _fake_materialized_result(
    plan: dict[str, Any],
    completion: str,
    **receipts: str,
) -> dict[str, Any]:
    parent = plan["parent_prediction"]
    body = {
        "call_key_sha256": receipts["call_key_sha256"],
        "changed_from_parent": False,
        "completion_receipt_sha256": receipts["completion_receipt_sha256"],
        "dated_question_sha256": plan["dated_question_sha256"],
        "decision": "keep_parent",
        "format": answer.RESULT_ROW_FORMAT,
        "ordinal": plan["ordinal"],
        "parent_prediction_sha256": quote_sha256(parent),
        "parse_error_code": "none",
        "parse_receipt_sha256": _sha(f"parse {plan['ordinal']}"),
        "prediction": parent,
        "prediction_sha256": quote_sha256(parent),
        "prediction_source": "typed_final_validated_keep_parent_v1",
        "prompt_row_receipt_sha256": plan["prompt_row_receipt_sha256"],
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "request_journal_sha256": receipts["request_journal_sha256"],
        "response_journal_sha256": receipts["response_journal_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "route_id": plan["route_id"],
        "solver_valid": True,
        "used_handle_ids": [],
        "validation_basis": "keep_parent_contract",
        "validator_policy_format": answer.VALIDATOR_POLICY_FORMAT,
    }
    return {**body, "source_row_sha256": identity_sha256(body)}


def test_materialize_replay_and_public_judge_reader_are_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, terminal_replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, terminal_replay)
    _install_source_reader(
        monkeypatch, construction, terminal_replay, plans
    )
    preflight_args = _preflight_args(tmp_path, construction, terminal_replay)
    preflight_result = answer.run_preflight(preflight_args)
    release_result, release = _approve_release(
        preflight_args, preflight_result["preflight_sha256"]
    )
    preflight = read_sealed_json(
        Path(preflight_args.output_root) / answer.PREFLIGHT_NAME
    )
    _prompts, rows = answer._validate_preflight(preflight)  # noqa: SLF001
    batch = _checkpoint_batch(rows)
    delegated: list[tuple[int, str]] = []

    def materializer(plan, completion, **kwargs):
        delegated.append((plan["ordinal"], completion))
        return _fake_materialized_result(plan, completion, **kwargs)

    monkeypatch.setattr(answer, "materialize_typed_final_result_row", materializer)
    monkeypatch.setattr(answer, "_checkpoint_batch", lambda *_a, **_k: batch)
    runtime_args = SimpleNamespace(
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        expected_release_sha256=release_result["release_sha256"],
        expected_postseal_audit_sha256=preflight_args.expected_postseal_audit_sha256,
        gateway_url=answer.DEFAULT_GATEWAY_URL,
        max_concurrency=3,
        model=answer.DEFAULT_MODEL,
        output_root=preflight_args.output_root,
        postseal_audit=preflight_args.postseal_audit,
    )
    materialized = answer.run_materialize(runtime_args)
    run = read_sealed_json(Path(runtime_args.output_root) / answer.RUN_NAME)

    replay_args = SimpleNamespace(
        **vars(runtime_args),
        expected_run_sha256=run.sha256,
        expected_terminal_construction_sha256=construction.sha256,
        expected_terminal_replay_sha256=terminal_replay.sha256,
        terminal_root=preflight_args.terminal_root,
    )
    replayed = answer.run_replay(replay_args)
    replay = read_sealed_json(Path(runtime_args.output_root) / answer.REPLAY_NAME)
    verified_run, verified_replay, judge_rows = answer.load_verified_answer_run(
        runtime_args.output_root,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=run.sha256,
        expected_replay_sha256=replay.sha256,
        postseal_audit=preflight_args.postseal_audit,
        expected_postseal_audit_sha256=preflight_args.expected_postseal_audit_sha256,
    )

    assert materialized["physical_provider_calls"] == 0
    assert materialized["checkpoint_hits"] == 11
    assert replayed["byte_identical"] is True
    assert replayed["physical_provider_calls"] == 0
    assert replay.payload["expected_run_sha256"] == run.sha256
    assert replay.payload["replayed_run_sha256"] == run.sha256
    assert verified_run.sha256 == run.sha256
    assert verified_replay.sha256 == replay.sha256
    assert tuple(row["ordinal"] for row in judge_rows) == answer.EXACT_ORDINALS
    assert len(delegated) == 22
    assert tuple(row[0] for row in delegated[:11]) == answer.EXACT_ORDINALS
    assert run.payload["retained_transformer_token_state_bytes"] == 0
    assert run.payload["physical_provider_calls_during_materialization"] == 0


def test_public_judge_reader_rejects_replay_binding_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction, terminal_replay = _terminal_sources(tmp_path)
    plans = _answer_plans(construction, terminal_replay)
    _install_source_reader(monkeypatch, construction, terminal_replay, plans)
    args = _preflight_args(tmp_path, construction, terminal_replay)
    preflight_result = answer.run_preflight(args)
    release_result, release = _approve_release(
        args, preflight_result["preflight_sha256"]
    )
    preflight = read_sealed_json(Path(args.output_root) / answer.PREFLIGHT_NAME)
    _prompts, rows = answer._validate_preflight(preflight)  # noqa: SLF001
    batch = _checkpoint_batch(rows)
    monkeypatch.setattr(
        answer, "materialize_typed_final_result_row", _fake_materialized_result
    )
    payload = answer._materialization_payload(  # noqa: SLF001
        preflight, release, rows, batch
    )
    run, _ = publish_sealed_json(Path(args.output_root) / answer.RUN_NAME, payload)
    bad_replay_payload = {
        "byte_identical": True,
        "expected_run_sha256": run.sha256,
        "format": answer.REPLAY_FORMAT,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        **{key: preflight.payload[key] for key in answer.POSTSEAL_BINDING_KEYS},
        "preflight_artifact_sha256": preflight.sha256,
        "replayed_run_sha256": _sha("different run"),
        "release_authorization_artifact_sha256": release_result[
            "release_sha256"
        ],
        "retained_transformer_token_state_bytes": 0,
        "terminal_construction_artifact_sha256": construction.sha256,
        "terminal_replay_artifact_sha256": terminal_replay.sha256,
    }
    replay, _ = publish_sealed_json(
        Path(args.output_root) / answer.REPLAY_NAME, bad_replay_payload
    )

    with pytest.raises(
        answer.LockedSemanticGlobalTerminalAnswerError,
        match="not exact replay-verified",
    ):
        answer.load_verified_answer_run(
            args.output_root,
            expected_preflight_sha256=preflight_result["preflight_sha256"],
            expected_run_sha256=run.sha256,
            expected_replay_sha256=replay.sha256,
            postseal_audit=args.postseal_audit,
            expected_postseal_audit_sha256=args.expected_postseal_audit_sha256,
        )
