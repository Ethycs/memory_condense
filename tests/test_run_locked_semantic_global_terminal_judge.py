from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_semantic_global_terminal_judge as judge
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_judging import TypedFinalJudgeGoldRow


def _sha(value: str) -> str:
    return quote_sha256(value)


def _artifact(path: Path, label: str) -> SealedArtifact:
    return SealedArtifact(path, _sha(label), {"label": label})


def _postseal_binding() -> dict[str, Any]:
    return {
        "postseal_promotion_audit_artifact_sha256": _sha("promotion audit"),
        "postseal_promotion_audit_identity_sha256": _sha("promotion identity"),
        "postseal_semantic_atom_count": (
            judge.answer_cli.postseal_cli.SEMANTIC_ATOM_COUNT
        ),
        "postseal_semantic_atom_final_usable_count": (
            judge.answer_cli.postseal_cli.SEMANTIC_ATOM_COUNT
        ),
        "postseal_semantic_atom_manifest_artifact_sha256": (
            judge.answer_cli.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
        ),
        "postseal_semantic_atom_manifest_identity_sha256": (
            judge.answer_cli.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_IDENTITY_SHA256
        ),
        "postseal_semantic_atom_population_sha256": (
            judge.answer_cli.postseal_cli.DEFAULT_SEMANTIC_ATOM_POPULATION_SHA256
        ),
        "postseal_source_final_usable_count": 7,
        "postseal_source_target_count": judge.answer_cli.postseal_cli.SOURCE_TARGET_COUNT,
        "postseal_target_plan_artifact_sha256": (
            judge.answer_cli.postseal_cli.DEFAULT_TARGET_PLAN_SHA256
        ),
        "postseal_target_plan_identity_sha256": (
            judge.answer_cli.postseal_cli.DEFAULT_TARGET_PLAN_IDENTITY_SHA256
        ),
        "postseal_witness_final_usable_count": 9,
        "postseal_witness_manifest_artifact_sha256": (
            judge.answer_cli.postseal_cli.DEFAULT_WITNESS_MANIFEST_SHA256
        ),
        "postseal_witness_manifest_identity_sha256": (
            judge.answer_cli.postseal_cli.DEFAULT_WITNESS_MANIFEST_IDENTITY_SHA256
        ),
        "postseal_witness_positive_count": (
            judge.answer_cli.postseal_cli.POSITIVE_WITNESS_COUNT
        ),
    }


def _source_and_gold() -> tuple[
    tuple[dict[str, Any], ...], tuple[TypedFinalJudgeGoldRow, ...]
]:
    sources: list[dict[str, Any]] = []
    gold: list[TypedFinalJudgeGoldRow] = []
    for ordinal in judge.EXACT_ORDINALS:
        question_id = f"terminal-question-{ordinal:03d}"
        question = f"What is the sealed terminal value for {ordinal}?"
        dated = f"[Question asked at 2024/02/20 23:40]\n{question}"
        prediction = f"sealed prediction {ordinal}"
        reference = f"sealed reference {ordinal}"
        source_body = {
            "changed_from_parent": ordinal % 2 == 0,
            "dated_question_sha256": _sha(dated),
            "format": "memory-condense-semantic-global-terminal-judge-row-v1",
            "ordinal": ordinal,
            "parent_prediction_sha256": _sha(f"parent prediction {ordinal}"),
            "prediction": prediction,
            "prediction_sha256": _sha(prediction),
            "prediction_source": "semantic_global_terminal_v1",
            "question_id": question_id,
            "question_sha256": _sha(question),
            "route_id": "semantic-global-terminal-terra-answer-v1",
        }
        sources.append(
            {
                **source_body,
                "source_row_sha256": identity_sha256(source_body),
            }
        )
        gold.append(
            TypedFinalJudgeGoldRow(
                ordinal,
                question_id,
                question,
                _sha(question),
                dated,
                _sha(dated),
                reference,
                _sha(reference),
                "synthetic-terminal",
            )
        )
    return tuple(sources), tuple(gold)


def _answer_artifacts(tmp_path: Path):
    run = SealedArtifact(
        tmp_path / "answer-run.json",
        _sha("answer run"),
        {"label": "answer run", **_postseal_binding()},
    )
    return run, _artifact(tmp_path / "answer-replay.json", "answer replay")


def _preflight_args(tmp_path: Path, run: SealedArtifact, replay: SealedArtifact):
    return SimpleNamespace(
        answer_root=tmp_path / "terra-answer",
        dataset=tmp_path / "locked.json",
        expected_answer_preflight_sha256=_sha("answer preflight"),
        expected_answer_replay_sha256=replay.sha256,
        expected_answer_run_sha256=run.sha256,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=tmp_path / "sol-judge",
        max_concurrency=3,
        model=judge.DEFAULT_MODEL,
        postseal_audit=tmp_path / "postseal-promotion.json",
        expected_postseal_audit_sha256=_sha("promotion audit"),
        split=tmp_path / "locked-split.json",
    )


def _install_preflight_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    run: SealedArtifact,
    replay: SealedArtifact,
    sources: tuple[dict[str, Any], ...],
    gold: tuple[TypedFinalJudgeGoldRow, ...],
) -> list[tuple[str, object]]:
    calls: list[tuple[str, object]] = []

    def answer_reader(
        root: str | Path,
        *,
        expected_preflight_sha256: str,
        expected_run_sha256: str,
        expected_replay_sha256: str,
        postseal_audit: str | Path,
        expected_postseal_audit_sha256: str,
    ):
        calls.append(
            (
                "answer",
                (
                    Path(root),
                    expected_preflight_sha256,
                    expected_run_sha256,
                    expected_replay_sha256,
                    Path(postseal_audit),
                    expected_postseal_audit_sha256,
                ),
            )
        )
        return run, replay, sources

    def gold_reader(**kwargs: Any):
        calls.append(("gold", kwargs))
        assert kwargs["allow_subset"] is True
        assert kwargs["source_rows"] == sources
        return gold, _sha("terminal exact11 gold population")

    monkeypatch.setattr(judge.answer_cli, "load_verified_answer_run", answer_reader)
    monkeypatch.setattr(judge, "load_locked_typed_final_gold", gold_reader)
    return calls


def _make_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimpleNamespace, dict[str, Any]]:
    sources, gold = _source_and_gold()
    run, replay = _answer_artifacts(tmp_path)
    _install_preflight_dependencies(monkeypatch, run, replay, sources, gold)
    args = _preflight_args(tmp_path, run, replay)
    return args, judge.run_preflight(args)


def test_preflight_authenticates_terra_before_gold_and_builds_11_unique_sol_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, gold = _source_and_gold()
    run, replay = _answer_artifacts(tmp_path)
    calls = _install_preflight_dependencies(
        monkeypatch, run, replay, sources, gold
    )
    args = _preflight_args(tmp_path, run, replay)

    first = judge.run_preflight(args)
    second = judge.run_preflight(args)
    artifact = read_sealed_json(
        Path(args.judge_output_root) / judge.PREFLIGHT_NAME
    )
    prompts, rows = judge._validate_preflight(artifact)  # noqa: SLF001

    assert [kind for kind, _ in calls] == ["answer", "gold", "answer", "gold"]
    assert first["created"] is True
    assert second["created"] is False
    assert first["preflight_sha256"] == second["preflight_sha256"] == artifact.sha256
    assert first["physical_provider_calls"] == 0
    assert first["judge_mode"] == "selected_subset"
    assert first["required_authorized_provider_calls"] == 11
    assert first["selected_ordinals"] == list(judge.EXACT_ORDINALS)
    assert artifact.payload["gold_loaded"] is True
    assert artifact.payload["question_count"] == 100
    assert artifact.payload["selected_question_count"] == 11
    assert artifact.payload["postseal_semantic_atom_count"] == 26
    assert artifact.payload["postseal_semantic_atom_final_usable_count"] == 26
    assert artifact.payload["postseal_source_final_usable_count"] == 7
    assert artifact.payload["postseal_witness_final_usable_count"] == 9
    assert tuple(row["ordinal"] for row in rows) == judge.EXACT_ORDINALS
    assert len(prompts) == len({row["messages_sha256"] for row in rows}) == 11
    for messages, source, reference in zip(prompts, sources, gold, strict=True):
        rendered = "\n".join(message["content"] for message in messages)
        assert reference.question in rendered
        assert reference.reference in rendered
        assert source["prediction"] in rendered
    assert not (
        Path(args.judge_output_root) / judge.CHECKPOINT_DIR_NAME
    ).exists()


def test_preflight_can_authenticate_the_distinct_validator_v4_run_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, gold = _source_and_gold()
    run, replay = _answer_artifacts(tmp_path)
    args = _preflight_args(tmp_path, run, replay)
    args.answer_validator_v4_root = tmp_path / "validator-v4"
    args.expected_answer_validator_v4_run_sha256 = run.sha256
    args.expected_answer_validator_v4_replay_sha256 = replay.sha256
    calls: list[str] = []

    monkeypatch.setattr(
        judge.answer_cli,
        "load_verified_answer_run",
        lambda *_args, **_kwargs: pytest.fail("legacy answer source selected"),
    )

    def v4_reader(root: str | Path, **kwargs: Any):
        calls.append("validator_v4")
        assert Path(root) == args.answer_validator_v4_root
        assert kwargs["answer_root"] == args.answer_root
        assert kwargs["expected_validator_run_sha256"] == run.sha256
        assert kwargs["expected_validator_replay_sha256"] == replay.sha256
        return run, replay, sources

    def gold_reader(**kwargs: Any):
        calls.append("gold")
        assert kwargs["source_rows"] == sources
        return gold, _sha("terminal exact11 gold population")

    monkeypatch.setattr(
        judge.validator_v4_cli,
        "load_verified_revalidated_answer_run",
        v4_reader,
    )
    monkeypatch.setattr(judge, "load_locked_typed_final_gold", gold_reader)

    result = judge.run_preflight(args)
    assert calls == ["validator_v4", "gold"]
    assert result["physical_provider_calls"] == 0
    assert result["selected_question_count"] == 11


def test_preflight_rejects_reordered_terra_rows_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, _gold = _source_and_gold()
    run, replay = _answer_artifacts(tmp_path)
    monkeypatch.setattr(
        judge.answer_cli,
        "load_verified_answer_run",
        lambda *_args, **_kwargs: (run, replay, tuple(reversed(sources))),
    )
    monkeypatch.setattr(
        judge,
        "load_locked_typed_final_gold",
        lambda **_kwargs: pytest.fail("gold opened for reordered Terra rows"),
    )

    with pytest.raises(
        judge.LockedSemanticGlobalTerminalJudgeError,
        match="population/order",
    ):
        judge.run_preflight(_preflight_args(tmp_path, run, replay))


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("postseal_semantic_atom_count", 25),
        ("postseal_semantic_atom_final_usable_count", 25),
        (
            "postseal_semantic_atom_manifest_artifact_sha256",
            _sha("wrong atom manifest"),
        ),
    ),
    ids=("atom-count", "atom-usable", "atom-manifest-binding"),
)
def test_preflight_rejects_semantic_atom_mutation_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: Any,
) -> None:
    sources, _gold = _source_and_gold()
    run, replay = _answer_artifacts(tmp_path)
    mutated_payload = {**run.payload, key: value}
    mutated_run = SealedArtifact(
        run.path,
        _sha(f"mutated {key}"),
        mutated_payload,
    )
    monkeypatch.setattr(
        judge.answer_cli,
        "load_verified_answer_run",
        lambda *_args, **_kwargs: (mutated_run, replay, sources),
    )
    monkeypatch.setattr(
        judge,
        "load_locked_typed_final_gold",
        lambda **_kwargs: pytest.fail("gold opened for invalid atom binding"),
    )

    with pytest.raises(
        judge.LockedSemanticGlobalTerminalJudgeError,
        match="promotion binding changed",
    ):
        judge.run_preflight(_preflight_args(tmp_path, mutated_run, replay))


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls.append(dict(request))
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="CORRECT"),
                ),
            ),
            id=f"fake-sol-response-{len(self.calls)}",
            model=judge.DEFAULT_MODEL,
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


def _provider_args(preflight_args: SimpleNamespace, preflight_sha: str):
    return SimpleNamespace(
        api_key_env="SEALED_SOL_KEY",
        authorized_provider_calls=7,
        enable_provider=True,
        expected_judge_preflight_sha256=preflight_sha,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=preflight_args.judge_output_root,
        max_concurrency=3,
        model=judge.DEFAULT_MODEL,
    )


def test_provider_authenticates_and_resumes_four_of_eleven_sol_journals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_args, result = _make_preflight(tmp_path, monkeypatch)
    preflight, prompts, _rows = judge._read_preflight(  # noqa: SLF001
        Path(preflight_args.judge_output_root), result["preflight_sha256"]
    )
    args = _provider_args(preflight_args, result["preflight_sha256"])
    seed_client = _FakeClient()
    runtime = judge._runtime(  # noqa: SLF001
        preflight, prompts, args=args, client=seed_client
    )
    assert runtime.provenance.benchmark_provenance[
        "postseal_semantic_atom_manifest_artifact_sha256"
    ] == judge.answer_cli.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
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
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment opened before remaining-call check"),
    )
    with pytest.raises(
        judge.LockedSemanticGlobalTerminalJudgeError,
        match="exactly equal remaining",
    ):
        judge.run_provider(args)

    resume_client = _FakeClient()
    args.authorized_provider_calls = 7
    monkeypatch.setattr(judge, "load_dotenv", lambda: None)
    monkeypatch.setenv("SEALED_SOL_KEY", "sealed-test-key")
    monkeypatch.setattr(
        judge.judging,
        "_make_provider_client",
        lambda *_args, **_kwargs: resume_client,
    )
    resumed = judge.run_provider(args)

    assert resumed["authorized_remaining_provider_calls"] == 7
    assert resumed["physical_provider_calls"] == 7
    assert resumed["checkpoint_hits"] == 4
    assert resumed["retained_transformer_token_state_bytes"] == 0
    assert len(resume_client.completions.calls) == 7

    args.authorized_provider_calls = 0
    monkeypatch.setattr(
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment opened for completed checkpoint replay"),
    )
    complete = judge.run_provider(args)
    assert complete["authorized_remaining_provider_calls"] == 0
    assert complete["physical_provider_calls"] == 0
    assert complete["checkpoint_hits"] == 11


def test_provider_rejects_foreign_checkpoint_before_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_args, result = _make_preflight(tmp_path, monkeypatch)
    checkpoint = (
        Path(preflight_args.judge_output_root) / judge.CHECKPOINT_DIR_NAME
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "foreign.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment opened before journal validation"),
    )
    args = _provider_args(preflight_args, result["preflight_sha256"])
    args.authorized_provider_calls = 11

    with pytest.raises(ValueError, match="unexpected JSON completion journal"):
        judge.run_provider(args)


def test_checkpoint_only_materialize_replay_and_public_reader_reject_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_args, result = _make_preflight(tmp_path, monkeypatch)
    preflight, prompts, _rows = judge._read_preflight(  # noqa: SLF001
        Path(preflight_args.judge_output_root), result["preflight_sha256"]
    )
    args = _provider_args(preflight_args, result["preflight_sha256"])
    seed = _FakeClient()
    runtime = judge._runtime(  # noqa: SLF001
        preflight, prompts, args=args, client=seed
    )
    try:
        runtime.run()
    finally:
        runtime.close()
    assert len(seed.completions.calls) == 11

    offline_args = SimpleNamespace(
        expected_judge_preflight_sha256=result["preflight_sha256"],
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=preflight_args.judge_output_root,
        max_concurrency=3,
        model=judge.DEFAULT_MODEL,
    )
    materialized = judge.run_materialize(offline_args)
    replay_args = SimpleNamespace(
        **vars(offline_args),
        expected_judge_sha256=materialized["judge_sha256"],
        expected_score_sha256=materialized["score_sha256"],
    )
    replayed = judge.run_replay(replay_args)
    judge_artifact, score_artifact, rows = judge.load_verified_judge_run(
        preflight_args.judge_output_root,
        expected_preflight_sha256=result["preflight_sha256"],
        expected_judge_sha256=materialized["judge_sha256"],
        expected_score_sha256=materialized["score_sha256"],
        expected_judge_replay_sha256=replayed["judge_replay_sha256"],
        expected_score_replay_sha256=replayed["score_replay_sha256"],
    )

    assert materialized["physical_provider_calls"] == 0
    assert materialized["checkpoint_hits"] == 11
    assert materialized["correct"] == 11
    assert replayed["physical_provider_calls"] == 0
    assert replayed["byte_identical"] is True
    assert replayed["judge_replay_sha256"] == judge_artifact.sha256
    assert replayed["score_replay_sha256"] == score_artifact.sha256
    assert len(rows) == 11
    assert tuple(row["ordinal"] for row in rows) == judge.EXACT_ORDINALS
    assert score_artifact.payload["selected_accuracy"] == 1.0
    assert score_artifact.payload["postseal_semantic_atom_count"] == 26
    assert judge_artifact.payload[
        "postseal_semantic_atom_manifest_artifact_sha256"
    ] == judge.answer_cli.postseal_cli.DEFAULT_SEMANTIC_ATOM_MANIFEST_SHA256
    assert judge_artifact.payload["retained_transformer_token_state_bytes"] == 0

    mutated = dict(judge_artifact.payload)
    mutated_rows = [dict(row) for row in mutated["questions"]]
    mutated_rows[0]["prediction_sha256"] = _sha("mutated prediction")
    mutated["questions"] = mutated_rows
    judge_path = Path(preflight_args.judge_output_root) / judge.JUDGE_NAME
    judge_path.unlink()
    judge_path.with_name(judge_path.name + ".sha256").unlink()
    mutated_artifact, _ = publish_sealed_json(
        judge_path,
        mutated,
    )
    with pytest.raises(
        judge.LockedSemanticGlobalTerminalJudgeError,
        match="artifacts changed|verdict row",
    ):
        judge.load_verified_judge_run(
            preflight_args.judge_output_root,
            expected_preflight_sha256=result["preflight_sha256"],
            expected_judge_sha256=mutated_artifact.sha256,
            expected_score_sha256=materialized["score_sha256"],
            expected_judge_replay_sha256=replayed["judge_replay_sha256"],
            expected_score_replay_sha256=replayed["score_replay_sha256"],
        )
