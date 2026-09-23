from __future__ import annotations

import copy
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import assay_hot_reduced30_construction as harness
from tools import run_hot_reduced30_answer_judge as runner
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256


def _selection_payload(*, duplicate_prompts: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for reduced_ordinal, (ordinal, question_id, question_sha) in enumerate(
        harness.LOCKED_QUESTIONS
    ):
        marker = 0 if duplicate_prompts else ordinal
        messages = [
            {"role": "system", "content": "Use only this sealed memory packet."},
            {"role": "user", "content": f"Memory packet and question {marker}."},
        ]
        encoded = canonical_json_bytes({"messages": messages})
        arm = {
            "provider_messages": messages,
            "provider_payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "provider_payload_utf8_bytes": len(encoded),
            "context_token_proxy": 100,
        }
        body = {
            "format": "synthetic-successor-reduced30-row-v1",
            "global_ordinal": ordinal,
            "prompt_question_sha256": question_sha,
            "question_id": question_id,
            "reduced_ordinal": reduced_ordinal,
            "source_row": {"successor": {"packet": arm}},
            "telemetry": {"arm_path": "successor.packet"},
        }
        rows.append({**body, "row_receipt_sha256": identity_sha256(body)})
    body = {
        "format": harness.FORMAT,
        "status": "sealed_provider_free_reduced30_construction",
        "gold_fields_present": False,
        "provider_calls": 0,
        "question_count": harness.QUESTION_COUNT,
        "locked_question_identity_sha256": harness.LOCKED_IDENTITY_SHA256,
        "questions": rows,
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _publish_selection(root: Path, *, duplicate_prompts: bool = False):
    return publish_sealed_json(
        root / "selection.json",
        _selection_payload(duplicate_prompts=duplicate_prompts),
    )[0]


def _prepare_answer_preflight(tmp_path: Path, *, max_concurrency: int = 4):
    selection = _publish_selection(tmp_path / "selection")
    output = tmp_path / "evaluation"
    result = runner.answer_preflight(
        selection_path=selection.path,
        expected_selection_sha256=selection.sha256,
        output_root=output,
        gateway_url=runner.DEFAULT_GATEWAY_URL,
        max_concurrency=max_concurrency,
    )
    preflight = read_sealed_json(output / runner.ANSWER_PREFLIGHT_NAME)
    return selection, output, preflight, result


class _FakeCompletions:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._fail = fail

    def create(self, **request: Any) -> Any:
        with self._lock:
            self.calls.append(copy.deepcopy(request))
            ordinal = len(self.calls)
        if self._fail:
            raise RuntimeError("simulated uncertain provider failure")
        if request["model"] == runner.SOL_MODEL:
            completion = "CORRECT"
        else:
            prompt_sha = identity_sha256(request["messages"])
            completion = f"sealed prediction {prompt_sha[:16]}"
        return SimpleNamespace(
            id=f"fake-{ordinal}",
            model=request["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        )


class _FakeClient:
    max_retries = 0

    def __init__(self, *, fail: bool = False) -> None:
        self.completions = _FakeCompletions(fail=fail)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _run_fake_answers(tmp_path: Path):
    selection, output, preflight, _result = _prepare_answer_preflight(tmp_path)
    client = _FakeClient()
    result = runner.answer_run(
        selection_path=selection.path,
        expected_selection_sha256=selection.sha256,
        output_root=output,
        expected_answer_preflight_sha256=preflight.sha256,
        authorized_provider_calls=30,
        enable_provider=True,
        client_factory=lambda: client,
    )
    answers = read_sealed_json(output / runner.ANSWERS_NAME)
    return selection, output, preflight, answers, client, result


def _fake_judge_material(
    _dataset: Path,
    _split_manifest: Path,
    prediction_rows: list[dict[str, Any]],
):
    prompts: list[list[dict[str, str]]] = []
    bindings: list[dict[str, Any]] = []
    for reduced_ordinal, ((ordinal, question_id, dated_sha), prediction) in enumerate(
        zip(harness.LOCKED_QUESTIONS, prediction_rows, strict=True)
    ):
        question = f"Plain benchmark question {ordinal}?"
        reference = f"Reference value {ordinal}."
        prompts.append(
            [
                {"role": "system", "content": "Return CORRECT or INCORRECT."},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\nReference: {reference}\n"
                        f"Prediction: {prediction['prediction']}"
                    ),
                },
            ]
        )
        bindings.append(
            {
                "reduced_ordinal": reduced_ordinal,
                "global_ordinal": ordinal,
                "question_id": question_id,
                "prompt_question_sha256": dated_sha,
                "judge_question_sha256": quote_sha256(question),
                "reference_sha256": quote_sha256(reference),
                "prediction_sha256": prediction["prediction_sha256"],
                "category": "synthetic",
            }
        )
    return "a" * 64, prompts, bindings


def test_judge_material_addresses_the_locked_validation_full100_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = [
        SimpleNamespace(
            question_id=f"unused-{ordinal}",
            dated_question=f"[Question asked at 2026-01-01] unused {ordinal}?",
            question=f"Unused question {ordinal}?",
            answer=f"Unused answer {ordinal}.",
            category="unused",
        )
        for ordinal in range(100)
    ]
    predictions: list[dict[str, Any]] = []
    for reduced_ordinal, (ordinal, question_id, dated_sha) in enumerate(
        harness.LOCKED_QUESTIONS
    ):
        dated_question = f"[Question asked at 2026-02-01] locked {ordinal}?"
        # The lock binds a historical digest.  Give the synthetic question an
        # explicit value whose digest path is controlled below so this test is
        # about global ordinal addressing rather than fixture text recovery.
        questions[ordinal] = SimpleNamespace(
            question_id=question_id,
            dated_question=dated_question,
            question=f"Locked question {ordinal}?",
            answer=f"Reference {ordinal}.",
            category="synthetic",
        )
        predictions.append(
            {
                "prediction": f"Prediction {reduced_ordinal}.",
                "prediction_sha256": quote_sha256(
                    f"Prediction {reduced_ordinal}."
                ),
            }
        )

    original_quote_sha256 = runner.quote_sha256

    def lock_aware_quote(value: str) -> str:
        for ordinal, _question_id, dated_sha in harness.LOCKED_QUESTIONS:
            if value == f"[Question asked at 2026-02-01] locked {ordinal}?":
                return dated_sha
        return original_quote_sha256(value)

    monkeypatch.setattr(runner, "quote_sha256", lock_aware_quote)
    monkeypatch.setattr(
        runner,
        "_load_locked_validation_question_population",
        lambda _dataset, _split: ("a" * 64, questions),
    )
    population_sha, prompts, bindings = runner._load_judge_material(
        Path("unused-dataset"),
        Path("unused-split"),
        predictions,
    )

    assert population_sha == "a" * 64
    assert len(prompts) == len(bindings) == 30
    assert [row["global_ordinal"] for row in bindings] == [
        row[0] for row in harness.LOCKED_QUESTIONS
    ]
    assert bindings[0]["question_id"] == "06878be2"
    assert "Locked question 5?" in prompts[0][1]["content"]


def test_answer_preflight_is_gold_free_provider_free_and_format_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("preflight crossed a provider or gold boundary")

    monkeypatch.setattr(runner, "_completion_client", forbidden)
    monkeypatch.setattr(runner, "_load_judge_material", forbidden)
    selection, output, preflight, result = _prepare_answer_preflight(tmp_path)

    assert called is False
    assert not (output / runner.ANSWER_CHECKPOINT_DIR).exists()
    assert result["new_provider_calls"] == 0
    assert result["required_authorized_provider_calls"] == 30
    assert preflight.payload["gold_fields_present"] is False
    assert preflight.payload["prompt_population"]["logical_prompt_count"] == 30
    assert preflight.payload["prompt_population"]["unique_prompt_count"] == 30
    assert preflight.payload["questions"][0]["global_ordinal"] == 5
    assert preflight.payload["questions"][0]["question_id"] == "06878be2"
    assert preflight.payload["questions"][0]["arm_path"] == "successor.packet"
    assert preflight.payload["selection_sha256"] == selection.sha256


def test_answer_preflight_rejects_prompt_deduplication(tmp_path: Path) -> None:
    selection = _publish_selection(tmp_path / "selection", duplicate_prompts=True)
    with pytest.raises(
        runner.Reduced30LifecycleError,
        match="exact unique reduced-30",
    ):
        runner.answer_preflight(
            selection_path=selection.path,
            expected_selection_sha256=selection.sha256,
            output_root=tmp_path / "evaluation",
            gateway_url=runner.DEFAULT_GATEWAY_URL,
            max_concurrency=4,
        )


@pytest.mark.parametrize(
    ("authorized", "enabled"),
    [(29, True), (30, False)],
)
def test_answer_run_rejects_bad_authorization_before_client_creation(
    tmp_path: Path,
    authorized: int,
    enabled: bool,
) -> None:
    selection, output, preflight, _result = _prepare_answer_preflight(tmp_path)
    factory_calls = 0

    def factory() -> _FakeClient:
        nonlocal factory_calls
        factory_calls += 1
        return _FakeClient()

    with pytest.raises(runner.Reduced30LifecycleError):
        runner.answer_run(
            selection_path=selection.path,
            expected_selection_sha256=selection.sha256,
            output_root=output,
            expected_answer_preflight_sha256=preflight.sha256,
            authorized_provider_calls=authorized,
            enable_provider=enabled,
            client_factory=factory,
        )
    assert factory_calls == 0


def test_terra_run_is_exact_checkpointed_sealed_and_replay_safe(
    tmp_path: Path,
) -> None:
    selection, output, preflight, answers, client, result = _run_fake_answers(
        tmp_path
    )
    assert len(client.completions.calls) == 30
    assert client.closed is True
    assert {row["model"] for row in client.completions.calls} == {
        runner.TERRA_MODEL
    }
    assert all("max_retries" not in row for row in client.completions.calls)
    assert result["new_provider_calls"] == 30
    assert result["authenticated_checkpoint_hits"] == 0
    assert result["completion_batch_wall_time_s"] >= 0
    assert answers.payload["model"] == runner.TERRA_MODEL
    assert answers.payload["gateway_url"] == runner.DEFAULT_GATEWAY_URL
    assert answers.payload["retries"] == 0
    assert answers.payload["provider_calls_completed"] == 30
    assert len(answers.payload["questions"]) == 30
    assert all(row["provider_elapsed_s"] >= 0 for row in answers.payload["questions"])
    assert answers.payload["completion_usage"]["recorded_provider_elapsed_s"] >= 0
    assert all(
        row["prediction_sha256"] == quote_sha256(row["prediction"])
        for row in answers.payload["questions"]
    )
    assert len(list((output / runner.ANSWER_CHECKPOINT_DIR).glob("*.request.json"))) == 30
    assert len(list((output / runner.ANSWER_CHECKPOINT_DIR).glob("*.response.json"))) == 30

    replay_factory_calls = 0

    def forbidden_factory() -> _FakeClient:
        nonlocal replay_factory_calls
        replay_factory_calls += 1
        raise AssertionError("sealed replay must not create a provider client")

    replay = runner.answer_run(
        selection_path=selection.path,
        expected_selection_sha256=selection.sha256,
        output_root=output,
        expected_answer_preflight_sha256=preflight.sha256,
        authorized_provider_calls=0,
        enable_provider=False,
        client_factory=forbidden_factory,
    )
    assert replay_factory_calls == 0
    assert replay["answers_sha256"] == answers.sha256
    assert replay["new_provider_calls"] == 0
    assert replay["authenticated_checkpoint_hits"] == 30

    (output / f"{runner.ANSWERS_NAME}.sha256").unlink()
    (output / runner.ANSWERS_NAME).unlink()
    rebuilt = runner.answer_run(
        selection_path=selection.path,
        expected_selection_sha256=selection.sha256,
        output_root=output,
        expected_answer_preflight_sha256=preflight.sha256,
        authorized_provider_calls=0,
        enable_provider=False,
        client_factory=forbidden_factory,
    )
    assert rebuilt["answers_sha256"] == answers.sha256
    assert rebuilt["new_provider_calls"] == 0


def test_request_only_checkpoint_refuses_duplicate_call(
    tmp_path: Path,
) -> None:
    selection, output, preflight, _result = _prepare_answer_preflight(
        tmp_path, max_concurrency=1
    )
    failing = _FakeClient(fail=True)
    with pytest.raises(RuntimeError, match="simulated uncertain"):
        runner.answer_run(
            selection_path=selection.path,
            expected_selection_sha256=selection.sha256,
            output_root=output,
            expected_answer_preflight_sha256=preflight.sha256,
            authorized_provider_calls=30,
            enable_provider=True,
            client_factory=lambda: failing,
        )
    checkpoints = output / runner.ANSWER_CHECKPOINT_DIR
    assert len(list(checkpoints.glob("*.request.json"))) == 1
    assert len(list(checkpoints.glob("*.response.json"))) == 0

    replacement_factory_calls = 0

    def replacement_factory() -> _FakeClient:
        nonlocal replacement_factory_calls
        replacement_factory_calls += 1
        return _FakeClient()

    with pytest.raises(RuntimeError, match="refusing an unsafe retry"):
        runner.answer_run(
            selection_path=selection.path,
            expected_selection_sha256=selection.sha256,
            output_root=output,
            expected_answer_preflight_sha256=preflight.sha256,
            authorized_provider_calls=30,
            enable_provider=True,
            client_factory=replacement_factory,
        )
    assert replacement_factory_calls == 0


def test_invalid_predictions_fail_before_first_gold_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, source_output, preflight, answers, _client, _result = (
        _run_fake_answers(tmp_path / "source")
    )
    bad_output = tmp_path / "bad-evaluation"
    publish_sealed_json(
        bad_output / runner.ANSWER_PREFLIGHT_NAME,
        preflight.payload,
    )
    bad = copy.deepcopy(answers.payload)
    bad_row = bad["questions"][0]
    bad_row["prediction"] = "tampered prediction"
    row_body = dict(bad_row)
    row_body.pop("row_receipt_sha256")
    bad_row["row_receipt_sha256"] = identity_sha256(row_body)
    body = dict(bad)
    body.pop("receipt_sha256")
    bad["receipt_sha256"] = identity_sha256(body)
    bad_artifact, _created = publish_sealed_json(
        bad_output / runner.ANSWERS_NAME,
        bad,
    )
    gold_reads = 0

    def forbidden_gold(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal gold_reads
        gold_reads += 1
        raise AssertionError("gold was read before predictions were authenticated")

    monkeypatch.setattr(runner, "_load_judge_material", forbidden_gold)
    with pytest.raises(
        runner.Reduced30LifecycleError,
        match="prediction text hash changed",
    ):
        runner.judge_preflight(
            selection_path=selection.path,
            expected_selection_sha256=selection.sha256,
            output_root=bad_output,
            expected_answer_preflight_sha256=preflight.sha256,
            expected_answers_sha256=bad_artifact.sha256,
            dataset=Path("not-opened.json"),
            split_manifest=Path("not-opened-split.json"),
            gateway_url=runner.DEFAULT_GATEWAY_URL,
            max_concurrency=4,
        )
    assert gold_reads == 0
    assert source_output != bad_output


def test_sol_lifecycle_uses_sealed_gold_plan_and_exact_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, output, answer_plan, answers, _terra, _answer_result = (
        _run_fake_answers(tmp_path)
    )
    gold_reads = 0

    def load_material(*args: Any, **kwargs: Any):
        nonlocal gold_reads
        gold_reads += 1
        return _fake_judge_material(*args, **kwargs)

    monkeypatch.setattr(runner, "_load_judge_material", load_material)
    plan_result = runner.judge_preflight(
        selection_path=selection.path,
        expected_selection_sha256=selection.sha256,
        output_root=output,
        expected_answer_preflight_sha256=answer_plan.sha256,
        expected_answers_sha256=answers.sha256,
        dataset=Path("synthetic-dataset.json"),
        split_manifest=Path("synthetic-split.json"),
        gateway_url=runner.DEFAULT_GATEWAY_URL,
        max_concurrency=5,
    )
    assert gold_reads == 1
    assert plan_result["new_provider_calls"] == 0
    judge_plan = read_sealed_json(output / runner.JUDGE_PREFLIGHT_NAME)
    assert judge_plan.payload["gold_fields_present"] is True
    assert judge_plan.payload["required_authorized_provider_calls"] == 30
    assert judge_plan.payload["questions"][0]["global_ordinal"] == 5

    sol = _FakeClient()
    result = runner.judge_run(
        selection_path=selection.path,
        expected_selection_sha256=selection.sha256,
        output_root=output,
        expected_answer_preflight_sha256=answer_plan.sha256,
        expected_answers_sha256=answers.sha256,
        expected_judge_preflight_sha256=judge_plan.sha256,
        authorized_provider_calls=30,
        enable_provider=True,
        client_factory=lambda: sol,
    )
    judgments = read_sealed_json(output / runner.JUDGMENTS_NAME)
    assert len(sol.completions.calls) == 30
    assert sol.closed is True
    assert {row["model"] for row in sol.completions.calls} == {runner.SOL_MODEL}
    assert result["new_provider_calls"] == 30
    assert result["correct_count"] == 30
    assert result["completion_batch_wall_time_s"] >= 0
    assert judgments.payload["correct_count"] == 30
    assert judgments.payload["accuracy"] == 1.0
    assert judgments.payload["retries"] == 0
    assert judgments.payload["provider_calls_completed"] == 30
    assert all(row["provider_elapsed_s"] >= 0 for row in judgments.payload["questions"])

    replay_factory_calls = 0

    def forbidden_factory() -> _FakeClient:
        nonlocal replay_factory_calls
        replay_factory_calls += 1
        raise AssertionError("sealed judgment replay must not call Sol")

    replay = runner.judge_run(
        selection_path=selection.path,
        expected_selection_sha256=selection.sha256,
        output_root=output,
        expected_answer_preflight_sha256=answer_plan.sha256,
        expected_answers_sha256=answers.sha256,
        expected_judge_preflight_sha256=judge_plan.sha256,
        authorized_provider_calls=0,
        enable_provider=False,
        client_factory=forbidden_factory,
    )
    assert replay_factory_calls == 0
    assert replay["judgments_sha256"] == judgments.sha256
    assert replay["new_provider_calls"] == 0
