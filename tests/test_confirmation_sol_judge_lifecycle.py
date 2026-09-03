from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval.benchmark import build_judge_prompt
from tools import confirmation_gold_judge_scaffold as judge
from tools import confirmation_sol_judge_lifecycle as subject
from tools.v4_population_firebreak.canonical import canonical_sha256


def _publish_plan(root: Path, count: int = 3):
    ids = [f"q-{index}" for index in range(count)]
    rows = []
    for index, question_id in enumerate(ids):
        body = {
            "format": judge.JUDGE_PLAN_ROW_FORMAT,
            "question_id": question_id,
            "question": f"Question {index}?",
            "reference_answer": f"Reference {index}",
            "prediction": f"Prediction {index}",
        }
        rows.append({**body, "row_receipt_sha256": canonical_sha256(body)})
    body = {
        "format": judge.JUDGE_PLAN_FORMAT,
        "status": "compiled",
        "bindings": {
            "policy_manifest_sha256": "a" * 64,
            "treatment_file_sha256": "b" * 64,
            "treatment_preflight_sha256": "c" * 64,
            "predictions_file_sha256": "d" * 64,
            "prediction_handoff_sha256": "2" * 64,
            "prediction_run_manifest_sha256": "3" * 64,
            "dataset_sha256": "e" * 64,
            "split_manifest_sha256": "f" * 64,
        },
        "population": {
            "question_count": count,
            "ordered_question_ids_sha256": canonical_sha256(ids),
        },
        "exposure_audit": {
            "audit_sha256": "1" * 64,
            "potentially_exposed_count": 0,
            "ordered_potentially_exposed_ids_sha256": canonical_sha256([]),
            "membership_emitted_to_judge_rows": False,
            "answer_values_emitted": False,
        },
        "execution": {
            "provider_class": "sol",
            "would_call_count": count,
            "count_basis": "one-call-per-sealed-confirmation-prediction",
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "authorization_released": False,
        },
        "rows": rows,
    }
    payload = {**body, "plan_identity_sha256": canonical_sha256(body)}
    path = root / "judge-plan.json"
    artifact, _ = judge.publish_sealed_json(path, payload)
    return path, artifact, rows


class _FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.lock = threading.Lock()

    def create(self, **request):
        with self.lock:
            self.requests.append(request)
            index = len(self.requests)
        return SimpleNamespace(
            id=f"sol-{index}",
            model="fake-sol",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="CORRECT" if index % 2 else "INCORRECT"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=1, total_tokens=11
            ),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.max_retries = 0
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _Factory:
    def __init__(self, explode: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.client = _FakeClient()
        self.explode = explode

    def __call__(self, gateway: str, key_env: str):
        self.calls.append((gateway, key_env))
        if self.explode:
            raise AssertionError("provider factory must be unreachable")
        return self.client


def _prepare(root: Path, count: int = 3):
    plan_path, plan, rows = _publish_plan(root, count)
    output = root / "run"
    preflight, created = subject.publish_preflight(
        judge_plan_path=plan_path,
        expected_judge_plan_sha256=plan.sha256,
        output_root=output,
    )
    assert created
    release, created = subject.approve_provider_release(
        judge_plan_path=plan_path,
        expected_judge_plan_sha256=plan.sha256,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        approve_provider_release=True,
        authorized_provider_calls=count,
    )
    assert created
    return plan_path, plan, rows, output, preflight, release


def _provider_args(prepared):
    plan_path, plan, _rows, output, preflight, release = prepared
    return {
        "judge_plan_path": plan_path,
        "expected_judge_plan_sha256": plan.sha256,
        "output_root": output,
        "expected_preflight_sha256": preflight.sha256,
        "expected_release_sha256": release.sha256,
    }


@pytest.mark.parametrize("count", [1, 4, 11])
def test_preflight_and_release_are_exact_and_provider_free(
    tmp_path: Path, count: int
) -> None:
    prepared = _prepare(tmp_path, count)
    *_, output, preflight, release = prepared
    assert preflight.payload["population"]["question_count"] == count
    assert preflight.payload["execution"]["would_call_count"] == count
    assert release.payload["required_authorized_provider_calls"] == count
    assert release.payload["provider_calls_during_release"] == 0
    assert not (output / subject.CHECKPOINT_DIR_NAME).exists()


def test_standard_prompt_contains_only_question_reference_prediction(
    tmp_path: Path,
) -> None:
    plan_path, plan, rows = _publish_plan(tmp_path, 2)
    verified = subject.verify_judge_plan(plan_path, expected_sha256=plan.sha256)
    assert verified.prompts == tuple(
        tuple(
            dict(message)
            for message in build_judge_prompt(
                row["question"], row["reference_answer"], row["prediction"]
            )
        )
        for row in rows
    )
    assert "potentially_exposed" not in str(verified.prompts).casefold()


def test_provider_requires_exact_release_before_factory(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path, 2)
    factory = _Factory(explode=True)
    with pytest.raises(
        subject.ConfirmationSolLifecycleError, match="exactly equal remaining"
    ):
        subject.run_provider(
            **_provider_args(prepared),
            enable_provider=True,
            authorized_provider_calls=1,
            client_factory=factory,
        )
    assert factory.calls == []


def test_run_materialize_replay_and_scaffold_decode(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path, 3)
    factory = _Factory()
    run = subject.run_provider(
        **_provider_args(prepared),
        enable_provider=True,
        authorized_provider_calls=3,
        client_factory=factory,
    )
    assert run["physical_provider_calls"] == 3
    assert len(factory.client.chat.completions.requests) == 3
    assert all(
        request["model"] == subject.SOL_MODEL
        and request["max_tokens"] == subject.JUDGE_MAX_TOKENS
        for request in factory.client.chat.completions.requests
    )

    completion, results = subject.materialize(**_provider_args(prepared))
    assert completion.payload["physical_provider_calls_during_materialization"] == 0
    assert [row["verdict"] for row in results.payload["rows"]] == [
        "correct",
        "incorrect",
        "correct",
    ]
    plan_path, plan, *_ = prepared
    ids = judge._validate_judge_plan(plan)  # noqa: SLF001
    assert judge._decode_judge_results(  # noqa: SLF001
        results, judge_plan_sha256=plan.sha256, question_ids=ids
    ) == (True, False, True)

    replay, same_results = subject.replay(
        **_provider_args(prepared),
        expected_completion_sha256=completion.sha256,
        expected_results_sha256=results.sha256,
    )
    assert replay.sha256 == completion.sha256
    assert same_results.sha256 == results.sha256

    no_call = _Factory(explode=True)
    resumed = subject.run_provider(
        **_provider_args(prepared),
        enable_provider=True,
        authorized_provider_calls=0,
        client_factory=no_call,
    )
    assert resumed["physical_provider_calls"] == 0
    assert no_call.calls == []


def test_request_only_journal_is_terminal_before_factory(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path, 2)
    output = prepared[3]
    checkpoint = output / subject.CHECKPOINT_DIR_NAME
    checkpoint.mkdir()
    (checkpoint / f"{'a' * 64}.request.json").write_text("{}\n", encoding="utf-8")
    factory = _Factory(explode=True)
    with pytest.raises(subject.ConfirmationSolLifecycleError, match="incomplete"):
        subject.run_provider(
            **_provider_args(prepared),
            enable_provider=True,
            authorized_provider_calls=2,
            client_factory=factory,
        )
    assert factory.calls == []


def test_invalid_verdict_fails_materialization(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path, 1)
    factory = _Factory()
    factory.client.chat.completions.create = lambda **_request: SimpleNamespace(
        id="invalid",
        model="fake-sol",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="maybe"), finish_reason="stop"
            )
        ],
        usage=None,
    )
    subject.run_provider(
        **_provider_args(prepared),
        enable_provider=True,
        authorized_provider_calls=1,
        client_factory=factory,
    )
    with pytest.raises(subject.ConfirmationSolLifecycleError, match="verdict 0"):
        subject.materialize(**_provider_args(prepared))


def test_default_client_factory_loads_dotenv_only_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dotenv
    from tools.matched_eval import live

    key_name = "CONFIRMATION_TEST_SOL_LITELLM_KEY"
    sentinel = object()
    loads: list[bool] = []
    monkeypatch.delenv(key_name, raising=False)

    def load_dotenv(*, override: bool) -> bool:
        loads.append(override)
        monkeypatch.setenv(key_name, "sealed-sol-test-key")
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", load_dotenv)
    monkeypatch.setattr(
        live,
        "_make_provider_client",
        lambda api_key, gateway: sentinel
        if (api_key, gateway) == ("sealed-sol-test-key", "https://gateway.test/v1")
        else None,
    )

    assert loads == []
    assert (
        subject._default_client_factory(  # noqa: SLF001 - construction seam test
            "https://gateway.test/v1", key_name
        )
        is sentinel
    )
    assert loads == [False]


def test_cli_has_closed_lifecycle_commands() -> None:
    subparsers = next(
        action for action in subject.build_parser()._actions if getattr(action, "choices", None)
    )
    assert set(subparsers.choices) == {
        "preflight",
        "approve-release",
        "provider-run",
        "materialize",
        "replay",
    }
