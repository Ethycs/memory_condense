from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from tools import confirmation_terra_completion_lifecycle as subject
from tools.v4_population_firebreak.canonical import canonical_sha256


def _messages(value: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Answer only from supplied memory."},
        {"role": "user", "content": value},
    ]


def _source_payload(values: list[str]) -> dict:
    prompts = [_messages(value) for value in values]
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=7232,
    )
    question_ids = [f"q-{index}" for index in range(len(values))]
    rows = []
    for index, (question_id, messages, prompt_row) in enumerate(
        zip(question_ids, prompts, population.ordered_rows, strict=True)
    ):
        provider_body = {
            "format": subject.PROVIDER_INPUT_FORMAT,
            "messages": messages,
            "messages_sha256": prompt_row.messages_sha256,
        }
        provider_input = {
            **provider_body,
            "provider_input_receipt_sha256": canonical_sha256(provider_body),
        }
        row_body = {
            "format": "memory-condense-confirmation-synthetic-terra-preflight-row-v1",
            "row_index": index,
            "question_id": question_id,
            "source_row_receipt_sha256": canonical_sha256(
                {"question_id": question_id}
            ),
            "messages_sha256": prompt_row.messages_sha256,
            "prompt_token_proxy": prompt_row.prompt_token_proxy,
            "provider_input": provider_input,
        }
        rows.append(
            {**row_body, "row_receipt_sha256": canonical_sha256(row_body)}
        )
    body = {
        "format": subject.S0_PROMPT_FORMAT,
        "status": "compiled",
        "gold_loaded": False,
        "bindings": {"source_sha256": "a" * 64},
        "population": {
            "question_count": len(values),
            "ordered_question_ids_sha256": canonical_sha256(question_ids),
        },
        "runtime": {
            "gateway_url": subject.TERRA_GATEWAY_URL,
            "hard_complete_chat_token_cap": 8000,
            "input_token_cap": 7232,
            "max_concurrency": 3,
            "model": subject.TERRA_MODEL,
            "output_token_reserve": 768,
            "retry_count": 0,
        },
        "execution": {
            "logical_prompt_count": len(values),
            "unique_prompt_count": population.unique_prompt_count,
            "would_call_count": population.unique_prompt_count,
            "would_call_count_status": "exact",
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "authorization_released": False,
        },
        "ordered_rows": rows,
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
    }
    return {**body, "preflight_identity_sha256": canonical_sha256(body)}


def _terminal_source_payload(
    values: list[str],
    *,
    provider_indexes: set[int],
) -> dict:
    source = _source_payload(values)
    rows = []
    for index, sealed_row in enumerate(source["ordered_rows"]):
        row = dict(sealed_row)
        row.pop("row_receipt_sha256")
        row.pop("row_index")
        row.pop("messages_sha256")
        would_call = index in provider_indexes
        row["would_call"] = would_call
        if not would_call:
            row["provider_input"] = None
            row["prompt_token_proxy"] = None
        rows.append({**row, "row_receipt_sha256": canonical_sha256(row)})
    question_ids = [f"q-{index}" for index in range(len(values))]
    body = {
        "format": subject.TERMINAL_PROMPT_FORMAT,
        "status": "compiled",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "provider_execution_available": False,
        "authorization_released": False,
        "bindings": {"source_sha256": "a" * 64},
        "population": {
            "question_count": len(values),
            "ordered_question_ids_sha256": canonical_sha256(question_ids),
        },
        "runtime": source["runtime"],
        "execution": {
            "logical_terminal_prompt_count": len(provider_indexes),
            "would_call_count": len(provider_indexes),
            "would_call_count_status": "exact",
            "physical_provider_calls": 0,
            "provider_execution_available": False,
            "retry_count": 0,
        },
        "ordered_rows": rows,
    }
    return {**body, "preflight_identity_sha256": canonical_sha256(body)}


def _publish_source(tmp_path: Path, values: list[str]):
    path = tmp_path / "source-prompt.json"
    artifact, created = subject.publish_sealed_artifact(
        path,
        _source_payload(values),
    )
    assert created is True
    return path, artifact


def _publish_terminal_source(
    tmp_path: Path,
    values: list[str],
    *,
    provider_indexes: set[int],
):
    path = tmp_path / "terminal-source-prompt.json"
    artifact, created = subject.publish_sealed_artifact(
        path,
        _terminal_source_payload(values, provider_indexes=provider_indexes),
    )
    assert created is True
    return path, artifact


class _FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def create(self, **request):
        with self._lock:
            self.requests.append(request)
            number = len(self.requests)
        return SimpleNamespace(
            id=f"fake-response-{number}",
            model="fake-terra-route",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"completion-{number}"),
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
    def __init__(self) -> None:
        self.max_retries = 0
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _Factory:
    def __init__(self, *, explode: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.client = _FakeClient()
        self.explode = explode

    def __call__(self, gateway_url: str, api_key_env: str):
        self.calls.append((gateway_url, api_key_env))
        if self.explode:
            raise AssertionError("client factory must be unreachable")
        return self.client


def _publish_preflight(source_path: Path, source_sha: str, output_root: Path):
    artifact, created = subject.publish_lifecycle_preflight(
        prompt_artifact_path=source_path,
        expected_prompt_artifact_sha256=source_sha,
        output_root=output_root,
    )
    assert created is True
    return artifact


def _approve(
    source_path: Path,
    source_sha: str,
    output_root: Path,
    preflight_sha: str,
    calls: int,
):
    artifact, created = subject.approve_provider_release(
        prompt_artifact_path=source_path,
        expected_prompt_artifact_sha256=source_sha,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=preflight_sha,
        approve_provider_release=True,
        authorized_provider_calls=calls,
    )
    assert created is True
    return artifact


def _provider_kwargs(source_path, source, output_root, preflight, release):
    return {
        "prompt_artifact_path": source_path,
        "expected_prompt_artifact_sha256": source.sha256,
        "output_root": output_root,
        "expected_lifecycle_preflight_sha256": preflight.sha256,
        "expected_release_sha256": release.sha256,
        "enable_provider": True,
    }


@pytest.mark.parametrize("count", [1, 5, 11])
def test_preflight_and_release_are_provider_free_for_arbitrary_n(
    tmp_path: Path,
    count: int,
) -> None:
    source_path, source = _publish_source(
        tmp_path,
        [f"question-{index}" for index in range(count)],
    )
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)
    release = _approve(
        source_path,
        source.sha256,
        output_root,
        preflight.sha256,
        count,
    )

    assert preflight.payload["population"]["logical_prompt_count"] == count
    assert preflight.payload["population"]["unique_prompt_count"] == count
    assert preflight.payload["execution"]["physical_provider_calls"] == 0
    assert release.payload["required_authorized_provider_calls"] == count
    assert release.payload["provider_calls_during_release"] == 0
    assert not (output_root / subject.CHECKPOINT_DIR_NAME).exists()


def test_intermediate_prompt_profile_is_reusable_by_non_query_stage(
    tmp_path: Path,
) -> None:
    question_ids = ["fact-a", "fact-b", "fact-c"]
    source_receipts = [
        canonical_sha256({"source": question_id}) for question_id in question_ids
    ]
    runtime = _source_payload(["seed"])["runtime"]
    payload = subject.compile_intermediate_prompt_artifact(
        stage_id="synthetic-fact-compression",
        ordered_question_ids=question_ids,
        source_row_receipts=source_receipts,
        messages=[_messages(f"compress facts for {value}") for value in question_ids],
        runtime=runtime,
        stage_bindings={
            "source_plane_sha256": canonical_sha256({"stage": "not-query"}),
            "profile": "synthetic-non-query-intermediate-v1",
        },
    )
    source_path = tmp_path / "intermediate-prompt.json"
    source, created = subject.publish_sealed_artifact(source_path, payload)
    assert created is True

    verified = subject.verify_prompt_artifact(
        source.path,
        expected_sha256=source.sha256,
    )
    lifecycle = subject.compile_lifecycle_preflight(verified)

    assert verified.source_format == subject.INTERMEDIATE_PROMPT_FORMAT
    assert verified.question_ids == tuple(question_ids)
    assert payload["stage_id"] == "synthetic-fact-compression"
    assert lifecycle["population"]["logical_prompt_count"] == 3
    assert lifecycle["execution"]["required_provider_calls"] == 3


def test_release_requires_exact_remaining_unique_calls(tmp_path: Path) -> None:
    source_path, source = _publish_source(tmp_path, ["one", "two", "one"])
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)

    with pytest.raises(
        subject.ConfirmationTerraLifecycleError,
        match="exactly equal remaining unique",
    ):
        _approve(
            source_path,
            source.sha256,
            output_root,
            preflight.sha256,
            3,
        )


def test_provider_factory_unreachable_without_release(tmp_path: Path) -> None:
    source_path, source = _publish_source(tmp_path, ["one", "two"])
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)
    factory = _Factory(explode=True)

    with pytest.raises(subject.ConfirmationTerraLifecycleError):
        subject.run_provider_completion(
            prompt_artifact_path=source_path,
            expected_prompt_artifact_sha256=source.sha256,
            output_root=output_root,
            expected_lifecycle_preflight_sha256=preflight.sha256,
            expected_release_sha256="f" * 64,
            enable_provider=True,
            authorized_provider_calls=2,
            client_factory=factory,
        )
    assert factory.calls == []


def test_provider_factory_unreachable_on_authorization_mismatch(
    tmp_path: Path,
) -> None:
    source_path, source = _publish_source(tmp_path, ["one", "two"])
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)
    release = _approve(
        source_path,
        source.sha256,
        output_root,
        preflight.sha256,
        2,
    )
    factory = _Factory(explode=True)

    with pytest.raises(
        subject.ConfirmationTerraLifecycleError,
        match="exactly equal remaining unique",
    ):
        subject.run_provider_completion(
            **_provider_kwargs(
                source_path, source, output_root, preflight, release
            ),
            authorized_provider_calls=1,
            client_factory=factory,
        )
    assert factory.calls == []


def test_provider_factory_unreachable_without_explicit_opt_in(
    tmp_path: Path,
) -> None:
    source_path, source = _publish_source(tmp_path, ["one", "two"])
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)
    release = _approve(
        source_path,
        source.sha256,
        output_root,
        preflight.sha256,
        2,
    )
    factory = _Factory(explode=True)

    with pytest.raises(
        subject.ConfirmationTerraLifecycleError,
        match="requires explicit opt-in",
    ):
        subject.run_provider_completion(
            **{
                **_provider_kwargs(
                    source_path, source, output_root, preflight, release
                ),
                "enable_provider": False,
            },
            authorized_provider_calls=2,
            client_factory=factory,
        )
    assert factory.calls == []


def test_exact_calls_checkpoint_resume_materialize_and_replay(
    tmp_path: Path,
) -> None:
    source_path, source = _publish_source(
        tmp_path,
        ["one", "two", "one", "three"],
    )
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)
    release = _approve(
        source_path,
        source.sha256,
        output_root,
        preflight.sha256,
        3,
    )
    factory = _Factory()

    first = subject.run_provider_completion(
        **_provider_kwargs(source_path, source, output_root, preflight, release),
        authorized_provider_calls=3,
        client_factory=factory,
    )

    assert first["logical_prompt_count"] == 4
    assert first["unique_prompt_count"] == 3
    assert first["physical_provider_calls"] == 3
    assert first["checkpoint_hits_before_run"] == 0
    assert len(factory.calls) == 1
    assert len(factory.client.chat.completions.requests) == 3
    assert all(
        request["model"] == subject.TERRA_MODEL
        and request["max_tokens"] == 768
        for request in factory.client.chat.completions.requests
    )
    assert factory.client.close_calls == 1

    no_call_factory = _Factory(explode=True)
    resumed = subject.run_provider_completion(
        **_provider_kwargs(source_path, source, output_root, preflight, release),
        authorized_provider_calls=0,
        client_factory=no_call_factory,
    )
    assert resumed["physical_provider_calls"] == 0
    assert resumed["checkpoint_hits_before_run"] == 3
    assert no_call_factory.calls == []

    completion, created = subject.materialize_completions(
        prompt_artifact_path=source_path,
        expected_prompt_artifact_sha256=source.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
    )
    assert created is True
    rows = completion.payload["ordered_rows"]
    assert [row["question_id"] for row in rows] == ["q-0", "q-1", "q-2", "q-3"]
    assert rows[0]["call_key_sha256"] == rows[2]["call_key_sha256"]
    assert completion.payload["physical_provider_calls_during_materialization"] == 0

    replay, replay_created = subject.replay_completions(
        prompt_artifact_path=source_path,
        expected_prompt_artifact_sha256=source.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        expected_completion_sha256=completion.sha256,
    )
    assert replay_created is True
    assert replay.sha256 == completion.sha256
    assert replay.payload == completion.payload


def test_terminal_preflight_executes_only_provider_bound_source_rows(
    tmp_path: Path,
) -> None:
    source_path, source = _publish_terminal_source(
        tmp_path,
        ["pass-through", "terminal-one", "fallback", "terminal-two"],
        provider_indexes={1, 3},
    )
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)
    release = _approve(
        source_path,
        source.sha256,
        output_root,
        preflight.sha256,
        2,
    )
    factory = _Factory()

    result = subject.run_provider_completion(
        **_provider_kwargs(source_path, source, output_root, preflight, release),
        authorized_provider_calls=2,
        client_factory=factory,
    )
    completion, _ = subject.materialize_completions(
        prompt_artifact_path=source_path,
        expected_prompt_artifact_sha256=source.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
    )

    assert result["logical_prompt_count"] == 2
    assert result["physical_provider_calls"] == 2
    assert [row["question_id"] for row in completion.payload["ordered_rows"]] == [
        "q-1",
        "q-3",
    ]
    assert [
        row["source_prompt_row_index"]
        for row in completion.payload["ordered_rows"]
    ] == [1, 3]


def test_incomplete_checkpoint_pair_is_terminal_before_client_creation(
    tmp_path: Path,
) -> None:
    source_path, source = _publish_source(tmp_path, ["one", "two"])
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)
    release = _approve(
        source_path,
        source.sha256,
        output_root,
        preflight.sha256,
        2,
    )
    checkpoint_root = output_root / subject.CHECKPOINT_DIR_NAME
    checkpoint_root.mkdir()
    (checkpoint_root / f"{'e' * 64}.request.json").write_text(
        "{}\n", encoding="utf-8"
    )
    factory = _Factory(explode=True)

    with pytest.raises(
        subject.ConfirmationTerraLifecycleError,
        match="pair is incomplete",
    ):
        subject.run_provider_completion(
            **_provider_kwargs(
                source_path, source, output_root, preflight, release
            ),
            authorized_provider_calls=2,
            client_factory=factory,
        )
    assert factory.calls == []


def test_materialize_refuses_incomplete_population(tmp_path: Path) -> None:
    source_path, source = _publish_source(tmp_path, ["one", "two"])
    output_root = tmp_path / "run"
    preflight = _publish_preflight(source_path, source.sha256, output_root)
    release = _approve(
        source_path,
        source.sha256,
        output_root,
        preflight.sha256,
        2,
    )

    with pytest.raises(
        subject.ConfirmationTerraLifecycleError,
        match="complete checkpoint population",
    ):
        subject.materialize_completions(
            prompt_artifact_path=source_path,
            expected_prompt_artifact_sha256=source.sha256,
            output_root=output_root,
            expected_lifecycle_preflight_sha256=preflight.sha256,
            expected_release_sha256=release.sha256,
        )


def test_publication_reuses_identical_and_refuses_different_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sealed.json"
    first, created = subject.publish_sealed_artifact(path, {"value": 1})
    second, reused = subject.publish_sealed_artifact(path, {"value": 1})

    assert created is True
    assert reused is False
    assert first.sha256 == second.sha256
    with pytest.raises(subject.ConfirmationTerraLifecycleError):
        subject.publish_sealed_artifact(path, {"value": 2})


def test_source_retry_policy_must_be_zero(tmp_path: Path) -> None:
    payload = _source_payload(["one"])
    body = dict(payload)
    body.pop("preflight_identity_sha256")
    body["runtime"] = {**body["runtime"], "retry_count": 1}
    payload = {**body, "preflight_identity_sha256": canonical_sha256(body)}
    path = tmp_path / "source.json"
    artifact, _ = subject.publish_sealed_artifact(path, payload)

    with pytest.raises(
        subject.ConfirmationTerraLifecycleError,
        match="retries must equal zero",
    ):
        subject.verify_prompt_artifact(path, expected_sha256=artifact.sha256)


def test_parser_exposes_closed_lifecycle_commands() -> None:
    parser = subject.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None)
    )
    assert set(subparsers.choices) == {
        "preflight",
        "approve-release",
        "provider-run",
        "materialize",
        "replay",
    }


def test_default_client_factory_loads_dotenv_only_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dotenv
    from tools.matched_eval import live

    key_name = "CONFIRMATION_TEST_LITELLM_KEY"
    sentinel = object()
    loads: list[bool] = []
    monkeypatch.delenv(key_name, raising=False)

    def load_dotenv(*, override: bool) -> bool:
        loads.append(override)
        monkeypatch.setenv(key_name, "sealed-test-key")
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", load_dotenv)
    monkeypatch.setattr(
        live,
        "_make_provider_client",
        lambda api_key, gateway: sentinel
        if (api_key, gateway) == ("sealed-test-key", "https://gateway.test/v1")
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
