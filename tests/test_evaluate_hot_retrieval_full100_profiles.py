from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import evaluate_hot_retrieval_full100 as evaluator


EXPECTED_SELECTION_SHA256 = "7" * 64


class _ConstructorReached(RuntimeError):
    pass


def _selection() -> dict[str, object]:
    return {
        "questions": [{"ordinal": ordinal} for ordinal in range(100)],
        "gold_fields_present": False,
        "provider_calls": 0,
    }


def _provider_selection() -> dict[str, object]:
    questions: list[dict[str, object]] = []
    for ordinal in range(100):
        questions.append(
            {
                "ordinal": ordinal,
                "shard_offset": ordinal - ordinal % 10,
                "question_id": f"question-{ordinal}",
                "arms": {
                    "a3_protected_union": {
                        "provider_messages": [
                            {"role": "user", "content": f"question {ordinal}"}
                        ],
                        "provider_payload_sha256": f"{ordinal:064x}",
                    }
                },
            }
        )
    return {
        "questions": questions,
        "gold_fields_present": False,
        "provider_calls": 0,
    }


def _answer_artifact(selection: dict[str, object]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for selected in selection["questions"]:  # type: ignore[index]
        prediction = f"prediction {selected['ordinal']}"
        rows.append(
            {
                "ordinal": selected["ordinal"],
                "shard_offset": selected["shard_offset"],
                "question_id": selected["question_id"],
                "provider_payload_sha256": selected["arms"][  # type: ignore[index]
                    "a3_protected_union"
                ]["provider_payload_sha256"],
                "prediction": prediction,
                "prediction_sha256": evaluator.quote_sha256(prediction),
                "messages_sha256": "1" * 64,
                "call_key_sha256": "2" * 64,
                "request_journal_sha256": "3" * 64,
                "response_journal_sha256": "4" * 64,
            }
        )
    return {
        "format": evaluator.ANSWER_FORMAT,
        "status": "sealed_terra_predictions_without_gold",
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "population_identity_sha256": evaluator.assay.EXPECTED_POPULATION_SHA256,
        "gold_fields_present": False,
        "retries": 0,
        "max_concurrency": 4,
        "gateway_url": "http://local.invalid",
        "questions": rows,
    }


def _install_constructor_spy(
    monkeypatch: pytest.MonkeyPatch,
    observed: dict[str, object],
) -> None:
    class RuntimeSpy:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)
            raise _ConstructorReached

    monkeypatch.setattr(evaluator, "FastCompletionRuntime", RuntimeSpy)


@pytest.mark.parametrize(
    ("profile", "module_name"),
    (
        ("adaptive-v7", "tools.assay_hot_retrieval_adaptive_full100"),
        (
            "source-seed-hybrid-v3",
            "tools.assay_hot_retrieval_source_seed_hybrid_full100",
        ),
        (
            "user-envelope-shadow-v1",
            "tools.assay_hot_v4_user_envelope_provider_selection",
        ),
        (
            "operation-aware-v1",
            "tools.assay_hot_v5_operation_aware_provider_selection",
        ),
        (
            "user-spine-v1",
            "tools.assay_hot_v5_user_spine_provider_selection",
        ),
        (
            evaluator.SPINE_EPISODIC_FACT_PROFILE,
            "tools.assay_hot_v6_spine_episode_fact_ledger_full100",
        ),
        (
            evaluator.SPINE_EPISODIC_FACT_RESERVED_PROFILE,
            "tools.assay_hot_v7_spine_episode_fact_reserved_full100",
        ),
    ),
)
def test_optional_profile_loader_is_resolved_lazily(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    module_name: str,
) -> None:
    """Importing the evaluator must not require optional assay modules."""

    synthetic_module = ModuleType(module_name)
    synthetic_module._load_selection = lambda _root: (  # type: ignore[attr-defined]
        _selection(),
        EXPECTED_SELECTION_SHA256,
    )
    monkeypatch.setitem(sys.modules, module_name, synthetic_module)

    assert evaluator._selection_assay(profile) is synthetic_module  # noqa: SLF001
    assert evaluator._selection_assay("frozen-v6") is evaluator.assay  # noqa: SLF001

    with pytest.raises(ValueError, match="unknown selection profile"):
        evaluator._selection_assay("future-v8")  # noqa: SLF001


def test_load_selection_dispatches_to_adaptive_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    observed: dict[str, object] = {}

    def select_loader(profile: str) -> object:
        observed["profile"] = profile

        def load(root: Path) -> tuple[dict[str, object], str]:
            observed["root"] = root
            return selection, EXPECTED_SELECTION_SHA256

        return SimpleNamespace(_load_selection=load)

    monkeypatch.setattr(evaluator, "_selection_assay", select_loader)

    loaded, digest = evaluator._load_selection(  # noqa: SLF001
        tmp_path,
        EXPECTED_SELECTION_SHA256,
        selection_profile="adaptive-v7",
    )

    assert loaded is selection
    assert digest == EXPECTED_SELECTION_SHA256
    assert observed == {"profile": "adaptive-v7", "root": tmp_path}


def test_load_selection_prefers_public_loader_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    observed: dict[str, object] = {}

    def load(root: Path) -> tuple[dict[str, object], str]:
        observed["root"] = root
        return selection, EXPECTED_SELECTION_SHA256

    def reject_private_loader(_root: Path) -> None:
        raise AssertionError("private loader should not be used")

    monkeypatch.setattr(
        evaluator,
        "_selection_assay",
        lambda _profile: SimpleNamespace(
            load_selection=load,
            _load_selection=reject_private_loader,
        ),
    )

    loaded, digest = evaluator._load_selection(  # noqa: SLF001
        tmp_path,
        EXPECTED_SELECTION_SHA256,
        selection_profile=evaluator.SPINE_EPISODIC_FACT_PROFILE,
    )

    assert loaded is selection
    assert digest == EXPECTED_SELECTION_SHA256
    assert observed == {"root": tmp_path}


def test_reserved_profile_loader_keeps_caller_sha_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    observed: dict[str, object] = {}

    def load(root: Path) -> tuple[dict[str, object], str]:
        observed["root"] = root
        return selection, "6" * 64

    def select_loader(profile: str) -> object:
        observed["profile"] = profile
        return SimpleNamespace(load_selection=load)

    monkeypatch.setattr(evaluator, "_selection_assay", select_loader)

    with pytest.raises(ValueError, match="selection digest changed"):
        evaluator._load_selection(  # noqa: SLF001
            tmp_path,
            EXPECTED_SELECTION_SHA256,
            selection_profile=evaluator.SPINE_EPISODIC_FACT_RESERVED_PROFILE,
        )

    assert observed == {
        "profile": evaluator.SPINE_EPISODIC_FACT_RESERVED_PROFILE,
        "root": tmp_path,
    }


def test_load_selection_dispatches_to_materialized_hybrid_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = _selection()
    messages = [{"role": "user", "content": "sealed hybrid evidence"}]
    for row in materialized["questions"]:  # type: ignore[index]
        row["arms"] = {  # type: ignore[index]
            "a3_protected_union": {"provider_messages": messages}
        }
    observed: dict[str, object] = {}

    def select_loader(profile: str) -> object:
        observed["profile"] = profile

        def load(root: Path) -> tuple[dict[str, object], str]:
            observed["root"] = root
            return materialized, EXPECTED_SELECTION_SHA256

        return SimpleNamespace(_load_selection=load)

    monkeypatch.setattr(evaluator, "_selection_assay", select_loader)

    loaded, digest = evaluator._load_selection(  # noqa: SLF001
        tmp_path,
        EXPECTED_SELECTION_SHA256,
        selection_profile="source-seed-hybrid-v3",
    )

    assert loaded is materialized
    assert digest == EXPECTED_SELECTION_SHA256
    assert loaded["questions"][0]["arms"]["a3_protected_union"][  # type: ignore[index]
        "provider_messages"
    ] == messages
    assert observed == {
        "profile": "source-seed-hybrid-v3",
        "root": tmp_path,
    }


@pytest.mark.parametrize(
    ("command", "extra_arguments"),
    (("answer", ()), ("judge", ("--dataset", "gold.json"))),
)
@pytest.mark.parametrize(
    "selection_profile",
    (
        "adaptive-v7",
        "source-seed-hybrid-v3",
        "user-envelope-shadow-v1",
        "operation-aware-v1",
        "user-spine-v1",
        evaluator.SPINE_EPISODIC_FACT_PROFILE,
        evaluator.SPINE_EPISODIC_FACT_RESERVED_PROFILE,
    ),
)
def test_cli_forwards_profile_to_each_provider_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    extra_arguments: tuple[str, ...],
    selection_profile: str,
) -> None:
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> str:
        observed.update(kwargs)
        return "result"

    monkeypatch.setattr(evaluator, command, capture)
    arguments = [
        "--output-root",
        str(tmp_path),
        "--expected-selection-sha256",
        EXPECTED_SELECTION_SHA256,
        "--authorized-provider-calls",
        "0",
        "--selection-profile",
        selection_profile,
        command,
        *extra_arguments,
    ]

    assert evaluator.main(arguments) == 0
    assert observed["selection_profile"] == selection_profile
    assert observed["expected_selection_sha256"] == EXPECTED_SELECTION_SHA256
    assert observed["output_root"] == tmp_path.resolve()


def test_cli_defaults_to_frozen_v6_profile(tmp_path: Path) -> None:
    args = evaluator._parser().parse_args(  # noqa: SLF001
        [
            "--output-root",
            str(tmp_path),
            "--expected-selection-sha256",
            EXPECTED_SELECTION_SHA256,
            "--authorized-provider-calls",
            "0",
            "answer",
        ]
    )

    assert args.selection_profile == "frozen-v6"


def test_reserved_profile_cli_still_requires_expected_selection_sha(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        evaluator._parser().parse_args(  # noqa: SLF001
            [
                "--output-root",
                str(tmp_path),
                "--authorized-provider-calls",
                "0",
                "--selection-profile",
                evaluator.SPINE_EPISODIC_FACT_RESERVED_PROFILE,
                "answer",
            ]
        )


def test_cli_forwards_explicit_sealed_r3_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> str:
        observed.update(kwargs)
        return "result"

    monkeypatch.setattr(evaluator, "answer", capture)
    assert (
        evaluator.main(
            [
                "--output-root",
                str(evaluator.SPINE_EPISODIC_FACT_R3_OUTPUT_ROOT),
                "--expected-selection-sha256",
                evaluator.SPINE_EPISODIC_FACT_R3_SELECTION_SHA256,
                "--authorized-provider-calls",
                "0",
                "--selection-profile",
                evaluator.SPINE_EPISODIC_FACT_PROFILE,
                "answer",
            ]
        )
        == 0
    )

    assert observed["selection_profile"] == evaluator.SPINE_EPISODIC_FACT_PROFILE
    assert (
        observed["expected_selection_sha256"]
        == evaluator.SPINE_EPISODIC_FACT_R3_SELECTION_SHA256
    )
    assert observed["output_root"] == (
        Path.cwd() / evaluator.SPINE_EPISODIC_FACT_R3_OUTPUT_ROOT
    ).resolve()


@pytest.mark.parametrize(
    ("selection_profile", "expected_cap"),
    (
        ("frozen-v6", evaluator.ANSWER_MAX_PROMPT_TOKENS),
        (
            evaluator.SPINE_EPISODIC_FACT_PROFILE,
            evaluator.SPINE_EPISODIC_FACT_ANSWER_MAX_PROMPT_TOKENS,
        ),
        (
            evaluator.SPINE_EPISODIC_FACT_RESERVED_PROFILE,
            evaluator.SPINE_EPISODIC_FACT_RESERVED_ANSWER_MAX_PROMPT_TOKENS,
        ),
    ),
)
def test_live_answer_runtime_uses_profile_specific_terra_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_profile: str,
    expected_cap: int,
) -> None:
    observed: dict[str, object] = {}
    selection = _provider_selection()
    monkeypatch.setattr(
        evaluator,
        "_load_selection",
        lambda *_args, **_kwargs: (selection, EXPECTED_SELECTION_SHA256),
    )
    monkeypatch.setattr(
        evaluator,
        "_completion_client",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    _install_constructor_spy(monkeypatch, observed)

    with pytest.raises(_ConstructorReached):
        evaluator.answer(
            output_root=tmp_path,
            expected_selection_sha256=EXPECTED_SELECTION_SHA256,
            gateway_url="http://local.invalid",
            api_key_env="UNUSED_TEST_KEY",
            dotenv_path=None,
            max_concurrency=1,
            authorized_provider_calls=0,
            selection_profile=selection_profile,
        )

    assert observed["model"] == evaluator.TERRA_MODEL
    assert observed["max_prompt_tokens"] == expected_cap


@pytest.mark.parametrize(
    ("selection_profile", "expected_cap"),
    (
        (
            evaluator.SPINE_EPISODIC_FACT_PROFILE,
            evaluator.SPINE_EPISODIC_FACT_ANSWER_MAX_PROMPT_TOKENS,
        ),
        (
            evaluator.SPINE_EPISODIC_FACT_RESERVED_PROFILE,
            evaluator.SPINE_EPISODIC_FACT_RESERVED_ANSWER_MAX_PROMPT_TOKENS,
        ),
    ),
)
def test_answer_checkpoint_replay_uses_profile_specific_terra_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_profile: str,
    expected_cap: int,
) -> None:
    observed: dict[str, object] = {}
    selection = _provider_selection()
    artifact = _answer_artifact(selection)
    monkeypatch.setattr(
        evaluator.assay.hot,
        "_read_json_artifact",
        lambda _path: (artifact, "8" * 64),
    )
    _install_constructor_spy(monkeypatch, observed)

    with pytest.raises(_ConstructorReached):
        evaluator._load_answers(  # noqa: SLF001
            tmp_path,
            EXPECTED_SELECTION_SHA256,
            selection=selection,
            selection_profile=selection_profile,
        )

    assert observed["model"] == evaluator.TERRA_MODEL
    assert observed["max_prompt_tokens"] == expected_cap


def test_existing_answer_path_forwards_selection_profile_to_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / evaluator.ANSWERS_NAME).touch()
    selection = _provider_selection()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        evaluator,
        "_load_selection",
        lambda *_args, **_kwargs: (selection, EXPECTED_SELECTION_SHA256),
    )

    def load_answers(*_args: object, **kwargs: object) -> tuple[dict[str, object], str]:
        observed.update(kwargs)
        return {}, "8" * 64

    monkeypatch.setattr(evaluator, "_load_answers", load_answers)
    evaluator.answer(
        output_root=tmp_path,
        expected_selection_sha256=EXPECTED_SELECTION_SHA256,
        gateway_url="http://local.invalid",
        api_key_env="UNUSED_TEST_KEY",
        dotenv_path=None,
        max_concurrency=1,
        authorized_provider_calls=0,
        selection_profile=evaluator.SPINE_EPISODIC_FACT_PROFILE,
    )

    assert observed["selection_profile"] == evaluator.SPINE_EPISODIC_FACT_PROFILE


@pytest.mark.parametrize("existing_judgment", (False, True))
def test_both_judge_paths_forward_selection_profile_to_answer_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_judgment: bool,
) -> None:
    if existing_judgment:
        (tmp_path / evaluator.JUDGMENTS_NAME).touch()
    selection = _provider_selection()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        evaluator,
        "_load_selection",
        lambda *_args, **_kwargs: (selection, EXPECTED_SELECTION_SHA256),
    )

    def stop_at_answers(*_args: object, **kwargs: object) -> None:
        observed.update(kwargs)
        raise _ConstructorReached

    monkeypatch.setattr(evaluator, "_load_answers", stop_at_answers)
    with pytest.raises(_ConstructorReached):
        evaluator.judge(
            output_root=tmp_path,
            expected_selection_sha256=EXPECTED_SELECTION_SHA256,
            dataset=tmp_path / "gold.json",
            split_manifest=tmp_path / "split.json",
            gateway_url="http://local.invalid",
            api_key_env="UNUSED_TEST_KEY",
            dotenv_path=None,
            max_concurrency=1,
            authorized_provider_calls=0,
            selection_profile=evaluator.SPINE_EPISODIC_FACT_PROFILE,
        )

    assert observed["selection_profile"] == evaluator.SPINE_EPISODIC_FACT_PROFILE


def test_sol_checkpoint_replay_keeps_4096_prompt_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    _install_constructor_spy(monkeypatch, observed)

    with pytest.raises(_ConstructorReached):
        evaluator._verify_judgment_journals(  # noqa: SLF001
            output_root=tmp_path,
            body={
                "rows": [],
                "max_concurrency": 1,
                "gateway_url": "http://local.invalid",
            },
            selection_sha256=EXPECTED_SELECTION_SHA256,
            answers={"questions": []},
            answers_sha256="8" * 64,
            questions=[],
        )

    assert observed["model"] == evaluator.SOL_MODEL
    assert observed["max_prompt_tokens"] == 4_096
