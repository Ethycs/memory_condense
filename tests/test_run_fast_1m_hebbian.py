from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval import run_fast_1m_hebbian as runner
from memory_condense.eval.consolidation_replay import RetrievalAccessEvent
from tests.test_fast_completion_runtime import _FakeClient


_D = "a" * 64


def _prompt_population():
    messages = (
        (
            {"role": "system", "content": "Answer only from evidence."},
            {"role": "user", "content": "Evidence A. Question A?"},
        ),
        (
            {"role": "system", "content": "Answer only from evidence."},
            {"role": "user", "content": "Evidence H1. Question A?"},
        ),
    )
    logical = tuple(
        SimpleNamespace(
            logical_ordinal=index,
            question_ordinal=0,
            question_id="q-1",
            stage_id=runner.S0_STAGE_ID,
            arm_id=arm,
            arm_prompt_sha256=hashlib.sha256(f"arm-{arm}".encode()).hexdigest(),
            messages_sha256=identity_sha256(list(row)),
            unique_prompt_ordinal=index,
            prompt_token_proxy=24,
            hard_prompt_token_cap=128,
            chunk_ids=(f"chunk-{arm}",),
            alias_order=(f"H{index}",),
        )
        for index, (arm, row) in enumerate(zip(runner.ARM_IDS, messages, strict=True))
    )
    receipts = (
        SimpleNamespace(
            effective_status="replaced",
            effective_h1_chunk_ids=("chunk-h1",),
            protected_chunk_ids=("chunk-base",),
        ),
    )
    identity = {
        "format": "test-hebbian-prompts-v1",
        "prompt_population_sha256": "b" * 64,
        "retained_request_token_state_bytes": 0,
    }
    return SimpleNamespace(
        stage_id=runner.S0_STAGE_ID,
        logical_prompt_count=2,
        unique_prompt_count=2,
        logical_prompts=logical,
        logical_message_population=messages,
        question_receipts=receipts,
        prompt_population_sha256="b" * 64,
        identity_payload=lambda: dict(identity),
    )


def _experiment(tmp_path: Path):
    source_receipt = SimpleNamespace(
        receipt_sha256="1" * 64,
        target_database_sha256="2" * 64,
        target_index_sha256="3" * 64,
    )
    source = SimpleNamespace(
        receipt=source_receipt,
        manifest_sha256="4" * 64,
    )
    history_receipt = SimpleNamespace(
        receipt_sha256="5" * 64,
        direct_capture_sha256="a" * 64,
        capture_policy_sha256="b" * 64,
        implementation_sha256="6" * 64,
        environment_lock_sha256="7" * 64,
    )
    history = SimpleNamespace(
        artifact_sha256="8" * 64,
        receipt=history_receipt,
        capture_policy_payload={
            "format": "memory-condense.hebbian-capture-policy.v1",
            "retrieval_k": 10,
            "expansion_tokens": 1_600,
            "max_prompt_tokens": 128,
            "direct_expansion_only": True,
            "event_id_scheme": "causal-user:{ordinal}",
            "capture_point": (
                "after_direct_context_pack_before_current_user_append"
            ),
            "exclude_current_and_future_turns": True,
            "query_embedding_model_id": runner.DEFAULT_MODEL_NAME,
            "query_embedding_model_revision": runner.DEFAULT_MODEL_REVISION,
            "query_embedding_checkpoint_sha256": (
                runner.BGE_M3_CHECKPOINT_SHA256
            ),
            "query_embedding_execution_sha256": "a" * 64,
        },
    )
    derived = SimpleNamespace(
        receipt_sha256="9" * 64,
        derived_database_sha256="1" * 64,
        derived_index_sha256="2" * 64,
        learning_policy_sha256="3" * 64,
        association_artifact_id="hebbian-test",
        association_artifact_sha256="c" * 64,
    )
    artifact = SimpleNamespace(
        raw_sha256="d" * 64,
        population_identity_sha256="e" * 64,
        source_store_receipt_sha256="4" * 64,
        retrieval_implementation_sha256="5" * 64,
        retrieval_policy_sha256="6" * 64,
        question_count=1,
        questions=(SimpleNamespace(question_id="q-1"),),
    )
    return runner._Experiment(
        artifact=artifact,
        source=source,
        history=history,
        history_file_sha256="f" * 64,
        derived=derived,
        derived_manifest_sha256="0" * 64,
        derived_store_path=tmp_path / "derived-store",
    )


def _stub_experiment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    experiment = _experiment(tmp_path)
    prompts = _prompt_population()
    monkeypatch.setattr(runner, "_load_experiment", lambda _args: experiment)
    monkeypatch.setattr(runner, "_build_prompts", lambda _experiment: prompts)
    return experiment, prompts


def _rewrite_canonical(path: Path, payload: dict) -> None:
    raw = runner._canonical_json_bytes(payload)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_name(path.name + ".sha256").write_bytes(
        f"{digest}  {path.name}\n".encode("ascii")
    )


def test_cli_defaults_are_fast_provider_free_and_pinned() -> None:
    args = runner.build_parser().parse_args([])

    assert args.phase == "preflight"
    assert args.expected_retrieval_sha256 == runner.ORIGINAL_1M_RETRIEVAL_SHA256
    assert args.max_prompt_tokens == 8_000
    assert args.max_new_tokens == 256
    assert args.history_root is None
    assert args.enable_provider is False
    assert args.authorized_provider_calls == 0
    assert args.gateway_model == "codex_sdk/gpt-5.6-terra"
    assert args.source_store.name == "combined-store"


def test_canonical_artifact_roundtrip_and_sidecar_tamper(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    digest = runner._atomic_write_json(path, {"b": 2, "a": 1})

    payload, observed = runner._read_canonical_json(path)

    assert payload == {"a": 1, "b": 2}
    assert observed == digest
    path.with_name("artifact.json.sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(ValueError, match="sidecar"):
        runner._read_canonical_json(path)


def test_preflight_reports_h1_population_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    args = runner.build_parser().parse_args(
        ["--output-root", str(tmp_path / "must-not-exist")]
    )

    result = runner.run_preflight(args)

    prompt = result["prompt_preflight"]
    assert result["writes"] == result["provider_calls"] == 0
    assert result["gold_loaded"] is False
    assert prompt["logical_prompt_count"] == prompt["unique_prompt_count"] == 2
    assert prompt["effective_statuses"] == {"replaced": 1}
    assert prompt["replacements"] == prompt["membership_changes"] == 1
    assert prompt["hard_prompt_token_cap"] == 8_000
    assert 1 <= prompt["max_observed_prompt_token_proxy"] <= 8_000
    assert not Path(args.output_root).exists()


def test_answer_exact_gate_precedes_key_client_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    monkeypatch.delenv("LITELLM_KEY", raising=False)
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: pytest.fail("provider client must not be constructed"),
    )
    args = runner.build_parser().parse_args(
        [
            "--phase",
            "answer",
            "--output-root",
            str(tmp_path / "output"),
            "--enable-provider",
            "--authorized-provider-calls",
            "1",
        ]
    )

    with pytest.raises(ValueError, match=r"1 != 2"):
        runner.run_answer(args)

    assert not Path(args.output_root).exists()


def test_invalid_history_policy_rejects_before_artifact_or_model_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_artifact",
        lambda *_args: pytest.fail("retrieval must remain unopened"),
    )
    monkeypatch.setattr(
        runner,
        "EmbeddingService",
        lambda **_kwargs: pytest.fail("embedding model must remain unloaded"),
    )
    args = runner.build_parser().parse_args(
        [
            "--phase",
            "history",
            "--output-root",
            str(tmp_path / "output"),
            "--retrieval-k",
            "65",
        ]
    )

    with pytest.raises(ValueError, match=r"retrieval-k.*\[1, 64\]"):
        runner.run_history(args)

    assert not Path(args.output_root).exists()


def test_existing_history_rejects_changed_capture_policy_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    output = tmp_path / "existing-history"
    output.mkdir()
    monkeypatch.setattr(runner, "_load_artifact", lambda *_args: experiment.artifact)
    monkeypatch.setattr(runner, "_validate_source_store", lambda *_args: experiment.source)
    monkeypatch.setattr(
        runner,
        "_load_experiment",
        lambda _args, *, history_root=None: experiment,
    )
    args = runner.build_parser().parse_args(
        [
            "--phase",
            "history",
            "--output-root",
            str(output),
            "--retrieval-k",
            "11",
        ]
    )

    with pytest.raises(ValueError, match="capture policy"):
        runner.run_history(args)


def test_answer_replay_and_score_use_immutable_journals_before_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _experiment_value, prompts = _stub_experiment(monkeypatch, tmp_path)
    output = tmp_path / "output"
    checkpoint = output / "completion-calls"
    client = _FakeClient(checkpoint, delay_s=0.0)
    monkeypatch.setenv("LITELLM_KEY", "never-persist-this-secret")
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)
    answer_args = runner.build_parser().parse_args(
        [
            "--phase",
            "answer",
            "--output-root",
            str(output),
            "--enable-provider",
            "--authorized-provider-calls",
            "2",
            "--gateway-model",
            "codex_sdk/fake-model",
            "--max-new-tokens",
            "32",
        ]
    )

    answers, _answer_digest = runner.run_answer(answer_args)

    assert answers["completion_batch"]["usage"]["physical_calls"] == 2
    assert b"never-persist-this-secret" not in (output / "answers.json").read_bytes()
    assert len(list(checkpoint.glob("*.request.json"))) == 2
    assert len(list(checkpoint.glob("*.response.json"))) == 2

    replay_args = runner.build_parser().parse_args(
        ["--phase", "replay", "--output-root", str(output)]
    )
    replay, _replay_digest = runner.run_replay(replay_args)
    assert replay["completion_batch"]["usage"]["physical_calls"] == 0
    assert replay["completion_batch"]["usage"]["checkpoint_hits"] == 2

    predictions = {
        row["arm_id"]: row["prediction"] for row in answers["answers"]
    }
    gold_calls: list[str] = []

    def load_gold(_dataset: Path, _split: Path):
        gold_calls.append("after-validation")
        # One answer must serve both arms; scoring correctness is secondary to
        # proving that gold remains unreachable until journal replay succeeds.
        return SimpleNamespace(
            questions=(
                SimpleNamespace(
                    question_id="q-1",
                    answer=predictions["base"],
                    category="test",
                ),
            )
        )

    monkeypatch.setattr(runner, "_load_gold_population", load_gold)
    score_args = runner.build_parser().parse_args(
        [
            "--phase",
            "score",
            "--output-root",
            str(output),
            "--dataset",
            str(tmp_path / "gold.json"),
        ]
    )
    scores, _score_digest = runner.run_score(score_args)

    assert gold_calls == ["after-validation"]
    assert [row["arm_id"] for row in scores["aggregates"]] == list(runner.ARM_IDS)
    assert scores["logical_score_count"] == len(prompts.logical_prompts)


@pytest.mark.parametrize("tamper_kind", ["record", "usage"])
def test_score_rejects_resealed_answer_metadata_before_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper_kind: str
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    output = tmp_path / "output"
    checkpoint = output / "completion-calls"
    monkeypatch.setenv("LITELLM_KEY", "test-only-key")
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: _FakeClient(checkpoint, delay_s=0.0),
    )
    answer_args = runner.build_parser().parse_args(
        [
            "--phase",
            "answer",
            "--output-root",
            str(output),
            "--enable-provider",
            "--authorized-provider-calls",
            "2",
            "--gateway-model",
            "codex_sdk/fake-model",
            "--max-new-tokens",
            "32",
        ]
    )
    runner.run_answer(answer_args)
    runner.run_replay(
        runner.build_parser().parse_args(
            ["--phase", "replay", "--output-root", str(output)]
        )
    )

    for name in ("answers.json", "replay.json"):
        path = output / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tamper_kind == "record":
            payload["completion_batch"]["unique_records"][0][
                "response_model"
            ] = "forged-model"
        else:
            payload["completion_batch"]["usage"]["prompt_token_proxy"] += 1
        _rewrite_canonical(path, payload)

    gold_calls: list[bool] = []
    monkeypatch.setattr(
        runner,
        "_load_gold_population",
        lambda *_args: gold_calls.append(True),
    )
    score_args = runner.build_parser().parse_args(
        [
            "--phase",
            "score",
            "--output-root",
            str(output),
            "--dataset",
            str(tmp_path / "gold.json"),
        ]
    )

    with pytest.raises(ValueError, match="immutable provider journals"):
        runner.run_score(score_args)

    assert gold_calls == []


def test_answer_can_reuse_history_root_without_coupling_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_roots: list[Path | None] = []
    experiment = _experiment(tmp_path)
    prompts = _prompt_population()

    def load_experiment(args, *, history_root=None):
        observed_roots.append(history_root or args.history_root or args.output_root)
        return experiment

    monkeypatch.setattr(runner, "_load_experiment", load_experiment)
    monkeypatch.setattr(runner, "_build_prompts", lambda _experiment: prompts)
    output = tmp_path / "answer-output"
    history = tmp_path / "reusable-history"
    args = runner.build_parser().parse_args(
        [
            "--output-root",
            str(output),
            "--history-root",
            str(history),
        ]
    )

    result = runner.run_preflight(args)

    assert observed_roots == [history]
    assert result["writes"] == 0
    assert not output.exists()


def test_score_cannot_reach_gold_after_replay_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_experiment(monkeypatch, tmp_path)
    gold_calls: list[bool] = []
    monkeypatch.setattr(
        runner,
        "_load_gold_population",
        lambda *_args: gold_calls.append(True),
    )
    monkeypatch.setattr(
        runner,
        "_read_and_validate_answers",
        lambda *_args, expected_mode, **_kwargs: (
            ({"mode": "answer"}, _D)
            if expected_mode == "answer"
            else (_ for _ in ()).throw(ValueError("tampered replay"))
        ),
    )
    args = runner.build_parser().parse_args(
        [
            "--phase",
            "score",
            "--output-root",
            str(tmp_path),
            "--dataset",
            str(tmp_path / "gold.json"),
        ]
    )

    with pytest.raises(ValueError, match="tampered replay"):
        runner.run_score(args)

    assert gold_calls == []


def test_history_batches_embeddings_once_and_publishes_root_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    database = source_root / "memory.db"
    database.write_bytes(b"immutable-source-database")
    index = source_root / "hnsw_index.bin"
    index.write_bytes(b"immutable-source-index")
    manifest = source_root / runner.COMBINED_CUMULATIVE_STORE_MANIFEST
    manifest.write_bytes(b"manifest")
    source_before = database.read_bytes(), index.read_bytes()
    source_receipt = SimpleNamespace(
        receipt_sha256="1" * 64,
        target_database_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
        target_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
    )
    source = runner._SourceBinding(
        root=source_root,
        database_path=database,
        index_path=index,
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
        receipt=source_receipt,
    )
    artifact = SimpleNamespace(raw_sha256="2" * 64)
    embed_calls: list[tuple[str, ...]] = []

    class FakeEmbeddingService:
        checkpoint_sha256 = runner.BGE_M3_CHECKPOINT_SHA256
        model_name = runner.DEFAULT_MODEL_NAME
        model_revision = runner.DEFAULT_MODEL_REVISION
        execution_identity = {
            "backend": "sentence-transformers.encode-v1",
            "device": "cpu",
            "batch_size": 4,
            "normalize_embeddings": False,
            "output_dtype": "float32",
        }

        def __init__(self, **_kwargs):
            pass

        def embed_queries(self, queries):
            embed_calls.append(tuple(queries))
            return np.asarray([[1.0, 0.0] for _query in queries], dtype=np.float32)

        def close(self):
            pass

    history_receipt = SimpleNamespace(
        source_database_sha256=source_receipt.target_database_sha256,
        receipt_sha256="3" * 64,
        direct_capture_sha256="8" * 64,
        event_count=1,
        empty_event_count=0,
    )
    history = SimpleNamespace(
        receipt=history_receipt,
        artifact_sha256="4" * 64,
        payload=lambda: {"format": "fake-history", "artifact_sha256": "4" * 64},
    )
    derived = SimpleNamespace(
        source_index_sha256=source_receipt.target_index_sha256,
        receipt_sha256="5" * 64,
        association_artifact_id="hebbian-test",
    )

    monkeypatch.setattr(runner, "_load_artifact", lambda *_args: artifact)
    monkeypatch.setattr(runner, "_validate_source_store", lambda *_args: source)
    monkeypatch.setattr(
        runner, "_eligible_historical_queries", lambda *_args, **_kwargs: ("q1", "q2")
    )
    monkeypatch.setattr(runner, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(runner, "DEFAULT_MODEL_DIM", 2)
    monkeypatch.setattr(runner, "implementation_sha256", lambda: "6" * 64)
    monkeypatch.setattr(runner, "environment_lock_sha256", lambda: "7" * 64)

    def stage(snapshot, derived_store, _frozen, **kwargs):
        from memory_condense.eval.consolidation_replay import (
            _mint_retrieval_access_capture,
        )

        assert Path(snapshot).read_bytes() == source_before[0]
        Path(derived_store).mkdir()
        (Path(derived_store) / "memory.db").write_bytes(b"derived")
        (Path(derived_store) / "hnsw_index.bin").write_bytes(b"index")
        event = RetrievalAccessEvent(
            event_id="causal-user:2", now_turn=1, chunk_ids=("c",)
        )
        kwargs["retrieval_access_capture_sink"]._capture = (
            _mint_retrieval_access_capture(
                source_database_sha256=hashlib.sha256(
                    Path(snapshot).read_bytes()
                ).hexdigest(),
                capture_policy_sha256=kwargs[
                    "retrieval_access_capture_policy_sha256"
                ],
                retrieval_k=kwargs["retrieval_k"],
                expansion_tokens=kwargs["expansion_tokens"],
                max_prompt_tokens=kwargs["max_prompt_tokens"],
                events=(event,),
            )
        )
        return [], {"events": 1}

    monkeypatch.setattr(runner, "stage_causal_store", stage)
    def seal(capture, **kwargs):
        assert capture.capture_policy_sha256 == identity_sha256(
            dict(kwargs["capture_policy"])
        )
        assert capture.source_database_sha256 == source_receipt.target_database_sha256
        return history

    monkeypatch.setattr(runner, "_seal_staged_history", seal)

    def apply(store, **_kwargs):
        (Path(store) / runner.DERIVED_MANIFEST_NAME).write_bytes(b"derived-manifest")
        return derived

    monkeypatch.setattr(runner, "apply_hebbian_history_to_staged_store", apply)
    monkeypatch.setattr(
        runner,
        "verify_hebbian_history_artifact",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "verify_hebbian_derived_store",
        lambda _path, expected: expected,
    )
    output = tmp_path / "published"
    args = runner.build_parser().parse_args(
        [
            "--phase",
            "history",
            "--source-store",
            str(source_root),
            "--output-root",
            str(output),
            "--embedding-device",
            "cpu",
            "--embedding-batch-size",
            "4",
        ]
    )

    result = runner.run_history(args)

    assert embed_calls == [("q1", "q2")]
    assert result["embedding_api_calls"] == 1
    assert result["embedding_forward_batches"] == 1
    assert result["provider_calls"] == 0 and result["gold_loaded"] is False
    assert (database.read_bytes(), index.read_bytes()) == source_before
    assert (output / "history.json").is_file()
    assert (output / "history.json.sha256").is_file()
    assert (output / "derived-store" / runner.DERIVED_MANIFEST_NAME).is_file()
    assert not (output / "source-memory.db").exists()
