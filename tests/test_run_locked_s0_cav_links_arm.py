from __future__ import annotations

import hashlib
import json
import re
import threading
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.search.fusion.fixed_cav_router import (
    FixedCAVForward,
    FixedCAVRuntimeReceipt,
)
from tests.test_run_locked_retrieval_mechanism_arm import _sources
from tools import run_locked_retrieval_mechanism_arm as s0_runner
from tools import load_locked_s0_cav_links_arm as cav_loader
from tools import run_locked_s0_cav_links_arm as runner


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _s0_run(plan: Any) -> dict[str, Any]:
    questions = []
    for ordinal, row in enumerate(plan.rows):
        prediction = f"baseline-{ordinal}"
        questions.append(
            {
                "ordinal": ordinal,
                "question_id": row.question_id,
                "question_sha256": row.question_sha256,
                "dated_question_sha256": row.dated_question_sha256,
                "retrieval_question_part_sha256": (
                    row.retrieval_question_part_sha256
                ),
                "source_stage_id": row.source_stage_id,
                "stage_receipt_sha256": row.stage_receipt_sha256,
                "evidence_projection_sha256": row.evidence_projection_sha256,
                "provider_messages_sha256": row.provider_messages_sha256,
                "prompt_token_proxy": row.prompt_token_proxy,
                "source_binding_sha256": row.binding_sha256,
                "prediction": {
                    "text": prediction,
                    "sha256": quote_sha256(prediction),
                },
            }
        )
    return {
        "format": runner.RUN_FORMAT,
        "arm_label": runner.PARENT_ARM_LABEL,
        "retrieval_sha256": plan.population.retrieval_sha256,
        "baseline_final_answers_sha256": (
            plan.population.baseline_final_answers_sha256
        ),
        "population_identity_sha256": plan.population.population_identity_sha256,
        "historical_validator_binding_sha256": plan.population.binding_sha256,
        "questions": questions,
    }


def _args(
    root: Path,
    phase: str,
    *,
    feature_model: bool = False,
    feature_calls: int = 0,
    provider: bool = False,
    provider_calls: int = 0,
) -> Any:
    return SimpleNamespace(
        phase=phase,
        retrieval=root / "unused-retrieval.json",
        expected_retrieval_sha256=runner.EXPECTED_RETRIEVAL_SHA256,
        baseline_answers=root / "unused-baseline.json",
        expected_baseline_answers_sha256=(
            runner.EXPECTED_BASELINE_ANSWERS_SHA256
        ),
        s0_run=root / "unused-s0.json",
        s0_checkpoint_dir=None,
        expected_s0_run_sha256="9" * 64,
        output_root=root,
        features=None,
        expected_features_sha256=None,
        model_dir=root / "unused-qwen",
        device="cuda",
        dtype="bfloat16",
        batch_size=8,
        event_cav=Path(runner.DEFAULT_EVENT_CAV),
        prefix_cav=Path(runner.DEFAULT_PREFIX_CAV),
        extraction_temperature=0.05,
        reinjection_temperature=0.05,
        alpha=1.0,
        expected_question_count=2,
        gateway_url=runner.DEFAULT_GATEWAY_URL,
        model=runner.DEFAULT_MODEL,
        api_key_env="TEST_CAV_KEY",
        max_concurrency=2,
        enable_feature_model=feature_model,
        authorized_feature_encoder_calls=feature_calls,
        enable_provider=provider,
        authorized_provider_calls=provider_calls,
    )


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> runner._Source:
    population, retrieval = _sources()
    monkeypatch.setattr(
        s0_runner,
        "_validated_sources",
        lambda *_args, **_kwargs: (population, retrieval),
    )
    plan = s0_runner._prepare(
        retrieval_path=tmp_path / "unused-retrieval.json",
        baseline_answers_path=tmp_path / "unused-baseline.json",
        expected_question_count=2,
    )
    return runner._source_from_verified(
        _args(tmp_path, "preflight"),
        plan,
        _s0_run(plan),
        "9" * 64,
    )


class _Encoder:
    feature_backend_identity_sha256 = _digest("feature-backend")
    dtype_name = "bfloat16"
    device = "cuda"
    layers = 1

    def __init__(self, checkpoint_sha256: str) -> None:
        self.checkpoint_sha256 = checkpoint_sha256
        self.calls: list[tuple[tuple[str, ...], tuple[int, ...], int]] = []

    def encode_layers(
        self,
        texts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        batch_size: int,
    ) -> dict[int, Any]:
        self.calls.append((tuple(texts), tuple(layers), batch_size))
        values = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values.append([0.25 + digest[index] / 255.0 for index in range(4)])
        return {layers[0]: torch.tensor(values, dtype=torch.float32)}


class _Router:
    max_atoms = 64

    def __init__(self, source: runner._Source) -> None:
        selections = source.feature_plan["router"]["selections"]
        artifacts = tuple(row["artifact_sha256"] for row in selections)
        keys = tuple(row["tensor_key"] for row in selections)
        bank_sha = identity_sha256(
            {
                "format": "memory-condense-fixed-cav-source-bank-v1",
                "artifact_file_sha256s": list(artifacts),
                "ordered_tensor_keys": list(keys),
                "layer": runner.FEATURE_LAYER,
                "num_cavs": runner.CONCEPT_COUNT,
                "hidden_dim": 4,
                "artifact_dtype": "torch.float32",
            }
        )
        self.runtime_receipt = FixedCAVRuntimeReceipt(
            artifact_file_sha256s=artifacts,
            ordered_tensor_keys=keys,
            layer=runner.FEATURE_LAYER,
            num_cavs=runner.CONCEPT_COUNT,
            hidden_dim=4,
            artifact_dtype="torch.float32",
            execution_dtype="torch.float32",
            device="cpu",
            extraction_temperature=0.05,
            reinjection_temperature=0.05,
            alpha=1.0,
            bank_identity_sha256=bank_sha,
            normalized_cav_bank_sha256=_digest("normalized-bank"),
        )
        self.layer = self.runtime_receipt.layer
        self.hidden_dim = self.runtime_receipt.hidden_dim
        self.num_cavs = self.runtime_receipt.num_cavs
        self.runtime_identity_sha256 = self.runtime_receipt.runtime_sha256
        self.bank_identity_sha256 = self.runtime_receipt.bank_identity_sha256
        self.calls = 0

    def route_one(self, node_features: Any) -> FixedCAVForward:
        self.calls += 1
        node_count = int(node_features.shape[0])
        return FixedCAVForward(
            steered_nodes=(node_features + 0.125).detach(),
            extraction_attention=torch.full(
                (self.num_cavs, node_count),
                1.0 / node_count,
                dtype=node_features.dtype,
            ),
            reinjection_attention=torch.full(
                (node_count, self.num_cavs),
                1.0 / self.num_cavs,
                dtype=node_features.dtype,
            ),
        )


class _Client:
    def __init__(self) -> None:
        self.max_retries = 0
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.closed = False
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._complete)
        )

    def _complete(self, **request: Any) -> Any:
        with self.lock:
            self.requests.append(dict(request))
        user = str(request["messages"][-1]["content"])
        match = re.search(r"Fact [0-9]+", user)
        assert match is not None
        fact = match.group(0)
        completion = (
            '{"answer":"'
            + fact
            + '","citations":[{"evidence_alias":"E001","quote":"'
            + fact
            + '"}]}'
        )
        return SimpleNamespace(
            id="fake-response",
            model=request["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )

    def close(self) -> None:
        self.closed = True


def _contains_tensor(value: object) -> bool:
    if type(value) is torch.Tensor:
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_tensor(getattr(value, field.name)) for field in fields(value)
        )
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _publish_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[runner._Source, dict[str, Any], str, _Encoder, _Router]:
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_load_source", lambda _args: source)
    encoder = _Encoder(
        source.feature_plan["encoder"]["expected_prefix_checkpoint_sha256"]
    )
    router = _Router(source)
    monkeypatch.setattr(
        runner,
        "_load_feature_runtime",
        lambda _args: (encoder, router),
    )
    feature_args = _args(
        tmp_path,
        "features",
        feature_model=True,
        feature_calls=1,
    )
    manifest, digest = runner.run_features(feature_args)
    return source, manifest, digest, encoder, router


def test_provider_free_preflight_seals_exact_s0_and_feature_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_load_source", lambda _args: source)
    monkeypatch.setattr(
        runner,
        "_load_feature_runtime",
        lambda _args: pytest.fail("preflight loaded the feature model"),
    )
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: pytest.fail("preflight constructed a provider client"),
    )

    result = runner.run_preflight(_args(tmp_path, "preflight"))

    assert result["question_count"] == 2
    assert result["feature_model_loads"] == 0
    assert result["feature_encoder_calls"] == 0
    assert result["provider_calls"] == 0
    assert result["writes"] == 0
    assert result["concept_count"] == 3
    assert result["top_extraction_links_per_concept"] == 4
    assert result["evidence_additions"] == 0
    command = result["feature_plan"]["exact_feature_command_argv"]
    assert command[-3:] == [
        "--enable-feature-model",
        "--authorized-feature-encoder-calls",
        "1",
    ]
    assert command[0:5] == ["pixi", "run", "-e", "dev", "python"]
    assert not (tmp_path / "features.json").exists()
    assert not (tmp_path / "run.json").exists()


def test_chunked_encoder_preserves_global_order_and_seals_every_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, monkeypatch)
    plan = deepcopy(source.feature_plan)
    texts = tuple(
        sorted(
            {
                row.evidence_text
                for question in source.artifact.questions
                for row in question.feature_rows
            }
            | {question.question for question in source.artifact.questions},
            key=lambda item: (len(item), item),
        )
    )
    plan["encoder"]["execution_chunk_row_cap"] = 2
    plan["encoder"]["execution_chunk_count"] = 2
    plan["encoder"]["qwen_encode_layers_call_count"] = 2
    plan["encoder"]["transformer_forward_batch_count"] = 2
    encoder = _Encoder(
        source.feature_plan["encoder"]["expected_prefix_checkpoint_sha256"]
    )
    wrapped = runner._ChunkedFeatureEncoder(encoder, plan)

    encoded = wrapped.encode_layers(texts, layers=(0,), batch_size=8)

    assert tuple(encoded) == (0,)
    assert tuple(encoded[0].shape) == (4, 4)
    assert [call[0] for call in encoder.calls] == [texts[:2], texts[2:]]
    receipt = wrapped.execution_receipt
    assert receipt is not None
    assert receipt["qwen_model_load_count"] == 1
    assert receipt["qwen_encode_layers_call_count"] == 2
    assert receipt["rows_truncated"] == 0
    assert [row["row_count"] for row in receipt["chunks"]] == [2, 2]
    assert not _contains_tensor(receipt)


def test_feature_answer_and_replays_are_bound_and_tensor_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, features, feature_sha, encoder, router = _publish_features(
        tmp_path,
        monkeypatch,
    )
    assert features["feature_model_loads"] == 1
    assert features["feature_encoder_calls"] == 1
    assert features["provider_calls"] == 0
    assert len(encoder.calls) == 1
    encoded, layers, batch_size = encoder.calls[0]
    assert encoded == tuple(sorted(encoded, key=lambda item: (len(item), item)))
    assert layers == (0,)
    assert batch_size == 8
    assert router.calls == 2
    assert not _contains_tensor(features)
    assert features["mechanism"]["x_x1_ordering_proxy_consumed"] is False

    replay_args = _args(tmp_path, "feature-replay")
    replay_args.expected_features_sha256 = feature_sha
    replay, replay_sha = runner.run_feature_replay(replay_args)
    assert replay == features
    assert replay_sha == feature_sha

    preflight = runner.run_answer_preflight(replay_args)
    assert preflight["valid_link_guide_count"] == 2
    assert preflight["s0_fallback_count"] == 0
    assert preflight["required_authorized_provider_calls"] == 2
    assert preflight["guide_tokens"]["maximum"] <= runner.MAX_GUIDE_TOKENS

    made_client = False

    def forbidden_client(*_args: Any) -> Any:
        nonlocal made_client
        made_client = True
        raise AssertionError("authorization must precede client construction")

    monkeypatch.setattr(runner, "_make_provider_client", forbidden_client)
    rejected = _args(tmp_path, "run", provider=True, provider_calls=1)
    rejected.expected_features_sha256 = feature_sha
    with pytest.raises(ValueError, match="dependent answer population"):
        runner.run_treatment(rejected)
    assert made_client is False

    client = _Client()
    monkeypatch.setenv("TEST_CAV_KEY", "test-key")
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: client,
    )
    run_args = _args(tmp_path, "run", provider=True, provider_calls=2)
    run_args.expected_features_sha256 = feature_sha
    artifact, run_sha = runner.run_treatment(run_args)
    assert len(client.requests) == 2
    assert client.closed
    assert artifact["required_answer_calls"] == 2
    assert artifact["budget"]["exact_s0_membership_and_order"] is True
    assert artifact["budget"]["evidence_additions"] == 0
    assert all(row["prediction_kind"].startswith("terra_") for row in artifact["questions"])
    assert all(row["x_x1_ordering_proxy_consumed"] is False for row in artifact["questions"])
    assert all(row["s0_evidence_row_count"] == 1 for row in artifact["questions"])

    ledger = artifact["structural_target_ledger"]
    assert ledger["format"] == "memory-condense-structural-target-ledger-v1"
    for row in ledger["questions"]:
        assert row["evidence_targets"][0]["discovering_method"] == (
            "causal_graph_coverage_predecessor"
        )
        assert row["candidate_relation_target_count"] == 4
        assert row["admitted_relation_target_count"] == 4
        assert {
            target["discovering_method"]
            for target in row["candidate_relation_targets_before_budget"]
        } == {"genuine_cav_v2_two_pass"}
    assert "primary_owner" not in json.dumps(ledger)

    final_replay_args = _args(tmp_path, "replay")
    final_replay_args.expected_features_sha256 = feature_sha
    final, final_sha = runner.run_replay(final_replay_args)
    assert final == artifact
    assert final_sha == run_sha
    replay_path = tmp_path / "run-replay.json"
    assert replay_path.is_file()
    assert json.loads(replay_path.read_text(encoding="utf-8")) == artifact
    assert replay_path.with_name("run-replay.json.sha256").read_text(
        encoding="ascii"
    ) == f"{run_sha}  run-replay.json\n"
    verified, verified_sha = cav_loader.load_verified_run(
        tmp_path / "run.json",
        expected_run_sha256=run_sha,
        retrieval_path=tmp_path / "unused-retrieval.json",
        baseline_answers_path=tmp_path / "unused-baseline.json",
        max_concurrency=2,
        expected_question_count=2,
    )
    assert verified == artifact
    assert verified_sha == run_sha


def test_guide_overflow_falls_back_to_exact_s0_without_answer_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _features, feature_sha, _encoder, _router = _publish_features(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(runner, "MAX_GUIDE_TOKENS", 0)
    monkeypatch.setattr(
        runner,
        "_make_provider_client",
        lambda *_args: pytest.fail("fallback-only run constructed a client"),
    )
    args = _args(tmp_path, "run", provider=True, provider_calls=0)
    args.expected_features_sha256 = feature_sha

    artifact, _digest_value = runner.run_treatment(args)

    assert artifact["required_answer_calls"] == 0
    assert artifact["answer_completion_batch"] is None
    assert all(
        row["prediction_kind"] == "sealed_s0_control_fallback"
        and row["s0_fallback_reason"] == "guide_overflow"
        and row["prediction"]["text"] == f"baseline-{row['ordinal']}"
        for row in artifact["questions"]
    )
    assert all(
        row["candidate_relation_target_count"] == 4
        and row["admitted_relation_target_count"] == 0
        for row in artifact["structural_target_ledger"]["questions"]
    )
