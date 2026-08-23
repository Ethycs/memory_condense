from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from memory_condense.domain.discourse import (
    canonical_json,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.fast_cav_prompts import (
    build_fast_cav_prompt_population,
)
from memory_condense.eval.fast_cav_feature_session import (
    run_fast_cav_feature_session,
)
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.eval.run_fast_1m_cav import (
    FEATURE_MANIFEST_FORMAT,
    ZERO_STATE_CONTRACT,
    _answer_artifact,
    _atomic_write_json,
    _for_stages,
    _orders_from_session,
    _orders_payload,
    _read_and_validate_answers,
    _read_feature_orders,
    _replay_answer_journals,
    _selected_stages,
    build_parser,
    run_preflight,
)
from memory_condense.search.fusion.fixed_cav_router import FixedCAVRouter
from tests.test_fast_completion_runtime import _FakeClient
from tests.test_fast_cav_feature_session import (
    _FakeEncoder,
    _artifact,
)


def _feature_manifest(tmp_path: Path):
    artifact = _artifact()
    cav_path = tmp_path / "fixture-cavs.safetensors"
    save_file(
        {
            "concept_a.layer_2": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "concept_b.layer_2": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        },
        cav_path,
    )
    router = FixedCAVRouter.load(
        [
            (cav_path, "concept_a.layer_2"),
            (cav_path, "concept_b.layer_2"),
        ],
        layer=2,
        device="cpu",
        dtype="float32",
    )
    session = run_fast_cav_feature_session(
        artifact,
        encoder=_FakeEncoder(),
        router=router,
        layer=2,
    )
    orders = _orders_from_session(session)
    payload = {
        "format": FEATURE_MANIFEST_FORMAT,
        "retrieval_sha256": artifact.raw_sha256,
        "feature_session": asdict(session),
        "router_runtime_receipt": asdict(router.runtime_receipt),
        "stage_orders": _orders_payload(orders),
        "zero_state": {
            "contract": ZERO_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
        },
    }
    path = tmp_path / "features.json"
    digest = _atomic_write_json(path, payload)
    return artifact, session, orders, payload, path, digest


def _completed_answer_manifest(tmp_path: Path):
    artifact, _session, orders, feature_payload, _feature_path, feature_digest = (
        _feature_manifest(tmp_path)
    )
    selected_stages = ("direct_episode_additions",)
    prompt_population = build_fast_cav_prompt_population(
        artifact,
        _for_stages(orders, selected_stages),
        stage_ids=selected_stages,
    )
    checkpoint_dir = tmp_path / "completion-calls"
    provenance = {
        "format": "memory-condense-fast-1m-cav-answer-binding-v1",
        "retrieval_sha256": artifact.raw_sha256,
        "feature_manifest_sha256": feature_digest,
        "feature_session_receipt_sha256": feature_payload["feature_session"][
            "session_receipt_sha256"
        ],
        "prompt_population_sha256": prompt_population.prompt_population_sha256,
        "selected_stage_ids": list(selected_stages),
        "authorized_unique_calls": prompt_population.unique_prompt_count,
        "caller_model_alias": "openai/codex_sdk/fake-model",
        "gateway_url": "https://example.invalid/v1",
        "gold_blind": True,
    }
    runtime = FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=prompt_population.logical_message_population,
        model="codex_sdk/fake-model",
        client=_FakeClient(checkpoint_dir, delay_s=0.0),
        max_prompt_tokens=8_000,
        max_new_tokens=32,
        max_concurrency=2,
        retries=0,
        benchmark_provenance=provenance,
    )
    with runtime:
        completion_batch = runtime.run()
    payload = _answer_artifact(
        mode="answer",
        artifact=artifact,
        feature_sha256=feature_digest,
        feature_payload=feature_payload,
        prompt_population=prompt_population,
        completion_batch=completion_batch,
    )
    path = tmp_path / "answers.json"
    _atomic_write_json(path, payload)
    return artifact, orders, feature_digest, payload, path, checkpoint_dir


def _reseal_journal(path: Path, **changes: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    body = dict(payload)
    body.pop("journal_sha256")
    payload["journal_sha256"] = identity_sha256(body)
    path.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))


def test_stage_alias_parser_is_canonical() -> None:
    assert _selected_stages("S1") == (
        "direct_episode_additions",
    )
    assert _selected_stages("s0,s2") == (
        "causal_graph_coverage_predecessor",
        "representative_episode_additions",
    )
    assert len(_selected_stages("all")) == 4
    with pytest.raises(ValueError, match="preserve"):
        _selected_stages("S2,S0")
    with pytest.raises(ValueError, match="unknown"):
        _selected_stages("S4")


def test_feature_manifest_round_trip_binds_tensor_free_orders(tmp_path: Path) -> None:
    artifact, session, expected, _payload, path, digest = _feature_manifest(tmp_path)

    observed, payload, observed_digest = _read_feature_orders(artifact, path)

    assert observed == expected
    assert observed_digest == digest
    assert payload["feature_session"]["session_receipt_sha256"] == (
        session.session_receipt_sha256
    )
    assert all(row.retained_tensor_bytes == 0 for row in observed)


def test_feature_manifest_rejects_order_receipt_tampering(tmp_path: Path) -> None:
    artifact, _session, _orders, payload, _path, _digest = _feature_manifest(tmp_path)
    row = next(
        item
        for item in payload["stage_orders"]
        if len(item["base_evidence_ids"]) > 1
    )
    row["base_evidence_ids"].reverse()
    tampered = tmp_path / "tampered.json"
    _atomic_write_json(tampered, payload)

    with pytest.raises(ValueError, match="receipt does not verify"):
        _read_feature_orders(artifact, tampered)


def test_preflight_uses_real_feature_orders_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _session, _orders, _payload, path, digest = _feature_manifest(tmp_path)
    args = build_parser().parse_args(
        [
            "--phase",
            "preflight",
            "--features",
            str(path),
            "--stages",
            "S1",
        ]
    )
    monkeypatch.setattr(
        "memory_condense.eval.run_fast_1m_cav._load_artifact",
        lambda _path: artifact,
    )

    result = run_preflight(args)

    assert result["writes"] == result["provider_calls"] == 0
    assert result["feature_manifest_sha256"] == digest
    assert result["order_population_kind"] == "actual_feature_orders"
    assert result["prompt_preflight"]["logical_prompt_count"] == 3
    assert 1 <= result["prompt_preflight"]["unique_prompt_count"] <= 3


def test_cli_defaults_to_provider_free_preflight() -> None:
    args = build_parser().parse_args([])
    assert args.phase == "preflight"
    assert args.stages == "S1"
    assert args.enable_provider is False
    assert args.authorized_provider_calls == 0
    assert args.gateway_model == "codex_sdk/gpt-5.6-terra"
    assert args.extraction_temperature == pytest.approx(0.05)
    assert args.reinjection_temperature == pytest.approx(0.05)


def test_real_sealed_preflight_is_model_and_provider_free(tmp_path: Path) -> None:
    retrieval = Path(
        "eval_results/longmemeval-1m-recall-guarded-cumulative-"
        "development-20260821/retrieval.json"
    )
    if not retrieval.is_file():
        pytest.skip("sealed local 1M retrieval artifact is not present")
    output = tmp_path / "must-not-exist"
    args = build_parser().parse_args(
        [
            "--phase",
            "preflight",
            "--retrieval",
            str(retrieval),
            "--output-root",
            str(output),
            "--stages",
            "S1",
        ]
    )

    result = run_preflight(args)

    assert result["question_count"] == 10
    assert result["logical_evidence_placements"] == 1_939
    assert result["deduplicated_feature_rows"] == 530
    assert result["prompt_preflight"]["logical_prompt_count"] == 30
    assert result["prompt_preflight"]["unique_prompt_count"] == 10
    assert not output.exists()


def test_answer_manifest_round_trip_cross_binds_completion_batch(
    tmp_path: Path,
) -> None:
    artifact, orders, feature_digest, payload, path, checkpoint_dir = (
        _completed_answer_manifest(tmp_path)
    )

    observed, _digest = _read_and_validate_answers(
        artifact,
        path,
        orders,
        feature_digest,
    )
    _replay_answer_journals(
        artifact=artifact,
        answers=observed,
        orders=orders,
        checkpoint_dir=checkpoint_dir,
    )

    assert observed == payload


def test_resealed_answer_prediction_cannot_diverge_from_completion_batch(
    tmp_path: Path,
) -> None:
    artifact, orders, feature_digest, payload, _path, _checkpoint_dir = (
        _completed_answer_manifest(tmp_path)
    )
    tampered = copy.deepcopy(payload)
    tampered["answers"][0]["prediction"] = "self-consistent forged prediction"
    tampered["answers"][0]["prediction_sha256"] = quote_sha256(
        tampered["answers"][0]["prediction"]
    )
    tampered_path = tmp_path / "answers-row-tampered.json"
    _atomic_write_json(tampered_path, tampered)

    with pytest.raises(ValueError, match="logical completions"):
        _read_and_validate_answers(
            artifact,
            tampered_path,
            orders,
            feature_digest,
        )


def test_resealed_logical_completion_cannot_diverge_from_unique_record(
    tmp_path: Path,
) -> None:
    artifact, orders, feature_digest, payload, _path, _checkpoint_dir = (
        _completed_answer_manifest(tmp_path)
    )
    tampered = copy.deepcopy(payload)
    forged = "coordinated row and logical-completion forgery"
    tampered["answers"][0]["prediction"] = forged
    tampered["answers"][0]["prediction_sha256"] = quote_sha256(forged)
    tampered["completion_batch"]["logical_completions"][0] = forged
    tampered_path = tmp_path / "answers-logical-tampered.json"
    _atomic_write_json(tampered_path, tampered)

    with pytest.raises(ValueError, match="journaled completion"):
        _read_and_validate_answers(
            artifact,
            tampered_path,
            orders,
            feature_digest,
        )


def test_runner_replay_rejects_canonical_resealed_request_journal(
    tmp_path: Path,
) -> None:
    artifact, orders, feature_digest, _payload, path, checkpoint_dir = (
        _completed_answer_manifest(tmp_path)
    )
    answers, _digest = _read_and_validate_answers(
        artifact,
        path,
        orders,
        feature_digest,
    )
    request_path = next(checkpoint_dir.glob("*.request.json"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _reseal_journal(
        request_path,
        max_new_tokens=int(request["max_new_tokens"]) + 1,
    )

    with pytest.raises(ValueError, match="request provenance changed"):
        _replay_answer_journals(
            artifact=artifact,
            answers=answers,
            orders=orders,
            checkpoint_dir=checkpoint_dir,
        )


def test_runner_replay_rejects_canonical_resealed_response_journal(
    tmp_path: Path,
) -> None:
    artifact, orders, feature_digest, _payload, path, checkpoint_dir = (
        _completed_answer_manifest(tmp_path)
    )
    answers, _digest = _read_and_validate_answers(
        artifact,
        path,
        orders,
        feature_digest,
    )
    response_path = next(checkpoint_dir.glob("*.response.json"))
    forged = "canonical but non-journaled response"
    _reseal_journal(
        response_path,
        completion=forged,
        completion_sha256=quote_sha256(forged),
        completion_token_proxy=count_tokens(forged),
    )

    with pytest.raises(ValueError, match="immutable provider journals"):
        _replay_answer_journals(
            artifact=artifact,
            answers=answers,
            orders=orders,
            checkpoint_dir=checkpoint_dir,
        )
