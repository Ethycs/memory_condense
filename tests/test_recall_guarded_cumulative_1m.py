from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memory_condense.eval import recall_guarded_cumulative_1m as campaign
from memory_condense.eval import recall_guarded_cumulative_1m_source as source
from memory_condense.eval.diffuse_longmemeval_runtime import (
    gold_blind_from_treatment_sample,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="provider-free-cumulative-fixture",
        turns=[("user", "fixture")],
        turn_source_ids=["fixture-source"],
        turn_created_at=[datetime(2026, 8, 21, tzinfo=timezone.utc)],
        questions=[
            BenchmarkQuestion(
                question_id="fixture-q",
                question="Which two codes were selected?",
                answer="amber, cobalt",
                evidence_sources=["source-amber", "source-cobalt"],
            )
        ],
    )


def _stage(stage_id: str, index: int, evidence: list[dict[str, str]]):
    return {
        "stage_id": stage_id,
        "stage_receipt": {
            "receipt_sha256": f"{index + 1:064x}",
            "selected_evidence_ids": [item["evidence_id"] for item in evidence],
            "context_token_proxy": 100 + index,
            "prompt_token_proxy": 200 + index,
            "admission_status": "root" if index == 0 else "added",
        },
        "provider_messages": [
            {"role": "system", "content": "answer from evidence"},
            {"role": "user", "content": "fixture prompt"},
        ],
        "evidence": evidence,
    }


def _source_receipt(
    sample: BenchmarkSample,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    key = "1" * 64
    query_key = "2" * 64
    body: dict[str, object] = {
        "format": campaign.CURRENT_SOURCE_FORMAT,
        "source_scope": campaign.CURRENT_SOURCE_SCOPE,
        "timestamp_semantics": campaign.CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "base_store_key": key,
        "selected_store_entry": f"stores/{key}",
        "store_manifest_sha256": "3" * 64,
        "store_artifact_sha256": "4" * 64,
        "database_sha256": "5" * 64,
        "index_sha256": "6" * 64,
        "corpus_sha256": gold_blind_from_treatment_sample(sample).corpus_sha256,
        "turn_count": len(sample.turns),
        "chunk_count": 1,
        "deterministic_turn_ids_sha256": "7" * 64,
        "turn_sequence_sha256": "8" * 64,
        "chunk_sequence_sha256": "9" * 64,
        "source_streams_sha256": "a" * 64,
        "embedding_identity": {
            "backend": "sentence-transformers.encode-v1",
            "model_id": DEFAULT_MODEL_NAME,
            "model_revision": DEFAULT_MODEL_REVISION,
            "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
            "dimension": DEFAULT_MODEL_DIM,
            "device": device,
            "batch_size": 32,
            "normalize_embeddings": False,
            "output_dtype": "float32",
        },
        "embedding_identity_sha256": "b" * 64,
        "build_runtime_identity_sha256": "c" * 64,
        "implementation_sha256": "d" * 64,
        "environment_lock_sha256": "e" * 64,
        "query_input_key": query_key,
        "selected_query_entry": f"query-inputs/{query_key}",
        "query_manifest_sha256": "f" * 64,
        "query_artifact_sha256": "0" * 64,
    }
    body["receipt_sha256"] = campaign.identity_sha256(body)
    return body


def test_posthoc_score_reads_already_published_gold_blind_stages(
    tmp_path,
    monkeypatch,
):
    sample = _sample()
    amber = {
        "evidence_id": "amber-id",
        "source_id": "source-amber",
        "text": "The first code was amber.",
    }
    cobalt = {
        "evidence_id": "cobalt-id",
        "source_id": "source-cobalt",
        "text": "The second code was cobalt.",
    }
    stages = [
        _stage(campaign.STAGE_IDS[0], 0, [amber]),
        _stage(campaign.STAGE_IDS[1], 1, [amber, cobalt]),
        _stage(campaign.STAGE_IDS[2], 2, [amber, cobalt]),
        _stage(campaign.STAGE_IDS[3], 3, [amber, cobalt]),
    ]
    question_part = {
        "question_id": "fixture-q",
        "retrieval_implementation_sha256": "1" * 64,
        "retrieval_receipt": {"receipt_sha256": "e" * 64},
        "stages": stages,
    }
    retrieval = {
        "format": campaign.RETRIEVAL_FORMAT,
        "campaign_format": campaign.CAMPAIGN_FORMAT,
        "archived_compiled_sample_sha256": campaign.ORIGINAL_SAMPLE_SHA256,
        "archived_source_provenance": (
            campaign.archived_source_provenance_payload()
        ),
        "source_timestamp_semantics": (
            campaign.CURRENT_SOURCE_TIMESTAMP_SEMANTICS
        ),
        "source_store_receipt": _source_receipt(sample, device="cpu"),
        "population_identity": campaign.population_identity_payload(sample),
        "population_identity_sha256": campaign.population_identity_sha256(sample),
        "transcript_tokens": 1,
        "turn_count": 1,
        "question_count": 1,
        "stage_ids": list(campaign.STAGE_IDS),
        "retrieval_policy_sha256": "a" * 64,
        "retrieval_implementation_sha256": "1" * 64,
        "combined_store_receipt": {
            "receipt_sha256": "b" * 64,
            "source_database_sha256": "5" * 64,
        },
        "compilation_receipt_sha256": "c" * 64,
        "question_part_sha256s": [
            hashlib.sha256(
                campaign._canonical_json_bytes(question_part)
            ).hexdigest()
        ],
        "questions": [question_part],
        "provider_calls": 0,
        "gold_fields_present": False,
    }
    retrieval_path = tmp_path / "retrieval.json"
    campaign._atomic_write_json(retrieval_path, retrieval)
    before = retrieval_path.read_bytes()
    validated: list[str] = []

    def validate_part(part, **_kwargs):
        validated.append(part["question_id"])

    monkeypatch.setattr(campaign, "_validate_question_part", validate_part)

    scores, _digest = campaign.score_published_retrieval(
        sample=sample,
        retrieval_path=retrieval_path,
        output_path=tmp_path / "scores.json",
        source_embedding_device="cpu",
    )

    assert retrieval_path.read_bytes() == before
    assert scores["responder_calls"] == 0
    assert scores["judge_calls"] == 0
    assert scores["retrieval_implementation_sha256"] == "1" * 64
    assert validated == ["fixture-q"]
    assert scores["aggregates"][0]["mean_evidence_source_recall"] == 0.5
    assert scores["aggregates"][1]["mean_evidence_source_recall"] == 1.0
    assert scores["questions"][0]["stages"][1]["retrieved_source_ids"] == [
        "source-amber",
        "source-cobalt",
    ]

    invalid = dict(retrieval)
    invalid["retrieval_implementation_sha256"] = "not-a-digest"
    invalid_path = tmp_path / "invalid-retrieval.json"
    campaign._atomic_write_json(invalid_path, invalid)
    with pytest.raises(ValueError, match="implementation digest"):
        campaign.score_published_retrieval(
            sample=sample,
            retrieval_path=invalid_path,
            output_path=tmp_path / "invalid-scores.json",
            source_embedding_device="cpu",
        )


def test_population_identity_binds_question_order_and_probe_bytes():
    sample = _sample()
    payload = campaign._canonical_json_bytes(
        campaign.population_identity_payload(sample)
    )
    baseline = campaign.population_identity_sha256(sample)
    changed = sample.model_copy(
        update={
            "questions": [
                sample.questions[0].model_copy(
                    update={"question": "Which codes were selected yesterday?"}
                )
            ]
        }
    )

    assert b"amber, cobalt" not in payload
    assert b"source-amber" not in payload
    assert campaign.population_identity_sha256(changed) != baseline


def test_1m_cli_requires_an_explicit_dataset_path() -> None:
    with pytest.raises(SystemExit):
        campaign._parser().parse_args([])

    parsed = campaign._parser().parse_args(["--dataset", "fixture.json"])

    assert parsed.dataset.as_posix() == "fixture.json"


def test_prepare_store_forwards_the_selected_qwen_prefix_directory(
    tmp_path,
    monkeypatch,
) -> None:
    selected = tmp_path / "custom-qwen-prefix"
    observed: list[object] = []

    class StopAfterBinding(RuntimeError):
        pass

    def current_source_binding(config, *, qwen_model_dir):
        observed.extend((config, qwen_model_dir))
        raise StopAfterBinding

    monkeypatch.setattr(
        campaign,
        "current_source_binding",
        current_source_binding,
    )
    config = EvalConfig(retrieval=RetrievalConfig(mode="dense"))

    with pytest.raises(StopAfterBinding):
        campaign.prepare_store(
            sample=_sample(),
            config=config,
            source_dir=tmp_path / "source",
            combined_dir=tmp_path / "combined",
            qwen_prefix_model_dir=selected,
        )

    assert observed == [config, selected]


def test_source_phase_forwards_the_selected_qwen_prefix_directory(
    tmp_path,
    monkeypatch,
) -> None:
    sample = _sample()
    config = EvalConfig(retrieval=RetrievalConfig(mode="dense"))
    selected = tmp_path / "custom-qwen-prefix"
    observed: list[object] = []

    class Embedder:
        closed = False

        def close(self) -> None:
            self.closed = True

    embedder = Embedder()
    binding = SimpleNamespace(embedder=embedder)
    monkeypatch.setattr(
        campaign,
        "load_original_population",
        lambda *_args, **_kwargs: sample,
    )
    monkeypatch.setattr(
        campaign,
        "load_frozen_config",
        lambda *_args, **_kwargs: config,
    )

    def current_source_binding(active_config, *, qwen_model_dir):
        observed.extend((active_config, qwen_model_dir))
        return active_config, binding

    monkeypatch.setattr(
        campaign,
        "current_source_binding",
        current_source_binding,
    )
    monkeypatch.setattr(
        campaign,
        "prepare_current_source_store",
        lambda **_kwargs: (
            tmp_path / "memory.db",
            {"receipt_sha256": "a" * 64},
            "fixture",
        ),
    )

    assert campaign.main(
        [
            "--phase",
            "source",
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--qwen-prefix-model-dir",
            str(selected),
            "--output-root",
            str(tmp_path / "output"),
        ]
    ) == 0
    assert observed == [config, selected]
    assert embedder.closed is True


def test_retrieval_and_main_score_forward_nondefault_source_device(
    tmp_path,
    monkeypatch,
) -> None:
    sample = _sample()
    config = EvalConfig(retrieval=RetrievalConfig(mode="dense"))
    retrieval_devices: list[str] = []

    class StopAfterSourceValidation(RuntimeError):
        pass

    def validate_source(_receipt, *, sample, expected_device):
        assert sample is not None
        retrieval_devices.append(expected_device)
        raise StopAfterSourceValidation

    monkeypatch.setattr(
        campaign,
        "validate_current_source_receipt",
        validate_source,
    )
    with pytest.raises(StopAfterSourceValidation):
        campaign.run_gold_blind_retrieval(
            prepared=object(),
            sample=sample,
            config=config,
            selector=object(),
            representative_linker=object(),
            output_root=tmp_path / "retrieval-output",
            source_store_receipt={},
            source_embedding_device="cpu",
        )
    assert retrieval_devices == ["cpu"]

    score_devices: list[str] = []
    monkeypatch.setattr(
        campaign,
        "load_original_population",
        lambda *_args, **_kwargs: sample,
    )
    monkeypatch.setattr(
        campaign,
        "load_frozen_config",
        lambda *_args, **_kwargs: config,
    )

    def score_published_retrieval(**kwargs):
        score_devices.append(kwargs["source_embedding_device"])
        return {}, "a" * 64

    monkeypatch.setattr(
        campaign,
        "score_published_retrieval",
        score_published_retrieval,
    )
    assert campaign.main(
        [
            "--phase",
            "score",
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--device",
            "cpu",
            "--output-root",
            str(tmp_path / "score-output"),
        ]
    ) == 0
    assert score_devices == ["cpu"]


def test_canonical_artifact_requires_matching_digest_sidecar(tmp_path):
    path = tmp_path / "artifact.json"
    campaign._atomic_write_json(path, {"value": 1})
    path.with_name(path.name + ".sha256").write_text(
        f"{'0' * 64}  {path.name}\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="digest sidecar"):
        campaign._read_canonical_json(path)


def test_declared_current_source_selection_is_verify_only(
    tmp_path,
    monkeypatch,
):
    sample = _sample()
    root = tmp_path / "source-current"
    store_key, query_key = "1" * 64, "2" * 64
    store_path = root / "stores" / store_key
    query_path = root / "query-inputs" / query_key
    payload_path = store_path / "store"
    payload_path.mkdir(parents=True)
    query_path.mkdir(parents=True)
    database = payload_path / "memory.db"
    index = payload_path / "hnsw_index.bin"
    database.write_bytes(b"verified database")
    index.write_bytes(b"verified index")
    embedding_payload = {
        "backend": "sentence-transformers.encode-v1",
        "model_id": DEFAULT_MODEL_NAME,
        "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "device": "cuda",
        "batch_size": 32,
        "normalize_embeddings": False,
        "output_dtype": "float32",
    }
    store = SimpleNamespace(
        base_store_key=store_key,
        artifact_sha256="3" * 64,
        database_sha256=campaign.file_sha256(database),
        index_sha256=campaign.file_sha256(index),
        corpus_sha256=gold_blind_from_treatment_sample(sample).corpus_sha256,
        turn_count=1,
        chunk_count=1,
        deterministic_turn_ids_sha256="4" * 64,
        turn_sequence_sha256="5" * 64,
        chunk_sequence_sha256="6" * 64,
        source_streams_sha256="7" * 64,
        embedding_identity=SimpleNamespace(
            model_dump=lambda **_kwargs: embedding_payload
        ),
        embedding_identity_sha256="8" * 64,
        build_runtime_identity_sha256="9" * 64,
        implementation_sha256="a" * 64,
        environment_lock_sha256="b" * 64,
    )
    query = SimpleNamespace(
        query_input_key=query_key,
        artifact_sha256="c" * 64,
    )
    base = SimpleNamespace(
        store_path=store_path,
        query_inputs_path=query_path,
        store_manifest=store,
        query_manifest=query,
        store_manifest_sha256="d" * 64,
        query_manifest_sha256="e" * 64,
    )
    selection = tmp_path / source.CURRENT_SOURCE_SELECTION_NAME
    source._write_selection(selection, source._source_receipt(base, source_root=root))
    calls = {"verify": 0, "publish": 0}
    verified_kwargs = {}

    def verify(*_args, **kwargs):
        calls["verify"] += 1
        verified_kwargs.update(kwargs)
        return base

    def publish(*_args, **_kwargs):
        calls["publish"] += 1
        raise AssertionError("declared source selection must never rebuild")

    monkeypatch.setattr(source, "owned_build_runtime_identity", lambda _value: object())
    monkeypatch.setattr(source, "verify_diffuse_longmemeval_base", verify)
    monkeypatch.setattr(source, "publish_diffuse_longmemeval_base", publish)
    binding = SimpleNamespace(
        new_condenser=lambda _path: None,
        embedding_identity=embedding_payload,
        embedder=object(),
    )
    config = EvalConfig(
        retrieval=RetrievalConfig(mode="dense"),
        embedding_device="cuda",
        max_prompt_tokens=8000,
    )

    observed_database, observed_receipt, mode = source.prepare_current_source_store(
        sample=sample,
        config=config,
        treatment_identity=object(),
        binding=binding,
        source_root=root,
        selection_path=selection,
    )

    assert observed_database == database
    assert observed_receipt["database_sha256"] == campaign.file_sha256(database)
    assert mode == "verified_cache_hit"
    assert calls == {"verify": 1, "publish": 0}
    assert verified_kwargs["implementation_digest"] == "a" * 64
    assert verified_kwargs["environment_digest"] == "b" * 64
