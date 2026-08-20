from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_diffuse_longmemeval_shared_base_replay as launcher
from memory_condense.domain.discourse import identity_sha256
from memory_condense.eval.diffuse_longmemeval_replay import ReplayExecutionIdentity
from tools.v4_population_firebreak import (
    AnalysisTreatmentInput,
    TreatmentQuestion,
    TreatmentSample,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _treatment() -> AnalysisTreatmentInput:
    samples = tuple(
        TreatmentSample(
            sample_id=f"sample-{index}",
            turns=(("user", f"private transcript {index}"),),
            turn_source_ids=(f"source-{index}",),
            turn_created_at=(None,),
            questions=(
                TreatmentQuestion(
                    question_id=f"sample-{index}",
                    question=f"private question {index}",
                    question_date=None,
                ),
            ),
        )
        for index in range(launcher.PINNED_SAMPLE_COUNT)
    )
    return AnalysisTreatmentInput(
        file_sha256=launcher.PINNED_TREATMENT_FILE_SHA256,
        sanitized_projection_sha256=(
            launcher.PINNED_SANITIZED_PROJECTION_SHA256
        ),
        dataset_sha256=launcher.PRODUCTION_LOCK.dataset_sha256,
        split_manifest_sha256=launcher.PRODUCTION_LOCK.split_manifest_sha256,
        ordered_question_ids_sha256=(
            launcher.PRODUCTION_LOCK.analysis_ordered_question_ids_sha256
        ),
        samples=samples,
    )


@dataclass
class _FakeBinding:
    binding_sha256: str

    def __post_init__(self) -> None:
        self.config = launcher._campaign_config("cuda:0")
        self.embedding_identity = {"device": "cuda:0"}
        self.new_condenser = lambda _path: None


def _install_fake_campaign(
    monkeypatch,
    *,
    runtime_sha256s: tuple[str, ...] = (_sha("runtime"), _sha("runtime")),
):
    treatment = _treatment()
    execution = ReplayExecutionIdentity(
        launcher_sha256=_sha("launcher"),
        source_commit="1" * 40,
        tracked_worktree_clean=True,
    )
    loads: list[Path] = []
    bindings: list[_FakeBinding] = []
    run_calls: list[dict] = []
    base_calls: list[dict] = []
    verify_calls: list[dict] = []
    certification_calls: list[Path] = []
    manifest_payload = b"{}\n"
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    store_manifest = SimpleNamespace(
        base_store_key=_sha("base-key"),
        artifact_sha256=_sha("base-artifact"),
    )
    query_manifest = SimpleNamespace(
        query_input_key=_sha("query-key"),
        artifact_sha256=_sha("query-artifact"),
    )

    def load(path):
        loads.append(Path(path))
        return treatment

    def new_binding(_qwen_path, _device):
        binding = _FakeBinding(runtime_sha256s[len(bindings)])
        bindings.append(binding)
        return binding

    def certify(path):
        certification_calls.append(Path(path))
        return execution

    def run(sample, **kwargs):
        run_calls.append({"sample": sample, **kwargs})
        cache_root = Path(kwargs["cache_root"])
        base_path = cache_root / "stores" / store_manifest.base_store_key
        query_path = cache_root / "query-inputs" / query_manifest.query_input_key
        base_path.mkdir(parents=True)
        query_path.mkdir(parents=True)
        (base_path.parent / f".{store_manifest.base_store_key}.publish.lock").write_bytes(
            b"0"
        )
        (
            query_path.parent / f".{query_manifest.query_input_key}.publish.lock"
        ).write_bytes(b"0")
        (base_path / "base-manifest.json").write_bytes(manifest_payload)
        (query_path / "query-manifest.json").write_bytes(manifest_payload)
        replay_root = Path(kwargs["replay_root"])
        replay_root.mkdir()
        (replay_root / "replay-manifest.json").write_bytes(manifest_payload)
        (replay_root.parent / ".replay.publish.lock").write_bytes(b"0")
        treatment_sha = identity_sha256(
            kwargs["treatment_identity"].model_dump(mode="json")
        )
        blind = launcher.gold_blind_from_treatment_sample(sample)
        question_rows = tuple(
            SimpleNamespace(
                question_id_sha256=identity_sha256(
                    {"question_id": item.question_id}
                ),
                question_probe_sha256=item.probe_sha256,
            )
            for item in blind.questions
        )
        runtime_payload = {
            "embedding": {
                "checkpoint_sha256": launcher.BGE_M3_CHECKPOINT_SHA256
            },
            "qwen": {"checkpoint_sha256": _sha("qwen-prefix")},
        }
        return SimpleNamespace(
            execution_identity=execution,
            launcher_binding_certified=True,
            runtime_binding=SimpleNamespace(
                identity_sha256=kwargs["binding"].binding_sha256,
                canonical_identity_json=json.dumps(
                    runtime_payload, sort_keys=True, separators=(",", ":")
                ),
            ),
            base_manifest=SimpleNamespace(
                base_store_key=store_manifest.base_store_key,
                artifact_sha256=store_manifest.artifact_sha256,
                embedding_identity=SimpleNamespace(
                    checkpoint_sha256=launcher.BGE_M3_CHECKPOINT_SHA256
                ),
                sample_id_sha256=identity_sha256(
                    {"sample_id": blind.sample_id}
                ),
                corpus_sha256=blind.corpus_sha256,
                turn_count=len(blind.turns),
            ),
            query_manifest=SimpleNamespace(
                treatment_identity_sha256=treatment_sha,
                treatment_identity=kwargs["treatment_identity"],
                query_input_key=query_manifest.query_input_key,
                artifact_sha256=query_manifest.artifact_sha256,
                query_count=len(blind.questions),
            ),
            arms=(SimpleNamespace(queries=question_rows),) * 3,
            treatment_population_membership_certified=False,
            retrieval_input_schema_contains_gold_fields=False,
            qa_responder_or_judge_calls=0,
            receipt_sha256=_sha("replay-receipt"),
            base_manifest_file_sha256=manifest_sha256,
            query_manifest_file_sha256=manifest_sha256,
        )

    verified_base = SimpleNamespace(
        store_manifest=store_manifest,
        query_manifest=query_manifest,
        store_manifest_sha256=manifest_sha256,
        query_manifest_sha256=manifest_sha256,
    )

    def verify_base(cache_root, **kwargs):
        base_calls.append({"cache_root": cache_root, **kwargs})
        return verified_base

    def verify_replay(path, **kwargs):
        verify_calls.append({"path": path, **kwargs})
        return run_calls[0]["result"]

    def run_with_result(sample, **kwargs):
        result = run(sample, **kwargs)
        run_calls[0]["result"] = result
        return result

    monkeypatch.setattr(launcher, "_load_pinned_treatment", load)
    monkeypatch.setattr(launcher, "_new_owned_binding", new_binding)
    monkeypatch.setattr(launcher, "_require_campaign_binding", lambda _value: None)
    monkeypatch.setattr(launcher, "certify_replay_launcher", certify)
    monkeypatch.setattr(
        launcher,
        "_verify_local_checkpoints",
        lambda _binding, _path: launcher.CampaignCheckpointIdentity(
            bge_m3_checkpoint_sha256=launcher.BGE_M3_CHECKPOINT_SHA256,
            qwen_prefix_checkpoint_sha256=_sha("qwen-prefix"),
            qwen_prefix_verified_file_count=7,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "run_diffuse_longmemeval_shared_base_replay",
        run_with_result,
    )
    monkeypatch.setattr(
        launcher, "owned_build_runtime_identity", lambda _factory: "owned-build"
    )
    monkeypatch.setattr(launcher, "verify_diffuse_longmemeval_base", verify_base)
    monkeypatch.setattr(
        launcher, "verify_diffuse_longmemeval_replay_package", verify_replay
    )
    monkeypatch.setattr(
        launcher,
        "_load_nested_replay_manifest",
        lambda _path: run_calls[0]["result"],
    )
    return SimpleNamespace(
        treatment=treatment,
        execution=execution,
        loads=loads,
        bindings=bindings,
        run_calls=run_calls,
        base_calls=base_calls,
        verify_calls=verify_calls,
        certification_calls=certification_calls,
    )


def test_campaign_pins_the_proven_canary_controls():
    config = launcher._campaign_config("cuda:0")
    arm = launcher._reference_arm()
    runtime = launcher._campaign_runtime(Path("local-qwen"), "cuda:0")

    assert launcher.PINNED_SAMPLE_ORDINAL == 169
    assert launcher.PINNED_SAMPLE_COUNT == 300
    assert config.embedding_device == "cuda:0"
    assert config.max_prompt_tokens == 8000
    assert config.chunker.min_tokens == 120
    assert config.chunker.max_tokens == 250
    assert config.retrieval.mode == "hybrid_graph"
    assert config.retrieval.k == 10
    assert config.retrieval.source_candidate_pool == 750
    assert config.retrieval.qwen_rerank is False
    assert config.retrieval.qwen_feedback is False
    assert runtime.residency_mode == "resident_bge_qwen"
    assert runtime.qwen_device == "cuda:0"
    assert arm.max_context_tokens == 7000
    assert arm.responder_output_token_reserve == 256
    assert arm.episode.max_anchor_episodes == 96
    assert arm.closure.max_bundles == 256


@pytest.mark.parametrize("device", ("cpu", "cuda:-1", "cuda:01", "cuda:any"))
def test_campaign_rejects_noncanonical_cuda_devices(device):
    with pytest.raises(ValueError, match="device must be"):
        launcher._canonical_cuda_device(device)


def test_campaign_normalizes_default_cuda():
    assert launcher._canonical_cuda_device("CUDA") == "cuda:0"
    assert launcher._canonical_cuda_device("cuda:2") == "cuda:2"


def test_campaign_requires_both_offline_environment_variables(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="TRANSFORMERS_OFFLINE"):
        launcher._require_offline_environment()


def test_campaign_requires_pixi_openmp_activation(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    with pytest.raises(RuntimeError, match="KMP_DUPLICATE_LIB_OK"):
        launcher._require_offline_environment()


def test_fake_campaign_reloads_and_independently_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    installed = _install_fake_campaign(monkeypatch)
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    treatment_path = tmp_path / "sanitized-treatment.json"
    treatment_path.write_text("fixture is intercepted", encoding="utf-8")
    launcher_path = tmp_path / "tracked-launcher.py"
    launcher_path.write_text("# certified by fake", encoding="utf-8")
    output = tmp_path / "campaign"

    receipt = launcher._run_campaign(
        treatment_input=treatment_path,
        qwen_model_dir=qwen,
        output_root=output,
        device="cuda",
        launcher_path=launcher_path,
    )

    assert len(installed.loads) == 2
    assert len(installed.bindings) == 2
    assert len(installed.base_calls) == 1
    assert len(installed.verify_calls) == 1
    assert installed.verify_calls[0]["expected_runtime_binding_sha256"] == (
        installed.bindings[1].binding_sha256
    )
    assert installed.run_calls[0]["cache_root"] == output / "cache"
    assert installed.run_calls[0]["replay_root"] == output / "replay"
    assert installed.run_calls[0]["launcher_path"] == launcher_path
    assert installed.run_calls[0]["treatment_identity"].sample_ordinal == 169
    assert installed.base_calls[0]["sample"].sample_id == "sample-169"
    assert installed.certification_calls == [launcher_path, launcher_path]
    assert receipt.launcher == installed.execution
    assert receipt.claims.pinned_sanitized_population_artifact_verified is True
    assert receipt.claims.network_calls_proven_zero is False
    assert receipt.claims.model_outputs_independently_reexecuted is False
    assert receipt.population.turn_count == 1
    assert receipt.population.question_count == 1

    receipt_path = output / launcher.CAMPAIGN_RECEIPT_NAME
    raw = receipt_path.read_bytes()
    assert raw == launcher._canonical_json_bytes(receipt.model_dump(mode="json"))
    parsed = launcher.PinnedReplayCampaignReceipt.model_validate_json(raw)
    assert parsed == receipt
    assert b"private transcript" not in raw
    assert b"private question" not in raw

    altered = receipt.model_dump(mode="json")
    altered["artifacts"]["replay_manifest_file_sha256"] = _sha("altered")
    altered["receipt_sha256"] = identity_sha256(
        {key: value for key, value in altered.items() if key != "receipt_sha256"}
    )
    receipt_path.write_bytes(launcher._canonical_json_bytes(altered))
    with pytest.raises(RuntimeError, match="replay manifest differs"):
        launcher.verify_pinned_replay_campaign_receipt(
            output,
            expected_population=receipt.population,
            expected_launcher=receipt.launcher,
            expected_runtime_binding_sha256=receipt.runtime_binding_sha256,
        )
    receipt_path.write_bytes(raw)

    for section, field in (
        ("artifacts", "base_artifact_sha256"),
        ("artifacts", "query_artifact_sha256"),
        ("checkpoints", "qwen_prefix_checkpoint_sha256"),
    ):
        altered = receipt.model_dump(mode="json")
        altered[section][field] = _sha(f"altered:{section}:{field}")
        altered["receipt_sha256"] = identity_sha256(
            {
                key: value
                for key, value in altered.items()
                if key != "receipt_sha256"
            }
        )
        receipt_path.write_bytes(launcher._canonical_json_bytes(altered))
        with pytest.raises(RuntimeError, match="nested replay differs"):
            launcher.verify_pinned_replay_campaign_receipt(
                output,
                expected_population=receipt.population,
                expected_launcher=receipt.launcher,
                expected_runtime_binding_sha256=receipt.runtime_binding_sha256,
            )
        receipt_path.write_bytes(raw)

    unexpected = output / "cache" / "unexpected"
    unexpected.write_bytes(b"not admitted")
    try:
        with pytest.raises(RuntimeError, match="unexpected or missing"):
            launcher.verify_pinned_replay_campaign_receipt(
                output,
                expected_population=receipt.population,
                expected_launcher=receipt.launcher,
                expected_runtime_binding_sha256=receipt.runtime_binding_sha256,
            )
    finally:
        unexpected.unlink()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        launcher._run_campaign(
            treatment_input=treatment_path,
            qwen_model_dir=qwen,
            output_root=output,
            device="cuda:0",
            launcher_path=launcher_path,
        )


def test_campaign_rejects_changed_independent_runtime_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    _install_fake_campaign(
        monkeypatch,
        runtime_sha256s=(_sha("runtime-a"), _sha("runtime-b")),
    )
    qwen = tmp_path / "qwen"
    qwen.mkdir()

    with pytest.raises(RuntimeError, match="independently derived runtime"):
        launcher._run_campaign(
            treatment_input=tmp_path / "sanitized-treatment.json",
            qwen_model_dir=qwen,
            output_root=tmp_path / "campaign",
            device="cuda:0",
            launcher_path=tmp_path / "tracked-launcher.py",
        )
    assert not (
        tmp_path / "campaign" / launcher.CAMPAIGN_RECEIPT_NAME
    ).exists()
