"""Provider-free tests for the closed production-corpus launcher boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from tools import _diffuse_latent_training_corpus_authority as authority
from tools import _diffuse_latent_training_corpus_authority_filesystem as candidate_fs
from tools import _diffuse_latent_training_corpus_authority_models as models
from tools import _diffuse_latent_training_corpus_candidate_verification as candidate_verify
from tools import run_diffuse_latent_training_corpus as launcher
from tools._diffuse_latent_training_corpus_authority_codec import (
    canonical_candidate_bytes,
    decode_production_candidate,
    encode_production_candidate,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _execution(*, package_sha256: str | None = None, corpus_sha256: str | None = None):
    lock = models.locked_production_external_lock()
    package = package_sha256 or _sha("memory-condense-package")
    corpus = corpus_sha256 or identity_sha256(
        {
            "format": "memory-condense-latent-training-corpus-implementation-v2",
            "memory_condense_package_sha256": package,
        }
    )
    retrieval_contract = identity_sha256(
        {
            "format": "qwen-prefix-layer-contract-v1",
            "model_id": lock.qwen_model_id,
            "model_revision": lock.qwen_model_revision,
            "checkpoint_sha256": lock.qwen_checkpoint_sha256,
            "retained_layers": lock.retrieval_prefix_layers,
            "selected_layer_kind": "attention",
            "selected_layer": lock.retrieval_attention_layer,
        }
    )
    feature_contract = identity_sha256(
        {
            "format": "qwen-prefix-layer-contract-v1",
            "model_id": lock.qwen_model_id,
            "model_revision": lock.qwen_model_revision,
            "checkpoint_sha256": lock.qwen_checkpoint_sha256,
            "retained_layers": lock.feature_prefix_layers,
            "selected_layer_kind": "output",
            "selected_layer": lock.feature_output_layer,
        }
    )
    return models.DeclaredProductionExecutionCoordinates(
        launcher_relative_path="tools/run_diffuse_latent_training_corpus.py",
        launcher_sha256=_sha("launcher"),
        source_commit="0" * 40,
        package_implementation_sha256=package,
        corpus_implementation_sha256=corpus,
        route_implementation_sha256=_sha("route"),
        runtime_binding_sha256=_sha("runtime"),
        ordered_legacy_input_provider_identities_sha256=_sha("providers"),
        representative_linker_identity_sha256=_sha("linker"),
        representative_policy_factory_identity_sha256=_sha("factory"),
        bge_checkpoint_sha256=lock.bge_checkpoint_sha256,
        qwen_retrieval_checkpoint_sha256=lock.qwen_checkpoint_sha256,
        qwen_feature_checkpoint_sha256=lock.qwen_checkpoint_sha256,
        qwen_retrieval_contract_sha256=retrieval_contract,
        qwen_feature_contract_sha256=feature_contract,
    )


def _candidate(
    execution: models.DeclaredProductionExecutionCoordinates | None = None,
) -> models.ProductionCorpusCandidateReceipt:
    coordinates = execution or _execution()
    return models.ProductionCorpusCandidateReceipt(
        generic_root_manifest_sha256=_sha("generic-root"),
        generic_root_manifest_bytes=101,
        generic_corpus_sha256=_sha("generic-corpus"),
        generic_inventory_sha256=_sha("generic-inventory"),
        generic_population_projection_sha256=_sha("generic-population"),
        generic_implementation_sha256=coordinates.corpus_implementation_sha256,
        generic_fit_partition_sha256=_sha("fit-partition"),
        generic_fit_manifest_file_sha256=_sha("fit-manifest"),
        generic_fit_manifest_file_bytes=102,
        generic_validation_partition_sha256=_sha("validation-partition"),
        generic_validation_manifest_file_sha256=_sha("validation-manifest"),
        generic_validation_manifest_file_bytes=103,
        external_lock=models.locked_production_external_lock(),
        declared_execution=coordinates,
    )


class _HostileArgument:
    def __init__(self) -> None:
        self.touched = False

    def _fail(self):
        self.touched = True
        raise AssertionError("closed launcher touched a caller argument")

    def __fspath__(self):
        return self._fail()

    def __bool__(self):
        return self._fail()

    def __str__(self):
        return self._fail()

    def __repr__(self):
        return self._fail()


def test_public_run_signature_and_status_are_closed_false() -> None:
    signature = inspect.signature(launcher.run)
    assert tuple(signature.parameters) == (
        "treatment_input",
        "qwen_model_dir",
        "output_root",
        "restart",
    )
    assert signature.parameters["restart"].default is False

    status = launcher.candidate_execution_status()
    assert type(status) is models.ProductionCandidateExecutionStatus
    assert status.reason == "candidate_path_handoffs_not_capability_safe"
    assert status.reason == models.CANDIDATE_EXECUTION_DISABLED_REASON
    assert status.candidate_execution_enabled is False
    assert status.source_runtime_verified is False
    assert status.production_authorized is False
    assert status.d1_eligible is False
    assert status.validation_eligible is False

    with pytest.raises(ValueError, match="must remain false"):
        models.ProductionCandidateExecutionStatus(
            candidate_execution_enabled=True,
        )


def test_public_run_rejects_before_argument_or_helper_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = tuple(_HostileArgument() for _ in range(4))

    def bomb(*_args, **_kwargs):
        raise AssertionError("closed launcher reached private execution plumbing")

    for name in (
        "_run_real_candidate",
        "_absolute_child",
        "load_analysis_treatment_input",
        "create_candidate_staging",
        "verify_latent_training_corpus_candidate",
    ):
        monkeypatch.setattr(launcher, name, bomb)
    monkeypatch.setattr(
        launcher,
        "CANDIDATE_EXECUTION_DISABLED_REASON",
        "caller_mutated_reason",
    )
    monkeypatch.setattr(launcher, "ProductionCandidateExecutionUnavailable", bomb)

    with pytest.raises(
        models.ProductionCandidateExecutionUnavailable,
        match=models.CANDIDATE_EXECUTION_DISABLED_REASON,
    ):
        launcher.run(*arguments)

    assert all(argument.touched is False for argument in arguments)
    assert not (tmp_path / "unused-output").exists()


def test_cli_reports_disabled_state_without_success_receipt(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-be-created"
    result = launcher.main(
        [
            "--treatment-input", str(tmp_path / "missing-treatment.json"),
            "--qwen-model-dir", str(tmp_path / "missing-qwen"),
            "--output-root", str(output),
            "--restart",
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert models.CANDIDATE_EXECUTION_DISABLED_REASON in captured.err
    assert "LATENT_TRAINING_CORPUS_CANDIDATE" not in captured.err
    assert "production_authorized" not in captured.err
    assert not output.exists()


def test_private_execution_workspace_uses_the_public_frozen_pointer_name() -> None:
    from tools._diffuse_latent_training_corpus_workspace import _schema

    _schema(
        "execution",
        ("cache", "query-inputs", "a" * 64),
        ("frozen-legacy-inputs.json", "query-manifest.json"),
    )
    with pytest.raises(
        models.ProductionLatentTrainingCorpusError, match="exact schema"
    ):
        _schema(
            "execution",
            ("cache", "query-inputs", "a" * 64),
            ("frozen-query-inputs.json", "query-manifest.json"),
        )


@pytest.mark.parametrize(
    "verifier",
    (
        authority.verify_production_latent_training_corpus,
        authority.verify_production_latent_training_fit_corpus,
        authority.verify_production_latent_training_validation_corpus,
    ),
)
def test_unpinned_production_verifiers_reject_before_path_access(verifier) -> None:
    hostile = _HostileArgument()
    with pytest.raises(
        models.ProductionAuthorityNotPinned,
        match=models.AUTHORITY_NOT_PINNED_REASON,
    ):
        verifier(hostile)
    assert hostile.touched is False


def test_candidate_codec_closes_package_implementation_identity() -> None:
    candidate = _candidate()
    payload = encode_production_candidate(candidate)
    decoded = decode_production_candidate(payload)
    assert type(decoded) is models.ProductionCorpusCandidateReceipt
    assert decoded == candidate
    assert decoded.declared_execution.package_implementation_sha256 == (
        candidate.declared_execution.package_implementation_sha256
    )
    assert decoded.production_authorized is False
    assert decoded.d1_eligible is False
    assert decoded.validation_eligible is False
    assert decoded.retrieval_qwen_execution_attested is False
    assert decoded.feature_qwen_execution_attested is False

    body = json.loads(payload)
    del body["declared_execution"]["package_implementation_sha256"]
    with pytest.raises(models.ProductionLatentTrainingCorpusError, match="closed schema"):
        decode_production_candidate(canonical_candidate_bytes(body))


def test_candidate_rejects_package_corpus_identity_mismatch() -> None:
    execution = _execution(
        package_sha256=_sha("other-package"),
        corpus_sha256=identity_sha256(
            {
                "format": "memory-condense-latent-training-corpus-implementation-v2",
                "memory_condense_package_sha256": _sha("original-package"),
            }
        ),
    )
    with pytest.raises(ValueError, match="package/corpus implementation join"):
        _candidate(execution)


@pytest.mark.skipif(os.name != "nt", reason="Windows held-handle regression")
@pytest.mark.parametrize("module", (candidate_fs, candidate_verify))
def test_windows_link_check_uses_the_held_handle(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile_path = _HostileArgument()
    entry = SimpleNamespace(handle=123, path=hostile_path, identity=(1, 2, 3, 4, 5, 6, 1))
    observed: list[object] = []

    def held_identity(value) -> tuple[int, ...]:
        observed.append(value)
        return entry.identity

    monkeypatch.setattr(module, "_current_identity", held_identity)
    monkeypatch.setattr(module, "_assert_current", lambda _entry: None)
    module._require_one_link(entry)
    assert observed == [entry]
    assert hostile_path.touched is False

    monkeypatch.setattr(
        module,
        "_current_identity",
        lambda _entry: (*entry.identity[:-1], 2),
    )
    with pytest.raises(
        models.ProductionLatentTrainingCorpusError,
        match="hard-linked",
    ):
        module._require_one_link(entry)
    assert hostile_path.touched is False


def test_subtree_helpers_reject_an_alternate_outer_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (1, 2, 3, 4, 5, 6, 1)
    alternate = (9, 8, 7, 6, 5, 4, 1)

    class FakePhase:
        def __init__(self, _path: Path) -> None:
            self.root = SimpleNamespace(identity=alternate)

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(candidate_verify, "_HeldPhase", FakePhase)
    monkeypatch.setattr(
        candidate_verify,
        "_verify_held_phase",
        lambda *_args: pytest.fail("alternate phase reached decoding"),
    )
    with pytest.raises(
        models.ProductionLatentTrainingCorpusError,
        match="another outer child",
    ):
        candidate_verify._verify_phase_path(
            Path("unused-fit"),
            "fit",
            expected_root_identity=expected,
        )

    class FakeSnapshot:
        def __init__(self, _path: Path) -> None:
            self._ancestors = [SimpleNamespace(identity=alternate)]

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(candidate_verify, "CorpusTreeSnapshot", FakeSnapshot)
    monkeypatch.setattr(
        candidate_verify,
        "_verify_generic_snapshot",
        lambda *_args: pytest.fail("alternate generic tree reached decoding"),
    )
    monkeypatch.setattr(
        candidate_verify,
        "latent_training_corpus_implementation_sha256",
        lambda: _sha("corpus-implementation"),
    )
    monkeypatch.setattr(
        candidate_verify,
        "live_route_v2_implementation_sha256",
        lambda: _sha("route-implementation"),
    )
    with pytest.raises(
        models.ProductionLatentTrainingCorpusError,
        match="another outer child",
    ):
        candidate_verify._verify_generic_with_binding(
            Path("unused-generic"),
            expected_root_identity=expected,
        )

def test_launcher_cold_import_is_scoring_and_model_free() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys;"
        f"sys.path.insert(0,{str(root)!r});"
        "import tools.run_diffuse_latent_training_corpus as launcher;"
        "denied={'accelerate','anthropic','cohere','google','httpx','litellm',"
        "'mistralai','openai','requests','safetensors','sentence_transformers',"
        "'torch','transformers'};"
        "bad=sorted(name for name in sys.modules if "
        "name == 'tools.v4_population_firebreak.scoring' or "
        "name.split('.')[0] in denied);"
        "assert 'AnalysisScoringLabel' not in launcher.__dict__;"
        "print(','.join(bad))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == ""
