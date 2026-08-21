from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import MappingProxyType

import pytest
import memory_condense.eval.diffuse_latent_training_corpus as corpus_api


_ROUTE_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_latent_training_route_fixture",
    Path(__file__).with_name("test_diffuse_longmemeval_route_v2.py"),
)
assert _ROUTE_FIXTURE_SPEC is not None and _ROUTE_FIXTURE_SPEC.loader is not None
route_fixture = importlib.util.module_from_spec(_ROUTE_FIXTURE_SPEC)
_ROUTE_FIXTURE_SPEC.loader.exec_module(route_fixture)

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.eval.diffuse_latent_training_corpus import (
    AnalysisPopulationProjection,
    LatentTrainingPopulationExpectation,
    StructuralRouteV2MappedRow,
    LatentTrainingCorpusManifest,
    LatentTrainingCorpusPartitionManifest,
    LatentTrainingCorpusError,
    VerifiedLatentTrainingFitCorpus,
    VerifiedLatentTrainingValidationCorpus,
    _publish_synthetic_structural_latent_training_corpus,
    verify_structural_latent_training_corpus,
    verify_structural_latent_training_fit_corpus,
    verify_structural_latent_training_validation_corpus,
)
from memory_condense.eval._diffuse_latent_training_corpus_codec import (
    decode_latent_training_payload,
    encode_latent_training_payload,
)
from memory_condense.eval._diffuse_latent_training_corpus_route import (
    live_route_v2_implementation_sha256,
    validate_persisted_route,
)
from memory_condense.eval import _diffuse_latent_training_corpus_io as corpus_io
from memory_condense.eval import _diffuse_latent_training_corpus_filesystem as corpus_fs
from memory_condense.domain.discourse import ClosureScopeWitness
from memory_condense.eval.diffuse_longmemeval_inputs import (
    GoldBlindLongMemEvalQuestion,
    _corpus_sha256,
)
from memory_condense.eval.diffuse_longmemeval_route_v2 import (
    EpisodePrimaryAnalysisArmV2,
    retrieve_episode_primary_analysis_phase_v2,
)


def _route(sample, destination):
    with route_fixture._new_condenser(destination) as condenser:
        route_fixture.ingest_gold_blind_sample_deterministically(condenser, sample)
        phase = retrieve_episode_primary_analysis_phase_v2(
            condenser,
            sample,
            config=route_fixture._config(),
            arm=EpisodePrimaryAnalysisArmV2(base_arm=route_fixture._base_arm()),
            legacy_input_provider=route_fixture._legacy_inputs,
            representative_linker=route_fixture._SelectEveryEpisodeLinker(),
            representative_policy_factory=route_fixture._representative_policy,
        )
    representative = phase.questions[0].inner.retrieval.representative_expansion
    assert representative is not None
    return StructuralRouteV2MappedRow(
        phase=phase,
        representative_policy=route_fixture._representative_policy(
            representative.artifact_id
        ),
    )


def _renamed_sample(question_id: str):
    sample = route_fixture._sample()
    question = GoldBlindLongMemEvalQuestion(
        question_id=question_id,
        retrieval_query=sample.questions[0].retrieval_query,
        prompt_question=sample.questions[0].prompt_question,
    )
    sample_id = f"corpus-{question_id}"
    return replace(
        sample,
        sample_id=sample_id,
        questions=(question,),
        corpus_sha256=_corpus_sha256(
            sample_id,
            sample.turns,
            sample.turn_source_ids,
            sample.turn_created_at,
        ),
    )


def _synthetic_population(question_ids):
    fit_ids = question_ids[:1]
    validation_ids = question_ids[1:]
    values = {
        "dataset_sha256": identity_sha256({"synthetic": "dataset"}),
        "split_manifest_sha256": identity_sha256({"synthetic": "split"}),
        "treatment_file_sha256": identity_sha256({"synthetic": "treatment"}),
        "sanitized_projection_sha256": identity_sha256(
            {"synthetic": "projection"}
        ),
        "excluded_confirmation_ordered_question_ids_sha256": identity_sha256(
            ["synthetic-confirmation"]
        ),
    }
    expected = LatentTrainingPopulationExpectation(
        **values,
        fit_count=1,
        fit_ordered_question_ids_sha256=identity_sha256(list(fit_ids)),
        validation_count=1,
        validation_ordered_question_ids_sha256=identity_sha256(
            list(validation_ids)
        ),
        analysis_ordered_question_ids_sha256=identity_sha256(list(question_ids)),
        excluded_confirmation_count=1,
    )
    projection = AnalysisPopulationProjection(
        **values,
        ordered_question_ids=question_ids,
        excluded_confirmation_count=1,
    )
    return projection, expected


@pytest.fixture(scope="module")
def published_corpus(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    route_fixture._patch_synthetic_owned_identity(monkeypatch)
    tmp_path = tmp_path_factory.mktemp("latent-training-corpus")
    question_ids = ("corpus-fit-q", "corpus-validation-q")
    mapped = {
        question_id: _route(
            _renamed_sample(question_id), tmp_path / f"route-{index}"
        )
        for index, question_id in enumerate(question_ids)
    }
    projection, expected = _synthetic_population(question_ids)
    destination = tmp_path / "corpus"
    receipt = _publish_synthetic_structural_latent_training_corpus(
        projection,
        destination,
        row_mapper=lambda row: mapped[row.question_id],
        expected=expected,
    )
    yield tmp_path, destination, receipt, projection, expected, mapped
    monkeypatch.undo()


def test_genuine_provider_free_two_row_publish_verify(published_corpus):
    tmp_path, destination, receipt, _, _, _ = published_corpus
    verified = verify_structural_latent_training_corpus(destination)
    assert receipt.corpus_sha256 == verified.manifest.corpus_sha256
    assert type(verified.fit) is VerifiedLatentTrainingFitCorpus
    assert type(verified.validation) is VerifiedLatentTrainingValidationCorpus
    assert verified.production_authorized is False
    assert verified.d1_eligible is False
    assert verify_structural_latent_training_fit_corpus(destination).rows == verified.fit.rows
    assert (
        verify_structural_latent_training_validation_corpus(destination).rows
        == verified.validation.rows
    )
    assert verified.fit.rows != verified.validation.rows
    assert not tuple(tmp_path.glob(".corpus.staging-*"))
    assert not tuple(
        path for path in tmp_path.glob("corpus*") if path.name != "corpus"
    )


def test_payload_codec_round_trip_is_exact(published_corpus):
    _, destination, _, _, _, _ = published_corpus
    verified = verify_structural_latent_training_corpus(destination)
    for decoded in (*verified.fit.rows, *verified.validation.rows):
        payload = decoded.payload
        encoded = encode_latent_training_payload(
            payload.retrieval_query,
            payload.plan,
            payload.packet,
            question_id=payload.question_id,
            prompt_question=payload.prompt_question,
        )
        assert decode_latent_training_payload(encoded) == payload
        assert encoded == (destination / decoded.manifest.payload_relative_path).read_bytes()


def test_payload_codec_rejects_trailing_and_duplicate_json(published_corpus):
    _, destination, _, _, _, _ = published_corpus
    verified = verify_structural_latent_training_corpus(destination)
    payload = (
        destination / verified.fit.rows[0].manifest.payload_relative_path
    ).read_bytes()
    with pytest.raises(ValueError, match="canonical"):
        decode_latent_training_payload(payload + b" ")
    duplicate = payload.replace(
        b'{"format":',
        b'{"format":"duplicate","format":',
        1,
    )
    with pytest.raises(ValueError, match="decode"):
        decode_latent_training_payload(duplicate)


def test_decoded_identity_mappings_are_deeply_immutable(published_corpus):
    _, destination, _, _, _, _ = published_corpus
    decoded = verify_structural_latent_training_corpus(destination).fit.rows[0]
    assert type(decoded.manifest.route_evidence.source_scope_body) is MappingProxyType
    with pytest.raises(TypeError):
        decoded.manifest.route_evidence.source_scope_body["artifact_id"] = "x"
    detail = decoded.payload.plan.scope_witnesses[0].detail
    assert type(detail) is MappingProxyType
    with pytest.raises(TypeError):
        detail["gold_answer"] = "x"


def test_open_mapping_firebreak_rejects_nested_output_channels(published_corpus):
    _, destination, _, _, _, _ = published_corpus
    decoded = verify_structural_latent_training_corpus(destination).fit.rows[0]
    plan = decoded.payload.plan
    witness = plan.scope_witnesses[0]
    malicious_witness = replace(
        witness,
        detail={"nested": {"gold_answer": "leak"}},
        witness_sha256="",
    )
    malicious_plan = replace(
        plan,
        scope_witnesses=(malicious_witness, *plan.scope_witnesses[1:]),
        plan_sha256="",
    )
    with pytest.raises(ValueError, match="forbidden output channel"):
        encode_latent_training_payload(
            decoded.payload.retrieval_query,
            malicious_plan,
            decoded.payload.packet,
            question_id=decoded.payload.question_id,
            prompt_question=decoded.payload.prompt_question,
        )
    malicious_receipt = replace(
        decoded.payload.packet.receipt,
        dropped_bundle_reasons={
            **decoded.payload.packet.receipt.dropped_bundle_reasons,
            "bundle-forged": "evaluator_score",
        },
        receipt_sha256="",
    )
    malicious_packet = replace(decoded.payload.packet, receipt=malicious_receipt)
    with pytest.raises(ValueError, match="forbidden output channel"):
        encode_latent_training_payload(
            decoded.payload.retrieval_query,
            plan,
            malicious_packet,
            question_id=decoded.payload.question_id,
            prompt_question=decoded.payload.prompt_question,
        )


def test_text_and_gold_blind_firebreak(published_corpus):
    _, destination, _, _, _, _ = published_corpus
    verified = verify_structural_latent_training_corpus(destination)
    metadata = b"\n".join(
        path.read_bytes()
        for path in sorted(destination.rglob("*.json"))
        if path.parent.name != "payloads"
    )
    prohibited_keys = {
        "gold", "gold_answer", "answer", "category", "prediction",
        "judge", "evaluator_score", "annotated_source_label",
    }
    for decoded in (*verified.fit.rows, *verified.validation.rows):
        payload = decoded.payload
        for text in (
            payload.retrieval_query,
            payload.prompt_question,
            *(atom.text for atom in payload.packet.atoms),
        ):
            assert text.encode("utf-8") not in metadata
        raw_payload = json.loads(
            (destination / decoded.manifest.payload_relative_path).read_text(
                encoding="utf-8"
            )
        )
        stack = [raw_payload]
        while stack:
            value = stack.pop()
            if type(value) is dict:
                assert prohibited_keys.isdisjoint(value)
                stack.extend(value.values())
            elif type(value) is list:
                stack.extend(value)
    assert verified.manifest.tensor_or_embedding_payload_present is False
    assert verified.manifest.scorer_labels_present is False
    assert verified.manifest.evaluator_label_schema_present is False


def test_population_failure_precedes_mapper_and_staging(tmp_path):
    projection, expected = _synthetic_population(("fit", "validation"))
    broken = replace(projection, ordered_question_ids=("validation", "fit"), projection_sha256="")
    calls = []
    destination = tmp_path / "population-firebreak"
    with pytest.raises(LatentTrainingCorpusError, match="before row mapping"):
        _publish_synthetic_structural_latent_training_corpus(
            broken,
            destination,
            row_mapper=lambda row: calls.append(row),
            expected=expected,
        )
    assert calls == []
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".population-firebreak.staging-*"))


def test_population_over_cap_precedes_mapper_and_filesystem(tmp_path):
    ids = tuple(f"q-{index}" for index in range(301))
    projection, expected = _synthetic_population(("fit", "validation"))
    values = {
        "dataset_sha256": expected.dataset_sha256,
        "split_manifest_sha256": expected.split_manifest_sha256,
        "treatment_file_sha256": expected.treatment_file_sha256,
        "sanitized_projection_sha256": expected.sanitized_projection_sha256,
        "excluded_confirmation_ordered_question_ids_sha256": expected.excluded_confirmation_ordered_question_ids_sha256,
    }
    oversized_projection = AnalysisPopulationProjection(
        **values, ordered_question_ids=ids, excluded_confirmation_count=1
    )
    oversized_expected = LatentTrainingPopulationExpectation(
        **values,
        fit_count=1,
        fit_ordered_question_ids_sha256=identity_sha256(list(ids[:1])),
        validation_count=300,
        validation_ordered_question_ids_sha256=identity_sha256(list(ids[1:])),
        analysis_ordered_question_ids_sha256=identity_sha256(list(ids)),
        excluded_confirmation_count=1,
    )
    calls = []
    destination = tmp_path / "oversized"
    with pytest.raises(LatentTrainingCorpusError, match="300-row"):
        _publish_synthetic_structural_latent_training_corpus(
            oversized_projection,
            destination,
            row_mapper=lambda row: calls.append(row),
            expected=oversized_expected,
        )
    assert not calls and not destination.exists()


def test_public_locked_population_binding_cannot_be_rebound_or_mutated(
    tmp_path, monkeypatch
):
    projection, _ = _synthetic_population(("fit", "validation"))
    calls = []
    original = corpus_api.LOCKED_LATENT_TRAINING_POPULATION
    monkeypatch.setattr(
        corpus_api,
        "LOCKED_LATENT_TRAINING_POPULATION",
        replace(original, fit_count=199),
    )
    with pytest.raises(RuntimeError, match="binding changed"):
        corpus_api.publish_structural_latent_training_corpus(
            projection,
            tmp_path / "rebound",
            row_mapper=lambda row: calls.append(row),
        )
    monkeypatch.setattr(corpus_api, "LOCKED_LATENT_TRAINING_POPULATION", original)
    object.__setattr__(original, "fit_count", 199)
    try:
        with pytest.raises(RuntimeError, match="value changed"):
            corpus_api.publish_structural_latent_training_corpus(
                projection,
                tmp_path / "mutated",
                row_mapper=lambda row: calls.append(row),
            )
    finally:
        object.__setattr__(original, "fit_count", 200)
    assert calls == []


def test_no_clobber_precedes_mapper(tmp_path, published_corpus):
    _, _, _, projection, expected, mapped = published_corpus
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    calls = []
    with pytest.raises(FileExistsError):
        _publish_synthetic_structural_latent_training_corpus(
            projection,
            destination,
            row_mapper=lambda row: calls.append(row) or mapped[row.question_id],
            expected=expected,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert calls == []


def test_partial_extra_and_symlink_packages_are_rejected(tmp_path, published_corpus):
    _, destination, _, _, _, _ = published_corpus
    partial = tmp_path / "partial"
    shutil.copytree(destination, partial)
    (partial / "manifest.json").unlink()
    with pytest.raises(LatentTrainingCorpusError):
        verify_structural_latent_training_corpus(partial)
    extra = tmp_path / "extra"
    shutil.copytree(destination, extra)
    (extra / "rows" / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(LatentTrainingCorpusError):
        verify_structural_latent_training_corpus(extra)
    link = tmp_path / "linked"
    try:
        os.symlink(destination, link, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    with pytest.raises(LatentTrainingCorpusError):
        verify_structural_latent_training_corpus(link)


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point regression")
def test_windows_junction_package_is_rejected(tmp_path, published_corpus):
    _, destination, _, _, _, _ = published_corpus
    junction = tmp_path / "junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(LatentTrainingCorpusError, match="reparse"):
        verify_structural_latent_training_corpus(junction)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_posix_fifo_is_rejected_without_blocking(tmp_path, published_corpus):
    _, destination, _, _, _, _ = published_corpus
    copied = tmp_path / "fifo"
    shutil.copytree(destination, copied)
    os.mkfifo(copied / "rows" / "999999.json")
    with pytest.raises(LatentTrainingCorpusError, match="filesystem type"):
        verify_structural_latent_training_corpus(copied)


def test_content_address_and_inventory_tamper_are_rejected(tmp_path, published_corpus):
    _, destination, _, _, _, _ = published_corpus
    tampered = tmp_path / "tampered"
    shutil.copytree(destination, tampered)
    verified = verify_structural_latent_training_corpus(tampered)
    payload_path = tampered / verified.fit.rows[0].manifest.payload_relative_path
    payload_path.write_bytes(payload_path.read_bytes() + b" ")
    with pytest.raises(LatentTrainingCorpusError, match="inventory"):
        verify_structural_latent_training_corpus(tampered)


def test_snapshot_detects_added_entry(published_corpus, tmp_path):
    _, destination, _, _, _, _ = published_corpus
    copied = tmp_path / "snapshot-added"
    shutil.copytree(destination, copied)
    with corpus_fs.CorpusTreeSnapshot(copied) as snapshot:
        (copied / "rows" / "999999.json").write_text("{}", encoding="utf-8")
        with pytest.raises(LatentTrainingCorpusError, match="entries changed"):
            snapshot.assert_unchanged()


def test_owned_capability_rejects_root_and_child_swaps(tmp_path):
    root_owned = corpus_fs.owned_staging(tmp_path, "owned")
    original = tmp_path / "moved-original"
    root_owned.path.rename(original)
    root_owned.path.mkdir()
    for name in ("partitions", "payloads", "rows"):
        (root_owned.path / name).mkdir()
    with pytest.raises(LatentTrainingCorpusError, match="replaced"):
        corpus_fs.write_new(root_owned, "rows/000000.json", b"{}")
    with pytest.raises(LatentTrainingCorpusError, match="replaced"):
        corpus_fs.remove_owned(root_owned)
    assert original.is_dir() and root_owned.path.is_dir()


def test_post_promotion_implementation_drift_rolls_back(
    tmp_path, published_corpus, monkeypatch
):
    _, _, _, projection, expected, mapped = published_corpus
    destination = tmp_path / "drift"
    promoted = False
    real_publish = corpus_io.publish_staging
    real_implementation = corpus_io.latent_training_corpus_implementation_sha256

    def promote(*args, **kwargs):
        nonlocal promoted
        result = real_publish(*args, **kwargs)
        promoted = True
        return result

    monkeypatch.setattr(corpus_io, "publish_staging", promote)
    monkeypatch.setattr(
        corpus_io,
        "latent_training_corpus_implementation_sha256",
        lambda: "0" * 64 if promoted else real_implementation(),
    )
    with pytest.raises(LatentTrainingCorpusError, match="implementation identity"):
        corpus_io.publish_structural_corpus(
            projection,
            destination,
            row_mapper=lambda row: mapped[row.question_id],
            expected=expected,
            population_status="synthetic_projection",
        )
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".drift.staging-*"))


def test_exact_number_alias_and_nested_route_tamper_are_rejected(published_corpus):
    _, destination, _, _, _, _ = published_corpus
    verified = verify_structural_latent_training_corpus(destination)
    decoded = verified.fit.rows[0]
    evidence = decoded.manifest.route_evidence
    diffuse_body = dict(evidence.inner_diffuse_query_receipt_body)
    diffuse_body["prompt_token_proxy"] = float(diffuse_body["prompt_token_proxy"])
    forged_evidence = replace(
        evidence,
        inner_diffuse_query_receipt_body=diffuse_body,
        evidence_sha256="",
    )
    forged_row = replace(
        decoded.manifest, route_evidence=forged_evidence, row_sha256=""
    )
    with pytest.raises((TypeError, LatentTrainingCorpusError)):
        validate_persisted_route(
            forged_row,
            decoded.payload,
            expected_route_implementation_sha256=live_route_v2_implementation_sha256(),
        )

    for owner, field, value in (
        (decoded.payload.plan, "artifact_id", "tampered-artifact"),
        (decoded.payload.packet.receipt, "plan_sha256", "0" * 64),
        (
            decoded.manifest.structural_target.structural_targets,
            "atom_count",
            decoded.manifest.structural_target.structural_targets.atom_count + 1,
        ),
    ):
        original = getattr(owner, field)
        object.__setattr__(owner, field, value)
        try:
            with pytest.raises((TypeError, ValueError, LatentTrainingCorpusError)):
                validate_persisted_route(
                    decoded.manifest,
                    decoded.payload,
                    expected_route_implementation_sha256=live_route_v2_implementation_sha256(),
                )
        finally:
            object.__setattr__(owner, field, original)

    original_artifact = decoded.payload.plan.artifact_id
    object.__setattr__(decoded.payload.plan, "artifact_id", "tampered-artifact")
    try:
        with pytest.raises((ValueError, LatentTrainingCorpusError)):
            VerifiedLatentTrainingFitCorpus(
                verified.manifest,
                verified.fit.partition,
                verified.fit.rows,
            )
    finally:
        object.__setattr__(decoded.payload.plan, "artifact_id", original_artifact)


def test_authority_flags_cannot_be_constructed_or_resealed(published_corpus):
    _, destination, _, _, _, _ = published_corpus
    verified = verify_structural_latent_training_corpus(destination)
    with pytest.raises(ValueError, match="authorize"):
        replace(verified.fit.partition, production_authorized=True, partition_sha256="")
    with pytest.raises(ValueError, match="authority"):
        replace(verified.manifest, d1_eligible=True, corpus_sha256="")

    manifest = verified.manifest
    object.__setattr__(manifest, "production_authorized", True)
    object.__setattr__(manifest, "corpus_sha256", identity_sha256(manifest.identity_payload(include_receipt=False)))
    try:
        with pytest.raises(ValueError, match="authority"):
            VerifiedLatentTrainingFitCorpus(manifest, verified.fit.partition, verified.fit.rows)
    finally:
        object.__setattr__(manifest, "production_authorized", False)
        object.__setattr__(manifest, "corpus_sha256", identity_sha256(manifest.identity_payload(include_receipt=False)))
