from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory_condense.eval.recall_guarded_cumulative_1m_source import (
    source_acquisition_config,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)

from tools import confirmation_namespace_store_adapter as adapter
from tools import plan_confirmation_treatment_pipeline as planner
from tools.confirmation_canonical import (
    canonical_json_bytes,
    canonical_sha256,
)
from tools.confirmation_treatment import (
    ConfirmationTreatmentInput,
    TreatmentQuestion,
    TreatmentSample,
)


def _samples(token_counts: tuple[int, ...]) -> tuple[TreatmentSample, ...]:
    rows: list[TreatmentSample] = []
    for index, token_count in enumerate(token_counts):
        question_id = f"synthetic-{index}"
        rows.append(
            TreatmentSample(
                sample_id=question_id,
                turns=(("user", " ".join([f"memory-{index}"] * token_count)),),
                turn_source_ids=(f"session-{index}",),
                turn_created_at=(
                    datetime(2026, 1, index + 1, tzinfo=timezone.utc),
                ),
                questions=(
                    TreatmentQuestion(
                        question_id=question_id,
                        question=f"What is memory {index}?",
                        question_date=f"2026-02-{index + 1:02d}T00:00:00Z",
                    ),
                ),
            )
        )
    return tuple(rows)


def _sample_projection(sample: TreatmentSample) -> dict[str, object]:
    question = sample.questions[0]
    return {
        "sample_id": sample.sample_id,
        "turns": [list(turn) for turn in sample.turns],
        "turn_source_ids": list(sample.turn_source_ids),
        "turn_created_at": [
            value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if value is not None
            else None
            for value in sample.turn_created_at
        ],
        "questions": [
            {
                "question_id": question.question_id,
                "question": question.question,
                "question_date": question.question_date,
            }
        ],
    }


def _treatment(
    samples: tuple[TreatmentSample, ...], *, tag: str
) -> ConfirmationTreatmentInput:
    projections = [_sample_projection(sample) for sample in samples]
    return ConfirmationTreatmentInput(
        file_sha256=canonical_sha256({"file": tag, "samples": projections}),
        sanitized_projection_sha256=canonical_sha256(projections),
        dataset_sha256=canonical_sha256({"dataset": tag}),
        split_manifest_sha256=canonical_sha256({"split": tag}),
        ordered_question_ids_sha256=canonical_sha256(
            [sample.sample_id for sample in samples]
        ),
        ordered_normalized_sample_bindings_sha256=canonical_sha256(
            {"normalized": tag, "samples": projections}
        ),
        ordered_raw_record_bindings_sha256=canonical_sha256(
            {"raw": tag, "samples": projections}
        ),
        samples=samples,
    )


def _sealed_plan(
    treatment: ConfirmationTreatmentInput,
    sizes: tuple[int, ...],
) -> planner.SealedConfirmationPipelinePlan:
    payload = planner.compile_confirmation_pipeline_preflight(
        treatment, namespace_sizes=sizes
    )
    raw = canonical_json_bytes(payload) + b"\n"
    return planner.SealedConfirmationPipelinePlan(
        Path("synthetic-preflight.json"), hashlib.sha256(raw).hexdigest(), payload
    )


def _sealed_freeze(treatment: ConfirmationTreatmentInput) -> adapter.SealedPayload:
    confirmation = {
        "ordered_normalized_sample_bindings_sha256": (
            treatment.ordered_normalized_sample_bindings_sha256
        ),
        "ordered_question_ids_sha256": treatment.ordered_question_ids_sha256,
        "ordered_raw_record_bindings_sha256": (
            treatment.ordered_raw_record_bindings_sha256
        ),
    }
    static = {
        "dataset_sha256": treatment.dataset_sha256,
        **confirmation,
        "sample_count": len(treatment.samples),
        "split_manifest_sha256": treatment.split_manifest_sha256,
    }
    source_sha256 = canonical_sha256(
        {"source_policy": treatment.ordered_question_ids_sha256}
    )
    treatment_policy = {
        "confirmation_guards": dict(adapter._REQUIRED_CONFIRMATION_GUARDS),
        "confirmation_population_static_root": static,
    }
    body = {
        "format": adapter.FREEZE_FORMAT,
        "source_policy_manifest_sha256": source_sha256,
        "status": adapter.FREEZE_STATUS,
        "treatment_policy": treatment_policy,
        "treatment_projection_sha256": canonical_sha256(treatment_policy),
    }
    payload = {**body, "runtime_policy_identity_sha256": canonical_sha256(body)}
    raw = canonical_json_bytes(payload) + b"\n"
    return adapter.SealedPayload(
        Path("synthetic-runtime-policy.json"),
        hashlib.sha256(raw).hexdigest(),
        payload,
    )


def _compile(
    treatment: ConfirmationTreatmentInput,
    sizes: tuple[int, ...],
    *,
    target_tokens: int,
) -> tuple[
    planner.SealedConfirmationPipelinePlan,
    adapter.ConfirmationNamespaceWorkset,
]:
    plan = _sealed_plan(treatment, sizes)
    return plan, adapter.compile_confirmation_namespace_workset(
        treatment,
        preflight=plan,
        freeze=_sealed_freeze(treatment),
        target_tokens=target_tokens,
        token_counter=lambda text: len(text.split()),
    )


def test_suffix_haystack_and_probe_populations_are_sealed_separately() -> None:
    treatment = _treatment(_samples((2, 2, 2, 3)), tag="suffix")
    _plan, workset = _compile(treatment, (2, 2), target_tokens=5)

    first, second = workset.namespaces
    assert [probe.question_id for probe in first.probes] == [
        "synthetic-0",
        "synthetic-1",
    ]
    assert len(first.haystack) == 3
    assert first.actual_tokens == 6
    assert [probe.question_id for probe in second.probes] == [
        "synthetic-2",
        "synthetic-3",
    ]
    assert len(second.haystack) == 2
    assert second.actual_tokens == 5
    assert first.haystack[-1].member_key_sha256 == second.haystack[0].member_key_sha256
    assert first.body()["probe_membership_sha256"] != first.body()[
        "haystack_membership_sha256"
    ]
    assert workset.body()["physical_provider_calls"] == 0
    assert workset.body()["gold_loaded"] is False


def test_target_reached_before_probe_block_fails_instead_of_changing_history_rule() -> None:
    treatment = _treatment(_samples((10, 1, 10, 1)), tag="early-stop")
    with pytest.raises(
        adapter.ConfirmationNamespaceError,
        match="before the declared probe block",
    ):
        _compile(treatment, (2, 2), target_tokens=5)


def test_renumbering_changes_probe_coordinates_not_store_or_corpus_identity() -> None:
    samples = _samples((3, 3, 3, 3))
    treatment = _treatment(samples, tag="original")
    renamed_samples = tuple(
        replace(
            sample,
            sample_id=f"foreign-{index}",
            questions=(
                replace(sample.questions[0], question_id=f"foreign-{index}"),
            ),
        )
        for index, sample in enumerate(samples)
    )
    renamed = _treatment(renamed_samples, tag="renamed")
    plan_a, workset_a = _compile(treatment, (2, 2), target_tokens=5)
    plan_b, workset_b = _compile(renamed, (2, 2), target_tokens=5)

    assert [row.namespace_store_id for row in workset_a.namespaces] == [
        row.namespace_store_id for row in workset_b.namespaces
    ]
    assert [
        [member.member_key_sha256 for member in row.haystack]
        for row in workset_a.namespaces
    ] == [
        [member.member_key_sha256 for member in row.haystack]
        for row in workset_b.namespaces
    ]
    assert [row.probes for row in workset_a.namespaces] != [
        row.probes for row in workset_b.namespaces
    ]
    for work_a, work_b in zip(
        workset_a.namespaces, workset_b.namespaces, strict=True
    ):
        sample_a = adapter.build_namespace_sample(treatment, plan_a, work_a)
        sample_b = adapter.build_namespace_sample(renamed, plan_b, work_b)
        assert sample_a.corpus_sha256 == sample_b.corpus_sha256


def test_block_permutation_and_growth_do_not_rewrite_store_identities() -> None:
    samples = _samples((3, 3, 3, 3, 3, 3))
    base = _treatment(samples[:4], tag="base")
    expanded = _treatment(samples, tag="expanded")
    permuted = _treatment(samples[2:4] + samples[0:2], tag="permuted")
    _base_plan, base_work = _compile(base, (2, 2), target_tokens=5)
    _expanded_plan, expanded_work = _compile(
        expanded, (2, 2, 2), target_tokens=5
    )
    _permuted_plan, permuted_work = _compile(
        permuted, (2, 2), target_tokens=5
    )

    assert [row.namespace_store_id for row in base_work.namespaces] == [
        row.namespace_store_id for row in expanded_work.namespaces[:2]
    ]
    assert sorted(row.namespace_store_id for row in base_work.namespaces) == sorted(
        row.namespace_store_id for row in permuted_work.namespaces
    )


class _FakeBackend:
    def __init__(self) -> None:
        self.executed: list[adapter.NamespaceExecutionRequest] = []
        self.verified: list[adapter.NamespaceExecutionRequest] = []
        self._identity = canonical_sha256({"backend": "synthetic"})

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @staticmethod
    def _result(
        request: adapter.NamespaceExecutionRequest,
    ) -> adapter.NamespaceBackendResult:
        work = request.work
        return adapter.NamespaceBackendResult(
            namespace_id=work.namespace_id,
            namespace_store_id=work.namespace_store_id,
            store_receipt_sha256=canonical_sha256(
                {"store": work.namespace_store_id}
            ),
            index_receipt_sha256=canonical_sha256(
                {"index": work.namespace_store_id}
            ),
            query_artifact_receipt_sha256=canonical_sha256(
                {"queries": [probe.row_receipt_sha256 for probe in work.probes]}
            ),
            artifact_projection={
                "local_store_key": work.namespace_store_id,
                "retained_request_token_state_bytes": 0,
            },
            questions=tuple(
                adapter.QuestionRetrievalBinding(
                    question_id=probe.question_id,
                    row_receipt_sha256=probe.row_receipt_sha256,
                    retrieval_receipt_sha256=canonical_sha256(
                        {
                            "namespace_id": work.namespace_id,
                            "question": probe.row_receipt_sha256,
                        }
                    ),
                )
                for probe in work.probes
            ),
        )

    def execute(
        self, request: adapter.NamespaceExecutionRequest
    ) -> adapter.NamespaceBackendResult:
        self.executed.append(request)
        return self._result(request)

    def verify(
        self,
        request: adapter.NamespaceExecutionRequest,
        expected: object,
    ) -> adapter.NamespaceBackendResult:
        self.verified.append(request)
        result = self._result(request)
        assert result.projection() == expected
        return result


def _frozen_bge_identity(device: str) -> dict[str, object]:
    return {
        "backend": "sentence-transformers.encode-v1",
        "model_id": DEFAULT_MODEL_NAME,
        "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "device": device,
        "batch_size": 32,
        "normalize_embeddings": False,
        "output_dtype": "float32",
    }


def test_production_source_contract_rejects_packed_policy_and_wrong_bge() -> None:
    full = EvalConfig(
        embedding_device="cpu",
        max_prompt_tokens=8192,
        retrieval=RetrievalConfig(mode="causal_graph"),
    )
    with pytest.raises(
        adapter.ConfirmationNamespaceError,
        match="source_acquisition_config",
    ):
        adapter.build_production_source_treatment_contract(
            full, _frozen_bge_identity("cpu")
        )

    source = source_acquisition_config(full)
    wrong = {**_frozen_bge_identity("cpu"), "batch_size": 31}
    with pytest.raises(
        adapter.ConfirmationNamespaceError,
        match="frozen BGE-M3",
    ):
        adapter.build_production_source_treatment_contract(source, wrong)

    contract = adapter.build_production_source_treatment_contract(
        source, _frozen_bge_identity("cpu")
    )
    assert contract["coordinate_semantics"] == adapter.SOURCE_COORDINATE_SEMANTICS
    assert contract["historical_coordinate_or_byte_identity"] is False
    assert contract["frozen_current_source_equivalence"] == (
        adapter.FROZEN_CURRENT_SOURCE_EQUIVALENCE
    )


def test_checkpoint_resume_and_question_namespace_isolation(tmp_path: Path) -> None:
    treatment = _treatment(_samples((2, 2, 2, 3)), tag="execute")
    plan, workset = _compile(treatment, (2, 2), target_tokens=5)
    backend = _FakeBackend()

    first = adapter.execute_confirmation_namespaces(
        treatment,
        preflight=plan,
        workset=workset,
        output_root=tmp_path,
        backend=backend,
    )
    assert (first.created_count, first.reused_count, first.physical_provider_calls) == (
        2,
        0,
        0,
    )
    assert len(backend.executed) == 2
    assert [item.question_id for item in backend.executed[0].probes] == [
        "synthetic-0",
        "synthetic-1",
    ]
    assert [item.question_id for item in backend.executed[1].probes] == [
        "synthetic-2",
        "synthetic-3",
    ]
    assert len(backend.executed[0].sample.turns) == 3
    assert len(backend.executed[1].sample.turns) == 2
    assert all(
        source is None or source.startswith(member.member_key_sha256)
        for request in backend.executed
        for member, source in zip(
            request.work.haystack,
            request.sample.turn_source_ids,
            strict=True,
        )
    )

    second = adapter.execute_confirmation_namespaces(
        treatment,
        preflight=plan,
        workset=workset,
        output_root=tmp_path,
        backend=backend,
    )
    assert (second.created_count, second.reused_count) == (0, 2)
    assert len(backend.executed) == 2
    assert len(backend.verified) == 2
    assert first.checkpoint_sha256s == second.checkpoint_sha256s


def test_backend_cannot_return_a_question_from_another_namespace(tmp_path: Path) -> None:
    treatment = _treatment(_samples((3, 3, 3, 3)), tag="escape")
    plan, workset = _compile(treatment, (2, 2), target_tokens=5)

    class EscapingBackend(_FakeBackend):
        def execute(
            self, request: adapter.NamespaceExecutionRequest
        ) -> adapter.NamespaceBackendResult:
            result = super().execute(request)
            foreign = workset.namespaces[1].probes[0]
            return replace(
                result,
                questions=(
                    adapter.QuestionRetrievalBinding(
                        question_id=foreign.question_id,
                        row_receipt_sha256=foreign.row_receipt_sha256,
                        retrieval_receipt_sha256=canonical_sha256(
                            {"foreign": foreign.row_receipt_sha256}
                        ),
                    ),
                    *result.questions[1:],
                ),
            )

    with pytest.raises(adapter.ConfirmationNamespaceError, match="probe membership"):
        adapter.execute_confirmation_namespaces(
            treatment,
            preflight=plan,
            workset=workset,
            output_root=tmp_path,
            backend=EscapingBackend(),
        )
    assert not (tmp_path / "checkpoints").exists()


def test_freeze_population_mismatch_fails_before_namespace_compilation() -> None:
    treatment = _treatment(_samples((3, 3)), tag="freeze")
    plan = _sealed_plan(treatment, (2,))
    freeze = _sealed_freeze(treatment)
    changed = dict(freeze.payload)
    policy = dict(changed["treatment_policy"])
    static = dict(policy["confirmation_population_static_root"])
    static["dataset_sha256"] = "0" * 64
    policy["confirmation_population_static_root"] = static
    changed["treatment_policy"] = policy
    changed["treatment_projection_sha256"] = canonical_sha256(policy)
    body = {
        key: value
        for key, value in changed.items()
        if key != "runtime_policy_identity_sha256"
    }
    changed["runtime_policy_identity_sha256"] = canonical_sha256(body)
    bad_raw = canonical_json_bytes(changed) + b"\n"
    bad = adapter.SealedPayload(
        freeze.path, hashlib.sha256(bad_raw).hexdigest(), changed
    )

    with pytest.raises(
        adapter.ConfirmationNamespaceError,
        match="static root changed",
    ):
        adapter.compile_confirmation_namespace_workset(
            treatment,
            preflight=plan,
            freeze=bad,
            target_tokens=5,
            token_counter=lambda text: len(text.split()),
        )
