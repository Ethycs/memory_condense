from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import (
    identity_sha256 as runtime_identity_sha256,
    quote_sha256,
)
from memory_condense.eval.benchmark import build_qa_prompt
from memory_condense.eval.recall_guarded_cumulative import (
    CausalCoveragePredecessorReceipt,
    CumulativeRetrievalLadder,
    CumulativeRetrievalStageReceipt,
    RecallGuardedCumulativeReceipt,
)
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    CombinedCumulativeStoreReceipt,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from tools import confirmation_cumulative_retrieval as cumulative
from tools import confirmation_namespace_store_adapter as base_adapter
from tools import confirmation_semantic_planes as semantic_planes
from tools import confirmation_staged_cumulative_coordinator as staged
from tools import plan_confirmation_treatment_pipeline as planner
from tools.matched_eval.renderer import V4_RENDERER_ID
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
    treatment: ConfirmationTreatmentInput, sizes: tuple[int, ...]
) -> planner.SealedConfirmationPipelinePlan:
    payload = planner.compile_confirmation_pipeline_preflight(
        treatment, namespace_sizes=sizes
    )
    raw = canonical_json_bytes(payload) + b"\n"
    return planner.SealedConfirmationPipelinePlan(
        Path("synthetic-preflight.json"), hashlib.sha256(raw).hexdigest(), payload
    )


def _sealed_freeze(
    treatment: ConfirmationTreatmentInput,
) -> base_adapter.SealedPayload:
    confirmation = {
        "count": len(treatment.samples),
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
        **{key: value for key, value in confirmation.items() if key != "count"},
        "sample_count": len(treatment.samples),
        "split_manifest_sha256": treatment.split_manifest_sha256,
    }
    treatment_policy = {
        "confirmation_guards": dict(
            base_adapter._REQUIRED_CONFIRMATION_GUARDS
        ),
        "confirmation_population_static_root": static,
    }
    body = {
        "format": base_adapter.FREEZE_FORMAT,
        "source_policy_manifest_sha256": canonical_sha256(
            {"source_policy": treatment.ordered_question_ids_sha256}
        ),
        "status": base_adapter.FREEZE_STATUS,
        "treatment_policy": treatment_policy,
        "treatment_projection_sha256": canonical_sha256(treatment_policy),
    }
    payload = {
        **body,
        "runtime_policy_identity_sha256": canonical_sha256(body),
    }
    raw = canonical_json_bytes(payload) + b"\n"
    return base_adapter.SealedPayload(
        Path("synthetic-runtime-policy.json"),
        hashlib.sha256(raw).hexdigest(),
        payload,
    )


_SYNTHETIC_EMBEDDING_IDENTITY = {"backend": "synthetic-bge"}


def _synthetic_source_contract() -> dict[str, object]:
    body: dict[str, object] = {
        "coordinate_semantics": base_adapter.SOURCE_COORDINATE_SEMANTICS,
        "embedding_identity_sha256": runtime_identity_sha256(
            _SYNTHETIC_EMBEDDING_IDENTITY
        ),
        "format": base_adapter.SOURCE_TREATMENT_CONTRACT_FORMAT,
        "frozen_current_source_equivalence": (
            base_adapter.FROZEN_CURRENT_SOURCE_EQUIVALENCE
        ),
        "historical_coordinate_or_byte_identity": False,
        "source_acquisition_config_sha256": canonical_sha256(
            {"source_config": "synthetic"}
        ),
        "source_retrieval_policy_sha256": canonical_sha256(
            {"source_retrieval": "synthetic-dense"}
        ),
        "source_scope": "synthetic-gold-blind-source",
        "timestamp_semantics": "synthetic-exact-timestamps",
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


class _BaseBackend:
    def __init__(self) -> None:
        self._identity = canonical_sha256({"backend": "synthetic-base"})

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @staticmethod
    def _result(
        request: base_adapter.NamespaceExecutionRequest,
    ) -> base_adapter.NamespaceBackendResult:
        work = request.work
        database = request.namespace_root / "synthetic-store" / "memory.db"
        return base_adapter.NamespaceBackendResult(
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
                "database_relative_path": "synthetic-store/memory.db",
                "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
                "local_store_key": work.namespace_store_id,
                "retained_request_token_state_bytes": 0,
                "source_treatment_contract": _synthetic_source_contract(),
            },
            questions=tuple(
                base_adapter.QuestionRetrievalBinding(
                    question_id=probe.question_id,
                    row_receipt_sha256=probe.row_receipt_sha256,
                    retrieval_receipt_sha256=canonical_sha256(
                        {"base_retrieval": probe.row_receipt_sha256}
                    ),
                )
                for probe in work.probes
            ),
        )

    def execute(
        self, request: base_adapter.NamespaceExecutionRequest
    ) -> base_adapter.NamespaceBackendResult:
        database = request.namespace_root / "synthetic-store" / "memory.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        if not database.exists():
            database.write_bytes(
                f"synthetic:{request.work.namespace_store_id}".encode("ascii")
            )
        return self._result(request)

    def verify(
        self,
        request: base_adapter.NamespaceExecutionRequest,
        expected: object,
    ) -> base_adapter.NamespaceBackendResult:
        assert (request.namespace_root / "synthetic-store" / "memory.db").is_file()
        result = self._result(request)
        assert result.projection() == expected
        return result


def _make_inputs(
    root: Path,
    treatment: ConfirmationTreatmentInput,
    sizes: tuple[int, ...],
    *,
    target_tokens: int,
) -> cumulative.ConfirmationCumulativeInput:
    plan = _sealed_plan(treatment, sizes)
    freeze = _sealed_freeze(treatment)
    workset = base_adapter.compile_confirmation_namespace_workset(
        treatment,
        preflight=plan,
        freeze=freeze,
        target_tokens=target_tokens,
        token_counter=lambda text: len(text.split()),
    )
    base = base_adapter.execute_confirmation_namespaces(
        treatment,
        preflight=plan,
        workset=workset,
        output_root=root / "base",
        backend=_BaseBackend(),
    )
    return cumulative.ConfirmationCumulativeInput(
        treatment=treatment,
        preflight=plan,
        policy_freeze=freeze,
        workset=workset,
        base_execution=base,
    )


def _stage_context(evidence: tuple[cumulative.CumulativeEvidence, ...]) -> str:
    return "\n".join(
        f"[{index}] {item.text}" for index, item in enumerate(evidence, start=1)
    )


class _CumulativeBackend:
    def __init__(self, freeze_sha256: str) -> None:
        self._freeze_sha256 = freeze_sha256
        self._policy_sha256 = runtime_identity_sha256(
            {"retrieval": "synthetic-population-neutral"}
        )
        self._identity = canonical_sha256(
            {
                "backend": "synthetic-cumulative",
                "freeze_sha256": freeze_sha256,
                "retrieval_policy_sha256": self._policy_sha256,
            }
        )
        self.executed: list[cumulative.CumulativeNamespaceRequest] = []
        self.verified: list[cumulative.CumulativeNamespaceRequest] = []

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def policy_freeze_sha256(self) -> str:
        return self._freeze_sha256

    def _combined(
        self, request: cumulative.CumulativeNamespaceRequest
    ) -> CombinedCumulativeStoreReceipt:
        store_identity = runtime_identity_sha256(
            {"store": request.work.namespace_store_id}
        )
        compilation = runtime_identity_sha256(
            {"compilation": request.work.namespace_store_id}
        )
        return CombinedCumulativeStoreReceipt(
            source_store_identity_sha256=store_identity,
            target_store_identity_sha256=store_identity,
            source_database_sha256=runtime_identity_sha256(
                {"source_database": request.work.namespace_store_id}
            ),
            target_database_sha256=runtime_identity_sha256(
                {"target_database": request.work.namespace_store_id}
            ),
            target_index_sha256=runtime_identity_sha256(
                {"target_index": request.work.namespace_store_id}
            ),
            retrieval_policy_sha256=self._policy_sha256,
            context_budget_sha256=runtime_identity_sha256(
                {"context_budget": "synthetic"}
            ),
            training_query_batch_sha256=runtime_identity_sha256(
                {"training": request.work.namespace_store_id}
            ),
            held_out_query_batch_sha256=runtime_identity_sha256(
                {"held_out": [query.question for query in request.queries]}
            ),
            compilation_receipt_sha256=compilation,
            artifact_id=f"artifact-{request.work.namespace_store_id}",
            snapshot_sha256=runtime_identity_sha256(
                {"snapshot": request.work.namespace_store_id}
            ),
            turn_count=len(request.work.haystack),
            chunk_count=len(request.work.haystack),
            causal_events=0,
            causal_graph_edges=0,
        )

    def _question(
        self,
        request: cumulative.CumulativeNamespaceRequest,
        query: cumulative.ConfirmationQuery,
    ) -> cumulative.CumulativeQuestion:
        def evidence(kind: str) -> cumulative.CumulativeEvidence:
            return cumulative.CumulativeEvidence(
                evidence_id=runtime_identity_sha256(
                    {
                        "kind": kind,
                        "namespace_store_id": request.work.namespace_store_id,
                        "question": query.question,
                    }
                ),
                source_id=f"{request.work.namespace_store_id}:{kind}",
                text=f"{kind} evidence for {query.question}",
            )

        root = evidence("root")
        direct = evidence("direct")
        global_evidence = evidence("global")
        evidence_by_stage = (
            (root,),
            (root, direct),
            (root, direct),
            (root, direct, global_evidence),
        )
        matched = runtime_identity_sha256(
            {
                "matched": query.question,
                "store": request.work.namespace_store_id,
            }
        )
        max_context = 50_000
        max_prompt = 50_000
        reserve = 64
        receipts: list[CumulativeRetrievalStageReceipt] = []
        stages: list[cumulative.CumulativeStage] = []
        for position, (stage_id, stage_evidence) in enumerate(
            zip(cumulative.STAGE_IDS, evidence_by_stage, strict=True)
        ):
            context = _stage_context(stage_evidence)
            messages = build_qa_prompt(
                query.dated_question, [item.text for item in stage_evidence]
            )
            ids = tuple(item.evidence_id for item in stage_evidence)
            parent_ids = () if position == 0 else receipts[-1].selected_evidence_ids
            added = ids[len(parent_ids) :]
            receipt = CumulativeRetrievalStageReceipt(
                stage_id=stage_id,
                matched_controls_sha256=matched,
                method_evidence_sha256=runtime_identity_sha256(
                    {"method": stage_id, "ids": list(ids)}
                ),
                parent_stage_receipt_sha256=(
                    None if position == 0 else receipts[-1].receipt_sha256
                ),
                parent_evidence_ids=parent_ids,
                selected_evidence_ids=ids,
                added_evidence_ids=added,
                admission_status=(
                    "root"
                    if position == 0
                    else ("added" if added else "no_novel_evidence")
                ),
                evidence_projection_sha256=runtime_identity_sha256(
                    {"evidence": list(ids)}
                ),
                context_sha256=quote_sha256(context),
                prompt_messages_sha256=runtime_identity_sha256(messages),
                context_token_proxy=count_tokens(context),
                max_context_token_proxy=max_context,
                prompt_token_proxy=count_chat_prompt_token_proxy(messages),
                max_prompt_token_proxy=max_prompt,
                responder_output_token_reserve=reserve,
            )
            receipts.append(receipt)
            stages.append(
                cumulative.CumulativeStage(
                    stage_id=stage_id,
                    stage_receipt=asdict(receipt),
                    provider_messages=tuple(messages),
                    evidence=stage_evidence,
                )
            )

        ladder = CumulativeRetrievalLadder(stages=tuple(receipts))
        root_messages = list(stages[0].provider_messages)
        predecessor = CausalCoveragePredecessorReceipt(
            matched_controls_sha256=matched,
            retrieval_query_sha256=runtime_identity_sha256(
                {"query": query.dated_question}
            ),
            prompt_question_sha256=runtime_identity_sha256(
                {"prompt_question": query.dated_question}
            ),
            retrieval_policy_sha256=self._policy_sha256,
            context_budget_sha256=runtime_identity_sha256(
                {"context_budget": "synthetic"}
            ),
            raw_graph_anchor_sequence_sha256=runtime_identity_sha256(
                {"raw_graph": root.evidence_id}
            ),
            raw_graph_chunk_ids=(f"chunk-{root.evidence_id}",),
            packed_chunk_ids=(f"chunk-{root.evidence_id}",),
            protected_chunk_ids=(f"chunk-{root.evidence_id}",),
            direct_protected_chunk_ids=(f"chunk-{root.evidence_id}",),
            protected_excerpt_projection_sha256=runtime_identity_sha256(
                {"protected": root.evidence_id}
            ),
            protected_context_sha256=quote_sha256(_stage_context((root,))),
            selected_anchor_sequence_sha256=runtime_identity_sha256(
                {"selected": root.evidence_id}
            ),
            coverage_selector_report_sha256=runtime_identity_sha256(
                {"coverage": root.evidence_id}
            ),
            coverage_candidate_trace_sha256=runtime_identity_sha256(
                {"trace": root.evidence_id}
            ),
            coverage_runtime_certified=True,
            packed_token_counts=(),
            packed_dropped_counts=(),
            prompt_messages_sha256=runtime_identity_sha256(root_messages),
            prompt_token_proxy=count_chat_prompt_token_proxy(root_messages),
            max_prompt_token_proxy=max_prompt,
            responder_output_token_reserve=reserve,
        )
        added_ids = (direct.evidence_id, global_evidence.evidence_id)
        final_stage = receipts[-1]
        final = RecallGuardedCumulativeReceipt(
            matched_controls_sha256=matched,
            predecessor_receipt_sha256=predecessor.receipt_sha256,
            direct_expansion_receipt_sha256=runtime_identity_sha256(
                {"direct_expansion": query.question}
            ),
            representative_expansion_receipt_sha256=runtime_identity_sha256(
                {"representative_expansion": query.question}
            ),
            closure_plan_sha256s=tuple(
                runtime_identity_sha256({"closure": stage_id, "q": query.question})
                for stage_id in cumulative.STAGE_IDS[1:]
            ),
            novel_projection_receipt_sha256s=tuple(
                runtime_identity_sha256({"novel": stage_id, "q": query.question})
                for stage_id in cumulative.STAGE_IDS[1:]
            ),
            addition_packet_receipt_sha256s=(
                runtime_identity_sha256({"packet": "direct", "q": query.question}),
                None,
                runtime_identity_sha256({"packet": "global", "q": query.question}),
            ),
            stage_admission_statuses=tuple(
                receipt.admission_status for receipt in receipts[1:]
            ),
            ladder_receipt_sha256=ladder.receipt_sha256,
            representative_runtime_certified=True,
            protected_chunk_ids=predecessor.protected_chunk_ids,
            protected_evidence_ids=(root.evidence_id,),
            added_atom_ids=added_ids,
            added_chunk_ids=(
                f"chunk-{direct.evidence_id}",
                f"chunk-{global_evidence.evidence_id}",
            ),
            final_chunk_ids=(
                *predecessor.protected_chunk_ids,
                f"chunk-{direct.evidence_id}",
                f"chunk-{global_evidence.evidence_id}",
            ),
            final_evidence_ids=final_stage.selected_evidence_ids,
            protected_excerpt_projection_sha256=(
                predecessor.protected_excerpt_projection_sha256
            ),
            addition_evidence_projection_sha256=runtime_identity_sha256(
                {"added": list(added_ids)}
            ),
            final_context_sha256=final_stage.context_sha256,
            prompt_messages_sha256=final_stage.prompt_messages_sha256,
            context_token_proxy=final_stage.context_token_proxy,
            max_context_token_proxy=max_context,
            prompt_token_proxy=final_stage.prompt_token_proxy,
            max_prompt_token_proxy=max_prompt,
            responder_output_token_reserve=reserve,
            prompt_workspace_token_proxy=final_stage.prompt_token_proxy + reserve,
        )
        return cumulative.CumulativeQuestion(
            question_id=query.question_id,
            row_receipt_sha256=query.row_receipt_sha256,
            content_binding_sha256=query.content_binding_sha256,
            question=query.question,
            dated_question=query.dated_question,
            base_retrieval_receipt_sha256=(
                request.base.retrieval_receipts_by_row[query.row_receipt_sha256]
            ),
            predecessor_receipt=asdict(predecessor),
            retrieval_receipt=asdict(final),
            stages=tuple(stages),
        )

    def _result(
        self, request: cumulative.CumulativeNamespaceRequest
    ) -> cumulative.CumulativeNamespaceResult:
        combined = self._combined(request)
        return cumulative.CumulativeNamespaceResult(
            namespace_id=request.work.namespace_id,
            namespace_store_id=request.work.namespace_store_id,
            base_checkpoint_sha256=request.base.checkpoint.sha256,
            combined_store_receipt=asdict(combined),
            compilation_receipt_sha256=combined.compilation_receipt_sha256,
            artifact_projection={
                "combined_store_key": request.work.namespace_store_id,
                "retained_request_token_state_bytes": 0,
            },
            questions=tuple(self._question(request, query) for query in request.queries),
        )

    def execute(
        self, request: cumulative.CumulativeNamespaceRequest
    ) -> cumulative.CumulativeNamespaceResult:
        self.executed.append(request)
        return self._result(request)

    def verify(
        self,
        request: cumulative.CumulativeNamespaceRequest,
        expected: object,
    ) -> None:
        self.verified.append(request)
        assert self._result(request).projection() == expected


def _execute(
    root: Path,
    inputs: cumulative.ConfirmationCumulativeInput,
) -> tuple[_CumulativeBackend, cumulative.ConfirmationCumulativeExecution]:
    backend = _CumulativeBackend(inputs.policy_freeze.sha256)
    execution = cumulative.execute_confirmation_cumulative_namespaces(
        inputs,
        output_root=root / "cumulative",
        backend=backend,
        token_counter=lambda text: len(text.split()),
    )
    return backend, execution


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def _evidence_signature(merge: dict[str, object]) -> dict[str, tuple[tuple[str, ...], ...]]:
    signatures: dict[str, tuple[tuple[str, ...], ...]] = {}
    for wrapper in merge["questions"]:  # type: ignore[index,union-attr]
        question = wrapper["question"]
        signatures[question["question"]] = tuple(
            tuple(item["evidence_id"] for item in stage["evidence"])
            for stage in question["stages"]
        )
    return signatures


def test_execute_resume_merge_and_generic_matched_s0(tmp_path: Path) -> None:
    treatment = _treatment(_samples((2, 2, 2, 3)), tag="execute")
    inputs = _make_inputs(tmp_path, treatment, (2, 2), target_tokens=5)
    backend, first = _execute(tmp_path, inputs)

    assert (first.created_count, first.reused_count, first.physical_provider_calls) == (
        2,
        0,
        0,
    )
    assert [
        tuple(query.question_id for query in request.queries)
        for request in backend.executed
    ] == [
        tuple(probe.question_id for probe in work.probes)
        for work in inputs.workset.namespaces
    ]
    assert all(
        request.base.checkpoint.payload["namespace_store_id"]
        == request.work.namespace_store_id
        for request in backend.executed
    )

    second = cumulative.execute_confirmation_cumulative_namespaces(
        inputs,
        output_root=tmp_path / "cumulative",
        backend=backend,
        token_counter=lambda text: len(text.split()),
    )
    assert (second.created_count, second.reused_count) == (0, 2)
    assert len(backend.executed) == 2
    assert len(backend.verified) == 2
    assert first.checkpoint_sha256s == second.checkpoint_sha256s

    forward = cumulative.replay_confirmation_cumulative_merge(
        inputs,
        cumulative_execution=first,
        token_counter=lambda text: len(text.split()),
    )
    reversed_execution = replace(
        first,
        checkpoint_paths=tuple(reversed(first.checkpoint_paths)),
        checkpoint_sha256s=tuple(reversed(first.checkpoint_sha256s)),
    )
    replayed = cumulative.replay_confirmation_cumulative_merge(
        inputs,
        cumulative_execution=reversed_execution,
        token_counter=lambda text: len(text.split()),
    )
    assert replayed == forward
    assert "ordinal" not in _all_keys(forward)
    assert forward["physical_provider_calls"] == 0
    assert [
        stage["stage_id"]
        for wrapper in forward["questions"]
        for stage in wrapper["question"]["stages"]
    ] == list(cumulative.STAGE_IDS) * len(treatment.samples)
    for wrapper in forward["questions"]:
        stage_ids = [
            tuple(item["evidence_id"] for item in stage["evidence"])
            for stage in wrapper["question"]["stages"]
        ]
        assert all(
            child[: len(parent)] == parent
            for parent, child in zip(stage_ids, stage_ids[1:])
        )

    sealed, created = cumulative.publish_confirmation_cumulative_merge(
        inputs,
        cumulative_execution=reversed_execution,
        output_path=tmp_path / "cumulative" / "merged.json",
        token_counter=lambda text: len(text.split()),
    )
    assert created is True
    same, created_again = cumulative.publish_confirmation_cumulative_merge(
        inputs,
        cumulative_execution=first,
        output_path=tmp_path / "cumulative" / "merged.json",
        token_counter=lambda text: len(text.split()),
    )
    assert created_again is False
    assert same.sha256 == sealed.sha256
    population = cumulative.matched_s0_population_from_confirmation_merge(
        inputs,
        cumulative_execution=first,
        merged=sealed,
        max_prompt_tokens=50_000,
        renderer_id=V4_RENDERER_ID,
        token_counter=lambda text: len(text.split()),
    )
    assert population.question_count == len(treatment.samples)
    assert population.prompt_population.logical_prompt_count == len(treatment.samples)
    assert population.preflight_projection()["provider_calls"] == 0


def test_semantic_checkpoint_tamper_fails_before_backend_verify(tmp_path: Path) -> None:
    treatment = _treatment(_samples((3, 3, 3, 3)), tag="tamper")
    inputs = _make_inputs(tmp_path, treatment, (2, 2), target_tokens=5)
    backend, execution = _execute(tmp_path, inputs)
    target = execution.checkpoint_paths[0]
    payload = json.loads(target.read_bytes())
    question = payload["execution"]["questions"][0]
    question["stages"][1]["evidence"].reverse()
    question_body = dict(question)
    question_body.pop("question_receipt_sha256")
    question["question_receipt_sha256"] = canonical_sha256(question_body)
    checkpoint_body = dict(payload)
    checkpoint_body.pop("checkpoint_receipt_sha256")
    payload["checkpoint_receipt_sha256"] = canonical_sha256(checkpoint_body)
    raw = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    target.write_bytes(raw)
    target.with_name(target.name + ".sha256").write_bytes(
        f"{digest}  {target.name}\n".encode("ascii")
    )

    verified_before = len(backend.verified)
    with pytest.raises(
        cumulative.ConfirmationCumulativeError,
        match="evidence coordinates|not cumulative",
    ):
        cumulative.execute_confirmation_cumulative_namespaces(
            inputs,
            output_root=tmp_path / "cumulative",
            backend=backend,
            token_counter=lambda text: len(text.split()),
        )
    assert len(backend.verified) == verified_before


def test_permutation_renumbering_and_growth_preserve_content_retrieval(
    tmp_path: Path,
) -> None:
    samples = _samples((3, 3, 3, 3, 3, 3))
    original = _treatment(samples[:4], tag="original")
    renamed_samples = tuple(
        replace(
            sample,
            sample_id=f"renamed-{index}",
            questions=(
                replace(sample.questions[0], question_id=f"renamed-{index}"),
            ),
        )
        for index, sample in enumerate(samples[:4])
    )
    renamed = _treatment(renamed_samples, tag="renamed")
    permuted = _treatment(samples[2:4] + samples[:2], tag="permuted")
    grown = _treatment(samples, tag="grown")

    def run(
        label: str,
        treatment: ConfirmationTreatmentInput,
        sizes: tuple[int, ...],
    ) -> tuple[cumulative.ConfirmationCumulativeInput, dict[str, object]]:
        root = tmp_path / label
        inputs = _make_inputs(root, treatment, sizes, target_tokens=5)
        _backend, execution = _execute(root, inputs)
        merged = cumulative.replay_confirmation_cumulative_merge(
            inputs,
            cumulative_execution=execution,
            token_counter=lambda text: len(text.split()),
        )
        return inputs, merged

    original_inputs, original_merge = run("original", original, (2, 2))
    renamed_inputs, renamed_merge = run("renamed", renamed, (2, 2))
    permuted_inputs, permuted_merge = run("permuted", permuted, (2, 2))
    grown_inputs, grown_merge = run("grown", grown, (2, 2, 2))

    original_stores = [
        work.namespace_store_id for work in original_inputs.workset.namespaces
    ]
    assert original_stores == [
        work.namespace_store_id for work in renamed_inputs.workset.namespaces
    ]
    assert sorted(original_stores) == sorted(
        work.namespace_store_id for work in permuted_inputs.workset.namespaces
    )
    assert original_stores == [
        work.namespace_store_id for work in grown_inputs.workset.namespaces[:2]
    ]
    original_signature = _evidence_signature(original_merge)
    assert _evidence_signature(renamed_merge) == original_signature
    assert _evidence_signature(permuted_merge) == original_signature
    assert {
        key: value
        for key, value in _evidence_signature(grown_merge).items()
        if key in original_signature
    } == original_signature


def test_arbitrary_namespace_schedule_and_foreign_question_fail_closed(
    tmp_path: Path,
) -> None:
    treatment = _treatment(_samples((3, 3, 3, 3, 3, 3)), tag="schedule")
    inputs = _make_inputs(tmp_path, treatment, (1, 2, 3), target_tokens=8)

    class EscapingBackend(_CumulativeBackend):
        def execute(
            self, request: cumulative.CumulativeNamespaceRequest
        ) -> cumulative.CumulativeNamespaceResult:
            result = super().execute(request)
            first = result.questions[0]
            return replace(
                result,
                questions=(replace(first, question_id="foreign-question"), *result.questions[1:]),
            )

    with pytest.raises(
        cumulative.ConfirmationCumulativeError,
        match="another probe",
    ):
        cumulative.execute_confirmation_cumulative_namespaces(
            inputs,
            output_root=tmp_path / "cumulative",
            backend=EscapingBackend(inputs.policy_freeze.sha256),
            token_counter=lambda text: len(text.split()),
    )
    assert not (tmp_path / "cumulative" / "checkpoints").exists()


def _runtime_result_from_projection(
    question: cumulative.CumulativeQuestion,
) -> object:
    projected = question.projection()
    typed_stages = tuple(
        CumulativeRetrievalStageReceipt(**stage["stage_receipt"])
        for stage in projected["stages"]
    )
    ladder = CumulativeRetrievalLadder(stages=typed_stages)
    root_evidence = projected["stages"][0]["evidence"]
    predecessor = SimpleNamespace(
        receipt=CausalCoveragePredecessorReceipt(
            **projected["predecessor_receipt"]
        ),
        excerpts=tuple(
            SimpleNamespace(source_id=row["source_id"], text=row["text"])
            for row in root_evidence
        ),
    )
    addition_packets: list[object | None] = []
    parent_size = len(root_evidence)
    for stage in projected["stages"][1:]:
        additions = stage["evidence"][parent_size:]
        parent_size = len(stage["evidence"])
        if additions:
            addition_packets.append(
                SimpleNamespace(
                    atoms=tuple(
                        SimpleNamespace(
                            span=SimpleNamespace(source_id=row["source_id"]),
                            text=row["text"],
                        )
                        for row in additions
                    )
                )
            )
        else:
            addition_packets.append(None)
    messages = {
        stage["stage_id"]: stage["provider_messages"]
        for stage in projected["stages"]
    }
    return SimpleNamespace(
        predecessor=predecessor,
        ladder=ladder,
        addition_packets=tuple(addition_packets),
        receipt=RecallGuardedCumulativeReceipt(**projected["retrieval_receipt"]),
        provider_messages_by_stage=lambda: messages,
    )


def test_production_adapter_reuses_mocked_store_and_retrieval_runtime(
    tmp_path: Path,
) -> None:
    treatment = _treatment(_samples((3, 3)), tag="production-adapter")
    inputs = _make_inputs(tmp_path, treatment, (2,), target_tokens=5)
    query_map = cumulative._validate_inputs(
        inputs, token_counter=lambda text: len(text.split())
    )
    base_locations = cumulative._base_namespaces(inputs)
    request = cumulative._request_for_work(
        inputs=inputs,
        output_root=tmp_path / "production-cumulative",
        work=inputs.workset.namespaces[0],
        queries_by_receipt=query_map,
        base_locations=base_locations,
    )
    synthetic = _CumulativeBackend(inputs.policy_freeze.sha256)
    typed_questions = {
        query.question: synthetic._question(request, query)
        for query in request.queries
    }
    source_path = (
        request.base.namespace_root / "synthetic-store" / "memory.db"
    )
    combined = replace(
        synthetic._combined(request),
        source_database_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        receipt_sha256="",
    )
    calls: dict[str, list[object]] = {"build": [], "open": [], "retrieve": []}

    class Condenser:
        def set_context_candidate_selector(self, selector: object) -> None:
            self.selector = selector

    class Prepared:
        def __init__(self) -> None:
            self.condenser = Condenser()
            self.receipt = combined
            self.compilation = SimpleNamespace(
                artifact=SimpleNamespace(artifact_id=combined.artifact_id),
                receipt_sha256=combined.compilation_receipt_sha256,
            )

        def close(self) -> None:
            return None

    def build_store(source: Path, target: Path, **kwargs: object) -> Prepared:
        calls["build"].append((source, target, kwargs))
        target.mkdir(parents=True)
        return Prepared()

    def open_store(target: Path, **kwargs: object) -> Prepared:
        calls["open"].append((target, kwargs))
        return Prepared()

    def retrieve(_condenser: object, **kwargs: object) -> object:
        calls["retrieve"].append(dict(kwargs))
        raw_question = str(kwargs["query"])
        assert kwargs["prompt_question"] == typed_questions[
            raw_question
        ].dated_question
        return _runtime_result_from_projection(typed_questions[raw_question])

    class Retrieval:
        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {"retrieval": "synthetic-population-neutral"}

    backend = cumulative.ProductionCumulativeNamespaceBackend(
        policy_freeze_sha256=inputs.policy_freeze.sha256,
        runtime_policy_binding={
            "model_residency_mode": cumulative.RESIDENT_PRODUCTION_MODE,
            "policy": "synthetic-frozen",
            "resident_preflight_receipt_sha256": canonical_sha256(
                {"resident_preflight": "synthetic"}
            ),
        },
        source_backend_identity_sha256=request.base.backend_identity_sha256,
        source_treatment_contract_sha256=_synthetic_source_contract()[
            "contract_sha256"
        ],
        model_residency_mode=cumulative.RESIDENT_PRODUCTION_MODE,
        config=SimpleNamespace(retrieval=Retrieval()),
        embedder=object(),
        compilation_policy={"boundary_mode": "synthetic"},
        coverage_selector=object(),
        representative_linker=object(),
        episode_policy_factory=lambda artifact_id: ("episode", artifact_id),
        representative_policy_factory=lambda artifact_id: (
            "representative",
            artifact_id,
        ),
        closure_policy=("closure",),
        max_context_tokens=50_000,
        max_prompt_tokens=50_000,
        responder_output_token_reserve=64,
        source_router_max_sources=7,
        source_router_rrf_constant=11,
        embedding_identity=_SYNTHETIC_EMBEDDING_IDENTITY,
        build_store=build_store,
        open_store=open_store,
        retrieve=retrieve,
    )
    first = cumulative.execute_confirmation_cumulative_namespaces(
        inputs,
        output_root=tmp_path / "production-cumulative",
        backend=backend,
        token_counter=lambda text: len(text.split()),
    )
    assert (len(calls["build"]), len(calls["retrieve"])) == (
        1,
        len(treatment.samples),
    )
    assert first.physical_provider_calls == 0

    second = cumulative.execute_confirmation_cumulative_namespaces(
        inputs,
        output_root=tmp_path / "production-cumulative",
        backend=backend,
        token_counter=lambda text: len(text.split()),
    )
    assert (second.created_count, second.reused_count) == (0, 1)
    assert len(calls["open"]) == 1
    assert len(calls["retrieve"]) == len(treatment.samples)

    source_path.write_bytes(b"tampered-after-base-checkpoint")
    with pytest.raises(
        cumulative.ConfirmationCumulativeError,
        match="database changed after source verification",
    ):
        cumulative.execute_confirmation_cumulative_namespaces(
            inputs,
            output_root=tmp_path / "production-cumulative",
            backend=backend,
            token_counter=lambda text: len(text.split()),
        )


def test_production_adapter_requires_sealed_vectors_for_staged_residency() -> None:
    digest = "a" * 64

    class Retrieval:
        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {"mode": "causal_graph"}

    with pytest.raises(
        cumulative.ConfirmationCumulativeError,
        match="sealed frozen query vectors",
    ):
        cumulative.ProductionCumulativeNamespaceBackend(
            policy_freeze_sha256=digest,
            runtime_policy_binding={
                "model_residency_mode": cumulative.STAGED_PRODUCTION_MODE,
                "resident_preflight_receipt_sha256": digest,
            },
            source_backend_identity_sha256=digest,
            source_treatment_contract_sha256=digest,
            model_residency_mode=cumulative.STAGED_PRODUCTION_MODE,
            config=SimpleNamespace(retrieval=Retrieval()),
            embedder=object(),
            compilation_policy={"boundary_mode": "fixed_interval"},
            coverage_selector=object(),
            representative_linker=object(),
            episode_policy_factory=lambda artifact_id: artifact_id,
            representative_policy_factory=lambda artifact_id: artifact_id,
            closure_policy=object(),
            max_context_tokens=1,
            max_prompt_tokens=1,
            responder_output_token_reserve=0,
            source_router_max_sources=1,
            source_router_rrf_constant=1,
            embedding_identity=_SYNTHETIC_EMBEDDING_IDENTITY,
        )

    staged_backend = cumulative.ProductionCumulativeNamespaceBackend(
        policy_freeze_sha256=digest,
        runtime_policy_binding={
            "model_residency_mode": cumulative.STAGED_PRODUCTION_MODE
        },
        source_backend_identity_sha256=digest,
        source_treatment_contract_sha256=digest,
        model_residency_mode=cumulative.STAGED_PRODUCTION_MODE,
        embedding_runtime_kind="sealed_frozen_queries",
        staged_barrier_receipt_sha256=digest,
        config=SimpleNamespace(retrieval=Retrieval()),
        embedder=object(),
        compilation_policy={"boundary_mode": "fixed_interval"},
        coverage_selector=object(),
        representative_linker=object(),
        episode_policy_factory=lambda artifact_id: artifact_id,
        representative_policy_factory=lambda artifact_id: artifact_id,
        closure_policy=object(),
        max_context_tokens=1,
        max_prompt_tokens=1,
        responder_output_token_reserve=0,
        source_router_max_sources=1,
        source_router_rrf_constant=1,
        embedding_identity=_SYNTHETIC_EMBEDDING_IDENTITY,
    )
    assert staged_backend.policy_freeze_sha256 == digest


class _StagedPreparationBackend:
    def __init__(self, freeze_sha256: str, events: list[str]) -> None:
        self._freeze = freeze_sha256
        self._events = events
        self._identity = canonical_sha256({"backend": "synthetic-staged-bge"})

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def policy_freeze_sha256(self) -> str:
        return self._freeze

    @property
    def embedding_identity(self) -> dict[str, str]:
        return _SYNTHETIC_EMBEDDING_IDENTITY

    @staticmethod
    def _vectors(
        request: cumulative.CumulativeNamespaceRequest,
    ) -> dict[str, tuple[float, float]]:
        return {
            query: (
                float(int(quote_sha256(query)[:4], 16)),
                float(len(query)),
            )
            for query in cumulative.held_out_queries(request.queries)
        }

    def _combined(
        self, request: cumulative.CumulativeNamespaceRequest
    ) -> CombinedCumulativeStoreReceipt:
        batch = tuple(self._vectors(request))
        return replace(
            _CumulativeBackend(self._freeze)._combined(request),
            held_out_query_batch_sha256=runtime_identity_sha256(
                [{"query_sha256": quote_sha256(query)} for query in batch]
            ),
            receipt_sha256="",
        )

    def prepare(
        self, request: cumulative.CumulativeNamespaceRequest
    ) -> staged.StagedPreparationResult:
        self._events.append(f"bge_prepare:{request.work.namespace_store_id}")
        combined = self._combined(request)
        vectors = self._vectors(request)
        return staged.StagedPreparationResult(
            namespace_id=request.work.namespace_id,
            namespace_store_id=request.work.namespace_store_id,
            base_checkpoint_sha256=request.base.checkpoint.sha256,
            combined_store_receipt=asdict(combined),
            compilation_receipt_sha256=combined.compilation_receipt_sha256,
            combined_store_mode="fresh_synthetic_build",
            query_batch=tuple(vectors),
            query_vectors=vectors,
        )

    def verify(
        self,
        request: cumulative.CumulativeNamespaceRequest,
        expected: object,
        query_vectors: object,
    ) -> None:
        self._events.append(f"bge_verify:{request.work.namespace_store_id}")
        assert isinstance(expected, dict)
        assert expected["combined_store_receipt"] == asdict(self._combined(request))
        assert {
            key: tuple(value) for key, value in query_vectors.items()
        } == self._vectors(request)

    def release_bge(self) -> object:
        if not self._events or self._events[-1] != "bge_close":
            self._events.append("bge_close")
        return staged.bge_release_receipt(
            preparation_backend_identity_sha256=self._identity,
            embedding_identity=self.embedding_identity,
        )

    def freeze_query_batch(
        self, queries: tuple[str, ...]
    ) -> dict[str, tuple[float, float]]:
        self._events.append("facet_embed")
        return {
            query: (
                float(int(quote_sha256(query)[:4], 16)),
                float(len(query)),
            )
            for query in queries
        }


class _StagedQwen:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._identity = canonical_sha256({"runtime": "synthetic-qwen"})
        self._coverage = object()
        self._linker = object()

    @property
    def identity_sha256(self) -> str:
        return self._identity

    @property
    def coverage_selector(self) -> object:
        return self._coverage

    @property
    def representative_linker(self) -> object:
        return self._linker

    @property
    def physical_provider_calls(self) -> int:
        return 0

    def close(self) -> None:
        self._events.append("qwen_close")


class _StagedQwenFactory:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._identity = canonical_sha256({"factory": "synthetic-qwen"})

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def load_after_barrier(self, barrier: object) -> _StagedQwen:
        assert isinstance(barrier, base_adapter.SealedPayload)
        assert self._events[-1] == "bge_close"
        self._events.append("qwen_load")
        return _StagedQwen(self._events)


class _StagedRetrievalFactory:
    def __init__(self, freeze_sha256: str, events: list[str]) -> None:
        self._freeze = freeze_sha256
        self._events = events
        self._identity = canonical_sha256({"factory": "synthetic-retrieval"})

    @property
    def identity_sha256(self) -> str:
        return self._identity

    def create(self, **kwargs: object) -> _CumulativeBackend:
        qwen_runtime = kwargs["qwen_runtime"]
        assert getattr(qwen_runtime, "physical_provider_calls") == 0
        assert getattr(qwen_runtime, "coverage_selector") is not None
        assert getattr(qwen_runtime, "representative_linker") is not None
        assert kwargs["frozen_queries"]
        self._events.append("retrieval_backend_create")
        return _CumulativeBackend(self._freeze)


def _run_staged(
    inputs: cumulative.ConfirmationCumulativeInput,
    root: Path,
    events: list[str],
) -> staged.StagedCoordinatorExecution:
    return staged.execute_staged_confirmation_cumulative(
        inputs,
        output_root=root,
        preparation_backend=_StagedPreparationBackend(
            inputs.policy_freeze.sha256, events
        ),
        qwen_factory=_StagedQwenFactory(events),
        retrieval_factory=_StagedRetrievalFactory(
            inputs.policy_freeze.sha256, events
        ),
        token_counter=lambda text: len(text.split()),
    )


def test_staged_coordinator_orders_barrier_and_resumes_all_namespaces(
    tmp_path: Path,
) -> None:
    treatment = _treatment(_samples((3, 3, 3)), tag="staged-order")
    inputs = _make_inputs(tmp_path, treatment, (1, 2), target_tokens=5)
    root = tmp_path / "staged"
    events: list[str] = []
    first = _run_staged(inputs, root, events)

    prepare_positions = [
        index for index, event in enumerate(events) if event.startswith("bge_prepare:")
    ]
    assert len(prepare_positions) == len(inputs.workset.namespaces)
    assert max(prepare_positions) < events.index("bge_close")
    assert events.index("bge_close") < events.index("qwen_load")
    assert events.index("qwen_load") < events.index("retrieval_backend_create")
    assert events[-1] == "qwen_close"
    assert (first.preparation.created_count, first.cumulative.created_count) == (2, 2)
    assert first.physical_provider_calls == 0
    replay = staged.SealedFrozenQueryEmbedder(first.preparation.descriptors)
    for descriptor in first.preparation.descriptors:
        assert replay.embed_queries(descriptor.query_batch).shape == (
            len(descriptor.query_batch),
            descriptor.dimension,
        )

    resumed_events: list[str] = []
    second = _run_staged(inputs, root, resumed_events)
    assert not any(event.startswith("bge_prepare:") for event in resumed_events)
    assert sum(event.startswith("bge_verify:") for event in resumed_events) == 2
    assert resumed_events.index("bge_close") < resumed_events.index("qwen_load")
    assert (second.preparation.reused_count, second.cumulative.reused_count) == (2, 2)
    assert first.preparation.checkpoint_sha256s == (
        second.preparation.checkpoint_sha256s
    )
    assert first.barrier.sha256 == second.barrier.sha256


def test_staged_before_release_hook_runs_after_prepare_and_before_bge_close(
    tmp_path: Path,
) -> None:
    treatment = _treatment(_samples((3, 3)), tag="staged-semantic-facet-hook")
    inputs = _make_inputs(tmp_path, treatment, (1, 1), target_tokens=3)
    events: list[str] = []
    preparation_backend = _StagedPreparationBackend(
        inputs.policy_freeze.sha256, events
    )

    def freeze_semantic_facets(
        preparation: staged.StagedPreparationExecution,
        backend: staged.StagedPreparationBackend,
    ) -> None:
        assert len(preparation.descriptors) == len(inputs.workset.namespaces)
        assert backend is preparation_backend
        assert all(event.startswith("bge_prepare:") for event in events)
        events.append("facet_freeze")

    staged.execute_staged_confirmation_cumulative(
        inputs,
        output_root=tmp_path / "staged-semantic-facet-hook",
        preparation_backend=preparation_backend,
        qwen_factory=_StagedQwenFactory(events),
        retrieval_factory=_StagedRetrievalFactory(
            inputs.policy_freeze.sha256, events
        ),
        token_counter=lambda text: len(text.split()),
        before_bge_release=freeze_semantic_facets,
    )

    assert max(
        index for index, event in enumerate(events) if event.startswith("bge_prepare:")
    ) < events.index("facet_freeze")
    assert events.index("facet_freeze") < events.index("bge_close")
    assert events.index("bge_close") < events.index("qwen_load")


def test_semantic_facet_hook_freezes_then_verifies_without_reembedding(
    tmp_path: Path,
) -> None:
    treatment = _treatment(_samples((3, 3)), tag="semantic-facet-cache")
    inputs = _make_inputs(tmp_path, treatment, (1, 1), target_tokens=3)
    root = tmp_path / "semantic-facet-cache"
    preparations: list[semantic_planes.ConfirmationSemanticFacetPreparation] = []

    first_events: list[str] = []

    def first_hook(
        preparation: staged.StagedPreparationExecution,
        backend: staged.StagedPreparationBackend,
    ) -> None:
        preparations.append(
            semantic_planes.prepare_confirmation_semantic_facet_vectors(
                inputs,
                preparation,
                backend=backend,
                output_root=root,
                token_counter=lambda text: len(text.split()),
            )
        )

    first = staged.execute_staged_confirmation_cumulative(
        inputs,
        output_root=root,
        preparation_backend=_StagedPreparationBackend(
            inputs.policy_freeze.sha256, first_events
        ),
        qwen_factory=_StagedQwenFactory(first_events),
        retrieval_factory=_StagedRetrievalFactory(
            inputs.policy_freeze.sha256, first_events
        ),
        token_counter=lambda text: len(text.split()),
        before_bge_release=first_hook,
    )
    facet_first = preparations.pop()
    assert facet_first.created_count == len(inputs.workset.namespaces)
    assert facet_first.local_embedding_batch_calls == len(inputs.workset.namespaces)
    assert first_events.count("facet_embed") == len(inputs.workset.namespaces)
    release = semantic_planes.publish_confirmation_semantic_facet_release(
        facet_first,
        first.barrier.payload["release_receipt"],
        output_root=root,
    )

    second_events: list[str] = []

    def second_hook(
        preparation: staged.StagedPreparationExecution,
        backend: staged.StagedPreparationBackend,
    ) -> None:
        preparations.append(
            semantic_planes.prepare_confirmation_semantic_facet_vectors(
                inputs,
                preparation,
                backend=backend,
                output_root=root,
                token_counter=lambda text: len(text.split()),
            )
        )

    second = staged.execute_staged_confirmation_cumulative(
        inputs,
        output_root=root,
        preparation_backend=_StagedPreparationBackend(
            inputs.policy_freeze.sha256, second_events
        ),
        qwen_factory=_StagedQwenFactory(second_events),
        retrieval_factory=_StagedRetrievalFactory(
            inputs.policy_freeze.sha256, second_events
        ),
        token_counter=lambda text: len(text.split()),
        before_bge_release=second_hook,
    )
    facet_second = preparations.pop()
    assert facet_second.reused_count == len(inputs.workset.namespaces)
    assert facet_second.local_embedding_batch_calls == 0
    assert "facet_embed" not in second_events
    assert facet_second.artifact.sha256 == facet_first.artifact.sha256
    assert second.barrier.sha256 == first.barrier.sha256
    assert semantic_planes.publish_confirmation_semantic_facet_release(
        facet_second,
        second.barrier.payload["release_receipt"],
        output_root=root,
    ).sha256 == release.sha256


def test_staged_coordinator_rejects_vector_tamper_before_qwen_load(
    tmp_path: Path,
) -> None:
    treatment = _treatment(_samples((3, 3)), tag="staged-tamper")
    inputs = _make_inputs(tmp_path, treatment, (1, 1), target_tokens=3)
    root = tmp_path / "staged"
    _run_staged(inputs, root, [])
    vector_path = next((root / "staged-preparation" / "vectors").glob("*.json"))
    vector_path.write_bytes(vector_path.read_bytes() + b" ")

    events: list[str] = []
    with pytest.raises(
        staged.StagedCoordinatorError,
        match="cannot verify",
    ):
        _run_staged(inputs, root, events)
    assert "bge_close" in events
    assert "qwen_load" not in events


def test_production_qwen_factory_loads_post_barrier_and_closes(tmp_path: Path) -> None:
    from memory_condense.modeling.qwen_prefix import (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
        expected_prefix_checkpoint_sha256,
    )
    from memory_condense.search.selectors.causal_choice_scorer import (
        QWEN_CHOICE_CHECKPOINT_SHA256,
        QWEN_CHOICE_MODEL_ID,
        QWEN_CHOICE_MODEL_REVISION,
    )

    events: list[str] = []

    class _ClosableSelector:
        def close(self) -> None:
            events.append("qwen_close")

    selector = _ClosableSelector()
    linker = SimpleNamespace(encoder=SimpleNamespace(_torch=None))

    def load_shared_qwen(
        config: object, prefix_dir: Path, choice_dir: Path
    ) -> tuple[object, object]:
        assert isinstance(config, EvalConfig)
        assert prefix_dir.name == "prefix"
        assert choice_dir.name == "choice"
        assert events[-1] == "bge_close"
        events.append("qwen_load")
        return selector, linker

    config = EvalConfig(
        retrieval=RetrievalConfig(
            mode="causal_graph",
            coverage_selection=True,
            coverage_selector_backend="qwen_prefix_choice",
            coverage_selector_prefix_layers=2,
            coverage_selector_attention_layer=1,
            coverage_selector_prefix_model_id=DEFAULT_MODEL_ID,
            coverage_selector_prefix_revision=DEFAULT_MODEL_REVISION,
            coverage_selector_prefix_checkpoint_sha256=(
                expected_prefix_checkpoint_sha256(2)
            ),
            coverage_selector_prefix_device="cuda",
            coverage_selector_prefix_dtype="float16",
            coverage_selector_choice_model_id=QWEN_CHOICE_MODEL_ID,
            coverage_selector_choice_revision=QWEN_CHOICE_MODEL_REVISION,
            coverage_selector_choice_checkpoint_sha256=(
                QWEN_CHOICE_CHECKPOINT_SHA256
            ),
            coverage_selector_choice_device="cuda",
            coverage_selector_choice_dtype="float16",
        )
    )
    treatment = _treatment(_samples((3, 3)), tag="production-qwen")
    inputs = _make_inputs(tmp_path, treatment, (1, 1), target_tokens=3)
    factory = staged.ProductionStagedQwenRuntimeFactory(
        config=config,
        qwen_prefix_model_dir=tmp_path / "prefix",
        qwen_choice_model_dir=tmp_path / "choice",
        load_shared_qwen=load_shared_qwen,
        loader_identity_sha256=canonical_sha256({"loader": "synthetic"}),
    )
    result = staged.execute_staged_confirmation_cumulative(
        inputs,
        output_root=tmp_path / "staged-production-qwen",
        preparation_backend=_StagedPreparationBackend(
            inputs.policy_freeze.sha256, events
        ),
        qwen_factory=factory,
        retrieval_factory=_StagedRetrievalFactory(
            inputs.policy_freeze.sha256, events
        ),
        token_counter=lambda text: len(text.split()),
    )

    assert events.index("bge_close") < events.index("qwen_load")
    assert events.index("qwen_load") < events.index("retrieval_backend_create")
    assert events[-1] == "qwen_close"
    assert result.physical_provider_calls == 0
