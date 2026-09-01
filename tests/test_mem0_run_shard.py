from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

import tools.mem0_eval.run_shard as run_shard_module
from tools.mem0_eval import resumable_runner
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.mem0_adapter import (
    MEM0_ATTRIBUTION_KIND,
    MEM0_CERTIFIED_RENDERING,
    Mem0AdapterStats,
)
from memory_condense.eval.sample_identity import canonical_sha256, sample_sha256
from memory_condense.eval.schemas import UsageStats
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_REVISION,
)
from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from tools.mem0_eval.preflight import tool_implementation_sha256
from tools.mem0_eval.prompt_pack import (
    MEM0_PROMPT_CAP_SEMANTICS,
    MEM0_SOURCE_JUDGE_MODEL,
    MEM0_SOURCE_RESPONDER_MODEL,
)
from tools.mem0_eval.protocol import (
    RawStressShard,
    build_composite_add_batches,
    compose_raw_stress_record,
    count_official_add_requests,
)
from tools.mem0_eval.run_shard import (
    Mem0ShardRunError,
    ProviderCallResult,
    RetrievalStageAuthorization,
    ScoringStageAuthorization,
    ShardProcessGuard,
    TrustedRuntimeBinding,
    build_adapter_prepared_corpus,
    canonical_json_sha256,
    main,
    run_retrieval_stage,
    run_scoring_stage,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_1 = "1" * 64
SHA_2 = "2" * 64
SHA_3 = "3" * 64
LOCK_PATH = Path(__file__).resolve().parents[1] / "pixi.lock"
LOCK_SHA256 = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()

STABLE_PAYLOAD = {
    "protocol": "mem0-oss-2.0.18-certified-local-v1",
    "config": {"api_key": "<redacted>"},
    "stack": {
        "dependency_versions": {
            "mem0ai": "2.0.18",
            "qdrant-client": "1.15.1",
        },
        "bm25_model": "Qdrant/bm25",
        "spacy_model": "en_core_web_sm",
        "bm25_operational": True,
        "entity_extraction_operational": True,
    },
}
SHA_C = canonical_json_sha256(STABLE_PAYLOAD)


def _declare_stateless(invoker: Any) -> Any:
    invoker.request_token_state_receipt = lambda: {
        "contract": "stateless-request-token-state-v1",
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    return invoker


def _shard() -> RawStressShard:
    questions = [
        BenchmarkQuestion(
            question_id=f"q-{index}",
            question=f"What is answer {index}?",
            answer=f"answer {index}",
            category="single-session-user",
            question_date="2025-01-15",
        )
        for index in range(10)
    ]
    sample = BenchmarkSample(
        sample_id="mem0-context-stress-1000000-offset-000",
        turns=[("user", "alpha"), ("assistant", "beta")],
        turn_source_ids=["q-0::session-1", "q-0::session-1"],
        questions=questions,
    )
    records: list[dict[str, Any]] = [
        {
            "question_id": "q-0",
            "haystack_sessions": [
                [],
                [
                    {"role": "user", "content": "alpha"},
                    {"role": "assistant", "content": "beta"},
                ],
            ],
            "haystack_session_ids": ["unused", "session-1"],
            "haystack_dates": ["2025-02-01", "2025-01-01"],
        },
        {
            "question_id": "q-1",
            "haystack_sessions": [
                [{"role": "assistant", "content": "gamma"}]
            ],
            "haystack_session_ids": ["session-9"],
            "haystack_dates": ["2025-01-02"],
        },
    ]
    records.extend(
        {
            "question_id": f"q-{index}",
            "haystack_sessions": [],
            "haystack_session_ids": [],
            "haystack_dates": [],
        }
        for index in range(2, 10)
    )
    history_ids = tuple(f"q-{index}" for index in range(10))
    raw_bundle = compose_raw_stress_record(
        records,
        sample_id="mem0-context-stress-1000000-offset-000",
    )
    batches = build_composite_add_batches(records)
    return RawStressShard(
        sample_offset=0,
        parsed_sample=sample,
        sample_sha256=sample_sha256(sample),
        history_sample_ids=history_ids,
        raw_history_bundle=raw_bundle,
        raw_history_bundle_sha256=canonical_sha256(raw_bundle),
        add_batches=batches,
        add_counts=count_official_add_requests(records),
    )


def _runtime_identity() -> dict[str, Any]:
    return {
        **STABLE_PAYLOAD,
        "stable_config_sha256": SHA_C,
        "effective_config_sha256": SHA_D,
        "local_owned_state": True,
        "on_disk": True,
        "certified": True,
    }


def _source_evaluation_identity() -> dict[str, Any]:
    return {
        "responder_model": MEM0_SOURCE_RESPONDER_MODEL,
        "judge_model": MEM0_SOURCE_JUDGE_MODEL,
        "use_judge": True,
        "provider_retries": 0,
        "max_provider_calls_per_shard": 20,
        "max_prompt_tokens": 8_000,
        "prompt_cap_semantics": MEM0_PROMPT_CAP_SEMANTICS,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": 256,
        "recent_window": 4,
        "accuracy_target": 0.95,
        "min_target_questions": 100,
        "stress_context_tokens": 1_000_000,
        "stress_questions": 10,
        "stress_question_offset": 0,
        "max_samples": 1,
        "sample_offsets": list(range(0, 100, 10)),
    }


def _model_identities() -> tuple[dict[str, Any], dict[str, Any]]:
    extraction_body = {
        "provider": "test-provider",
        "model": "test-extractor",
        "revision": "test-revision",
        "provider_retries": 0,
        "logical_call_boundary": "Memory.llm.generate_response",
        "logical_calls_per_add": 1,
        "http_attempts_certified": False,
    }
    extraction = {
        **extraction_body,
        "model_identity_sha256": canonical_json_sha256(extraction_body),
    }
    embedder_body = {
        "provider": "huggingface",
        "model": "BAAI/bge-m3",
        "revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": 1024,
        "device": "cpu",
        "dtype": "float32",
        "execution": "local_offline",
        "network_calls_authorized": 0,
        "runtime_probe_required": True,
    }
    embedder = {
        **embedder_body,
        "model_identity_sha256": canonical_json_sha256(embedder_body),
    }
    return extraction, embedder


def _retrieval_authorization(
    shard: RawStressShard,
    *,
    mem0_environment_lock_sha256: str = LOCK_SHA256,
    source_environment_lock_sha256: str = LOCK_SHA256,
) -> RetrievalStageAuthorization:
    extraction, embedder = _model_identities()
    return RetrievalStageAuthorization(
        sample_offset=shard.sample_offset,
        sample_sha256=shard.sample_sha256,
        raw_history_bundle_sha256=shard.raw_history_bundle_sha256,
        question_ids=shard.question_ids,
        authorized_add_operations=shard.add_counts.add_requests,
        authorized_extraction_calls=shard.add_counts.add_requests,
        authorized_search_operations=len(shard.question_ids),
        source_validation_policy_sha256=SHA_E,
        source_implementation_sha256=SHA_F,
        source_environment_lock_sha256=source_environment_lock_sha256,
        mem0_policy_sha256=SHA_2,
        mem0_tool_implementation_sha256=tool_implementation_sha256(),
        mem0_environment_lock_sha256=mem0_environment_lock_sha256,
        mem0_stable_config_sha256=SHA_C,
        source_evaluation_identity=_source_evaluation_identity(),
        mem0_stable_payload=STABLE_PAYLOAD,
        extraction_model_identity=extraction,
        extraction_model_identity_sha256=canonical_json_sha256(extraction),
        embedder_model_identity=embedder,
        embedder_model_identity_sha256=canonical_json_sha256(embedder),
    )


class _FakeAdapter:
    def __init__(
        self,
        state: Path,
        *,
        extraction_calls_per_add: Sequence[int] = (1, 1),
        fail_extraction_call: int | None = None,
        swallow_extraction_failure: bool = False,
    ) -> None:
        self.state = state
        state.mkdir()
        (state / "owned").write_text("test", encoding="utf-8")
        self.active_user_scope: str | None = None
        self.stats = Mem0AdapterStats()
        self.corpus = None
        self.cleaned = False
        self._closed = False
        self._ledger: dict[Any, Any] = {}
        self._scopes: list[str] = []
        self._scope_protocol: dict[str, bool] = {}
        self.restored_before_cleanup = False
        self.cleanup_order: list[str] = []
        self.extraction_calls_per_add = tuple(extraction_calls_per_add)
        self.fail_extraction_call = fail_extraction_call
        self.swallow_extraction_failure = swallow_extraction_failure
        self.provider_invocations = 0

        def generate_response(*_args: Any, **_kwargs: Any) -> str:
            self.provider_invocations += 1
            if self.provider_invocations == self.fail_extraction_call:
                raise RuntimeError("synthetic extraction failure")
            return '{"facts": []}'

        self.original_generate_response = generate_response
        self.llm = SimpleNamespace(
            generate_response=generate_response,
            request_token_state_receipt=lambda: {
                "contract": "stateless-request-token-state-v1",
                "persisted_request_token_state": False,
                "retained_request_token_state_bytes": 0,
                "request_token_state_evidence_kind": (
                    "local_injected_request_token_state_contract"
                ),
                "external_provider_persistence_certified": False,
            },
        )

        self._add_target: Any | None = None

        def add(*args: Any, **kwargs: Any) -> dict[str, Any]:
            if self._add_target is None:
                raise AssertionError("per-batch fake add was not installed")
            return self._add_target(*args, **kwargs)

        self.original_add = add
        self.memory = SimpleNamespace(llm=self.llm, add=add)
        self.owned_backend = SimpleNamespace(
            backend=self.memory,
            _closed=False,
            runtime_identity=_runtime_identity(),
        )
        self._backend = self.owned_backend

    @property
    def ledger(self) -> Mapping[Any, Any]:
        return self._ledger

    def _get_backend(self) -> Any:
        return self.owned_backend

    def _runtime_identity_snapshot(self) -> Mapping[str, Any]:
        return self.owned_backend.runtime_identity

    def _ingest_prepared(self, corpus: Any) -> Any:
        self.corpus = corpus
        self.active_user_scope = "isolated-scope"
        self._scopes.append(self.active_user_scope)
        self._scope_protocol[self.active_user_scope] = True
        if len(self.extraction_calls_per_add) != len(corpus.batches):
            raise AssertionError("fake extraction plan must cover every add")

        def add(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            batch_index = self.stats.add_attempted_calls
            operation_error: BaseException | None = None
            try:
                for _ in range(self.extraction_calls_per_add[batch_index]):
                    try:
                        self.memory.llm.generate_response(messages=[])
                    except BaseException:
                        if not self.swallow_extraction_failure:
                            raise
                return {"results": [{"id": f"memory-{batch_index}"}]}
            except BaseException as exc:
                operation_error = exc
                raise
            finally:
                self.stats = replace(
                    self.stats,
                    add_calls=self.stats.add_calls + 1,
                    add_attempted_calls=self.stats.add_attempted_calls + 1,
                    add_completed_calls=(
                        self.stats.add_completed_calls
                        + (0 if operation_error is not None else 1)
                    ),
                    add_failed_calls=(
                        self.stats.add_failed_calls
                        + (1 if operation_error is not None else 0)
                    ),
                )

        # The runner has already installed its supervisor on Memory.add. Keep
        # that wrapper and swap only the callback it invokes by rebuilding the
        # fake before installation in tests that need alternate behavior.
        supervised_add = self.memory.add
        original_callback = self.original_add
        if supervised_add is original_callback:
            raise AssertionError("extraction supervisor was not installed")
        # ``supervised_add`` closed over ``original_add`` at install time, so
        # the fake's original add must itself dispatch to this mutable target.
        self._add_target = add
        for batch in corpus.batches:
            supervised_add(
                [{"role": role, "content": content} for role, content in batch.messages],
                user_id=self.active_user_scope,
                infer=True,
            )
        adds = len(corpus.batches)
        self.stats = replace(
            self.stats,
            add_returned_memories=adds,
            unique_ledger_memories=adds,
        )
        self._ledger.update(
            {
                (self.active_user_scope, f"memory-{index}"): (batch.ref,)
                for index, batch in enumerate(corpus.batches)
            }
        )
        return SimpleNamespace(
            comparison_certified=True,
            official_longmemeval_protocol=True,
            supports_exact_source_provenance=False,
            runtime_identity=_runtime_identity(),
            stats=self.stats,
            user_scope=self.active_user_scope,
            batches_added=corpus.batches,
        )

    def search(self, query: str, **kwargs: Any) -> Any:
        assert kwargs["user_scope"] == "isolated-scope"
        assert kwargs["threshold"] == 0.1
        assert kwargs["rendering_mode"] == MEM0_CERTIFIED_RENDERING
        index = self.stats.search_calls
        self.stats = replace(
            self.stats,
            search_calls=index + 1,
            search_latency_s=self.stats.search_latency_s + 0.01,
            search_returned_memories=self.stats.search_returned_memories + 1,
        )
        candidate = SimpleNamespace(
            rank=1,
            memory_id=f"memory-{index}",
            text=f"answer {index}",
            score=0.99,
            created_at=f"2025-01-{index + 1:02d}T00:00:00Z",
            attribution_kind=MEM0_ATTRIBUTION_KIND,
        )
        return SimpleNamespace(
            query=query,
            raw_pool=(candidate,),
            official_longmemeval_protocol=True,
            official_search_protocol=True,
            rendering_mode=MEM0_CERTIFIED_RENDERING,
            certified_rendering=True,
            comparison_certified=True,
            runtime_identity=_runtime_identity(),
            attribution_kind=MEM0_ATTRIBUTION_KIND,
            supports_exact_source_provenance=False,
            stats=self.stats,
        )

    def cleanup(self) -> None:
        self.cleanup_order.append("cleanup")
        self.restored_before_cleanup = (
            self.memory.add is self.original_add
            and self.llm.generate_response is self.original_generate_response
        )
        self._ledger.clear()
        self._scopes.clear()
        self._scope_protocol.clear()
        self._closed = True
        self.owned_backend._closed = True
        (self.state / "owned").unlink()
        self.state.rmdir()
        self.active_user_scope = None
        self.cleaned = True


def _run_stage_a(
    tmp_path: Path,
    *,
    source_environment_lock_sha256: str = LOCK_SHA256,
) -> tuple[RawStressShard, Any, _FakeAdapter]:
    shard = _shard()
    adapters: list[_FakeAdapter] = []

    def factory(state: Path) -> _FakeAdapter:
        adapter = _FakeAdapter(state)
        adapters.append(adapter)
        return adapter

    result = run_retrieval_stage(
        shard=shard,
        authorization=_retrieval_authorization(
            shard,
            source_environment_lock_sha256=source_environment_lock_sha256,
        ),
        mem0_environment_lock_path=LOCK_PATH,
        owned_state_dir=tmp_path / "state",
        artifact_path=tmp_path / "retrieval.json",
        trace_path=tmp_path / "retrieval.trace.json",
        adapter_factory=factory,
        process_guard=ShardProcessGuard("test-retrieval"),
    )
    return shard, result, adapters[0]


def _scoring_authorization(
    shard: RawStressShard,
    artifact_sha256: str,
    *,
    root_environment_lock_sha256: str = LOCK_SHA256,
) -> ScoringStageAuthorization:
    extraction, embedder = _model_identities()
    return ScoringStageAuthorization(
        sample_offset=shard.sample_offset,
        sample_sha256=shard.sample_sha256,
        raw_history_bundle_sha256=shard.raw_history_bundle_sha256,
        question_ids=shard.question_ids,
        retrieval_artifact_sha256=artifact_sha256,
        source_validation_policy_sha256=SHA_E,
        source_implementation_sha256=SHA_F,
        source_environment_lock_sha256=root_environment_lock_sha256,
        mem0_policy_sha256=SHA_2,
        mem0_tool_implementation_sha256=tool_implementation_sha256(),
        mem0_environment_lock_sha256=LOCK_SHA256,
        mem0_stable_config_sha256=SHA_C,
        source_evaluation_identity=_source_evaluation_identity(),
        mem0_stable_payload=STABLE_PAYLOAD,
        scoring_policy_sha256="4" * 64,
        responder_model=MEM0_SOURCE_RESPONDER_MODEL,
        judge_model=MEM0_SOURCE_JUDGE_MODEL,
        responder_model_identity_sha256="5" * 64,
        judge_model_identity_sha256="6" * 64,
        extraction_model_identity=extraction,
        extraction_model_identity_sha256=canonical_json_sha256(extraction),
        embedder_model_identity=embedder,
        embedder_model_identity_sha256=canonical_json_sha256(embedder),
    )


def _self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(body)
    return {**row, "receipt_sha256": canonical_json_sha256(row)}


def _resumable_retrieval_fixture(
    tmp_path: Path,
) -> tuple[RawStressShard, Any, Path, Path]:
    shard, legacy, _adapter = _run_stage_a(tmp_path)
    retrieval_authorization = _retrieval_authorization(shard)
    retrieval_authorization_sha256 = canonical_json_sha256(
        asdict(retrieval_authorization)
    )
    plan = resumable_runner.build_resume_plan(
        shard=shard,
        authorization=retrieval_authorization,
        authorization_sha256=retrieval_authorization_sha256,
    )
    exact_adds = shard.add_counts.add_requests
    extraction, embedder_identity = _model_identities()
    logical_segment = {
        "kind": "mem0_memory_llm_generate_response_logical_calls",
        "boundary": "Memory.llm.generate_response",
        "count_semantics": "local_logical_wrapper_calls_not_http_attempts",
        "external_http_attempts_certified": False,
        "authorized_local_wrapper_retries": 0,
        "external_retry_attempts_certified": False,
        "authorized": exact_adds,
        "attempted": exact_adds,
        "completed": exact_adds,
        "failed": 0,
        "rejected": 0,
        "infer_true_adds_started": exact_adds,
        "infer_true_adds_exactly_one_call": exact_adds,
        "one_logical_call_per_infer_true_add_certified": True,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    logical = {
        "kind": "resumable_cumulative_logical_extraction",
        "authorized": exact_adds,
        "seeded_prefix": 0,
        "segment_authorized": exact_adds,
        "attempted": exact_adds,
        "completed": exact_adds,
        "failed": 0,
        "rejected": 0,
        "segment_receipt": logical_segment,
        "segment_receipt_sha256": canonical_json_sha256(logical_segment),
        "retries_authorized": 0,
        "infer_true_adds_started": exact_adds,
        "infer_true_adds_exactly_one_call": exact_adds,
    }
    request_identity = {
        "format": "memory-condense-mem0-extraction-request-v1",
        "route_identity_sha256": SHA_A,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 2_000,
        "sampling_parameters": "omitted",
        "sdk_retries": 0,
        "http_transport_retries": 0,
        "follow_redirects": False,
        "trust_env": False,
        "timeout_seconds": 600.0,
        "connect_timeout_seconds": 30.0,
        "cap_boundary": "httpx.BaseTransport.handle_request",
    }
    transport_segment = {
        "kind": "local_transport_send_cap",
        "role": "extraction",
        "authorized": exact_adds,
        "attempted": exact_adds,
        "completed": exact_adds,
        "failed": 0,
        "rejected": 0,
        "retries_authorized": 0,
        "provider_usage_status": "provider_reported_exact",
        "provider_usage_records": exact_adds,
        "provider_input_tokens": 120,
        "provider_output_tokens": 20,
        "provider_total_tokens": 140,
        "provider_latency_s": 1.25,
        "production_eligible": True,
        "provider": extraction["provider"],
        "model": extraction["model"],
        "revision": extraction["revision"],
        "route_identity_sha256": SHA_A,
        "request_identity_sha256": canonical_json_sha256(request_identity),
        "gateway_url": "https://controlled.invalid/v1",
        "max_completion_tokens": 2_000,
        "sampling_parameters_omitted": True,
        "sdk_retries": 0,
        "http_transport_retries": 0,
        "follow_redirects": False,
        "trust_env": False,
        "cap_boundary": "httpx.BaseTransport.handle_request",
        "external_http_attempts_certified": True,
        "external_provider_persistence_certified": False,
    }
    transport = {
        "kind": "resumable_cumulative_http_transport",
        "authorized": exact_adds,
        "seeded_prefix": 0,
        "segment_authorized": exact_adds,
        "attempted": exact_adds,
        "completed": exact_adds,
        "failed": 0,
        "rejected": 0,
        "segment_receipt": transport_segment,
        "segment_receipt_sha256": canonical_json_sha256(transport_segment),
        "retries_authorized": 0,
        "provider_usage_status": "provider_reported_exact",
        "provider_usage_records": exact_adds,
        "provider_input_tokens": 120,
        "provider_output_tokens": 20,
        "provider_total_tokens": 140,
        "provider_latency_s": 1.25,
    }
    bound_embedder = _self_hashed(
        {
            "format": "memory-condense-bound-mem0-bge-m3-v1",
            "concrete_type": (
                "mem0.embeddings.huggingface.HuggingFaceEmbedding"
            ),
            "model": embedder_identity["model"],
            "revision": embedder_identity["revision"],
            "authorized_checkpoint_sha256": embedder_identity[
                "checkpoint_sha256"
            ],
            "checkpoint_bytes_rehashed_per_factory": False,
            "dimension": embedder_identity["dimension"],
            "device": embedder_identity["device"],
            "local_files_only": True,
            "trust_remote_code": False,
            "network_calls_authorized": 0,
        }
    )
    bound_bm25 = _self_hashed(
        {
            "format": "memory-condense-bound-mem0-bm25-v1",
            "model": "Qdrant/bm25",
            "revision": "synthetic-locked-revision",
            "asset_tree_sha256": SHA_D,
            "cache_root": "C:/controlled/cache",
            "file_count": 1,
            "local_files_only": True,
            "network_calls_authorized": 0,
            "specific_model_path": "C:/controlled/cache/snapshot",
            "retry_sleep_attempts": 0,
            "bound_store_roles": ["memory", "entity"],
            "encoder_instances": 2,
            "distinct_encoder_instances": True,
            "internal_lazy_download_path_reachable": False,
        }
    )
    zero_transport = {
        "kind": "local_transport_send_cap",
        "role": "extraction",
        "authorized": 0,
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "rejected": 0,
        "retries_authorized": 0,
        "provider_usage_status": "not_applicable_zero_authorized",
        "provider_usage_records": 0,
        "provider_input_tokens": 0,
        "provider_output_tokens": 0,
        "provider_total_tokens": 0,
        "provider_latency_s": 0.0,
        "production_eligible": True,
        "provider": extraction["provider"],
        "model": extraction["model"],
        "revision": extraction["revision"],
        "route_identity_sha256": SHA_A,
        "gateway_url": "https://controlled.invalid/v1",
        "sdk_retries": 0,
        "http_transport_retries": 0,
        "cap_boundary": "deny_before_provider_transport",
        "external_http_attempts_certified": True,
        "external_provider_persistence_certified": False,
    }
    factory = _self_hashed(
        {
            "format": "memory-condense-mem0-resumable-factory-v1",
            "mode": "adopt",
            "segment_authorized_calls": 0,
            "full_authorized_calls": exact_adds,
            "user_scope_sha256": plan.as_dict()["user_scope_sha256"],
            "bound_embedder": bound_embedder,
            "bound_bm25": bound_bm25,
            "transport": zero_transport,
        }
    )
    manifest = {
        ".memory-condense-owned-state": {
            "type": "file",
            "bytes": 32,
            "sha256": SHA_E,
        }
    }
    manifest_sha = canonical_json_sha256(manifest)
    state_tree = {
        "path_name": "working",
        "ownership_token_sha256": SHA_F,
        "manifest": manifest,
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_tree_sha256": canonical_json_sha256(
            {
                "ownership_token_sha256": SHA_F,
                "manifest_sha256": manifest_sha,
            }
        ),
        "file_count": 1,
        "total_bytes": 32,
    }
    ingestion_transport_closure = _self_hashed(
        {
            "format": "memory-condense-mem0-resumable-transport-closure-v1",
            "segment_authorized_calls": exact_adds,
            "transport_closed": True,
            "budget_closed_exactly": True,
            "provider_usage_complete": True,
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "transport_receipt": transport_segment,
            "transport_receipt_sha256": canonical_json_sha256(
                transport_segment
            ),
        }
    )
    live_launch_authority = {
        "format": "memory-condense-mem0-live-launch-authority-v1",
        "preflight_sha256": SHA_A,
        "launch_manifest_sha256": SHA_B,
        "shard_launch_sha256": SHA_D,
        "shard_launch_payload_sha256": SHA_E,
        "plan_sha256": plan.sha256,
        "authorization_sha256": retrieval_authorization_sha256,
        "journal_path_sha256": SHA_F,
        "sample_offset": shard.sample_offset,
        "namespace": plan.as_dict()["user_scope"],
        "namespace_sha256": plan.as_dict()["user_scope_sha256"],
        "mem0_policy_sha256": plan.as_dict()["mem0_policy_sha256"],
        "mem0_tool_implementation_sha256": plan.as_dict()[
            "mem0_tool_implementation_sha256"
        ],
        "mem0_environment_lock_sha256": plan.as_dict()[
            "mem0_environment_lock_sha256"
        ],
        "retained_transformer_token_state_bytes": 0,
    }
    segment_authorization = _self_hashed(
        {
            "format": (
                "memory-condense-mem0-one-use-segment-authorization-v1"
            ),
            "plan_sha256": plan.sha256,
            "authorization_sha256": retrieval_authorization_sha256,
            "journal_path_sha256": SHA_F,
            "prefix_before": 0,
            "prefix_after": exact_adds,
            "generation": 0,
            "prior_checkpoint_authority_sha256": SHA_3,
            "authorized_provider_calls": exact_adds,
            "authorized_add_operations": exact_adds,
            "provider_retries": 0,
            "namespace": plan.as_dict()["user_scope"],
            "retained_transformer_token_state_bytes": 0,
            "live_launch_authority": live_launch_authority,
            "live_launch_authority_sha256": canonical_json_sha256(
                live_launch_authority
            ),
        }
    )
    write_activity = _self_hashed(
        {
            "format": "memory-condense-mem0-resumable-write-activity-v1",
            "embedding_attempted": exact_adds,
            "embedding_completed": exact_adds,
            "embedding_failed": 0,
            "embedding_input_token_proxy": 4,
            "embedding_latency_s": 0.5,
            "storage_attempted": exact_adds,
            "storage_completed": exact_adds,
            "storage_failed": 0,
            "storage_latency_s": 0.25,
            "wrappers_installed": True,
            "wrappers_restored": True,
        }
    )
    observed_write = {
        "add_attempted": exact_adds,
        "add_completed": exact_adds,
        "add_failed": 0,
        "extraction_attempted": exact_adds,
        "extraction_completed": exact_adds,
        "extraction_failed": 0,
        "extraction_raw_message_token_proxy": 0,
        "extraction_provider_input_tokens": 120,
        "extraction_provider_output_tokens": 20,
        "extraction_usage_status": "provider_reported_exact",
        "embedding_operations": exact_adds,
        "embedding_input_token_proxy": 4,
        "returned_memory_count": exact_adds,
        "persisted_memory_count": exact_adds,
        "persisted_storage_bytes": state_tree["total_bytes"],
        "add_latency_s": 0.0,
        "extraction_latency_s": 1.25,
        "embedding_latency_s": 0.5,
        "storage_latency_s": 0.25,
    }
    write_usage = _self_hashed(
        {
            "format": (
                "memory-condense-mem0-complete-write-usage-attestation-v1"
            ),
            "plan_sha256": plan.sha256,
            "authorization_sha256": retrieval_authorization_sha256,
            "generation": 0,
            "committed_prefix": exact_adds,
            "prior_write_usage_attestation_sha256": None,
            "segment_authorization_receipt": segment_authorization,
            "segment_authorization_receipt_sha256": segment_authorization[
                "receipt_sha256"
            ],
            "segment_write_activity_receipt": write_activity,
            "segment_write_activity_receipt_sha256": write_activity[
                "receipt_sha256"
            ],
            "transport_closure_receipt_sha256": (
                ingestion_transport_closure["receipt_sha256"]
            ),
            "observed": observed_write,
            "observed_sha256": canonical_json_sha256(observed_write),
            "retained_transformer_token_state_bytes": 0,
        }
    )
    terminal_transport_closure = _self_hashed(
        {
            "format": "memory-condense-mem0-resumable-transport-closure-v1",
            "segment_authorized_calls": 0,
            "transport_closed": True,
            "budget_closed_exactly": True,
            "provider_usage_complete": True,
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "transport_receipt": zero_transport,
            "transport_receipt_sha256": canonical_json_sha256(zero_transport),
        }
    )
    suspend = _self_hashed(
        {
            "format": "memory-condense-mem0-resumable-suspend-v1",
            "history_sqlite_closed": True,
            "qdrant_local_collections_closed": 2,
            "qdrant_clients_closed": 1,
            "qdrant_local_registries_closed": 1,
            "transport_closed": True,
            "transport_closure": terminal_transport_closure,
            "transport_closure_sha256": terminal_transport_closure[
                "receipt_sha256"
            ],
            "delete_col_calls": 0,
            "owned_state_retained": True,
            "owned_state_tree": state_tree,
            "namespace_persisted_memory_count": exact_adds,
        }
    )
    active_commit = SHA_1
    checkpoint = SHA_2
    seal_body = {
        "format": "memory-condense-mem0-resume-journal-v2",
        "kind": "prefix_sealed",
        "sequence": 7,
        "previous_entry_sha256": active_commit,
        "generation": 0,
        "committed_prefix": exact_adds,
        "active_commit_entry_sha256": active_commit,
        "snapshot_path": "state/snapshots/prefix-000002",
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_tree_sha256": state_tree["snapshot_tree_sha256"],
        "ownership_token_sha256": SHA_F,
        "handles_closed_receipt_sha256": SHA_3,
        "transport_closure_receipt_sha256": ingestion_transport_closure[
            "receipt_sha256"
        ],
        "write_usage_attestation": write_usage,
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
        "snapshot_authority_sha256": checkpoint,
        "snapshot_authority_artifact_sha256": SHA_D,
        "snapshot_receipt_sha256": SHA_E,
        "rehydration_sha256": SHA_F,
        "cumulative_extraction_attempted": exact_adds,
        "cumulative_extraction_completed": exact_adds,
        "cumulative_http_attempted": exact_adds,
        "cumulative_http_completed": exact_adds,
        "failures": 0,
        "rejections": 0,
    }
    seal = {**seal_body, "entry_sha256": canonical_json_sha256(seal_body)}
    execution_body = {
        "kind": "exact_mem0_resumable_execution_v2",
        "comparison_certified": True,
        "external_http_attempts_certified": True,
        "external_provider_persistence_certified": False,
        "authorization_sha256": retrieval_authorization_sha256,
        "plan_sha256": plan.sha256,
        "checkpoint_authority_sha256": checkpoint,
        "full_prefix": exact_adds,
        "active_commit_entry_sha256": active_commit,
        "logical_meter_receipt_sha256": canonical_json_sha256(logical),
        "transport_receipt_sha256": canonical_json_sha256(transport),
        "transport_closure_receipt_sha256": ingestion_transport_closure[
            "receipt_sha256"
        ],
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
        "source_implementation_sha256": (
            retrieval_authorization.source_implementation_sha256
        ),
        "source_environment_lock_sha256": (
            retrieval_authorization.source_environment_lock_sha256
        ),
        "mem0_tool_implementation_sha256": (
            retrieval_authorization.mem0_tool_implementation_sha256
        ),
        "mem0_environment_lock_sha256": (
            retrieval_authorization.mem0_environment_lock_sha256
        ),
        "terminal_factory_receipt_sha256": factory["receipt_sha256"],
        "terminal_suspend_receipt_sha256": suspend["receipt_sha256"],
    }
    execution = _self_hashed(execution_body)
    result = json.loads(legacy.artifact_path.read_text(encoding="utf-8"))
    for field in ("content_sha256", "retrieval_trace", "environment_lock"):
        result.pop(field, None)
    result.update(
        format=resumable_runner.RESUMABLE_TERMINAL_RESULT_FORMAT,
        certification_status="exact_resumable_production",
        comparison_certified=True,
        execution_binding=execution,
    )
    result["identity"]["runtime_model_identity_probe"] = {
        "kind": "exact_resumable_factory_bound",
        "bound_embedder_receipt_sha256": bound_embedder["receipt_sha256"],
        "bound_bm25_receipt_sha256": bound_bm25["receipt_sha256"],
        "comparison_certified": True,
    }
    result["ingestion_receipt"]["comparison_certified"] = True
    result["ingestion_receipt"]["user_scope_sha256"] = plan.as_dict()[
        "user_scope_sha256"
    ]
    result["search_receipt"]["extraction_transport_calls_during_search"] = 0
    result["mem0_usage"].update(
        provider_prompt_tokens=120,
        provider_completion_tokens=20,
        provider_usage_status="provider_reported_exact",
        token_counter_identity="synthetic-exact-token-counter-v1",
        token_counter_identity_verified=True,
    )
    result["provenance"]["external_http_attempts_certified"] = True
    result["provenance"]["external_retry_attempts_certified"] = True
    result["provenance"]["provider_usage_status"] = "provider_reported_exact"
    result["write_usage_attestation"] = write_usage
    result["resumable_closure"] = {
        "plan_sha256": plan.sha256,
        "resume_plan": plan.as_dict(),
        "checkpoint_authority_sha256": checkpoint,
        "active_commit_entry_sha256": active_commit,
        "journal_tail_entry_sha256": seal["entry_sha256"],
        "journal_chain_sha256": SHA_A,
        "commit_population_sha256": SHA_B,
        "full_prefix_seal": seal,
        "logical_meter_receipt": logical,
        "transport_receipt": transport,
        "transport_closure_receipt_sha256": ingestion_transport_closure[
            "receipt_sha256"
        ],
        "write_usage_attestation": write_usage,
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
        "factory_receipt": factory,
        "suspend_receipt": suspend,
    }
    search_events = [
        {
            "sequence": index,
            "question_id": row["question_id"],
            "query_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
            "raw_memory_count": row["raw_memory_count"],
            "raw_pool_sha256": row["raw_pool_sha256"],
            "retrieval_row_sha256": row["retrieval_row_sha256"],
        }
        for index, row in enumerate(result["retrieval_rows"], start=1)
    ]
    stage_trace = {
        "format": resumable_runner.RESUMABLE_TERMINAL_TRACE_FORMAT,
        "status": "search_staged",
        "sample_offset": shard.sample_offset,
        "sample_id": shard.parsed_sample.sample_id,
        "sample_sha256": shard.sample_sha256,
        "plan_sha256": plan.sha256,
        "checkpoint_authority_sha256": checkpoint,
        "events": search_events,
        "completed_search_operations": len(shard.question_ids),
        "extraction_transport_calls": 0,
        "handles_closed_receipt_sha256": suspend["receipt_sha256"],
        "checkpoint_retained": True,
        "transport_closure_receipt_sha256": ingestion_transport_closure[
            "receipt_sha256"
        ],
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
    }
    stage = {
        "format": resumable_runner.RESUME_TERMINAL_FORMAT,
        "plan_sha256": plan.sha256,
        "authorization_sha256": retrieval_authorization_sha256,
        "committed_prefix": exact_adds,
        "full_checkpoint_authority_sha256": checkpoint,
        "completed_search_operations": len(shard.question_ids),
        "extraction_calls_closed": True,
        "provider_retries": 0,
        "transport_closure_receipt_sha256": ingestion_transport_closure[
            "receipt_sha256"
        ],
        "write_usage_attestation_sha256": write_usage["receipt_sha256"],
        "terminal_result_sha256": canonical_json_sha256(result),
        "terminal_trace_sha256": canonical_json_sha256(stage_trace),
        "result": result,
        "trace": stage_trace,
    }
    stage_file_sha = hashlib.sha256(
        run_shard_module._canonical_bytes(stage) + b"\n"
    ).hexdigest()
    state = SimpleNamespace(
        checkpoint_authority_sha256=checkpoint,
        terminal_search={"terminal_stage_sha256": stage_file_sha},
    )
    artifact_path = tmp_path / "resumable-retrieval.json"
    trace_path = tmp_path / "resumable-retrieval.trace.json"
    artifact_bytes, trace_bytes, _artifact, _trace = (
        resumable_runner._official_terminal_payloads(
            stage=stage,
            state=state,
            artifact_target=artifact_path,
            trace_target=trace_path,
            environment_lock_path=LOCK_PATH,
            environment_lock_sha256=LOCK_SHA256,
        )
    )
    artifact_path.write_bytes(artifact_bytes)
    trace_path.write_bytes(trace_bytes)
    authorization = _scoring_authorization(
        shard, hashlib.sha256(artifact_bytes).hexdigest()
    )
    return shard, authorization, artifact_path, trace_path


def _successful_scoring_provider(
    calls: list[str],
) -> Any:
    def provider(
        _messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        del max_output_tokens
        calls.append(model)
        is_judge = model == MEM0_SOURCE_JUDGE_MODEL
        return ProviderCallResult(
            text="CORRECT synthetic verdict" if is_judge else "answer",
            usage=UsageStats(input_tokens=10, output_tokens=2, calls=1),
        )

    return _declare_stateless(provider)


def test_prepared_corpus_preserves_locked_batch_order_and_messages() -> None:
    shard = _shard()
    corpus = build_adapter_prepared_corpus(shard)

    assert corpus.sample_id == shard.parsed_sample.sample_id
    assert corpus.official_longmemeval_protocol is True
    assert [batch.messages for batch in corpus.batches] == [
        row.messages for row in shard.add_batches
    ]
    assert [batch.ref.source for batch in corpus.batches] == [
        "q-0::session-1",
        "q-1::session-9",
    ]
    assert corpus.batches[0].ref.roles == ("user", "assistant")
    assert corpus.batches[1].ref.turn_count == 1
    assert corpus.batches[0].ref.original_session_index == 2


def test_retrieval_stage_publishes_only_after_cleanup(tmp_path: Path) -> None:
    shard, result, adapter = _run_stage_a(tmp_path)

    assert adapter.cleaned is True
    assert adapter.restored_before_cleanup is True
    assert not (tmp_path / "state").exists()
    assert result.artifact_path.is_file()
    assert result.trace_path.is_file()
    assert result.artifact["question_ids"] == list(shard.question_ids)
    assert result.artifact["certification_status"] == "injected_nonproduction"
    assert result.artifact["comparison_certified"] is False
    assert result.artifact["execution_binding"] == {
        "kind": "injected_nonproduction",
        "trusted_runtime_binding_receipt_sha256": None,
        "comparison_certified": False,
        "external_http_attempts_certified": False,
        "external_provider_persistence_certified": False,
    }
    assert result.artifact["ingestion_receipt"]["completed_add_operations"] == 2
    assert result.artifact["ingestion_receipt"]["extraction_model_calls"] == {
        "kind": "mem0_memory_llm_generate_response_logical_calls",
        "boundary": "Memory.llm.generate_response",
        "count_semantics": "local_logical_wrapper_calls_not_http_attempts",
        "external_http_attempts_certified": False,
        "authorized_local_wrapper_retries": 0,
        "external_retry_attempts_certified": False,
        "authorized": 2,
        "attempted": 2,
        "completed": 2,
        "failed": 0,
        "rejected": 0,
        "infer_true_adds_started": 2,
        "infer_true_adds_exactly_one_call": 2,
        "one_logical_call_per_infer_true_add_certified": True,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    assert result.artifact["search_receipt"]["completed_search_operations"] == 10
    assert len(result.artifact["retrieval_rows"]) == 10
    assert all(
        len(row["messages"]) == 2
        and row["max_prompt_token_proxy"] == 8_000
        and row["independently_verified"] is True
        for row in result.artifact["retrieval_rows"]
    )
    assert result.trace["cleanup"]["state_absent_after"] is True
    assert result.trace["cleanup"]["active_scope_cleared"] is True
    assert result.artifact["ingestion_receipt"][
        "persisted_request_token_state"
    ] is False
    assert result.artifact["ingestion_receipt"][
        "retained_request_token_state_bytes"
    ] == 0
    assert (
        result.trace["cleanup"]["extraction_meter_restored_before_cleanup"]
        is True
    )
    assert all(
        result.trace["cleanup"][field] is True
        for field in (
            "adapter_closed",
            "ledger_empty",
            "registered_scopes_empty",
            "scope_protocol_empty",
            "backend_closed_or_cleared",
            "owned_state_path_absent",
        )
    )
    assert result.artifact["environment_lock"] == {
        "filename": LOCK_PATH.name,
        "authorized_sha256": LOCK_SHA256,
        "sha256_before": LOCK_SHA256,
        "sha256_after": LOCK_SHA256,
        "unchanged": True,
    }
    assert not list(tmp_path.glob("*.staging"))


def test_retrieval_pair_publication_rolls_back_first_complete_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = _shard()
    adapters: list[_FakeAdapter] = []

    def factory(state: Path) -> _FakeAdapter:
        adapter = _FakeAdapter(state)
        adapters.append(adapter)
        return adapter

    authorization = _retrieval_authorization(shard)
    real_link = run_shard_module.os.link
    link_calls = 0

    def fail_second_link(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("synthetic second-link failure")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(run_shard_module.os, "link", fail_second_link)
    artifact_path = tmp_path / "retrieval.json"
    trace_path = tmp_path / "retrieval.trace.json"
    with pytest.raises(Mem0ShardRunError) as raised:
        run_retrieval_stage(
            shard=shard,
            authorization=authorization,
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=artifact_path,
            trace_path=trace_path,
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-publication-rollback"),
        )

    assert raised.value.trace_path is None
    assert link_calls == 2
    assert adapters[0].cleaned is True
    assert not artifact_path.exists()
    assert not trace_path.exists()
    assert not list(tmp_path.glob("*.staging"))


def test_retrieval_rejects_output_under_frozen_tool_root_before_adapter(
    tmp_path: Path,
) -> None:
    shard = _shard()
    called = False

    def factory(_state: Path) -> _FakeAdapter:
        nonlocal called
        called = True
        raise AssertionError("protected output must fail before adapter construction")

    protected_artifact = (
        Path(run_shard_module.__file__).resolve().parent
        / "forbidden-publication.json"
    )
    failure_trace = tmp_path / "retrieval.failure.trace.json"
    with pytest.raises(Mem0ShardRunError) as raised:
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=protected_artifact,
            trace_path=failure_trace,
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-protected-output"),
        )

    assert called is False
    assert not protected_artifact.exists()
    assert raised.value.trace_path == failure_trace
    assert json.loads(failure_trace.read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize(
    ("calls_per_add", "fail_call", "swallow_failure", "expected"),
    [
        ((0, 1), None, False, (0, 0, 0, 0)),
        ((2, 1), None, False, (1, 1, 0, 1)),
        ((1, 1), 1, True, (2, 1, 1, 0)),
    ],
    ids=("under-call", "over-call", "swallowed-provider-failure"),
)
def test_retrieval_extraction_meter_fails_closed_and_restores_before_cleanup(
    tmp_path: Path,
    calls_per_add: tuple[int, int],
    fail_call: int | None,
    swallow_failure: bool,
    expected: tuple[int, int, int, int],
) -> None:
    shard = _shard()
    adapters: list[_FakeAdapter] = []

    def factory(state: Path) -> _FakeAdapter:
        adapter = _FakeAdapter(
            state,
            extraction_calls_per_add=calls_per_add,
            fail_extraction_call=fail_call,
            swallow_extraction_failure=swallow_failure,
        )
        adapters.append(adapter)
        return adapter

    artifact_path = tmp_path / "retrieval.json"
    trace_path = tmp_path / "retrieval.failure.trace.json"
    with pytest.raises(Mem0ShardRunError) as raised:
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=artifact_path,
            trace_path=trace_path,
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-extraction-failure"),
        )

    assert raised.value.trace_path == trace_path
    assert not artifact_path.exists()
    assert len(adapters) == 1
    adapter = adapters[0]
    assert adapter.cleaned is True
    assert adapter.restored_before_cleanup is True
    assert not (tmp_path / "state").exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert trace["cleanup"]["extraction_meter_restored_before_cleanup"] is True
    receipt = trace["cleanup"]["extraction_model_calls"]
    assert (
        receipt["attempted"],
        receipt["completed"],
        receipt["failed"],
        receipt["rejected"],
    ) == expected
    assert receipt["external_http_attempts_certified"] is False
    assert receipt["authorized_local_wrapper_retries"] == 0
    assert receipt["external_retry_attempts_certified"] is False
    assert receipt["one_logical_call_per_infer_true_add_certified"] is False


def test_retrieval_rejects_swallowed_model_calls_outside_infer_add(
    tmp_path: Path,
) -> None:
    shard = _shard()
    adapters: list[_FakeAdapter] = []

    class SearchCallingAdapter(_FakeAdapter):
        def search(self, query: str, **kwargs: Any) -> Any:
            try:
                self.llm.generate_response(messages=[])
            except Mem0ShardRunError:
                pass
            return super().search(query, **kwargs)

    def factory(state: Path) -> _FakeAdapter:
        adapter = SearchCallingAdapter(state)
        adapters.append(adapter)
        return adapter

    artifact_path = tmp_path / "retrieval.json"
    trace_path = tmp_path / "retrieval.failure.trace.json"
    with pytest.raises(Mem0ShardRunError):
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=artifact_path,
            trace_path=trace_path,
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-search-llm-call"),
        )

    assert not artifact_path.exists()
    assert adapters[0].restored_before_cleanup is True
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    receipt = trace["cleanup"]["extraction_model_calls"]
    assert receipt["attempted"] == receipt["completed"] == 2
    assert receipt["rejected"] == 10
    assert receipt["one_logical_call_per_infer_true_add_certified"] is False


def test_retrieval_authorization_fails_before_adapter_factory(tmp_path: Path) -> None:
    shard = _shard()
    authorization = replace(
        _retrieval_authorization(shard), authorized_add_operations=3
    )
    called = False

    def factory(_state: Path) -> _FakeAdapter:
        nonlocal called
        called = True
        raise AssertionError("must not run")

    with pytest.raises(Mem0ShardRunError) as raised:
        run_retrieval_stage(
            shard=shard,
            authorization=authorization,
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=tmp_path / "artifact.json",
            trace_path=tmp_path / "failure.trace.json",
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-preflight"),
        )

    assert called is False
    assert raised.value.trace_path == tmp_path / "failure.trace.json"
    trace = json.loads(raised.value.trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert trace["cleanup"]["attempted"] is False
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("nested_output", ["artifact", "trace"])
def test_retrieval_rejects_outputs_under_owned_state_without_recreating_it(
    tmp_path: Path, nested_output: str
) -> None:
    shard = _shard()
    state = tmp_path / "state"
    artifact = (
        state / "retrieval.json"
        if nested_output == "artifact"
        else tmp_path / "retrieval.json"
    )
    trace = (
        state / "retrieval.trace.json"
        if nested_output == "trace"
        else tmp_path / "retrieval.failure.trace.json"
    )
    called = False

    def factory(_state: Path) -> _FakeAdapter:
        nonlocal called
        called = True
        raise AssertionError("unsafe output paths must fail before construction")

    with pytest.raises(Mem0ShardRunError) as raised:
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=state,
            artifact_path=artifact,
            trace_path=trace,
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-nested-output"),
        )

    assert called is False
    assert not state.exists()
    assert not artifact.exists()
    if nested_output == "trace":
        assert raised.value.trace_path is None
        assert not trace.exists()
    else:
        assert raised.value.trace_path == trace
        assert trace.is_file()


def test_retrieval_rejects_mutated_add_message_before_any_output_or_adapter(
    tmp_path: Path,
) -> None:
    shard = _shard()
    first = shard.add_batches[0]
    object.__setattr__(
        first,
        "messages",
        (("user", "tampered"), ("assistant", "beta")),
    )
    called = False

    def factory(_state: Path) -> _FakeAdapter:
        nonlocal called
        called = True
        raise AssertionError("mutated shards must not reach adapter construction")

    with pytest.raises(Mem0ShardRunError) as raised:
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=tmp_path / "retrieval.json",
            trace_path=tmp_path / "retrieval.trace.json",
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-mutated-message"),
        )

    assert called is False
    assert raised.value.trace_path is None
    assert not list(tmp_path.iterdir())


def test_retrieval_hashes_environment_lock_internally_before_adapter(
    tmp_path: Path,
) -> None:
    shard = _shard()
    lock_path = tmp_path / "isolated.lock"
    lock_path.write_bytes(b"different lock")
    called = False

    def factory(_state: Path) -> _FakeAdapter:
        nonlocal called
        called = True
        raise AssertionError("lock mismatch must precede adapter construction")

    with pytest.raises(Mem0ShardRunError):
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=lock_path,
            owned_state_dir=tmp_path / "state",
            artifact_path=tmp_path / "retrieval.json",
            trace_path=tmp_path / "retrieval.failure.trace.json",
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-lock-mismatch"),
        )

    assert called is False
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "retrieval.json").exists()


def test_retrieval_rechecks_environment_lock_after_cleanup(tmp_path: Path) -> None:
    shard = _shard()
    lock_path = tmp_path / "isolated.lock"
    lock_path.write_bytes(b"initial lock")
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    adapters: list[_FakeAdapter] = []

    class LockMutatingAdapter(_FakeAdapter):
        def _ingest_prepared(self, corpus: Any) -> Any:
            result = super()._ingest_prepared(corpus)
            lock_path.write_bytes(b"mutated lock")
            return result

    def factory(state: Path) -> _FakeAdapter:
        adapter = LockMutatingAdapter(state)
        adapters.append(adapter)
        return adapter

    artifact_path = tmp_path / "retrieval.json"
    trace_path = tmp_path / "retrieval.failure.trace.json"
    with pytest.raises(Mem0ShardRunError):
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(
                shard, mem0_environment_lock_sha256=lock_sha
            ),
            mem0_environment_lock_path=lock_path,
            owned_state_dir=tmp_path / "state",
            artifact_path=artifact_path,
            trace_path=trace_path,
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-lock-recheck"),
        )

    assert adapters[0].cleaned is True
    assert not artifact_path.exists()
    failed = json.loads(trace_path.read_text(encoding="utf-8"))
    assert failed["cleanup"]["environment_lock"]["unchanged"] is False


def test_retrieval_rejects_residual_in_process_memory_after_cleanup(
    tmp_path: Path,
) -> None:
    shard = _shard()

    class LeakyCleanupAdapter(_FakeAdapter):
        def cleanup(self) -> None:
            super().cleanup()
            self._ledger[("scope", "memory")] = ("retained memory text",)
            self.owned_backend._closed = False

    artifact_path = tmp_path / "retrieval.json"
    trace_path = tmp_path / "retrieval.failure.trace.json"
    with pytest.raises(Mem0ShardRunError):
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=artifact_path,
            trace_path=trace_path,
            adapter_factory=LeakyCleanupAdapter,
            process_guard=ShardProcessGuard("test-retrieval-residual-memory"),
        )

    assert not artifact_path.exists()
    cleanup = json.loads(trace_path.read_text(encoding="utf-8"))["cleanup"]
    assert cleanup["ledger_empty"] is False
    assert cleanup["backend_closed_or_cleared"] is False
    assert cleanup["owned_state_path_absent"] is True


def test_retrieval_failure_trace_does_not_hash_external_error_text(
    tmp_path: Path,
) -> None:
    shard = _shard()
    secret = "TOPSECRET-cleanup-message"

    class FailingCleanupAdapter(_FakeAdapter):
        def cleanup(self) -> None:
            raise RuntimeError(secret)

    trace_path = tmp_path / "retrieval.failure.trace.json"
    with pytest.raises(Mem0ShardRunError):
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=tmp_path / "retrieval.json",
            trace_path=trace_path,
            adapter_factory=FailingCleanupAdapter,
            process_guard=ShardProcessGuard("test-retrieval-error-oracle"),
        )

    rendered = trace_path.read_text(encoding="utf-8")
    assert secret not in rendered
    assert hashlib.sha256(secret.encode("utf-8")).hexdigest() not in rendered


@pytest.mark.parametrize(
    ("runtime_config", "secret"),
    [
        ({"api_key": "test-secret-must-not-escape"}, "test-secret-must-not-escape"),
        ({"Authorization": "Bearer TOPSECRET"}, "TOPSECRET"),
        ({"azure_openai_api_key": "TOPSECRET"}, "TOPSECRET"),
    ],
)
def test_retrieval_rejects_runtime_secret_and_never_publishes_it(
    tmp_path: Path, runtime_config: dict[str, str], secret: str
) -> None:
    shard = _shard()

    class SecretRuntimeAdapter(_FakeAdapter):
        def _ingest_prepared(self, corpus: Any) -> Any:
            result = super()._ingest_prepared(corpus)
            runtime = dict(result.runtime_identity)
            runtime["config"] = runtime_config
            result.runtime_identity = runtime
            return result

    artifact_path = tmp_path / "retrieval.json"
    trace_path = tmp_path / "retrieval.failure.trace.json"
    with pytest.raises(Mem0ShardRunError):
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=artifact_path,
            trace_path=trace_path,
            adapter_factory=SecretRuntimeAdapter,
            process_guard=ShardProcessGuard("test-retrieval-secret-runtime"),
        )

    assert not artifact_path.exists()
    assert secret not in trace_path.read_text(encoding="utf-8")


def test_retrieval_rejects_runtime_identity_before_first_add(
    tmp_path: Path,
) -> None:
    shard = _shard()
    adapters: list[_FakeAdapter] = []

    class WrongPreflightRuntimeAdapter(_FakeAdapter):
        def __init__(self, state: Path) -> None:
            super().__init__(state)
            wrong = dict(self.owned_backend.runtime_identity)
            wrong["config"] = {"provider": "unauthorized"}
            self.owned_backend.runtime_identity = wrong

    def factory(state: Path) -> _FakeAdapter:
        adapter = WrongPreflightRuntimeAdapter(state)
        adapters.append(adapter)
        return adapter

    artifact_path = tmp_path / "retrieval.json"
    trace_path = tmp_path / "retrieval.failure.trace.json"
    with pytest.raises(Mem0ShardRunError):
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=artifact_path,
            trace_path=trace_path,
            adapter_factory=factory,
            process_guard=ShardProcessGuard("test-retrieval-pre-add-runtime"),
        )

    assert adapters[0].provider_invocations == 0
    assert not artifact_path.exists()
    assert adapters[0].cleaned is True


@pytest.mark.parametrize(
    "forged_binding",
    [object(), object.__new__(TrustedRuntimeBinding)],
    ids=["arbitrary-object", "constructor-bypass"],
)
def test_arbitrary_binding_object_cannot_elevate_injected_core(
    tmp_path: Path,
    forged_binding: object,
) -> None:
    shard = _shard()
    with pytest.raises(Mem0ShardRunError) as raised:
        run_retrieval_stage(
            shard=shard,
            authorization=_retrieval_authorization(shard),
            mem0_environment_lock_path=LOCK_PATH,
            owned_state_dir=tmp_path / "state",
            artifact_path=tmp_path / "retrieval.json",
            trace_path=tmp_path / "retrieval.trace.json",
            adapter_factory=_FakeAdapter,
            trusted_runtime_binding=forged_binding,  # type: ignore[arg-type]
            process_guard=ShardProcessGuard("test-retrieval-forged-binding"),
        )

    assert raised.value.trace_path is None
    assert not list(tmp_path.iterdir())


def test_scoring_rejects_mutated_question_before_any_output_or_provider(
    tmp_path: Path,
) -> None:
    shard, retrieval, _adapter = _run_stage_a(tmp_path)
    shard.parsed_sample.questions[0].question = "tampered question"
    calls = 0

    def provider(
        _messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        del model, max_output_tokens
        nonlocal calls
        calls += 1
        raise AssertionError("mutated shards must not reach providers")

    _declare_stateless(provider)
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "scoring.trace.json"
    with pytest.raises(Mem0ShardRunError) as raised:
        run_scoring_stage(
            shard=shard,
            authorization=_scoring_authorization(
                shard, retrieval.artifact_sha256
            ),
            root_environment_lock_path=LOCK_PATH,
            retrieval_artifact_path=retrieval.artifact_path,
            retrieval_trace_path=retrieval.trace_path,
            report_path=report_path,
            scoring_trace_path=trace_path,
            responder=provider,
            judge=provider,
            process_guard=ShardProcessGuard("test-scoring-mutated-question"),
        )

    assert calls == 0
    assert raised.value.trace_path is None
    assert not report_path.exists()
    assert not trace_path.exists()


def test_scoring_stage_verifies_artifact_and_closes_exact_twenty_calls(
    tmp_path: Path,
) -> None:
    shard, retrieval, _adapter = _run_stage_a(tmp_path)
    responder_calls: list[tuple[Sequence[Mapping[str, str]], str, int]] = []
    judge_calls: list[tuple[Sequence[Mapping[str, str]], str, int]] = []

    def responder(
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        index = len(responder_calls)
        responder_calls.append((messages, model, max_output_tokens))
        return ProviderCallResult(
            text=f"answer {index}",
            usage=UsageStats(
                input_tokens=100,
                output_tokens=2,
                elapsed_s=0.01,
                calls=1,
            ),
        )

    def judge(
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        judge_calls.append((messages, model, max_output_tokens))
        return ProviderCallResult(
            text="CORRECT same answer",
            usage=UsageStats(
                input_tokens=50,
                output_tokens=3,
                elapsed_s=0.01,
                calls=1,
            ),
        )

    _declare_stateless(responder)
    _declare_stateless(judge)

    result = run_scoring_stage(
        shard=shard,
        authorization=_scoring_authorization(shard, retrieval.artifact_sha256),
        root_environment_lock_path=LOCK_PATH,
        retrieval_artifact_path=retrieval.artifact_path,
        retrieval_trace_path=retrieval.trace_path,
        report_path=tmp_path / "report.json",
        scoring_trace_path=tmp_path / "scoring.trace.json",
        responder=responder,
        judge=judge,
        process_guard=ShardProcessGuard("test-scoring"),
    )

    assert len(responder_calls) == len(judge_calls) == 10
    assert all(
        call[1:] == (MEM0_SOURCE_RESPONDER_MODEL, 256)
        for call in responder_calls
    )
    assert all(
        call[1:] == (MEM0_SOURCE_JUDGE_MODEL, 1024) for call in judge_calls
    )
    for index, (messages, _model, _max_output_tokens) in enumerate(judge_calls):
        question = shard.parsed_sample.questions[index]
        assert list(messages) == build_judge_prompt(
            question.question,
            question.answer,
            f"answer {index}",
        )
        assert "Reference date:" not in messages[1]["content"]
    assert result.report["run_status"] == "complete"
    assert result.report["certification_status"] == "injected_nonproduction"
    assert result.report["comparison_certified"] is False
    assert result.trace["certification_status"] == "injected_nonproduction"
    assert result.trace["comparison_certified"] is False
    assert result.report["scoring_receipt"][
        "answer_judge_logical_wrapper_calls"
    ] == 20
    assert result.report["scoring_receipt"][
        "persisted_request_token_state"
    ] is False
    assert result.report["scoring_receipt"][
        "retained_request_token_state_bytes"
    ] == 0
    assert result.trace["persisted_request_token_state"] is False
    assert result.trace["retained_request_token_state_bytes"] == 0
    assert result.trace["request_token_state_evidence_kind"] == (
        "local_injected_request_token_state_contract"
    )
    assert result.trace["external_provider_persistence_certified"] is False
    assert result.trace["external_http_attempts_certified"] is False
    assert result.report["scoring_receipt"][
        "responder_logical_wrapper_calls"
    ] == {
        "authorized": 10,
        "attempted": 10,
        "completed": 10,
        "failed": 0,
    }
    assert all(row["judge_correct"] for row in result.report["question_results"])
    assert result.trace["mem0_state_touched"] is False
    assert not list(tmp_path.glob("*.staging"))


def test_scoring_verifier_accepts_exact_resumable_terminal_artifact(
    tmp_path: Path,
) -> None:
    shard, authorization, artifact_path, trace_path = (
        _resumable_retrieval_fixture(tmp_path)
    )

    artifact, payload, rows = run_shard_module._verify_retrieval_artifact(
        artifact_path=artifact_path,
        trace_path=trace_path,
        shard=shard,
        authorization=authorization,
        prompt_packer=run_shard_module._default_prompt_packer,
    )

    assert artifact["certification_status"] == "exact_resumable_production"
    assert artifact["comparison_certified"] is True
    assert artifact["execution_binding"]["external_http_attempts_certified"] is True
    assert artifact["mem0_usage"]["provider_prompt_tokens"] == 120
    assert artifact["mem0_usage"]["provider_completion_tokens"] == 20
    assert artifact["mem0_usage"]["provider_usage_status"] == (
        "provider_reported_exact"
    )
    assert artifact["write_usage_attestation"]["observed"] == artifact[
        "resumable_closure"
    ]["write_usage_attestation"]["observed"]
    assert artifact["execution_binding"][
        "transport_closure_receipt_sha256"
    ] == artifact["resumable_closure"]["transport_closure_receipt_sha256"]
    assert len(payload) == artifact_path.stat().st_size
    assert len(rows) == len(shard.question_ids)


@pytest.mark.parametrize(
    "mutation",
    [
        "factory_extra_field",
        "http_route_digest",
        "terminal_stage_digest",
        "resume_plan_population",
        "partial_provider_usage",
        "mem0_provider_usage",
        "write_usage_duplicate",
        "transport_closure_digest",
    ],
)
def test_scoring_verifier_rejects_rehashed_resumable_proof_tamper(
    tmp_path: Path, mutation: str
) -> None:
    shard, _authorization, artifact_path, trace_path = (
        _resumable_retrieval_fixture(tmp_path)
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if mutation == "factory_extra_field":
        artifact["resumable_closure"]["factory_receipt"]["unexpected"] = True
    elif mutation == "http_route_digest":
        artifact["resumable_closure"]["transport_receipt"][
            "segment_receipt"
        ]["route_identity_sha256"] = "not-a-digest"
    elif mutation == "terminal_stage_digest":
        artifact["resumable_terminal"]["terminal_stage_file_sha256"] = SHA_D
    elif mutation == "resume_plan_population":
        artifact["resumable_closure"]["resume_plan"][
            "authorized_add_operations"
        ] += 1
    elif mutation == "partial_provider_usage":
        artifact["resumable_closure"]["transport_receipt"][
            "segment_receipt"
        ].pop("provider_output_tokens")
    elif mutation == "mem0_provider_usage":
        artifact["mem0_usage"]["provider_prompt_tokens"] += 1
    elif mutation == "write_usage_duplicate":
        duplicate = json.loads(
            json.dumps(artifact["write_usage_attestation"])
        )
        duplicate["observed"]["persisted_memory_count"] -= 1
        duplicate["observed_sha256"] = canonical_json_sha256(
            duplicate["observed"]
        )
        duplicate.pop("receipt_sha256")
        duplicate["receipt_sha256"] = canonical_json_sha256(duplicate)
        artifact["write_usage_attestation"] = duplicate
    else:
        artifact["resumable_closure"][
            "transport_closure_receipt_sha256"
        ] = SHA_D
    artifact.pop("content_sha256")
    artifact["content_sha256"] = canonical_json_sha256(artifact)
    payload = run_shard_module._render_json_bytes(artifact)
    artifact_path.write_bytes(payload)
    authorization = _scoring_authorization(
        shard, hashlib.sha256(payload).hexdigest()
    )

    with pytest.raises((Mem0ShardRunError, ValueError)):
        run_shard_module._verify_retrieval_artifact(
            artifact_path=artifact_path,
            trace_path=trace_path,
            shard=shard,
            authorization=authorization,
            prompt_packer=run_shard_module._default_prompt_packer,
        )


def test_scoring_treats_zero_provider_input_usage_as_unavailable(
    tmp_path: Path,
) -> None:
    shard, retrieval, _adapter = _run_stage_a(tmp_path)
    observed_messages: list[Sequence[Mapping[str, str]]] = []

    def provider(
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        del max_output_tokens
        observed_messages.append(messages)
        is_judge = model == MEM0_SOURCE_JUDGE_MODEL
        return ProviderCallResult(
            text="CORRECT same answer" if is_judge else "answer",
            # Some compatible gateways complete requests but report zero
            # provider input usage.  Zero means unknown, not an empty request.
            usage=UsageStats(input_tokens=0, output_tokens=2, calls=1),
        )

    _declare_stateless(provider)
    result = run_scoring_stage(
        shard=shard,
        authorization=_scoring_authorization(shard, retrieval.artifact_sha256),
        root_environment_lock_path=LOCK_PATH,
        retrieval_artifact_path=retrieval.artifact_path,
        retrieval_trace_path=retrieval.trace_path,
        report_path=tmp_path / "zero-usage-report.json",
        scoring_trace_path=tmp_path / "zero-usage-scoring.trace.json",
        responder=provider,
        judge=provider,
        process_guard=ShardProcessGuard("test-scoring-zero-provider-usage"),
    )

    assert len(observed_messages) == 20
    assert all(messages for messages in observed_messages)
    assert all(
        row["prompt_token_proxy"] > 0
        and row["prompt_token_proxy"] <= row["max_prompt_tokens"] == 8_000
        and row["provider_prompt_budget_compliant"] is None
        for row in result.report["question_results"]
    )
    receipt = result.report["scoring_receipt"]
    assert receipt["responder_input_usage_status"] == "unavailable"
    assert receipt["judge_input_usage_status"] == "unavailable"
    assert receipt["responder_usage"]["input_tokens"] == 0
    assert receipt["judge_usage"]["input_tokens"] == 0
    assert not list(tmp_path.glob("*.staging"))


def test_scoring_pair_publication_rolls_back_first_complete_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard, retrieval, _adapter = _run_stage_a(tmp_path)
    authorization = _scoring_authorization(shard, retrieval.artifact_sha256)
    calls: list[str] = []
    provider = _successful_scoring_provider(calls)
    real_link = run_shard_module.os.link
    link_calls = 0

    def fail_second_link(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("synthetic second-link failure")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(run_shard_module.os, "link", fail_second_link)
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "scoring.trace.json"
    with pytest.raises(Mem0ShardRunError) as raised:
        run_scoring_stage(
            shard=shard,
            authorization=authorization,
            root_environment_lock_path=LOCK_PATH,
            retrieval_artifact_path=retrieval.artifact_path,
            retrieval_trace_path=retrieval.trace_path,
            report_path=report_path,
            scoring_trace_path=trace_path,
            responder=provider,
            judge=provider,
            process_guard=ShardProcessGuard("test-scoring-publication-rollback"),
        )

    assert raised.value.trace_path is None
    assert link_calls == 2
    assert len(calls) == 20
    assert not report_path.exists()
    assert not trace_path.exists()
    assert not list(tmp_path.glob("*.staging"))


def test_scoring_reads_hashed_retrieval_trace_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard, retrieval, _adapter = _run_stage_a(tmp_path)
    trace_target = retrieval.trace_path.resolve()
    original_read_bytes = Path.read_bytes
    trace_reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal trace_reads
        if path.resolve() == trace_target:
            trace_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    calls: list[str] = []
    provider = _successful_scoring_provider(calls)
    run_scoring_stage(
        shard=shard,
        authorization=_scoring_authorization(shard, retrieval.artifact_sha256),
        root_environment_lock_path=LOCK_PATH,
        retrieval_artifact_path=retrieval.artifact_path,
        retrieval_trace_path=retrieval.trace_path,
        report_path=tmp_path / "report.json",
        scoring_trace_path=tmp_path / "scoring.trace.json",
        responder=provider,
        judge=provider,
        process_guard=ShardProcessGuard("test-scoring-single-trace-read"),
    )

    assert trace_reads == 1
    assert len(calls) == 20


def test_scoring_rejects_output_alias_of_stage_a_input_before_calls(
    tmp_path: Path,
) -> None:
    shard, retrieval, _adapter = _run_stage_a(tmp_path)
    artifact_before = retrieval.artifact_path.read_bytes()
    calls: list[str] = []
    provider = _successful_scoring_provider(calls)
    failure_trace = tmp_path / "scoring.failure.trace.json"

    with pytest.raises(Mem0ShardRunError) as raised:
        run_scoring_stage(
            shard=shard,
            authorization=_scoring_authorization(shard, retrieval.artifact_sha256),
            root_environment_lock_path=LOCK_PATH,
            retrieval_artifact_path=retrieval.artifact_path,
            retrieval_trace_path=retrieval.trace_path,
            report_path=retrieval.artifact_path,
            scoring_trace_path=failure_trace,
            responder=provider,
            judge=provider,
            process_guard=ShardProcessGuard("test-scoring-protected-output"),
        )

    assert calls == []
    assert retrieval.artifact_path.read_bytes() == artifact_before
    assert raised.value.trace_path == failure_trace
    assert json.loads(failure_trace.read_text(encoding="utf-8"))["status"] == "failed"


def test_scoring_rechecks_root_environment_lock_after_all_calls(
    tmp_path: Path,
) -> None:
    root_lock = tmp_path / "root.lock"
    root_lock.write_bytes(b"initial root lock")
    root_sha = hashlib.sha256(root_lock.read_bytes()).hexdigest()
    shard, retrieval, _adapter = _run_stage_a(
        tmp_path,
        source_environment_lock_sha256=root_sha,
    )
    responder_count = 0
    judge_count = 0

    def responder(
        _messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        del model, max_output_tokens
        nonlocal responder_count
        responder_count += 1
        return ProviderCallResult(
            text=f"answer {responder_count - 1}",
            usage=UsageStats(input_tokens=10, output_tokens=2, calls=1),
        )

    def judge(
        _messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        del model, max_output_tokens
        nonlocal judge_count
        judge_count += 1
        if judge_count == 10:
            root_lock.write_bytes(b"mutated root lock")
        return ProviderCallResult(
            text="CORRECT",
            usage=UsageStats(input_tokens=10, output_tokens=1, calls=1),
        )

    _declare_stateless(responder)
    _declare_stateless(judge)
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "scoring.failure.trace.json"
    with pytest.raises(Mem0ShardRunError):
        run_scoring_stage(
            shard=shard,
            authorization=_scoring_authorization(
                shard,
                retrieval.artifact_sha256,
                root_environment_lock_sha256=root_sha,
            ),
            root_environment_lock_path=root_lock,
            retrieval_artifact_path=retrieval.artifact_path,
            retrieval_trace_path=retrieval.trace_path,
            report_path=report_path,
            scoring_trace_path=trace_path,
            responder=responder,
            judge=judge,
            process_guard=ShardProcessGuard("test-scoring-lock-recheck"),
        )

    assert responder_count == judge_count == 10
    assert not report_path.exists()
    failed = json.loads(trace_path.read_text(encoding="utf-8"))
    assert failed["cleanup"]["environment_lock"]["unchanged"] is False


def test_scoring_tamper_fails_before_any_provider_call(tmp_path: Path) -> None:
    shard, retrieval, _adapter = _run_stage_a(tmp_path)
    artifact = json.loads(retrieval.artifact_path.read_text(encoding="utf-8"))
    artifact["retrieval_rows"][0]["context"] = "tampered"
    # This direct write is test-only corruption of an already isolated temp file.
    retrieval.artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    calls = 0

    def provider(
        _messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        del model, max_output_tokens
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run")

    _declare_stateless(provider)

    with pytest.raises(Mem0ShardRunError) as raised:
        run_scoring_stage(
            shard=shard,
            authorization=_scoring_authorization(
                shard, retrieval.artifact_sha256
            ),
            root_environment_lock_path=LOCK_PATH,
            retrieval_artifact_path=retrieval.artifact_path,
            retrieval_trace_path=retrieval.trace_path,
            report_path=tmp_path / "report.json",
            scoring_trace_path=tmp_path / "scoring.failure.trace.json",
            responder=provider,
            judge=provider,
            process_guard=ShardProcessGuard("test-scoring-tamper"),
        )

    assert calls == 0
    assert raised.value.trace_path is not None
    failed = json.loads(raised.value.trace_path.read_text(encoding="utf-8"))
    assert failed["cleanup"]["mem0_state_touched"] is False
    assert failed["cleanup"][
        "responder_logical_wrapper_calls"
    ]["attempted"] == 0


def test_cli_is_provider_free_and_fail_closed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--show-contract"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked_pending_frozen_runtime_policy"
    assert payload["provider_calls_permitted"] is False


def test_canonical_hash_rejects_nan() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_sha256({"unsafe": float("nan")})
