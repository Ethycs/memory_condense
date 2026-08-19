from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from memory_condense.domain.discourse import ClosurePolicy
from memory_condense.eval.diffuse_compilation import DiffuseCompilationPolicy
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalArm,
    matched_diffuse_boundary_arms,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalRuntimeConfig,
    build_diffuse_longmemeval_execution_binding,
    run_diffuse_treatment_sample,
)
from memory_condense.eval.diffuse_longmemeval_runtime_matched import (
    validate_matched_diffuse_runtime_results,
)
from memory_condense.eval.schemas import ChunkerConfig, EvalConfig, RetrievalConfig
from memory_condense.search.episodes import EpisodeRetrievalPolicy
from tools.v4_population_firebreak.treatment import load_analysis_treatment_input


REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "eval_results/v4-analysis-input/longmemeval-analysis-treatment-v2.json"
OUTPUT_ROOT = REPO / "eval_results/v4-canary-ecbb9dd-s169"
SAMPLE_INDEX = 169


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query_summary(row):
    retrieval = row.retrieval
    plan = retrieval.plan
    packet = retrieval.packet
    representative = retrieval.representative_expansion
    scope = row.legacy_inputs.candidates.source_candidate_scope
    return {
        "question_id_sha256": _sha256_text(row.probe.question_id),
        "analysis_query_receipt_sha256": row.receipt.receipt_sha256,
        "legacy_input_receipt": row.legacy_inputs.receipt.identity_payload(),
        "diffuse_query_receipt": retrieval.receipt.identity_payload(),
        "source_scope": None
        if scope is None
        else {
            "receipt_sha256": scope.receipt_sha256,
            "universe_sources": len(scope.universe_source_ids),
            "selected_sources": len(scope.candidates),
            "truncated_sources": len(scope.truncated_source_ids),
            "universe_enumerated": scope.universe_enumerated,
        },
        "representative": None
        if representative is None
        else {
            "receipt_sha256": representative.receipt_sha256,
            "runtime_binding_certified": representative.runtime_binding_certified,
            "candidate_scope_exhaustive": representative.candidate_scope_exhaustive,
            "returned_plan_transformer_state_bytes": (
                representative.returned_plan_transformer_state_bytes
            ),
            "seeds": len(representative.seeds),
            "truncated_sources": len(representative.truncated_source_ids),
            "truncated_episodes": len(representative.truncated_episode_ids),
            "unavailable_episodes": len(representative.unavailable_episode_ids),
            "passes": representative.passes,
            "max_workspace_candidates": representative.max_workspace_candidates,
            "max_workspace_tokens": representative.max_workspace_tokens,
            "total_candidate_inspections": representative.total_candidate_inspections,
        },
        "closure": {
            "plan_sha256": plan.plan_sha256,
            "stopping_reason": plan.stopping_reason,
            "complete_claimed": plan.complete_claimed,
            "scope_exhaustive": all(item.exhaustive for item in plan.scope_witnesses),
            "scope_witnesses": [item.identity_payload() for item in plan.scope_witnesses],
            "atoms": len(plan.atoms),
            "bundles": len(plan.bundles),
            "visited_episodes": len(plan.visited_episode_ids),
            "visited_units": len(plan.visited_unit_ids),
            "visited_relations": len(plan.visited_relation_ids),
        },
        "packet": {
            "receipt_sha256": packet.receipt.receipt_sha256,
            "context_sha256": packet.receipt.context_sha256,
            "context_token_proxy": packet.receipt.context_token_proxy,
            "prompt_token_proxy": packet.receipt.prompt_token_proxy,
            "prompt_workspace_token_proxy": packet.receipt.prompt_workspace_token_proxy,
            "max_prompt_workspace_token_proxy": packet.receipt.max_prompt_token_proxy,
            "selected_atoms": len(packet.atoms),
            "selected_bundles": len(packet.bundles),
            "retained_request_token_state_bytes": (
                packet.receipt.retained_request_token_state_bytes
            ),
        },
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to reuse {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    print("loading verified gold-blind analysis artifact", flush=True)
    treatment = load_analysis_treatment_input(
        INPUT,
        expected_file_sha256=(
            "b4d1d34538fdabbd6127c339bff8167293d290eb732afc18a5d8963d12b15001"
        ),
        expected_sanitized_projection_sha256=(
            "58a1982122d259e046ac5268de8fc3c2857a63d24c859e3bc13e4e6b9aa52ad8"
        ),
        expected_dataset_sha256=(
            "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
        ),
        expected_split_manifest_sha256=(
            "8d5c1885903b199a4ab0859ccabc5ce41d9a105d0c755d3daf33cbfd959995f4"
        ),
        expected_ordered_question_ids_sha256=(
            "cf5e8648b71634e4e22be872881766e37e0dc24a2931d0c63365e075b2742046"
        ),
        expected_sample_count=300,
    )
    sample = treatment.samples[SAMPLE_INDEX]
    print(
        f"sample loaded: turns={len(sample.turns)} questions={len(sample.questions)}",
        flush=True,
    )

    retrieval = RetrievalConfig(
        mode="hybrid_graph",
        k=10,
        ef_search=50,
        candidates=100,
        alpha=0.65,
        neighbor_radius=5,
        neighbor_slots=24,
        neighbor_direction="next",
        source_slots=48,
        source_candidate_pool=750,
        source_activation_k=65,
        role_aware_retrieval=True,
        source_tfisf_activation=True,
        source_tfisf_slots=8,
        source_hsc_activation=True,
        source_hsc_slots=8,
        source_hsc_hops=2,
        source_hsc_chunk_slots=4,
        source_local_search=True,
        source_partition_routing=True,
        source_partition_slots=4,
        qwen_rerank=False,
        qwen_feedback=False,
        qwen_rerank_prefix_layers=2,
        qwen_rerank_attention_layer=1,
        qwen_rerank_max_workspace_tokens=2048,
    )
    config = EvalConfig(
        chunker=ChunkerConfig(min_tokens=120, max_tokens=250),
        retrieval=retrieval,
        embedding_device="cuda",
        max_prompt_tokens=8000,
    )
    runtime = DiffuseLongMemEvalRuntimeConfig(
        qwen_model_dir=REPO / ".cache/models/Qwen3-8B",
        residency_mode="resident_bge_qwen",
    )
    binding = build_diffuse_longmemeval_execution_binding(
        config=config,
        runtime=runtime,
    )
    if not binding.runtime_binding_certified:
        raise RuntimeError("owned runtime binding is not certified")

    reference = DiffuseLongMemEvalArm(
        arm_id="fixed_interval",
        compilation=DiffuseCompilationPolicy(boundary_mode="fixed_interval"),
        episode=EpisodeRetrievalPolicy(
            max_anchor_episodes=96,
            previous_episodes=1,
            next_episodes=1,
            max_episode_seeds=256,
            max_direct_fallbacks=96,
        ),
        closure=ClosurePolicy(
            max_hops=3,
            max_units=1024,
            max_relations=2048,
            max_degree=32,
            max_episode_neighbors=2,
            max_frontier=1024,
            max_bundles=256,
            beam_width=128,
            min_relation_confidence=0.5,
        ),
        max_context_tokens=7000,
        responder_output_token_reserve=256,
        require_owned_representative_runtime=True,
    )
    arms = matched_diffuse_boundary_arms(reference)
    results = []
    arm_summaries = []
    started = time.perf_counter()
    for arm in arms:
        arm_started = time.perf_counter()
        print(f"starting arm={arm.arm_id}", flush=True)
        result = run_diffuse_treatment_sample(
            sample,
            binding=binding,
            arm=arm,
            data_dir=OUTPUT_ROOT / arm.arm_id,
        )
        elapsed = time.perf_counter() - arm_started
        results.append(result)
        phase = result.phase
        compilation = phase.compilation
        arm_summaries.append(
            {
                "arm_id": arm.arm_id,
                "elapsed_s": elapsed,
                "runtime_result": result.identity_payload(),
                "phase_receipt_sha256": phase.receipt_sha256,
                "compilation_receipt_sha256": compilation.receipt_sha256,
                "compilation_policy_sha256": compilation.compilation_policy_sha256,
                "artifact_policy_sha256": compilation.policy_sha256,
                "artifact_id": compilation.artifact.artifact_id,
                "snapshot_sha256": compilation.final_snapshot.snapshot_sha256,
                "sources": len(compilation.source_receipts),
                "content_chunks": sum(
                    item.content_chunks for item in compilation.source_receipts
                ),
                "metadata_chunks": sum(
                    item.metadata_chunks for item in compilation.source_receipts
                ),
                "episodes": sum(
                    len(item.episode_ids) for item in compilation.source_receipts
                ),
                "units": sum(
                    len(item.unit_ids) for item in compilation.source_receipts
                ),
                "relations": sum(
                    len(item.relation_ids) for item in compilation.source_receipts
                ),
                "persisted_request_token_state_bytes": (
                    compilation.persisted_request_token_state_bytes
                ),
                "questions": [_query_summary(row) for row in phase.questions],
            }
        )
        print(f"finished arm={arm.arm_id} elapsed_s={elapsed:.2f}", flush=True)
    suite = validate_matched_diffuse_runtime_results(results)
    receipt = {
            "format": "memory-condense-v4-gold-blind-real-model-canary-v1",
            "source_commit": "ecbb9dd8813528b30b2096d98d47e383fcbbb282",
            "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "treatment": {
                "file_sha256": treatment.file_sha256,
                "sanitized_projection_sha256": treatment.sanitized_projection_sha256,
                "dataset_sha256": treatment.dataset_sha256,
                "split_manifest_sha256": treatment.split_manifest_sha256,
                "ordered_question_ids_sha256": treatment.ordered_question_ids_sha256,
                "sample_count": len(treatment.samples),
                "sample_index": SAMPLE_INDEX,
                "sample_id_sha256": _sha256_text(sample.sample_id),
                "turn_count": len(sample.turns),
                "question_count": len(sample.questions),
            },
            "runtime_suite": suite.identity_payload(),
            "matched_suite": suite.matched_suite.identity_payload(),
            "arms": arm_summaries,
            "total_elapsed_s": time.perf_counter() - started,
            "gold_fields_available_to_process": False,
            "provider_calls": 0,
    }
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    receipt_path = OUTPUT_ROOT / "canary-receipt.json"
    with receipt_path.open("xb") as handle:
        handle.write(payload)
    with (OUTPUT_ROOT / "canary-receipt.json.sha256").open("x", encoding="ascii") as handle:
        handle.write(hashlib.sha256(payload).hexdigest() + "\n")
    print(
        "CANARY_PASS "
        f"suite={suite.receipt_sha256} "
        f"receipt={hashlib.sha256(payload).hexdigest()} "
        f"elapsed_s={receipt['total_elapsed_s']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
