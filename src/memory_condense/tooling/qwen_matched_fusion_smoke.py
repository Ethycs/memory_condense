"""Real-checkpoint happy-path smoke for resident Qwen matched fusion.

The returned mapping is a local diagnostic summary, not a receipt artifact or
a performance attestation.  The public fusion builder remains the
only operation that executes Qwen features and the latent router.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseSnapshot,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceObligation,
    EvidencePacket,
    EvidenceSpan,
    ObligationResult,
    QueryProgram,
    make_atom_id,
    quote_sha256,
)
from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3PrefixEncoder,
    _qwen_prefix_gate_state,
    expected_prefix_checkpoint_sha256,
)
from memory_condense.search.fusion.latent_router import LatentEvidenceRouter
from memory_condense.search.fusion.models import FusionCaps
from memory_condense.search.fusion.qwen_feature_models import QwenAtomFeatureCaps
from memory_condense.search.fusion.qwen_features import QwenAtomFeatureProvider
from memory_condense.search.fusion.qwen_matched import (
    build_qwen_matched_fusion_pair,
)
from memory_condense.search.fusion.resident_models import (
    MatchedEvidenceFusionPair,
)


_DEFAULT_MODEL_DIR = Path(".cache/models/Qwen3-8B")
_MODEL_ID = "Qwen/Qwen3-8B"
_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
_RETAINED_LAYERS = 1
_OUTPUT_LAYER = 0
_HIDDEN_DIM = 4096
_NUM_LATENTS = 2
_NUM_HEADS = 4
_DEVICE = "cuda:0"
_DTYPE_NAME = "float16"
_TORCH_DTYPE_NAME = "torch.float16"
_MIN_FREE_CUDA_BYTES = 3 * 1024**3
_PINNED_PREFIX_CHECKPOINT_SHA256 = (
    "76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d"
)


def _expected_checkpoint_sha256() -> str:
    if DEFAULT_MODEL_ID != _MODEL_ID or DEFAULT_MODEL_REVISION != _MODEL_REVISION:
        raise RuntimeError("Qwen matched-fusion smoke model identity drifted")
    observed = expected_prefix_checkpoint_sha256(
        _RETAINED_LAYERS,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
    )
    if observed != _PINNED_PREFIX_CHECKPOINT_SHA256:
        raise RuntimeError("Qwen matched-fusion smoke checkpoint identity drifted")
    return observed


def _synthetic_atom(index: int, text: str) -> EvidenceAtom:
    span = EvidenceSpan(
        chunk_id=f"synthetic-chunk-{index}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=index,
        source_id="qwen-matched-fusion-smoke",
        turn_id=f"synthetic-turn-{index}",
        role="user",
    )
    return EvidenceAtom(
        atom_id=make_atom_id(span),
        span=span,
        text=text,
        label=f"synthetic-observation-{index}",
    )


def _synthetic_packet_and_plan() -> tuple[EvidencePacket, ClosurePlan]:
    """Create a route-agnostic N=2 packet with one exact synthetic bundle."""

    atoms = (
        _synthetic_atom(1, "First synthetic observation is retained exactly."),
        _synthetic_atom(2, "Second synthetic observation is retained exactly."),
    )
    bundle = EvidenceBundle(
        bundle_id="synthetic-two-observation-bundle",
        atom_ids=tuple(atom.atom_id for atom in atoms),
        obligation_ids=("relate-observations",),
        required=True,
        utility=1.0,
    )
    program = QueryProgram(
        query="How do the two exact observations relate?",
        intent="relate",
        subject_terms=("observations",),
        obligations=(
            EvidenceObligation(
                obligation_id="relate-observations",
                kind="answer_fact",
                required=True,
                weight=1.0,
            ),
        ),
    )
    plan = ClosurePlan(
        query_program=program,
        policy=ClosurePolicy(max_bundles=1, beam_width=2),
        snapshot=DiscourseSnapshot(
            max_turn_ordinal=2,
            chunk_count=2,
            graph_revision=0,
            schema_version=1,
            artifact_ids=(),
            source_content_sha256="1" * 64,
        ),
        seeds=(),
        atoms=atoms,
        bundles=(bundle,),
        obligation_results=(
            ObligationResult(
                obligation_id="relate-observations",
                status="satisfied",
                bundle_ids=(bundle.bundle_id,),
            ),
        ),
        visited_episode_ids=(),
        visited_unit_ids=(),
        visited_relation_ids=(),
        stopping_reason="complete",
        complete_claimed=True,
        scope_witnesses=(
            ClosureScopeWitness(
                kind="synthetic_scope",
                subject_id="qwen-matched-fusion-smoke",
                requested_limit=2,
                returned_count=2,
                exhaustive=True,
            ),
        ),
    )
    context = "Synthetic packet containing two exact observations."
    receipt = ClosureReceipt(
        plan_sha256=plan.plan_sha256,
        context_sha256=quote_sha256(context),
        selected_bundle_ids=(bundle.bundle_id,),
        selected_atom_ids=tuple(atom.atom_id for atom in atoms),
        dropped_bundle_reasons={},
        context_token_proxy=8,
        max_context_token_proxy=64,
        tokenizer_identity="synthetic-smoke-tokenizer-proxy",
        stopping_reason="complete",
        complete_claimed=True,
    )
    return EvidencePacket(context, atoms, (bundle,), receipt), plan


def _fusion_caps() -> FusionCaps:
    return FusionCaps(
        max_atoms=2,
        max_latents=2,
        max_hidden_dim=4096,
        max_route_cells=4,
        max_topology_links=1,
        max_hyperedges=1,
        max_groups=2,
        max_group_atoms=2,
        max_latent_memberships_per_atom=2,
    )


def _feature_caps() -> QwenAtomFeatureCaps:
    return QwenAtomFeatureCaps(
        max_row_tokens=128,
        max_query_tail_tokens=64,
        max_rows_per_forward=2,
        max_workspace_tokens=256,
        max_evidence_characters=256,
        max_query_characters=128,
        batch_invariance_atol=1e-3,
        batch_invariance_rtol=1e-3,
    )


def _contains_tensor(value: object, tensor_type: type) -> bool:
    if isinstance(value, tensor_type):
        return True
    if is_dataclass(value):
        return any(
            _contains_tensor(getattr(value, item.name), tensor_type)
            for item in fields(value)
        )
    if isinstance(value, Mapping):
        return any(
            _contains_tensor(key, tensor_type)
            or _contains_tensor(item, tensor_type)
            for key, item in value.items()
        )
    if type(value) in {tuple, list, set, frozenset}:
        return any(_contains_tensor(item, tensor_type) for item in value)
    return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _assert_success(
    pair: MatchedEvidenceFusionPair,
    *,
    packet: EvidencePacket,
    plan: ClosurePlan,
    caps: FusionCaps,
    feature_caps: QwenAtomFeatureCaps,
    provider: QwenAtomFeatureProvider,
    provider_before: object,
    router: LatentEvidenceRouter,
    router_before: object,
    encoder: Qwen3PrefixEncoder,
    torch: Any,
) -> None:
    _require(type(pair) is MatchedEvidenceFusionPair, "fusion returned a foreign pair")
    control = pair.topology_only
    treatment = pair.latent_router
    operation = pair.operation
    feature = operation.feature_suboperation
    receipt = pair.receipt

    _require(control.mode == "topology_only", "control mode changed")
    _require(treatment.mode == "latent_router", "treatment mode changed")
    _require(control.caps == caps == treatment.caps == feature.caps, "caps join failed")
    _require(feature.feature_caps == feature_caps, "feature caps join failed")
    _require(control.atoms == treatment.atoms == feature.atoms, "atom join failed")
    _require(
        control.hyperedges == treatment.hyperedges == feature.hyperedges,
        "hyperedge join failed",
    )
    _require(len(feature.atoms) == 2, "smoke must bind exactly two atoms")
    _require(len(feature.hyperedges) == 1, "smoke must bind exactly one hyperedge")
    _require(
        tuple(item.atom_id for item in feature.atoms)
        == tuple(item.atom_id for item in packet.atoms),
        "feature atom order changed",
    )
    _require(
        feature.hyperedges[0].bundle_id == packet.bundles[0].bundle_id
        and feature.hyperedges[0].atom_ids
        == tuple(item.atom_id for item in packet.atoms),
        "synthetic bundle provenance changed",
    )

    _require(feature.packet_receipt_sha256 == packet.receipt.receipt_sha256, "packet join failed")
    _require(feature.closure_plan_sha256 == plan.plan_sha256, "plan join failed")
    _require(
        feature.query_program_sha256 == plan.query_program.program_sha256,
        "query-program join failed",
    )
    _require(feature.query_sha256 == quote_sha256(plan.query_program.query), "query join failed")
    _require(feature.closure_policy_sha256 == plan.policy.policy_sha256, "policy join failed")
    _require(feature.snapshot_sha256 == plan.snapshot.snapshot_sha256, "snapshot join failed")
    _require(feature.provider == provider_before == provider.receipt, "provider join failed")
    _require(feature.feature_shape == (2, _HIDDEN_DIM), "feature shape changed")
    _require(feature.feature_device == _DEVICE, "feature device changed")
    _require(feature.feature_execution_dtype == _TORCH_DTYPE_NAME, "feature dtype changed")
    _require(len(feature.rows) == 2, "row coverage changed")
    _require(tuple(row.row_index for row in feature.rows) == (0, 1), "row indices changed")
    _require(
        tuple(row.atom_id for row in feature.rows)
        == tuple(atom.atom_id for atom in packet.atoms),
        "row atom order changed",
    )
    for row, atom, atom_ref in zip(feature.rows, packet.atoms, feature.atoms, strict=True):
        _require(
            row.atom_identity_sha256
            == atom_ref.atom_identity_sha256
            == identity_sha256(atom.identity_payload()),
            "row atom identity changed",
        )
        _require(
            row.span_identity_sha256
            == atom_ref.span_identity_sha256
            == identity_sha256(atom.span.identity_payload()),
            "row span identity changed",
        )
        _require(
            row.quote_sha256 == atom_ref.quote_sha256 == atom.span.quote_sha256,
            "row quote identity changed",
        )
        _require(
            row.evidence_character_count == len(atom.text)
            and row.query_character_count == len(plan.query_program.query),
            "row character accounting changed",
        )
        _require(
            row.total_row_tokens
            == row.prefix_tokens
            + row.evidence_tokens_admitted
            + row.query_tail_tokens,
            "row token accounting changed",
        )
        _require(
            0 < row.evidence_tokens_admitted == row.evidence_tokens_observed,
            "fixed smoke evidence row was truncated",
        )
        _require(row.evidence_truncated is False, "fixed smoke row claimed truncation")
        _require(
            row.total_row_tokens <= feature_caps.max_row_tokens
            and row.query_tail_tokens <= feature_caps.max_query_tail_tokens
            and row.evidence_character_count
            <= feature_caps.max_evidence_characters
            and row.query_character_count <= feature_caps.max_query_characters,
            "row exceeded fixed feature caps",
        )
    _require(
        all(row.readout_end_index == row.total_row_tokens - 1 for row in feature.rows),
        "row readout marker changed",
    )
    _require(len(feature.batches) == 1, "smoke requires one bounded Qwen batch")
    batch = feature.batches[0]
    _require(
        (batch.batch_index, batch.start_row, batch.row_count) == (0, 0, 2),
        "Qwen batch coverage changed",
    )
    _require(
        batch.padded_workspace_tokens <= feature_caps.max_workspace_tokens,
        "Qwen workspace exceeded caps",
    )
    _require(
        batch.padded_width == max(row.total_row_tokens for row in feature.rows)
        and batch.padded_workspace_tokens == batch.row_count * batch.padded_width,
        "Qwen padded workspace accounting changed",
    )
    _require(
        feature.qwen_forward_count == feature.primary_qwen_forward_count == 1,
        "smoke requires exactly one primary Qwen forward",
    )
    _require(feature.batch_invariance_forward_count == 0, "invariance diagnostic ran")
    _require(feature.runtime_batch_invariance_attested is False, "invariance was overstated")
    _require(feature.qwen_executed is True, "Qwen execution was not attested")
    _require(feature.router_executed is False, "feature sub-operation claimed routing")
    _require(feature.matched_pair_produced is False, "feature sub-operation claimed a pair")
    _require(feature.performance_attested is False, "feature performance was overstated")
    _require(feature.feature_tensor_sha256 is None, "full feature tensor was hashed")
    _require(feature.steered_tensor_produced is False, "feature sub-operation claimed steering")
    _require(feature.steered_tensor_sha256 is None, "full steered tensor was hashed")
    _require(feature.feature_tensor_content_attested is False, "feature content was overstated")
    _require(feature.operation_inputs_attested is True, "feature inputs were not attested")
    _require(feature.retrieval_route_attested is False, "retrieval route was overstated")
    _require(feature.retained_request_tensor_bytes == 0, "feature receipt retained tensors")

    runtime = operation.router_runtime
    _require(runtime == router_before == router.resident_runtime_receipt, "router changed")
    _require(runtime.device == _DEVICE, "router device changed")
    _require(runtime.execution_dtype == _TORCH_DTYPE_NAME, "router dtype changed")
    _require(runtime.sealed_for_inference is True, "router is not sealed")
    _require(runtime.state.training_status == "untrained", "router status changed")
    _require(runtime.state.parameter_dtypes == (_TORCH_DTYPE_NAME,), "router dtype set changed")
    _require(runtime.architecture.hidden_dim == _HIDDEN_DIM, "router width changed")
    _require(runtime.architecture.num_latents == _NUM_LATENTS, "latent count changed")
    _require(runtime.architecture.num_heads == _NUM_HEADS, "router head count changed")
    _require(
        (runtime.max_atoms, runtime.max_hidden_dim, runtime.max_route_cells)
        == (2, _HIDDEN_DIM, 4),
        "router bounds changed",
    )
    _require(treatment.router_runtime == runtime, "treatment router join failed")
    _require(control.router_runtime is None, "control bound a router")
    _require(operation.extraction_shape == treatment.extraction_shape == (2, 2), "extraction shape changed")
    _require(operation.reinjection_shape == treatment.reinjection_shape == (2, 2), "reinjection shape changed")
    _require(
        operation.extraction_matrix_sha256 == treatment.extraction_matrix_sha256,
        "extraction digest join failed",
    )
    _require(
        operation.reinjection_matrix_sha256 == treatment.reinjection_matrix_sha256,
        "reinjection digest join failed",
    )
    _require(operation.route_matrix_canonical_dtype == "float32-le", "route canonical dtype changed")
    _require(
        operation.route_weight_normalization_policy == "source_dtype_softmax_sum_v1",
        "route normalization policy changed",
    )

    feature_sha = feature.operation_sha256
    _require(control.feature_suboperation_sha256 == feature_sha, "control feature join failed")
    _require(treatment.feature_suboperation_sha256 == feature_sha, "treatment feature join failed")
    _require(operation.topology_plan_sha256 == control.plan_sha256, "control plan join failed")
    _require(operation.latent_plan_sha256 == treatment.plan_sha256, "treatment plan join failed")
    _require(
        operation.matched_input_sha256
        == control.matched_input_sha256
        == treatment.matched_input_sha256
        == receipt.matched_input_sha256,
        "matched-input join failed",
    )
    _require(receipt.operation_sha256 == operation.operation_sha256, "operation receipt join failed")
    _require(receipt.feature_suboperation_sha256 == feature_sha, "feature receipt join failed")
    _require(receipt.topology_plan_sha256 == control.plan_sha256, "control receipt join failed")
    _require(receipt.latent_plan_sha256 == treatment.plan_sha256, "treatment receipt join failed")
    _require(receipt.exact_atom_set_shared is True, "shared atom claim missing")
    _require(receipt.exact_hyperedges_shared is True, "shared hyperedge claim missing")
    _require(receipt.feature_operation_shared is True, "shared feature claim missing")

    _require(operation.router_forward_count == 1, "smoke requires one router forward")
    _require(operation.topology_reencode_count == 0, "control re-encoded features")
    _require(operation.qwen_executed is True, "outer operation omitted Qwen")
    _require(operation.router_executed is True, "outer operation omitted router")
    _require(operation.matched_pair_produced is True, "outer operation omitted pair")
    _require(operation.bounded_route_matrix_content_attested is True, "route matrices were not bound")
    _require(operation.route_matrix_values_retained is False, "route values were retained")
    _require(operation.feature_tensor_content_attested is False, "feature content was overstated")
    _require(operation.steered_tensor_produced is True, "steered tensor was not produced")
    _require(operation.steered_tensor_content_attested is False, "steered content was overstated")
    _require(operation.single_feature_workspace_attested is True, "workspace sharing was not attested")
    _require(operation.operation_inputs_attested is True, "outer inputs were not attested")
    _require(operation.retrieval_route_attested is False, "retrieval route was overstated")
    _require(operation.performance_attested is False, "performance was overstated")
    _require(operation.feature_tensor_sha256 is None, "full feature tensor was hashed")
    _require(operation.steered_tensor_sha256 is None, "full steered tensor was hashed")
    _require(operation.retained_request_tensor_bytes == 0, "operation retained tensors")
    _require(control.plan_retained_request_tensor_bytes == 0, "control retained tensors")
    _require(treatment.plan_retained_request_tensor_bytes == 0, "treatment retained tensors")

    provider_receipt = provider.receipt
    _require(provider_receipt.model_id == _MODEL_ID, "provider model changed")
    _require(provider_receipt.model_revision == _MODEL_REVISION, "provider revision changed")
    _require(
        provider_receipt.checkpoint_sha256 == _expected_checkpoint_sha256(),
        "provider checkpoint changed",
    )
    _require(provider_receipt.retained_layers == _RETAINED_LAYERS, "provider layers changed")
    _require(provider_receipt.output_layer == _OUTPUT_LAYER, "provider output changed")
    _require(provider_receipt.hidden_dim == _HIDDEN_DIM, "provider width changed")
    _require(provider_receipt.device == _DEVICE, "provider device changed")
    _require(provider_receipt.execution_dtype == _TORCH_DTYPE_NAME, "provider dtype changed")
    _require(
        provider_receipt.provider_id == "qwen3_prefix.query_readout_last.v1"
        and provider_receipt.pooling == "last_token"
        and provider_receipt.truncation_rule == "evidence_prefix_only"
        and provider_receipt.checkpoint_status == "checkpoint_files_verified",
        "provider execution declaration changed",
    )
    _require(provider_receipt.model_behavior_verified is False, "model behavior was overstated")
    _require(provider_receipt.general_concurrency_safe is False, "concurrency was overstated")
    _require(
        provider_receipt.exclusive_synchronous_ownership_required is True
        and provider_receipt.exclusive_synchronous_ownership_verified is False,
        "provider ownership boundary changed",
    )
    _require(provider_receipt.loaded_parameter_content_attested is False, "parameter content was overstated")
    _require(provider_receipt.loaded_tokenizer_content_attested is False, "tokenizer content was overstated")
    _require(
        provider_receipt.loaded_module_runtime_constants_attested is False,
        "module runtime constants were overstated",
    )
    _require(provider_receipt.tokenizer_behavior_verified is False, "tokenizer behavior was overstated")
    _require(provider_receipt.legacy_hook_paths_serialized is False, "legacy hooks were overstated")
    _require(
        provider_receipt.supported_structural_mutation_checks_attested is True
        and provider_receipt.supported_mutation_scope
        == "structure_parameter_metadata_bounded_scalar_fields"
        and provider_receipt.execution_gate_scope == "fusion_provider_only",
        "provider mutation-check boundary changed",
    )

    provider._assert_provider_state()
    provider._assert_implementation()
    provider._assert_runtime()
    router._assert_inference_seal()
    gate = _qwen_prefix_gate_state(encoder)
    _require(gate.active_token is None, "provider gate retained an active token")
    gate_available = gate.lock.acquire(blocking=False)
    _require(gate_available, "provider gate remained locked")
    gate.lock.release()
    _require(not _contains_tensor(pair, torch.Tensor), "public pair retained a tensor")
    encoded = canonical_json(
        {
            "control": control.identity_payload(),
            "treatment": treatment.identity_payload(),
            "operation": operation.identity_payload(),
            "receipt": receipt.identity_payload(),
        }
    )
    request_text = (packet.context, plan.query_program.query, *(atom.text for atom in packet.atoms))
    _require(all(text not in encoded for text in request_text), "public pair retained request text")


def _preflight_cuda(torch: Any) -> tuple[Any, int, int, int]:
    device = torch.device(_DEVICE)
    if device.type != "cuda" or device.index != 0 or str(device) != _DEVICE:
        raise RuntimeError("matched-fusion smoke requires canonical indexed cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("matched-fusion smoke requires CUDA device 0")
    if str(getattr(torch, "float16", None)) != _TORCH_DTYPE_NAME:
        raise RuntimeError("matched-fusion smoke requires canonical torch.float16")
    free_bytes, total_bytes = (
        int(value) for value in torch.cuda.mem_get_info(device)
    )
    if free_bytes < _MIN_FREE_CUDA_BYTES:
        raise MemoryError(
            "matched-fusion smoke requires at least 3 GiB of free CUDA memory"
        )
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    return device, free_bytes, total_bytes, baseline_allocated


def run_qwen_matched_fusion_smoke(
    model_dir: str | Path = _DEFAULT_MODEL_DIR,
) -> dict[str, object]:
    """Run one real local smoke and return a non-artifact diagnostic mapping."""

    import torch

    expected_checkpoint = _expected_checkpoint_sha256()
    device, free_bytes, total_bytes, baseline_allocated = _preflight_cuda(torch)
    packet, plan = _synthetic_packet_and_plan()
    caps = _fusion_caps()
    feature_caps = _feature_caps()
    encoder: Qwen3PrefixEncoder | None = None
    provider: QwenAtomFeatureProvider | None = None
    router: LatentEvidenceRouter | None = None
    pair: MatchedEvidenceFusionPair | None = None
    report: dict[str, object] = {}
    try:
        load_started = time.perf_counter()
        encoder = Qwen3PrefixEncoder(
            Path(model_dir),
            layers=_RETAINED_LAYERS,
            device=_DEVICE,
            dtype=_DTYPE_NAME,
            model_id=_MODEL_ID,
            model_revision=_MODEL_REVISION,
            expected_checkpoint_sha256=expected_checkpoint,
        )
        torch.cuda.synchronize(device)
        checkpoint_load_seconds = time.perf_counter() - load_started
        _require(int(encoder.config.hidden_size) == _HIDDEN_DIM, "Qwen width changed")

        provider = QwenAtomFeatureProvider(encoder, output_layer=_OUTPUT_LAYER)
        provider_before = provider.receipt
        router_started = time.perf_counter()
        router = LatentEvidenceRouter(
            _HIDDEN_DIM,
            num_latents=_NUM_LATENTS,
            num_heads=_NUM_HEADS,
            training_status="untrained",
            max_atoms=2,
            max_hidden_dim=_HIDDEN_DIM,
            max_route_cells=4,
        ).seal_for_inference(device=device, dtype=torch.float16)
        router_before = router.resident_runtime_receipt
        torch.cuda.synchronize(device)
        router_setup_seconds = time.perf_counter() - router_started

        resident_allocated_before_builder = int(
            torch.cuda.memory_allocated(device)
        )
        torch.cuda.reset_peak_memory_stats(device)
        fusion_started = time.perf_counter()
        pair = build_qwen_matched_fusion_pair(
            packet,
            plan,
            provider=provider,
            router=router,
            caps=caps,
            feature_caps=feature_caps,
        )
        torch.cuda.synchronize(device)
        fusion_seconds = time.perf_counter() - fusion_started
        allocated_after_builder = int(torch.cuda.memory_allocated(device))
        operation_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        _require(
            allocated_after_builder <= resident_allocated_before_builder,
            "matched-fusion builder retained CUDA tensor allocation",
        )
        _assert_success(
            pair,
            packet=packet,
            plan=plan,
            caps=caps,
            feature_caps=feature_caps,
            provider=provider,
            provider_before=provider_before,
            router=router,
            router_before=router_before,
            encoder=encoder,
            torch=torch,
        )
        report.update(
            {
                "format": "qwen_matched_fusion_local_diagnostic_v1",
                "diagnostic_non_artifact": True,
                "performance_attested": False,
                "model_id": _MODEL_ID,
                "model_revision": _MODEL_REVISION,
                "checkpoint_sha256": expected_checkpoint,
                "retained_layers": _RETAINED_LAYERS,
                "output_layer": _OUTPUT_LAYER,
                "hidden_dim": _HIDDEN_DIM,
                "device": _DEVICE,
                "execution_dtype": _TORCH_DTYPE_NAME,
                "atom_count": 2,
                "latent_count": 2,
                "qwen_forward_count": pair.operation.feature_suboperation.qwen_forward_count,
                "router_forward_count": pair.operation.router_forward_count,
                "pair_sha256": pair.receipt.pair_sha256,
                "operation_sha256": pair.operation.operation_sha256,
                "feature_suboperation_sha256": pair.operation.feature_suboperation.operation_sha256,
                "observed_checkpoint_load_seconds": checkpoint_load_seconds,
                "observed_router_setup_seconds": router_setup_seconds,
                "observed_fusion_seconds": fusion_seconds,
                "observed_free_cuda_bytes_before_load": free_bytes,
                "observed_total_cuda_bytes": total_bytes,
                "observed_cuda_allocated_bytes_before_load": baseline_allocated,
                "observed_cuda_allocated_bytes_before_builder": (
                    resident_allocated_before_builder
                ),
                "observed_cuda_allocated_bytes_after_builder": (
                    allocated_after_builder
                ),
                "observed_operation_peak_cuda_allocated_bytes": (
                    operation_peak_allocated
                ),
            }
        )
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []
        pair = None
        router = None
        if provider is not None:
            try:
                provider.close()
            except BaseException as exc:  # pragma: no cover - real runtime failure
                cleanup_errors.append(exc)
        provider = None
        encoder = None
        try:
            gc.collect()
        except BaseException as exc:  # pragma: no cover - real runtime failure
            cleanup_errors.append(exc)
        try:
            torch.cuda.synchronize(device)
        except BaseException as exc:  # pragma: no cover - real runtime failure
            cleanup_errors.append(exc)
        try:
            torch.cuda.empty_cache()
        except BaseException as exc:  # pragma: no cover - real runtime failure
            cleanup_errors.append(exc)
        try:
            torch.cuda.synchronize(device)
            post_cleanup_allocated = int(torch.cuda.memory_allocated(device))
            report["observed_cuda_allocated_bytes_after_cleanup"] = (
                post_cleanup_allocated
            )
            if primary_error is None and post_cleanup_allocated > baseline_allocated:
                cleanup_errors.append(
                    RuntimeError(
                        "matched-fusion smoke did not restore pre-load CUDA allocation"
                    )
                )
        except BaseException as exc:  # pragma: no cover - real runtime failure
            cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is None:
                raise cleanup_errors[0]
            for cleanup_error in cleanup_errors:
                primary_error.add_note(f"cleanup failure: {cleanup_error!r}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_qwen_matched_fusion_smoke(args.model_dir)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
