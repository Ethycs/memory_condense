from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import ClosurePlan, EvidencePacket
from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    expected_prefix_checkpoint_sha256,
)
from memory_condense.tooling import qwen_matched_fusion_smoke as smoke
from memory_condense.search.packing.evidence_packet import render_evidence_context


SOURCE_PATH = Path(smoke.__file__)
REPO_ROOT = SOURCE_PATH.parents[3]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(call: ast.Call) -> str | None:
    value = call.func
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def test_parser_has_only_the_local_checkpoint_directory_option() -> None:
    parser = smoke._parser()
    args = parser.parse_args([])
    assert args.model_dir == Path(".cache/models/Qwen3-8B")
    assert {action.dest for action in parser._actions} == {"help", "model_dir"}
    assert parser.parse_args(["--model-dir", "local-prefix"]).model_dir == Path(
        "local-prefix"
    )


def test_synthetic_inputs_are_exact_route_agnostic_n2_one_bundle() -> None:
    packet, plan = smoke._synthetic_packet_and_plan()
    assert type(packet) is EvidencePacket
    assert type(plan) is ClosurePlan
    assert len(packet.atoms) == len(plan.atoms) == 2
    assert len(packet.bundles) == len(plan.bundles) == 1
    assert packet.atoms == plan.atoms
    assert packet.bundles == plan.bundles
    assert packet.receipt.plan_sha256 == plan.plan_sha256
    assert packet.receipt.selected_atom_ids == tuple(
        atom.atom_id for atom in packet.atoms
    )
    assert packet.receipt.selected_bundle_ids == (packet.bundles[0].bundle_id,)
    assert packet.bundles[0].atom_ids == packet.receipt.selected_atom_ids
    assert packet.bundles[0].obligation_ids == ("relate-observations",)
    assert len(plan.query_program.obligations) == 1
    assert len(plan.obligation_results) == 1
    assert plan.seeds == ()
    assert plan.visited_episode_ids == ()
    assert plan.visited_unit_ids == ()
    assert plan.visited_relation_ids == ()
    assert all(
        atom.span.source_id == "qwen-matched-fusion-smoke"
        for atom in packet.atoms
    )
    assert packet.context == render_evidence_context(plan.atoms, plan.bundles)
    assert packet.receipt.context_sha256 == quote_sha256(packet.context)
    assert packet.receipt.context_token_proxy == count_tokens(
        packet.context,
        encoding=smoke._ENCODING,
    )
    assert packet.receipt.max_context_token_proxy == smoke._MAX_CONTEXT_TOKENS == 256
    tokenizer_body = tokenizer_proxy_identity(smoke._ENCODING)
    assert packet.receipt.tokenizer_identity == (
        f"{tokenizer_body['encoding']}:{identity_sha256(tokenizer_body)}"
    )
    base_messages = smoke._synthetic_base_messages(plan)
    assert base_messages == (
        {"role": "system", "content": smoke._PROMPT_SYSTEM_MESSAGE},
        {"role": "user", "content": plan.query_program.query},
    )
    prompt_messages = (
        *base_messages,
        {
            "role": smoke._EVIDENCE_MESSAGE_ROLE,
            "content": (
                smoke._EVIDENCE_PREFIX
                + packet.context
                + smoke._EVIDENCE_SUFFIX
            ),
        },
    )
    prompt_tokens = count_chat_prompt_token_proxy(
        prompt_messages,
        encoding=smoke._ENCODING,
    )
    assert packet.receipt.base_messages_sha256 == identity_sha256(
        list(base_messages)
    )
    assert packet.receipt.evidence_message_role == smoke._EVIDENCE_MESSAGE_ROLE
    assert packet.receipt.evidence_prefix_sha256 == quote_sha256(
        smoke._EVIDENCE_PREFIX
    )
    assert packet.receipt.evidence_suffix_sha256 == quote_sha256(
        smoke._EVIDENCE_SUFFIX
    )
    assert packet.receipt.prompt_messages_sha256 == identity_sha256(
        list(prompt_messages)
    )
    assert packet.receipt.prompt_token_proxy == prompt_tokens == 181
    assert packet.receipt.responder_output_token_reserve == 64
    assert packet.receipt.prompt_workspace_token_proxy == prompt_tokens + 64 == 245
    assert packet.receipt.max_prompt_token_proxy == smoke._MAX_PROMPT_TOKENS == 512
    assert not hasattr(plan, "gold")


def test_synthetic_packet_uses_one_public_full_prompt_packer() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = _function(tree, "_synthetic_packet_and_plan")
    call_names = [
        _call_name(node)
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
    ]
    assert call_names.count("pack_evidence_plan") == 1
    assert "ClosureReceipt" not in ast.get_source_segment(source, helper)
    assert (
        smoke._ENCODING,
        smoke._MAX_CONTEXT_TOKENS,
        smoke._MAX_PROMPT_TOKENS,
        smoke._OUTPUT_TOKEN_RESERVE,
        smoke._EVIDENCE_MESSAGE_ROLE,
    ) == ("cl100k_base", 256, 512, 64, "system")


def test_smoke_caps_are_the_frozen_real_checkpoint_profile() -> None:
    caps = smoke._fusion_caps()
    assert (
        caps.max_atoms,
        caps.max_latents,
        caps.max_hidden_dim,
        caps.max_route_cells,
        caps.max_topology_links,
        caps.max_hyperedges,
        caps.max_groups,
        caps.max_group_atoms,
        caps.max_latent_memberships_per_atom,
    ) == (2, 2, 4096, 4, 1, 1, 2, 2, 2)
    feature_caps = smoke._feature_caps()
    assert (
        feature_caps.max_row_tokens,
        feature_caps.max_query_tail_tokens,
        feature_caps.max_rows_per_forward,
        feature_caps.max_workspace_tokens,
        feature_caps.max_evidence_characters,
        feature_caps.max_query_characters,
        feature_caps.batch_invariance_atol,
        feature_caps.batch_invariance_rtol,
    ) == (128, 64, 2, 256, 256, 128, 1e-3, 1e-3)


def test_model_revision_and_prefix_digest_are_fixed_by_production_helper() -> None:
    expected = expected_prefix_checkpoint_sha256(
        1,
        model_id="Qwen/Qwen3-8B",
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
    )
    assert DEFAULT_MODEL_ID == smoke._MODEL_ID == "Qwen/Qwen3-8B"
    assert (
        DEFAULT_MODEL_REVISION
        == smoke._MODEL_REVISION
        == "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert expected == smoke._expected_checkpoint_sha256()
    assert expected == "76273516aa6924b12344d5e83daa485b66459b663c745cb3b9ef51cc17c7440d"
    assert (
        smoke._RETAINED_LAYERS,
        smoke._OUTPUT_LAYER,
        smoke._HIDDEN_DIM,
        smoke._DEVICE,
        smoke._DTYPE_NAME,
    ) == (1, 0, 4096, "cuda:0", "float16")


def test_public_runner_has_no_runtime_or_receipt_injection_surface() -> None:
    signature = inspect.signature(smoke.run_qwen_matched_fusion_smoke)
    assert tuple(signature.parameters) == ("model_dir",)
    assert signature.parameters["model_dir"].default == Path(
        ".cache/models/Qwen3-8B"
    )
    main_signature = inspect.signature(smoke.main)
    assert tuple(main_signature.parameters) == ("argv",)

    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    runner = _function(tree, "run_qwen_matched_fusion_smoke")
    calls = [node for node in ast.walk(runner) if isinstance(node, ast.Call)]
    call_names = [_call_name(call) for call in calls]
    assert call_names.count("build_qwen_matched_fusion_pair") == 1
    assert call_names.count("render_matched_fusion_contexts") == 1
    assert "run_execution_smoke" not in call_names
    assert "_run_feature_execution_smoke" not in call_names
    assert "build_evidence_fusion_plan" not in call_names
    assert "validate_matched_fusion_pair" not in call_names
    assert "_render_structural_matched_contexts" not in call_names
    assert "_arm_receipt" not in call_names
    assert "FusionRenderArmReceipt" not in call_names
    assert "MatchedFusionRenderReceipt" not in call_names
    assert "RenderedFusionContext" not in call_names
    assert "MatchedFusionContexts" not in call_names
    assert "factory" not in source.casefold()
    assert "corpus" not in source.casefold()
    assert "gold" not in source.casefold()


def test_runner_preflights_memory_and_calls_only_public_builder_and_renderer() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    runner = _function(tree, "run_qwen_matched_fusion_smoke")
    named_calls = [
        (_call_name(node), node.lineno)
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
    ]
    lines_by_name: dict[str, list[int]] = {}
    for name, line in named_calls:
        if name is not None:
            lines_by_name.setdefault(name, []).append(line)
    assert max(lines_by_name["_preflight_cuda"]) < min(
        lines_by_name["Qwen3PrefixEncoder"]
    )
    assert max(lines_by_name["_preflight_cuda"]) < min(
        lines_by_name["LatentEvidenceRouter"]
    )
    assert len(lines_by_name["build_qwen_matched_fusion_pair"]) == 1
    assert len(lines_by_name["render_matched_fusion_contexts"]) == 1
    assert min(lines_by_name["seal_for_inference"]) < min(
        lines_by_name["build_qwen_matched_fusion_pair"]
    )
    assert min(lines_by_name["reset_peak_memory_stats"]) < min(
        lines_by_name["build_qwen_matched_fusion_pair"]
    )
    assert max(lines_by_name["_assert_no_retained_cuda_allocation"]) < min(
        lines_by_name["_assert_success"]
    )
    assert max(lines_by_name["_assert_success"]) < min(
        lines_by_name["render_matched_fusion_contexts"]
    )
    assert max(lines_by_name["render_matched_fusion_contexts"]) < min(
        lines_by_name["_assert_render_success"]
    )

    preflight = _function(tree, "_preflight_cuda")
    preflight_calls = {
        _call_name(node)
        for node in ast.walk(preflight)
        if isinstance(node, ast.Call)
    }
    assert {
        "device",
        "is_available",
        "device_count",
        "mem_get_info",
        "memory_allocated",
        "getattr",
    } <= preflight_calls
    assert smoke._MIN_FREE_CUDA_BYTES == 3 * 1024**3
    assert "torch.float16" in ast.get_source_segment(
        SOURCE_PATH.read_text(encoding="utf-8"), preflight
    )
    preflight_source = ast.get_source_segment(
        SOURCE_PATH.read_text(encoding="utf-8"), preflight
    )
    assert preflight_source is not None
    assert "_require_cublas_workspace_clearer(torch)" in preflight_source


def test_cublas_clearer_requires_the_exact_owned_torch_builtin() -> None:
    torch = pytest.importorskip("torch")
    clearer = smoke._require_cublas_workspace_clearer(torch)
    assert inspect.isbuiltin(clearer)
    assert clearer.__module__ == "torch._C"
    assert clearer.__name__ == "_cuda_clearCublasWorkspaces"
    assert clearer.__self__ is torch._C

    helper_source = inspect.getsource(smoke._require_cublas_workspace_clearer)
    for required in (
        "inspect.isbuiltin",
        '"__module__"',
        '"torch._C"',
        '"__name__"',
        '"__self__"',
        "is not extension",
    ):
        assert required in helper_source


def test_missing_or_foreign_cublas_clearer_fails_closed() -> None:
    class MissingTorch:
        _C = object()

    class ForeignExtension:
        def _cuda_clearCublasWorkspaces(self) -> None:
            return None

    class ForeignTorch:
        _C = ForeignExtension()

    with pytest.raises(RuntimeError, match="exact cuBLAS workspace clearer"):
        smoke._require_cublas_workspace_clearer(MissingTorch())
    with pytest.raises(RuntimeError, match="exact cuBLAS workspace clearer"):
        smoke._require_cublas_workspace_clearer(ForeignTorch())


def test_cublas_clearer_drift_failure_and_return_value_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def captured() -> None:
        return None

    def replacement() -> None:
        return None

    monkeypatch.setattr(
        smoke,
        "_require_cublas_workspace_clearer",
        lambda _torch: replacement,
    )
    with pytest.raises(RuntimeError, match="changed during smoke"):
        smoke._clear_cublas_workspaces(object(), captured)

    def throwing() -> None:
        raise ValueError("synthetic clear failure")

    monkeypatch.setattr(
        smoke,
        "_require_cublas_workspace_clearer",
        lambda _torch: throwing,
    )
    with pytest.raises(RuntimeError, match="workspace clear failed"):
        smoke._clear_cublas_workspaces(object(), throwing)

    def returning() -> int:
        return 1

    monkeypatch.setattr(
        smoke,
        "_require_cublas_workspace_clearer",
        lambda _torch: returning,
    )
    with pytest.raises(RuntimeError, match="clearer returned a value"):
        smoke._clear_cublas_workspaces(object(), returning)


def test_retained_or_unexpectedly_released_cuda_allocation_fails_closed() -> None:
    baseline = {
        "resident_allocated": 100,
        "resident_reserved": 200,
        "raw_allocated": 130,
        "raw_reserved": 260,
        "post_clear_reserved": 240,
    }
    smoke._assert_no_retained_cuda_allocation(
        **baseline,
        post_clear_allocated=100,
    )
    with pytest.raises(RuntimeError) as retained:
        smoke._assert_no_retained_cuda_allocation(
            **baseline,
            post_clear_allocated=101,
        )
    assert "raw_allocated_delta_bytes=30" in str(retained.value)
    assert "raw_reserved_delta_bytes=60" in str(retained.value)
    assert "post_clear_allocated_delta_bytes=1" in str(retained.value)
    assert "post_clear_reserved_delta_bytes=40" in str(retained.value)
    with pytest.raises(RuntimeError, match="normalized resident baseline"):
        smoke._assert_no_retained_cuda_allocation(
            **baseline,
            post_clear_allocated=99,
        )


def test_cublas_normalization_surrounds_only_the_measured_builder() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    runner = _function(tree, "run_qwen_matched_fusion_smoke")
    runner_source = ast.get_source_segment(source, runner)
    assert runner_source is not None
    assert runner_source.count("_clear_cublas_workspaces(") == 2

    router_setup = runner_source.index("router_setup_seconds =")
    prebaseline_sync = runner_source.index(
        "torch.cuda.synchronize(device)", router_setup
    )
    prebaseline_clear = runner_source.index(
        "_clear_cublas_workspaces(torch, cublas_workspace_clearer)",
        prebaseline_sync,
    )
    prebaseline_post_sync = runner_source.index(
        "torch.cuda.synchronize(device)", prebaseline_clear
    )
    resident_allocated = runner_source.index(
        "resident_allocated_before_builder =", prebaseline_post_sync
    )
    resident_reserved = runner_source.index(
        "resident_reserved_before_builder =", resident_allocated
    )
    reset_peak = runner_source.index("reset_peak_memory_stats", resident_reserved)
    builder = runner_source.index("pair = build_qwen_matched_fusion_pair(", reset_peak)
    raw_allocated = runner_source.index("raw_allocated_after_builder =", builder)
    raw_reserved = runner_source.index("raw_reserved_after_builder =", raw_allocated)
    operation_peak = runner_source.index("operation_peak_allocated =", raw_reserved)
    postraw_sync = runner_source.index(
        "torch.cuda.synchronize(device)", operation_peak
    )
    postbuilder_clear = runner_source.index(
        "_clear_cublas_workspaces(torch, cublas_workspace_clearer)",
        postraw_sync,
    )
    postclear_sync = runner_source.index(
        "torch.cuda.synchronize(device)", postbuilder_clear
    )
    postclear_allocated = runner_source.index(
        "post_clear_allocated =", postclear_sync
    )
    postclear_reserved = runner_source.index(
        "post_clear_reserved =", postclear_allocated
    )
    allocation_gate = runner_source.index(
        "_assert_no_retained_cuda_allocation(", postclear_reserved
    )
    assert (
        router_setup
        < prebaseline_sync
        < prebaseline_clear
        < prebaseline_post_sync
        < resident_allocated
        < resident_reserved
        < reset_peak
        < builder
        < raw_allocated
        < raw_reserved
        < operation_peak
        < postraw_sync
        < postbuilder_clear
        < postclear_sync
        < postclear_allocated
        < postclear_reserved
        < allocation_gate
    )
    measured_section = runner_source[builder:allocation_gate]
    assert "gc.collect()" not in measured_section
    assert "empty_cache()" not in measured_section


def test_success_checks_cover_claims_gate_stability_and_tensor_absence() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    success = _function(tree, "_assert_success")
    success_source = ast.get_source_segment(source, success)
    assert success_source is not None
    for required in (
        "qwen_forward_count",
        "router_forward_count",
        "topology_reencode_count",
        "bounded_route_matrix_content_attested",
        "route_matrix_values_retained",
        "feature_tensor_content_attested",
        "steered_tensor_content_attested",
        "single_feature_workspace_attested",
        "operation_inputs_attested",
        "retrieval_route_attested",
        "performance_attested",
        "retained_request_tensor_bytes",
        "atom_identity_sha256",
        "span_identity_sha256",
        "quote_sha256",
        "evidence_character_count",
        "query_character_count",
        "evidence_tokens_admitted",
        "evidence_tokens_observed",
        "evidence_truncated",
        "checkpoint_status",
        "pooling",
        "truncation_rule",
        "loaded_module_runtime_constants_attested",
        "resident_runtime_receipt",
        "_assert_runtime",
        "_assert_inference_seal",
        "_qwen_prefix_gate_state",
        "active_token",
        "lock.acquire",
        "lock.release",
        "_contains_tensor",
    ):
        assert required in success_source


def test_render_success_checks_exact_public_output_without_fabricated_receipts() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    render_success = _function(tree, "_assert_render_success")
    render_source = ast.get_source_segment(source, render_success)
    assert render_source is not None
    for required in (
        "MatchedFusionContexts",
        "packet_receipt_sha256",
        "matched_pair_sha256",
        "fusion_plan_sha256",
        "render_receipt_sha256",
        "renderer_implementation_sha256",
        "prompt_frame_sha256",
        "memory-condense-evidence-prompt-frame-v1",
        "tokenizer_identity",
        "base_messages_sha256",
        "evidence_message_role",
        "evidence_prefix_sha256",
        "evidence_suffix_sha256",
        "max_prompt_token_proxy",
        "responder_output_token_reserve",
        "pair_wide_fallback_applied",
        "plan_applied",
        "context_cap_compliant",
        "prompt_cap_compliant",
        "exact_atom_set_preserved",
        "exact_bundle_set_preserved",
        "exact_evidence_bytes_preserved",
        "retained_request_tensor_bytes",
        "resident_values_sha256",
        "resident_atom_order_sha256",
        "render_evidence_context",
        "render_grouped_evidence_context",
        "count_tokens",
        "count_chat_prompt_token_proxy",
        "prompt_messages_sha256",
        "prompt_workspace_token_proxy",
        "qwen_forward_count",
        "router_forward_count",
        "_contains_tensor",
        "identity_payload",
        "bundle_id",
        "obligation_ids",
        "source_id",
        "chunk_id",
        "turn_id",
    ):
        assert required in render_source
    assert "FusionRenderArmReceipt(" not in render_source
    assert "MatchedFusionRenderReceipt(" not in render_source
    assert "RenderedFusionContext(" not in render_source
    assert "MatchedFusionContexts(" not in render_source


def test_runner_finally_closes_and_releases_without_certifying_measurements() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    runner = _function(tree, "run_qwen_matched_fusion_smoke")
    finalizers = [node.finalbody for node in ast.walk(runner) if isinstance(node, ast.Try)]
    assert finalizers
    finalizer_source = "\n".join(
        ast.get_source_segment(source, node) or ""
        for body in finalizers
        for node in body
    )
    for required in (
        "rendered = None",
        "pair = None",
        "router = None",
        "provider.close()",
        "provider = None",
        "encoder = None",
        "gc.collect()",
        "torch.cuda.empty_cache()",
        "torch.cuda.synchronize(device)",
        "primary_error",
    ):
        assert required in finalizer_source
    assert '"diagnostic_non_artifact": True' in source
    assert '"performance_attested": False' in source
    assert '"format": "qwen_matched_fusion_local_diagnostic_v2"' in source
    assert "matched_render_sha256" in source
    assert "renderer_implementation_sha256" in source
    assert "render_prompt_frame_sha256" in source
    assert "topology_render_receipt_sha256" in source
    assert "latent_render_receipt_sha256" in source
    assert "render_pair_wide_fallback_applied" in source
    assert "observed_render_seconds" not in source
    assert "render_seconds" not in source
    assert "observed_fusion_seconds" in source
    assert "observed_cuda_allocated_bytes_before_load" in source
    assert "observed_cuda_allocated_bytes_before_builder" in source
    assert "observed_cuda_reserved_bytes_before_builder" in source
    assert "observed_cuda_allocated_bytes_after_builder" in source
    assert "observed_cuda_reserved_bytes_after_builder" in source
    assert "observed_raw_cuda_allocated_bytes_after_builder" in source
    assert "observed_raw_cuda_reserved_bytes_after_builder" in source
    assert "observed_raw_cuda_allocated_delta_bytes_after_builder" in source
    assert "observed_raw_cuda_reserved_delta_bytes_after_builder" in source
    assert "observed_cuda_allocated_bytes_after_cublas_clear" in source
    assert "observed_cuda_reserved_bytes_after_cublas_clear" in source
    assert "observed_cuda_allocated_delta_bytes_after_cublas_clear" in source
    assert "observed_cuda_reserved_delta_bytes_after_cublas_clear" in source
    assert "observed_operation_peak_cuda_allocated_bytes" in source
    assert (
        '"post_cublas_clear_allocated_equals_normalized_resident_v1"'
        in source
    )
    assert '"torch._C._cuda_clearCublasWorkspaces"' in source
    assert "certified" not in source.casefold().replace("uncertified", "")


def test_recursive_tensor_guard_descends_supported_return_shapes() -> None:
    class TensorSentinel:
        pass

    @dataclass(frozen=True)
    class Wrapper:
        value: object

    assert smoke._contains_tensor(Wrapper({"nested": (TensorSentinel(),)}), TensorSentinel)
    assert not smoke._contains_tensor(Wrapper({"nested": ("digest", 1)}), TensorSentinel)


def test_tooling_module_keeps_torch_and_transformers_cold() -> None:
    code = (
        "import sys; import memory_condense.tooling.qwen_matched_fusion_smoke; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False", result.stdout


def test_pixi_exposes_one_fixed_real_matched_fusion_smoke_task() -> None:
    pixi = (REPO_ROOT / "pixi.toml").read_text(encoding="utf-8")
    assert (
        'qwen-matched-fusion-smoke = "python -m '
        'memory_condense.tooling.qwen_matched_fusion_smoke"'
    ) in pixi
