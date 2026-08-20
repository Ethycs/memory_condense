"""Resident Qwen atom rows and a receipt-only tranche-A execution smoke.

The public provider never returns node features.  Its private consume-once
workspace exists only so the later atomic matched-pair tranche can route the
same resident ``[N, D]`` tensor without a host copy or a second Qwen pass.
"""

from __future__ import annotations

import hashlib
import inspect
import marshal
from dataclasses import dataclass, fields
from pathlib import Path
from types import CodeType
from typing import Any, Sequence

import memory_condense.search.fusion.qwen_feature_models as feature_models_module
import memory_condense.search.fusion.qwen_feature_executor as executor_module
import memory_condense.search.fusion.qwen_feature_runtime as runtime_module
import memory_condense.modeling.qwen_prefix as prefix_module
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain.discourse import (
    ClosurePlan,
    ClosurePolicy,
    ClosureReceipt,
    ClosureScopeWitness,
    DiscourseSnapshot,
    EpisodeSeed,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceObligation,
    EvidencePacket,
    EvidenceSpan,
    ObligationResult,
    QueryProgram,
)
from memory_condense.search.fusion.models import (
    AuthoritativeHyperedge,
    FusionAtomRef,
    FusionCaps,
)
from memory_condense.search.fusion.qwen_feature_models import (
    QwenAtomBatchReceipt,
    QwenAtomFeatureCaps,
    QwenAtomFeatureOperationReceipt,
    QwenAtomFeatureProviderReceipt,
    QwenAtomRowReceipt,
)
from memory_condense.search.fusion.qwen_feature_executor import (
    _DiscardFeatures,
    _FeatureExecutionDiagnostics,
    _QwenFeatureLease,
    _execute_feature_batches,
    _validate_feature_output,
)
from memory_condense.search.fusion.qwen_feature_runtime import (
    _OwnedRuntimeSnapshot,
    _reject_global_module_hooks,
    _runtime_fingerprint,
)


_EVIDENCE_PREFIX = "[Evidence]\n"
_QUESTION_PREFIX = "\n[Question]\n"
_READOUT_SUFFIX = "\n[Readout]"
_READOUT_MARKER = "[Readout]"
_PROVIDER_ID = "qwen3_prefix.query_readout_last.v1"
_PINNED_GATE_FACTORY = prefix_module._qwen_prefix_execution_gate
_PINNED_REQUIRE_GATE = prefix_module._require_qwen_prefix_gate
_PINNED_GATE_STATE = prefix_module._qwen_prefix_gate_state
_PINNED_VALIDATED_LAYERS = prefix_module._validated_layers
_PINNED_REQUIRE_TORCH_STACK = prefix_module._require_torch_stack
_PINNED_EXPECTED_CHECKPOINT_SHA256 = (
    prefix_module.expected_prefix_checkpoint_sha256
)
_PINNED_KNOWN_REQUIRED_SHARDS = prefix_module._known_required_shards
_PINNED_PREFIX_METADATA_FILES = prefix_module._PREFIX_METADATA_FILES
_PINNED_ENCODER_TYPE = prefix_module.Qwen3PrefixEncoder
_PINNED_CHECKPOINT_IDENTITY_TYPE = prefix_module.QwenPrefixCheckpointIdentity
_PINNED_DEFAULT_MODEL_ID = prefix_module.DEFAULT_MODEL_ID
_PINNED_DEFAULT_MODEL_REVISION = prefix_module.DEFAULT_MODEL_REVISION
_PINNED_FINAL_READOUT_PRIMITIVE = (
    _PINNED_ENCODER_TYPE._encode_selected_layer_final_readout
)
_PINNED_CHECKPOINT_SHA256_BY_LAYER = tuple(
    _PINNED_EXPECTED_CHECKPOINT_SHA256(
        layers,
        model_id=_PINNED_DEFAULT_MODEL_ID,
        model_revision=_PINNED_DEFAULT_MODEL_REVISION,
    )
    for layers in range(1, 37)
)
_PINNED_VERIFIED_FILES_BY_LAYER = tuple(
    (
        *_PINNED_PREFIX_METADATA_FILES,
        *_PINNED_KNOWN_REQUIRED_SHARDS(layers),
    )
    for layers in range(1, 37)
)
_PROMPT_TEMPLATE_SHA256 = identity_sha256(
    {
        "format": "memory-condense-qwen-atom-row-v1",
        "evidence_prefix_sha256": quote_sha256(_EVIDENCE_PREFIX),
        "question_prefix_sha256": quote_sha256(_QUESTION_PREFIX),
        "readout_suffix_sha256": quote_sha256(_READOUT_SUFFIX),
        "segments_tokenized_separately": True,
        "add_special_tokens": False,
        "bos_added": False,
        "eos_added": False,
        "evidence_truncation": "prefix_only",
        "pooling": "final_readout_token",
    }
)


@dataclass(frozen=True, slots=True)
class _QwenAtomTokenRow:
    token_ids: tuple[int, ...]
    receipt: QwenAtomRowReceipt

    def __post_init__(self) -> None:
        if type(self.receipt) is not QwenAtomRowReceipt:
            raise TypeError("row receipt must be an exact QwenAtomRowReceipt")
        token_ids = tuple(self.token_ids)
        if len(token_ids) != self.receipt.total_row_tokens:
            raise ValueError("token row length disagrees with its receipt")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids
        ):
            raise TypeError("token rows must contain non-negative exact integers")
        object.__setattr__(self, "token_ids", token_ids)


@dataclass(frozen=True, slots=True)
class _ValidatedOperationInputs:
    """Bounded pre-execution snapshot used for post-execution TOCTOU checks."""

    fingerprint: str
    packet_receipt_sha256: str
    closure_plan_sha256: str
    query_program_sha256: str
    query_sha256: str
    closure_policy_sha256: str
    snapshot_sha256: str
    query: str
    atom_values: tuple[EvidenceAtom, ...]
    caps: FusionCaps
    feature_caps: QwenAtomFeatureCaps
    atoms: tuple[FusionAtomRef, ...]
    hyperedges: tuple[AuthoritativeHyperedge, ...]


@dataclass(frozen=True, slots=True)
class _OwnedImplementation:
    preflight_packet: Any
    validate_packet_plan: Any
    capture_operation_inputs: Any
    revalidate_operation_inputs: Any
    build_atom_rows: Any
    batch_rows: Any
    execute_feature_batches: Any
    diagnostics_type: type
    batch_receipt_type: type
    operation_receipt_type: type


def _exact_token_ids(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} token IDs must be a sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{label} token IDs must be a sequence") from exc
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in values
    ):
        raise TypeError(f"{label} token IDs must be non-negative exact integers")
    if not values:
        raise ValueError(f"{label} must produce at least one token")
    return values


def _tokenize_segment(
    tokenizer: Any,
    text: str,
    label: str,
    *,
    observe_at_most: int | None = None,
) -> tuple[int, ...]:
    kwargs: dict[str, Any] = {"add_special_tokens": False}
    if observe_at_most is not None:
        if observe_at_most < 1:
            raise ValueError("bounded token observation must be positive")
        kwargs.update(truncation=True, max_length=observe_at_most)
    encoded = tokenizer(text, **kwargs)
    try:
        raw_ids = encoded["input_ids"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Qwen tokenizer did not return input_ids for {label}") from exc
    return _exact_token_ids(raw_ids, label)


def _build_qwen_atom_rows(
    tokenizer: Any,
    atoms: Sequence[EvidenceAtom],
    query: str,
    caps: QwenAtomFeatureCaps,
    *,
    _row_receipt_type: type = QwenAtomRowReceipt,
    _token_row_type: type = _QwenAtomTokenRow,
) -> tuple[_QwenAtomTokenRow, ...]:
    """Build exact query-preserving token rows without decode/re-tokenize."""

    if type(caps) is not QwenAtomFeatureCaps:
        raise TypeError("caps must be an exact QwenAtomFeatureCaps")
    exact_atoms = tuple(atoms)
    if not exact_atoms or any(type(atom) is not EvidenceAtom for atom in exact_atoms):
        raise TypeError("atoms must be a non-empty exact EvidenceAtom sequence")
    if getattr(tokenizer, "truncation_side", None) != "right":
        raise RuntimeError("Qwen atom rows require right-side segment truncation")
    if len(query) > caps.max_query_characters:
        raise MemoryError("raw query exceeds max_query_characters")
    if any(len(atom.text) > caps.max_evidence_characters for atom in exact_atoms):
        raise MemoryError("an evidence atom exceeds max_evidence_characters")
    prefix_ids = _tokenize_segment(tokenizer, _EVIDENCE_PREFIX, "evidence prefix")
    tail_ids = _tokenize_segment(
        tokenizer,
        _QUESTION_PREFIX + query + _READOUT_SUFFIX,
        "query/readout tail",
        observe_at_most=caps.max_query_tail_tokens + 1,
    )
    readout_ids = _tokenize_segment(tokenizer, _READOUT_MARKER, "readout marker")
    if len(tail_ids) > caps.max_query_tail_tokens:
        raise MemoryError("query/readout tail exceeds max_query_tail_tokens")
    if len(tail_ids) < len(readout_ids) or tail_ids[-len(readout_ids) :] != readout_ids:
        raise RuntimeError("query tail does not end with the complete readout marker")
    evidence_budget = caps.max_row_tokens - len(prefix_ids) - len(tail_ids)
    if evidence_budget < 1:
        raise MemoryError("Qwen row caps leave no evidence-token budget")

    rows: list[_QwenAtomTokenRow] = []
    for row_index, atom in enumerate(exact_atoms):
        if quote_sha256(atom.text) != atom.span.quote_sha256:
            raise ValueError("atom text no longer matches its authoritative span")
        evidence_ids = _tokenize_segment(
            tokenizer,
            atom.text,
            f"evidence atom {row_index}",
            observe_at_most=evidence_budget + 1,
        )
        if len(evidence_ids) > evidence_budget + 1:
            raise RuntimeError("Qwen tokenizer ignored the bounded evidence observation")
        admitted = evidence_ids[:evidence_budget]
        token_ids = (*prefix_ids, *admitted, *tail_ids)
        if len(token_ids) > caps.max_row_tokens:  # pragma: no cover - arithmetic guard
            raise MemoryError("constructed Qwen row exceeds max_row_tokens")
        if token_ids[-len(readout_ids) :] != readout_ids:
            raise RuntimeError("constructed row lost the complete readout marker")
        receipt = _row_receipt_type(
            row_index=row_index,
            atom_id=atom.atom_id,
            atom_identity_sha256=identity_sha256(atom.identity_payload()),
            span_identity_sha256=identity_sha256(atom.span.identity_payload()),
            quote_sha256=atom.span.quote_sha256,
            evidence_character_count=len(atom.text),
            query_character_count=len(query),
            prefix_tokens=len(prefix_ids),
            evidence_tokens_observed=len(evidence_ids),
            evidence_tokens_admitted=len(admitted),
            query_tail_tokens=len(tail_ids),
            total_row_tokens=len(token_ids),
            readout_end_index=len(token_ids) - 1,
            evidence_truncated=len(evidence_ids) == evidence_budget + 1,
        )
        rows.append(_token_row_type(token_ids=token_ids, receipt=receipt))
    return tuple(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _callable_sha256(value: Any) -> str:
    code = getattr(value, "__code__", None)
    if code is not None:
        callable_name = (
            f"{getattr(value, '__module__', '')}."
            f"{getattr(value, '__qualname__', '')}"
        )
        encoded = marshal.dumps(
            _canonical_code_object(code, stable_filename=callable_name)
        )
    else:
        try:
            encoded = inspect.getsource(value).encode("utf-8")
        except (OSError, TypeError):
            encoded = (
                f"{getattr(value, '__module__', '')}:"
                f"{getattr(value, '__qualname__', '')}"
            ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_code_object(code: CodeType, *, stable_filename: str) -> CodeType:
    constants = tuple(
        (
            _canonical_code_object(value, stable_filename=stable_filename)
            if isinstance(value, CodeType)
            else value
        )
        for value in code.co_consts
    )
    return code.replace(co_filename=stable_filename, co_consts=constants)


def _expected_runtime_types() -> tuple[Any, type, type, type]:
    from transformers.models.qwen2.tokenization_qwen2_fast import (
        Qwen2TokenizerFast,
    )

    stack = _PINNED_REQUIRE_TORCH_STACK()
    return stack[0], stack[7], Qwen2TokenizerFast, stack[6]


def _assert_pinned_prefix_seams() -> None:
    if (
        prefix_module._qwen_prefix_execution_gate is not _PINNED_GATE_FACTORY
        or prefix_module._require_qwen_prefix_gate is not _PINNED_REQUIRE_GATE
        or prefix_module._qwen_prefix_gate_state is not _PINNED_GATE_STATE
        or prefix_module._validated_layers is not _PINNED_VALIDATED_LAYERS
        or prefix_module._require_torch_stack is not _PINNED_REQUIRE_TORCH_STACK
        or prefix_module.expected_prefix_checkpoint_sha256
        is not _PINNED_EXPECTED_CHECKPOINT_SHA256
        or prefix_module._known_required_shards
        is not _PINNED_KNOWN_REQUIRED_SHARDS
        or prefix_module._PREFIX_METADATA_FILES
        is not _PINNED_PREFIX_METADATA_FILES
        or prefix_module.Qwen3PrefixEncoder is not _PINNED_ENCODER_TYPE
        or prefix_module.QwenPrefixCheckpointIdentity
        is not _PINNED_CHECKPOINT_IDENTITY_TYPE
        or prefix_module.DEFAULT_MODEL_ID != _PINNED_DEFAULT_MODEL_ID
        or prefix_module.DEFAULT_MODEL_REVISION != _PINNED_DEFAULT_MODEL_REVISION
        or _PINNED_ENCODER_TYPE._encode_selected_layer_final_readout
        is not _PINNED_FINAL_READOUT_PRIMITIVE
    ):
        raise RuntimeError("owned Qwen prefix execution seams were replaced")


def _expected_checkpoint_sha256(
    encoder: Any,
    _table: tuple[str, ...] = _PINNED_CHECKPOINT_SHA256_BY_LAYER,
) -> str:
    layers = int(encoder.layers)
    if not 1 <= layers <= len(_table):
        raise ValueError("Qwen retained layers lie outside the pinned manifest")
    return _table[layers - 1]


def _expected_verified_files(
    encoder: Any,
    _table: tuple[tuple[str, ...], ...] = _PINNED_VERIFIED_FILES_BY_LAYER,
) -> tuple[str, ...]:
    layers = int(encoder.layers)
    if not 1 <= layers <= len(_table):
        raise ValueError("Qwen retained layers lie outside the pinned manifest")
    return _table[layers - 1]


def _owned_local_objects() -> tuple[Any, ...]:
    return (
        QwenAtomFeatureProvider.__init__,
        QwenAtomFeatureProvider.__dict__["__init_subclass__"].__func__,
        QwenAtomFeatureProvider.run_execution_smoke,
        QwenAtomFeatureProvider._run_feature_execution_smoke,
        QwenAtomFeatureProvider._assert_provider_state,
        QwenAtomFeatureProvider._assert_implementation,
        QwenAtomFeatureProvider._assert_runtime,
        QwenAtomFeatureProvider.close,
        _QwenAtomTokenRow,
        _QwenFeatureLease,
        _QwenFeatureLease._discard_once,
        _QwenFeatureLease._close,
        _DiscardFeatures,
        _ValidatedOperationInputs,
        _FeatureExecutionDiagnostics,
        _OwnedImplementation,
        _pinned_owned_implementation,
        QwenAtomBatchReceipt,
        QwenAtomFeatureCaps,
        QwenAtomFeatureOperationReceipt,
        QwenAtomFeatureProviderReceipt,
        QwenAtomRowReceipt,
        _preflight_packet,
        _validate_packet_plan,
        _capture_operation_inputs,
        _revalidate_operation_inputs,
        _build_qwen_atom_rows,
        _tokenize_segment,
        _batch_rows,
        _validate_feature_output,
        _execute_feature_batches,
        _packet_refs,
        _packet_hyperedges,
        _runtime_fingerprint,
        _reject_global_module_hooks,
        runtime_module._canonical_cuda_device,
        runtime_module._reject_hooks,
        runtime_module._bounded_runtime_value,
        runtime_module._declared_runtime_fields,
        _expected_runtime_types,
        _assert_pinned_prefix_seams,
        _expected_checkpoint_sha256,
        _expected_verified_files,
        _file_sha256,
        _callable_sha256,
        _canonical_code_object,
        _implementation_callables,
        _implementation_callable_fingerprint,
        _implementation_sha256,
        _PINNED_GATE_FACTORY,
        _PINNED_REQUIRE_GATE,
        _PINNED_GATE_STATE,
        _PINNED_VALIDATED_LAYERS,
        _PINNED_REQUIRE_TORCH_STACK,
        _PINNED_EXPECTED_CHECKPOINT_SHA256,
        _PINNED_KNOWN_REQUIRED_SHARDS,
        _PINNED_CHECKPOINT_SHA256_BY_LAYER,
        _PINNED_VERIFIED_FILES_BY_LAYER,
        _PINNED_ENCODER_TYPE,
        _PINNED_CHECKPOINT_IDENTITY_TYPE,
        _PINNED_FINAL_READOUT_PRIMITIVE,
        executor_module._QwenFeatureLease,
        executor_module._QwenFeatureLease._discard_once,
        executor_module._DiscardFeatures,
        executor_module._FeatureExecutionDiagnostics,
        executor_module._validate_feature_output,
        executor_module._execute_feature_batches,
    )


def _owned_local_fingerprint() -> tuple[Any, ...]:
    return tuple(
        (
            str(getattr(value, "__module__", "")),
            str(getattr(value, "__qualname__", "")),
            id(value),
            id(getattr(value, "__code__", None)),
            id(getattr(value, "__defaults__", None)),
            tuple(id(item) for item in (getattr(value, "__defaults__", None) or ())),
            tuple(
                sorted(
                    (name, id(item))
                    for name, item in (getattr(value, "__kwdefaults__", None) or {}).items()
                )
            ),
            _callable_sha256(value),
        )
        for value in _owned_local_objects()
    )


def _assert_owned_local_implementation() -> None:
    if _owned_local_fingerprint() != _OWNED_LOCAL_IMPLEMENTATION_FINGERPRINT:
        raise RuntimeError("owned Qwen feature implementation was replaced")


def _implementation_callables(
    encoder: Any,
    *,
    expected_model_type: type,
    expected_tokenizer_type: type,
) -> tuple[Any, ...]:
    import memory_condense.modeling.qwen_prefix as prefix_module

    modules = tuple(encoder.model.modules())
    forward_types = tuple(dict.fromkeys(type(module) for module in modules))
    return (
        *_owned_local_objects(),
        _implementation_callables,
        _implementation_callable_fingerprint,
        _implementation_sha256,
        prefix_module._qwen_prefix_execution_gate,
        prefix_module._require_qwen_prefix_gate,
        _PINNED_FINAL_READOUT_PRIMITIVE,
        expected_model_type.forward,
        expected_tokenizer_type.__call__,
        *(module_type.forward for module_type in forward_types),
    )


def _implementation_callable_fingerprint(
    encoder: Any,
    *,
    expected_model_type: type,
    expected_tokenizer_type: type,
) -> tuple[Any, ...]:
    return tuple(
        (
            str(getattr(value, "__module__", "")),
            str(getattr(value, "__qualname__", "")),
            id(value),
            id(getattr(value, "__code__", None)),
            id(getattr(value, "__defaults__", None)),
            tuple(id(item) for item in (getattr(value, "__defaults__", None) or ())),
            tuple(
                sorted(
                    (name, id(item))
                    for name, item in (getattr(value, "__kwdefaults__", None) or {}).items()
                )
            ),
            _callable_sha256(value),
        )
        for value in _implementation_callables(
            encoder,
            expected_model_type=expected_model_type,
            expected_tokenizer_type=expected_tokenizer_type,
        )
    )


def _implementation_sha256(
    encoder: Any,
    *,
    expected_model_type: type,
    expected_tokenizer_type: type,
) -> str:
    import memory_condense.modeling.qwen_prefix as prefix_module

    callables = _implementation_callables(
        encoder,
        expected_model_type=expected_model_type,
        expected_tokenizer_type=expected_tokenizer_type,
    )
    return identity_sha256(
        {
            "format": "memory-condense-qwen-atom-feature-implementation-v1",
            "source_files": {
                "qwen_features": _file_sha256(Path(__file__)),
                "qwen_feature_models": _file_sha256(
                    Path(feature_models_module.__file__)
                ),
                "qwen_feature_executor": _file_sha256(Path(executor_module.__file__)),
                "qwen_feature_runtime": _file_sha256(Path(runtime_module.__file__)),
                "qwen_prefix": _file_sha256(Path(prefix_module.__file__)),
            },
            "runtime_versions": {
                "torch": str(getattr(encoder._torch, "__version__", "")),
                "transformers": str(
                    __import__("transformers").__version__
                ),
                "tokenizers": str(__import__("tokenizers").__version__),
            },
            "callables": [
                {
                    "module": str(getattr(value, "__module__", "")),
                    "qualname": str(getattr(value, "__qualname__", "")),
                    "code_sha256": _callable_sha256(value),
                }
                for value in callables
            ],
        }
    )


def _packet_refs(packet: EvidencePacket) -> tuple[FusionAtomRef, ...]:
    return tuple(
        FusionAtomRef(
            atom_id=atom.atom_id,
            atom_identity_sha256=identity_sha256(atom.identity_payload()),
            span_identity_sha256=identity_sha256(atom.span.identity_payload()),
            quote_sha256=atom.span.quote_sha256,
        )
        for atom in packet.atoms
    )


def _packet_hyperedges(packet: EvidencePacket) -> tuple[AuthoritativeHyperedge, ...]:
    return tuple(
        AuthoritativeHyperedge(
            bundle_id=bundle.bundle_id,
            atom_ids=bundle.atom_ids,
            obligation_ids=bundle.obligation_ids,
            unit_witness_ids=bundle.unit_ids,
            relation_witness_ids=bundle.relation_ids,
            required=bundle.required,
            utility=bundle.utility,
        )
        for bundle in packet.bundles
    )


def _preflight_packet(
    packet: Any,
    plan: Any,
    caps: Any,
    feature_caps: Any,
    *,
    hidden_dim: int,
) -> None:
    if type(packet) is not EvidencePacket:
        raise TypeError("packet must be an exact EvidencePacket")
    if type(plan) is not ClosurePlan:
        raise TypeError("plan must be an exact ClosurePlan")
    if type(caps) is not FusionCaps:
        raise TypeError("caps must be an exact FusionCaps")
    if type(feature_caps) is not QwenAtomFeatureCaps:
        raise TypeError("feature_caps must be an exact QwenAtomFeatureCaps")
    caps._seal()
    feature_caps._seal()
    if type(packet.receipt) is not ClosureReceipt:
        raise TypeError("packet receipt must be an exact ClosureReceipt")
    if type(plan.query_program) is not QueryProgram:
        raise TypeError("closure query_program must be an exact QueryProgram")
    if type(plan.policy) is not ClosurePolicy:
        raise TypeError("closure policy must be an exact ClosurePolicy")
    if type(plan.snapshot) is not DiscourseSnapshot:
        raise TypeError("closure snapshot must be an exact DiscourseSnapshot")
    if type(packet.context) is not str or type(plan.query_program.query) is not str:
        raise TypeError("packet context and closure query must be exact strings")
    if type(packet.atoms) is not tuple or type(packet.bundles) is not tuple:
        raise TypeError("packet atom and bundle collections must be exact tuples")
    if type(plan.atoms) is not tuple or type(plan.bundles) is not tuple:
        raise TypeError("plan atom and bundle collections must be exact tuples")
    if not packet.atoms:
        raise ValueError("Qwen feature operation requires selected atoms")
    if any(type(atom) is not EvidenceAtom for atom in packet.atoms):
        raise TypeError("packet atoms must be exact EvidenceAtom values")
    if any(type(bundle) is not EvidenceBundle for bundle in packet.bundles):
        raise TypeError("packet bundles must be exact EvidenceBundle values")
    if any(type(atom) is not EvidenceAtom for atom in plan.atoms):
        raise TypeError("plan atoms must be exact EvidenceAtom values")
    if any(type(bundle) is not EvidenceBundle for bundle in plan.bundles):
        raise TypeError("plan bundles must be exact EvidenceBundle values")
    all_atoms = (*packet.atoms, *plan.atoms)
    if any(
        type(atom.span) is not EvidenceSpan
        or type(atom.text) is not str
        or type(atom.atom_id) is not str
        for atom in all_atoms
    ):
        raise TypeError("atom bodies must retain exact span and string fields")
    all_bundles = (*packet.bundles, *plan.bundles)
    bundle_id_fields = (
        "atom_ids",
        "obligation_ids",
        "unit_ids",
        "relation_ids",
    )
    if any(
        type(bundle.bundle_id) is not str
        or any(
            type(getattr(bundle, name)) is not tuple
            or any(type(value) is not str for value in getattr(bundle, name))
            for name in bundle_id_fields
        )
        for bundle in all_bundles
    ):
        raise TypeError("bundle bodies must retain exact tuple/string fields")
    plan_tuple_types = (
        (plan.query_program.obligations, EvidenceObligation, "query obligations"),
        (plan.seeds, EpisodeSeed, "plan seeds"),
        (plan.obligation_results, ObligationResult, "obligation results"),
        (plan.scope_witnesses, ClosureScopeWitness, "scope witnesses"),
    )
    for values, expected_type, label in plan_tuple_types:
        if type(values) is not tuple or any(
            type(value) is not expected_type for value in values
        ):
            raise TypeError(f"{label} must be an exact tuple of owned values")
    if len(packet.atoms) > caps.max_atoms:
        raise MemoryError("packet atom count exceeds FusionCaps.max_atoms")
    if len(packet.bundles) > caps.max_hyperedges:
        raise MemoryError("packet bundle count exceeds FusionCaps.max_hyperedges")
    raw_links = sum(
        len(bundle.atom_ids) * (len(bundle.atom_ids) - 1) // 2
        for bundle in packet.bundles
    )
    if raw_links > caps.max_topology_links:
        raise MemoryError("packet co-memberships exceed FusionCaps.max_topology_links")
    if hidden_dim > caps.max_hidden_dim:
        raise MemoryError("Qwen hidden width exceeds FusionCaps.max_hidden_dim")
    if len(plan.query_program.query) > feature_caps.max_query_characters:
        raise MemoryError("raw query exceeds max_query_characters")
    if any(
        len(atom.text) > feature_caps.max_evidence_characters
        for atom in packet.atoms
    ):
        raise MemoryError("an evidence atom exceeds max_evidence_characters")


def _validate_packet_plan(packet: EvidencePacket, plan: ClosurePlan) -> None:
    # Recompute every stored seal whose child identity is projected by digest,
    # then recheck the unsealed policy through the parent plan digest.
    plan.query_program._seal()
    plan.snapshot._seal()
    plan._seal()
    packet.receipt._seal()
    if quote_sha256(packet.context) != packet.receipt.context_sha256:
        raise ValueError("packet context no longer matches its closure receipt")
    if packet.receipt.plan_sha256 != plan.plan_sha256:
        raise ValueError("packet receipt does not bind the supplied closure plan")
    if packet.receipt.selected_atom_ids != tuple(atom.atom_id for atom in packet.atoms):
        raise ValueError("packet atom order disagrees with its receipt")
    if packet.receipt.selected_bundle_ids != tuple(
        bundle.bundle_id for bundle in packet.bundles
    ):
        raise ValueError("packet bundle order disagrees with its receipt")
    if any(type(atom) is not EvidenceAtom for atom in packet.atoms):
        raise TypeError("packet atoms must be exact EvidenceAtom values")
    if any(type(bundle) is not EvidenceBundle for bundle in packet.bundles):
        raise TypeError("packet bundles must be exact EvidenceBundle values")
    atom_ids = tuple(atom.atom_id for atom in packet.atoms)
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError("packet atom IDs must be unique")
    plan_atoms = {atom.atom_id: atom for atom in plan.atoms}
    if len(plan_atoms) != len(plan.atoms):
        raise ValueError("closure plan atom IDs must be unique")
    for atom in packet.atoms:
        planned = plan_atoms.get(atom.atom_id)
        if planned is None or identity_sha256(atom.identity_payload()) != identity_sha256(
            planned.identity_payload()
        ):
            raise ValueError("packet atom does not exactly match the closure plan")
        if quote_sha256(atom.text) != atom.span.quote_sha256:
            raise ValueError("packet atom text does not match its source span")
    plan_bundles = {bundle.bundle_id: bundle for bundle in plan.bundles}
    selected = set(atom_ids)
    for bundle in packet.bundles:
        planned = plan_bundles.get(bundle.bundle_id)
        if planned is None or identity_sha256(bundle.identity_payload()) != identity_sha256(
            planned.identity_payload()
        ):
            raise ValueError("packet bundle does not exactly match the closure plan")
        if any(atom_id not in selected for atom_id in bundle.atom_ids):
            raise ValueError("packet bundle references an unselected atom")


def _capture_operation_inputs(
    packet: EvidencePacket,
    plan: ClosurePlan,
    caps: FusionCaps,
    feature_caps: QwenAtomFeatureCaps,
    *,
    _packet_refs_fn: Any = _packet_refs,
    _packet_hyperedges_fn: Any = _packet_hyperedges,
) -> _ValidatedOperationInputs:
    detached_caps = FusionCaps(
        **{
            item.name: getattr(caps, item.name)
            for item in fields(FusionCaps)
            if item.name != "caps_sha256"
        }
    )
    detached_feature_caps = QwenAtomFeatureCaps(
        **{
            item.name: getattr(feature_caps, item.name)
            for item in fields(QwenAtomFeatureCaps)
            if item.name != "caps_sha256"
        }
    )
    atom_values = tuple(
        EvidenceAtom(
            **{
                **{
                    item.name: getattr(atom, item.name)
                    for item in fields(EvidenceAtom)
                    if item.name != "span"
                },
                "span": EvidenceSpan(
                    **{
                        item.name: getattr(atom.span, item.name)
                        for item in fields(EvidenceSpan)
                    }
                ),
            }
        )
        for atom in packet.atoms
    )
    atoms = _packet_refs_fn(packet)
    hyperedges = _packet_hyperedges_fn(packet)
    query = plan.query_program.query
    fingerprint = identity_sha256(
        {
            "format": "qwen_atom_feature_validated_inputs_v1",
            "packet_receipt": packet.receipt.identity_payload(),
            "closure_plan": plan.identity_payload(),
            "packet_context_sha256": quote_sha256(packet.context),
            "fusion_caps": caps.identity_payload(),
            "feature_caps": feature_caps.identity_payload(),
            "atoms": [item.identity_payload() for item in atoms],
            "hyperedges": [item.identity_payload() for item in hyperedges],
        }
    )
    return _ValidatedOperationInputs(
        fingerprint=fingerprint,
        packet_receipt_sha256=packet.receipt.receipt_sha256,
        closure_plan_sha256=plan.plan_sha256,
        query_program_sha256=plan.query_program.program_sha256,
        query_sha256=quote_sha256(query),
        closure_policy_sha256=plan.policy.policy_sha256,
        snapshot_sha256=plan.snapshot.snapshot_sha256,
        query=query,
        atom_values=atom_values,
        caps=detached_caps,
        feature_caps=detached_feature_caps,
        atoms=atoms,
        hyperedges=hyperedges,
    )


def _revalidate_operation_inputs(
    expected: _ValidatedOperationInputs,
    packet: EvidencePacket,
    plan: ClosurePlan,
    caps: FusionCaps,
    feature_caps: QwenAtomFeatureCaps,
    *,
    hidden_dim: int,
    _preflight_fn: Any = _preflight_packet,
    _validate_fn: Any = _validate_packet_plan,
    _capture_fn: Any = _capture_operation_inputs,
) -> None:
    _preflight_fn(
        packet,
        plan,
        caps,
        feature_caps,
        hidden_dim=hidden_dim,
    )
    _validate_fn(packet, plan)
    current = _capture_fn(packet, plan, caps, feature_caps)
    if current.fingerprint != expected.fingerprint:
        raise RuntimeError("Qwen feature operation inputs changed during execution")


def _batch_rows(
    rows: tuple[_QwenAtomTokenRow, ...],
    caps: QwenAtomFeatureCaps,
) -> tuple[tuple[int, int, int, int], ...]:
    batches: list[tuple[int, int, int, int]] = []
    start = 0
    while start < len(rows):
        stop = start
        padded_width = 0
        while stop < len(rows) and stop - start < caps.max_rows_per_forward:
            candidate_width = max(padded_width, len(rows[stop].token_ids))
            candidate_count = stop - start + 1
            if candidate_count * candidate_width > caps.max_workspace_tokens:
                break
            padded_width = candidate_width
            stop += 1
        if stop == start:
            raise MemoryError("one Qwen atom row cannot fit the padded workspace")
        count = stop - start
        batches.append((start, count, padded_width, count * padded_width))
        start = stop
    return tuple(batches)


_PINNED_OWNED_IMPLEMENTATION = _OwnedImplementation(
    preflight_packet=_preflight_packet,
    validate_packet_plan=_validate_packet_plan,
    capture_operation_inputs=_capture_operation_inputs,
    revalidate_operation_inputs=_revalidate_operation_inputs,
    build_atom_rows=_build_qwen_atom_rows,
    batch_rows=_batch_rows,
    execute_feature_batches=_execute_feature_batches,
    diagnostics_type=_FeatureExecutionDiagnostics,
    batch_receipt_type=QwenAtomBatchReceipt,
    operation_receipt_type=QwenAtomFeatureOperationReceipt,
)


def _pinned_owned_implementation(
    _value: _OwnedImplementation = _PINNED_OWNED_IMPLEMENTATION,
) -> _OwnedImplementation:
    return _value


class QwenAtomFeatureProvider:
    """Exact-owned resident provider with a receipt-only public smoke API.

    The supplied encoder remains externally aliased.  This provider validates
    its exact supported runtime boundary before and after every operation, but
    it does not attest exclusive ownership or loaded parameter bytes.  Callers
    must give it exclusive synchronous use of the encoder because legacy hook
    paths do not yet participate in the fusion-provider gate.
    """

    __slots__ = (
        "_encoder",
        "_output_layer",
        "_torch",
        "_expected_model_type",
        "_expected_tokenizer_type",
        "_expected_config_type",
        "_runtime_fingerprint",
        "_implementation_fingerprint",
        "_implementation",
        "_gate_factory",
        "_receipt",
        "_closed",
    )

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("QwenAtomFeatureProvider does not support subclassing")

    def __init__(self, encoder: Any, *, output_layer: int) -> None:
        _assert_owned_local_implementation()
        _assert_pinned_prefix_seams()
        pinned = _pinned_owned_implementation()
        if type(encoder) is not _PINNED_ENCODER_TYPE:
            raise TypeError("provider requires the exact owned Qwen3PrefixEncoder")
        if isinstance(output_layer, bool) or not isinstance(output_layer, int):
            raise ValueError("output_layer must be an exact integer")
        if not 0 <= output_layer < int(encoder.layers):
            raise ValueError("output_layer lies outside the retained Qwen prefix")
        checkpoint = getattr(encoder, "checkpoint_identity", None)
        if type(checkpoint) is not _PINNED_CHECKPOINT_IDENTITY_TYPE:
            raise TypeError("Qwen encoder requires an exact verified checkpoint identity")
        if (
            encoder.model_id != _PINNED_DEFAULT_MODEL_ID
            or encoder.model_revision != _PINNED_DEFAULT_MODEL_REVISION
        ):
            raise ValueError("resident Qwen provider requires the pinned model and revision")
        expected_checkpoint = _expected_checkpoint_sha256(encoder)
        if (
            checkpoint.model_id != encoder.model_id
            or checkpoint.model_revision != encoder.model_revision
            or checkpoint.checkpoint_sha256 != expected_checkpoint
            or encoder.checkpoint_sha256 != expected_checkpoint
        ):
            raise ValueError("Qwen checkpoint identity disagrees with the pinned manifest")
        if type(checkpoint.verified_files) is not tuple or not checkpoint.verified_files:
            raise ValueError("Qwen checkpoint identity requires verified files")
        if any(type(name) is not str or not name.strip() for name in checkpoint.verified_files):
            raise ValueError("Qwen verified file identities are malformed")
        if checkpoint.verified_files != _expected_verified_files(encoder):
            raise ValueError("Qwen verified files disagree with the pinned prefix manifest")

        torch, model_type, tokenizer_type, config_type = _expected_runtime_types()
        self._encoder = encoder
        self._output_layer = output_layer
        self._torch = torch
        self._expected_model_type = model_type
        self._expected_tokenizer_type = tokenizer_type
        self._expected_config_type = config_type
        runtime = _runtime_fingerprint(
            encoder,
            torch=torch,
            expected_encoder_type=_PINNED_ENCODER_TYPE,
            expected_model_type=model_type,
            expected_tokenizer_type=tokenizer_type,
            expected_config_type=config_type,
        )
        device = runtime.device
        execution_dtype = runtime.execution_dtype
        hidden_dim = runtime.hidden_dim
        if hidden_dim <= 0:
            raise RuntimeError("Qwen hidden width must be positive")
        implementation = _implementation_sha256(
            encoder,
            expected_model_type=model_type,
            expected_tokenizer_type=tokenizer_type,
        )
        implementation_fingerprint = _implementation_callable_fingerprint(
            encoder,
            expected_model_type=model_type,
            expected_tokenizer_type=tokenizer_type,
        )
        tokenizer = encoder.tokenizer
        tokenizer_identity = identity_sha256(
            {
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "tokenizer_type": (
                    f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
                ),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "vocab_size": int(len(tokenizer)),
            }
        )
        self._runtime_fingerprint = runtime.fingerprint
        self._implementation_fingerprint = implementation_fingerprint
        self._implementation = pinned
        self._gate_factory = _PINNED_GATE_FACTORY
        self._receipt = QwenAtomFeatureProviderReceipt(
            implementation_sha256=implementation,
            model_id=checkpoint.model_id,
            model_revision=checkpoint.model_revision,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            verified_files_sha256=identity_sha256(
                {"verified_files": list(checkpoint.verified_files)}
            ),
            tokenizer_identity_sha256=tokenizer_identity,
            retained_layers=int(encoder.layers),
            output_layer=output_layer,
            hidden_dim=hidden_dim,
            device=device,
            execution_dtype=execution_dtype,
            provider_id=_PROVIDER_ID,
            prompt_template_sha256=_PROMPT_TEMPLATE_SHA256,
        )
        self._closed = False

    def _assert_provider_state(self) -> None:
        if type(self) is not QwenAtomFeatureProvider:
            raise TypeError("provider must retain its exact owned type")
        if self._closed or self._encoder is None:
            raise RuntimeError("Qwen atom feature provider is closed")
        if type(self._receipt) is not QwenAtomFeatureProviderReceipt:
            raise TypeError("provider receipt lost its exact owned type")
        pinned = _pinned_owned_implementation()
        if self._implementation is not pinned:
            raise RuntimeError("provider lost its pinned owned implementation")
        self._receipt._seal()
        if self._output_layer != self._receipt.output_layer:
            raise RuntimeError("provider output layer changed after construction")
        if self._torch is not self._encoder._torch:
            raise RuntimeError("provider torch runtime changed after construction")
        if self._gate_factory is not _PINNED_GATE_FACTORY:
            raise RuntimeError("provider implementation changed after construction")

    def _assert_implementation(self) -> None:
        _assert_owned_local_implementation()
        _assert_pinned_prefix_seams()
        current = _implementation_callable_fingerprint(
            self._encoder,
            expected_model_type=self._expected_model_type,
            expected_tokenizer_type=self._expected_tokenizer_type,
        )
        if current != self._implementation_fingerprint:
            raise RuntimeError("Qwen feature implementation changed after construction")
        if self._receipt.implementation_sha256 != _implementation_sha256(
            self._encoder,
            expected_model_type=self._expected_model_type,
            expected_tokenizer_type=self._expected_tokenizer_type,
        ):
            raise RuntimeError("Qwen feature implementation digest changed")

    @property
    def receipt(self) -> QwenAtomFeatureProviderReceipt:
        if self._closed:
            raise RuntimeError("Qwen atom feature provider is closed")
        return self._receipt

    def _assert_runtime(self) -> None:
        self._assert_provider_state()
        current = _runtime_fingerprint(
            self._encoder,
            torch=self._torch,
            expected_encoder_type=_PINNED_ENCODER_TYPE,
            expected_model_type=self._expected_model_type,
            expected_tokenizer_type=self._expected_tokenizer_type,
            expected_config_type=self._expected_config_type,
        )
        if current.fingerprint != self._runtime_fingerprint:
            raise RuntimeError("owned Qwen runtime changed after provider construction")

    def run_execution_smoke(
        self,
        packet: EvidencePacket,
        plan: ClosurePlan,
        *,
        caps: FusionCaps,
        feature_caps: QwenAtomFeatureCaps,
    ) -> QwenAtomFeatureOperationReceipt:
        """Execute and discard resident features, returning only a smoke receipt."""

        return self._run_feature_execution_smoke(
            packet,
            plan,
            caps=caps,
            feature_caps=feature_caps,
        )

    def _run_feature_execution_smoke(
        self,
        packet: EvidencePacket,
        plan: ClosurePlan,
        *,
        caps: FusionCaps,
        feature_caps: QwenAtomFeatureCaps,
    ) -> QwenAtomFeatureOperationReceipt:
        """Private single-use workspace lifecycle for the public smoke."""

        self._assert_provider_state()
        encoder = self._encoder
        owned = self._implementation
        rows: tuple[_QwenAtomTokenRow, ...] = ()
        with self._gate_factory(encoder) as gate_token:
            self._assert_provider_state()
            owned.preflight_packet(
                packet,
                plan,
                caps,
                feature_caps,
                hidden_dim=self._receipt.hidden_dim,
            )
            owned.validate_packet_plan(packet, plan)
            inputs = owned.capture_operation_inputs(packet, plan, caps, feature_caps)
            self._assert_implementation()
            self._assert_runtime()
            try:
                rows = owned.build_atom_rows(
                    encoder.tokenizer,
                    inputs.atom_values,
                    inputs.query,
                    inputs.feature_caps,
                )
                batches = owned.batch_rows(rows, inputs.feature_caps)
                batch_receipts = tuple(
                    owned.batch_receipt_type(
                        batch_index=index,
                        start_row=start,
                        row_count=count,
                        padded_width=width,
                        padded_workspace_tokens=workspace,
                    )
                    for index, (start, count, width, workspace) in enumerate(batches)
                )
                execution = owned.execute_feature_batches(
                    encoder=encoder,
                    torch=self._torch,
                    output_layer=self._output_layer,
                    provider_receipt=self._receipt,
                    feature_caps=inputs.feature_caps,
                    rows=rows,
                    batches=batches,
                    gate_token=gate_token,
                )
                if type(execution) is not owned.diagnostics_type:
                    raise RuntimeError("owned Qwen executor omitted smoke diagnostics")
                row_receipts = tuple(row.receipt for row in rows)
            finally:
                # Drop all row token IDs before postchecking and sealing the
                # zero-retained request-state receipt.
                rows = ()
                owned.revalidate_operation_inputs(
                    inputs,
                    packet,
                    plan,
                    caps,
                    feature_caps,
                    hidden_dim=self._receipt.hidden_dim,
                )
                self._assert_implementation()
                self._assert_runtime()
            receipt = owned.operation_receipt_type(
                packet_receipt_sha256=inputs.packet_receipt_sha256,
                closure_plan_sha256=inputs.closure_plan_sha256,
                query_program_sha256=inputs.query_program_sha256,
                query_sha256=inputs.query_sha256,
                closure_policy_sha256=inputs.closure_policy_sha256,
                snapshot_sha256=inputs.snapshot_sha256,
                caps=inputs.caps,
                feature_caps=inputs.feature_caps,
                provider=self._receipt,
                atoms=inputs.atoms,
                hyperedges=inputs.hyperedges,
                rows=row_receipts,
                batches=batch_receipts,
                feature_shape=(len(inputs.atoms), self._receipt.hidden_dim),
                feature_device=self._receipt.device,
                feature_execution_dtype=self._receipt.execution_dtype,
                qwen_forward_count=(
                    execution.primary_forward_count
                    + execution.batch_invariance_forward_count
                ),
                primary_qwen_forward_count=execution.primary_forward_count,
                batch_invariance_forward_count=(
                    execution.batch_invariance_forward_count
                ),
                runtime_batch_invariance_attested=False,
                max_observed_padded_workspace_tokens=max(
                    batch.padded_workspace_tokens for batch in batch_receipts
                ),
            )
            return receipt

    def close(self) -> None:
        if self._closed:
            return
        self._assert_provider_state()
        encoder = self._encoder
        with self._gate_factory(encoder):
            self._assert_implementation()
            self._assert_runtime()
            self._encoder = None
            self._closed = True

    def __copy__(self) -> Any:
        raise TypeError("Qwen atom feature providers cannot be copied")

    def __deepcopy__(self, _memo: Any) -> Any:
        raise TypeError("Qwen atom feature providers cannot be deep-copied")

    def __reduce__(self) -> Any:
        raise TypeError("Qwen atom feature providers cannot be pickled")


_OWNED_LOCAL_IMPLEMENTATION_FINGERPRINT = _owned_local_fingerprint()


__all__ = ["QwenAtomFeatureProvider"]
