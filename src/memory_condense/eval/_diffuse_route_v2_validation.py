"""Strict, provider-free validation helpers for diffuse route-v2 receipts."""

from __future__ import annotations

import hashlib
import marshal
from types import FunctionType
from typing import Any

from memory_condense.domain.discourse import (
    DiscourseArtifact,
    DiscourseSnapshot,
    EpisodeSeed,
    EvidenceAtom,
    EvidenceBundle,
    EvidenceSpan,
    identity_sha256,
)
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval.diffuse_compilation import (
    DiffuseCompilationReceipt,
    DiffuseSourceCompilationReceipt,
)
from memory_condense.eval.diffuse_longmemeval_inputs import (
    LegacyDiffuseCandidates,
    LegacyDiffuseInputReceipt,
    _receipt_bindings,
)
from memory_condense.eval.reproducibility import implementation_sha256
from memory_condense.search.episodes import (
    EpisodeSourceCandidate,
    EpisodeSourceCandidateScope,
)
from memory_condense.search.packing.evidence_packet import (
    _proved_obligation_ids,
    _required_proof_ids,
)


_ROUTE_V2_IMPLEMENTATION_FORMAT = (
    "memory-condense-longmemeval-episode-primary-route-implementation-v2"
)


def freeze_loaded_callable(value: Any, label: str) -> Any:
    """Copy one loaded Python function with a snapshot of its global bindings."""

    if type(value) is not FunctionType:
        raise TypeError(f"{label} must be an exact Python function")
    frozen = FunctionType(
        value.__code__,
        dict(value.__globals__),
        value.__name__,
        value.__defaults__,
        value.__closure__,
    )
    frozen.__kwdefaults__ = (
        None if value.__kwdefaults__ is None else dict(value.__kwdefaults__)
    )
    frozen.__annotations__ = dict(value.__annotations__)
    frozen.__dict__.update(value.__dict__)
    frozen.__doc__ = value.__doc__
    frozen.__module__ = value.__module__
    frozen.__qualname__ = value.__qualname__
    return frozen


def _runtime_value_projection(value: Any) -> object:
    """Produce an in-process comparison payload for callable defaults."""

    if type(value) in (str, int, float, bool, type(None)):
        return {"type": type(value).__name__, "value": value}
    if type(value) is bytes:
        return {"type": "bytes", "hex": value.hex()}
    if type(value) in (tuple, list):
        return {
            "type": type(value).__name__,
            "items": [_runtime_value_projection(item) for item in value],
        }
    if type(value) is dict:
        return {
            "type": "dict",
            "items": [
                [
                    _runtime_value_projection(key),
                    _runtime_value_projection(item),
                ]
                for key, item in value.items()
            ],
        }
    if type(value) is FunctionType:
        return {
            "type": "function",
            "module": value.__module__,
            "qualname": value.__qualname__,
            "code_sha256": hashlib.sha256(
                marshal.dumps(value.__code__)
            ).hexdigest(),
        }
    if isinstance(value, type):
        return {
            "type": "class",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    return {
        "type": "object",
        "class_module": type(value).__module__,
        "class_qualname": type(value).__qualname__,
        "loaded_identity": id(value),
    }


def loaded_callable_fingerprint(value: Any, label: str) -> str:
    """Bind a loaded function's code and mutable call-default surfaces."""

    if type(value) is not FunctionType:
        raise TypeError(f"{label} must be an exact Python function")
    closure = () if value.__closure__ is None else tuple(
        _runtime_value_projection(cell.cell_contents) for cell in value.__closure__
    )
    return identity_sha256(
        {
            "module": value.__module__,
            "qualname": value.__qualname__,
            "code_sha256": hashlib.sha256(
                marshal.dumps(value.__code__)
            ).hexdigest(),
            "defaults": _runtime_value_projection(value.__defaults__),
            "kwdefaults": _runtime_value_projection(value.__kwdefaults__),
            "closure": list(closure),
        }
    )


def bind_loaded_dependency_guard(
    bindings: tuple[tuple[str, Any, Any], ...],
    *,
    immediate_labels: tuple[str, ...],
) -> tuple[Any, Any]:
    """Bind identity and loaded-code guards for owned execution seams."""

    labels = tuple(label for label, _observer, _expected in bindings)
    if len(set(labels)) != len(labels):
        raise ValueError("loaded dependency labels must be unique")
    if not set(immediate_labels) <= set(labels):
        raise ValueError("immediate dependency labels must be bound")
    fingerprints = {
        label: loaded_callable_fingerprint(expected, label)
        for label, _observer, expected in bindings
    }

    def reject() -> None:
        raise RuntimeError(
            "owned episode-primary retrieval implementation was replaced"
        )

    def assert_subset(active_labels: tuple[str, ...]) -> None:
        active = set(active_labels)
        for label, observer, expected in bindings:
            if label not in active:
                continue
            if observer() is not expected or (
                loaded_callable_fingerprint(expected, label)
                != fingerprints[label]
            ):
                reject()

    def assert_all() -> None:
        assert_subset(labels)

    def assert_immediate() -> None:
        assert_subset(immediate_labels)

    return assert_all, assert_immediate


def bind_route_v2_dependency_guard(
    *,
    analysis_module: Any,
    diffuse_module: Any,
    route_globals: dict[str, Any],
    analysis_retriever: Any,
    packet_retriever: Any,
    certifier: Any,
    implementation_observer: Any,
    expected_receipt: Any,
) -> tuple[Any, Any]:
    """Bind every public/module route seam plus the immediate packet seams."""

    bindings = (
        (
            "analysis core",
            lambda: analysis_module._retrieve_diffuse_longmemeval_sample_with_route,
            analysis_retriever,
        ),
        (
            "analysis packet",
            lambda: analysis_module.retrieve_longmemeval_diffuse_packet,
            packet_retriever,
        ),
        (
            "diffuse packet",
            lambda: diffuse_module.retrieve_longmemeval_diffuse_packet,
            packet_retriever,
        ),
        (
            "certifier",
            lambda: route_globals.get("certify_episode_primary_analysis_phase_v2"),
            certifier,
        ),
        (
            "implementation observer",
            lambda: route_globals.get("route_v2_implementation_sha256"),
            implementation_observer,
        ),
        (
            "expected receipt builder",
            lambda: route_globals.get("_expected_route_receipt"),
            expected_receipt,
        ),
    )
    return bind_loaded_dependency_guard(
        bindings,
        immediate_labels=("analysis packet", "diffuse packet"),
    )


def _bind_implementation_observer(
    package_observer: Any,
    identity_hasher: Any,
    implementation_format: str,
) -> Any:
    """Freeze the package-wide source digest behind a zero-argument API."""

    implementation_digest = identity_hasher(
        {
            "format": implementation_format,
            "package_implementation_sha256": package_observer(),
        }
    )

    def route_v2_implementation_sha256() -> str:
        """Return the import-time full-package implementation identity."""

        return implementation_digest

    return route_v2_implementation_sha256


route_v2_implementation_sha256 = freeze_loaded_callable(
    _bind_implementation_observer(
        implementation_sha256,
        identity_sha256,
        _ROUTE_V2_IMPLEMENTATION_FORMAT,
    ),
    "route-v2 implementation observer",
)
del _bind_implementation_observer


def require_exact(value: Any, expected: type, label: str) -> Any:
    if type(value) is not expected:
        raise TypeError(f"{label} must be exact {expected.__name__}")
    return value


def require_exact_tuple(
    values: Any,
    expected: type,
    label: str,
) -> tuple[Any, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    for value in values:
        require_exact(value, expected, label)
    return values


def require_exact_scalar(value: Any, expected: type, label: str) -> Any:
    if type(value) is not expected:
        raise TypeError(f"{label} must be exact {expected.__name__}")
    return value


def require_optional_exact_scalar(
    value: Any,
    expected: type,
    label: str,
) -> Any:
    if value is not None:
        require_exact_scalar(value, expected, label)
    return value


def assert_exact_value(actual: Any, expected: Any, label: str) -> None:
    """Compare JSON-shaped values without Python's bool/int coercive equality."""

    if type(actual) is not type(expected):
        raise TypeError(
            f"{label} changed scalar or container type "
            f"({type(actual).__name__} != {type(expected).__name__})"
        )
    if type(expected) is dict:
        if any(type(key) is not str for key in (*actual.keys(), *expected.keys())):
            raise TypeError(f"{label} keys must be exact strings")
        if set(actual) != set(expected):
            raise ValueError(f"{label} changed its exact key set")
        for key in expected:
            assert_exact_value(actual[key], expected[key], f"{label}.{key}")
        return
    if type(expected) in (list, tuple):
        if len(actual) != len(expected):
            raise ValueError(f"{label} changed its exact sequence length")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_exact_value(actual_item, expected_item, f"{label}[{index}]")
        return
    if type(expected) not in (str, int, float, bool, type(None)):
        raise TypeError(f"{label} contains a non-canonical value")
    if actual != expected:
        raise ValueError(f"{label} changed its exact value")


def assert_same_identity(actual: Any, expected: Any, label: str) -> None:
    """Require exact class, scalar schema, and canonical identity payload."""

    require_exact(actual, type(expected), label)
    actual_payload = actual.identity_payload()
    expected_payload = expected.identity_payload()
    assert_exact_value(actual_payload, expected_payload, f"{label} identity")
    if identity_sha256(actual_payload) != identity_sha256(expected_payload):
        raise ValueError(f"{label} changed its canonical identity")


def assert_same_identity_sequence(
    actual: Any,
    expected: Any,
    item_type: type,
    label: str,
) -> None:
    require_exact_tuple(actual, item_type, label)
    require_exact_tuple(expected, item_type, f"expected {label}")
    if len(actual) != len(expected):
        raise ValueError(f"{label} changed its exact sequence length")
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        assert_same_identity(actual_item, expected_item, f"{label}[{index}]")


def require_exact_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    return require_exact_tuple(value, str, label)


def assert_current_identity(value: Any, field: str, label: str) -> None:
    payload = getattr(value, "identity_payload", None)
    if not callable(payload):
        raise TypeError(f"{label} lacks an identity payload")
    expected = identity_sha256(payload(include_receipt=False))
    if getattr(value, field, None) != expected:
        raise ValueError(f"{label} identity changed after construction")


def validate_compilation_receipt(value: Any) -> DiffuseCompilationReceipt:
    compilation = require_exact(value, DiffuseCompilationReceipt, "compilation")
    require_exact(compilation.artifact, DiscourseArtifact, "compiled artifact")
    require_exact(
        compilation.final_snapshot,
        DiscourseSnapshot,
        "compilation snapshot",
    )
    sources = require_exact_tuple(
        compilation.source_receipts,
        DiffuseSourceCompilationReceipt,
        "source compilation receipts",
    )
    assert_current_identity(compilation, "receipt_sha256", "compilation receipt")
    assert_current_identity(
        compilation.final_snapshot,
        "snapshot_sha256",
        "compilation snapshot",
    )
    for source in sources:
        assert_current_identity(
            source,
            "receipt_sha256",
            "source compilation receipt",
        )
    return compilation


def validate_evidence_span(span: Any, label: str) -> EvidenceSpan:
    value = require_exact(span, EvidenceSpan, label)
    require_exact_scalar(value.chunk_id, str, f"{label}.chunk_id")
    for name in ("start_char", "end_char", "ordinal", "turn_start_char"):
        require_exact_scalar(getattr(value, name), int, f"{label}.{name}")
    require_exact_scalar(value.quote_sha256, str, f"{label}.quote_sha256")
    for name in ("source_id", "turn_id", "role", "created_at"):
        require_optional_exact_scalar(getattr(value, name), str, f"{label}.{name}")
    return value


def validate_evidence_atom(atom: Any, label: str) -> EvidenceAtom:
    value = require_exact(atom, EvidenceAtom, label)
    require_exact_scalar(value.atom_id, str, f"{label}.atom_id")
    validate_evidence_span(value.span, f"{label}.span")
    require_exact_scalar(value.text, str, f"{label}.text")
    require_exact_scalar(value.label, str, f"{label}.label")
    require_optional_exact_scalar(value.role, str, f"{label}.role")
    require_optional_exact_scalar(value.created_at, str, f"{label}.created_at")
    return value


def validate_evidence_bundle(bundle: Any, label: str) -> EvidenceBundle:
    value = require_exact(bundle, EvidenceBundle, label)
    require_exact_scalar(value.bundle_id, str, f"{label}.bundle_id")
    for name in ("atom_ids", "obligation_ids", "unit_ids", "relation_ids"):
        require_exact_string_tuple(getattr(value, name), f"{label}.{name}")
    require_exact_scalar(value.required, bool, f"{label}.required")
    require_exact_scalar(value.utility, float, f"{label}.utility")
    return value


def validate_episode_seed(seed: Any, label: str) -> EpisodeSeed:
    value = require_exact(seed, EpisodeSeed, label)
    for name in ("episode_id", "anchor_chunk_id", "route"):
        require_exact_scalar(getattr(value, name), str, f"{label}.{name}")
    require_exact_scalar(value.score, float, f"{label}.score")
    require_exact_string_tuple(value.path, f"{label}.path")
    return value


def episode_seed_payload(seed: EpisodeSeed) -> dict[str, object]:
    validate_episode_seed(seed, "episode seed")
    return {
        "episode_id": seed.episode_id,
        "anchor_chunk_id": seed.anchor_chunk_id,
        "score": seed.score,
        "route": seed.route,
        "path": list(seed.path),
    }


episode_seed_payload = freeze_loaded_callable(
    episode_seed_payload,
    "episode seed payload builder",
)


def seed_projection_sha256(seeds: tuple[EpisodeSeed, ...]) -> str:
    return identity_sha256([episode_seed_payload(seed) for seed in seeds])


seed_projection_sha256 = freeze_loaded_callable(
    seed_projection_sha256,
    "episode seed projection hasher",
)


def validate_source_candidate(
    candidate: Any,
    label: str,
) -> EpisodeSourceCandidate:
    value = require_exact(candidate, EpisodeSourceCandidate, label)
    require_exact_scalar(value.source_id, str, f"{label}.source_id")
    require_exact_scalar(value.score, float, f"{label}.score")
    require_exact_scalar(value.route, str, f"{label}.route")
    return value


def validate_legacy_source_scope(
    candidates: LegacyDiffuseCandidates,
    receipt: LegacyDiffuseInputReceipt,
) -> EpisodeSourceCandidateScope:
    """Rebind exact legacy candidates to both their receipt and source scope."""

    require_exact_tuple(candidates.anchors, RetrievalResult, "legacy anchors")
    require_exact_tuple(
        candidates.source_candidates,
        EpisodeSourceCandidate,
        "source candidates",
    )
    for index, candidate in enumerate(candidates.source_candidates):
        validate_source_candidate(candidate, f"source candidates[{index}]")
    for name, binding in _receipt_bindings(candidates).items():
        assert_exact_value(
            getattr(receipt, name),
            binding,
            f"legacy receipt.{name}",
        )
    scope = candidates.source_candidate_scope
    if scope is None:
        raise ValueError("episode-primary query lacks a source candidate scope")
    require_exact(scope, EpisodeSourceCandidateScope, "source candidate scope")
    assert_current_identity(scope, "receipt_sha256", "source candidate scope")
    require_exact_tuple(
        scope.candidates,
        EpisodeSourceCandidate,
        "scoped source candidates",
    )
    require_exact_scalar(scope.artifact_id, str, "source scope.artifact_id")
    require_exact_scalar(scope.snapshot_sha256, str, "source scope.snapshot_sha256")
    require_exact_scalar(scope.source_revision, int, "source scope.source_revision")
    require_exact_scalar(
        scope.source_content_sha256,
        str,
        "source scope.source_content_sha256",
    )
    require_exact_scalar(scope.query_sha256, str, "source scope.query_sha256")
    require_exact_scalar(
        scope.router_policy_sha256,
        str,
        "source scope.router_policy_sha256",
    )
    require_exact_string_tuple(
        scope.universe_source_ids,
        "source scope.universe_source_ids",
    )
    require_exact_string_tuple(
        scope.truncated_source_ids,
        "source scope.truncated_source_ids",
    )
    require_exact_scalar(
        scope.universe_enumerated,
        bool,
        "source scope.universe_enumerated",
    )
    for index, candidate in enumerate(scope.candidates):
        validate_source_candidate(candidate, f"scoped source candidates[{index}]")
    assert_same_identity_sequence(
        candidates.source_candidates,
        scope.candidates,
        EpisodeSourceCandidate,
        "source candidate sequence",
    )
    if not scope.candidates:
        raise ValueError("episode-primary source scope has no candidates")
    return scope


def _bind_closure_stopping_validator(
    required_proof_ids: Any,
    proved_obligation_ids: Any,
) -> Any:
    """Capture the exact packer proof helpers behind receipt validation."""

    def validate_closure_stopping_state(
        diffuse_receipt: Any,
        plan: Any,
        packet_receipt: Any,
    ) -> None:
        """Join plan state and derive the packet outcome from selected proof."""

        require_exact_scalar(plan.stopping_reason, str, "plan stopping_reason")
        require_exact_scalar(
            diffuse_receipt.closure_stopping_reason,
            str,
            "diffuse receipt closure_stopping_reason",
        )
        require_exact_scalar(
            packet_receipt.stopping_reason,
            str,
            "packet receipt stopping_reason",
        )
        assert_exact_value(
            diffuse_receipt.closure_stopping_reason,
            plan.stopping_reason,
            "diffuse closure stopping reason",
        )
        require_exact_scalar(plan.complete_claimed, bool, "plan complete_claimed")
        require_exact_scalar(
            diffuse_receipt.closure_complete_claimed,
            bool,
            "diffuse receipt closure_complete_claimed",
        )
        require_exact_scalar(
            packet_receipt.complete_claimed,
            bool,
            "packet receipt complete_claimed",
        )
        assert_exact_value(
            diffuse_receipt.closure_complete_claimed,
            plan.complete_claimed,
            "diffuse closure completion claim",
        )

        selected_bundle_ids = require_exact_string_tuple(
            packet_receipt.selected_bundle_ids,
            "packet receipt selected_bundle_ids",
        )
        bundle_by_id = {bundle.bundle_id: bundle for bundle in plan.bundles}
        required = required_proof_ids(plan.query_program)
        proved = proved_obligation_ids(
            selected_bundle_ids,
            plan=plan,
            bundle_by_id=bundle_by_id,
        )
        required_selected = required <= proved
        expected_complete = plan.complete_claimed and required_selected
        expected_reason = (
            "budget_impossible"
            if not required_selected
            else "complete"
            if plan.stopping_reason == "complete"
            else plan.stopping_reason
        )
        assert_exact_value(
            packet_receipt.complete_claimed,
            expected_complete,
            "packet closure completion claim",
        )
        assert_exact_value(
            packet_receipt.stopping_reason,
            expected_reason,
            "packet closure stopping reason",
        )

    return validate_closure_stopping_state


validate_closure_stopping_state = freeze_loaded_callable(
    _bind_closure_stopping_validator(
        _required_proof_ids,
        _proved_obligation_ids,
    ),
    "closure stopping-state validator",
)
del _bind_closure_stopping_validator


__all__ = [
    "assert_current_identity",
    "assert_exact_value",
    "assert_same_identity",
    "assert_same_identity_sequence",
    "bind_loaded_dependency_guard",
    "bind_route_v2_dependency_guard",
    "episode_seed_payload",
    "freeze_loaded_callable",
    "loaded_callable_fingerprint",
    "require_exact",
    "require_exact_scalar",
    "require_exact_string_tuple",
    "require_exact_tuple",
    "route_v2_implementation_sha256",
    "seed_projection_sha256",
    "validate_evidence_atom",
    "validate_evidence_bundle",
    "validate_compilation_receipt",
    "validate_closure_stopping_state",
    "validate_episode_seed",
    "validate_legacy_source_scope",
    "validate_source_candidate",
]
