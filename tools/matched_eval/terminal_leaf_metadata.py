"""Authenticate selected-leaf labels for terminal provider context.

This adapter is intentionally narrower than a linker.  It exposes metadata
that is already committed by an A1 :class:`SelectedHLeaf`; it does not infer
entities, manufacture source locators, traverse an association graph, or give
topic labels exclusion authority.  The resulting receipt binds every exposed
field to the exact selected-leaf and source receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

from .after_union_fact_closure import SELECTED_LEAF_FORMAT, SelectedHLeaf
from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)


FORMAT = "memory-condense-a1-terminal-leaf-provider-metadata-v1"
METADATA_AUTHORITY = "authenticated_selected_leaf_labels_context_only"

_TOPIC_FIELDS = frozenset({"entity", "kind", "status"})
_BOUNDARY_FIELDS = frozenset({"date", "group", "relation"})


class TerminalLeafMetadataError(MatchedEvalContractError):
    """An authenticated selected leaf or its label grammar changed."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise TerminalLeafMetadataError(message)


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    return value  # type: ignore[return-value]


def _normalized_label_value(value: str, label: str) -> str:
    require_text(value, label)
    parts = value.replace("_", "-").split("-")
    _require(
        value.casefold() == value
        and all(part and part.isalnum() for part in parts),
        f"{label} must be a normalized lowercase label",
    )
    return value


def _parse_labels(
    labels: Sequence[str],
    *,
    allowed: frozenset[str],
    label: str,
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in labels:
        value = require_text(raw, label)
        field, separator, payload = value.partition(":")
        _require(separator == ":" and bool(payload), f"{label} grammar changed")
        _require(field in allowed, f"{label} has an unsupported field: {field}")
        _require(field not in parsed, f"{label} repeats field: {field}")
        parsed[field] = payload
    return parsed


def authenticate_selected_leaf_projection(raw: object) -> SelectedHLeaf:
    """Reconstruct and authenticate one exact persisted A1 leaf projection."""

    _require(type(raw) is dict, "selected leaf must be an exact object")
    row: dict[str, Any] = raw  # type: ignore[assignment]
    expected_keys = {
        "boundary_labels",
        "cross_boundary_edge_ids",
        "format",
        "group_handle",
        "handle_id",
        "receipt_sha256",
        "source_receipt_sha256",
        "text",
        "token_count",
        "topic_labels",
    }
    _require(
        set(row) == expected_keys and row.get("format") == SELECTED_LEAF_FORMAT,
        "selected leaf projection schema changed",
    )
    leaf = SelectedHLeaf(
        require_text(row.get("handle_id"), "selected leaf handle"),
        require_text(row.get("group_handle"), "selected leaf group"),
        require_text(row.get("text"), "selected leaf text"),
        require_sha256(row.get("source_receipt_sha256"), "selected leaf source"),
        tuple(
            require_text(value, "selected leaf topic label")
            for value in _exact_list(row.get("topic_labels"), "topic labels")
        ),
        tuple(
            require_text(value, "selected leaf boundary label")
            for value in _exact_list(row.get("boundary_labels"), "boundary labels")
        ),
        tuple(
            require_text(value, "selected leaf edge ID")
            for value in _exact_list(
                row.get("cross_boundary_edge_ids"), "cross-boundary edge IDs"
            )
        ),
        require_sha256(row.get("receipt_sha256"), "selected leaf receipt"),
    )
    _require(
        row == leaf.projection(),
        "selected leaf differs from its authenticated projection",
    )
    return leaf


@dataclass(frozen=True, slots=True)
class AuthenticatedTerminalLeafMetadata:
    """Provider-safe metadata bound to one exact selected A1 leaf."""

    handle_id: str
    group_handle: str
    leaf_receipt_sha256: str
    source_receipt_sha256: str
    event_date: str | None = None
    source_relation: str | None = None
    kind: str | None = None
    status: str | None = None
    entity_label: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_text(self.handle_id, "terminal metadata handle")
        require_text(self.group_handle, "terminal metadata group")
        require_sha256(self.leaf_receipt_sha256, "terminal metadata leaf")
        require_sha256(self.source_receipt_sha256, "terminal metadata source")
        if self.event_date is not None:
            require_text(self.event_date, "terminal metadata event date")
            try:
                parsed = date.fromisoformat(self.event_date)
            except ValueError as exc:
                raise TerminalLeafMetadataError(
                    "terminal metadata event date must be an ISO calendar date"
                ) from exc
            _require(
                parsed.isoformat() == self.event_date,
                "terminal metadata event date must be canonical ISO",
            )
        for field in ("source_relation", "kind", "status", "entity_label"):
            value = getattr(self, field)
            if value is not None:
                _normalized_label_value(value, f"terminal metadata {field}")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(
                require_sha256(
                    self.receipt_sha256, "terminal metadata receipt"
                )
                == expected,
                "terminal metadata receipt changed",
            )
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="terminal_leaf_metadata")

    def _metadata_fields(self) -> dict[str, str]:
        values = {
            "entity_label": self.entity_label,
            "event_date": self.event_date,
            "kind": self.kind,
            "source_relation": self.source_relation,
            "status": self.status,
        }
        return {key: value for key, value in values.items() if value is not None}

    def provider_projection(self) -> dict[str, object]:
        """Return the locator-free row that a later terminal arm may expose."""

        value: dict[str, object] = {
            "format": FORMAT,
            "group_handle": self.group_handle,
            "handle_id": self.handle_id,
            "leaf_receipt_sha256": self.leaf_receipt_sha256,
            "metadata_authority": METADATA_AUTHORITY,
            "metadata_receipt_sha256": self.receipt_sha256,
            **self._metadata_fields(),
        }
        assert_gold_blind(value, path="terminal_leaf_metadata.provider")
        return value

    def projection(self, *, include_receipt: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "format": FORMAT,
            "group_handle": self.group_handle,
            "handle_id": self.handle_id,
            "leaf_receipt_sha256": self.leaf_receipt_sha256,
            "metadata_authority": METADATA_AUTHORITY,
            "retained_transformer_token_state_bytes": 0,
            "source_receipt_sha256": self.source_receipt_sha256,
            **self._metadata_fields(),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def compile_terminal_leaf_metadata(
    leaf: SelectedHLeaf,
) -> AuthenticatedTerminalLeafMetadata:
    """Parse only the closed A1 label grammar and bind it to ``leaf``."""

    _require(type(leaf) is SelectedHLeaf, "terminal metadata requires SelectedHLeaf")
    # Replaying the projection revalidates the receipt rather than trusting an
    # object passed through an untyped caller boundary.
    authenticated = authenticate_selected_leaf_projection(leaf.projection())
    topics = _parse_labels(
        authenticated.topic_labels,
        allowed=_TOPIC_FIELDS,
        label="selected leaf topic label",
    )
    boundaries = _parse_labels(
        authenticated.boundary_labels,
        allowed=_BOUNDARY_FIELDS,
        label="selected leaf boundary label",
    )
    _require(
        boundaries.get("group") == authenticated.group_handle,
        "selected leaf boundary group is missing or foreign",
    )
    event_date = boundaries.get("date")
    if event_date is not None:
        try:
            parsed_date = date.fromisoformat(event_date)
        except ValueError as exc:
            raise TerminalLeafMetadataError(
                "selected leaf boundary date must be an ISO calendar date"
            ) from exc
        _require(
            parsed_date.isoformat() == event_date,
            "selected leaf boundary date must be canonical ISO",
        )
    for field, value in (*topics.items(), *boundaries.items()):
        if field not in {"date", "group"}:
            _normalized_label_value(value, f"selected leaf {field}")
    return AuthenticatedTerminalLeafMetadata(
        handle_id=authenticated.handle_id,
        group_handle=authenticated.group_handle,
        leaf_receipt_sha256=authenticated.receipt_sha256,
        source_receipt_sha256=authenticated.source_receipt_sha256,
        event_date=event_date,
        source_relation=boundaries.get("relation"),
        kind=topics.get("kind"),
        status=topics.get("status"),
        entity_label=topics.get("entity"),
    )


def compile_terminal_leaf_metadata_population(
    leaves: Sequence[SelectedHLeaf],
) -> tuple[AuthenticatedTerminalLeafMetadata, ...]:
    """Compile an ordered, non-repeating selected-leaf population."""

    exact = tuple(leaves)
    _require(bool(exact), "terminal metadata population must be non-empty")
    compiled = tuple(compile_terminal_leaf_metadata(leaf) for leaf in exact)
    handles = tuple(row.handle_id for row in compiled)
    _require(
        len(handles) == len(set(handles)),
        "terminal metadata population repeats a selected handle",
    )
    return compiled


__all__ = [
    "AuthenticatedTerminalLeafMetadata",
    "FORMAT",
    "METADATA_AUTHORITY",
    "TerminalLeafMetadataError",
    "authenticate_selected_leaf_projection",
    "compile_terminal_leaf_metadata",
    "compile_terminal_leaf_metadata_population",
]
