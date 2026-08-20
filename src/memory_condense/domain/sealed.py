"""Reflective sealing for identity-hashed frozen dataclasses.

A "sealed" dataclass carries a derived ``*_sha256`` field that is either
empty (computed and bound on construction) or supplied (verified against the
freshly computed identity).  Canonical JSON sorts keys, so the payload's dict
order never affects the digest — only field content does.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, ClassVar, Mapping

from memory_condense.domain._discourse_identity import (
    _sha256,
    identity_sha256,
)


def identity_value(value: Any) -> Any:
    """Return one field value in its canonical identity-payload form."""

    payload = getattr(value, "identity_payload", None)
    if callable(payload):
        return payload()
    if isinstance(value, (list, tuple)):
        return [identity_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): identity_value(item) for key, item in value.items()}
    return value


def reflect_payload(obj: Any, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return every dataclass field (minus ``exclude``) in identity form."""

    return {
        item.name: identity_value(getattr(obj, item.name))
        for item in fields(obj)
        if item.name not in exclude
    }


class SealedIdentity:
    """Mixin: reflective ``identity_payload`` plus verify-or-bind sealing.

    Subclasses may narrow the payload with ``_PAYLOAD_EXCLUDE`` (derived
    fields that must not feed the digest) or override ``identity_payload``
    entirely when a field needs projection.  The payload produced here must
    stay value-equal to any hand-written payload it replaces — receipts are
    persisted, and a changed payload is a changed digest.
    """

    __slots__ = ()

    # Subclasses override these with PLAIN (un-annotated) assignments.  An
    # annotated ``ClassVar`` inside a dataclass body lands in the raw
    # ``__dataclass_fields__`` mapping as a pseudo-field, and external code
    # that iterates that mapping to rebuild identity payloads would pick the
    # configuration knobs up as data.
    _SEAL_FIELD: ClassVar[str] = "receipt_sha256"
    _SEAL_MISMATCH: ClassVar[str] = "receipt does not match its identity payload"
    _PAYLOAD_EXCLUDE: ClassVar[frozenset[str]] = frozenset()

    def identity_payload(
        self,
        *,
        include_receipt: bool = True,
        include_sha: bool | None = None,
    ) -> dict[str, Any]:
        """Return the canonical body, accepting the pre-mixin flag spelling."""

        if include_sha is not None:
            if include_receipt is not True and include_receipt != include_sha:
                raise TypeError("include_receipt and include_sha disagree")
            include_receipt = include_sha
        payload = reflect_payload(self, self._PAYLOAD_EXCLUDE | {self._SEAL_FIELD})
        if include_receipt:
            payload[self._SEAL_FIELD] = getattr(self, self._SEAL_FIELD)
        return payload

    def _seal(self) -> None:
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        current = getattr(self, self._SEAL_FIELD)
        if current:
            if _sha256(current, self._SEAL_FIELD) != expected:
                raise ValueError(self._SEAL_MISMATCH)
        else:
            object.__setattr__(self, self._SEAL_FIELD, expected)


__all__ = ["SealedIdentity", "identity_value", "reflect_payload"]
