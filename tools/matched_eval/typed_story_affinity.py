"""Opaque story-affinity keys derived from already selected provenance.

The locked 1M evaluation concatenates unrelated histories into one physical
search namespace.  A selected exact citation therefore identifies both a
source stream and, when the durable source ID carries the namespace's declared
partition separator, the larger history component containing that stream.

This module never selects a component from a question ID and never exposes a
raw source or partition locator to a provider.  It only turns provenance that
has *already survived mechanism selection* into content-addressed local keys.
Those keys can then connect opaque evidence groups in the common link plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .contracts import identity_sha256, require_sha256, require_text


FORMAT = "memory-condense-evidence-derived-story-affinity-v1"
SOURCE_KEY_FORMAT = f"{FORMAT}-source-key"
HISTORY_KEY_FORMAT = f"{FORMAT}-history-key"


def evidence_source_story_key_sha256(namespace_id: str, source_id: str, /) -> str:
    """Return the canonical opaque key for one exact selected source."""

    require_sha256(namespace_id, "story-affinity namespace")
    require_text(source_id, "story-affinity source")
    return identity_sha256(
        {
            "affinity_scope": "exact_source_stream",
            "format": SOURCE_KEY_FORMAT,
            "namespace_id": namespace_id,
            "source_id": source_id,
        }
    )


def evidence_history_story_key_sha256(
    namespace_id: str,
    history_component: str,
    /,
    *,
    partition_separator: str = "::",
) -> str:
    """Return the canonical opaque key for an exact selected component."""

    require_sha256(namespace_id, "story-affinity namespace")
    require_text(history_component, "story-affinity history component")
    require_text(partition_separator, "story-affinity separator")
    return identity_sha256(
        {
            "affinity_scope": "evidence_derived_history_component",
            "format": HISTORY_KEY_FORMAT,
            "history_component": history_component,
            "namespace_id": namespace_id,
            "partition_separator": partition_separator,
        }
    )


@dataclass(frozen=True, slots=True)
class EvidenceDerivedStoryAffinity:
    """Local-only proof for source and enclosing-history story keys."""

    namespace_id: str
    source_id: str
    partition_separator: str
    source_story_key_sha256: str
    history_story_key_sha256: str
    history_key_distinct_from_source: bool
    derivation: Literal["post_selection_exact_provenance"] = (
        "post_selection_exact_provenance"
    )
    provider_visible_raw_locator_count: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.namespace_id, "story-affinity namespace")
        require_text(self.source_id, "story-affinity source")
        require_text(self.partition_separator, "story-affinity separator")
        require_sha256(self.source_story_key_sha256, "source story key")
        require_sha256(self.history_story_key_sha256, "history story key")
        if type(self.history_key_distinct_from_source) is not bool:
            raise TypeError("history/source key distinction must be exact")
        expected_distinct = self.partition_separator in self.source_id
        if self.history_key_distinct_from_source is not expected_distinct:
            raise ValueError("history/source key distinction changed")
        if (
            self.derivation != "post_selection_exact_provenance"
            or self.provider_visible_raw_locator_count != 0
            or self.retained_transformer_token_state_bytes != 0
        ):
            raise ValueError("story affinity escaped its local zero-state boundary")
        expected = identity_sha256(self.local_projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("story-affinity receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def story_keys(self) -> tuple[str, ...]:
        """Ordered unique keys safe to use only as opaque link identities."""

        return tuple(
            dict.fromkeys(
                (self.source_story_key_sha256, self.history_story_key_sha256)
            )
        )

    def local_projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        """Prompt-external audit projection; raw provenance intentionally local."""

        value: dict[str, Any] = {
            "derivation": self.derivation,
            "format": FORMAT,
            "history_key_distinct_from_source": (
                self.history_key_distinct_from_source
            ),
            "history_story_key_sha256": self.history_story_key_sha256,
            "namespace_id": self.namespace_id,
            "partition_separator": self.partition_separator,
            "provider_visible_raw_locator_count": 0,
            "retained_transformer_token_state_bytes": 0,
            "source_id": self.source_id,
            "source_story_key_sha256": self.source_story_key_sha256,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def opaque_projection(self) -> dict[str, Any]:
        """Provider-safe receipt surface with no raw source/component values."""

        return {
            "derivation": self.derivation,
            "format": FORMAT,
            "history_key_distinct_from_source": (
                self.history_key_distinct_from_source
            ),
            "provider_visible_raw_locator_count": 0,
            "receipt_sha256": self.receipt_sha256,
            "retained_transformer_token_state_bytes": 0,
            "story_key_count": len(self.story_keys),
        }


def derive_evidence_story_affinity(
    namespace_id: str,
    source_id: str,
    /,
    *,
    partition_separator: str = "::",
) -> EvidenceDerivedStoryAffinity:
    """Derive source and enclosing-history keys from selected exact lineage."""

    require_sha256(namespace_id, "story-affinity namespace")
    require_text(source_id, "story-affinity source")
    require_text(partition_separator, "story-affinity separator")
    source_key = evidence_source_story_key_sha256(namespace_id, source_id)
    has_component = partition_separator in source_id
    component = source_id.split(partition_separator, 1)[0]
    history_key = (
        evidence_history_story_key_sha256(
            namespace_id,
            component,
            partition_separator=partition_separator,
        )
        if has_component
        else source_key
    )
    return EvidenceDerivedStoryAffinity(
        namespace_id=namespace_id,
        source_id=source_id,
        partition_separator=partition_separator,
        source_story_key_sha256=source_key,
        history_story_key_sha256=history_key,
        history_key_distinct_from_source=has_component,
    )


__all__ = [
    "FORMAT",
    "HISTORY_KEY_FORMAT",
    "SOURCE_KEY_FORMAT",
    "EvidenceDerivedStoryAffinity",
    "derive_evidence_story_affinity",
    "evidence_history_story_key_sha256",
    "evidence_source_story_key_sha256",
]
