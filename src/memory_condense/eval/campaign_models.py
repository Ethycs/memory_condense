"""Immutable contracts for locked validation campaign assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

class CampaignMergeError(ValueError):
    """A shard cannot participate in a locked validation campaign."""


@dataclass(frozen=True, slots=True)
class ExpectedStressShard:
    sample_offset: int
    sample_id: str
    sample_sha256: str
    num_turns: int
    transcript_tokens: int
    questions: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LockedValidationPlan:
    dataset_path: Path
    split_manifest_path: Path
    policy_manifest_path: Path
    selection_artifact_path: Path
    dataset_sha256: str
    split_manifest_sha256: str
    policy_manifest_sha256: str
    implementation_sha256: str
    environment_lock_sha256: str
    selection_artifact_sha256: str
    retrieval: dict[str, Any]
    evaluation: dict[str, Any]
    sample_offsets: tuple[int, ...]
    shards: dict[int, ExpectedStressShard]
    question_ids: frozenset[str]
    claim_profile: str
    claim_profile_verified: bool
