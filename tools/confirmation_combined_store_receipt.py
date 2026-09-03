"""Prediction-safe identity for one causal-plus-discourse memory store."""

from __future__ import annotations

from dataclasses import dataclass

from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._identity import exact_int, sha256_digest


COMBINED_CUMULATIVE_STORE_FORMAT = (
    "memory-condense-recall-guarded-combined-store-v1"
)


@dataclass(frozen=True, slots=True)
class CombinedCumulativeStoreReceipt(SealedIdentity):
    """Text-free proof that causal and discourse layers share one corpus."""

    _SEAL_MISMATCH = "combined cumulative store receipt does not match"

    source_store_identity_sha256: str
    target_store_identity_sha256: str
    source_database_sha256: str
    target_database_sha256: str
    target_index_sha256: str
    retrieval_policy_sha256: str
    context_budget_sha256: str
    training_query_batch_sha256: str
    held_out_query_batch_sha256: str
    compilation_receipt_sha256: str
    artifact_id: str
    snapshot_sha256: str
    turn_count: int
    chunk_count: int
    causal_events: int
    causal_graph_edges: int
    retained_request_token_state_bytes: int = 0
    format: str = COMBINED_CUMULATIVE_STORE_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != COMBINED_CUMULATIVE_STORE_FORMAT:
            raise ValueError("unsupported combined cumulative store format")
        for name in (
            "source_store_identity_sha256",
            "target_store_identity_sha256",
            "source_database_sha256",
            "target_database_sha256",
            "target_index_sha256",
            "retrieval_policy_sha256",
            "context_budget_sha256",
            "training_query_batch_sha256",
            "held_out_query_batch_sha256",
            "compilation_receipt_sha256",
            "snapshot_sha256",
        ):
            sha256_digest(getattr(self, name), name)
        if self.source_store_identity_sha256 != self.target_store_identity_sha256:
            raise ValueError("combined store changed source turn/chunk identities")
        artifact = str(self.artifact_id).strip()
        if not artifact:
            raise ValueError("artifact_id must be non-empty")
        object.__setattr__(self, "artifact_id", artifact)
        for name in (
            "turn_count",
            "chunk_count",
            "causal_events",
            "causal_graph_edges",
            "retained_request_token_state_bytes",
        ):
            object.__setattr__(
                self,
                name,
                exact_int(getattr(self, name), name, minimum=0),
            )
        if self.turn_count < 1 or self.chunk_count < 1:
            raise ValueError("combined store cannot be empty")
        if self.retained_request_token_state_bytes != 0:
            raise ValueError("combined store retained request-token state")
        self._seal()


__all__ = ["COMBINED_CUMULATIVE_STORE_FORMAT", "CombinedCumulativeStoreReceipt"]
