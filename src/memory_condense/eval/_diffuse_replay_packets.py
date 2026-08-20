"""Immutable packet views reconstructed from a verified replay database."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from memory_condense.domain.discourse import EvidencePacket, EvidenceSpan, quote_sha256
from memory_condense.eval._diffuse_replay_contracts import (
    DiffuseLongMemEvalReplayReceipt,
)
from memory_condense.eval._identity import exact_int, sha256_digest
from memory_condense.eval.diffuse_longmemeval import LongMemEvalDiffuseQueryReceipt


ReplayBoundaryMode = Literal["fixed_interval", "lexical_embedding", "qwen_head"]


@dataclass(frozen=True, slots=True)
class VerifiedDiffuseReplayPacket:
    """One exact final packet plus an authoritative read-only span resolver."""

    boundary_mode: ReplayBoundaryMode
    question_ordinal: int
    question_id_sha256: str
    question_probe_sha256: str
    packet: EvidencePacket
    receipt: LongMemEvalDiffuseQueryReceipt
    _authoritative_span_texts: tuple[tuple[EvidenceSpan, str], ...]

    def __post_init__(self) -> None:
        if self.boundary_mode not in {
            "fixed_interval",
            "lexical_embedding",
            "qwen_head",
        }:
            raise ValueError("unsupported replay boundary mode")
        exact_int(self.question_ordinal, "question_ordinal", minimum=0)
        sha256_digest(self.question_id_sha256, "question_id_sha256")
        sha256_digest(self.question_probe_sha256, "question_probe_sha256")
        if self.receipt.packet_receipt_sha256 != self.packet.receipt.receipt_sha256:
            raise ValueError("verified replay packet and query receipt disagree")
        expected_spans = tuple(atom.span for atom in self.packet.atoms)
        observed_spans = tuple(span for span, _text in self._authoritative_span_texts)
        if observed_spans != expected_spans:
            raise ValueError("authoritative span resolver differs from the packet")
        for atom, (_span, text) in zip(
            self.packet.atoms,
            self._authoritative_span_texts,
            strict=True,
        ):
            if text != atom.text or quote_sha256(text) != atom.span.quote_sha256:
                raise ValueError("authoritative replay span text changed")

    def hydrate_span(self, span: EvidenceSpan) -> str:
        """Resolve only an exact selected span captured from the final database."""

        for expected, text in self._authoritative_span_texts:
            if span == expected:
                return text
        raise KeyError("span is outside this verified replay packet")


@dataclass(frozen=True, slots=True)
class VerifiedDiffuseReplayPackage:
    """One replay receipt and the exact packets reconstructed from its stores."""

    receipt: DiffuseLongMemEvalReplayReceipt
    manifest_file_sha256: str
    packets: tuple[VerifiedDiffuseReplayPacket, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.manifest_file_sha256) is not str or sha256_digest(
            self.manifest_file_sha256,
            "replay manifest file SHA-256",
        ) != self.manifest_file_sha256:
            raise ValueError("replay manifest file SHA-256 must be lowercase")
        expected = tuple(
            (
                arm.boundary_mode,
                query.question_ordinal,
                query.question_id_sha256,
                query.question_probe_sha256,
                query.query_receipt.receipt_sha256,
            )
            for arm in self.receipt.arms
            for query in arm.queries
        )
        observed = tuple(
            (
                packet.boundary_mode,
                packet.question_ordinal,
                packet.question_id_sha256,
                packet.question_probe_sha256,
                packet.receipt.receipt_sha256,
            )
            for packet in self.packets
        )
        if observed != expected:
            raise ValueError("reconstructed packets do not exactly cover the replay")


__all__ = [
    "ReplayBoundaryMode",
    "VerifiedDiffuseReplayPackage",
    "VerifiedDiffuseReplayPacket",
]
