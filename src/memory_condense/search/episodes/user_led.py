"""Deterministic user-led segmentation of exact conversation evidence.

Each user turn starts one logical exchange.  Later machine turns belong to
that exchange until the next user turn.  This module deliberately operates on
``EvidenceSpan`` metadata only: it reads no text and invokes no model.

When an exchange is too large, its attached evidence is partitioned into
bounded shards.  Every shard can hydrate the complete user lead through an
exact retained-evidence sidecar, while persisted ``Episode.evidence`` remains
a disjoint, monotonic, exactly-once partition of the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from memory_condense.domain._discourse_identity import (
    identity_sha256,
    make_episode_id,
)
from memory_condense.domain.discourse import (
    Episode,
    EvidenceSpan,
    evidence_span_sort_key,
)
from memory_condense.domain.sealed import SealedIdentity


USER_LED_EPISODE_FORMAT = "user-led-episodes-v1"
USER_LED_BOUNDARY_METHOD = "user_turn_lead"
ORPHAN_PRELUDE_BOUNDARY_METHOD = "orphan_prelude"

UserLedExchangeKind = Literal["user_led", "orphan_prelude"]


def _evidence_sha256(evidence: Sequence[EvidenceSpan]) -> str:
    return identity_sha256(
        {"evidence": [span.identity_payload() for span in evidence]}
    )


def _span_identity(span: EvidenceSpan) -> str:
    return identity_sha256(span.identity_payload())


def make_user_led_exchange_id(
    *,
    artifact_id: str,
    source_id: str,
    exchange_kind: UserLedExchangeKind,
    lead_evidence: Sequence[EvidenceSpan],
) -> str:
    """Return an identity that is invariant to exchange shard size.

    A user exchange is anchored to its authoritative lead turn rather than to
    attached response spans.  The pre-user prelude is unique per artifact and
    source.  Consequently adding response spans or changing a shard bound does
    not rename the logical exchange.
    """

    artifact = str(artifact_id).strip()
    source = str(source_id).strip()
    lead = tuple(lead_evidence)
    if not artifact or not source:
        raise ValueError("artifact_id and source_id must be non-empty")
    if exchange_kind not in {"user_led", "orphan_prelude"}:
        raise ValueError("exchange_kind is not supported")
    if exchange_kind == "user_led":
        if not lead or any(span.role != "user" for span in lead):
            raise ValueError("a user-led exchange requires user lead evidence")
        lead_ordinals = {span.ordinal for span in lead}
        if len(lead_ordinals) != 1:
            raise ValueError("user lead evidence must come from one turn ordinal")
        lead_turn_ids = {span.turn_id for span in lead if span.turn_id is not None}
        if len(lead_turn_ids) > 1:
            raise ValueError("user lead evidence must come from one turn")
        anchor = {
            "ordinal": lead[0].ordinal,
            "turn_id": next(iter(lead_turn_ids), None),
        }
    else:
        if lead:
            raise ValueError("an orphan prelude cannot have user lead evidence")
        anchor = None
    payload = {
        "format": USER_LED_EPISODE_FORMAT,
        "artifact_id": artifact,
        "source_id": source,
        "exchange_kind": exchange_kind,
        "lead_turn": anchor,
    }
    return f"exchange-{identity_sha256(payload)[:24]}"


@dataclass(frozen=True, slots=True)
class UserLedEpisodeShard(SealedIdentity):
    """One immutable episode plus its logical-exchange ownership receipt."""

    _SEAL_MISMATCH = "user-led episode shard receipt does not match its payload"

    episode: Episode
    exchange_id: str
    exchange_kind: UserLedExchangeKind
    shard_index: int
    shard_count: int
    lead_evidence: tuple[EvidenceSpan, ...]
    owned_evidence: tuple[EvidenceSpan, ...]
    retained_lead_evidence: tuple[EvidenceSpan, ...] = ()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "lead_evidence", tuple(self.lead_evidence))
        object.__setattr__(self, "owned_evidence", tuple(self.owned_evidence))
        object.__setattr__(
            self,
            "retained_lead_evidence",
            tuple(self.retained_lead_evidence),
        )
        if not self.exchange_id.strip():
            raise ValueError("exchange_id must be non-empty")
        if self.exchange_kind not in {"user_led", "orphan_prelude"}:
            raise ValueError("exchange_kind is not supported")
        if self.shard_count < 1 or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index must address a positive shard_count")
        if not self.owned_evidence:
            raise ValueError("each shard must own at least one input span")

        for label, evidence in (
            ("lead", self.lead_evidence),
            ("owned", self.owned_evidence),
            ("retained lead", self.retained_lead_evidence),
        ):
            if tuple(sorted(evidence, key=evidence_span_sort_key)) != evidence:
                raise ValueError(f"{label} evidence must be in source order")
            identities = tuple(_span_identity(span) for span in evidence)
            if len(set(identities)) != len(identities):
                raise ValueError(f"{label} evidence cannot contain duplicates")

        expected_exchange_id = make_user_led_exchange_id(
            artifact_id=self.episode.artifact_id,
            source_id=self.episode.source_id,
            exchange_kind=self.exchange_kind,
            lead_evidence=self.lead_evidence,
        )
        if self.exchange_id != expected_exchange_id:
            raise ValueError("exchange_id does not match its stable lead anchor")

        lead_ids = {_span_identity(span) for span in self.lead_evidence}
        owned_ids = {_span_identity(span) for span in self.owned_evidence}
        retained_ids = {
            _span_identity(span) for span in self.retained_lead_evidence
        }
        if self.exchange_kind == "user_led":
            if not self.lead_evidence:
                raise ValueError("user-led shards require lead evidence")
            if self.shard_index == 0:
                if self.retained_lead_evidence:
                    raise ValueError("the first shard owns rather than retains its lead")
                if self.owned_evidence[: len(self.lead_evidence)] != self.lead_evidence:
                    raise ValueError("the first shard must own the complete lead")
            else:
                if self.retained_lead_evidence != self.lead_evidence:
                    raise ValueError("later shards must retain the complete lead")
                if lead_ids & owned_ids:
                    raise ValueError("retained lead evidence cannot also be owned")
        else:
            if self.lead_evidence or self.retained_lead_evidence:
                raise ValueError("orphan preludes cannot carry user lead evidence")
            if any(span.role == "user" for span in self.owned_evidence):
                raise ValueError("orphan preludes cannot own user evidence")

        if owned_ids & retained_ids:
            raise ValueError("owned and retained evidence must be disjoint")
        retrieval_evidence = tuple(
            sorted(
                (*self.retained_lead_evidence, *self.owned_evidence),
                key=evidence_span_sort_key,
            )
        )
        if self.episode.evidence != self.owned_evidence:
            raise ValueError("persisted episode evidence must equal owned evidence")
        if len({_span_identity(span) for span in retrieval_evidence}) != len(
            retrieval_evidence
        ):
            raise ValueError("retrieval evidence cannot contain duplicates")
        expected_method = (
            USER_LED_BOUNDARY_METHOD
            if self.exchange_kind == "user_led"
            else ORPHAN_PRELUDE_BOUNDARY_METHOD
        )
        if self.shard_count > 1:
            expected_method += "_shard"
        if self.episode.boundary_method != expected_method:
            raise ValueError("episode boundary_method does not match its shard receipt")
        self._seal()

    @property
    def retrieval_evidence(self) -> tuple[EvidenceSpan, ...]:
        """Exact evidence to hydrate for this shard, including its user lead."""

        return tuple(
            sorted(
                (*self.retained_lead_evidence, *self.owned_evidence),
                key=evidence_span_sort_key,
            )
        )


@dataclass(frozen=True, slots=True)
class UserLedEpisodeBuildResult(SealedIdentity):
    """Sealed output whose ownership rows partition the input exactly once."""

    _SEAL_MISMATCH = "user-led episode build receipt does not match its payload"

    source_id: str
    artifact_id: str
    sequence_start: int
    input_span_count: int
    input_evidence_sha256: str
    shards: tuple[UserLedEpisodeShard, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shards", tuple(self.shards))
        if not self.source_id.strip() or not self.artifact_id.strip():
            raise ValueError("source_id and artifact_id must be non-empty")
        if self.sequence_start < 0 or self.input_span_count < 0:
            raise ValueError("sequence_start and input_span_count must be non-negative")
        if len(self.input_evidence_sha256) != 64:
            raise ValueError("input_evidence_sha256 must be a SHA-256 digest")

        expected_sequences = tuple(
            range(self.sequence_start, self.sequence_start + len(self.shards))
        )
        actual_sequences = tuple(shard.episode.sequence_no for shard in self.shards)
        if actual_sequences != expected_sequences:
            raise ValueError("episode sequence numbers must be contiguous")
        if any(
            shard.episode.source_id != self.source_id
            or shard.episode.artifact_id != self.artifact_id
            for shard in self.shards
        ):
            raise ValueError("build results cannot cross source or artifact boundaries")

        owned = tuple(span for shard in self.shards for span in shard.owned_evidence)
        if len(owned) != self.input_span_count:
            raise ValueError("owned evidence count does not match the input")
        if tuple(sorted(owned, key=evidence_span_sort_key)) != owned:
            raise ValueError("owned evidence must preserve global source order")
        identities = tuple(_span_identity(span) for span in owned)
        if len(set(identities)) != len(identities):
            raise ValueError("owned evidence must partition input spans exactly once")
        if _evidence_sha256(owned) != self.input_evidence_sha256:
            raise ValueError("owned evidence does not match the input evidence digest")

        seen: set[str] = set()
        cursor = 0
        while cursor < len(self.shards):
            first = self.shards[cursor]
            if first.exchange_id in seen or first.shard_index != 0:
                raise ValueError("exchange shards must be contiguous and start at zero")
            seen.add(first.exchange_id)
            group = self.shards[cursor : cursor + first.shard_count]
            if len(group) != first.shard_count or any(
                shard.exchange_id != first.exchange_id
                or shard.exchange_kind != first.exchange_kind
                or shard.shard_count != first.shard_count
                or shard.shard_index != index
                or shard.lead_evidence != first.lead_evidence
                for index, shard in enumerate(group)
            ):
                raise ValueError("exchange shard receipts are inconsistent")
            cursor += first.shard_count
        self._seal()

    @property
    def episodes(self) -> tuple[Episode, ...]:
        return tuple(shard.episode for shard in self.shards)

    @property
    def exchange_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(shard.exchange_id for shard in self.shards)
        )


@dataclass(frozen=True, slots=True)
class _TurnGroup:
    ordinal: int
    role: str
    turn_id: str | None
    evidence: tuple[EvidenceSpan, ...]


@dataclass(frozen=True, slots=True)
class _LogicalExchange:
    kind: UserLedExchangeKind
    lead: tuple[EvidenceSpan, ...]
    attached: tuple[EvidenceSpan, ...]


class UserLedEpisodeBuilder:
    """Build user-led exact episodes without text, facts, or providers."""

    __slots__ = ("max_spans_per_shard",)

    def __init__(self, *, max_spans_per_shard: int | None = None) -> None:
        if max_spans_per_shard is not None and (
            isinstance(max_spans_per_shard, bool)
            or not isinstance(max_spans_per_shard, int)
            or max_spans_per_shard < 1
        ):
            raise ValueError("max_spans_per_shard must be a positive integer or None")
        self.max_spans_per_shard = max_spans_per_shard

    def build(
        self,
        *,
        source_id: str,
        artifact_id: str,
        spans: Sequence[EvidenceSpan],
        sequence_start: int = 0,
    ) -> UserLedEpisodeBuildResult:
        source = str(source_id).strip()
        artifact = str(artifact_id).strip()
        if not source or not artifact:
            raise ValueError("source_id and artifact_id must be non-empty")
        if (
            isinstance(sequence_start, bool)
            or not isinstance(sequence_start, int)
            or sequence_start < 0
        ):
            raise ValueError("sequence_start must be a non-negative integer")

        evidence = tuple(spans)
        self._validate_evidence(source, evidence)
        exchanges = self._form_exchanges(self._group_turns(evidence))
        shards: list[UserLedEpisodeShard] = []
        sequence_no = sequence_start
        for exchange in exchanges:
            rows = self._shard_exchange(exchange)
            exchange_id = make_user_led_exchange_id(
                artifact_id=artifact,
                source_id=source,
                exchange_kind=exchange.kind,
                lead_evidence=exchange.lead,
            )
            shard_count = len(rows)
            for shard_index, (owned, retained) in enumerate(rows):
                method = (
                    USER_LED_BOUNDARY_METHOD
                    if exchange.kind == "user_led"
                    else ORPHAN_PRELUDE_BOUNDARY_METHOD
                )
                if shard_count > 1:
                    method += "_shard"
                episode = Episode(
                    episode_id=make_episode_id(
                        artifact_id=artifact,
                        source_id=source,
                        sequence_no=sequence_no,
                        evidence=owned,
                    ),
                    artifact_id=artifact,
                    source_id=source,
                    sequence_no=sequence_no,
                    first_ordinal=owned[0].ordinal,
                    last_ordinal=owned[-1].ordinal,
                    evidence=owned,
                    boundary_method=method,
                )
                shards.append(
                    UserLedEpisodeShard(
                        episode=episode,
                        exchange_id=exchange_id,
                        exchange_kind=exchange.kind,
                        shard_index=shard_index,
                        shard_count=shard_count,
                        lead_evidence=exchange.lead,
                        owned_evidence=owned,
                        retained_lead_evidence=retained,
                    )
                )
                sequence_no += 1

        return UserLedEpisodeBuildResult(
            source_id=source,
            artifact_id=artifact,
            sequence_start=sequence_start,
            input_span_count=len(evidence),
            input_evidence_sha256=_evidence_sha256(evidence),
            shards=tuple(shards),
        )

    @staticmethod
    def _validate_evidence(
        source_id: str,
        evidence: tuple[EvidenceSpan, ...],
    ) -> None:
        if tuple(sorted(evidence, key=evidence_span_sort_key)) != evidence:
            raise ValueError("spans must be supplied in deterministic source order")
        identities = tuple(_span_identity(span) for span in evidence)
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate evidence spans are not allowed")
        if any(span.source_id != source_id for span in evidence):
            raise ValueError("user-led segmentation requires one explicit source_id")
        if any(span.role not in {"user", "assistant", "system"} for span in evidence):
            raise ValueError(
                "user-led segmentation requires an explicit user, assistant, "
                "or system role"
            )

    @staticmethod
    def _group_turns(evidence: tuple[EvidenceSpan, ...]) -> tuple[_TurnGroup, ...]:
        groups: list[_TurnGroup] = []
        seen_turn_ids: dict[str, int] = {}
        cursor = 0
        while cursor < len(evidence):
            ordinal = evidence[cursor].ordinal
            end = cursor + 1
            while end < len(evidence) and evidence[end].ordinal == ordinal:
                end += 1
            rows = evidence[cursor:end]
            roles = {span.role for span in rows}
            if len(roles) != 1:
                raise ValueError("all chunks from one turn ordinal must share one role")
            turn_ids = {span.turn_id for span in rows if span.turn_id is not None}
            if len(turn_ids) > 1:
                raise ValueError("one turn ordinal cannot contain multiple turn_ids")
            turn_id = next(iter(turn_ids), None)
            if turn_id is not None:
                prior = seen_turn_ids.setdefault(turn_id, ordinal)
                if prior != ordinal:
                    raise ValueError("a turn_id cannot occur at multiple ordinals")
            groups.append(
                _TurnGroup(
                    ordinal=ordinal,
                    role=next(iter(roles)),
                    turn_id=turn_id,
                    evidence=rows,
                )
            )
            cursor = end
        return tuple(groups)

    @staticmethod
    def _form_exchanges(
        turns: tuple[_TurnGroup, ...],
    ) -> tuple[_LogicalExchange, ...]:
        exchanges: list[_LogicalExchange] = []
        prelude: list[EvidenceSpan] = []
        current_lead: tuple[EvidenceSpan, ...] | None = None
        attached: list[EvidenceSpan] = []
        for turn in turns:
            if turn.role == "user":
                if current_lead is None:
                    if prelude:
                        exchanges.append(
                            _LogicalExchange(
                                kind="orphan_prelude",
                                lead=(),
                                attached=tuple(prelude),
                            )
                        )
                        prelude.clear()
                else:
                    exchanges.append(
                        _LogicalExchange(
                            kind="user_led",
                            lead=current_lead,
                            attached=tuple(attached),
                        )
                    )
                    attached.clear()
                current_lead = turn.evidence
            elif current_lead is None:
                prelude.extend(turn.evidence)
            else:
                attached.extend(turn.evidence)
        if current_lead is not None:
            exchanges.append(
                _LogicalExchange(
                    kind="user_led",
                    lead=current_lead,
                    attached=tuple(attached),
                )
            )
        elif prelude:
            exchanges.append(
                _LogicalExchange(
                    kind="orphan_prelude",
                    lead=(),
                    attached=tuple(prelude),
                )
            )
        return tuple(exchanges)

    def _shard_exchange(
        self,
        exchange: _LogicalExchange,
    ) -> tuple[tuple[tuple[EvidenceSpan, ...], tuple[EvidenceSpan, ...]], ...]:
        bound = self.max_spans_per_shard
        if exchange.kind == "orphan_prelude":
            width = len(exchange.attached) if bound is None else bound
            return tuple(
                (exchange.attached[start : start + width], ())
                for start in range(0, len(exchange.attached), width)
            )

        lead = exchange.lead
        if bound is not None and len(lead) > bound:
            raise ValueError(
                "max_spans_per_shard cannot split a multi-chunk user lead"
            )
        if bound is None or len(lead) + len(exchange.attached) <= bound:
            return (((*lead, *exchange.attached), ()),)
        payload_width = bound - len(lead)
        if payload_width < 1:
            raise ValueError(
                "max_spans_per_shard leaves no room for attached evidence "
                "after retaining the user lead"
            )
        payloads = tuple(
            exchange.attached[start : start + payload_width]
            for start in range(0, len(exchange.attached), payload_width)
        )
        return tuple(
            (((*lead, *payload), ()))
            if index == 0
            else ((payload, lead))
            for index, payload in enumerate(payloads)
        )


def build_user_led_episodes(
    *,
    source_id: str,
    artifact_id: str,
    spans: Sequence[EvidenceSpan],
    sequence_start: int = 0,
    max_spans_per_shard: int | None = None,
) -> UserLedEpisodeBuildResult:
    """Convenience wrapper around :class:`UserLedEpisodeBuilder`."""

    return UserLedEpisodeBuilder(
        max_spans_per_shard=max_spans_per_shard
    ).build(
        source_id=source_id,
        artifact_id=artifact_id,
        spans=spans,
        sequence_start=sequence_start,
    )


__all__ = [
    "ORPHAN_PRELUDE_BOUNDARY_METHOD",
    "USER_LED_BOUNDARY_METHOD",
    "USER_LED_EPISODE_FORMAT",
    "UserLedEpisodeBuildResult",
    "UserLedEpisodeBuilder",
    "UserLedEpisodeShard",
    "build_user_led_episodes",
    "make_user_led_exchange_id",
]
