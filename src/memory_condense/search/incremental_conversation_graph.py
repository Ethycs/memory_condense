"""Provider-free incremental graph over exact conversational chunks.

The graph keeps physical chunks as the only evidence-bearing nodes.  A small,
deterministic phrase compiler creates immutable occurrence records whose
postings act as virtual phrase hubs.  That representation gives cross-source
connectivity without materialising an all-pairs edge set: appending a chunk is
linear in that chunk's text, not in the size of the memory corpus.

Two kinds of traversal edge are exposed:

* adjacent chunks in the same source, ordered by turn ordinal and character
  offset; and
* nonadjacent chunks that share a bounded phrase posting cohort, including
  long-range jumps within one source.

Querying never generates an answer.  It returns ranked physical chunks and an
auditable path from question-matched or caller-supplied seeds.  There are no
model, embedding, database, or gold-answer inputs in this module.
"""

from __future__ import annotations

import math
import re
import unicodedata
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from threading import RLock
from typing import Literal, Sequence

from memory_condense.domain._discourse_identity import (
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.schemas import Chunk, Turn
from memory_condense.search.indexes.lexical import STOPWORDS


GraphRelation = Literal["same_source_sequence", "shared_phrase"]
SequenceDirection = Literal["previous", "next"]

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CLAUSE_RE = re.compile(r"[^.!?;:\r\n]+", re.UNICODE)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# These are discourse scaffolding rather than durable topics.  The ordinary
# lexical stop list remains deliberately small; this extra set is restricted
# to terms whose unigram postings would otherwise become conversation-wide
# hubs.  They can still participate in a more specific bigram or trigram.
_GENERIC_UNIGRAMS = frozenset(
    {
        "answer",
        "assistant",
        "conversation",
        "memory",
        "question",
        "remember",
        "something",
        "thing",
        "things",
    }
)

# Source-story links deliberately use only user-authored lexical material.
# These terms are useful for recognizing an assertion locally, but are poor
# evidence that two whole sessions belong to the same story.  Keeping this
# inventory in the graph policy (rather than in a query-specific evaluator)
# makes the ingest image reproducible and gold blind.
_STORY_LINK_STOP = frozenset(
    {
        "about",
        "actually",
        "activity",
        "advice",
        "also",
        "answer",
        "because",
        "could",
        "earliest",
        "event",
        "good",
        "help",
        "initial",
        "just",
        "know",
        "last",
        "latest",
        "like",
        "looking",
        "make",
        "memory",
        "month",
        "need",
        "order",
        "past",
        "question",
        "really",
        "recent",
        "recommend",
        "remember",
        "sequence",
        "something",
        "suggest",
        "thanks",
        "thing",
        "things",
        "think",
        "three",
        "time",
        "today",
        "took",
        "want",
        "week",
        "what",
        "when",
        "wonder",
        "would",
    }
)


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    return value


def _exact_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _exact_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_unit_interval(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError(f"{label} must be finite and in (0, 1]")
    return number


def _is_ordered_subsequence(values: Sequence[str], parent: Sequence[str]) -> bool:
    iterator = iter(parent)
    return all(any(candidate == value for candidate in iterator) for value in values)


@dataclass(frozen=True, slots=True)
class PhraseExtractionPolicy:
    """Question-independent bounds for the deterministic phrase compiler."""

    max_ngram_tokens: int = 3
    min_unigram_chars: int = 4
    max_phrases_per_chunk: int = 2_048

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_ngram_tokens",
            _exact_positive_int(self.max_ngram_tokens, "max_ngram_tokens"),
        )
        if self.max_ngram_tokens > 3:
            raise ValueError("max_ngram_tokens cannot exceed 3")
        object.__setattr__(
            self,
            "min_unigram_chars",
            _exact_positive_int(self.min_unigram_chars, "min_unigram_chars"),
        )
        object.__setattr__(
            self,
            "max_phrases_per_chunk",
            _exact_positive_int(
                self.max_phrases_per_chunk,
                "max_phrases_per_chunk",
            ),
        )

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(
            {
                "schema": "incremental-phrase-extraction-v2",
                **asdict(self),
                "implementation": {
                    "canonicalization": "unicode_nfkc_then_casefold",
                    "cap_policy": "balanced_width_round_robin",
                    "clause_pattern": _CLAUSE_RE.pattern,
                    "generic_unigrams": sorted(_GENERIC_UNIGRAMS),
                    "occurrences_per_chunk_key": 1,
                    "single_digit_identifiers": "retained",
                    "stopword_policy": "ngram_boundary_not_deletion",
                    "stopwords_sha256": identity_sha256(sorted(STOPWORDS)),
                    "token_pattern": _TOKEN_RE.pattern,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class StoryAffinityIndexPolicy:
    """Question-independent bounds for source/session story membership.

    Membership is first-seen and append-only.  A term or source that reaches
    a cap is saturated rather than evicting an earlier member, so publishing a
    later conversation cannot orphan an existing source-story connection.
    """

    max_terms_per_chunk: int = 512
    max_terms_per_source: int = 4_096
    max_sources_per_term: int = 64
    max_user_chunks_per_source: int = 32

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _exact_positive_int(getattr(self, field_name), field_name)

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(
            {
                "schema": "incremental-source-story-index-v2",
                **asdict(self),
                "implementation": {
                    "membership": "first_exact_user_unigram_occurrence",
                    "morphology": "idempotent_bounded_suffix_normalization_v2",
                    "source_cap": "first_seen_saturation_no_eviction",
                    "stop_terms": sorted(_STORY_LINK_STOP),
                    "term_cap": "first_seen_saturation_no_eviction",
                },
            }
        )


@dataclass(frozen=True, slots=True)
class OrderedStorySearchPolicy:
    """Hard work bounds for one source-scoped exact-cardinality read."""

    max_seed_chunks: int = 16
    max_candidate_chunks: int = 32
    max_candidate_sources: int = 32
    max_neighbor_sources: int = 31
    max_bundle_members: int = 12
    max_terms_per_pair: int = 32
    max_combinations: int = 50_000
    max_evidence_chunks_per_source: int = 8
    max_source_fraction: float = 0.12

    def __post_init__(self) -> None:
        for field_name in (
            "max_seed_chunks",
            "max_candidate_chunks",
            "max_candidate_sources",
            "max_neighbor_sources",
            "max_bundle_members",
            "max_terms_per_pair",
            "max_combinations",
            "max_evidence_chunks_per_source",
        ):
            _exact_positive_int(getattr(self, field_name), field_name)
        if self.max_bundle_members > self.max_candidate_sources:
            raise ValueError("story bundle cap exceeds candidate source cap")
        fraction = float(self.max_source_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("max_source_fraction must be finite and in (0, 1]")
        object.__setattr__(self, "max_source_fraction", fraction)

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(
            {
                "schema": "incremental-ordered-source-story-search-v1",
                **asdict(self),
                "implementation": {
                    "affinity": "candidate_local_inverse_source_frequency",
                    "chronology": "distinct_utc_source_dates",
                    "coherence": "connected_then_clique_density_then_weight",
                    "explicit_candidates": "one_first_ranked_chunk_per_source",
                    "large_pool": "deterministic_anchor_greedy",
                    "seed_discovery": "bounded_rare_term_source_postings",
                },
            }
        )


@dataclass(frozen=True, slots=True)
class CanonicalPhraseSpan:
    """One canonical phrase backed by an exact substring of its input."""

    key: str
    start_offset: int
    end_offset: int
    quote: str
    token_count: int

    def __post_init__(self) -> None:
        _nonempty(self.key, "key")
        _exact_nonnegative_int(self.start_offset, "start_offset")
        _exact_nonnegative_int(self.end_offset, "end_offset")
        if self.end_offset <= self.start_offset:
            raise ValueError("phrase span must be non-empty")
        if not isinstance(self.quote, str) or not self.quote:
            raise ValueError("quote must be non-empty")
        _exact_positive_int(self.token_count, "token_count")


def _canonical_token(raw: str) -> str:
    return unicodedata.normalize("NFKC", raw).casefold()


def _story_suffix_candidate(word: str) -> str:
    """Apply at most one legacy story suffix rewrite to normalized text."""

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    elif word.endswith("oes") and len(word) > 4:
        return word[:-2]
    elif word.endswith("ing") and len(word) > 5:
        candidate = word[:-3]
        if len(candidate) >= 2 and candidate[-1] == candidate[-2]:
            candidate = candidate[:-1]
        return candidate
    elif word.endswith("ed") and len(word) > 4:
        return word[:-2]
    elif word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def _story_term(raw: str) -> str:
    """Return a deterministic, idempotent story-link term.

    The legacy one-pass rules can emit a value that another rule would stem
    again (for example ``closed -> clos -> clo``). Such a value cannot cross
    the durable membership boundary because replay validates canonical terms
    independently. Keep the normalized surface form only for those unstable
    cases; every legacy result that was already a fixed point is unchanged.
    """

    word = _canonical_token(raw).replace("’", "'").strip("' -_")
    candidate = _story_suffix_candidate(word)
    if candidate != word and _story_suffix_candidate(candidate) != candidate:
        return word
    return candidate


def _clause_tokens(text: str) -> tuple[tuple[tuple[str, int, int], ...], ...]:
    """Return contiguous content-token runs; stopwords terminate a run."""

    clauses: list[tuple[tuple[str, int, int], ...]] = []
    for clause_match in _CLAUSE_RE.finditer(text):
        tokens: list[tuple[str, int, int]] = []
        for token_match in _TOKEN_RE.finditer(
            text,
            clause_match.start(),
            clause_match.end(),
        ):
            token = _canonical_token(token_match.group(0))
            if token in STOPWORDS or (len(token) < 2 and not token.isdigit()):
                if tokens:
                    clauses.append(tuple(tokens))
                    tokens = []
                continue
            tokens.append((token, token_match.start(), token_match.end()))
        if tokens:
            clauses.append(tuple(tokens))
    return tuple(clauses)


def extract_canonical_phrases(
    text: str,
    *,
    policy: PhraseExtractionPolicy | None = None,
) -> tuple[CanonicalPhraseSpan, ...]:
    """Extract conservative uni/bi/trigrams with exact source coordinates.

    Punctuation and stopwords terminate an n-gram, so phrase keys never claim
    contiguity that the source bytes do not have.  Unigrams are kept only when
    sufficiently distinctive, except that a single digit is retained as an
    identifier.  The same short token can survive in a contiguous bigram or
    trigram.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    selected_policy = policy or PhraseExtractionPolicy()
    candidates: list[CanonicalPhraseSpan] = []
    # One exact representative per (chunk, canonical key) is sufficient for
    # the virtual hub.  Keeping every repetition lets boilerplate exhaust the
    # chunk cap before a later rare connector is observed.
    seen_keys: set[str] = set()
    for tokens in _clause_tokens(text):
        for start_index, (token, start, end) in enumerate(tokens):
            if (
                len(token) >= selected_policy.min_unigram_chars
                or any(character.isdigit() for character in token)
            ) and token not in _GENERIC_UNIGRAMS:
                if token not in seen_keys:
                    seen_keys.add(token)
                    candidates.append(
                        CanonicalPhraseSpan(token, start, end, text[start:end], 1)
                    )
            maximum = min(
                selected_policy.max_ngram_tokens,
                len(tokens) - start_index,
            )
            for width in range(2, maximum + 1):
                window = tokens[start_index : start_index + width]
                key = " ".join(item[0] for item in window)
                phrase_start = window[0][1]
                phrase_end = window[-1][2]
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(
                    CanonicalPhraseSpan(
                        key=key,
                        start_offset=phrase_start,
                        end_offset=phrase_end,
                        quote=text[phrase_start:phrase_end],
                        token_count=width,
                    )
                )

    if len(candidates) > selected_policy.max_phrases_per_chunk:
        # Balance the three phrase widths so a long diverse chunk cannot spend
        # every slot on trigrams and orphan all of its unigram connectors.
        by_width: dict[int, list[CanonicalPhraseSpan]] = {}
        for span in candidates:
            by_width.setdefault(span.token_count, []).append(span)
        for values in by_width.values():
            values.sort(
                key=lambda span: (
                    -int(any(character.isdigit() for character in span.key)),
                    -len(span.key),
                    span.start_offset,
                    span.end_offset,
                    span.key,
                )
            )
        balanced: list[CanonicalPhraseSpan] = []
        positions = {width: 0 for width in by_width}
        widths = tuple(sorted(by_width, reverse=True))
        while len(balanced) < selected_policy.max_phrases_per_chunk:
            emitted = False
            for width in widths:
                position = positions[width]
                values = by_width[width]
                if position < len(values):
                    balanced.append(values[position])
                    positions[width] += 1
                    emitted = True
                    if len(balanced) == selected_policy.max_phrases_per_chunk:
                        break
            if not emitted:
                break
        candidates = balanced
    return tuple(
        sorted(
            candidates,
            key=lambda span: (
                span.start_offset,
                span.end_offset,
                span.token_count,
                span.key,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class ConversationGraphChunk:
    """One immutable physical evidence node with complete turn provenance."""

    chunk_id: str
    source_id: str
    turn_id: str
    ordinal: int
    role: str
    text: str
    start_char: int
    end_char: int
    created_at: str | None = None
    text_sha256: str = ""

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "source_id", "turn_id", "role"):
            _nonempty(getattr(self, field_name), field_name)
        _exact_nonnegative_int(self.ordinal, "ordinal")
        _exact_nonnegative_int(self.start_char, "start_char")
        _exact_nonnegative_int(self.end_char, "end_char")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be non-empty")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("chunk character bounds must exactly cover text")
        if self.created_at is not None:
            _nonempty(self.created_at, "created_at")
        actual = quote_sha256(self.text)
        if self.text_sha256 and self.text_sha256 != actual:
            raise ValueError("text_sha256 does not match exact chunk text")
        object.__setattr__(self, "text_sha256", actual)

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        turn: Turn,
        *,
        ordinal: int,
    ) -> ConversationGraphChunk:
        """Adapt the repository's durable ``Chunk``/``Turn`` contracts."""

        if chunk.turn_id != turn.turn_id:
            raise ValueError("chunk and turn IDs disagree")
        if (
            chunk.start_char < 0
            or chunk.end_char > len(turn.text)
            or chunk.end_char <= chunk.start_char
            or turn.text[chunk.start_char : chunk.end_char] != chunk.text
        ):
            raise ValueError("chunk text is not the exact cited turn substring")
        return cls(
            chunk_id=chunk.chunk_id,
            source_id=turn.source_id or turn.turn_id,
            turn_id=turn.turn_id,
            ordinal=ordinal,
            role=turn.role,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            created_at=turn.created_at.isoformat(),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "turn_id": self.turn_id,
            "ordinal": self.ordinal,
            "role": self.role,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "created_at": self.created_at,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class PhraseOccurrence:
    """Immutable posting from a canonical key to one exact chunk substring."""

    occurrence_id: str
    phrase_key: str
    chunk_id: str
    source_id: str
    turn_id: str
    start_char: int
    end_char: int
    quote: str
    quote_sha256: str
    token_count: int


@dataclass(frozen=True, slots=True)
class StoryTermEvidence:
    """One exact, source-to-source lexical bridge used by a story score."""

    term: str
    candidate_source_count: int
    weight: float
    left_occurrence: PhraseOccurrence
    right_occurrence: PhraseOccurrence

    def __post_init__(self) -> None:
        _nonempty(self.term, "story term")
        _exact_positive_int(
            self.candidate_source_count,
            "candidate_source_count",
        )
        if self.candidate_source_count < 2:
            raise ValueError("story term must link at least two candidate sources")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("story term weight must be finite and positive")
        if (
            not isinstance(self.left_occurrence, PhraseOccurrence)
            or not isinstance(self.right_occurrence, PhraseOccurrence)
            or self.left_occurrence.source_id
            == self.right_occurrence.source_id
            or _story_term(self.left_occurrence.phrase_key) != self.term
            or _story_term(self.right_occurrence.phrase_key) != self.term
        ):
            raise ValueError("story term lost its exact cross-source occurrences")


@dataclass(frozen=True, slots=True)
class StoryPairAffinity:
    """Bounded and fully witnessed affinity between two candidate sources."""

    left_source_id: str
    right_source_id: str
    score: float
    terms: tuple[StoryTermEvidence, ...]

    def __post_init__(self) -> None:
        _nonempty(self.left_source_id, "left story source")
        _nonempty(self.right_source_id, "right story source")
        if not self.left_source_id < self.right_source_id:
            raise ValueError("story pair sources must be distinct lexical order")
        if (
            not isinstance(self.terms, tuple)
            or not all(isinstance(row, StoryTermEvidence) for row in self.terms)
            or len({row.term for row in self.terms}) != len(self.terms)
            or any(
                row.left_occurrence.source_id != self.left_source_id
                or row.right_occurrence.source_id != self.right_source_id
                for row in self.terms
            )
        ):
            raise ValueError("story pair term evidence changed")
        expected = round(sum(row.weight for row in self.terms), 8)
        if not math.isfinite(self.score) or self.score != expected:
            raise ValueError("story pair score differs from its exact terms")


@dataclass(frozen=True, slots=True)
class OrderedStorySource:
    """One chronologically placed source/session in an exact-k story."""

    source_id: str
    representative_chunk: ConversationGraphChunk
    evidence_chunks: tuple[ConversationGraphChunk, ...]
    event_time_utc: str
    candidate_rank: int

    def __post_init__(self) -> None:
        _nonempty(self.source_id, "ordered story source")
        _exact_nonnegative_int(self.candidate_rank, "candidate_rank")
        parsed = _parse_story_time(self.event_time_utc)
        representative_time = _parse_story_time(
            self.representative_chunk.created_at
        )
        if (
            parsed is None
            or representative_time is None
            or parsed != representative_time
            or self.representative_chunk.source_id != self.source_id
            or self.representative_chunk.role != "user"
            or not self.evidence_chunks
            or self.evidence_chunks[0] != self.representative_chunk
            or len({row.chunk_id for row in self.evidence_chunks})
            != len(self.evidence_chunks)
            or any(
                row.source_id != self.source_id or row.role != "user"
                for row in self.evidence_chunks
            )
        ):
            raise ValueError("ordered story source lost chronology or provenance")


OrderedStoryStatus = Literal[
    "selected",
    "insufficient_distinct_dated_sources",
    "no_connected_story",
]


@dataclass(frozen=True, slots=True)
class OrderedStorySearchResult:
    """Auditable exact-cardinality source story, or an explicit abstention."""

    status: OrderedStoryStatus
    requested_count: int
    seed_chunk_ids: tuple[str, ...]
    explicit_candidate_chunk_ids: tuple[str, ...]
    candidate_source_ids: tuple[str, ...]
    selected_sources: tuple[OrderedStorySource, ...]
    selected_pair_affinities: tuple[StoryPairAffinity, ...]
    graph_revision: int
    story_index_policy_sha256: str
    search_policy_sha256: str
    candidate_derivation: Literal["explicit_candidates", "seed_source_expansion"]
    receipt_sha256: str

    def __post_init__(self) -> None:
        _exact_positive_int(self.requested_count, "requested_count")
        _exact_nonnegative_int(self.graph_revision, "graph_revision")
        for digest, label in (
            (self.story_index_policy_sha256, "story index policy"),
            (self.search_policy_sha256, "story search policy"),
            (self.receipt_sha256, "story search receipt"),
        ):
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} must be a SHA-256 digest")
        for values, label in (
            (self.seed_chunk_ids, "story seeds"),
            (self.explicit_candidate_chunk_ids, "story candidates"),
            (self.candidate_source_ids, "story candidate sources"),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != len(set(values))
                or any(not value or value != value.strip() for value in values)
            ):
                raise ValueError(f"{label} must be ordered unique exact text")
        selected = self.status == "selected"
        if selected != (len(self.selected_sources) == self.requested_count):
            raise ValueError("ordered story status differs from exact-k selection")
        if not selected and (
            self.selected_sources or self.selected_pair_affinities
        ):
            raise ValueError("ordered story abstention retained hidden evidence")
        if selected:
            source_ids = tuple(row.source_id for row in self.selected_sources)
            times = tuple(
                _parse_story_time(row.event_time_utc)
                for row in self.selected_sources
            )
            expected_pairs = {
                tuple(sorted(pair)) for pair in combinations(source_ids, 2)
            }
            actual_pairs = {
                (row.left_source_id, row.right_source_id)
                for row in self.selected_pair_affinities
            }
            if (
                len(source_ids) != len(set(source_ids))
                or any(
                    source_id not in self.candidate_source_ids
                    for source_id in source_ids
                )
                or any(value is None for value in times)
                or tuple(sorted(times)) != times
                or len({value.date() for value in times if value is not None})
                != len(times)
                or actual_pairs != expected_pairs
            ):
                raise ValueError("ordered story selection lost scope or chronology")

    @property
    def selected_source_ids(self) -> tuple[str, ...]:
        return tuple(row.source_id for row in self.selected_sources)

    @property
    def selected_representative_chunk_ids(self) -> tuple[str, ...]:
        return tuple(row.representative_chunk.chunk_id for row in self.selected_sources)


@dataclass(frozen=True, slots=True)
class GraphAppendReceipt:
    """Stable receipt for the first successful append of a physical chunk."""

    revision: int
    chunk_id: str
    chunk_identity_sha256: str
    extraction_policy_sha256: str
    story_index_policy_sha256: str
    occurrence_count: int
    phrase_key_count: int
    new_story_term_membership_count: int
    story_evidence_chunk_retained: bool
    predecessor_chunk_id: str | None
    successor_chunk_id: str | None
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class StoryTermMembership:
    """One first-seen source-story term and its exact unary witness."""

    term: str
    occurrence: PhraseOccurrence

    def __post_init__(self) -> None:
        _nonempty(self.term, "story membership term")
        if (
            not isinstance(self.occurrence, PhraseOccurrence)
            or self.occurrence.token_count != 1
            or _story_term(self.occurrence.phrase_key) != self.term
        ):
            raise ValueError("story membership lost its normalized unary witness")


@dataclass(frozen=True, slots=True)
class ConversationGraphAppendDelta:
    """Complete authenticated material needed to replay one graph append.

    This is the public persistence boundary.  Restoring it does not rerun
    extraction, and the graph validates every physical occurrence, story
    membership, policy binding, sequence coordinate, count, and receipt hash
    before publishing any state.
    """

    chunk: ConversationGraphChunk
    occurrences: tuple[PhraseOccurrence, ...]
    new_story_term_memberships: tuple[StoryTermMembership, ...]
    story_evidence_chunk_retained: bool
    receipt: GraphAppendReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, ConversationGraphChunk):
            raise TypeError("append delta chunk must be a ConversationGraphChunk")
        if (
            not isinstance(self.occurrences, tuple)
            or not all(
                isinstance(row, PhraseOccurrence) for row in self.occurrences
            )
            or len({row.occurrence_id for row in self.occurrences})
            != len(self.occurrences)
        ):
            raise ValueError("append delta occurrences must be ordered and unique")
        if (
            not isinstance(self.new_story_term_memberships, tuple)
            or not all(
                isinstance(row, StoryTermMembership)
                for row in self.new_story_term_memberships
            )
            or len({row.term for row in self.new_story_term_memberships})
            != len(self.new_story_term_memberships)
        ):
            raise ValueError("append delta story memberships must be ordered unique")
        if type(self.story_evidence_chunk_retained) is not bool:
            raise ValueError("story evidence retention flag must be boolean")
        if not isinstance(self.receipt, GraphAppendReceipt):
            raise TypeError("append delta receipt must be a GraphAppendReceipt")


def _graph_append_receipt_payload(
    *,
    revision: int,
    chunk: ConversationGraphChunk,
    extraction_policy_sha256: str,
    story_index_policy_sha256: str,
    occurrences: Sequence[PhraseOccurrence],
    phrase_keys: Sequence[str],
    story_memberships: Sequence[StoryTermMembership],
    story_evidence_chunk_retained: bool,
    predecessor_chunk_id: str | None,
    successor_chunk_id: str | None,
) -> dict[str, object]:
    """Canonical append receipt image shared by live append and restore."""

    return {
        "schema": "incremental-conversation-graph-append-v1",
        "revision": revision,
        "chunk": chunk.identity_payload(),
        "extraction_policy_sha256": extraction_policy_sha256,
        "story_index_policy_sha256": story_index_policy_sha256,
        "occurrence_ids": [row.occurrence_id for row in occurrences],
        "phrase_keys": list(phrase_keys),
        "new_story_terms": [
            {
                "occurrence_id": row.occurrence.occurrence_id,
                "term": row.term,
            }
            for row in story_memberships
        ],
        "story_evidence_chunk_retained": story_evidence_chunk_retained,
        "predecessor_chunk_id": predecessor_chunk_id,
        "successor_chunk_id": successor_chunk_id,
    }


@dataclass(frozen=True, slots=True)
class GraphAppendResult:
    """Append result; retries return the original receipt with ``created=False``."""

    created: bool
    receipt: GraphAppendReceipt


@dataclass(frozen=True, slots=True)
class ConversationGraphStats:
    revision: int
    chunk_count: int
    source_count: int
    phrase_count: int
    occurrence_count: int
    sequence_edge_count: int
    story_term_count: int
    story_term_membership_count: int
    story_evidence_chunk_count: int


@dataclass(frozen=True, slots=True)
class GraphTraversalPolicy:
    """Hard work bounds for one question-seeded graph traversal.

    ``max_results`` counts non-explicit results.  Caller-supplied activation
    seeds are returned for audit without consuming that allowance.
    ``max_degree`` bounds phrase jumps; the at-most-two physical sequence
    neighbors are considered independently.
    """

    max_hops: int = 2
    max_degree: int = 12
    max_frontier: int = 48
    max_results: int = 24
    max_seed_chunks: int = 24
    max_query_phrases: int = 32
    max_phrases_per_node: int = 32
    max_phrase_postings: int = 64
    hop_decay: float = 0.72
    sequence_weight: float = 0.58

    def __post_init__(self) -> None:
        hops = _exact_nonnegative_int(self.max_hops, "max_hops")
        if hops > 2:
            raise ValueError("max_hops cannot exceed 2")
        for field_name in (
            "max_degree",
            "max_frontier",
            "max_results",
            "max_seed_chunks",
            "max_query_phrases",
            "max_phrases_per_node",
            "max_phrase_postings",
        ):
            _exact_positive_int(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "hop_decay",
            _finite_unit_interval(self.hop_decay, "hop_decay"),
        )
        object.__setattr__(
            self,
            "sequence_weight",
            _finite_unit_interval(self.sequence_weight, "sequence_weight"),
        )

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(
            {
                "schema": "incremental-conversation-traversal-v2",
                **asdict(self),
                "implementation": {
                    "path_policy": "best_score_then_shorter_then_lexical_path",
                    "phrase_cohort": "first_seen_bounded_monotone",
                    "phrase_width_weights": [0.56, 0.78, 0.94],
                    "query_match_boost": 1.08,
                    "same_source_shared_phrase": "nonadjacent_only",
                    "sequence_edges": "reserved_outside_phrase_degree",
                },
            }
        )


@dataclass(frozen=True, slots=True)
class GraphTransition:
    """One bounded physical-to-physical graph edge used during traversal."""

    source_chunk_id: str
    target_chunk_id: str
    relation: GraphRelation
    weight: float
    shared_phrases: tuple[str, ...] = ()
    sequence_direction: SequenceDirection | None = None
    source_occurrence: PhraseOccurrence | None = None
    target_occurrence: PhraseOccurrence | None = None


@dataclass(frozen=True, slots=True)
class GraphEvidence:
    """Ranked raw evidence plus the exact graph path that reached it."""

    chunk: ConversationGraphChunk
    score: float
    hop: int
    supporting_seed_chunk_ids: tuple[str, ...]
    matched_query_phrases: tuple[str, ...]
    path: tuple[GraphTransition, ...]

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass(frozen=True, slots=True)
class ConversationGraphSearchResult:
    """Immutable result of a provider-free bounded graph read."""

    question: str
    query_phrase_keys: tuple[str, ...]
    seed_chunk_ids: tuple[str, ...]
    evidence: tuple[GraphEvidence, ...]
    graph_revision: int
    traversal_policy_sha256: str
    question_seed_derivation_enabled: bool

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(row.chunk_id for row in self.evidence)


@dataclass(slots=True)
class _MutableHit:
    score: float
    hop: int
    best_contribution: float
    path: tuple[GraphTransition, ...]
    seed_chunk_ids: set[str] = field(default_factory=set)
    query_phrases: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _StoryCandidate:
    source_id: str
    representative_chunk_id: str
    event_time: datetime
    event_date: str
    rank: int


def _phrase_token_count(key: str) -> int:
    return key.count(" ") + 1


def _phrase_weight(key: str, posting_count: int, *, query_match: bool) -> float:
    # Longer phrases are more selective; virtual-hub degree discounts even a
    # superficially specific phrase that became common in the live corpus.
    width = _phrase_token_count(key)
    width_weight = (0.56, 0.78, 0.94)[min(width, 3) - 1]
    degree_discount = 1.0 / math.sqrt(max(1, posting_count - 1))
    query_boost = 1.08 if query_match else 1.0
    return min(1.0, width_weight * degree_discount * query_boost)


def _noisy_or(left: float, right: float) -> float:
    return 1.0 - (1.0 - left) * (1.0 - right)


def _parse_story_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


class IncrementalConversationGraph:
    """Append-only conversational phrase graph with bounded live traversal.

    Phrase occurrences and chunks are never edited after publication.  A
    backfilled source coordinate changes only the two neighboring sequence
    links needed to splice that physical chunk into the source order.
    """

    def __init__(
        self,
        *,
        extraction_policy: PhraseExtractionPolicy | None = None,
        story_index_policy: StoryAffinityIndexPolicy | None = None,
    ) -> None:
        self._extraction_policy = extraction_policy or PhraseExtractionPolicy()
        self._story_index_policy = story_index_policy or StoryAffinityIndexPolicy()
        self._chunks: dict[str, ConversationGraphChunk] = {}
        self._chunk_occurrences: dict[str, tuple[PhraseOccurrence, ...]] = {}
        self._chunk_phrase_keys: dict[str, tuple[str, ...]] = {}
        self._postings: dict[str, list[PhraseOccurrence]] = {}
        self._phrase_chunk_ids: dict[str, set[str]] = {}
        self._source_order: dict[str, list[tuple[int, int, str]]] = {}
        self._coordinate_owner: dict[tuple[str, int, int], str] = {}
        self._turn_coordinate: dict[str, tuple[str, int]] = {}
        self._source_ordinal_turn: dict[tuple[str, int], str] = {}
        self._sequence_neighbors: dict[str, set[str]] = {}
        self._receipts: dict[str, GraphAppendReceipt] = {}
        self._story_source_terms: dict[
            str, dict[str, PhraseOccurrence]
        ] = {}
        self._story_term_sources: dict[str, list[str]] = {}
        self._story_source_user_chunks: dict[str, list[str]] = {}
        self._occurrence_count = 0
        self._sequence_edge_count = 0
        self._story_term_membership_count = 0
        self._story_evidence_chunk_count = 0
        self._revision = 0
        self._lock = RLock()

    @property
    def extraction_policy(self) -> PhraseExtractionPolicy:
        return self._extraction_policy

    @property
    def story_index_policy(self) -> StoryAffinityIndexPolicy:
        return self._story_index_policy

    def stats(self) -> ConversationGraphStats:
        with self._lock:
            return ConversationGraphStats(
                revision=self._revision,
                chunk_count=len(self._chunks),
                source_count=len(self._source_order),
                phrase_count=len(self._postings),
                occurrence_count=self._occurrence_count,
                sequence_edge_count=self._sequence_edge_count,
                story_term_count=len(self._story_term_sources),
                story_term_membership_count=self._story_term_membership_count,
                story_evidence_chunk_count=self._story_evidence_chunk_count,
            )

    def chunk(self, chunk_id: str) -> ConversationGraphChunk | None:
        with self._lock:
            return self._chunks.get(str(chunk_id))

    def occurrences(self, phrase_key: str) -> tuple[PhraseOccurrence, ...]:
        """Return the immutable append-ordered posting for a canonical key."""

        key = _canonical_token(str(phrase_key).strip())
        with self._lock:
            return tuple(self._postings.get(key, ()))

    def chunk_occurrences(self, chunk_id: str) -> tuple[PhraseOccurrence, ...]:
        with self._lock:
            return self._chunk_occurrences.get(str(chunk_id), ())

    def story_source_terms(self, source_id: str) -> tuple[str, ...]:
        """Return accepted source-story terms in first-occurrence order."""

        with self._lock:
            return tuple(self._story_source_terms.get(str(source_id), ()))

    def story_term_sources(self, term: str) -> tuple[str, ...]:
        """Return the monotone bounded source cohort for a normalized term."""

        normalized = _story_term(str(term))
        with self._lock:
            return tuple(self._story_term_sources.get(normalized, ()))

    def _append_delta_unlocked(
        self,
        chunk_id: str,
    ) -> ConversationGraphAppendDelta:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            raise KeyError(f"unknown conversation graph chunk: {chunk_id}")
        receipt = self._receipts[chunk_id]
        memberships = tuple(
            StoryTermMembership(term=term, occurrence=occurrence)
            for term, occurrence in self._story_source_terms.get(
                chunk.source_id,
                {},
            ).items()
            if occurrence.chunk_id == chunk_id
        )
        if len(memberships) != receipt.new_story_term_membership_count:
            raise AssertionError("graph story delta projection lost a membership")
        return ConversationGraphAppendDelta(
            chunk=chunk,
            occurrences=self._chunk_occurrences[chunk_id],
            new_story_term_memberships=memberships,
            story_evidence_chunk_retained=(
                receipt.story_evidence_chunk_retained
            ),
            receipt=receipt,
        )

    def append_delta(self, chunk_id: str) -> ConversationGraphAppendDelta:
        """Project one immutable append for authenticated durable storage."""

        normalized = _nonempty(str(chunk_id), "chunk_id")
        with self._lock:
            return self._append_delta_unlocked(normalized)

    def _validate_append_delta(
        self,
        delta: ConversationGraphAppendDelta,
    ) -> tuple[str, ...]:
        """Validate persisted material without extracting text again."""

        chunk = delta.chunk
        receipt = delta.receipt
        _exact_positive_int(receipt.revision, "append receipt revision")
        for value, label in (
            (receipt.chunk_id, "append receipt chunk_id"),
            (receipt.chunk_identity_sha256, "chunk identity receipt"),
            (receipt.extraction_policy_sha256, "extraction policy receipt"),
            (receipt.story_index_policy_sha256, "story policy receipt"),
            (receipt.receipt_sha256, "append receipt"),
        ):
            _nonempty(value, label)
        for digest, label in (
            (receipt.chunk_identity_sha256, "chunk identity receipt"),
            (receipt.extraction_policy_sha256, "extraction policy receipt"),
            (receipt.story_index_policy_sha256, "story policy receipt"),
            (receipt.receipt_sha256, "append receipt"),
        ):
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{label} must be a SHA-256 digest")
        for value, label in (
            (receipt.occurrence_count, "append occurrence_count"),
            (receipt.phrase_key_count, "append phrase_key_count"),
            (
                receipt.new_story_term_membership_count,
                "append story membership count",
            ),
        ):
            _exact_nonnegative_int(value, label)
        if type(receipt.story_evidence_chunk_retained) is not bool:
            raise ValueError("append story evidence flag must be boolean")
        for neighbor, label in (
            (receipt.predecessor_chunk_id, "append predecessor"),
            (receipt.successor_chunk_id, "append successor"),
        ):
            if neighbor is not None:
                _nonempty(neighbor, label)
                if neighbor == chunk.chunk_id:
                    raise ValueError("append receipt cannot name itself as a neighbor")
        if (
            receipt.chunk_id != chunk.chunk_id
            or receipt.chunk_identity_sha256
            != identity_sha256(chunk.identity_payload())
        ):
            raise ValueError("append receipt lost its physical chunk identity")
        if receipt.extraction_policy_sha256 != self._extraction_policy.policy_sha256:
            raise ValueError("append delta extraction policy does not match graph")
        if (
            receipt.story_index_policy_sha256
            != self._story_index_policy.policy_sha256
        ):
            raise ValueError("append delta story policy does not match graph")

        occurrence_positions: dict[str, int] = {}
        occurrence_by_id: dict[str, PhraseOccurrence] = {}
        for position, occurrence in enumerate(delta.occurrences):
            for value, label in (
                (occurrence.occurrence_id, "occurrence_id"),
                (occurrence.phrase_key, "occurrence phrase_key"),
                (occurrence.chunk_id, "occurrence chunk_id"),
                (occurrence.source_id, "occurrence source_id"),
                (occurrence.turn_id, "occurrence turn_id"),
                (occurrence.quote, "occurrence quote"),
                (occurrence.quote_sha256, "occurrence quote_sha256"),
            ):
                _nonempty(value, label)
            _exact_nonnegative_int(occurrence.start_char, "occurrence start_char")
            _exact_nonnegative_int(occurrence.end_char, "occurrence end_char")
            _exact_positive_int(occurrence.token_count, "occurrence token_count")
            local_start = occurrence.start_char - chunk.start_char
            local_end = occurrence.end_char - chunk.start_char
            exact_quote = (
                0 <= local_start < local_end <= len(chunk.text)
                and chunk.text[local_start:local_end] == occurrence.quote
            )
            if (
                occurrence.chunk_id != chunk.chunk_id
                or occurrence.source_id != chunk.source_id
                or occurrence.turn_id != chunk.turn_id
                or occurrence.phrase_key
                != _canonical_token(occurrence.phrase_key)
                or occurrence.token_count
                != _phrase_token_count(occurrence.phrase_key)
                or not exact_quote
                or occurrence.quote_sha256 != quote_sha256(occurrence.quote)
            ):
                raise ValueError("append occurrence lost exact chunk provenance")
            expected_occurrence_id = identity_sha256(
                {
                    "schema": "conversation-phrase-occurrence-v1",
                    "chunk_id": chunk.chunk_id,
                    "phrase_key": occurrence.phrase_key,
                    "start_char": occurrence.start_char,
                    "end_char": occurrence.end_char,
                    "quote_sha256": occurrence.quote_sha256,
                }
            )
            if occurrence.occurrence_id != expected_occurrence_id:
                raise ValueError("append occurrence identity does not match evidence")
            occurrence_positions[occurrence.occurrence_id] = position
            occurrence_by_id[occurrence.occurrence_id] = occurrence
        expected_occurrence_order = tuple(
            sorted(
                delta.occurrences,
                key=lambda row: (
                    row.start_char,
                    row.end_char,
                    row.token_count,
                    row.phrase_key,
                ),
            )
        )
        if delta.occurrences != expected_occurrence_order:
            raise ValueError("append occurrences are not in compiler order")

        membership_positions: list[int] = []
        for membership in delta.new_story_term_memberships:
            stored = occurrence_by_id.get(
                membership.occurrence.occurrence_id
            )
            if stored != membership.occurrence:
                raise ValueError("story membership cites an unbound occurrence")
            membership_positions.append(
                occurrence_positions[membership.occurrence.occurrence_id]
            )
        if membership_positions != sorted(membership_positions):
            raise ValueError("story memberships are not in occurrence order")
        if chunk.role != "user" and delta.new_story_term_memberships:
            raise ValueError("non-user append cannot publish story memberships")

        phrase_keys = tuple(
            sorted({row.phrase_key for row in delta.occurrences})
        )
        if (
            receipt.occurrence_count != len(delta.occurrences)
            or receipt.phrase_key_count != len(phrase_keys)
            or receipt.new_story_term_membership_count
            != len(delta.new_story_term_memberships)
            or receipt.story_evidence_chunk_retained
            != delta.story_evidence_chunk_retained
        ):
            raise ValueError("append delta counts disagree with its receipt")
        payload = _graph_append_receipt_payload(
            revision=receipt.revision,
            chunk=chunk,
            extraction_policy_sha256=receipt.extraction_policy_sha256,
            story_index_policy_sha256=receipt.story_index_policy_sha256,
            occurrences=delta.occurrences,
            phrase_keys=phrase_keys,
            story_memberships=delta.new_story_term_memberships,
            story_evidence_chunk_retained=(
                delta.story_evidence_chunk_retained
            ),
            predecessor_chunk_id=receipt.predecessor_chunk_id,
            successor_chunk_id=receipt.successor_chunk_id,
        )
        if receipt.receipt_sha256 != identity_sha256(payload):
            raise ValueError("append delta does not reproduce its receipt")
        return phrase_keys

    def restore_append_delta(
        self,
        delta: ConversationGraphAppendDelta,
    ) -> GraphAppendResult:
        """Restore one authenticated append without text re-extraction.

        Validation is completed before mutation.  Identical retries are
        idempotent; any conflicting retry, policy mismatch, broken physical
        witness, non-contiguous revision, or cap violation fails closed.
        """

        if not isinstance(delta, ConversationGraphAppendDelta):
            raise TypeError("delta must be a ConversationGraphAppendDelta")
        phrase_keys = self._validate_append_delta(delta)
        chunk = delta.chunk
        receipt = delta.receipt
        coordinate = (chunk.ordinal, chunk.start_char, chunk.chunk_id)
        owner_key = (chunk.source_id, chunk.ordinal, chunk.start_char)
        with self._lock:
            existing = self._chunks.get(chunk.chunk_id)
            if existing is not None:
                if self._append_delta_unlocked(chunk.chunk_id) != delta:
                    raise ValueError("chunk_id is already bound to a different delta")
                return GraphAppendResult(False, receipt)
            if receipt.revision != self._revision + 1:
                raise ValueError("graph delta revision is not contiguous")
            if owner_key in self._coordinate_owner:
                raise ValueError("graph delta repeats a source coordinate")
            turn_coordinate = (chunk.source_id, chunk.ordinal)
            existing_turn = self._turn_coordinate.get(chunk.turn_id)
            if existing_turn is not None and existing_turn != turn_coordinate:
                raise ValueError("graph delta has conflicting turn provenance")
            source_ordinal = (chunk.source_id, chunk.ordinal)
            ordinal_turn = self._source_ordinal_turn.get(source_ordinal)
            if ordinal_turn is not None and ordinal_turn != chunk.turn_id:
                raise ValueError("graph delta has conflicting source chronology")

            source_rows = self._source_order.get(chunk.source_id, [])
            position = bisect_left(source_rows, coordinate)
            predecessor = source_rows[position - 1][2] if position else None
            successor = (
                source_rows[position][2] if position < len(source_rows) else None
            )
            if (
                predecessor != receipt.predecessor_chunk_id
                or successor != receipt.successor_chunk_id
            ):
                raise ValueError("graph delta sequence receipt does not match")

            source_story_terms = self._story_source_terms.get(chunk.source_id, {})
            memberships = delta.new_story_term_memberships
            if (
                len(source_story_terms) + len(memberships)
                > self._story_index_policy.max_terms_per_source
            ):
                raise ValueError("graph story source cap was exceeded")
            for membership in memberships:
                term_sources = self._story_term_sources.get(membership.term, [])
                if (
                    membership.term in source_story_terms
                    or chunk.source_id in term_sources
                ):
                    raise ValueError("graph story delta repeats a source term")
                if (
                    len(term_sources)
                    >= self._story_index_policy.max_sources_per_term
                ):
                    raise ValueError("graph story term cap was exceeded")
            source_evidence = self._story_source_user_chunks.get(
                chunk.source_id,
                [],
            )
            expected_evidence_retention = (
                chunk.role == "user"
                and len(source_evidence)
                < self._story_index_policy.max_user_chunks_per_source
            )
            if (
                delta.story_evidence_chunk_retained
                != expected_evidence_retention
            ):
                raise ValueError("graph story evidence receipt is invalid")

            source_rows = self._source_order.setdefault(chunk.source_id, [])
            self._chunks[chunk.chunk_id] = chunk
            self._chunk_occurrences[chunk.chunk_id] = delta.occurrences
            self._chunk_phrase_keys[chunk.chunk_id] = phrase_keys
            self._coordinate_owner[owner_key] = chunk.chunk_id
            self._turn_coordinate.setdefault(chunk.turn_id, turn_coordinate)
            self._source_ordinal_turn.setdefault(source_ordinal, chunk.turn_id)
            self._sequence_neighbors[chunk.chunk_id] = set()
            source_rows.insert(position, coordinate)
            if predecessor is not None and successor is not None:
                self._unlink_sequence(predecessor, successor)
            if predecessor is not None:
                self._link_sequence(predecessor, chunk.chunk_id)
            if successor is not None:
                self._link_sequence(chunk.chunk_id, successor)
            for occurrence in delta.occurrences:
                self._postings.setdefault(occurrence.phrase_key, []).append(
                    occurrence
                )
                self._phrase_chunk_ids.setdefault(
                    occurrence.phrase_key,
                    set(),
                ).add(chunk.chunk_id)
            source_story_terms = self._story_source_terms.setdefault(
                chunk.source_id,
                {},
            )
            for membership in memberships:
                source_story_terms[membership.term] = membership.occurrence
                self._story_term_sources.setdefault(
                    membership.term,
                    [],
                ).append(chunk.source_id)
            source_evidence = self._story_source_user_chunks.setdefault(
                chunk.source_id,
                [],
            )
            if delta.story_evidence_chunk_retained:
                source_evidence.append(chunk.chunk_id)
            self._occurrence_count += len(delta.occurrences)
            self._story_term_membership_count += len(memberships)
            self._story_evidence_chunk_count += int(
                delta.story_evidence_chunk_retained
            )
            self._revision = receipt.revision
            self._receipts[chunk.chunk_id] = receipt
            return GraphAppendResult(True, receipt)

    def _compile_occurrences(
        self,
        chunk: ConversationGraphChunk,
    ) -> tuple[PhraseOccurrence, ...]:
        spans = extract_canonical_phrases(
            chunk.text,
            policy=self._extraction_policy,
        )
        output: list[PhraseOccurrence] = []
        for span in spans:
            absolute_start = chunk.start_char + span.start_offset
            absolute_end = chunk.start_char + span.end_offset
            digest = quote_sha256(span.quote)
            occurrence_id = identity_sha256(
                {
                    "schema": "conversation-phrase-occurrence-v1",
                    "chunk_id": chunk.chunk_id,
                    "phrase_key": span.key,
                    "start_char": absolute_start,
                    "end_char": absolute_end,
                    "quote_sha256": digest,
                }
            )
            output.append(
                PhraseOccurrence(
                    occurrence_id=occurrence_id,
                    phrase_key=span.key,
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    turn_id=chunk.turn_id,
                    start_char=absolute_start,
                    end_char=absolute_end,
                    quote=span.quote,
                    quote_sha256=digest,
                    token_count=span.token_count,
                )
            )
        return tuple(output)

    def _compile_story_terms(
        self,
        chunk: ConversationGraphChunk,
        occurrences: Sequence[PhraseOccurrence],
    ) -> tuple[tuple[str, PhraseOccurrence], ...]:
        """Return bounded first exact unigram witnesses for a user chunk."""

        if chunk.role != "user":
            return ()
        output: list[tuple[str, PhraseOccurrence]] = []
        seen: set[str] = set()
        for occurrence in occurrences:
            if occurrence.token_count != 1:
                continue
            term = _story_term(occurrence.phrase_key)
            if len(term) < 4 or term in _STORY_LINK_STOP or term in seen:
                continue
            seen.add(term)
            output.append((term, occurrence))
            if len(output) == self._story_index_policy.max_terms_per_chunk:
                break
        return tuple(output)

    def _unlink_sequence(self, left: str, right: str) -> None:
        if right not in self._sequence_neighbors[left]:
            return
        self._sequence_neighbors[left].remove(right)
        self._sequence_neighbors[right].remove(left)
        self._sequence_edge_count -= 1

    def _link_sequence(self, left: str, right: str) -> None:
        if right in self._sequence_neighbors[left]:
            return
        self._sequence_neighbors[left].add(right)
        self._sequence_neighbors[right].add(left)
        self._sequence_edge_count += 1

    def append_chunk(self, chunk: ConversationGraphChunk) -> GraphAppendResult:
        """Incrementally publish one physical chunk.

        Identical retries are idempotent.  Reusing a physical ID or a source
        coordinate for different evidence fails before any graph mutation.
        """

        if not isinstance(chunk, ConversationGraphChunk):
            raise TypeError("chunk must be a ConversationGraphChunk")
        # The common retry path is O(1): do not re-run phrase extraction for
        # a receipt the graph has already published.  A second check below
        # closes the race with another appender while compilation runs.
        with self._lock:
            existing = self._chunks.get(chunk.chunk_id)
            if existing is not None:
                if existing != chunk:
                    raise ValueError(
                        "chunk_id is already bound to different evidence"
                    )
                return GraphAppendResult(False, self._receipts[chunk.chunk_id])
        occurrences = self._compile_occurrences(chunk)
        compiled_story_terms = self._compile_story_terms(chunk, occurrences)
        coordinate = (chunk.ordinal, chunk.start_char, chunk.chunk_id)
        owner_key = (chunk.source_id, chunk.ordinal, chunk.start_char)
        with self._lock:
            existing = self._chunks.get(chunk.chunk_id)
            if existing is not None:
                if existing != chunk:
                    raise ValueError("chunk_id is already bound to different evidence")
                return GraphAppendResult(False, self._receipts[chunk.chunk_id])
            owner = self._coordinate_owner.get(owner_key)
            if owner is not None:
                raise ValueError(
                    "source ordinal/start_char is already bound to another chunk"
                )
            turn_coordinate = (chunk.source_id, chunk.ordinal)
            existing_turn = self._turn_coordinate.get(chunk.turn_id)
            if existing_turn is not None and existing_turn != turn_coordinate:
                raise ValueError("turn_id has conflicting source or ordinal provenance")
            source_ordinal = (chunk.source_id, chunk.ordinal)
            ordinal_turn = self._source_ordinal_turn.get(source_ordinal)
            if ordinal_turn is not None and ordinal_turn != chunk.turn_id:
                raise ValueError(
                    "source/ordinal is already bound to a different turn"
                )

            source_rows = self._source_order.setdefault(chunk.source_id, [])
            position = bisect_left(source_rows, coordinate)
            predecessor = source_rows[position - 1][2] if position else None
            successor = (
                source_rows[position][2] if position < len(source_rows) else None
            )

            self._chunks[chunk.chunk_id] = chunk
            self._chunk_occurrences[chunk.chunk_id] = occurrences
            phrase_keys = tuple(
                sorted({occurrence.phrase_key for occurrence in occurrences})
            )
            self._chunk_phrase_keys[chunk.chunk_id] = phrase_keys
            self._coordinate_owner[owner_key] = chunk.chunk_id
            self._turn_coordinate.setdefault(chunk.turn_id, turn_coordinate)
            self._source_ordinal_turn.setdefault(source_ordinal, chunk.turn_id)
            self._sequence_neighbors[chunk.chunk_id] = set()
            source_rows.insert(position, coordinate)
            if predecessor is not None and successor is not None:
                self._unlink_sequence(predecessor, successor)
            if predecessor is not None:
                self._link_sequence(predecessor, chunk.chunk_id)
            if successor is not None:
                self._link_sequence(chunk.chunk_id, successor)
            for occurrence in occurrences:
                self._postings.setdefault(occurrence.phrase_key, []).append(occurrence)
                self._phrase_chunk_ids.setdefault(
                    occurrence.phrase_key, set()
                ).add(chunk.chunk_id)
            source_story_terms = self._story_source_terms.setdefault(
                chunk.source_id,
                {},
            )
            new_story_memberships: list[StoryTermMembership] = []
            remaining_source_slots = (
                self._story_index_policy.max_terms_per_source
                - len(source_story_terms)
            )
            for term, occurrence in compiled_story_terms:
                if remaining_source_slots <= 0:
                    break
                if term in source_story_terms:
                    continue
                term_sources = self._story_term_sources.setdefault(term, [])
                if len(term_sources) >= self._story_index_policy.max_sources_per_term:
                    continue
                source_story_terms[term] = occurrence
                term_sources.append(chunk.source_id)
                new_story_memberships.append(
                    StoryTermMembership(term=term, occurrence=occurrence)
                )
                remaining_source_slots -= 1
            source_evidence = self._story_source_user_chunks.setdefault(
                chunk.source_id,
                [],
            )
            story_evidence_chunk_retained = (
                chunk.role == "user"
                and len(source_evidence)
                < self._story_index_policy.max_user_chunks_per_source
            )
            if story_evidence_chunk_retained:
                source_evidence.append(chunk.chunk_id)
            self._occurrence_count += len(occurrences)
            self._story_term_membership_count += len(new_story_memberships)
            self._story_evidence_chunk_count += int(story_evidence_chunk_retained)
            self._revision += 1

            payload = _graph_append_receipt_payload(
                revision=self._revision,
                chunk=chunk,
                extraction_policy_sha256=(
                    self._extraction_policy.policy_sha256
                ),
                story_index_policy_sha256=(
                    self._story_index_policy.policy_sha256
                ),
                occurrences=occurrences,
                phrase_keys=phrase_keys,
                story_memberships=new_story_memberships,
                story_evidence_chunk_retained=(
                    story_evidence_chunk_retained
                ),
                predecessor_chunk_id=predecessor,
                successor_chunk_id=successor,
            )
            receipt = GraphAppendReceipt(
                revision=self._revision,
                chunk_id=chunk.chunk_id,
                chunk_identity_sha256=identity_sha256(chunk.identity_payload()),
                extraction_policy_sha256=(
                    self._extraction_policy.policy_sha256
                ),
                story_index_policy_sha256=(
                    self._story_index_policy.policy_sha256
                ),
                occurrence_count=len(occurrences),
                phrase_key_count=len(phrase_keys),
                new_story_term_membership_count=len(new_story_memberships),
                story_evidence_chunk_retained=story_evidence_chunk_retained,
                predecessor_chunk_id=predecessor,
                successor_chunk_id=successor,
                receipt_sha256=identity_sha256(payload),
            )
            self._receipts[chunk.chunk_id] = receipt
            return GraphAppendResult(True, receipt)

    def _first_occurrence(self, chunk_id: str, key: str) -> PhraseOccurrence:
        for occurrence in self._chunk_occurrences[chunk_id]:
            if occurrence.phrase_key == key:
                return occurrence
        raise AssertionError("chunk phrase index lost its occurrence")

    def _sequence_transitions(
        self,
        chunk_id: str,
        policy: GraphTraversalPolicy,
    ) -> list[GraphTransition]:
        source = self._chunks[chunk_id]
        transitions: list[GraphTransition] = []
        source_coordinate = (source.ordinal, source.start_char, source.chunk_id)
        for target_id in self._sequence_neighbors[chunk_id]:
            target = self._chunks[target_id]
            target_coordinate = (target.ordinal, target.start_char, target.chunk_id)
            direction: SequenceDirection = (
                "previous" if target_coordinate < source_coordinate else "next"
            )
            transitions.append(
                GraphTransition(
                    source_chunk_id=chunk_id,
                    target_chunk_id=target_id,
                    relation="same_source_sequence",
                    weight=policy.sequence_weight,
                    sequence_direction=direction,
                )
            )
        transitions.sort(
            key=lambda edge: (
                edge.sequence_direction or "",
                edge.target_chunk_id,
            )
        )
        return transitions

    def _phrase_cohort(self, key: str, limit: int) -> tuple[str, ...]:
        """First-seen bounded chunk cohort for a monotone virtual hub.

        Once the cohort reaches ``limit``, later document-frequency growth
        cannot remove an edge that earlier members could traverse.  Late
        members simply do not join that already-saturated hub.
        """

        output: list[str] = []
        seen: set[str] = set()
        for occurrence in self._postings[key]:
            if occurrence.chunk_id in seen:
                continue
            seen.add(occurrence.chunk_id)
            output.append(occurrence.chunk_id)
            if len(output) == limit:
                break
        return tuple(output)

    def _shared_phrase_transitions(
        self,
        chunk_id: str,
        query_keys: frozenset[str],
        policy: GraphTraversalPolicy,
    ) -> list[GraphTransition]:
        source = self._chunks[chunk_id]
        cohorts = {
            key: self._phrase_cohort(key, policy.max_phrase_postings)
            for key in self._chunk_phrase_keys[chunk_id]
        }
        usable_keys = [
            key
            for key, cohort in cohorts.items()
            if chunk_id in cohort and len(cohort) > 1
        ]
        usable_keys.sort(
            key=lambda key: (
                -int(key in query_keys),
                -_phrase_token_count(key),
                len(cohorts[key]),
                key,
            )
        )
        del usable_keys[policy.max_phrases_per_node :]

        shared_by_target: dict[str, list[tuple[str, float]]] = {}
        for key in usable_keys:
            cohort = cohorts[key]
            posting_count = len(cohort)
            weight = _phrase_weight(
                key,
                posting_count,
                query_match=key in query_keys,
            )
            for target_id in sorted(cohort):
                if target_id == chunk_id:
                    continue
                target = self._chunks[target_id]
                if (
                    target.source_id == source.source_id
                    and target_id in self._sequence_neighbors[chunk_id]
                ):
                    continue
                shared_by_target.setdefault(target_id, []).append((key, weight))

        transitions: list[GraphTransition] = []
        for target_id, matches in shared_by_target.items():
            matches.sort(key=lambda item: (-item[1], item[0]))
            best_key, best_weight = matches[0]
            transitions.append(
                GraphTransition(
                    source_chunk_id=chunk_id,
                    target_chunk_id=target_id,
                    relation="shared_phrase",
                    weight=best_weight,
                    shared_phrases=tuple(key for key, _weight in matches[:3]),
                    source_occurrence=self._first_occurrence(chunk_id, best_key),
                    target_occurrence=self._first_occurrence(target_id, best_key),
                )
            )
        return transitions

    def _transitions_unlocked(
        self,
        chunk_id: str,
        query_keys: frozenset[str],
        policy: GraphTraversalPolicy,
    ) -> tuple[GraphTransition, ...]:
        # Physical adjacency has its own fixed degree (at most two) and cannot
        # be crowded out by a phrase hub.  ``max_degree`` bounds phrase jumps.
        sequence = self._sequence_transitions(chunk_id, policy)
        shared = self._shared_phrase_transitions(chunk_id, query_keys, policy)
        shared.sort(
            key=lambda edge: (
                -edge.weight,
                edge.target_chunk_id,
                edge.shared_phrases,
            )
        )
        return tuple((*sequence, *shared[: policy.max_degree]))

    def transitions(
        self,
        chunk_id: str,
        *,
        question: str = "",
        policy: GraphTraversalPolicy | None = None,
    ) -> tuple[GraphTransition, ...]:
        """Expose the deterministic bounded neighborhood of a physical chunk."""

        selected_policy = policy or GraphTraversalPolicy()
        query_keys = frozenset(
            span.key
            for span in extract_canonical_phrases(
                str(question),
                policy=self._extraction_policy,
            )
        )
        with self._lock:
            normalized = str(chunk_id)
            if normalized not in self._chunks:
                raise KeyError(f"unknown graph chunk: {normalized}")
            return self._transitions_unlocked(
                normalized,
                query_keys,
                selected_policy,
            )

    def _question_seeds(
        self,
        query_keys: tuple[str, ...],
        policy: GraphTraversalPolicy,
    ) -> dict[str, _MutableHit]:
        cohorts = {
            key: self._phrase_cohort(key, policy.max_phrase_postings)
            for key in set(query_keys)
            if key in self._postings
        }
        usable = list(cohorts)
        usable.sort(
            key=lambda key: (
                -_phrase_token_count(key),
                len(cohorts[key]),
                key,
            )
        )
        del usable[policy.max_query_phrases :]
        seeds: dict[str, _MutableHit] = {}
        for key in usable:
            cohort = cohorts[key]
            posting_count = len(cohort)
            evidence = _phrase_weight(key, posting_count, query_match=True)
            for chunk_id in sorted(cohort):
                state = seeds.get(chunk_id)
                if state is None:
                    state = _MutableHit(
                        score=evidence,
                        hop=0,
                        best_contribution=evidence,
                        path=(),
                        seed_chunk_ids={chunk_id},
                        query_phrases={key},
                    )
                    seeds[chunk_id] = state
                else:
                    state.score = _noisy_or(state.score, evidence)
                    state.best_contribution = max(
                        state.best_contribution,
                        evidence,
                    )
                    state.query_phrases.add(key)
        return seeds

    @staticmethod
    def _rank_states(
        states: dict[str, _MutableHit],
    ) -> list[tuple[str, _MutableHit]]:
        return sorted(
            states.items(),
            key=lambda item: (-item[1].score, item[1].hop, item[0]),
        )

    def search(
        self,
        question: str,
        *,
        seed_chunk_ids: Sequence[str] = (),
        derive_question_seeds: bool = True,
        policy: GraphTraversalPolicy | None = None,
    ) -> ConversationGraphSearchResult:
        """Retrieve raw chunks through a deterministic, at-most-two-hop walk.

        ``seed_chunk_ids`` lets an existing lexical/dense lane activate the
        graph.  Exact question phrases independently provide seeds, so the core
        can also be assayed on its own.
        """

        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if type(derive_question_seeds) is not bool:
            raise TypeError("derive_question_seeds must be a bool")
        selected_policy = policy or GraphTraversalPolicy()
        query_spans = extract_canonical_phrases(
            question,
            policy=self._extraction_policy,
        )
        query_keys = tuple(dict.fromkeys(span.key for span in query_spans))
        explicit = tuple(dict.fromkeys(str(value) for value in seed_chunk_ids))

        with self._lock:
            unknown = [
                chunk_id for chunk_id in explicit if chunk_id not in self._chunks
            ]
            if unknown:
                raise KeyError(f"unknown graph seed chunks: {unknown}")
            if len(explicit) > selected_policy.max_seed_chunks:
                raise ValueError("explicit graph seeds exceed max_seed_chunks")
            seeds = (
                self._question_seeds(query_keys, selected_policy)
                if derive_question_seeds
                else {}
            )
            query_key_set = frozenset(query_keys)
            for chunk_id in explicit:
                matched = query_key_set.intersection(
                    self._chunk_phrase_keys[chunk_id]
                )
                state = seeds.get(chunk_id)
                if state is None:
                    seeds[chunk_id] = _MutableHit(
                        score=1.0,
                        hop=0,
                        best_contribution=1.0,
                        path=(),
                        seed_chunk_ids={chunk_id},
                        query_phrases=set(matched),
                    )
                else:
                    state.score = 1.0
                    state.best_contribution = 1.0
                    # This path is the physical seed itself; do not retain
                    # aggregate provenance from any other activation route.
                    state.seed_chunk_ids = {chunk_id}
                    state.query_phrases.update(matched)

            explicit_pairs = [(chunk_id, seeds[chunk_id]) for chunk_id in explicit]
            explicit_set = set(explicit)
            derived_pairs = self._rank_states(
                {
                    chunk_id: state
                    for chunk_id, state in seeds.items()
                    if chunk_id not in explicit_set
                }
            )
            derived_pairs = derived_pairs[
                : selected_policy.max_seed_chunks - len(explicit_pairs)
            ]
            ranked_seeds = [*explicit_pairs, *derived_pairs]
            frontier = dict(ranked_seeds)
            states = dict(ranked_seeds)
            frozen_query_keys = query_key_set

            for hop in range(1, selected_policy.max_hops + 1):
                next_frontier: dict[str, _MutableHit] = {}
                for parent_id, parent in self._rank_states(frontier):
                    path_nodes = {parent_id}
                    for prior in parent.path:
                        path_nodes.add(prior.source_chunk_id)
                        path_nodes.add(prior.target_chunk_id)
                    for edge in self._transitions_unlocked(
                        parent_id,
                        frozen_query_keys,
                        selected_policy,
                    ):
                        target_id = edge.target_chunk_id
                        if target_id in path_nodes:
                            continue
                        contribution = min(
                            1.0,
                            parent.score * edge.weight * selected_policy.hop_decay,
                        )
                        if contribution <= 0.0:
                            continue
                        candidate_path = parent.path + (edge,)
                        state = next_frontier.get(target_id)
                        candidate_signature = tuple(
                            (step.relation, step.target_chunk_id, step.shared_phrases)
                            for step in candidate_path
                        )
                        existing_signature = (
                            ()
                            if state is None
                            else tuple(
                                (
                                    step.relation,
                                    step.target_chunk_id,
                                    step.shared_phrases,
                                )
                                for step in state.path
                            )
                        )
                        if state is None or (
                            contribution > state.best_contribution
                            or (
                                contribution == state.best_contribution
                                and candidate_signature < existing_signature
                            )
                        ):
                            next_frontier[target_id] = _MutableHit(
                                score=contribution,
                                hop=hop,
                                best_contribution=contribution,
                                path=candidate_path,
                                seed_chunk_ids=set(parent.seed_chunk_ids),
                                query_phrases=set(parent.query_phrases),
                            )
                ranked_next = self._rank_states(next_frontier)[
                    : selected_policy.max_frontier
                ]
                frontier = dict(ranked_next)
                if not frontier:
                    break
                for chunk_id, candidate in frontier.items():
                    existing = states.get(chunk_id)
                    if existing is None:
                        states[chunk_id] = candidate
                        continue
                    candidate_signature = tuple(
                        (step.relation, step.target_chunk_id, step.shared_phrases)
                        for step in candidate.path
                    )
                    existing_signature = tuple(
                        (step.relation, step.target_chunk_id, step.shared_phrases)
                        for step in existing.path
                    )
                    if (
                        candidate.score > existing.score
                        or (
                            candidate.score == existing.score
                            and (
                                candidate.hop < existing.hop
                                or (
                                    candidate.hop == existing.hop
                                    and candidate_signature < existing_signature
                                )
                            )
                        )
                    ):
                        states[chunk_id] = candidate

            ranked_all = self._rank_states(states)
            # Retain only the top max_results non-explicit rows, while every
            # explicit seed remains visible without consuming that budget.
            ranked_novel = [
                pair for pair in ranked_all if pair[0] not in explicit_set
            ][: selected_policy.max_results]
            retained_ids = explicit_set | {pair[0] for pair in ranked_novel}
            ranked = [pair for pair in ranked_all if pair[0] in retained_ids]
            evidence = tuple(
                GraphEvidence(
                    chunk=self._chunks[chunk_id],
                    score=state.score,
                    hop=state.hop,
                    supporting_seed_chunk_ids=tuple(sorted(state.seed_chunk_ids)),
                    matched_query_phrases=tuple(sorted(state.query_phrases)),
                    path=state.path,
                )
                for chunk_id, state in ranked
            )
            return ConversationGraphSearchResult(
                question=question,
                query_phrase_keys=query_keys,
                seed_chunk_ids=tuple(chunk_id for chunk_id, _state in ranked_seeds),
                evidence=evidence,
                graph_revision=self._revision,
                traversal_policy_sha256=selected_policy.policy_sha256,
                question_seed_derivation_enabled=derive_question_seeds,
            )

    def _story_pair_affinity_unlocked(
        self,
        left_source_id: str,
        right_source_id: str,
        candidate_source_ids: frozenset[str],
        policy: OrderedStorySearchPolicy,
    ) -> StoryPairAffinity:
        left_source_id, right_source_id = sorted(
            (left_source_id, right_source_id)
        )
        left_terms = self._story_source_terms.get(left_source_id, {})
        right_terms = self._story_source_terms.get(right_source_id, {})
        if len(left_terms) > len(right_terms):
            shared = (term for term in right_terms if term in left_terms)
        else:
            shared = (term for term in left_terms if term in right_terms)
        maximum_fanout = max(
            3,
            math.ceil(len(candidate_source_ids) * policy.max_source_fraction),
        )
        matches: list[StoryTermEvidence] = []
        for term in shared:
            candidate_fanout = sum(
                source_id in candidate_source_ids
                for source_id in self._story_term_sources[term]
            )
            if not 2 <= candidate_fanout <= maximum_fanout:
                continue
            weight = math.log(
                (len(candidate_source_ids) + 1) / candidate_fanout
            ) + 1.0
            matches.append(
                StoryTermEvidence(
                    term=term,
                    candidate_source_count=candidate_fanout,
                    weight=round(weight, 8),
                    left_occurrence=left_terms[term],
                    right_occurrence=right_terms[term],
                )
            )
        matches.sort(key=lambda row: (-row.weight, -len(row.term), row.term))
        del matches[policy.max_terms_per_pair :]
        return StoryPairAffinity(
            left_source_id=left_source_id,
            right_source_id=right_source_id,
            score=round(sum(row.weight for row in matches), 8),
            terms=tuple(matches),
        )

    def _story_neighbor_sources_unlocked(
        self,
        seed_source_ids: frozenset[str],
        policy: OrderedStorySearchPolicy,
    ) -> tuple[str, ...]:
        """Rank bounded global source neighbors without scanning source rows."""

        source_count = len(self._story_source_terms)
        maximum_fanout = max(
            3,
            math.ceil(source_count * policy.max_source_fraction),
        )
        scores: dict[str, float] = {}
        for seed_source_id in sorted(seed_source_ids):
            for term in self._story_source_terms.get(seed_source_id, {}):
                cohort = self._story_term_sources[term]
                if not 2 <= len(cohort) <= maximum_fanout:
                    continue
                weight = math.log((source_count + 1) / len(cohort)) + 1.0
                for target_source_id in cohort:
                    if target_source_id in seed_source_ids:
                        continue
                    scores[target_source_id] = (
                        scores.get(target_source_id, 0.0) + weight
                    )
        ranked = sorted(scores, key=lambda source_id: (-scores[source_id], source_id))
        return tuple(ranked[: policy.max_neighbor_sources])

    def _story_evidence_chunks_unlocked(
        self,
        source_id: str,
        representative_chunk_id: str,
        limit: int,
    ) -> tuple[ConversationGraphChunk, ...]:
        retained = list(self._story_source_user_chunks.get(source_id, ()))
        retained.sort(
            key=lambda chunk_id: (
                self._chunks[chunk_id].ordinal,
                self._chunks[chunk_id].start_char,
                chunk_id,
            )
        )
        ordered = [representative_chunk_id]
        ordered.extend(
            chunk_id
            for chunk_id in retained
            if chunk_id != representative_chunk_id
        )
        return tuple(self._chunks[chunk_id] for chunk_id in ordered[:limit])

    def search_ordered_story(
        self,
        *,
        seed_chunk_ids: Sequence[str],
        requested_count: int,
        candidate_chunk_ids: Sequence[str] = (),
        policy: OrderedStorySearchPolicy | None = None,
    ) -> OrderedStorySearchResult:
        """Select an exact-k, chronologically ordered, content-linked story.

        Seeds and optional candidates are physical IDs supplied by an upstream
        authenticated retrieval.  This method never derives seeds from a
        question or source-name prefix.  When candidates are omitted, only a
        bounded rare-term neighborhood of the explicit seed sources is used.
        """

        selected_policy = policy or OrderedStorySearchPolicy()
        count = _exact_positive_int(requested_count, "requested_count")
        if count > selected_policy.max_bundle_members:
            raise ValueError("requested story count exceeds max_bundle_members")
        seeds = tuple(str(value) for value in seed_chunk_ids)
        candidates = tuple(str(value) for value in candidate_chunk_ids)
        if not seeds:
            raise ValueError("ordered story search requires explicit seed chunks")
        if any(not value or value != value.strip() for value in (*seeds, *candidates)):
            raise ValueError("story chunk IDs must be non-empty exact text")
        if len(seeds) != len(set(seeds)):
            raise ValueError("story seed chunks must be ordered unique")
        if len(candidates) != len(set(candidates)):
            raise ValueError("story candidate chunks must be ordered unique")
        if len(seeds) > selected_policy.max_seed_chunks:
            raise ValueError("story seed population exceeds max_seed_chunks")
        if len(candidates) > selected_policy.max_candidate_chunks:
            raise ValueError("story candidate population exceeds max_candidate_chunks")

        with self._lock:
            unknown = [
                chunk_id
                for chunk_id in (*seeds, *candidates)
                if chunk_id not in self._chunks
            ]
            if unknown:
                raise KeyError(f"unknown ordered story chunks: {unknown}")
            if candidates and not _is_ordered_subsequence(seeds, candidates):
                raise ValueError(
                    "story seeds must be an ordered subset of explicit candidates"
                )

            seed_source_ids = frozenset(
                self._chunks[chunk_id].source_id for chunk_id in seeds
            )
            derivation: Literal[
                "explicit_candidates", "seed_source_expansion"
            ]
            candidate_rows: list[_StoryCandidate] = []
            if candidates:
                derivation = "explicit_candidates"
                source_seen: set[str] = set()
                for rank, chunk_id in enumerate(candidates):
                    chunk = self._chunks[chunk_id]
                    event_time = _parse_story_time(chunk.created_at)
                    if (
                        chunk.role != "user"
                        or event_time is None
                        or chunk.source_id in source_seen
                    ):
                        continue
                    source_seen.add(chunk.source_id)
                    candidate_rows.append(
                        _StoryCandidate(
                            source_id=chunk.source_id,
                            representative_chunk_id=chunk_id,
                            event_time=event_time,
                            event_date=event_time.date().isoformat(),
                            rank=rank,
                        )
                    )
            else:
                derivation = "seed_source_expansion"
                neighbors = self._story_neighbor_sources_unlocked(
                    seed_source_ids,
                    selected_policy,
                )
                source_order = tuple(
                    dict.fromkeys(
                        (
                            *(self._chunks[chunk_id].source_id for chunk_id in seeds),
                            *neighbors,
                        )
                    )
                )[: selected_policy.max_candidate_sources]
                seed_by_source: dict[str, list[str]] = {}
                for chunk_id in seeds:
                    seed_by_source.setdefault(
                        self._chunks[chunk_id].source_id,
                        [],
                    ).append(chunk_id)
                for rank, source_id in enumerate(source_order):
                    source_options = [
                        *seed_by_source.get(source_id, ()),
                        *self._story_source_user_chunks.get(source_id, ()),
                    ]
                    seen_options: set[str] = set()
                    representative: ConversationGraphChunk | None = None
                    event_time: datetime | None = None
                    for chunk_id in source_options:
                        if chunk_id in seen_options:
                            continue
                        seen_options.add(chunk_id)
                        chunk = self._chunks[chunk_id]
                        parsed = _parse_story_time(chunk.created_at)
                        if chunk.role == "user" and parsed is not None:
                            representative = chunk
                            event_time = parsed
                            break
                    if representative is None or event_time is None:
                        continue
                    candidate_rows.append(
                        _StoryCandidate(
                            source_id=source_id,
                            representative_chunk_id=representative.chunk_id,
                            event_time=event_time,
                            event_date=event_time.date().isoformat(),
                            rank=rank,
                        )
                    )

            if len(candidate_rows) > selected_policy.max_candidate_sources:
                raise ValueError(
                    "story source population exceeds max_candidate_sources"
                )
            candidate_source_ids = tuple(row.source_id for row in candidate_rows)
            candidate_source_set = frozenset(candidate_source_ids)

            pair_affinities = {
                tuple(sorted((left.source_id, right.source_id))): (
                    self._story_pair_affinity_unlocked(
                        left.source_id,
                        right.source_id,
                        candidate_source_set,
                        selected_policy,
                    )
                )
                for left, right in combinations(candidate_rows, 2)
            }

            def pair(
                left: _StoryCandidate,
                right: _StoryCandidate,
            ) -> StoryPairAffinity:
                key = tuple(sorted((left.source_id, right.source_id)))
                return pair_affinities[key]

            def connected(group: tuple[_StoryCandidate, ...]) -> bool:
                if len(group) <= 1:
                    return True
                reached = {group[0].source_id}
                while True:
                    expanded = {
                        row.source_id
                        for row in group
                        if row.source_id not in reached
                        and any(
                            pair(row, prior).score > 0.0
                            for prior in group
                            if prior.source_id in reached
                        )
                    }
                    if not expanded:
                        break
                    reached.update(expanded)
                return len(reached) == len(group)

            def group_key(group: tuple[_StoryCandidate, ...]) -> tuple[object, ...]:
                values = tuple(
                    pair(left, right).score
                    for left, right in combinations(group, 2)
                )
                positive = tuple(value for value in values if value > 0.0)
                complete_pair_count = len(group) * (len(group) - 1) // 2
                return (
                    int(len(positive) == complete_pair_count),
                    len(positive),
                    min(positive, default=0.0),
                    sum(positive),
                    -sum(row.rank for row in group),
                    tuple(sorted(row.source_id for row in group)),
                )

            selected_group: tuple[_StoryCandidate, ...] = ()
            status: OrderedStoryStatus
            if len(candidate_rows) < count:
                status = "insufficient_distinct_dated_sources"
            else:
                total_combinations = math.comb(len(candidate_rows), count)
                if total_combinations <= selected_policy.max_combinations:
                    groups = tuple(combinations(candidate_rows, count))
                else:
                    grown: list[tuple[_StoryCandidate, ...]] = []
                    for anchor in candidate_rows:
                        group = [anchor]
                        while len(group) < count:
                            used = {row.source_id for row in group}
                            dates = {row.event_date for row in group}
                            choices = [
                                row
                                for row in candidate_rows
                                if row.source_id not in used
                                and row.event_date not in dates
                            ]
                            if not choices:
                                break
                            addition = max(
                                choices,
                                key=lambda row: (
                                    min(
                                        (
                                            pair(row, prior).score
                                            for prior in group
                                        ),
                                        default=0.0,
                                    ),
                                    sum(pair(row, prior).score for prior in group),
                                    -row.rank,
                                    row.source_id,
                                ),
                            )
                            group.append(addition)
                        if len(group) == count:
                            grown.append(tuple(group))
                    groups = tuple(grown)
                eligible = tuple(
                    group
                    for group in groups
                    if len({row.event_date for row in group}) == count
                    and any(row.source_id in seed_source_ids for row in group)
                    and connected(group)
                )
                if eligible:
                    selected_group = max(eligible, key=group_key)
                    status = "selected"
                else:
                    distinct_dates = {row.event_date for row in candidate_rows}
                    status = (
                        "insufficient_distinct_dated_sources"
                        if len(distinct_dates) < count
                        else "no_connected_story"
                    )

            ordered_group = tuple(
                sorted(
                    selected_group,
                    key=lambda row: (
                        row.event_time,
                        self._chunks[row.representative_chunk_id].ordinal,
                        self._chunks[row.representative_chunk_id].start_char,
                        row.source_id,
                    ),
                )
            )
            selected_sources = tuple(
                OrderedStorySource(
                    source_id=row.source_id,
                    representative_chunk=self._chunks[
                        row.representative_chunk_id
                    ],
                    evidence_chunks=self._story_evidence_chunks_unlocked(
                        row.source_id,
                        row.representative_chunk_id,
                        selected_policy.max_evidence_chunks_per_source,
                    ),
                    event_time_utc=row.event_time.isoformat(),
                    candidate_rank=row.rank,
                )
                for row in ordered_group
            )
            selected_pair_affinities = tuple(
                pair(left, right)
                for left, right in combinations(
                    tuple(sorted(selected_group, key=lambda row: row.source_id)),
                    2,
                )
            )
            receipt_projection = {
                "schema": "incremental-ordered-source-story-result-v1",
                "status": status,
                "requested_count": count,
                "seed_chunk_ids": list(seeds),
                "explicit_candidate_chunk_ids": list(candidates),
                "candidate_source_ids": list(candidate_source_ids),
                "candidate_derivation": derivation,
                "selected_sources": [
                    {
                        "source_id": row.source_id,
                        "representative_chunk": (
                            row.representative_chunk.identity_payload()
                        ),
                        "evidence_chunk_ids": [
                            chunk.chunk_id for chunk in row.evidence_chunks
                        ],
                        "event_time_utc": row.event_time_utc,
                        "candidate_rank": row.candidate_rank,
                    }
                    for row in selected_sources
                ],
                "selected_pair_affinities": [
                    {
                        "left_source_id": row.left_source_id,
                        "right_source_id": row.right_source_id,
                        "score": row.score,
                        "terms": [
                            {
                                "term": term.term,
                                "candidate_source_count": (
                                    term.candidate_source_count
                                ),
                                "weight": term.weight,
                                "left_occurrence_id": (
                                    term.left_occurrence.occurrence_id
                                ),
                                "right_occurrence_id": (
                                    term.right_occurrence.occurrence_id
                                ),
                            }
                            for term in row.terms
                        ],
                    }
                    for row in selected_pair_affinities
                ],
                "graph_revision": self._revision,
                "story_index_policy_sha256": (
                    self._story_index_policy.policy_sha256
                ),
                "search_policy_sha256": selected_policy.policy_sha256,
            }
            return OrderedStorySearchResult(
                status=status,
                requested_count=count,
                seed_chunk_ids=seeds,
                explicit_candidate_chunk_ids=candidates,
                candidate_source_ids=candidate_source_ids,
                selected_sources=selected_sources,
                selected_pair_affinities=selected_pair_affinities,
                graph_revision=self._revision,
                story_index_policy_sha256=(
                    self._story_index_policy.policy_sha256
                ),
                search_policy_sha256=selected_policy.policy_sha256,
                candidate_derivation=derivation,
                receipt_sha256=identity_sha256(receipt_projection),
            )


__all__ = [
    "CanonicalPhraseSpan",
    "ConversationGraphAppendDelta",
    "ConversationGraphChunk",
    "ConversationGraphSearchResult",
    "ConversationGraphStats",
    "GraphAppendReceipt",
    "GraphAppendResult",
    "GraphEvidence",
    "GraphRelation",
    "GraphTransition",
    "GraphTraversalPolicy",
    "IncrementalConversationGraph",
    "OrderedStorySearchPolicy",
    "OrderedStorySearchResult",
    "OrderedStorySource",
    "OrderedStoryStatus",
    "PhraseExtractionPolicy",
    "PhraseOccurrence",
    "SequenceDirection",
    "StoryAffinityIndexPolicy",
    "StoryPairAffinity",
    "StoryTermMembership",
    "StoryTermEvidence",
    "extract_canonical_phrases",
]
