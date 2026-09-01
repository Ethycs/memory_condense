"""Gold-blind profile/preference retrieval over one immutable full-store index.

Generic recommendation questions are a poor fit for ordinary lexical search:
``recommend a show`` does not repeat the durable facts that the user is an
aspiring comedian who values storytelling.  This specialist scans the frozen
index without a question/source route, admits only user-authored profile,
preference, and recommendation-request sentences, then chooses one coherent
source/interest cluster.  Freshness and explicit identity are ranked before
uncommon domain/entity constraints; literal question-cue frequency is only a
late tie-breaker.

The result is intentionally expressed with the repository's existing exact
pointer contracts.  Raw source locators remain in local bindings, provider
rows carry only opaque group handles, and the module retains no transformer
tokens, embeddings, provider state, or mutable memory between prompt ticks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from memory_condense.domain.discourse import EvidenceSpan, quote_sha256

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .full_store_slot_closure import (
    FullStoreSlotCandidate,
    FullStoreWindowIndex,
    LocalCitationBinding,
    indexed_surface_terms,
)
from .typed_operator_spec import TypedOperatorSpec, compile_typed_operator_spec
from .typed_story_affinity import evidence_source_story_key_sha256


MECHANISM_ID = "profile_preference_specialist_v1"
BUDGET_FORMAT = "memory-condense-profile-preference-budget-v1"
AUDIT_FORMAT = "memory-condense-profile-preference-audit-v1"
RESULT_FORMAT = "memory-condense-profile-preference-result-v1"
CANDIDATE_ID_FORMAT = "memory-condense-profile-preference-index-candidate-v1"
RANKING_POLICY = (
    "one_exact_source_interest_cluster;intrinsic_profile_then_freshness_then_"
    "explicit_identity_then_"
    "uncommon_domain_entity_constraints_then_profile_signal;question_cue_"
    "frequency_is_late_tiebreak;future_rows_rejected"
)

SelectionStatus = Literal[
    "selected",
    "no_profile_evidence",
    "not_recommendation_or_preference",
    "unsupported_query_domain",
]


class ProfilePreferenceSpecialistError(MatchedEvalContractError):
    """Raised when specialist input, provenance, or a hard cap changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ProfilePreferenceSpecialistError(message)


def _ordered_unique(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    _require(
        type(values) is tuple
        and all(type(value) is str and value for value in values)
        and len(values) == len(set(values)),
        f"{label} must be an ordered unique exact tuple",
    )
    return values


@dataclass(frozen=True, slots=True)
class ProfilePreferenceBudget:
    """Hard output and per-cluster consideration limits."""

    max_selected_candidates: int = 6
    max_selected_tokens: int = 768
    max_windows_per_cluster: int = 12

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"{name} must be positive")
        _require(
            self.max_selected_candidates <= self.max_windows_per_cluster,
            "selected-candidate cap exceeds the per-cluster consideration cap",
        )

    def projection(self) -> dict[str, int | str]:
        return {"format": BUDGET_FORMAT, **asdict(self)}

    @property
    def budget_id(self) -> str:
        return identity_sha256(
            {
                "budget": self.projection(),
                "mechanism_id": MECHANISM_ID,
                "ranking_policy": RANKING_POLICY,
            }
        )


@dataclass(frozen=True, slots=True)
class ProfilePreferenceAudit:
    """Sealed prompt-external proof of the gold-blind bounded selection."""

    question_sha256: str
    operator_spec_receipt_sha256: str
    index_receipt_sha256: str
    cache_receipt_sha256: str
    budget_id: str
    status: SelectionStatus
    recognized_domain_ids: tuple[str, ...]
    physical_sentence_window_count: int
    physical_sentence_windows_scanned: int
    user_sentence_window_count: int
    eligible_candidate_count: int
    eligible_cluster_count: int
    role_rejected_window_count: int
    future_rejected_window_count: int
    candidate_population_sha256: str
    cluster_ranking_sha256: str
    selected_cluster_key_sha256: str | None
    selected_cluster_rank_projection_sha256: str | None
    selected_candidate_ids: tuple[str, ...]
    selected_local_binding_receipt_sha256s: tuple[str, ...]
    selected_evidence_tokens: int
    selection_truncated: bool
    max_selected_candidates: int
    max_selected_tokens: int
    max_windows_per_cluster: int
    ranking_policy: str = RANKING_POLICY
    physical_index_scan_exhaustive: Literal[True] = True
    selected_source_cluster_count: int = 0
    question_id_filter_used: Literal[False] = False
    known_source_filter_used: Literal[False] = False
    partition_routing_used: Literal[False] = False
    raw_source_ids_provider_visible: Literal[False] = False
    new_provider_calls: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0
    gold_loaded: Literal[False] = False
    semantic_completeness_status: Literal["not_claimed"] = "not_claimed"
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.question_sha256, "profile question"),
            (self.operator_spec_receipt_sha256, "profile operator spec"),
            (self.index_receipt_sha256, "profile index"),
            (self.cache_receipt_sha256, "profile cache"),
            (self.budget_id, "profile budget"),
            (self.candidate_population_sha256, "profile candidate population"),
            (self.cluster_ranking_sha256, "profile cluster ranking"),
        ):
            require_sha256(value, label)
        if self.selected_cluster_key_sha256 is not None:
            require_sha256(self.selected_cluster_key_sha256, "selected cluster")
            require_sha256(
                self.selected_cluster_rank_projection_sha256 or "",
                "selected cluster rank projection",
            )
        else:
            _require(
                self.selected_cluster_rank_projection_sha256 is None,
                "empty selection retained a cluster-rank receipt",
            )
        _ordered_unique(self.recognized_domain_ids, "recognized domains")
        _ordered_unique(self.selected_candidate_ids, "selected candidates")
        _ordered_unique(
            self.selected_local_binding_receipt_sha256s,
            "selected local bindings",
        )
        for value in self.selected_local_binding_receipt_sha256s:
            require_sha256(value, "selected local binding")
        for name in (
            "physical_sentence_window_count",
            "physical_sentence_windows_scanned",
            "user_sentence_window_count",
            "eligible_candidate_count",
            "eligible_cluster_count",
            "role_rejected_window_count",
            "future_rejected_window_count",
            "selected_evidence_tokens",
            "max_selected_candidates",
            "max_selected_tokens",
            "max_windows_per_cluster",
            "selected_source_cluster_count",
        ):
            value = getattr(self, name)
            _require(type(value) is int and value >= 0, f"{name} changed")
        _require(
            self.status
            in {
                "selected",
                "no_profile_evidence",
                "not_recommendation_or_preference",
                "unsupported_query_domain",
            },
            "profile selection status changed",
        )
        _require(
            self.physical_sentence_windows_scanned
            == self.physical_sentence_window_count
            and self.user_sentence_window_count
            + self.role_rejected_window_count
            == self.physical_sentence_window_count,
            "profile physical scan counts changed",
        )
        _require(
            len(self.selected_candidate_ids)
            == len(self.selected_local_binding_receipt_sha256s)
            <= self.max_selected_candidates
            and self.selected_evidence_tokens <= self.max_selected_tokens,
            "profile output escaped its hard cap",
        )
        has_selection = bool(self.selected_candidate_ids)
        _require(
            has_selection is (self.status == "selected")
            and self.selected_source_cluster_count == (1 if has_selection else 0)
            and (self.selected_cluster_key_sha256 is not None) is has_selection,
            "profile status and selected cluster disagree",
        )
        _require(
            self.ranking_policy == RANKING_POLICY
            and self.physical_index_scan_exhaustive is True
            and self.question_id_filter_used is False
            and self.known_source_filter_used is False
            and self.partition_routing_used is False
            and self.raw_source_ids_provider_visible is False
            and self.new_provider_calls == 0
            and self.retained_transformer_token_state_bytes == 0
            and self.gold_loaded is False
            and self.semantic_completeness_status == "not_claimed",
            "profile audit escaped its gold-blind zero-state boundary",
        )
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256:
            _require(self.receipt_sha256 == expected, "profile audit receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)
        assert_gold_blind(self.projection(), path="profile_preference_audit")

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "budget_id": self.budget_id,
            "cache_receipt_sha256": self.cache_receipt_sha256,
            "candidate_population_sha256": self.candidate_population_sha256,
            "cluster_ranking_sha256": self.cluster_ranking_sha256,
            "eligible_candidate_count": self.eligible_candidate_count,
            "eligible_cluster_count": self.eligible_cluster_count,
            "format": AUDIT_FORMAT,
            "future_rejected_window_count": self.future_rejected_window_count,
            "gold_loaded": False,
            "index_receipt_sha256": self.index_receipt_sha256,
            "known_source_filter_used": False,
            "max_selected_candidates": self.max_selected_candidates,
            "max_selected_tokens": self.max_selected_tokens,
            "max_windows_per_cluster": self.max_windows_per_cluster,
            "new_provider_calls": 0,
            "operator_spec_receipt_sha256": self.operator_spec_receipt_sha256,
            "partition_routing_used": False,
            "physical_index_scan_exhaustive": True,
            "physical_sentence_window_count": self.physical_sentence_window_count,
            "physical_sentence_windows_scanned": (
                self.physical_sentence_windows_scanned
            ),
            "question_id_filter_used": False,
            "question_sha256": self.question_sha256,
            "ranking_policy": self.ranking_policy,
            "raw_source_ids_provider_visible": False,
            "recognized_domain_ids": list(self.recognized_domain_ids),
            "retained_transformer_token_state_bytes": 0,
            "role_rejected_window_count": self.role_rejected_window_count,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_cluster_key_sha256": self.selected_cluster_key_sha256,
            "selected_cluster_rank_projection_sha256": (
                self.selected_cluster_rank_projection_sha256
            ),
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "selected_local_binding_receipt_sha256s": list(
                self.selected_local_binding_receipt_sha256s
            ),
            "selected_source_cluster_count": self.selected_source_cluster_count,
            "selection_truncated": self.selection_truncated,
            "semantic_completeness_status": self.semantic_completeness_status,
            "status": self.status,
            "user_sentence_window_count": self.user_sentence_window_count,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def provider_projection(self) -> dict[str, Any]:
        """Return the small provider-safe audit surface."""

        value = {
            "format": f"{AUDIT_FORMAT}-provider",
            "new_provider_calls": 0,
            "raw_source_ids_provider_visible": False,
            "receipt_sha256": self.receipt_sha256,
            "recognized_domain_ids": list(self.recognized_domain_ids),
            "retained_transformer_token_state_bytes": 0,
            "selected_candidate_count": len(self.selected_candidate_ids),
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "selection_truncated": self.selection_truncated,
            "semantic_completeness_status": "not_claimed",
            "status": self.status,
        }
        assert_gold_blind(value, path="profile_preference_provider_audit")
        return value


@dataclass(frozen=True, slots=True)
class ProfilePreferenceResult:
    """Bounded exact evidence plus local-only provenance."""

    dated_question: str
    operator_spec: TypedOperatorSpec
    budget: ProfilePreferenceBudget
    candidates: tuple[FullStoreSlotCandidate, ...]
    local_bindings: tuple[LocalCitationBinding, ...]
    audit: ProfilePreferenceAudit

    def __post_init__(self) -> None:
        require_text(self.dated_question, "profile dated question")
        _require(
            type(self.operator_spec) is TypedOperatorSpec
            and type(self.budget) is ProfilePreferenceBudget
            and type(self.audit) is ProfilePreferenceAudit,
            "profile result contract changed",
        )
        _require(
            type(self.candidates) is tuple
            and type(self.local_bindings) is tuple
            and all(type(row) is FullStoreSlotCandidate for row in self.candidates)
            and all(type(row) is LocalCitationBinding for row in self.local_bindings)
            and len(self.candidates) == len(self.local_bindings),
            "profile result evidence inventory changed",
        )
        for candidate, binding in zip(
            self.candidates, self.local_bindings, strict=True
        ):
            _require(
                candidate.candidate_id == binding.candidate_id
                and candidate.source_group_handle == binding.source_group_handle
                and candidate.quote_sha256 == binding.quote_sha256
                and candidate.citation_binding_receipt_sha256
                == binding.receipt_sha256,
                "profile candidate/local provenance changed",
            )
        _require(
            tuple(row.candidate_id for row in self.candidates)
            == self.audit.selected_candidate_ids
            and tuple(row.receipt_sha256 for row in self.local_bindings)
            == self.audit.selected_local_binding_receipt_sha256s
            and sum(row.token_count for row in self.candidates)
            == self.audit.selected_evidence_tokens
            and self.operator_spec.receipt_sha256
            == self.audit.operator_spec_receipt_sha256
            and self.budget.budget_id == self.audit.budget_id,
            "profile result disagrees with its sealed audit",
        )
        _require(
            len({row.source_id for row in self.local_bindings})
            <= 1,
            "profile result mixed source/interest clusters",
        )
        assert_gold_blind(self.local_projection(), path="profile_preference_result")

    @property
    def receipt_sha256(self) -> str:
        return self.audit.receipt_sha256

    def provider_projection(self) -> dict[str, Any]:
        value = {
            "audit": self.audit.provider_projection(),
            "candidates": [row.projection() for row in self.candidates],
            "format": RESULT_FORMAT,
            "mechanism_id": MECHANISM_ID,
        }
        assert_gold_blind(value, path="profile_preference_provider_result")
        return value

    def local_projection(self) -> dict[str, Any]:
        value = {
            "audit": self.audit.projection(),
            "budget": self.budget.projection(),
            "candidates": [row.projection() for row in self.candidates],
            "dated_question_sha256": quote_sha256(self.dated_question),
            "format": RESULT_FORMAT,
            "local_bindings": [row.projection() for row in self.local_bindings],
            "mechanism_id": MECHANISM_ID,
            "operator_spec": self.operator_spec.projection(),
        }
        assert_gold_blind(value, path="profile_preference_local_result")
        return value


_DATED_QUESTION_RE = re.compile(
    r"^\[Question asked at (?P<stamp>[^\]]+)\]\s*(?P<body>.+)$",
    re.DOTALL,
)
_WEEKDAY_RE = re.compile(r"\s*\([A-Za-z]{3,9}\)\s*")
_RECOMMENDATION_RE = re.compile(
    r"\b(?:recommend|recommendations?|suggest|suggestions?|what should i|"
    r"which .{0,40} should i|what (?:do|did) i (?:like|love|enjoy|prefer)|"
    r"my favou?rite|based on my (?:taste|preference|history))\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.IGNORECASE
)
_IDENTITY_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bas an? aspir(?:ing|ant)\b", re.IGNORECASE), 5),
    (
        re.compile(
            r"\b(?:i am|i'm) (?:an? )?aspir(?:ing|ant)\b", re.IGNORECASE
        ),
        5,
    ),
    (
        re.compile(
            r"\b(?:i work as|my (?:career|profession|craft|vocation)|"
            r"i'm (?:an?|the) (?:writer|artist|musician|comedian|filmmaker|"
            r"developer|designer|athlete|student|teacher|chef))\b",
            re.IGNORECASE,
        ),
        4,
    ),
    (
        re.compile(
            r"\b(?:i identify as|i consider myself|i'm passionate about|"
            r"my long[- ]term goal)\b",
            re.IGNORECASE,
        ),
        3,
    ),
)
_PREFERENCE_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"\b(?:my favou?rite|i (?:really )?(?:prefer|love|adore)|"
            r"i'm a fan of)\b",
            re.IGNORECASE,
        ),
        5,
    ),
    (
        re.compile(
            r"\b(?:i (?:really )?(?:like|enjoy|appreciate)|"
            r"i'm (?:interested in|looking for)|i've been looking for)\b",
            re.IGNORECASE,
        ),
        4,
    ),
    (
        re.compile(
            r"\b(?:i (?:want|need|hope|plan)|i'm (?:thinking|trying|planning)|"
            r"i've been (?:thinking|trying|watching|reading|listening))\b",
            re.IGNORECASE,
        ),
        3,
    ),
)
_REQUEST_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"\b(?:can|could|would|will) you (?:please )?"
            r"(?:recommend|suggest)\b",
            re.IGNORECASE,
        ),
        4,
    ),
    (
        re.compile(
            r"\b(?:recommendations?|recommend|suggestions?|suggest)\b",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        re.compile(
            r"\b(?:do you have any (?:tips|advice|ideas)|can you help me)\b",
            re.IGNORECASE,
        ),
        2,
    ),
)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?\b")
_QUOTED_RE = re.compile(
    r'"(?P<double>[^"\n]{2,80})"|(?<!\w)\'(?P<single>[^\'\n]{2,80})\'(?!\w)'
)
_ENTITY_STOP = frozenset(
    indexed_surface_terms(
        "I I'm I've My As Can Could Would Will What Which When Where Why How "
        "The This That These Those Do Does Did Please By And But Or On In At"
    )
)


def _terms(value: str) -> frozenset[str]:
    return frozenset(indexed_surface_terms(value))


_DOMAIN_QUERY_TERMS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "beverages": _terms(
            "beverage beverages drink drinks cocktail cocktails mocktail mocktails "
            "mixology bartending gin vodka rum tequila whiskey whisky bourbon wine "
            "beer cider liquor liqueur spirits punch martini margarita mojito"
        ),
        "books": _terms("book books novel novels author reading read literature"),
        "food": _terms("food eat restaurant recipe recipes cooking meal dinner lunch"),
        "games": _terms("game games gaming video-game board-game play"),
        "home": _terms("home furniture decor lamp lighting appliance room garden"),
        "music": _terms("music song songs album albums band artist listen concert"),
        "screen": _terms(
            "show shows movie movies film films watch television tv series "
            "stream streaming documentary documentaries episode cinema"
        ),
        "style": _terms("clothes clothing wear outfit style shoes fashion"),
        "technology": _terms(
            "laptop computer phone software app gadget headphones camera keyboard"
        ),
        "travel": _terms(
            "travel trip vacation visit destination hotel campsite flight itinerary"
        ),
        "wellness": _terms(
            "fitness workout exercise running gym health meditation sport training"
        ),
    }
)
_DOMAIN_EVIDENCE_TERMS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "beverages": _terms(
            "beverage beverages drink drinks cocktail cocktails mocktail mocktails "
            "mixology mixologist bartending bartender gin vodka rum tequila whiskey "
            "whisky bourbon wine beer cider liquor liqueur spirits punch martini "
            "margarita mojito negroni daiquiri highball collins tonic bitters"
        ),
        "books": _terms(
            "book novel author reading read literature fiction nonfiction memoir "
            "story prose poetry mystery fantasy biography"
        ),
        "food": _terms(
            "food eat restaurant recipe cooking meal dinner lunch cuisine dish chef "
            "vegan vegetarian spicy flavor"
        ),
        "games": _terms(
            "game gaming videogame board-game play console puzzle strategy rpg steam"
        ),
        "home": _terms(
            "home furniture decor lamp lighting appliance room garden sofa desk "
            "interior bedroom kitchen"
        ),
        "music": _terms(
            "music song album band artist listen concert singer guitar jazz rock pop "
            "classical hip-hop playlist"
        ),
        "screen": _terms(
            "show movie film watch television tv series stream streaming netflix hulu "
            "documentary episode cinema actor actress director genre thriller sci-fi "
            "anime comedy stand-up comedian special storytelling sitcom drama"
        ),
        "style": _terms(
            "clothes clothing wear outfit style shoes fashion color fit fabric jacket"
        ),
        "technology": _terms(
            "laptop computer phone software app gadget headphones camera keyboard "
            "portable lightweight battery screen processor android ios windows mac"
        ),
        "travel": _terms(
            "travel trip vacation visit destination hotel campsite flight itinerary "
            "beach mountain city museum hiking sightseeing"
        ),
        "wellness": _terms(
            "fitness workout exercise running gym health meditation sport training "
            "swimming yoga strength cardio"
        ),
    }
)


def _question_parts(dated_question: str) -> tuple[datetime, str]:
    require_text(dated_question, "profile dated question")
    match = _DATED_QUESTION_RE.fullmatch(dated_question)
    _require(match is not None, "profile question must carry an exact asked-at header")
    stamp = _WEEKDAY_RE.sub(" ", match.group("stamp")).strip()
    parsed: datetime | None = None
    for pattern in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(stamp, pattern)
            break
        except ValueError:
            continue
    _require(parsed is not None, "profile question asked-at timestamp changed")
    body = match.group("body").strip()
    require_text(body, "profile question body")
    return parsed.replace(tzinfo=timezone.utc), body


def _row_datetime(value: str) -> datetime:
    require_text(value, "profile evidence created-at")
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise ProfilePreferenceSpecialistError(
            "profile evidence timestamp changed"
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _maximum_pattern_score(
    value: str, patterns: Sequence[tuple[re.Pattern[str], int]]
) -> int:
    return max((score for pattern, score in patterns if pattern.search(value)), default=0)


def _entity_terms(value: str) -> frozenset[str]:
    surfaces = [match.group(0) for match in _ENTITY_RE.finditer(value)]
    surfaces.extend(
        match.group("double") or match.group("single")
        for match in _QUOTED_RE.finditer(value)
    )
    return frozenset(
        term
        for surface in surfaces
        for term in indexed_surface_terms(surface)
        if term not in _ENTITY_STOP and len(term) >= 3
    )


def _recognized_domains(body_terms: frozenset[str]) -> tuple[str, ...]:
    scored = {
        domain: len(body_terms & triggers)
        for domain, triggers in _DOMAIN_QUERY_TERMS.items()
    }
    maximum = max(scored.values(), default=0)
    if maximum == 0:
        return ()
    return tuple(sorted(domain for domain, score in scored.items() if score == maximum))


def _rarity_weight(index: FullStoreWindowIndex, term: str) -> int:
    frequency = len(index.term_postings.get(term, ()))
    ratio = max(1, (len(index.windows) + 1) // (frequency + 1))
    return min(32, 1 + ratio.bit_length())


def _recency_bucket(age_days: int) -> int:
    if age_days <= 3:
        return 6
    if age_days <= 7:
        return 5
    if age_days <= 30:
        return 4
    if age_days <= 90:
        return 3
    if age_days <= 365:
        return 2
    return 1


@dataclass(frozen=True, slots=True)
class _ScoredWindow:
    window_index: int
    cluster_key_sha256: str
    age_days: int
    recency: int
    identity_strength: int
    preference_strength: int
    request_strength: int
    uncommon_constraint_score: int
    domain_match_count: int
    question_overlap_count: int
    constraint_terms: tuple[str, ...]
    created_epoch_minute: int
    ordinal: int
    window_identity_sha256: str

    @property
    def profile_signal_strength(self) -> int:
        return (
            self.identity_strength * 4
            + self.preference_strength * 2
            + self.request_strength
        )

    @property
    def rank(self) -> tuple[int, ...]:
        # Literal question overlap is deliberately behind every substantive
        # profile/domain dimension.
        return (
            self.identity_strength,
            self.uncommon_constraint_score,
            self.profile_signal_strength,
            self.domain_match_count,
            self.recency,
            self.created_epoch_minute,
            self.ordinal,
            self.question_overlap_count,
            -self.window_index,
        )

    def projection(self) -> dict[str, Any]:
        return {
            "age_days": self.age_days,
            "cluster_key_sha256": self.cluster_key_sha256,
            "constraint_terms": list(self.constraint_terms),
            "created_epoch_minute": self.created_epoch_minute,
            "domain_match_count": self.domain_match_count,
            "identity_strength": self.identity_strength,
            "ordinal": self.ordinal,
            "preference_strength": self.preference_strength,
            "profile_signal_strength": self.profile_signal_strength,
            "question_overlap_count": self.question_overlap_count,
            "recency": self.recency,
            "request_strength": self.request_strength,
            "uncommon_constraint_score": self.uncommon_constraint_score,
            "window_identity_sha256": self.window_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class _RankedCluster:
    cluster_key_sha256: str
    source_id: str
    windows: tuple[_ScoredWindow, ...]
    freshness: int
    explicit_identity_strength: int
    uncommon_constraint_score: int
    profile_signal_strength: int
    coherent_constraint_score: int
    domain_constraint_coverage: int
    newest_epoch_minute: int
    question_overlap_count: int

    @property
    def intrinsic_profile_priority(self) -> int:
        """Keep durable identity/preference above request-only cue clusters."""

        return int(
            any(
                row.identity_strength or row.preference_strength
                for row in self.windows
            )
        )

    @property
    def rank(self) -> tuple[int, ...]:
        return (
            self.intrinsic_profile_priority,
            self.freshness,
            self.explicit_identity_strength,
            self.uncommon_constraint_score,
            self.profile_signal_strength,
            self.coherent_constraint_score,
            self.domain_constraint_coverage,
            self.newest_epoch_minute,
            self.question_overlap_count,
        )

    def opaque_projection(self) -> dict[str, Any]:
        return {
            "cluster_key_sha256": self.cluster_key_sha256,
            "coherent_constraint_score": self.coherent_constraint_score,
            "domain_constraint_coverage": self.domain_constraint_coverage,
            "explicit_identity_strength": self.explicit_identity_strength,
            "freshness": self.freshness,
            "intrinsic_profile_priority": self.intrinsic_profile_priority,
            "newest_epoch_minute": self.newest_epoch_minute,
            "profile_signal_strength": self.profile_signal_strength,
            "question_overlap_count": self.question_overlap_count,
            "uncommon_constraint_score": self.uncommon_constraint_score,
            "window_identity_sha256s": [
                row.window_identity_sha256 for row in self.windows
            ],
        }


def _window_identity(index: FullStoreWindowIndex, window_index: int) -> str:
    window = index.windows[window_index]
    return identity_sha256(
        {
            "end_char": window.end_char,
            "format": CANDIDATE_ID_FORMAT,
            "index_receipt_sha256": index.receipt_sha256,
            "row_receipt_sha256": identity_sha256(window.row.receipt_projection()),
            "start_char": window.start_char,
            "text_sha256": window.text_sha256,
        }
    )


def _score_windows(
    index: FullStoreWindowIndex,
    *,
    asked_at: datetime,
    question_terms: frozenset[str],
    domains: tuple[str, ...],
) -> tuple[tuple[_ScoredWindow, ...], int, int]:
    evidence_domain_terms = frozenset().union(
        *(_DOMAIN_EVIDENCE_TERMS[domain] for domain in domains)
    )
    scored: list[_ScoredWindow] = []
    role_rejected = 0
    future_rejected = 0
    for window_index, window in enumerate(index.windows):
        row = window.row
        if row.role != "user":
            role_rejected += 1
            continue
        created_at = _row_datetime(row.created_at)
        if created_at > asked_at:
            future_rejected += 1
            continue
        quote = row.text[window.start_char : window.end_char]
        identity = _maximum_pattern_score(quote, _IDENTITY_PATTERNS)
        preference = _maximum_pattern_score(quote, _PREFERENCE_PATTERNS)
        request = _maximum_pattern_score(quote, _REQUEST_PATTERNS)
        if not (identity or preference or request):
            continue
        window_terms = window.terms
        domain_terms = window_terms & evidence_domain_terms
        if not domain_terms:
            continue
        entities = _entity_terms(quote)
        constraints = tuple(sorted(domain_terms | entities))
        rarity = sorted(
            (_rarity_weight(index, term) for term in constraints), reverse=True
        )
        uncommon_score = sum(rarity[:8])
        age_days = max(0, (asked_at.date() - created_at.date()).days)
        cluster_key = evidence_source_story_key_sha256(
            row.namespace_id, row.source_id
        )
        scored.append(
            _ScoredWindow(
                window_index=window_index,
                cluster_key_sha256=cluster_key,
                age_days=age_days,
                recency=_recency_bucket(age_days),
                identity_strength=identity,
                preference_strength=preference,
                request_strength=request,
                uncommon_constraint_score=uncommon_score,
                domain_match_count=len(domain_terms),
                question_overlap_count=len(window_terms & question_terms),
                constraint_terms=constraints,
                created_epoch_minute=int(created_at.timestamp() // 60),
                ordinal=row.ordinal,
                window_identity_sha256=_window_identity(index, window_index),
            )
        )
    return tuple(scored), role_rejected, future_rejected


def _rank_clusters(
    index: FullStoreWindowIndex,
    windows: Sequence[_ScoredWindow],
    *,
    budget: ProfilePreferenceBudget,
) -> tuple[_RankedCluster, ...]:
    by_source: dict[str, list[_ScoredWindow]] = defaultdict(list)
    for row in windows:
        by_source[index.windows[row.window_index].row.source_id].append(row)
    ranked: list[_RankedCluster] = []
    for source_id, source_windows in by_source.items():
        ordered = tuple(
            sorted(source_windows, key=lambda row: row.rank, reverse=True)[
                : budget.max_windows_per_cluster
            ]
        )
        constraint_counts = Counter(
            term for row in ordered for term in row.constraint_terms
        )
        coherent = sum(
            min(count, 4) * _rarity_weight(index, term)
            for term, count in constraint_counts.items()
            if count >= 2
        )
        ranked.append(
            _RankedCluster(
                cluster_key_sha256=ordered[0].cluster_key_sha256,
                source_id=source_id,
                windows=ordered,
                freshness=max(row.recency for row in ordered),
                explicit_identity_strength=max(
                    row.identity_strength for row in ordered
                ),
                uncommon_constraint_score=sum(
                    sorted(
                        (row.uncommon_constraint_score for row in ordered),
                        reverse=True,
                    )[:3]
                ),
                profile_signal_strength=sum(
                    sorted(
                        (row.profile_signal_strength for row in ordered),
                        reverse=True,
                    )[:4]
                ),
                coherent_constraint_score=coherent,
                domain_constraint_coverage=len(constraint_counts),
                newest_epoch_minute=max(
                    row.created_epoch_minute for row in ordered
                ),
                question_overlap_count=sum(
                    row.question_overlap_count for row in ordered
                ),
            )
        )
    return tuple(
        sorted(
            ranked,
            key=lambda row: (
                tuple(-value for value in row.rank),
                row.cluster_key_sha256,
            ),
        )
    )


def _candidate_and_binding(
    index: FullStoreWindowIndex,
    scored: _ScoredWindow,
    *,
    asked_at: datetime,
    question_terms: frozenset[str],
) -> tuple[FullStoreSlotCandidate, LocalCitationBinding]:
    window = index.windows[scored.window_index]
    row = window.row
    quote = row.text[window.start_char : window.end_char]
    candidate_id = scored.window_identity_sha256
    group_handle = "G0001"
    span = EvidenceSpan(
        chunk_id=row.chunk_id,
        start_char=window.start_char,
        end_char=window.end_char,
        quote_sha256=quote_sha256(quote),
        ordinal=row.ordinal,
        source_id=row.source_id,
        turn_start_char=row.turn_start_char,
        turn_id=row.turn_id,
        role=row.role,
        created_at=row.created_at,
    )
    binding = LocalCitationBinding(
        candidate_id=candidate_id,
        source_group_handle=group_handle,
        namespace_id=row.namespace_id,
        cache_receipt_sha256=index.cache.cache_receipt_sha256,
        source_database_sha256=index.cache.source_database_sha256,
        source_store_receipt_sha256=index.cache.source_store_receipt_sha256,
        source_id=row.source_id,
        partition_id=row.partition_id,
        span=span,
        quote_sha256=window.text_sha256,
    )
    axes = ["one_source_interest_cluster", "recent_first_person_user_memory"]
    if _FIRST_PERSON_RE.search(quote):
        axes.append("first_person_surface")
    if scored.identity_strength:
        axes.append("explicit_user_identity")
    if scored.preference_strength:
        axes.append("explicit_user_preference")
    if scored.request_strength:
        axes.append("user_recommendation_request")
    if scored.uncommon_constraint_score:
        axes.append("uncommon_domain_entity_constraint")
    candidate = FullStoreSlotCandidate(
        candidate_id=candidate_id,
        source_group_handle=group_handle,
        quote=quote,
        quote_sha256=window.text_sha256,
        token_count=window.token_count,
        role=row.role,
        created_at=row.created_at,
        event_date=window.event_date,
        event_date_basis=window.event_date_basis,
        supported_slot_ids=(),
        matched_query_terms=tuple(
            term for term in indexed_surface_terms(quote) if term in question_terms
        ),
        contains_numeric_value=window.contains_numeric_value,
        temporal_distance_days=max(
            0, (asked_at.date() - _row_datetime(row.created_at).date()).days
        ),
        selection_axes=tuple(axes),
        citation_binding_receipt_sha256=binding.receipt_sha256,
    )
    return candidate, binding


def _empty_digests() -> tuple[str, str]:
    return (
        identity_sha256(
            {"format": f"{RESULT_FORMAT}-candidate-population", "rows": []}
        ),
        identity_sha256(
            {"format": f"{RESULT_FORMAT}-cluster-ranking", "rows": []}
        ),
    )


def select_profile_preference_evidence(
    index: FullStoreWindowIndex,
    dated_question: str,
    /,
    *,
    budget: ProfilePreferenceBudget = ProfilePreferenceBudget(),
) -> ProfilePreferenceResult:
    """Select one recent, coherent user profile/preference cluster.

    The function accepts no question ID, reference, prediction, source prefix,
    partition route, callback, model, or mutable state.  Every physical window
    is visited exactly once for the applicable path; output is then greedily
    bounded by the explicit candidate and token caps.
    """

    _require(type(index) is FullStoreWindowIndex, "profile index changed type")
    _require(type(budget) is ProfilePreferenceBudget, "profile budget changed type")
    asked_at, body = _question_parts(dated_question)
    operator_spec = compile_typed_operator_spec(dated_question)
    question_terms = frozenset(indexed_surface_terms(body))
    domains = _recognized_domains(question_terms)
    is_recommendation = bool(_RECOMMENDATION_RE.search(body))

    scored: tuple[_ScoredWindow, ...] = ()
    ranked_clusters: tuple[_RankedCluster, ...] = ()
    role_rejected = 0
    future_rejected = 0
    if is_recommendation and domains:
        scored, role_rejected, future_rejected = _score_windows(
            index,
            asked_at=asked_at,
            question_terms=question_terms,
            domains=domains,
        )
        ranked_clusters = _rank_clusters(index, scored, budget=budget)
    else:
        # The inventory is still fully observed so every result has the same
        # physical-scan accounting contract; no semantic candidate is scored.
        role_rejected = sum(window.row.role != "user" for window in index.windows)
        user_count = len(index.windows) - role_rejected
        future_rejected = sum(
            window.row.role == "user"
            and _row_datetime(window.row.created_at) > asked_at
            for window in index.windows
        )
        _require(user_count >= future_rejected, "future user count changed")

    population_sha = identity_sha256(
        {
            "format": f"{RESULT_FORMAT}-candidate-population",
            "rows": [row.projection() for row in scored],
        }
    )
    cluster_sha = identity_sha256(
        {
            "format": f"{RESULT_FORMAT}-cluster-ranking",
            "rows": [row.opaque_projection() for row in ranked_clusters],
        }
    )
    empty_population_sha, empty_cluster_sha = _empty_digests()
    if not is_recommendation:
        status: SelectionStatus = "not_recommendation_or_preference"
        _require(
            population_sha == empty_population_sha
            and cluster_sha == empty_cluster_sha,
            "non-applicable profile route produced candidates",
        )
    elif not domains:
        status = "unsupported_query_domain"
        _require(
            population_sha == empty_population_sha
            and cluster_sha == empty_cluster_sha,
            "unsupported profile domain produced candidates",
        )
    elif not ranked_clusters:
        status = "no_profile_evidence"
    else:
        status = "selected"

    candidates: list[FullStoreSlotCandidate] = []
    bindings: list[LocalCitationBinding] = []
    selected_cluster = ranked_clusters[0] if ranked_clusters else None
    if selected_cluster is not None:
        tokens = 0
        for row in selected_cluster.windows:
            if len(candidates) >= budget.max_selected_candidates:
                break
            window_tokens = index.windows[row.window_index].token_count
            if tokens + window_tokens > budget.max_selected_tokens:
                continue
            candidate, binding = _candidate_and_binding(
                index,
                row,
                asked_at=asked_at,
                question_terms=question_terms,
            )
            candidates.append(candidate)
            bindings.append(binding)
            tokens += candidate.token_count
    if status == "selected" and not candidates:
        status = "no_profile_evidence"
        selected_cluster = None

    selected_candidates = tuple(candidates)
    selected_bindings = tuple(bindings)
    selected_cluster_projection = (
        None if selected_cluster is None else selected_cluster.opaque_projection()
    )
    audit = ProfilePreferenceAudit(
        question_sha256=quote_sha256(dated_question),
        operator_spec_receipt_sha256=operator_spec.receipt_sha256,
        index_receipt_sha256=index.receipt_sha256,
        cache_receipt_sha256=index.cache.cache_receipt_sha256,
        budget_id=budget.budget_id,
        status=status,
        recognized_domain_ids=domains,
        physical_sentence_window_count=len(index.windows),
        physical_sentence_windows_scanned=len(index.windows),
        user_sentence_window_count=sum(
            window.row.role == "user" for window in index.windows
        ),
        eligible_candidate_count=len(scored),
        eligible_cluster_count=len(ranked_clusters),
        role_rejected_window_count=role_rejected,
        future_rejected_window_count=future_rejected,
        candidate_population_sha256=population_sha,
        cluster_ranking_sha256=cluster_sha,
        selected_cluster_key_sha256=(
            None if selected_cluster is None else selected_cluster.cluster_key_sha256
        ),
        selected_cluster_rank_projection_sha256=(
            None
            if selected_cluster_projection is None
            else identity_sha256(selected_cluster_projection)
        ),
        selected_candidate_ids=tuple(row.candidate_id for row in selected_candidates),
        selected_local_binding_receipt_sha256s=tuple(
            row.receipt_sha256 for row in selected_bindings
        ),
        selected_evidence_tokens=sum(row.token_count for row in selected_candidates),
        selection_truncated=bool(
            selected_cluster is not None
            and len(selected_cluster.windows) > len(selected_candidates)
        ),
        max_selected_candidates=budget.max_selected_candidates,
        max_selected_tokens=budget.max_selected_tokens,
        max_windows_per_cluster=budget.max_windows_per_cluster,
        selected_source_cluster_count=1 if selected_candidates else 0,
    )
    return ProfilePreferenceResult(
        dated_question=dated_question,
        operator_spec=operator_spec,
        budget=budget,
        candidates=selected_candidates,
        local_bindings=selected_bindings,
        audit=audit,
    )


def adapt_profile_preference_to_typed_contribution(
    result: ProfilePreferenceResult,
    /,
    *,
    handle_start: int,
    group_start: int,
) -> "TypedEvidenceContribution":
    """Convert selected pointers to one bounded typed contribution."""

    from .typed_operator_adapter import (
        EvidenceHandleBinding,
        EvidenceOrigin,
        FrontierMode,
        ProvenanceGrade,
        TypedEvidenceContribution,
        parse_typed_items,
    )

    _require(
        type(result) is ProfilePreferenceResult,
        "typed profile contribution requires an exact result",
    )
    for value, label in (
        (handle_start, "profile handle start"),
        (group_start, "profile group start"),
    ):
        _require(type(value) is int and value >= 1, f"{label} must be positive")
    _require(
        handle_start + len(result.candidates) - 1 <= 999_999,
        "profile handle range exceeds the opaque contract",
    )
    local_groups = tuple(
        dict.fromkeys(row.source_group_handle for row in result.candidates)
    )
    _require(
        group_start + len(local_groups) - 1 <= 999_999,
        "profile group range exceeds the opaque contract",
    )
    global_groups = {
        local: f"G{group_start + offset:03d}"
        for offset, local in enumerate(local_groups)
    }
    sealed_artifact_sha256 = identity_sha256(result.local_projection())
    typed_bindings: list[EvidenceHandleBinding] = []
    raw_items: list[dict[str, Any]] = []
    for offset, (candidate, local) in enumerate(
        zip(result.candidates, result.local_bindings, strict=True)
    ):
        handle_id = f"H{handle_start + offset:03d}"
        group_handle = global_groups[candidate.source_group_handle]
        typed_bindings.append(
            EvidenceHandleBinding(
                handle_id=handle_id,
                origin=EvidenceOrigin.DIRECT_POINTER,
                provenance_grade=ProvenanceGrade.DIRECT_POINTER,
                source_group_handle=group_handle,
                sealed_artifact_sha256=sealed_artifact_sha256,
                parent_receipt_sha256=result.audit.receipt_sha256,
                evidence_receipt_sha256=local.receipt_sha256,
                payload_sha256=identity_sha256(candidate.projection()),
                citation_sha256=candidate.quote_sha256,
                citation_char_count=len(candidate.quote),
                local_source_locator_sha256=local.receipt_sha256,
            )
        )
        raw_items.append(
            {
                "handle_ids": [handle_id],
                "included": True,
                "kind": "direct",
                "numeric_role": "none",
                "personalization_anchors": ["first-person user preference"],
                "relation": "authored_by_user;profile_preference_memory",
                "specificity_terms": [],
                "status": "unknown",
                "summary": candidate.quote,
                "value_authority": "explicit",
                **(
                    {"date": candidate.event_date}
                    if candidate.event_date is not None
                    else {}
                ),
            }
        )
    frozen_bindings = tuple(typed_bindings)
    parsed = parse_typed_items(
        raw_items,
        operator_spec=result.operator_spec,
        bindings=frozen_bindings,
    )
    contribution = TypedEvidenceContribution(
        mechanism_id=MECHANISM_ID,
        bindings=frozen_bindings,
        parsed=parsed,
        sealed_artifact_sha256=sealed_artifact_sha256,
        frontier_mode=FrontierMode.BOUNDED,
        truncated=result.audit.selection_truncated,
    )
    _require(
        contribution.frontier_mode is FrontierMode.BOUNDED
        and len(contribution.bindings) == len(result.candidates),
        "profile typed contribution changed its bounded exact-pointer inventory",
    )
    return contribution


__all__ = [
    "AUDIT_FORMAT",
    "BUDGET_FORMAT",
    "MECHANISM_ID",
    "RANKING_POLICY",
    "RESULT_FORMAT",
    "ProfilePreferenceAudit",
    "ProfilePreferenceBudget",
    "ProfilePreferenceResult",
    "ProfilePreferenceSpecialistError",
    "adapt_profile_preference_to_typed_contribution",
    "select_profile_preference_evidence",
]
