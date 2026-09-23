#!/usr/bin/env python3
"""Build gold-free user-spine episodic packets over the sealed locked100.

Operation C treats the user turn as the address of an exchange.  The exact
sources activated by the sealed v4 packet bound the search domain; within that
domain a deterministic question/typed-operation scorer ranks complete user-led
exchanges.  Winning exchanges are hydrated before quote facts are compiled.

The original selected raw rows remain authoritative and protected.  Episode
raw rows and quote facts have independent budgets and are deduplicated only
after their selections are fixed.  A source/provenance conservation failure
returns the exact Operation-A raw prompt.  This program loads no benchmark
labels and performs no provider I/O.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.integrity import file_sha256
from tools import assay_hot_retrieval_1m as hot
from tools import assay_hot_retrieval_source_seed_hybrid_full100 as v3
from tools import assay_hot_v3_typed_operator_full100 as typed
from tools import assay_hot_v4_user_envelope_provider_selection as source_assay
from tools import assay_hot_v4_user_envelope_shadow_full100 as shadow
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.hot_v3_user_led_envelope_shadow import (
    UserLedEnvelopeShadowIndex,
    build_user_led_envelope_shadow_index,
)
from tools.matched_eval import hot_v3_user_led_envelope_shadow as envelope_shadow
from tools.matched_eval.hot_v5_user_spine_prompt import (
    OPERATION_AWARE_SYSTEM_PROMPT,
    replace_system_prompt_only,
)
from tools.matched_eval.hot_v6_query_fact_ledger import (
    QueryFactLedger,
    QueryFactLedgerError,
    compile_query_fact_ledger,
    select_and_render_query_fact_ledger,
)
from tools.matched_eval.query_expansion import load_locked_query_expansion_context
from tools.matched_eval.query_guided_scan import CachedContentRow
from tools.matched_eval.typed_action_semantics import (
    canonical_action_concepts,
    linked_action_concepts,
)
from tools.matched_eval.typed_operator_spec import (
    TemporalMode,
    TypedOperatorSpec,
    compile_typed_operator_spec,
    normalized_terms,
)


FORMAT = "memory-condense-hot-v6-spine-episode-fact-ledger-selection-v3"
ROW_FORMAT = f"{FORMAT}-row-v1"
SELECTION_NAME = "selection.json"
REPLAY_NAME = "semantic-replay.json"
EXPECTED_QUESTION_COUNT = source_assay.EXPECTED_QUESTION_COUNT
EXPECTED_POPULATION_SHA256 = source_assay.EXPECTED_POPULATION_SHA256
MAX_CONTEXT_TOKENS = 10_000
MAX_WORKSPACE_TOKENS = 11_000
OUTPUT_TOKEN_RESERVE = hot.RESPONDER_OUTPUT_TOKEN_RESERVE

# Independent selection budgets.  Neither lane may borrow from the other.
MAX_EPISODES = 24
MAX_EPISODES_PER_SOURCE = 2
MAX_TURNS_PER_EPISODE = 6
MAX_EPISODE_RAW_CHUNKS = 24
MAX_EPISODE_RAW_TOKENS = 1_600
MAX_FACTS = 20
MAX_FACT_TOKENS = 500
LEXICAL_LANE_EPISODES = 2
ANCHOR_LANE_EPISODES = 2
SLOT_LANE_EPISODES = 2
TYPED_LANE_EPISODES = 2
PHYSICAL_OWNER_LANE_EPISODES = 8
PHYSICAL_TRANSITION_LANE_EPISODES = 8
EPISODE_LANE_TOKEN_BUDGET = 400
EPISODE_LANE_CHUNK_BUDGET = 6
PHYSICAL_OWNER_LANE_TOKEN_BUDGET = 800
PHYSICAL_OWNER_LANE_CHUNK_BUDGET = 12
PHYSICAL_TRANSITION_LANE_TOKEN_BUDGET = 800
PHYSICAL_TRANSITION_LANE_CHUNK_BUDGET = 12

DEFAULT_SOURCE_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v4-user-envelope-shadow-full100-20260907-r2"
)
DEFAULT_STORE_ROOT = v3.DEFAULT_SOURCE_ROOT
DEFAULT_RETRIEVAL = DEFAULT_STORE_ROOT / "retrieval.json"
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-v6-spine-episode-fact-ledger-full100-20260907-r3"
)

_DATE_TERM_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?|today|yesterday|last|latest|current|first|"
    r"before|after|between|ago|week|month|year|day)s?\b",
    re.IGNORECASE,
)
_STATUS_TERM_RE = re.compile(
    r"\b(?:plan(?:ned|ning)?|propos(?:e|ed)|intend(?:ed|ing)?|pending|"
    r"attempt(?:ed|ing)?|tried|completed|finished|bought|visited|returned|"
    r"cancel(?:ed|led)?|failed|current|latest|replaced)\b",
    re.IGNORECASE,
)
_DATED_QUESTION_RE = re.compile(
    r"^\[Question asked at .+?\]\s*", re.IGNORECASE | re.DOTALL
)
_CONDITIONED_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "did",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "me",
        "my",
        "of",
        "on",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
    }
)

USER_TEMPLATE = (
    "Retrieved memory. <G#> blocks are protected global evidence. <E#> blocks "
    "are isolated user-led exchanges; every <A owner=U> belongs to the user "
    "lead in that same block. <F#> entries are exact quote facts compiled only "
    "from the selected exchanges. Do not merge blocks or sources.\n{context}\n\n"
    "Question: {question}\nShort answer:"
)


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/domain/integrity.py",
        "src/memory_condense/domain/text_numbers.py",
        "src/memory_condense/search/episodes/user_led.py",
        "tools/assay_hot_retrieval_1m.py",
        "tools/assay_hot_retrieval_source_seed_hybrid_full100.py",
        "tools/assay_hot_v3_typed_operator_full100.py",
        "tools/assay_hot_v4_user_envelope_provider_selection.py",
        "tools/assay_hot_v4_user_envelope_shadow_full100.py",
        "tools/assay_hot_v6_spine_episode_fact_ledger_full100.py",
        "tools/_routed_repair_routing.py",
        "tools/matched_eval/contracts.py",
        "tools/matched_eval/hot_v3_user_led_envelope_shadow.py",
        "tools/matched_eval/hot_v5_user_spine_prompt.py",
        "tools/matched_eval/hot_v6_query_fact_ledger.py",
        "tools/matched_eval/query_guided_scan.py",
        "tools/matched_eval/query_expansion.py",
        "tools/matched_eval/typed_action_semantics.py",
        "tools/matched_eval/typed_numeric_semantics.py",
        "tools/matched_eval/typed_operator_spec.py",
    )
    files = {path: file_sha256(root / path) for path in paths}
    return {
        "files": files,
        "format": "memory-condense-hot-v6-spine-episode-implementation-v2",
        "sha256": identity_sha256(
            [{"path": path, "sha256": files[path]} for path in paths]
        ),
    }


def _row_text(row: CachedContentRow) -> str:
    return row.text


def _score_terms(question: str, spec: TypedOperatorSpec) -> dict[str, frozenset[str]]:
    body = _DATED_QUESTION_RE.sub("", question).strip()
    slot_terms = {
        term for slot in spec.required_slots for term in slot.match_terms
    }
    return {
        "question": frozenset(normalized_terms(body)),
        "slots": frozenset(slot_terms),
        "actions": frozenset(canonical_action_concepts(body)),
        "dates": frozenset(value.casefold() for value in _DATE_TERM_RE.findall(body)),
        "statuses": frozenset(
            value.casefold() for value in _STATUS_TERM_RE.findall(body)
        ),
    }


def _compatible_hits(
    needles: Sequence[str] | frozenset[str],
    haystack: Sequence[str] | set[str] | frozenset[str],
) -> int:
    """Count conservative exact-or-morphological-prefix term matches."""

    targets = set(haystack)
    return sum(
        any(
            needle == candidate
            or (
                min(len(needle), len(candidate)) >= 5
                and (
                    needle.startswith(candidate)
                    or candidate.startswith(needle)
                )
            )
            for candidate in targets
        )
        for needle in set(needles)
    )


def _weighted_compatible_score(
    needles: Sequence[str] | frozenset[str],
    haystack: Sequence[str] | set[str] | frozenset[str],
    idf: Mapping[str, float],
) -> int:
    targets = set(haystack)
    return int(
        round(
            100
            * math.fsum(
                idf.get(needle, 1.0)
                for needle in sorted(set(needles))
                if _compatible_hits((needle,), targets)
            )
        )
    )


def _episode_rank(
    index: UserLedEnvelopeShadowIndex,
    envelope: object,
    *,
    active_rank: Mapping[str, int],
    source_anchor_terms: Mapping[str, frozenset[str]],
    terms: Mapping[str, frozenset[str]],
    temporal_intent: bool,
    status_intent: bool,
    spec: TypedOperatorSpec,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    source_id = str(getattr(envelope, "source_id"))
    chunk_ids = tuple(getattr(envelope, "chunk_ids"))
    rows = tuple(index.row_by_chunk_id[chunk_id] for chunk_id in chunk_ids)
    user_rows = tuple(row for row in rows if row.role == "user")
    assistant_rows = tuple(row for row in rows if row.role == "assistant")
    user_terms = set(normalized_terms(" ".join(_row_text(row) for row in user_rows)))
    assistant_terms = set(
        normalized_terms(" ".join(_row_text(row) for row in assistant_rows))
    )
    all_terms = user_terms | assistant_terms
    question_overlap = _compatible_hits(terms["question"], all_terms)
    user_overlap = _compatible_hits(terms["question"], user_terms)
    slot_hits = _compatible_hits(terms["slots"], all_terms)
    source_anchor_hits = _compatible_hits(
        source_anchor_terms.get(source_id, frozenset()), all_terms
    )
    actions = {
        concept for row in rows for concept in linked_action_concepts(row.text)
    }
    action_hits = len(terms["actions"] & actions)
    surface = " ".join(row.text for row in rows)
    episode_date_markers = {
        value.casefold() for value in _DATE_TERM_RE.findall(surface)
    }
    # Temporal language rarely repeats byte-for-byte: "last year" in the
    # question can be supported by "in June" or "three months ago". Reward
    # any event-time marker when the question carries temporal intent.
    date_hits = int(temporal_intent and bool(episode_date_markers))
    episode_status_markers = {
        value.casefold() for value in _STATUS_TERM_RE.findall(surface)
    }
    status_hits = int(status_intent and bool(episode_status_markers))
    required_role_hit = int(
        spec.required_evidence_role is None
        or any(row.role == spec.required_evidence_role for row in rows)
    )
    numeric_hit = int(
        not any(slot.requires_numeric for slot in spec.required_slots)
        or bool(re.search(r"(?<!\w)[+-]?(?:\d[\d,.]*|\.\d+)", surface))
    )
    newest = max(row.ordinal for row in rows)
    recency = newest if spec.temporal_mode is TemporalMode.LATEST_STATE else 0
    # Tuple order is the policy: typed obligations, user-lead lexical match,
    # then whole-exchange match.  Source activation and chronology are only
    # deterministic tie breakers, never hidden labels.
    rank = (
        slot_hits,
        action_hits,
        date_hits,
        status_hits,
        required_role_hit,
        numeric_hit,
        user_overlap,
        source_anchor_hits,
        question_overlap,
        recency,
        -active_rank[source_id],
        -min(row.ordinal for row in rows),
    )
    audit = {
        "action_hits": action_hits,
        "assistant_chunk_count": len(assistant_rows),
        "date_hits": date_hits,
        "envelope_id": str(getattr(envelope, "envelope_id")),
        "numeric_requirement_hit": bool(numeric_hit),
        "question_overlap": question_overlap,
        "rank_vector": list(rank),
        "required_role_hit": bool(required_role_hit),
        "slot_hits": slot_hits,
        "source_id": source_id,
        "source_anchor_hits": source_anchor_hits,
        "status_hits": status_hits,
        "user_chunk_count": len(user_rows),
        "user_overlap": user_overlap,
    }
    return rank, audit


def _physical_anchor_neighborhoods(
    index: UserLedEnvelopeShadowIndex,
    parent_rows: Sequence[Mapping[str, Any]],
    *,
    active_source_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve each exact physical parent to one owner and local neighbors."""

    active = set(active_source_ids)
    by_source: dict[str, list[object]] = {}
    for envelope in index.envelopes:
        if envelope.exchange_kind == "user_led" and envelope.source_id in active:
            by_source.setdefault(str(envelope.source_id), []).append(envelope)
    for envelopes in by_source.values():
        envelopes.sort(
            key=lambda envelope: (
                min(envelope.turn_ordinals),
                str(envelope.envelope_id),
            )
        )
    position_by_envelope = {
        str(envelope.envelope_id): position
        for envelopes in by_source.values()
        for position, envelope in enumerate(envelopes)
    }
    neighborhoods: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for parent_position, raw in enumerate(parent_rows):
        logical_id = str(raw.get("evidence_id", raw.get("chunk_id", "")))
        physical_id = str(raw.get("backing_chunk_id", raw.get("chunk_id", "")))
        cached = index.row_by_chunk_id.get(physical_id)
        if cached is None:
            diagnostics.append(
                {
                    "logical_evidence_id": logical_id,
                    "parent_position": parent_position,
                    "physical_chunk_id": physical_id,
                    "reason": "physical_chunk_not_cached",
                }
            )
            continue
        if cached.source_id not in active or not envelope_shadow._exact_physical_parent(  # noqa: SLF001
            raw, cached
        ):
            diagnostics.append(
                {
                    "logical_evidence_id": logical_id,
                    "parent_position": parent_position,
                    "physical_chunk_id": physical_id,
                    "reason": "physical_anchor_mismatch",
                }
            )
            continue
        if cached.role != "user":
            diagnostics.append(
                {
                    "logical_evidence_id": logical_id,
                    "parent_position": parent_position,
                    "physical_chunk_id": physical_id,
                    "reason": "physical_anchor_not_user_spine",
                }
            )
            continue
        owner_id = index.envelope_by_chunk_id[physical_id]
        owner = index.envelope_by_id[owner_id]
        if owner.exchange_kind != "user_led":
            diagnostics.append(
                {
                    "logical_evidence_id": logical_id,
                    "parent_position": parent_position,
                    "physical_chunk_id": physical_id,
                    "reason": "physical_anchor_has_no_user_lead",
                }
            )
            continue
        source_envelopes = by_source[str(owner.source_id)]
        owner_position = position_by_envelope[str(owner.envelope_id)]
        neighbors: list[dict[str, Any]] = []
        for delta, direction in ((-1, "previous"), (1, "next")):
            candidate_position = owner_position + delta
            if not 0 <= candidate_position < len(source_envelopes):
                continue
            candidate = source_envelopes[candidate_position]
            neighbors.append(
                {
                    "direction": direction,
                    "envelope_id": str(candidate.envelope_id),
                }
            )
        neighborhoods.append(
            {
                "excerpt_backed": "backing_chunk_id" in raw,
                "neighbors": neighbors,
                "owner_envelope_id": str(owner.envelope_id),
                "parent_logical_evidence_id": logical_id,
                "parent_position": parent_position,
                "parent_raw_text": str(raw["raw_text"]),
                "physical_anchor_chunk_id": physical_id,
                "source_id": str(owner.source_id),
            }
        )
    return neighborhoods, diagnostics


def select_spine_episodes(
    index: UserLedEnvelopeShadowIndex,
    *,
    active_source_ids: Sequence[str],
    dated_question: str,
    parent_packed_rows: Sequence[Mapping[str, Any]] = (),
    typed_spec: TypedOperatorSpec | None = None,
) -> dict[str, Any]:
    """Rank and atomically hydrate user-led exchanges in active sources."""

    _require(type(index) is UserLedEnvelopeShadowIndex, "episode index type changed")
    active = tuple(active_source_ids)
    _require(active and len(active) == len(set(active)), "active sources changed")
    active_rank = {source_id: rank for rank, source_id in enumerate(active)}
    parent = tuple(dict(row) for row in parent_packed_rows)
    _require(
        all(str(row.get("source_id", "")) in active_rank for row in parent),
        "parent anchors escaped active sources",
    )
    raw_anchor_terms: dict[str, frozenset[str]] = {}
    for source_id in active:
        source_rows = [
            row for row in parent if row.get("source_id") == source_id
        ]
        user_rows = [row for row in source_rows if row.get("role") == "user"]
        # User evidence is the spine. Assistant text is only a fallback for a
        # source whose fast packet contains no user row at all.
        surface = " ".join(
            str(row.get("raw_text", ""))
            for row in (user_rows or source_rows)
        )
        raw_anchor_terms[source_id] = frozenset(normalized_terms(surface))
    document_frequency = Counter(
        term for values in raw_anchor_terms.values() for term in values
    )
    frequent_cutoff = max(2, (len(active) + 2) // 3)
    source_anchor_terms = {
        source_id: frozenset(
            term
            for term in values
            if document_frequency[term] < frequent_cutoff
        )
        for source_id, values in raw_anchor_terms.items()
    }
    spec = typed_spec or compile_typed_operator_spec(dated_question)
    terms = _score_terms(dated_question, spec)
    candidates: list[tuple[tuple[int, ...], str, object, dict[str, Any]]] = []
    source_envelope_counts = Counter()
    for envelope in index.envelopes:
        source_id = str(envelope.source_id)
        if source_id not in active_rank or envelope.exchange_kind != "user_led":
            continue
        source_envelope_counts[source_id] += 1
        rank, audit = _episode_rank(
            index,
            envelope,
            active_rank=active_rank,
            source_anchor_terms=source_anchor_terms,
            terms=terms,
            temporal_intent=(
                spec.temporal_mode is not TemporalMode.NONE
                or spec.temporal_window_days is not None
            ),
            status_intent=(
                spec.temporal_mode is TemporalMode.LATEST_STATE
                or spec.include_proposed
                or bool(terms["statuses"] - {"current", "latest"})
            ),
            spec=spec,
        )
        candidates.append((rank, str(envelope.envelope_id), envelope, audit))
    # Specialist lanes rank the same immutable episode universe independently.
    # IDF is computed only over user leads, so assistant boilerplate cannot
    # inflate lexical or source-anchor scores.
    user_terms_by_envelope: dict[str, frozenset[str]] = {}
    all_terms_by_envelope: dict[str, frozenset[str]] = {}
    user_surface_by_envelope: dict[str, str] = {}
    for _rank, envelope_id, envelope, _audit in candidates:
        rows = [index.row_by_chunk_id[chunk_id] for chunk_id in envelope.chunk_ids]
        user_surface = " ".join(row.text for row in rows if row.role == "user")
        all_surface = " ".join(row.text for row in rows)
        user_surface_by_envelope[envelope_id] = user_surface
        user_terms_by_envelope[envelope_id] = frozenset(normalized_terms(user_surface))
        all_terms_by_envelope[envelope_id] = frozenset(normalized_terms(all_surface))
    envelope_count = max(1, len(candidates))
    term_df = Counter(
        term for values in user_terms_by_envelope.values() for term in values
    )
    idf = {
        term: math.log((envelope_count + 1) / (frequency + 1)) + 1.0
        for term, frequency in term_df.items()
    }
    question_body_terms = normalized_terms(
        _DATED_QUESTION_RE.sub("", dated_question).strip()
    )
    question_bigrams = {
        (question_body_terms[index], question_body_terms[index + 1])
        for index in range(max(0, len(question_body_terms) - 1))
    }
    enriched: list[tuple[str, object, dict[str, Any]]] = []
    for _rank, envelope_id, envelope, audit in candidates:
        user_terms = user_terms_by_envelope[envelope_id]
        all_terms = all_terms_by_envelope[envelope_id]
        user_sequence = normalized_terms(user_surface_by_envelope[envelope_id])
        episode_bigrams = {
            (user_sequence[index], user_sequence[index + 1])
            for index in range(max(0, len(user_sequence) - 1))
        }
        lexical_score = _weighted_compatible_score(
            terms["question"], user_terms, idf
        ) + 250 * len(question_bigrams & episode_bigrams)
        anchor_score = _weighted_compatible_score(
            source_anchor_terms.get(str(envelope.source_id), frozenset()),
            user_terms,
            idf,
        )
        typed_core = (
            160 * int(audit["slot_hits"])
            + 80 * int(audit["action_hits"])
            + 50 * int(audit["date_hits"])
            + 50 * int(audit["status_hits"])
        )
        typed_score = typed_core + (
            10 * int(audit["required_role_hit"])
            + 10 * int(audit["numeric_requirement_hit"])
            if typed_core
            else 0
        )
        slot_scores: dict[str, int] = {}
        envelope_surface = " ".join(
            index.row_by_chunk_id[chunk_id].text
            for chunk_id in envelope.chunk_ids
        )
        for slot in spec.required_slots:
            lexical_slot_score = _weighted_compatible_score(
                slot.match_terms, all_terms, idf
            )
            numeric_bonus = 50 * int(
                slot.requires_numeric
                and bool(
                    re.search(
                        r"(?<!\w)[+-]?(?:\d[\d,.]*|\.\d+)",
                        envelope_surface,
                    )
                )
            )
            slot_scores[slot.slot_id] = (
                lexical_slot_score + numeric_bonus
                if lexical_slot_score
                else 0
            )
        audit = {
            **audit,
            "anchor_lane_score": anchor_score,
            "lexical_lane_score": lexical_score,
            "slot_lane_scores": slot_scores,
            "typed_lane_score": typed_score,
        }
        enriched.append((envelope_id, envelope, audit))

    def lane_order(score_name: str) -> list[tuple[str, object, dict[str, Any]]]:
        return sorted(
            enriched,
            key=lambda item: (
                int(item[2][score_name]),
                -active_rank[str(item[1].source_id)],
                -min(item[1].turn_ordinals),
                item[0],
            ),
            reverse=True,
        )

    physical_neighborhoods, physical_diagnostics = _physical_anchor_neighborhoods(
        index,
        parent,
        active_source_ids=active,
    )
    enriched_by_id = {envelope_id: (envelope, audit) for envelope_id, envelope, audit in enriched}
    anchor_terms = [
        frozenset(normalized_terms(neighborhood["parent_raw_text"]))
        for neighborhood in physical_neighborhoods
    ]
    anchor_df = Counter(term for values in anchor_terms for term in values)
    anchor_count = max(1, len(anchor_terms))
    anchor_idf = {
        term: math.log((anchor_count + 1) / (frequency + 1)) + 1.0
        for term, frequency in anchor_df.items()
    }
    conditioned_question_terms = frozenset(
        term
        for term in terms["question"]
        if term not in _CONDITIONED_STOP_TERMS and not term.isdigit()
    )

    def conditioned_score(
        neighborhood: Mapping[str, Any], envelope_id: str
    ) -> tuple[int, ...]:
        _envelope, audit = enriched_by_id[envelope_id]
        parent_terms = frozenset(normalized_terms(str(neighborhood["parent_raw_text"])))
        parent_actions = linked_action_concepts(str(neighborhood["parent_raw_text"]))
        candidate_terms = user_terms_by_envelope[envelope_id]
        union_terms = parent_terms | candidate_terms
        bridge_terms = frozenset(
            term
            for term in parent_terms
            if term not in _CONDITIONED_STOP_TERMS and not term.isdigit()
        )
        union_score = _weighted_compatible_score(
            conditioned_question_terms, union_terms, anchor_idf
        )
        bridge_score = _weighted_compatible_score(
            bridge_terms, candidate_terms, anchor_idf
        )
        pair_score = (
            2 * union_score
            + min(bridge_score, 2_000)
            + 2 * int(audit["lexical_lane_score"])
            + 200 * int(audit["action_hits"])
            + 100 * int(audit["date_hits"])
            + 100 * int(audit["status_hits"])
        )
        return (
            pair_score,
            union_score,
            int(audit["lexical_lane_score"]),
            bridge_score,
            _compatible_hits(terms["slots"], union_terms),
            len(terms["actions"] & set(parent_actions)) + int(audit["action_hits"]),
            int(audit["slot_hits"]),
            int(audit["action_hits"]),
            int(audit["date_hits"]),
            int(audit["status_hits"]),
            _weighted_compatible_score(
                conditioned_question_terms, parent_terms, anchor_idf
            ),
            int(audit["user_overlap"]),
            int(audit["question_overlap"]),
        )

    owner_choices: list[tuple[tuple[int, ...], int, str, object, dict[str, Any]]] = []
    transition_choices: list[
        tuple[tuple[int, ...], int, str, object, dict[str, Any]]
    ] = []
    frozen_local_choices: list[dict[str, Any]] = []
    for neighborhood in physical_neighborhoods:
        owner_id = str(neighborhood["owner_envelope_id"])
        owner_envelope, owner_audit = enriched_by_id[owner_id]
        base_owner_score = conditioned_score(neighborhood, owner_id)
        owner_lead = " ".join(
            index.row_by_chunk_id[chunk_id].text
            for chunk_id, turn_id in zip(
                owner_envelope.chunk_ids,
                owner_envelope.chunk_turn_ids,
                strict=True,
            )
            if turn_id == owner_envelope.opener_turn_id
            and index.row_by_chunk_id[chunk_id].role == "user"
        )
        parent_excerpt = str(neighborhood["parent_raw_text"])
        hidden_delta = owner_lead.replace(parent_excerpt, " ", 1)
        hidden_terms = frozenset(normalized_terms(hidden_delta))
        hidden_actions = linked_action_concepts(hidden_delta)
        hidden_query_score = _weighted_compatible_score(
            conditioned_question_terms, hidden_terms, anchor_idf
        )
        hidden_action_hits = len(terms["actions"] & set(hidden_actions))
        hidden_slot_hits = _compatible_hits(terms["slots"], hidden_terms)
        hidden_date_hit = int(bool(_DATE_TERM_RE.search(hidden_delta)))
        hidden_status_hit = int(bool(_STATUS_TERM_RE.search(hidden_delta)))
        hidden_token_count = count_tokens(hidden_delta.strip()) if hidden_delta.strip() else 0
        owner_score = (
            hidden_slot_hits,
            hidden_action_hits,
            hidden_query_score,
            hidden_date_hit,
            hidden_status_hit,
            -hidden_token_count,
            *base_owner_score,
        )
        owner_link = {
            "direction": "owner",
            "distance": 0,
            "owner_envelope_id": owner_id,
            "parent_logical_evidence_id": neighborhood["parent_logical_evidence_id"],
            "parent_position": neighborhood["parent_position"],
            "physical_anchor_chunk_id": neighborhood["physical_anchor_chunk_id"],
        }
        # A complete physical parent already exposes its owner bytes in G.
        # The owner-completion lane is reserved for excerpt-backed anchors,
        # where hydrating the full opener can reveal omitted same-turn facts.
        if (
            neighborhood["excerpt_backed"]
            or spec.required_evidence_role == "assistant"
        ):
            owner_choices.append(
                (
                    owner_score,
                    1,
                    owner_id,
                    owner_envelope,
                    {
                        **owner_audit,
                        "hidden_delta_action_hits": hidden_action_hits,
                        "hidden_delta_date_hit": hidden_date_hit,
                        "hidden_delta_query_score": hidden_query_score,
                        "hidden_delta_slot_hits": hidden_slot_hits,
                        "hidden_delta_status_hit": hidden_status_hit,
                        "hidden_delta_token_count": hidden_token_count,
                        "physical_anchor_links": [owner_link],
                    },
                )
            )
        neighbor_candidates = []
        for neighbor in neighborhood["neighbors"]:
            neighbor_id = str(neighbor["envelope_id"])
            neighbor_envelope, neighbor_audit = enriched_by_id[neighbor_id]
            neighbor_candidates.append(
                (
                    conditioned_score(neighborhood, neighbor_id),
                    neighbor_id,
                    neighbor_envelope,
                    neighbor_audit,
                    str(neighbor["direction"]),
                )
            )
        if neighbor_candidates:
            local_score, neighbor_id, neighbor_envelope, neighbor_audit, direction = max(
                neighbor_candidates,
                key=lambda item: (
                    item[0],
                    item[4] == "next"
                    if spec.temporal_mode is TemporalMode.LATEST_STATE
                    else item[4] == "previous",
                    item[1],
                ),
            )
            parent_terms = frozenset(
                normalized_terms(str(neighborhood["parent_raw_text"]))
            )
            candidate_terms = user_terms_by_envelope[neighbor_id]
            candidate_query_score = _weighted_compatible_score(
                conditioned_question_terms, candidate_terms, anchor_idf
            )
            novel_query_score = _weighted_compatible_score(
                conditioned_question_terms,
                candidate_terms - parent_terms,
                anchor_idf,
            )
            bridge_score = _weighted_compatible_score(
                frozenset(
                    term
                    for term in parent_terms
                    if term not in _CONDITIONED_STOP_TERMS and not term.isdigit()
                ),
                candidate_terms,
                anchor_idf,
            )
            temporal_direction_bonus = 1_500 * int(
                spec.temporal_mode is TemporalMode.LATEST_STATE
                and direction == "next"
            )
            transition_score = (
                3 * novel_query_score
                + 2 * candidate_query_score
                + min(bridge_score, 1_000)
                + 2 * int(neighbor_audit["lexical_lane_score"])
                + 200 * int(neighbor_audit["action_hits"])
                + 100 * int(neighbor_audit["date_hits"])
                + 100 * int(neighbor_audit["status_hits"])
                + temporal_direction_bonus
            )
            local_score = (transition_score, *local_score)
            neighbor_link = {
                "direction": direction,
                "distance": 1,
                "owner_envelope_id": owner_id,
                "parent_logical_evidence_id": neighborhood["parent_logical_evidence_id"],
                "parent_position": neighborhood["parent_position"],
                "physical_anchor_chunk_id": neighborhood["physical_anchor_chunk_id"],
            }
            transition_choices.append(
                (
                    local_score,
                    0,
                    neighbor_id,
                    neighbor_envelope,
                    {**neighbor_audit, "physical_anchor_links": [neighbor_link]},
                )
            )
            frozen_local_choices.append(
                {
                    "candidate_envelope_id": neighbor_id,
                    "direction": direction,
                    "local_candidate_count": len(neighbor_candidates),
                    "owner_envelope_id": owner_id,
                    "parent_logical_evidence_id": neighborhood["parent_logical_evidence_id"],
                    "physical_anchor_chunk_id": neighborhood["physical_anchor_chunk_id"],
                    "score_vector": list(local_score),
                }
            )

    def physical_order(
        values: Sequence[tuple[tuple[int, ...], int, str, object, dict[str, Any]]]
    ) -> list[tuple[str, object, dict[str, Any]]]:
        ordered = sorted(
            values,
            key=lambda item: (
                item[0],
                item[1],
                -int(item[4]["physical_anchor_links"][0]["parent_position"]),
                item[2],
            ),
            reverse=True,
        )
        return [
            (
                envelope_id,
                envelope,
                {
                    **audit,
                    "physical_candidate_chunk_count": sum(
                        1
                        for turn_id in envelope.chunk_turn_ids
                        if spec.required_evidence_role == "assistant"
                        or turn_id == envelope.opener_turn_id
                    ),
                    "physical_candidate_token_count": sum(
                        token_count
                        for token_count, turn_id in zip(
                            envelope.chunk_token_counts,
                            envelope.chunk_turn_ids,
                            strict=True,
                        )
                        if spec.required_evidence_role == "assistant"
                        or turn_id == envelope.opener_turn_id
                    ),
                    "physical_conditioned_score": list(score),
                    "physical_lane_rank": rank,
                },
            )
            for rank, (score, _excerpt, envelope_id, envelope, audit) in enumerate(
                ordered, 1
            )
        ]

    physical_owner_ranked = physical_order(owner_choices)
    physical_transition_ranked = physical_order(transition_choices)
    slot_proposals: list[tuple[int, str, str, object, dict[str, Any]]] = []
    for slot in spec.required_slots:
        ordered = sorted(
            enriched,
            key=lambda item: (
                int(item[2]["slot_lane_scores"].get(slot.slot_id, 0)),
                int(item[2]["lexical_lane_score"]),
                item[0],
            ),
            reverse=True,
        )
        for slot_rank, (envelope_id, envelope, audit) in enumerate(ordered, 1):
            slot_score = int(audit["slot_lane_scores"].get(slot.slot_id, 0))
            if slot_score <= 0:
                continue
            slot_proposals.append(
                (
                    slot_score,
                    slot.slot_id,
                    envelope_id,
                    envelope,
                    {**audit, "required_slot_lane_rank": slot_rank},
                )
            )
    required_slot_ranked = [
        (envelope_id, envelope, audit, f"required_slot:{slot_id}")
        for _score, slot_id, envelope_id, envelope, audit in sorted(
            slot_proposals, reverse=True
        )
    ]
    owner_lane = (
        "physical_anchor_owner",
        PHYSICAL_OWNER_LANE_EPISODES,
        [
            (envelope_id, envelope, audit, "physical_anchor_owner")
            for envelope_id, envelope, audit in physical_owner_ranked
        ],
    )
    transition_lane = (
        "physical_anchor_transition",
        PHYSICAL_TRANSITION_LANE_EPISODES,
        [
            (envelope_id, envelope, audit, "physical_anchor_transition")
            for envelope_id, envelope, audit in physical_transition_ranked
        ],
    )
    physical_lanes = (
        [transition_lane, owner_lane]
        if spec.temporal_mode is TemporalMode.LATEST_STATE
        else [owner_lane, transition_lane]
    )
    lane_queues: list[
        tuple[str, int, list[tuple[str, object, dict[str, Any], str]]]
    ] = [
        *physical_lanes,
        (
            "required_slot",
            SLOT_LANE_EPISODES,
            required_slot_ranked,
        ),
    ]
    for lane_name, score_name, budget in (
        ("query_user_lead", "lexical_lane_score", LEXICAL_LANE_EPISODES),
        ("source_local_anchor", "anchor_lane_score", ANCHOR_LANE_EPISODES),
        ("typed_operation", "typed_lane_score", TYPED_LANE_EPISODES),
    ):
        lane_queues.append(
            (
                lane_name,
                budget,
                [
                    (envelope_id, envelope, audit, lane_name)
                    for envelope_id, envelope, audit in lane_order(score_name)
                    if int(audit[score_name]) > 0
                ],
            )
        )

    # Lanes run in canonical specialist priority. A lane backfills past a
    # rejected candidate until it fills its own episode cap or exhausts its
    # deterministic ranking. Exact envelope-ID dedup is recorded afterward.
    selected_groups_by_envelope: dict[str, dict[str, Any]] = {}
    selected_group_order: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    selected_by_source: Counter[str] = Counter()
    lane_chunks: Counter[str] = Counter()
    lane_tokens: Counter[str] = Counter()
    lane_decisions: list[dict[str, Any]] = []
    used_chunks = used_tokens = 0
    for lane_name, lane_cap, queue in lane_queues:
        lane_selected = 0
        lane_token_cap = (
            PHYSICAL_OWNER_LANE_TOKEN_BUDGET
            if lane_name == "physical_anchor_owner"
            else PHYSICAL_TRANSITION_LANE_TOKEN_BUDGET
            if lane_name == "physical_anchor_transition"
            else EPISODE_LANE_TOKEN_BUDGET
        )
        lane_chunk_cap = (
            PHYSICAL_OWNER_LANE_CHUNK_BUDGET
            if lane_name == "physical_anchor_owner"
            else PHYSICAL_TRANSITION_LANE_CHUNK_BUDGET
            if lane_name == "physical_anchor_transition"
            else EPISODE_LANE_CHUNK_BUDGET
        )
        for lane_rank, (envelope_id, envelope, audit, selected_lane) in enumerate(
            queue, 1
        ):
            if lane_selected >= lane_cap:
                break
            decision = {
                "envelope_id": envelope_id,
                "lane": selected_lane,
                "lane_rank": lane_rank,
            }
            if selected_lane.startswith("physical_anchor_"):
                decision.update(
                    {
                        "candidate_chunk_count": audit["physical_candidate_chunk_count"],
                        "candidate_token_count": audit["physical_candidate_token_count"],
                        "conditioned_score": audit["physical_conditioned_score"],
                        "physical_anchor_links": audit["physical_anchor_links"],
                    }
                )
            existing = selected_groups_by_envelope.get(envelope_id)
            if existing is not None:
                existing["rank_audit"]["selected_lanes"].append(selected_lane)
                decision["decision"] = "selected_then_exact_envelope_dedup"
                lane_decisions.append(decision)
                continue
            if len(selected_groups_by_envelope) >= MAX_EPISODES:
                decision["decision"] = "rejected_global_episode_cap"
                lane_decisions.append(decision)
                continue
            if selected_by_source[str(envelope.source_id)] >= MAX_EPISODES_PER_SOURCE:
                decision["decision"] = "rejected_source_episode_cap"
                lane_decisions.append(decision)
                continue
            all_rows = [
                index.row_by_chunk_id[chunk_id] for chunk_id in envelope.chunk_ids
            ]
            if not all_rows or all_rows[0].role != "user":
                decision["decision"] = "rejected_missing_user_lead"
                lane_decisions.append(decision)
                continue
            opener_turn_id = str(envelope.opener_turn_id)
            lead_rows = [row for row in all_rows if row.turn_id == opener_turn_id]
            followers = [row for row in all_rows if row.turn_id != opener_turn_id]
            if (
                lane_name.startswith("physical_anchor_")
                and spec.required_evidence_role != "assistant"
            ):
                followers = []
            if spec.required_evidence_role == "assistant":
                followers.sort(
                    key=lambda row: (
                        row.role == "assistant",
                        _compatible_hits(terms["question"], normalized_terms(row.text)),
                        -abs(row.ordinal - lead_rows[-1].ordinal),
                        row.chunk_id,
                    ),
                    reverse=True,
                )
            rows = list(lead_rows)
            local_tokens = sum(row.token_count for row in rows)
            local_chunks = len(rows)
            if (
                lane_tokens[lane_name] + local_tokens > lane_token_cap
                or lane_chunks[lane_name] + local_chunks > lane_chunk_cap
            ):
                decision["decision"] = "rejected_lane_lead_budget"
                lane_decisions.append(decision)
                continue
            selected_turns = {row.turn_id for row in rows}
            truncation_reasons: list[str] = []
            for row in followers:
                if len(selected_turns | {row.turn_id}) > MAX_TURNS_PER_EPISODE:
                    truncation_reasons.append("turn_bound")
                    continue
                if lane_chunks[lane_name] + local_chunks + 1 > lane_chunk_cap:
                    truncation_reasons.append("lane_chunk_bound")
                    continue
                if lane_tokens[lane_name] + local_tokens + row.token_count > lane_token_cap:
                    truncation_reasons.append("lane_token_bound")
                    continue
                rows.append(row)
                local_chunks += 1
                local_tokens += row.token_count
                selected_turns.add(row.turn_id)
            if (
                spec.required_evidence_role == "assistant"
                and not any(row.role == "assistant" for row in rows)
            ):
                decision["decision"] = "rejected_required_assistant_not_hydrated"
                lane_decisions.append(decision)
                continue
            rows.sort(key=lambda row: (row.ordinal, row.turn_start_char, row.chunk_id))
            chunk_count = len(rows)
            token_count = sum(row.token_count for row in rows)
            if (
                used_chunks + chunk_count > MAX_EPISODE_RAW_CHUNKS
                or used_tokens + token_count > MAX_EPISODE_RAW_TOKENS
            ):
                decision["decision"] = "rejected_global_episode_raw_budget"
                lane_decisions.append(decision)
                continue
            opener = lead_rows[0]
            group_rows: list[dict[str, Any]] = []
            for row in rows:
                projected = {
                    "chunk_id": row.chunk_id,
                    "created_at": row.created_at,
                    "envelope_id": str(envelope.envelope_id),
                    "opener_user_chunk_id": opener.chunk_id,
                    "opener_user_turn_id": opener.turn_id,
                    "ordinal": row.ordinal,
                    "role": row.role,
                    "source_id": row.source_id,
                    "text": row.text,
                    "text_sha256": row.text_sha256,
                    "token_count": row.token_count,
                    "turn_id": row.turn_id,
                }
                _require(
                    row.role != "assistant"
                    or projected["opener_user_chunk_id"] == opener.chunk_id,
                    "assistant row lost its user lead",
                )
                group_rows.append(projected)
                selected_rows.append(projected)
            group = {
                "envelope_id": str(envelope.envelope_id),
                "opener_user_chunk_id": opener.chunk_id,
                "opener_user_turn_id": opener.turn_id,
                "physical_anchor_links": audit.get("physical_anchor_links", []),
                "rank_audit": {**audit, "selected_lanes": [selected_lane]},
                "row_count": len(group_rows),
                "rows": group_rows,
                "source_id": str(envelope.source_id),
                "token_count": token_count,
                "truncation_reasons": sorted(set(truncation_reasons)),
                "turn_ids": list(dict.fromkeys(row.turn_id for row in rows)),
            }
            selected_groups_by_envelope[envelope_id] = group
            selected_group_order.append(envelope_id)
            used_chunks += chunk_count
            used_tokens += token_count
            selected_by_source[str(envelope.source_id)] += 1
            lane_chunks[lane_name] += chunk_count
            lane_tokens[lane_name] += token_count
            lane_selected += 1
            decision["decision"] = "selected"
            lane_decisions.append(decision)
    selected_groups = []
    for envelope_id in selected_group_order:
        group = selected_groups_by_envelope[envelope_id]
        selected_groups.append({**group, "group_sha256": identity_sha256(group)})
    body = {
        "active_source_ids": list(active),
        "candidate_envelope_count": len(candidates),
        "candidate_ranking_sha256": identity_sha256(
            [
                item[2]
                for item in sorted(enriched, key=lambda item: item[0])
            ]
        ),
        "lane_budgets": {
            "episode_chunk_cap_each": EPISODE_LANE_CHUNK_BUDGET,
            "episode_token_cap_each": EPISODE_LANE_TOKEN_BUDGET,
            "physical_owner_chunk_cap": PHYSICAL_OWNER_LANE_CHUNK_BUDGET,
            "physical_owner_episode_cap": PHYSICAL_OWNER_LANE_EPISODES,
            "physical_owner_token_cap": PHYSICAL_OWNER_LANE_TOKEN_BUDGET,
            "physical_transition_chunk_cap": PHYSICAL_TRANSITION_LANE_CHUNK_BUDGET,
            "physical_transition_episode_cap": PHYSICAL_TRANSITION_LANE_EPISODES,
            "physical_transition_token_cap": PHYSICAL_TRANSITION_LANE_TOKEN_BUDGET,
            "query_user_lead_episode_cap": LEXICAL_LANE_EPISODES,
            "required_slot_total_episode_cap": SLOT_LANE_EPISODES,
            "source_local_anchor_episode_cap": ANCHOR_LANE_EPISODES,
            "typed_operation_episode_cap": TYPED_LANE_EPISODES,
        },
        "lane_used_chunks": dict(sorted(lane_chunks.items())),
        "lane_used_tokens": dict(sorted(lane_tokens.items())),
        "lane_decisions": lane_decisions,
        "lane_proposal_count": sum(len(queue) for _name, _cap, queue in lane_queues),
        "lane_rankings": [
            {
                "candidate_count": len(queue),
                "candidate_envelope_ids_sha256": identity_sha256(
                    [envelope_id for envelope_id, _envelope, _audit, _selected in queue]
                ),
                "episode_cap": cap,
                "lane": name,
            }
            for name, cap, queue in lane_queues
        ],
        "provider_calls": 0,
        "physical_anchor_diagnostics": physical_diagnostics,
        "physical_anchor_local_choices": frozen_local_choices,
        "physical_anchor_rankings": {
            lane_name: [
                {
                    "candidate_chunk_count": audit["physical_candidate_chunk_count"],
                    "candidate_token_count": audit["physical_candidate_token_count"],
                    "conditioned_score": audit["physical_conditioned_score"],
                    "envelope_id": envelope_id,
                    "physical_anchor_links": audit["physical_anchor_links"],
                    "rank": rank,
                    "source_id": str(envelope.source_id),
                }
                for rank, (envelope_id, envelope, audit) in enumerate(values, 1)
            ]
            for lane_name, values in (
                ("physical_anchor_owner", physical_owner_ranked),
                ("physical_anchor_transition", physical_transition_ranked),
            )
        },
        "candidate_index_binding": {
            "cache_receipt_sha256": index.cache_receipt_sha256,
            "index_receipt_sha256": identity_sha256(
                {
                    "cache_receipt_sha256": index.cache_receipt_sha256,
                    "envelopes": [
                        envelope.projection()
                        for envelope in sorted(
                            index.envelopes,
                            key=lambda value: value.envelope_id,
                        )
                    ],
                    "namespace_id": index.namespace_id,
                }
            ),
            "namespace_id": index.namespace_id,
        },
        "source_anchor_populations_sha256": identity_sha256(
            {
                source_id: sorted(values)
                for source_id, values in source_anchor_terms.items()
            }
        ),
        "sources_without_user_led_episode": [
            source_id for source_id in active if source_envelope_counts[source_id] == 0
        ],
        "selected_chunk_count": len(selected_rows),
        "selected_episode_count": len(selected_groups),
        "selected_groups": selected_groups,
        "selected_raw_token_count": used_tokens,
        "typed_spec_sha256": spec.receipt_sha256,
    }
    result = {**body, "receipt_sha256": identity_sha256(body)}
    assert_gold_blind(result, path="hot_v6_spine_episode_selection")
    return result


def _active_source_ids(parent_rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for row in parent_rows:
        source_id = str(row.get("source_id", ""))
        _require(bool(source_id), "parent row omitted source identity")
        if source_id not in seen:
            seen.add(source_id)
            result.append(source_id)
    return tuple(result)


def _candidate_rows(
    index: UserLedEnvelopeShadowIndex, active_source_ids: Sequence[str]
) -> list[dict[str, Any]]:
    active = set(active_source_ids)
    rows = [
        {
            "chunk_id": row.chunk_id,
            "created_at": row.created_at,
            "ordinal": row.ordinal,
            "role": row.role,
            "source_id": row.source_id,
            "text": row.text,
            "text_sha256": row.text_sha256,
            "token_count": row.token_count,
            "turn_id": row.turn_id,
        }
        for row in index.row_by_chunk_id.values()
        if row.source_id in active
    ]
    rows.sort(key=lambda row: (row["ordinal"], row["chunk_id"]))
    return rows


def _episode_fact_rows(
    groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project hydrated episode rows into the reusable exact-fact seam."""

    result: list[dict[str, Any]] = []
    for group in groups:
        envelope_id = str(group["envelope_id"])
        user_lead = str(group["opener_user_chunk_id"])
        for row in group["rows"]:
            result.append(
                {
                    "chunk_id": row["chunk_id"],
                    "created_at": row["created_at"],
                    "envelope_id": envelope_id,
                    "evidence_id": row["chunk_id"],
                    "exchange_id": envelope_id,
                    "raw_text": row["text"],
                    "raw_text_sha256": row["text_sha256"],
                    "role": row["role"],
                    "source_id": row["source_id"],
                    "user_lead_evidence_id": user_lead,
                }
            )
    _require(
        len({row["evidence_id"] for row in result}) == len(result),
        "hydrated episode evidence IDs repeat",
    )
    return result


def _compact_fact_selection(
    ledger: QueryFactLedger, *, selected_slice: object | None = None
) -> dict[str, Any]:
    """Apply the independent fact budget after the ledger is fully compiled."""

    selected_slice = selected_slice or select_and_render_query_fact_ledger(
        ledger, max_facts=MAX_FACTS, max_tokens=MAX_FACT_TOKENS
    )
    selected = [
        {**fact.projection(), "token_count": count_tokens(fact.exact_quote)}
        for fact in selected_slice.facts
    ]
    body = {
        "candidate_population_sha256": ledger.candidate_population_sha256,
        "compiled_fact_count": len(ledger.facts),
        "compiled_ledger_receipt_sha256": ledger.receipt_sha256,
        "fact_token_budget": MAX_FACT_TOKENS,
        "provider_calls": 0,
        "selected_fact_count": len(selected),
        "selected_fact_token_count": selected_slice.rendered_token_count,
        "selected_facts": selected,
        "selected_slice": selected_slice.projection(),
        "selected_population_sha256": ledger.selected_population_sha256,
        "selection_status": "selected_with_mandatory_coverage",
        "unresolved_slot_ids": list(ledger.unresolved_slot_ids),
    }
    result = {**body, "receipt_sha256": identity_sha256(body)}
    assert_gold_blind(result, path="hot_v6_compact_fact_selection")
    return result


def _empty_fact_selection(ledger: QueryFactLedger, *, status: str) -> dict[str, Any]:
    body = {
        "candidate_population_sha256": ledger.candidate_population_sha256,
        "compiled_fact_count": len(ledger.facts),
        "compiled_ledger_receipt_sha256": ledger.receipt_sha256,
        "fact_token_budget": MAX_FACT_TOKENS,
        "provider_calls": 0,
        "selected_fact_count": 0,
        "selected_fact_token_count": 0,
        "selected_facts": [],
        "selected_slice": None,
        "selected_population_sha256": ledger.selected_population_sha256,
        "selection_status": status,
        "unresolved_slot_ids": list(ledger.unresolved_slot_ids),
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _conserves_parent_sources(
    parent_rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[Mapping[str, Any]],
    active_source_ids: Sequence[str],
) -> tuple[bool, str]:
    if set(active_source_ids) != {str(row["source_id"]) for row in candidates}:
        return False, "active_source_partition_incomplete"
    for parent in parent_rows:
        raw = parent.get("raw_text")
        source_id = parent.get("source_id")
        if source_id not in active_source_ids:
            return False, "parent_source_escaped_active_partition"
        if not isinstance(raw, str) or not raw:
            return False, "parent_raw_quote_missing"
        raw_sha = parent.get("raw_text_sha256")
        if raw_sha is not None and raw_sha != quote_sha256(raw):
            return False, "parent_raw_quote_digest_mismatch"
    # Physical IDs are intentionally not required here. The fast packet may
    # contain content-addressed excerpts or graph/story rows whose logical ID
    # is not a frozen chunk ID. Their exact raw bytes remain protected in the
    # global lane; source conservation proves only that every activated opaque
    # source was exhaustively represented by the authenticated cache scan.
    return True, "validated"


def _evidence_line(row: Mapping[str, Any]) -> str:
    return f"[{row['created_at']} | {row['role']}] {row.get('raw_text', row.get('text'))}"


def _global_citation_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = [
        {
            "citation": f"G{position}",
            "created_at": str(row.get("created_at", "")),
            "evidence_id": str(row["evidence_id"]),
            "raw_text_sha256": str(
                row.get("raw_text_sha256")
                or quote_sha256(str(row.get("raw_text", "")))
            ),
            "role": str(row.get("role", "")),
            "row_sha256": identity_sha256(dict(row)),
            "source_id": str(row["source_id"]),
        }
        for position, row in enumerate(rows, 1)
    ]
    body = {
        "entries": entries,
        "render_order": "sealed_parent_order",
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _episode_global_match(
    episode_row: Mapping[str, Any], global_row: Mapping[str, Any]
) -> bool:
    raw = global_row.get("raw_text")
    return (
        str(global_row.get("evidence_id")) == str(episode_row["chunk_id"])
        and global_row.get("source_id") == episode_row.get("source_id")
        and global_row.get("role") == episode_row.get("role")
        and global_row.get("created_at") == episode_row.get("created_at")
        and raw == episode_row.get("text")
        and global_row.get("raw_text_sha256") == episode_row.get("text_sha256")
        and quote_sha256(str(raw)) == episode_row.get("text_sha256")
    )


def _normalized_projection_relationship(
    episode_text: str, global_text: str
) -> str | None:
    """Prove that a transformed G row is a bounded projection of physical E."""

    normalized_episode = " ".join(episode_text.split())
    normalized_global = " ".join(global_text.split())
    if normalized_global and normalized_global in normalized_episode:
        return "whitespace_normalized_contiguous_projection"
    segments = [" ".join(value.split()) for value in global_text.splitlines()]
    segments = [value for value in segments if value]
    if len(segments) < 2:
        return None
    position = 0
    for segment in segments:
        match = normalized_episode.find(segment, position)
        if match < 0:
            return None
        position = match + len(segment)
    return "whitespace_normalized_ordered_segment_projection"


def _episode_collision_binding(
    episode_row: Mapping[str, Any],
    global_row: Mapping[str, Any],
    *,
    global_citation: str,
) -> dict[str, Any] | None:
    """Bind a physical turn and a same-ID transformed G projection without dedup."""

    comparable = {
        "source_id": (episode_row.get("source_id"), global_row.get("source_id")),
        "role": (episode_row.get("role"), global_row.get("role")),
        "created_at": (episode_row.get("created_at"), global_row.get("created_at")),
        "text": (episode_row.get("text"), global_row.get("raw_text")),
        "text_sha256": (
            episode_row.get("text_sha256"),
            global_row.get("raw_text_sha256"),
        ),
    }
    mismatch_fields = [
        field
        for field, (episode_value, global_value) in comparable.items()
        if episode_value != global_value
    ]
    episode_text = episode_row.get("text")
    global_text = global_row.get("raw_text")
    if (
        str(global_row.get("evidence_id")) != str(episode_row.get("chunk_id"))
        or any(field in mismatch_fields for field in ("source_id", "role", "created_at"))
        or mismatch_fields != ["text", "text_sha256"]
        or type(episode_text) is not str
        or type(global_text) is not str
        or quote_sha256(episode_text) != episode_row.get("text_sha256")
        or quote_sha256(global_text) != global_row.get("raw_text_sha256")
    ):
        return None
    relationship = _normalized_projection_relationship(episode_text, global_text)
    if relationship is None:
        return None
    address_body = {
        "episode_evidence_id": str(episode_row["chunk_id"]),
        "episode_text_sha256": str(episode_row["text_sha256"]),
        "global_citation": global_citation,
        "global_evidence_id": str(global_row["evidence_id"]),
        "global_raw_text_sha256": str(global_row["raw_text_sha256"]),
        "projection_relationship": relationship,
    }
    address = f"episode-collision:{identity_sha256(address_body)}"
    body = {
        **address_body,
        "collision_address": address,
        "created_at": str(episode_row["created_at"]),
        "global_projection_text": global_text,
        "mismatch_fields": mismatch_fields,
        "role": str(episode_row["role"]),
        "source_id": str(episode_row["source_id"]),
    }
    return {**body, "collision_receipt_sha256": identity_sha256(body)}


def _collision_binding_valid(
    collision: Mapping[str, Any],
    episode_row: Mapping[str, Any],
    citation_entry: Mapping[str, Any],
) -> bool:
    unsigned = dict(collision)
    receipt = unsigned.pop("collision_receipt_sha256", None)
    address_body = {
        key: collision.get(key)
        for key in (
            "episode_evidence_id",
            "episode_text_sha256",
            "global_citation",
            "global_evidence_id",
            "global_raw_text_sha256",
            "projection_relationship",
        )
    }
    episode_text = episode_row.get("text")
    global_text = collision.get("global_projection_text")
    return (
        type(collision) is dict
        and receipt == identity_sha256(unsigned)
        and collision.get("collision_address")
        == f"episode-collision:{identity_sha256(address_body)}"
        and collision.get("mismatch_fields") == ["text", "text_sha256"]
        and collision.get("episode_evidence_id") == episode_row.get("chunk_id")
        and collision.get("episode_text_sha256") == episode_row.get("text_sha256")
        and collision.get("global_citation") == citation_entry.get("citation")
        and collision.get("global_evidence_id") == citation_entry.get("evidence_id")
        and collision.get("global_raw_text_sha256")
        == citation_entry.get("raw_text_sha256")
        and collision.get("source_id") == episode_row.get("source_id")
        == citation_entry.get("source_id")
        and collision.get("role") == episode_row.get("role")
        == citation_entry.get("role")
        and collision.get("created_at") == episode_row.get("created_at")
        == citation_entry.get("created_at")
        and type(episode_text) is str
        and quote_sha256(episode_text) == episode_row.get("text_sha256")
        and type(global_text) is str
        and quote_sha256(global_text) == collision.get("global_raw_text_sha256")
        and collision.get("projection_relationship")
        == _normalized_projection_relationship(episode_text, global_text)
    )


def _collision_partition_valid(manifest: Mapping[str, Any]) -> bool:
    collisions = manifest.get("representation_collisions")
    manifest_rows = manifest.get("manifest_rows")
    if type(collisions) is not list or type(manifest_rows) is not list:
        return False
    collision_rows = [
        row
        for row in manifest_rows
        if type(row) is dict
        and row.get("representation") == "episode_collision_raw"
    ]
    addresses = [collision.get("collision_address") for collision in collisions]
    return (
        all(type(collision) is dict for collision in collisions)
        and len(set(addresses)) == len(addresses)
        and [row.get("collision_ordinal") for row in collision_rows]
        == list(range(1, len(collisions) + 1))
        and [
            (row.get("collision_address"), row.get("chunk_id"))
            for row in collision_rows
        ]
        == [
            (
                collision.get("collision_address"),
                collision.get("episode_evidence_id"),
            )
            for collision in collisions
        ]
    )


def _episode_manifest(
    group: Mapping[str, Any],
    global_by_id: Mapping[str, Mapping[str, Any]],
    *,
    global_citations: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Represent every selected episode row as exact E raw or an exact G ref."""

    raw_rows: list[dict[str, Any]] = []
    global_refs: list[dict[str, Any]] = []
    representation_collisions: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for source in group["rows"]:
        row = copy.deepcopy(dict(source))
        chunk_id = str(row["chunk_id"])
        global_row = global_by_id.get(chunk_id)
        if global_row is None:
            raw_rows.append(row)
            manifest_rows.append({"chunk_id": chunk_id, "representation": "episode_raw"})
            continue
        if not _episode_global_match(row, global_row):
            citation = (global_citations or {}).get(chunk_id)
            if citation is None:
                return None, f"episode_global_collision_citation_missing:{chunk_id}"
            collision = _episode_collision_binding(
                row, global_row, global_citation=citation
            )
            if collision is None:
                return None, f"episode_global_exact_id_mismatch:{chunk_id}"
            raw_rows.append(row)
            representation_collisions.append(collision)
            manifest_rows.append(
                {
                    "chunk_id": chunk_id,
                    "collision_address": str(collision["collision_address"]),
                    "collision_ordinal": len(representation_collisions),
                    "representation": "episode_collision_raw",
                }
            )
            continue
        reference = {
            "episode_chunk_id": chunk_id,
            "global_evidence_id": str(global_row["evidence_id"]),
            "global_raw_text_sha256": str(global_row["raw_text_sha256"]),
        }
        global_refs.append(reference)
        manifest_rows.append({"chunk_id": chunk_id, "representation": "global_ref"})
    body = {
        "envelope_id": str(group["envelope_id"]),
        "global_refs": global_refs,
        "manifest_rows": manifest_rows,
        "opener_user_chunk_id": str(group["opener_user_chunk_id"]),
        "raw_rows": raw_rows,
        "representation_collisions": representation_collisions,
        "source_id": str(group["source_id"]),
    }
    _require(
        [row["chunk_id"] for row in manifest_rows]
        == [str(row["chunk_id"]) for row in group["rows"]],
        "episode manifest changed atomic row order",
    )
    return {**body, "manifest_sha256": identity_sha256(body)}, None


def _select_global_raw(
    parent_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Preserve the complete sealed Operation-A parent, byte rows in order."""

    retained = [copy.deepcopy(dict(raw)) for raw in parent_rows]
    _require(
        len({str(row["evidence_id"]) for row in retained}) == len(retained),
        "sealed parent evidence IDs repeat",
    )
    return retained, [], []


def _operation_a_reference(source_arm: Mapping[str, Any]) -> dict[str, Any]:
    messages = replace_system_prompt_only(source_arm["provider_messages"])
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    prompt_tokens = hot.count_chat_prompt_token_proxy(messages)
    body = {
        "context_token_proxy": int(source_arm["context_token_proxy"]),
        "packed_chunk_ids": list(source_arm["packed_chunk_ids"]),
        "parent_population_sha256": identity_sha256(source_arm["packed_evidence"]),
        "prompt_token_proxy": prompt_tokens,
        "prompt_workspace_token_proxy": prompt_tokens + OUTPUT_TOKEN_RESERVE,
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _render_context(
    parent_rows: Sequence[Mapping[str, Any]],
    episode_groups: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]] | str,
) -> str:
    sections: list[str] = []
    for block, row in enumerate(parent_rows, 1):
        # G ordinals are an unambiguous compact citation into the sealed
        # rendered_parent_evidence_ids array. Repeating 64-byte IDs and opaque
        # source addresses inside every provider line would consume the entire
        # monotone overlay allowance before any episode could be admitted.
        lines = [f"<G{block}>"]
        lines.append(_evidence_line(row))
        sections.append("\n".join(lines))
    for block, group in enumerate(episode_groups, 1):
        lines = [
            f"<E{block} envelope={group.get('envelope_id', 'unknown')} "
            f"source={group.get('source_id', 'unknown')} "
            f"owner_user={group.get('opener_user_chunk_id', 'unknown')}>"
        ]
        raw_rows = group.get("raw_rows", group.get("rows", []))
        raw_by_id = {str(row["chunk_id"]): row for row in raw_rows}
        refs_by_id = {
            str(reference["episode_chunk_id"]): reference
            for reference in group.get("global_refs", [])
        }
        collisions_by_id = {
            str(collision["episode_evidence_id"]): collision
            for collision in group.get("representation_collisions", [])
        }
        manifest_rows = group.get(
            "manifest_rows",
            [
                {"chunk_id": str(row["chunk_id"]), "representation": "episode_raw"}
                for row in raw_rows
            ],
        )
        for manifest_row in manifest_rows:
            chunk_id = str(manifest_row["chunk_id"])
            if manifest_row["representation"] == "global_ref":
                reference = refs_by_id[chunk_id]
                lines.append(
                    f"<GREF episode_evidence={reference['episode_chunk_id']} "
                    f"global_evidence={reference['global_evidence_id']} "
                    f"sha256={reference['global_raw_text_sha256']}>"
                )
                continue
            row = raw_by_id[chunk_id]
            owner = "U" if row["role"] == "assistant" else "self"
            label = "U" if row["role"] == "user" else "A" if row["role"] == "assistant" else "X"
            collision = collisions_by_id.get(chunk_id)
            collision_label = ""
            if manifest_row["representation"] == "episode_collision_raw":
                _require(collision is not None, "episode collision binding missing")
                collision_label = (
                    f" projection_of={collision['global_citation']}"
                    f" collision=C{manifest_row['collision_ordinal']}"
                )
            lines.append(
                f"<{label} evidence={row['chunk_id']} source={row['source_id']} "
                f"envelope={group.get('envelope_id', row.get('envelope_id', 'unknown'))} "
                f"owner={owner}{collision_label}> {_evidence_line(row)}"
            )
        sections.append("\n".join(lines))
    if isinstance(facts, str):
        if facts:
            sections.append(facts)
    elif facts:
        lines = ["<FACT_LEDGER>"]
        for number, fact in enumerate(facts, 1):
            owner = str(fact["user_lead_evidence_id"])
            lines.append(
                f"<F{number} fact={fact['fact_id']} "
                f"backs={fact['backing_evidence_id']} "
                f"source={fact['backing_source_id']} "
                f"exchange={fact.get('exchange_id')} "
                f"envelope={fact.get('envelope_id')} "
                f"role={fact['source_role']} owner_user={owner}> "
                f"[{fact['source_created_at']}] {fact['exact_quote']}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _prompt(question: str, context: str) -> tuple[list[dict[str, str]], int, int]:
    messages = [
        {"role": "system", "content": OPERATION_AWARE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=context, question=question),
        },
    ]
    context_tokens = count_tokens(context)
    workspace_tokens = hot.count_chat_prompt_token_proxy(messages) + OUTPUT_TOKEN_RESERVE
    return messages, context_tokens, workspace_tokens


def _exact_operation_a_fallback(
    source_arm: Mapping[str, Any],
    reason: str,
    *,
    candidate_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reference = _operation_a_reference(source_arm)
    parent_rows = source_arm["packed_evidence"]
    return {
        "active_source_ids": list(_active_source_ids(source_arm["packed_evidence"])),
        "candidate_universe_binding": (
            copy.deepcopy(dict(candidate_binding)) if candidate_binding else None
        ),
        "context_token_proxy": reference["context_token_proxy"],
        "dedup_stage": "post_selection_exact_evidence_id",
        "episode_selection": None,
        "fact_ledger": None,
        "fallback_reason": reason,
        "mode": "exact_operation_a_raw_fail_open",
        "operation_a_reference": reference,
        "packed_chunk_ids": list(source_arm["packed_chunk_ids"]),
        "parent_population_sha256": identity_sha256(source_arm["packed_evidence"]),
        "parent_rows_protected": True,
        "parent_all_rendered": True,
        "prompt_token_proxy": reference["prompt_token_proxy"],
        "prompt_workspace_token_proxy": reference["prompt_workspace_token_proxy"],
        "provider_messages": reference["provider_messages"],
        "provider_payload_sha256": reference["provider_payload_sha256"],
        "provider_payload_utf8_bytes": reference["provider_payload_utf8_bytes"],
        "raw_collision_bindings": [],
        "raw_dedup_bindings": [],
        "global_raw_omitted_evidence_ids": [],
        "global_citation_manifest": _global_citation_manifest(parent_rows),
        "global_raw_selected_evidence_ids": list(source_arm["packed_chunk_ids"]),
        "raw_selected_chunk_ids": [],
        "rendered_fact_ids": [],
        "rendered_parent_evidence_ids": [row["evidence_id"] for row in source_arm["packed_evidence"]],
        "rendered_raw_chunk_ids": [],
        "source_conservation_validated": False,
    }


def _compose_arm(
    source_arm: Mapping[str, Any],
    *,
    dated_question: str,
    index: UserLedEnvelopeShadowIndex,
    candidate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    parent = [copy.deepcopy(row) for row in source_arm["packed_evidence"]]
    active = _active_source_ids(parent)
    candidates = _candidate_rows(index, active)
    conserved, reason = _conserves_parent_sources(
        parent, candidates=candidates, active_source_ids=active
    )
    if not conserved:
        return _exact_operation_a_fallback(
            source_arm, reason, candidate_binding=candidate_binding
        )
    spec = compile_typed_operator_spec(dated_question)
    episode_selection = select_spine_episodes(
        index,
        active_source_ids=active,
        dated_question=dated_question,
        parent_packed_rows=parent,
        typed_spec=spec,
    )
    global_rows, global_omitted, raw_dedup = _select_global_raw(parent)
    operation_a_reference = _operation_a_reference(source_arm)
    _require(not global_omitted, "monotone parent lane omitted a sealed row")
    global_by_id = {str(row["evidence_id"]): row for row in global_rows}
    global_citations = {
        str(row["evidence_id"]): f"G{position}"
        for position, row in enumerate(global_rows, 1)
    }
    groups = list(episode_selection["selected_groups"])
    admitted_groups: list[Mapping[str, Any]] = []
    admitted_manifests: list[dict[str, Any]] = []
    rejected_manifests: list[dict[str, str]] = []
    for group in groups:
        manifest, manifest_error = _episode_manifest(
            group, global_by_id, global_citations=global_citations
        )
        if manifest_error is not None:
            return _exact_operation_a_fallback(
                source_arm, manifest_error, candidate_binding=candidate_binding
            )
        _require(manifest is not None, "episode manifest unexpectedly absent")
        context = _render_context(
            global_rows,
            [*admitted_manifests, manifest],
            [],
        )
        _messages, context_tokens, workspace_tokens = _prompt(dated_question, context)
        if context_tokens <= MAX_CONTEXT_TOKENS and workspace_tokens <= MAX_WORKSPACE_TOKENS:
            admitted_groups.append(group)
            admitted_manifests.append(manifest)
        else:
            rejected_manifests.append(
                {
                    "envelope_id": str(group["envelope_id"]),
                    "reason": "hard_prompt_cap",
                }
            )
    if not admitted_groups:
        return _exact_operation_a_fallback(
            source_arm,
            "no_admissible_user_led_episode",
            candidate_binding=candidate_binding,
        )

    # Facts are compiled only after atomic episode manifests are admitted.
    # Candidate-source cache rows never bypass the episode gate.
    ledger = compile_query_fact_ledger(
        dated_question,
        _episode_fact_rows(admitted_groups),
    )
    base_context = _render_context(global_rows, admitted_manifests, [])
    _base_messages, base_context_tokens, base_workspace_tokens = _prompt(
        dated_question, base_context
    )
    fact_allowance = min(
        MAX_FACT_TOKENS,
        MAX_CONTEXT_TOKENS - base_context_tokens - 4,
        MAX_WORKSPACE_TOKENS - base_workspace_tokens - 4,
    )
    fact_slice = None
    fact_status = "mandatory_coverage_did_not_fit"
    while fact_allowance > 0:
        try:
            candidate_slice = select_and_render_query_fact_ledger(
                ledger,
                max_facts=MAX_FACTS,
                max_tokens=fact_allowance,
            )
        except QueryFactLedgerError:
            break
        candidate_context = _render_context(
            global_rows, admitted_manifests, candidate_slice.text
        )
        _candidate_messages, candidate_context_tokens, candidate_workspace_tokens = _prompt(
            dated_question, candidate_context
        )
        if (
            candidate_context_tokens <= MAX_CONTEXT_TOKENS
            and candidate_workspace_tokens <= MAX_WORKSPACE_TOKENS
        ):
            fact_slice = candidate_slice
            fact_status = "selected_with_mandatory_coverage"
            break
        overflow = max(
            candidate_context_tokens - MAX_CONTEXT_TOKENS,
            candidate_workspace_tokens - MAX_WORKSPACE_TOKENS,
            1,
        )
        fact_allowance -= overflow + 4
    fact_ledger = (
        _compact_fact_selection(ledger, selected_slice=fact_slice)
        if fact_slice is not None
        else _empty_fact_selection(ledger, status=fact_status)
    )
    represented_ids = set(global_by_id)
    represented_ids.update(
        str(row["chunk_id"])
        for manifest in admitted_manifests
        for row in manifest["raw_rows"]
    )
    represented_ids.update(
        str(reference["episode_chunk_id"])
        for manifest in admitted_manifests
        for reference in manifest["global_refs"]
    )
    if fact_slice is not None and any(
        fact.backing_evidence_id not in represented_ids for fact in fact_slice.facts
    ):
        return _exact_operation_a_fallback(
            source_arm,
            "fact_backing_raw_not_rendered",
            candidate_binding=candidate_binding,
        )
    context = _render_context(
        global_rows,
        admitted_manifests,
        fact_slice.text if fact_slice is not None else [],
    )
    messages, context_tokens, workspace_tokens = _prompt(dated_question, context)
    if context_tokens > MAX_CONTEXT_TOKENS or workspace_tokens > MAX_WORKSPACE_TOKENS:
        raise ValueError("admitted raw packet exceeded its hard prompt cap")
    payload = hot._canonical_json_bytes({"messages": messages})  # noqa: SLF001
    prompt_tokens = hot.count_chat_prompt_token_proxy(messages)
    raw_ids = [str(row["chunk_id"]) for group in groups for row in group["rows"]]
    rendered_raw_ids = [
        str(row["chunk_id"])
        for manifest in admitted_manifests
        for row in manifest["raw_rows"]
    ]
    raw_dedup = [
        {
            "address": str(reference["episode_chunk_id"]),
            "excluded_episode_chunk_id": str(reference["episode_chunk_id"]),
            "retained_global_evidence_id": str(reference["global_evidence_id"]),
        }
        for manifest in admitted_manifests
        for reference in manifest["global_refs"]
    ]
    return {
        "active_source_ids": list(active),
        "candidate_universe_binding": copy.deepcopy(dict(candidate_binding)),
        "context_token_proxy": context_tokens,
        "dedup_stage": "post_independent_selection_exact_evidence_id",
        "episode_selection": episode_selection,
        "episode_manifests": admitted_manifests,
        "episode_manifest_rejections": rejected_manifests,
        "fact_ledger": fact_ledger,
        "fallback_reason": None,
        "mode": "spine_indexed_episodic_fact_ledger",
        "operation_a_reference": operation_a_reference,
        "packed_chunk_ids": [
            *[row["evidence_id"] for row in global_rows],
            *rendered_raw_ids,
        ],
        "parent_population_sha256": identity_sha256(parent),
        "parent_all_rendered": True,
        "parent_rows_protected": True,
        "prompt_token_proxy": prompt_tokens,
        "prompt_workspace_token_proxy": workspace_tokens,
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
        "raw_collision_bindings": [
            copy.deepcopy(collision)
            for manifest in admitted_manifests
            for collision in manifest["representation_collisions"]
        ],
        "raw_dedup_bindings": raw_dedup,
        "global_raw_omitted_evidence_ids": global_omitted,
        "global_citation_manifest": _global_citation_manifest(global_rows),
        "global_raw_selected_evidence_ids": [
            str(row["evidence_id"]) for row in global_rows
        ],
        "raw_selected_chunk_ids": raw_ids,
        "rendered_fact_ids": list(fact_slice.fact_ids) if fact_slice else [],
        "rendered_parent_evidence_ids": [
            str(row["evidence_id"]) for row in global_rows
        ],
        "rendered_raw_chunk_ids": rendered_raw_ids,
        "rendered_partitions": {
            "episode_global_ref_count": sum(
                len(manifest["global_refs"]) for manifest in admitted_manifests
            ),
            "episode_manifest_count": len(admitted_manifests),
            "episode_raw_chunk_count": len(rendered_raw_ids),
            "episode_representation_collision_count": sum(
                len(manifest["representation_collisions"])
                for manifest in admitted_manifests
            ),
            "fact_count": len(fact_slice.fact_ids) if fact_slice else 0,
            "global_parent_count": len(global_rows),
        },
        "overlay_context_token_delta": context_tokens
        - int(operation_a_reference["context_token_proxy"]),
        "overlay_workspace_token_delta": workspace_tokens
        - int(operation_a_reference["prompt_workspace_token_proxy"]),
        "source_conservation_validated": True,
    }


def _project_selection(
    construction: Mapping[str, Any],
    *,
    construction_sha256: str,
    runtime_sha256: str,
    replay_sha256: str,
    retrieval_path: Path,
    store_root: Path,
) -> dict[str, Any]:
    rows = construction.get("questions")
    _require(type(rows) is list and len(rows) == EXPECTED_QUESTION_COUNT, "source population changed")
    context = load_locked_query_expansion_context(
        retrieval_path,
        store_root=store_root,
        expected_retrieval_sha256=shadow.EXPECTED_RETRIEVAL_SHA256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    population_by_question = {
        str(row.source.packet.question_id): row for row in context.population.rows
    }
    indexes: dict[str, UserLedEnvelopeShadowIndex] = {}
    candidate_bindings: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    for ordinal, source_row in enumerate(rows):
        _require(
            source_row.get("ordinal") == ordinal
            and type(source_row.get("question_id")) is str,
            f"source order changed at {ordinal}",
        )
        unsigned = dict(source_row)
        source_receipt = unsigned.pop("row_receipt_sha256", None)
        _require(source_receipt == identity_sha256(unsigned), f"source receipt changed at {ordinal}")
        population_row = population_by_question[str(source_row["question_id"])]
        namespace = population_row.namespace
        namespace_id = str(namespace.namespace_id)
        if namespace_id not in indexes:
            cache, timing = shadow._build_cache(context, namespace)  # noqa: SLF001
            indexes[namespace_id] = build_user_led_envelope_shadow_index(cache)
            candidate_bindings[namespace_id] = {
                "cache": cache.projection(),
                "cache_build_binding": {
                    key: value
                    for key, value in timing.items()
                    if key != "cache_build_ns"
                },
                "index": indexes[namespace_id].projection(),
                "namespace_id": namespace_id,
                "store": {
                    "database_sha256": cache.source_database_sha256,
                    "physical_store_row_count": cache.physical_store_row_count,
                    "store_receipt_sha256": cache.source_store_receipt_sha256,
                },
            }
        source_arm = source_row["effective_arm"]
        dated_question = typed._extract_dated_question(source_arm)  # noqa: SLF001
        _require(
            quote_sha256(dated_question) == source_row["prompt_question_sha256"],
            f"question binding changed at {ordinal}",
        )
        arm = _compose_arm(
            source_arm,
            dated_question=dated_question,
            index=indexes[namespace_id],
            candidate_binding=candidate_bindings[namespace_id],
        )
        body = {
            "arms": {"a3_protected_union": arm},
            "format": ROW_FORMAT,
            "local_ordinal": ordinal % 10,
            "namespace_id": namespace_id,
            "ordinal": ordinal,
            "prompt_question_sha256": source_row["prompt_question_sha256"],
            "question_id": source_row["question_id"],
            "shard_offset": ordinal - ordinal % 10,
            "source_row_receipt_sha256": source_receipt,
        }
        output_rows.append({**body, "row_receipt_sha256": identity_sha256(body)})
    aggregate = {
        "exact_operation_a_fail_open_count": sum(
            row["arms"]["a3_protected_union"]["mode"]
            == "exact_operation_a_raw_fail_open"
            for row in output_rows
        ),
        "max_context_token_proxy": max(
            row["arms"]["a3_protected_union"]["context_token_proxy"]
            for row in output_rows
        ),
        "max_prompt_workspace_token_proxy": max(
            row["arms"]["a3_protected_union"]["prompt_workspace_token_proxy"]
            for row in output_rows
        ),
        "global_raw_omitted_evidence_count": sum(
            len(
                row["arms"]["a3_protected_union"][
                    "global_raw_omitted_evidence_ids"
                ]
            )
            for row in output_rows
        ),
        "global_raw_selected_evidence_count": sum(
            len(
                row["arms"]["a3_protected_union"][
                    "global_raw_selected_evidence_ids"
                ]
            )
            for row in output_rows
        ),
        "parent_all_rendered_count": sum(
            bool(row["arms"]["a3_protected_union"].get("parent_all_rendered"))
            for row in output_rows
        ),
        "rendered_episode_raw_chunk_count": sum(
            len(row["arms"]["a3_protected_union"]["rendered_raw_chunk_ids"])
            for row in output_rows
        ),
        "rendered_fact_count": sum(
            len(row["arms"]["a3_protected_union"]["rendered_fact_ids"])
            for row in output_rows
        ),
    }
    selection = {
        "aggregate": aggregate,
        "budget": {
            "anchor_lane_episode_cap": ANCHOR_LANE_EPISODES,
            "episode_chunk_cap_each_lane": EPISODE_LANE_CHUNK_BUDGET,
            "episode_raw_chunk_cap": MAX_EPISODE_RAW_CHUNKS,
            "episode_raw_token_cap": MAX_EPISODE_RAW_TOKENS,
            "episode_token_cap_each_lane": EPISODE_LANE_TOKEN_BUDGET,
            "fact_count_cap": MAX_FACTS,
            "fact_token_cap": MAX_FACT_TOKENS,
            "hard_context_token_cap": MAX_CONTEXT_TOKENS,
            "hard_workspace_token_cap": MAX_WORKSPACE_TOKENS,
            "lexical_lane_episode_cap": LEXICAL_LANE_EPISODES,
            "max_episodes_per_source": MAX_EPISODES_PER_SOURCE,
            "physical_owner_lane_chunk_cap": PHYSICAL_OWNER_LANE_CHUNK_BUDGET,
            "physical_owner_lane_episode_cap": PHYSICAL_OWNER_LANE_EPISODES,
            "physical_owner_lane_token_cap": PHYSICAL_OWNER_LANE_TOKEN_BUDGET,
            "physical_transition_lane_chunk_cap": PHYSICAL_TRANSITION_LANE_CHUNK_BUDGET,
            "physical_transition_lane_episode_cap": PHYSICAL_TRANSITION_LANE_EPISODES,
            "physical_transition_lane_token_cap": PHYSICAL_TRANSITION_LANE_TOKEN_BUDGET,
            "required_slot_lane_episode_cap": SLOT_LANE_EPISODES,
            "typed_lane_episode_cap": TYPED_LANE_EPISODES,
        },
        "format": FORMAT,
        "candidate_universe_bindings": [
            candidate_bindings[key] for key in sorted(candidate_bindings)
        ],
        "gold_fields_present": False,
        "implementation": _implementation_identity(),
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "provider_calls": 0,
        "question_count": len(output_rows),
        "questions": output_rows,
        "retrieval_sha256": file_sha256(retrieval_path),
        "source_construction_sha256": construction_sha256,
        "source_replay_sha256": replay_sha256,
        "source_runtime_sha256": runtime_sha256,
        "status": "sealed_gold_free_spine_indexed_episodic_fact_packets",
    }
    assert_gold_blind(selection, path="hot_v6_spine_episode_fact_selection")
    return selection


def _load_selection(output_root: Path) -> tuple[dict[str, Any], str]:
    """Load and authenticate one sealed r3 provider-free selection."""

    selection, digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / SELECTION_NAME
    )
    rows = selection.get("questions")
    bindings = selection.get("candidate_universe_bindings")
    _require(
        selection.get("format") == FORMAT
        and selection.get("status")
        == "sealed_gold_free_spine_indexed_episodic_fact_packets"
        and selection.get("gold_fields_present") is False
        and selection.get("provider_calls") == 0
        and selection.get("population_identity_sha256")
        == EXPECTED_POPULATION_SHA256
        and selection.get("retrieval_sha256") == shadow.EXPECTED_RETRIEVAL_SHA256
        and selection.get("source_construction_sha256")
        == source_assay.EXPECTED_SOURCE_CONSTRUCTION_SHA256
        and selection.get("source_runtime_sha256")
        == source_assay.EXPECTED_SOURCE_RUNTIME_SHA256
        and selection.get("source_replay_sha256")
        == source_assay.EXPECTED_SOURCE_REPLAY_SHA256
        and selection.get("implementation") == _implementation_identity()
        and type(rows) is list
        and len(rows) == EXPECTED_QUESTION_COUNT
        and selection.get("question_count") == len(rows)
        and type(bindings) is list
        and bool(bindings),
        "sealed r3 selection header changed",
    )
    binding_by_namespace: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        _require(type(binding) is dict, "candidate binding type changed")
        namespace_id = str(binding.get("namespace_id", ""))
        cache = binding.get("cache")
        index = binding.get("index")
        store = binding.get("store")
        _require(
            len(namespace_id) == 64
            and type(cache) is dict
            and type(index) is dict
            and type(store) is dict
            and cache.get("namespace_id") == namespace_id
            and index.get("namespace_id") == namespace_id
            and index.get("cache_receipt_sha256")
            == cache.get("cache_receipt_sha256")
            and store.get("database_sha256")
            == cache.get("source_database_sha256")
            and store.get("store_receipt_sha256")
            == cache.get("source_store_receipt_sha256")
            and store.get("physical_store_row_count")
            == cache.get("physical_store_row_count"),
            "candidate cache/index/store binding changed",
        )
        for projection, receipt_name in (
            (cache, "cache_receipt_sha256"),
            (index, "receipt_sha256"),
        ):
            unsigned = dict(projection)
            receipt = unsigned.pop(receipt_name, None)
            _require(
                receipt == identity_sha256(unsigned),
                f"candidate {receipt_name} changed",
            )
        _require(namespace_id not in binding_by_namespace, "namespace binding repeated")
        binding_by_namespace[namespace_id] = binding
    question_ids: set[str] = set()
    for ordinal, row in enumerate(rows):
        _require(type(row) is dict, f"r3 row type changed at {ordinal}")
        unsigned = dict(row)
        receipt = unsigned.pop("row_receipt_sha256", None)
        arm = row.get("arms", {}).get("a3_protected_union")
        namespace_id = str(row.get("namespace_id", ""))
        _require(
            receipt == identity_sha256(unsigned)
            and row.get("format") == ROW_FORMAT
            and row.get("ordinal") == ordinal
            and row.get("local_ordinal") == ordinal % 10
            and row.get("shard_offset") == ordinal - ordinal % 10
            and type(arm) is dict
            and arm.get("candidate_universe_binding")
            == binding_by_namespace.get(namespace_id),
            f"r3 row receipt/binding changed at {ordinal}",
        )
        question_id = str(row.get("question_id", ""))
        _require(question_id and question_id not in question_ids, "question identity repeated")
        question_ids.add(question_id)
        payload = hot._canonical_json_bytes(  # noqa: SLF001
            {"messages": arm.get("provider_messages")}
        )
        messages = arm.get("provider_messages")
        _require(
            type(messages) is list
            and len(messages) == 2
            and messages[0] == {
                "role": "system",
                "content": OPERATION_AWARE_SYSTEM_PROMPT,
            }
            and type(messages[1]) is dict
            and set(messages[1]) == {"role", "content"}
            and messages[1].get("role") == "user"
            and type(messages[1].get("content")) is str,
            f"r3 provider envelope changed at {ordinal}",
        )
        dated_question = typed._extract_dated_question(arm)  # noqa: SLF001
        user_content = str(messages[1]["content"])
        if arm.get("mode") == "spine_indexed_episodic_fact_ledger":
            prefix, suffix_template = USER_TEMPLATE.split("{context}", 1)
            suffix = suffix_template.format(question=dated_question)
            _require(
                user_content.startswith(prefix) and user_content.endswith(suffix),
                f"r3 user prompt framing changed at {ordinal}",
            )
            rendered_context = user_content[len(prefix) : -len(suffix)]
            context_tokens = count_tokens(rendered_context)
        else:
            reference = arm.get("operation_a_reference")
            _require(
                type(reference) is dict
                and reference.get("provider_messages") == messages,
                f"r3 fail-open reference changed at {ordinal}",
            )
            context_tokens = reference.get("context_token_proxy")
        prompt_tokens = hot.count_chat_prompt_token_proxy(messages)
        _require(
            arm.get("mode")
            in {
                "spine_indexed_episodic_fact_ledger",
                "exact_operation_a_raw_fail_open",
            }
            and
            arm.get("provider_payload_sha256")
            == hashlib.sha256(payload).hexdigest()
            and arm.get("provider_payload_utf8_bytes") == len(payload)
            and type(arm.get("context_token_proxy")) is int
            and type(context_tokens) is int
            and arm.get("context_token_proxy") == context_tokens
            and context_tokens <= MAX_CONTEXT_TOKENS
            and type(arm.get("prompt_token_proxy")) is int
            and arm.get("prompt_token_proxy") == prompt_tokens
            and type(arm.get("prompt_workspace_token_proxy")) is int
            and arm.get("prompt_workspace_token_proxy")
            == prompt_tokens + OUTPUT_TOKEN_RESERVE
            and arm.get("prompt_workspace_token_proxy") <= MAX_WORKSPACE_TOKENS
            and quote_sha256(dated_question) == row.get("prompt_question_sha256")
            and arm.get("parent_all_rendered") is True
            and arm.get("parent_rows_protected") is True
            and arm.get("global_raw_omitted_evidence_ids") == []
            and arm.get("global_raw_selected_evidence_ids")
            == arm.get("rendered_parent_evidence_ids"),
            f"r3 packet integrity changed at {ordinal}",
        )
        citation_manifest = arm.get("global_citation_manifest")
        _require(type(citation_manifest) is dict, "global citation manifest changed")
        citation_unsigned = dict(citation_manifest)
        citation_receipt = citation_unsigned.pop("receipt_sha256", None)
        citation_entries = citation_manifest.get("entries")
        _require(
            citation_receipt == identity_sha256(citation_unsigned)
            and citation_manifest.get("render_order") == "sealed_parent_order"
            and type(citation_entries) is list
            and [entry.get("citation") for entry in citation_entries]
            == [f"G{position}" for position in range(1, len(citation_entries) + 1)]
            and [entry.get("evidence_id") for entry in citation_entries]
            == arm.get("rendered_parent_evidence_ids"),
            f"global citation order changed at {ordinal}",
        )
        citation_by_label = {
            str(entry["citation"]): entry for entry in citation_entries
        }
        if arm.get("mode") == "spine_indexed_episodic_fact_ledger":
            represented = set(arm["rendered_parent_evidence_ids"])
            sealed_collisions: list[dict[str, Any]] = []
            for manifest in arm.get("episode_manifests", []):
                manifest_unsigned = dict(manifest)
                manifest_sha = manifest_unsigned.pop("manifest_sha256", None)
                raw_by_id = {
                    str(raw["chunk_id"]): raw for raw in manifest["raw_rows"]
                }
                collisions = manifest.get("representation_collisions")
                collision_by_address = {
                    str(collision.get("collision_address")): collision
                    for collision in collisions or []
                    if type(collision) is dict
                }
                collision_rows = [
                    value
                    for value in manifest.get("manifest_rows", [])
                    if value.get("representation") == "episode_collision_raw"
                ]
                _require(
                    manifest_sha == identity_sha256(manifest_unsigned),
                    f"episode manifest changed at {ordinal}",
                )
                _require(
                    _collision_partition_valid(manifest)
                    and len(collision_by_address) == len(collisions),
                    f"episode collision partition changed at {ordinal}",
                )
                for collision_row in collision_rows:
                    collision = collision_by_address.get(
                        str(collision_row.get("collision_address"))
                    )
                    episode_row = raw_by_id.get(str(collision_row.get("chunk_id")))
                    citation_entry = citation_by_label.get(
                        str(collision and collision.get("global_citation"))
                    )
                    _require(
                        collision is not None
                        and episode_row is not None
                        and citation_entry is not None
                        and collision_row.get("chunk_id")
                        == collision.get("episode_evidence_id")
                        and _collision_binding_valid(
                            collision, episode_row, citation_entry
                        ),
                        f"episode collision binding changed at {ordinal}",
                    )
                sealed_collisions.extend(copy.deepcopy(collisions))
                represented.update(
                    str(raw["chunk_id"]) for raw in manifest["raw_rows"]
                )
                represented.update(
                    str(ref["episode_chunk_id"])
                    for ref in manifest["global_refs"]
                )
            _require(
                arm.get("raw_collision_bindings") == sealed_collisions,
                f"raw collision projection changed at {ordinal}",
            )
            fact_body = arm["fact_ledger"]
            fact_unsigned = dict(fact_body)
            fact_receipt = fact_unsigned.pop("receipt_sha256", None)
            _require(
                fact_receipt == identity_sha256(fact_unsigned)
                and all(
                    str(fact["backing_evidence_id"]) in represented
                    for fact in fact_body["selected_facts"]
                )
                and arm["rendered_fact_ids"]
                == [str(fact["fact_id"]) for fact in fact_body["selected_facts"]],
                f"fact slice/raw binding changed at {ordinal}",
            )
    replay_body, _replay_digest = hot._read_json_artifact(  # noqa: SLF001
        output_root / REPLAY_NAME
    )
    replay_unsigned = dict(replay_body)
    replay_receipt = replay_unsigned.pop("replay_receipt_sha256", None)
    _require(
        replay_receipt == identity_sha256(replay_unsigned)
        and replay_body.get("format") == f"{FORMAT}-semantic-replay-v1"
        and replay_body.get("canonical_semantic_identity") is True
        and replay_body.get("provider_calls") == 0
        and replay_body.get("question_count") == EXPECTED_QUESTION_COUNT
        and replay_body.get("selection_sha256") == digest
        and replay_body.get("row_receipts_sha256")
        == identity_sha256([row["row_receipt_sha256"] for row in rows])
        and replay_body.get("source_construction_sha256")
        == selection.get("source_construction_sha256")
        and replay_body.get("source_runtime_sha256")
        == selection.get("source_runtime_sha256")
        and replay_body.get("source_replay_sha256")
        == selection.get("source_replay_sha256"),
        "r3 semantic replay binding changed",
    )
    assert_gold_blind(replay_body, path="loaded_hot_v6_spine_episode_semantic_replay")
    assert_gold_blind(selection, path="loaded_hot_v6_spine_episode_fact_selection")
    return selection, digest


def load_selection(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], str]:
    """Public authenticated loader for the canonical r3 selection."""

    return _load_selection(output_root.resolve())


def materialize(
    *,
    source_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
) -> str:
    _require(not output_root.exists(), "output root must be unique and absent")
    construction, construction_sha, runtime_sha, replay_sha = source_assay._load_source(source_root)  # noqa: SLF001
    selection = _project_selection(
        construction,
        construction_sha256=construction_sha,
        runtime_sha256=runtime_sha,
        replay_sha256=replay_sha,
        retrieval_path=retrieval_path,
        store_root=store_root,
    )
    digest = hot._atomic_write_json(output_root / SELECTION_NAME, selection)  # noqa: SLF001
    print(
        "Spine-episodic fact ledger: "
        f"{selection['question_count']} prompts; "
        f"episode_raw={selection['aggregate']['rendered_episode_raw_chunk_count']}; "
        f"facts={selection['aggregate']['rendered_fact_count']}; selection={digest}",
        flush=True,
    )
    return digest


def replay(
    *,
    source_root: Path,
    retrieval_path: Path,
    store_root: Path,
    output_root: Path,
) -> str:
    """Rebuild the gold-free selection and require canonical semantic identity."""

    sealed, sealed_sha = hot._read_json_artifact(output_root / SELECTION_NAME)  # noqa: SLF001
    _require(not (output_root / REPLAY_NAME).exists(), "refusing to overwrite replay")
    construction, construction_sha, runtime_sha, replay_sha = source_assay._load_source(source_root)  # noqa: SLF001
    rebuilt = _project_selection(
        construction,
        construction_sha256=construction_sha,
        runtime_sha256=runtime_sha,
        replay_sha256=replay_sha,
        retrieval_path=retrieval_path,
        store_root=store_root,
    )
    _require(rebuilt == sealed, "spine-episodic selection replay changed")
    body = {
        "canonical_semantic_identity": True,
        "format": f"{FORMAT}-semantic-replay-v1",
        "provider_calls": 0,
        "question_count": len(rebuilt["questions"]),
        "row_receipts_sha256": identity_sha256(
            [row["row_receipt_sha256"] for row in rebuilt["questions"]]
        ),
        "selection_sha256": sealed_sha,
        "source_construction_sha256": construction_sha,
        "source_replay_sha256": replay_sha,
        "source_runtime_sha256": runtime_sha,
    }
    artifact = {**body, "replay_receipt_sha256": identity_sha256(body)}
    assert_gold_blind(artifact, path="hot_v6_spine_episode_replay")
    digest = hot._atomic_write_json(output_root / REPLAY_NAME, artifact)  # noqa: SLF001
    print(f"Spine-episodic semantic replay: 100/100 exact; replay={digest}", flush=True)
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("materialize")
    commands.add_parser("replay")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        materialize(
            source_root=args.source_root.resolve(),
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
            output_root=args.output_root.resolve(),
        )
    elif args.command == "replay":
        replay(
            source_root=args.source_root.resolve(),
            retrieval_path=args.retrieval.resolve(),
            store_root=args.store_root.resolve(),
            output_root=args.output_root.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "_load_selection",
    "load_selection",
    "materialize",
    "replay",
    "select_spine_episodes",
]
