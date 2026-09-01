"""Small provider-free action normalization shared by ranking and cueing."""

from __future__ import annotations

import re

from .typed_operator_spec import normalized_terms


_ACTION_VARIANTS: dict[str, frozenset[str]] = {
    "acquire": frozenset(
        {
            "acquire",
            "acquired",
            "acquisition",
            "buy",
            "bought",
            "get",
            "got",
            "purchase",
            "purchased",
        }
    ),
    "assemble": frozenset({"assemble", "assembled", "assembly"}),
    "assist": frozenset({"assist", "assisted", "help", "helped"}),
    "cancel": frozenset({"cancel", "canceled", "cancelled"}),
    "clean": frozenset({"clean", "cleaned", "cleaning"}),
    "collect": frozenset({"collect", "collected", "collection"}),
    "donate": frozenset({"donate", "donated", "donation"}),
    "complete": frozenset({"complete", "completed", "finish", "finished"}),
    "cook": frozenset({"cook", "cooked", "cooking"}),
    "fix": frozenset({"fix", "fixed", "repair", "repaired"}),
    "join": frozenset({"join", "joined", "joining"}),
    "learn": frozenset({"learn", "learned", "learnt"}),
    "pickup": frozenset({"pick", "picked", "pickup"}),
    "receive": frozenset({"receive", "received", "receipt"}),
    "return": frozenset({"return", "returned", "returning"}),
    "sell": frozenset({"sell", "sold", "selling"}),
    "service": frozenset({"service", "serviced", "servicing"}),
    "spend": frozenset({"spend", "spent", "purchase", "purchased", "paid"}),
    "try": frozenset({"try", "tried", "trying"}),
    "visit": frozenset({"visit", "visited", "visiting"}),
}
_COMPLETED_VARIANTS = frozenset(
    {
        "acquired",
        "assembled",
        "assisted",
        "bought",
        "canceled",
        "cancelled",
        "cleaned",
        "collected",
        "completed",
        "cooked",
        "donated",
        "finished",
        "fixed",
        "helped",
        "joined",
        "learned",
        "learnt",
        "maintained",
        "paid",
        "picked",
        "purchased",
        "repaired",
        "replaced",
        "received",
        "returned",
        "serviced",
        "sold",
        "spent",
        "tried",
        "visited",
    }
)
_ACTION_SURFACE_WORD_RE = re.compile(
    r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE
)
_COMPLETED_ACQUIRE_GOT_RE = re.compile(
    r"\b(?:i|we)\s+(?:have\s+)?(?:just\s+|recently\s+)?got\s+"
    r"(?:a|an|the|my|our|some|new|another|\d+)\b",
    re.IGNORECASE,
)
_COMPLETED_ACQUIRE_BROUGHT_HOME_RE = re.compile(
    r"\b(?:i|we)\s+(?:have\s+|had\s+)?(?:just\s+|recently\s+)?brought\s+"
    r"(?:(?:it|them|this|that|one)|"
    r"(?:a|an|the|my|our|some|new|another)\s+"
    r"(?:[^\W_]+(?:['’-][^\W_]+)?\s+){0,3}"
    r"[^\W_]+(?:['’-][^\W_]+)?)\s+(?:back\s+)?home\b",
    re.IGNORECASE,
)
_COMPLETED_VISIT_TOOK_PERSON_RE = re.compile(
    r"\b(?:i|we)\s+(?:have\s+)?(?:just\s+|recently\s+)?took\s+"
    r"(?:him|her|them|"
    r"(?:my|our|a|an|the)\s+"
    r"(?:niece|nephew|daughter|son|kid|child|friend|partner|"
    r"mother|father|mom|dad|sister|brother|wife|husband|"
    r"family|coworker|colleague))\s+to\s+(?:the\s+)?"
    r"(?:[^\W_]+(?:['’-][^\W_]+)?\s+){0,5}"
    r"(?:museum|gallery|zoo|aquarium|park|library|theater|theatre|"
    r"stadium|arena|exhibit|restaurant|cafe|concert|festival)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_PLANNING_CUE_RE = re.compile(
    r"(?:\b(?:i|we)(?:['’](?:m|re)|\s+(?:am|are|was|were))?\s+"
    r"(?:plan(?:ned|ning)?(?:\s+to)?|intend(?:ed|ing)?(?:\s+to)?|"
    r"consider(?:ed|ing)?|look(?:ed|ing)?\s+into|"
    r"think(?:ing)?\s+(?:about|of)|hope(?:d|ing)?\s+to|"
    r"(?:need|want|expect)\s+to|going\s+to)\b|"
    r"\b(?:i|we)\s+think\s+it\s+(?:is|was)\s+time\s+to\b)",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(r"[^.!?;\r\n]+")
_RAW_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_LINKED_SERVICE_SURFACES = frozenset(
    {
        "maintain",
        "maintained",
        "maintenance",
        "replace",
        "replaced",
        "replacement",
    }
)
# Keep completed tense/aspect surfaces exact before the canonical action
# vocabulary is normalized.  Stemming ``visited`` to ``visit`` must not turn
# a proposal such as "might visit" into a completed event.
_COMPLETED_ACTION_SURFACES = {
    concept: frozenset(
        variant.casefold()
        for variant in variants
        if variant in _COMPLETED_VARIANTS
    )
    for concept, variants in _ACTION_VARIANTS.items()
}
# Use the exact same lexical normalization as typed slot compilation/evidence
# parsing (for example, ``purchased`` becomes ``purchas``).
_ACTION_VARIANTS = {
    concept: frozenset(
        term
        for variant in variants
        for term in normalized_terms(variant)
    )
    for concept, variants in _ACTION_VARIANTS.items()
}
def canonical_action_concepts(text: str) -> tuple[str, ...]:
    if type(text) is not str:
        raise TypeError("action semantic text must be exact")
    terms = set(normalized_terms(text))
    concepts = {
        concept
        for concept, variants in _ACTION_VARIANTS.items()
        if terms & variants
    }
    if {"pick", "up"} <= terms or {"picked", "up"} <= terms:
        concepts.add("pickup")
    if {"dry", "clean"} <= terms or {"dry", "cleaning"} <= terms:
        concepts.add("clean")
    return tuple(sorted(concepts))


def linked_action_concepts(text: str) -> tuple[str, ...]:
    """Add conservative phrase links without changing the sealed base index.

    The R7 question-neutral index is built with ``canonical_action_concepts``.
    These extra local-to-global bridges are intentionally applied only after
    V7 hydrates an exact quote, preserving the authenticated R7 lifecycle.
    """

    if type(text) is not str:
        raise TypeError("linked action semantic text must be exact")
    concepts = set(canonical_action_concepts(text))
    surface_words = {
        word.casefold() for word in _RAW_WORD_RE.findall(text)
    }
    if surface_words & _LINKED_SERVICE_SURFACES:
        concepts.add("service")
    if _COMPLETED_ACQUIRE_BROUGHT_HOME_RE.search(text):
        concepts.add("acquire")
    if _COMPLETED_VISIT_TOOK_PERSON_RE.search(text):
        concepts.add("visit")
    return tuple(sorted(concepts))


def canonical_action_proof_terms(
    text: str, action_concept: str, /
) -> tuple[str, ...]:
    """Return sealed single-semantic terms proving one action concept.

    Existing one-term action surfaces stay byte-identical (for example,
    ``purchased`` remains ``purchased``).  A hyphenated or possessive surface
    can normalize to multiple semantic terms even though it carries a valid
    action edge.  In that case, retain only normalized component terms that
    independently prove the requested concept.  This keeps proof terms inside
    the active-match single-semantic-term contract without weakening it.
    """

    if type(text) is not str or type(action_concept) is not str:
        raise TypeError("action proof text and concept must be exact")
    if canonical_action_concepts(action_concept) != (action_concept,):
        raise ValueError("action proof concept must be canonical")

    proof_terms: list[str] = []
    for surface in _ACTION_SURFACE_WORD_RE.findall(text.casefold()):
        if action_concept not in canonical_action_concepts(surface):
            continue
        normalized = normalized_terms(surface)
        candidates = (
            (surface,)
            if len(normalized) == 1
            else tuple(
                term
                for term in normalized
                if action_concept in canonical_action_concepts(term)
            )
        )
        for term in candidates:
            if term not in proof_terms:
                proof_terms.append(term)
    return tuple(proof_terms)


def completed_action_concepts(text: str) -> tuple[str, ...]:
    if type(text) is not str:
        raise TypeError("completed action semantic text must be exact")
    surface_words = {
        word.casefold() for word in _RAW_WORD_RE.findall(text)
    }
    concepts = set(linked_action_concepts(text))
    completed = {
        concept
        for concept in concepts
        if _COMPLETED_ACTION_SURFACES[concept] & surface_words
    }
    # ``got`` is intentionally not a global completed-action token: phrases
    # such as "got to try" or "got sick" are not acquisitions.  The common
    # first-person object construction is, however, a positive completed
    # acquisition witness (for example, "I just got a smoker today").
    if "acquire" in concepts and _COMPLETED_ACQUIRE_GOT_RE.search(text):
        completed.add("acquire")
    if _COMPLETED_ACQUIRE_BROUGHT_HOME_RE.search(text):
        completed.add("acquire")
    if _COMPLETED_VISIT_TOOK_PERSON_RE.search(text):
        completed.add("visit")
    if surface_words & {"maintained", "replaced"}:
        completed.add("service")
    return tuple(sorted(completed))


def planned_action_concepts(text: str) -> tuple[str, ...]:
    """Return actions under an explicit first-person proposal cue.

    Planning is scoped to the containing clause so a proposal elsewhere in a
    multi-sentence segment cannot relabel an already completed action.  This
    function provides positive ranking evidence only; callers must still gate
    it on the question's explicit ``include_proposed`` operator.
    """

    if type(text) is not str:
        raise TypeError("planned action semantic text must be exact")
    planned: set[str] = set()
    for match in _CLAUSE_RE.finditer(text):
        clause = match.group(0)
        if not _FIRST_PERSON_PLANNING_CUE_RE.search(clause):
            continue
        planned.update(linked_action_concepts(clause))
    return tuple(sorted(planned))


def matched_action_concepts(question: str, evidence: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(canonical_action_concepts(question))
            & set(canonical_action_concepts(evidence))
        )
    )


__all__ = [
    "canonical_action_proof_terms",
    "canonical_action_concepts",
    "completed_action_concepts",
    "linked_action_concepts",
    "matched_action_concepts",
    "planned_action_concepts",
]
