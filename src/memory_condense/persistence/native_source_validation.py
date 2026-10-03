"""Reuse authenticated, immutable raw prefixes during native publication.

Cold admission validates every address. A warm append validates changed span
partitions and new turns, with the same complete transcript hash as a rebuild.
No cache is accepted from disk without cold reconstruction.
"""
from dataclasses import dataclass
import hashlib

from memory_condense.application.section_retrieval import HydratedSectionSpan
from memory_condense.domain._discourse_identity import canonical_json, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.native_spine_context_routing import NativeSpineContextRouter


@dataclass(frozen=True)
class SourceValidation:
    turns: tuple
    body_tokens: int
    transcript_sha256: str
    _prefix_hash: object


def validate(semantic, hierarchy, turns, previous=None):
    old = previous.source_validation if previous is not None else None
    prefix = len(old.turns) if old is not None else 0
    if old is not None and (len(turns) < prefix or any(
            a is not b and a != b for a,b in zip(old.turns, turns))):
        raise ValueError('Native publication must extend its authenticated raw prefix')
    if not turns or len({t.turn_id for t in turns}) != len(turns):
        raise ValueError('native snapshot requires unique ingested raw turns')
    if previous is None or (semantic.hierarchy is not previous.native.semantic.hierarchy
                            or hierarchy is not previous.native.hierarchy):
        router = NativeSpineContextRouter(semantic, hierarchy)
        if len(router.owners) != len(semantic.sections):
            raise ValueError('native snapshot requires a complete attention hierarchy')
    old_atoms = previous.native.semantic.hierarchy._by_id if previous is not None else {}
    atoms = semantic.hierarchy._by_id
    affected = {t.turn_id for t in turns[prefix:]}
    for sid, section in old_atoms.items():
        replacement = atoms.get(sid)
        if replacement is None or replacement.spans != section.spans:
            affected.update(s.turn_id for s in section.spans)
    for sid, section in atoms.items():
        former = old_atoms.get(sid)
        if former is None or former.spans != section.spans:
            affected.update(s.turn_id for s in section.spans)
    raw = {t.turn_id: t for t in turns if t.turn_id in affected}
    by_turn = {tid: [] for tid in affected}
    for section in semantic.sections:
        span, = section.spans
        if span.turn_id in affected:
            by_turn[span.turn_id].append(span)
    for tid, spans in by_turn.items():
        turn = raw.get(tid)
        if turn is None:
            raise ValueError('native snapshot differs from ingested raw provenance')
        turn_sha = quote_sha256(turn.text)
        cursor = 0
        for span in sorted(spans, key=lambda s: s.start_char):
            if (span.source_id != turn.source_id or span.role != turn.role
                    or span.created_at != turn.created_at.isoformat()
                    or span.turn_text_sha256 != turn_sha or span.end_char > len(turn.text)):
                raise ValueError('native snapshot differs from ingested raw provenance')
            if span.start_char != cursor:
                raise ValueError('native snapshot raw coverage has a gap or overlap')
            HydratedSectionSpan(span, turn.text[span.start_char:span.end_char])
            cursor = span.end_char
        if not spans or cursor != len(turn.text):
            raise ValueError('native snapshot raw coverage is incomplete')
    hasher = old._prefix_hash.copy() if old is not None else hashlib.sha256(b'[')
    body_tokens = old.body_tokens if old is not None else 0
    for i, turn in enumerate(turns[prefix:], start=prefix):
        if i:
            hasher.update(b',')
        hasher.update(canonical_json(dict(turn_id=turn.turn_id, source_id=turn.source_id,
            role=turn.role, created_at=turn.created_at.isoformat(),
            text_sha256=quote_sha256(turn.text))).encode('utf-8'))
        body_tokens += count_tokens(turn.text)
    complete = hasher.copy()
    complete.update(b']')
    return SourceValidation(turns, body_tokens, complete.hexdigest(), hasher)
