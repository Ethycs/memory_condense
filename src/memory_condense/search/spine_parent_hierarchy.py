"""Restore attention-selected parent sections from a complete leaf projection.

The saved cuts determine the topology. Only the two explicit summary channels
enter merge requests; raw span descriptors validate ownership and coverage.
Original leaf descriptors are preserved byte for byte.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.episodes.user_spine_hierarchy import _fold, _render_channels
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.spine_summary import SpineSummaryFragment


def summary_channels(section):
    body = json.loads(section.summary)
    if type(body) is not dict or set(body) != {
        "user_spine", "attached_context_not_user_assertions", "transcript_date_range"
    }:
        raise ValueError("parent compilation requires explicit user-spine summary channels")
    values = body["user_spine"], body["attached_context_not_user_assertions"]
    if any(value is not None and (type(value) is not str or not value.strip()
                                or count_tokens(value) > 128) for value in values):
        raise ValueError("parent input summary exceeds the channel contract")
    if section.summary != _render_channels(*values, section.spans):
        raise ValueError("summary dates or channel encoding changed")
    if (values[0] is not None) != any(span.role == "user" for span in section.spans):
        raise ValueError("user-spine summary attribution changed")
    return values


@dataclass(frozen=True, slots=True)
class SpineParentSpec:
    section_id: str
    source_id: str
    spans: tuple[RawSectionSpan, ...]
    child_section_ids: tuple[str, str]
    split_exchange: int
    attention_change: float


class SourceSpineParentPlan:
    """Validate one entire source before scheduling any summary-only work."""

    def __init__(self, leaves, ordered_spans, cuts):
        self.leaves = tuple(leaves)
        spans = tuple(ordered_spans)
        if not self.leaves or any(type(s) is not SectionSummary or s.child_section_ids for s in self.leaves):
            raise ValueError("parent restoration requires original nonempty leaf sections")
        sources = {s.source_id for s in self.leaves}
        if len(sources) != 1 or not spans or any(type(s) is not RawSectionSpan for s in spans):
            raise ValueError("one parent plan must contain one source and its exact ordered spans")
        self.source_id = next(iter(sources))
        # Enforces source identity, contiguous fragments and no revisited turns.
        SectionSummary("validate", self.source_id, "validate", spans, "parent-plan-validation")
        expected = [s.receipt_sha256 for s in spans]
        if len(set(expected)) != len(expected):
            raise ValueError("duplicate source spans")
        positions = {sha: i for i, sha in enumerate(expected)}
        if sorted(s.receipt_sha256 for leaf in self.leaves for s in leaf.spans) != sorted(expected):
            raise ValueError("leaves must partition every original source span exactly once")
        self.leaves = tuple(sorted(self.leaves, key=lambda leaf: positions[leaf.spans[0].receipt_sha256]))
        if tuple(s for leaf in self.leaves for s in leaf.spans) != spans:
            raise ValueError("leaf spans changed transcript order")
        SectionSummaryIndex(self.leaves)
        self.channels = {s.section_id: summary_channels(s) for s in self.leaves}
        # Reconstruct exchange boundaries from roles and turn identity, never
        # timestamps, lexical source IDs, summaries, queries or raw text.
        starts = [0]
        for i, span in enumerate(spans[1:], 1):
            if span.role == "user" and span.turn_id != spans[i - 1].turn_id:
                starts.append(i)
        starts.append(len(spans))
        cut_map = {}
        for cut in cuts:
            if type(cut) is not dict or set(cut) != {"section_id", "split_atom", "attention_change"}:
                raise ValueError("invalid saved attention cut")
            if cut["section_id"] in cut_map or type(cut["split_atom"]) is not int:
                raise ValueError("duplicate or invalid saved attention cut")
            if type(cut["attention_change"]) not in (int, float) or not math.isfinite(cut["attention_change"]):
                raise ValueError("nonfinite saved attention cut")
            cut_map[cut["section_id"]] = cut
        by_id = {s.section_id: s for s in self.leaves}
        specs, used_leaves, used_cuts = [], set(), set()

        def descend(left, right):
            covered = spans[starts[left]:starts[right]]
            sid = "spine-section-" + identity_sha256([s.receipt_sha256 for s in covered])
            if sid in by_id:
                if by_id[sid].spans != covered or sid in cut_map:
                    raise ValueError("saved topology changed a leaf")
                used_leaves.add(sid)
                return sid
            cut = cut_map.get(sid)
            if cut is None or not left < cut["split_atom"] < right:
                raise ValueError("missing or out-of-range saved attention cut")
            used_cuts.add(sid)
            middle = cut["split_atom"]
            children = descend(left, middle), descend(middle, right)
            specs.append(SpineParentSpec(sid, self.source_id, covered, children,
                                         middle, float(cut["attention_change"])))
            return sid

        self.root_section_id = descend(0, len(starts) - 1)
        if used_leaves != set(by_id) or used_cuts != set(cut_map):
            raise ValueError("saved topology contains unused leaves or attention cuts")
        self.parents = tuple(specs)
        self.receipt_sha256 = identity_sha256({
            "leaf_receipts": [s.receipt_sha256 for s in self.leaves],
            "cuts": list(cuts), "root_section_id": self.root_section_id,
        })

    def compile(self, *, summarize, summarizer_identity):
        sections = {s.section_id: s for s in self.leaves}
        channels = dict(self.channels)
        for spec in self.parents:
            children = [sections[sid] for sid in spec.child_section_ids]
            users = _fold([
                SpineSummaryFragment("user_summary", s.spans[0].created_at, channels[s.section_id][0])
                for s in children if channels[s.section_id][0] is not None
            ], kind="user_spine", user_spine=None, summarize=summarize, cap=128, prompt_cap=2048)
            attached = _fold([
                SpineSummaryFragment("attached_summary", s.spans[0].created_at, channels[s.section_id][1])
                for s in children if channels[s.section_id][1] is not None
            ], kind="attached_context", user_spine=users, summarize=summarize, cap=128, prompt_cap=2048)
            section = SectionSummary(spec.section_id, spec.source_id,
                _render_channels(users, attached, spec.spans), spec.spans, summarizer_identity,
                child_section_ids=spec.child_section_ids)
            sections[section.section_id] = section
            channels[section.section_id] = users, attached
        index = SectionSummaryIndex(tuple(sections.values()))
        if tuple(s for s in index.sections if not s.child_section_ids) != tuple(sorted(self.leaves, key=lambda s: s.section_id)):
            raise ValueError("parent compilation changed original leaves")
        return index
