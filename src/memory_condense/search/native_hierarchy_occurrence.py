"""Bind a cached native body tree to another exact source occurrence, without models."""
from dataclasses import dataclass

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.episodes.user_spine_hierarchy import _render_channels
from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_parent_hierarchy import summary_channels


@dataclass(frozen=True, slots=True)
class NativeHierarchyBinding(SealedIdentity):
    body_sha256: str
    occurrence_id: str
    template_index_sha256: str
    template_atomic_index_sha256: str
    index_sha256: str
    atomic_index_sha256: str
    root_section_ids: tuple[str, ...]
    raw_text_reads: int = 0
    model_calls: int = 0
    receipt_sha256: str = ""

    def __post_init__(self):
        object.__setattr__(self, "root_section_ids", tuple(self.root_section_ids))
        if len(self.root_section_ids) != 1 or self.raw_text_reads != 0 or self.model_calls != 0:
            raise ValueError("native body binding requires one exact source root and no model/raw reads")
        self._seal()


def bind_native_hierarchy(template, source, atoms):
    """Accept only source records and already-materialized atomic descriptors."""
    source_fields = {"original_session_ordinal", "session_id", "created_at", "metadata_text",
                     "body_sha256", "dataset_origin", "occurrence_id"}
    if (type(source) is not dict or set(source) != source_fields
            or source["body_sha256"] != template["body_sha256"]
            or source["occurrence_id"] != identity_sha256({k: v for k, v in source.items() if k != "occurrence_id"})
            or template["original_atomic_addresses_preserved"] is not True):
        raise ValueError("native hierarchy body or actual occurrence changed")
    atoms = tuple(atoms)
    original = SectionSummaryIndex.from_json(template["index_json"])
    original_atoms = SectionSummaryIndex.from_json(template["atomic_index_json"])
    atom_index = SectionSummaryIndex(atoms)
    if any(len(a.spans) != 1 or a.child_section_ids for a in (*atoms, *original_atoms.sections)):
        raise ValueError("native hierarchy binding requires original single-span atoms")
    by_id = {s.section_id: s for s in original.sections}
    roots = tuple(template["root_section_ids"])
    parented = {child for s in original.sections for child in s.child_section_ids}
    if len(roots) != 1 or set(roots) != set(by_id)-parented:
        raise ValueError("native template root population changed")
    old_spans = by_id[roots[0]].spans
    old_atoms = {a.spans[0].receipt_sha256: a for a in original_atoms.sections}
    if (len(old_atoms) != len(old_spans) or len(old_atoms) != len(original_atoms.sections)
            or set(old_atoms) != {s.receipt_sha256 for s in old_spans}
            or identity_sha256([s.receipt_sha256 for s in old_spans]) != template["raw_span_population_sha256"]
            or len(atoms) != len(old_spans)):
        raise ValueError("native template atomic partition changed")
    source_id = "native-source-" + source["occurrence_id"]
    span_map, turn_map, reverse_turn_map = {}, {}, {}
    raw_fields = ("role", "start_char", "end_char", "turn_text_sha256", "span_text_sha256", "token_count")
    for old, atom in zip(old_spans, atoms, strict=True):
        new = atom.spans[0]
        if (atom.summary != old_atoms[old.receipt_sha256].summary
                or (new.source_id, new.created_at) != (source_id, source["created_at"])
                or any(getattr(old, field) != getattr(new, field) for field in raw_fields)
                or turn_map.setdefault(old.turn_id, new.turn_id) != new.turn_id
                or reverse_turn_map.setdefault(new.turn_id, old.turn_id) != old.turn_id):
            raise ValueError("rebinding changed raw content, turn ownership, summary or actual date")
        span_map[old.receipt_sha256] = new
    new_ids = {s.section_id: "spine-section-" + identity_sha256(
        [span_map[p.receipt_sha256].receipt_sha256 for p in s.spans]) for s in original.sections}
    if len(set(new_ids.values())) != len(new_ids):
        raise ValueError("rebinding aliased distinct hierarchy sections")
    sections = []
    for old in original.sections:
        users, attached = summary_channels(old)
        spans = tuple(span_map[p.receipt_sha256] for p in old.spans)
        sections.append(SectionSummary(new_ids[old.section_id], source_id,
            _render_channels(users, attached, spans), spans, old.summarizer_identity,
            child_section_ids=tuple(new_ids[child] for child in old.child_section_ids)))
    index = SectionSummaryIndex(tuple(sections))
    bound_roots = tuple(new_ids[root] for root in roots)
    rebound = {s.section_id: s for s in index.sections}
    if rebound[bound_roots[0]].spans != tuple(a.spans[0] for a in atoms):
        raise ValueError("rebinding changed complete original raw coverage")
    binding = NativeHierarchyBinding(source["body_sha256"], source["occurrence_id"], original.receipt_sha256,
                                    original_atoms.receipt_sha256, index.receipt_sha256,
                                    atom_index.receipt_sha256, bound_roots)
    return index, atom_index, binding
