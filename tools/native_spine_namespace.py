"""Cold-load one exact native history; incomplete snapshots require explicit use."""
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.native_hierarchy_occurrence import bind_native_hierarchy
from memory_condense.search.native_spine_memory import materialize_history
from memory_condense.search.section_routing import SectionSummaryIndex
from tools.assemble_native_spine_admitted import AdmittedSummaryBodies
from tools.assemble_native_spine_summaries import digest
from tools.compile_native_spine_hierarchy import implementation as hierarchy_implementation
from tools.matched_eval.artifacts import read_sealed_json


@dataclass(frozen=True)
class NativeNamespace:
    history: object
    atomic_index: SectionSummaryIndex
    hierarchy: SectionSummaryIndex
    audit: dict

    def require_complete(self, *, minimum_body_tokens=1_000_000):
        if (not self.audit["complete_namespace"]
                or self.audit["body_tokens"] < minimum_body_tokens):
            raise ValueError("native evaluation requires a complete history at the requested token scale")
        return self


def materialize_namespace(sessions, *, body_ids, templates, load_body, load_summaries,
                          compiler_identity, allow_partial=False):
    """Keep all admitted atoms even when their parent compilation is pending."""
    sessions = tuple(sessions)
    if not sessions:
        raise ValueError("native namespace source population is empty")
    missing_bodies = tuple(s["occurrence_id"] for s in sessions if s["body_sha256"] not in body_ids)
    missing_trees = tuple(s["occurrence_id"] for s in sessions if s["body_sha256"] not in templates)
    if (missing_bodies or missing_trees) and not allow_partial:
        raise ValueError("native namespace has missing summaries or hierarchies")
    selected = tuple(s for s in sessions if s["body_sha256"] in body_ids)
    if not selected:
        raise ValueError("native namespace has no complete admitted bodies")
    history = materialize_history(selected, load_body=load_body, load_summaries=load_summaries,
                                  compiler_identity=compiler_identity)
    by_source = defaultdict(list)
    for atom in history.atoms:
        by_source[atom.source_id].append(atom)
    trees, bindings = [], []
    for source in selected:
        template = templates.get(source["body_sha256"])
        if template is None:
            continue
        tree, _, binding = bind_native_hierarchy(template, source,
            by_source["native-source-" + source["occurrence_id"]])
        trees.extend(tree.sections)
        bindings.append(binding.identity_payload())
    atomic = SectionSummaryIndex(history.atoms)
    hierarchy = SectionSummaryIndex(trees)
    return NativeNamespace(history, atomic, hierarchy, {
        "source_occurrences": len(sessions), "admitted_occurrences": len(selected),
        "hierarchy_occurrences": len(bindings), "atomic_sections": len(atomic.sections),
        "body_tokens": sum(count_tokens(t.text) for t in history.turns.values()),
        "missing_summary_occurrence_ids": list(missing_bodies),
        "missing_hierarchy_occurrence_ids": list(missing_trees),
        "complete_namespace": not missing_bodies and not missing_trees,
        "partial_use_explicit": allow_partial, "occurrence_bindings": bindings,
        "atomic_index_sha256": atomic.receipt_sha256, "hierarchy_sha256": hierarchy.receipt_sha256,
        "new_model_calls": 0, "full100_target_passed": False,
    })


class NativeSpineCorpus:
    """Authenticate immutable inputs once, then materialize separate namespaces."""

    def __init__(self, source_root, store_root, hierarchy_report):
        self.store = self.bank = None
        self.source_root = Path(source_root).resolve()
        hierarchy_report = Path(hierarchy_report)
        root = hierarchy_report.parent
        self.sources = read_sealed_json(self.source_root / "sources.json")
        self.report = read_sealed_json(hierarchy_report)
        preflight = read_sealed_json(root / "preflight.json")
        exchange_root = Path(preflight.payload["exchange_root"])
        inputs = read_sealed_json(exchange_root / "inputs.json")
        exchange = read_sealed_json(exchange_root / "result.json")
        exchange_preflight = read_sealed_json(exchange_root / "preflight.json")
        if (self.report.payload["preflight_sha256"] != preflight.sha256
                or preflight.payload["implementation"] != hierarchy_implementation()
                or preflight.payload["exchange_result_sha256"] != exchange.sha256
                or exchange.payload["preflight_sha256"] != exchange_preflight.sha256
                or exchange_preflight.payload["inputs_sha256"] != inputs.sha256
                or self.sources.sha256 != inputs.payload["sources_sha256"]
                or self.sources.payload["question_inputs"] is not False
                or self.sources.payload["gold_inputs"] is not False):
            raise ValueError("native corpus input binding changed")
        self.templates = {}
        for binding in self.report.payload["compiled_bodies"]:
            path = (root / binding["path"]).resolve()
            path.relative_to((root / "hierarchies").resolve())
            template = read_sealed_json(path)
            if (template.sha256 != binding["sha256"] or template.payload["preflight_sha256"] != preflight.sha256
                    or template.payload["body_sha256"] in self.templates):
                raise ValueError("native hierarchy template binding changed")
            self.templates[template.payload["body_sha256"]] = template.payload
        if len(self.templates) != self.report.payload["body_count"]:
            raise ValueError("native hierarchy template count changed")
        bank_path = (self.source_root / self.sources.payload["body_bank_path"]).resolve()
        bank_path.relative_to(self.source_root)
        if digest(bank_path) != self.sources.payload["body_bank_sha256"]:
            raise ValueError("native source body bank changed")
        try:
            self.store = AdmittedSummaryBodies(store_root)
            if (self.store.manifest.sha256 != inputs.payload["summary_body_store_sha256"]
                    or self.store.manifest.payload["sources_sha256"] != self.sources.sha256):
                raise ValueError("native summaries belong to a different source bank")
            self.body_ids = frozenset(row[0] for row in self.store.connection.execute("SELECT body_sha256 FROM bodies"))
            if len(self.body_ids) != self.store.manifest.payload["body_count"] or not self.templates.keys() <= self.body_ids:
                raise ValueError("native available body population changed")
            self.bank = sqlite3.connect(bank_path.as_uri() + "?mode=ro", uri=True)
            self.namespaces = {n["namespace_id"]: n for n in self.sources.payload["namespaces"]}
            if len(self.namespaces) != len(self.sources.payload["namespaces"]):
                raise ValueError("native namespace identity is duplicated")
        except Exception:
            self.close()
            raise

    def close(self):
        if self.bank is not None:
            self.bank.close()
        if self.store is not None:
            self.store.close()

    def load_namespace(self, namespace_id, *, allow_partial=False):
        binding = self.namespaces[namespace_id]
        path = (self.source_root / binding["path"]).resolve()
        path.relative_to((self.source_root / "namespaces").resolve())
        source = read_sealed_json(path)
        if source.sha256 != binding["sha256"] or source.payload["namespace_id"] != namespace_id:
            raise ValueError("native namespace source changed")

        def load_body(sha):
            row = self.bank.execute("SELECT body_json FROM bodies WHERE body_sha256=?", (sha,)).fetchone()
            if row is None:
                raise ValueError("native raw body is missing")
            return json.loads(row[0])

        namespace = materialize_namespace(source.payload["sessions"], body_ids=self.body_ids,
            templates=self.templates, load_body=load_body, load_summaries=self.store.load,
            compiler_identity=self.store.manifest.sha256, allow_partial=allow_partial)
        namespace.audit.update(namespace_id=namespace_id, sources_sha256=self.sources.sha256,
            namespace_source_sha256=source.sha256, summary_store_sha256=self.store.manifest.sha256,
            hierarchy_report_sha256=self.report.sha256)
        return namespace
