"""Use an expanded body store with explicitly authenticated older hierarchies."""
from tools.assemble_native_spine_direct_repairs import DirectRepairSummaryBodies
from tools.assemble_native_spine_summaries import digest
from tools.native_spine_namespace import NativeSpineCorpus


def verify_extension(original, expanded):
    """Cached templates remain usable only if every original address is exact."""
    old, new = original.manifest.payload, expanded.manifest.payload
    if any(old[key] != new[key] for key in (
            "sources_sha256", "compiler_preflight_sha256", "model")):
        raise ValueError("expanded summaries belong to a different source compilation")
    old_ids = frozenset(r[0] for r in original.connection.execute("SELECT body_sha256 FROM bodies"))
    new_ids = frozenset(r[0] for r in expanded.connection.execute("SELECT body_sha256 FROM bodies"))
    if (len(old_ids) != old["body_count"] or len(new_ids) != new["body_count"]
            or not old_ids <= new_ids):
        raise ValueError("expanded summary population removed an original body")
    atoms = 0
    for sha in sorted(old_ids):
        previous = original.load(sha)
        if previous != expanded.load(sha):
            raise ValueError("expanded summaries changed an original summary or raw pointer")
        atoms += len(previous)
    if atoms != old["atom_count"]:
        raise ValueError("original summary population changed")
    return new_ids, {
        "template_summary_store_sha256": original.manifest.sha256,
        "active_summary_store_sha256": expanded.manifest.sha256,
        "preserved_body_count": len(old_ids), "preserved_atomic_count": atoms,
        "added_body_count": len(new_ids)-len(old_ids),
        "all_original_summary_and_pointer_records_unchanged": True,
        "expanded_store_producer_format": new["producer_format"],
        "expanded_store_producer_implementation": new["producer_implementation"],
        "adapter_implementation_sha256": digest(__file__),
    }


class ExpandedNativeSpineCorpus(NativeSpineCorpus):
    """Original corpus admission plus a checked extension; no new compilation."""

    def __init__(self, source_root, template_store_root, active_store_root, hierarchy_report):
        super().__init__(source_root, template_store_root, hierarchy_report)
        expanded = None
        try:
            expanded = DirectRepairSummaryBodies(active_store_root)
            body_ids, extension = verify_extension(self.store, expanded)
            self.store.close()
            self.store, self.body_ids, self.extension = expanded, body_ids, extension
        except Exception:
            if expanded is not None:
                expanded.close()
            self.close()
            raise

    def load_namespace(self, namespace_id, *, allow_partial=False):
        namespace = super().load_namespace(namespace_id, allow_partial=allow_partial)
        namespace.audit["summary_store_extension"] = dict(self.extension)
        return namespace
