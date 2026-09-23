"""Native namespace admission for the explicitly recovered body-store producer."""
from tools.assemble_native_spine_recovered import RecoveredSummaryBodies
from tools.assemble_native_spine_summaries import digest
from tools.expanded_native_spine_namespace import verify_extension
from tools.native_spine_namespace import NativeSpineCorpus


class RecoveredNativeSpineCorpus(NativeSpineCorpus):
    def __init__(self, source_root, template_store_root, active_store_root, hierarchy_report):
        super().__init__(source_root, template_store_root, hierarchy_report)
        recovered = None
        try:
            recovered = RecoveredSummaryBodies(active_store_root)
            body_ids, extension = verify_extension(self.store, recovered)
            extension["recovered_corpus_adapter_sha256"] = digest(__file__)
            self.store.close()
            self.store, self.body_ids, self.extension = recovered, body_ids, extension
        except Exception:
            if recovered is not None:
                recovered.close()
            self.close()
            raise

    def load_namespace(self, namespace_id, *, allow_partial=False):
        namespace = super().load_namespace(namespace_id, allow_partial=allow_partial)
        namespace.audit["summary_store_extension"] = dict(self.extension)
        return namespace
