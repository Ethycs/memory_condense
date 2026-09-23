"""Admit expanded parent trees after checking their additional producer lineage."""
from contextlib import closing
from pathlib import Path

from tools.assemble_native_spine_recovered import RecoveredSummaryBodies
from tools.assemble_native_spine_summaries import digest
from tools.compile_recovered_native_spine_hierarchy import FORMAT, implementation
from tools.matched_eval.artifacts import read_sealed_json
from tools.parent_budget_native_spine_namespace import ParentBudgetNativeSpineCorpus
from tools.prepare_recovered_native_spine_attention import validate_admission


class RecoveredParentNativeSpineCorpus(ParentBudgetNativeSpineCorpus):
    def __init__(self, source_root, template_store_root, hierarchy_report, *, active_store_root=None):
        report = read_sealed_json(hierarchy_report)
        preflight = read_sealed_json(Path(hierarchy_report).parent/"preflight.json")
        p = preflight.payload
        if (p.get("producer_format") != FORMAT or p["producer_implementation"] != implementation()
                or report.payload.get("producer_format") != FORMAT
                or report.payload["preflight_sha256"] != preflight.sha256):
            raise ValueError("expanded native parent producer changed")
        admission = validate_admission(Path(p["attention_root"]))
        if (admission.sha256 != p["attention_admission_sha256"]
                or Path(admission.payload["exchange_root"]).resolve() != Path(p["exchange_root"]).resolve()):
            raise ValueError("expanded hierarchy changed its exchange/attention admission")
        with closing(RecoveredSummaryBodies(template_store_root)) as store:
            inputs = read_sealed_json(Path(p["exchange_root"])/"inputs.json")
            if store.manifest.sha256 != inputs.payload["summary_body_store_sha256"]:
                raise ValueError("expanded hierarchy template store changed")
        self.expanded_producer = {"format": FORMAT, "preflight_sha256": preflight.sha256,
            "attention_admission_sha256": admission.sha256,
            "producer_implementation": p["producer_implementation"], "adapter_sha256": digest(__file__)}
        super().__init__(source_root, template_store_root, hierarchy_report, active_store_root=active_store_root)

    def load_namespace(self, namespace_id, *, allow_partial=False):
        namespace = super().load_namespace(namespace_id, allow_partial=allow_partial)
        namespace.audit["expanded_hierarchy_producer"] = dict(self.expanded_producer)
        return namespace
