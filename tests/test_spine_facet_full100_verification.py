import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from memory_condense.search.section_routing import SectionSummaryIndex
from memory_condense.search.spine_summary_facets import SpineFacetAddressIndex, summary_facets
from tests.test_spine_summary_facets import section
from tools.compile_spine_facet_addresses import IMPLEMENTATION
from tools.matched_eval.artifacts import publish_sealed_json
from tools.report_joint_source_spine_facets_full100 import full100_gate, load_facet_policy
from tools import report_joint_source_spine_facets_full100_v3 as facet_v3
from tools import report_joint_spine_reader_v3_full100 as reader_v3
from tools import report_joint_source_spine_facets_full100_v4 as facet_v4
from tools import report_joint_spine_reader_v3_full100_v2 as reader_v3_v2
from tools import report_joint_spine_term_coverage_full100 as term_coverage
from tools import report_joint_spine_source_coverage_full100 as source_coverage
from tools import report_joint_spine_source_coverage_full100_v2 as source_coverage_v2
from tools import report_joint_spine_combined_reader_full100 as combined_reader
from tools import report_joint_spine_semantic_seeds_full100 as semantic_seeds
from tools import report_joint_spine_semantic_seeds_full100_v2 as semantic_seeds_v2


MEMORY_ARMS = ("source_spine_overflow", "source_spine_facets")


@pytest.fixture(autouse=True, params=(
    (full100_gate, load_facet_policy, MEMORY_ARMS),
    (facet_v3.full100_gate, facet_v3.load_facet_policy, facet_v3.evaluation.MEMORY_ARMS),
    (reader_v3.full100_gate, reader_v3.load_facet_policy, reader_v3.evaluation.MEMORY_ARMS),
    (facet_v4.full100_gate, facet_v4.load_facet_policy, facet_v4.evaluation.MEMORY_ARMS),
    (reader_v3_v2.full100_gate, reader_v3_v2.load_facet_policy, reader_v3_v2.evaluation.MEMORY_ARMS),
    (term_coverage.full100_gate, term_coverage.load_facet_policy, term_coverage.evaluation.MEMORY_ARMS),
    (source_coverage.full100_gate, source_coverage.load_facet_policy, source_coverage.evaluation.MEMORY_ARMS),
    (source_coverage_v2.full100_gate, source_coverage_v2.load_facet_policy, source_coverage_v2.evaluation.MEMORY_ARMS),
    (combined_reader.full100_gate, combined_reader.load_facet_policy, combined_reader.evaluation.MEMORY_ARMS),
    (semantic_seeds.full100_gate, semantic_seeds.load_facet_policy, semantic_seeds.evaluation.MEMORY_ARMS),
    (semantic_seeds_v2.full100_gate, semantic_seeds_v2.load_facet_policy, semantic_seeds_v2.evaluation.MEMORY_ARMS),
))
def report_version(request, monkeypatch):
    for name, value in zip(("full100_gate", "load_facet_policy", "MEMORY_ARMS"), request.param, strict=True):
        monkeypatch.setattr(sys.modules[__name__], name, value)


def compiled(root, name, *, omitted=False, raw_inputs=False):
    root.mkdir()
    _, leaf = section("User purchased a pass. User asks about a trip.", name)
    hierarchy = SectionSummaryIndex((leaf,))
    index = SimpleNamespace(sections=hierarchy.sections, hierarchy=hierarchy, embedding_identity="fixture")
    base = SimpleNamespace(sha256=hashlib.sha256(name.encode()).hexdigest())
    facets = [f.identity_payload() for f in summary_facets(leaf)]
    matrix = np.array([[1, 0]] * len(facets), dtype=np.float32)
    np.save(root / "facet-vectors.npy", matrix, allow_pickle=False)
    addresses = SpineFacetAddressIndex(hierarchy, matrix, embedding_identity="fixture")
    preflight, _ = publish_sealed_json(root / "preflight.json", {
        "base_index_sha256": base.sha256, "facets": facets[:-1] if omitted else facets,
        "document_inputs": "exact passages within stored user-spine summaries only",
        "raw_inputs": raw_inputs, "gold_inputs": False, "query_inputs_during_compilation": False,
        "implementation": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in IMPLEMENTATION}})
    manifest, _ = publish_sealed_json(root / "addresses.json", {
        "base_index_sha256": base.sha256, "preflight_sha256": preflight.sha256,
        "matrix_sha256": hashlib.sha256((root / "facet-vectors.npy").read_bytes()).hexdigest(),
        "address_index_sha256": addresses.receipt_sha256, "embedding_identity": "fixture",
        "section_count": 1, "facet_count": len(facets)})
    return manifest, base, index


def test_full_population_compiler_policy_is_common_but_verification_binds_each_memory(tmp_path):
    results = []
    for name in ("alpha", "beta"):
        root = tmp_path / name
        manifest, base, index = compiled(root, name)
        results.append(load_facet_policy(root, manifest.sha256, base, index))
    assert results[0][0] == results[1][0]
    assert results[0][1] != results[1][1]


@pytest.mark.parametrize("variation", ["omitted", "raw_inputs", "matrix", "binding"])
def test_resealed_omission_or_changed_input_cannot_certify_the_full_memory(tmp_path, variation):
    root = tmp_path / variation
    manifest, base, index = compiled(root, "alpha", omitted=variation == "omitted", raw_inputs=variation == "raw_inputs")
    if variation == "matrix":
        np.save(root / "facet-vectors.npy", np.array([[0, 1], [0, 1]], dtype=np.float32), allow_pickle=False)
    if variation == "binding":
        base = SimpleNamespace(sha256="foreign")
    with pytest.raises(ValueError):
        load_facet_policy(root, manifest.sha256, base, index)


def population():
    questions = [{"ordinal": i, "question_id": str(i)} for i in range(100)]
    rows = [{"ordinal": i, "question_id": str(i), "arm": arm, "correct": i < 95, "raw_tokens": 1_041_276,
        "memory": {"ttft_s": 5.1, "total_s": 5.2}, "matched_api": {"ttft_s": 5, "total_s": 5.1},
        "short_api": {"ttft_s": 5, "total_s": 5.1}}
        for arm in MEMORY_ARMS for i in range(100)]
    return questions, rows


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, 9])
def test_invalid_or_reversed_latency_cannot_pass(bad):
    questions, rows = population()
    rows[0]["memory"]["ttft_s"] = bad
    with pytest.raises(ValueError, match="chronological"):
        full100_gate(rows, questions)


def test_population_duplicates_and_relaxed_latency_allowance_are_rejected():
    questions, rows = population()
    with pytest.raises(ValueError, match="complete locked"):
        full100_gate(rows, questions + [questions[0]])
    changed = copy.deepcopy(questions)
    changed[-1]["question_id"] = changed[0]["question_id"]
    with pytest.raises(ValueError, match="complete locked"):
        full100_gate(rows, changed)
    with pytest.raises(ValueError, match="cannot be relaxed"):
        full100_gate(rows, questions, latency_ratio_limit=2)
