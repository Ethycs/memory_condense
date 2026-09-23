import json

import pytest

from memory_condense.search.spine_merge_batch import (
    PendingMerge, SummaryMergeCache, lossless_merge, merge_batch_messages, pack_merge_batches, parse_merge_batch,
)
from memory_condense.search.spine_summary import SpineSummaryFragment, SpineSummaryRequest
from tools.build_spine_corpus_hierarchy import compile_waves, restore_request


def request(topic="orchard", kind="user_spine"):
    role = "user_summary" if kind == "user_spine" else "attached_summary"
    return SpineSummaryRequest(kind, tuple(SpineSummaryFragment(role, "2026-09-09", (topic + " ") * 80)
                                          for _ in range(2)),
                               None if kind == "user_spine" else "User asks about orchards.", 128)


def response(summaries):
    return json.dumps({"summaries": [{"label": f"S{i}", "summary": text} for i, text in enumerate(summaries)]})


def test_batch_keeps_jobs_and_role_channels_separate_and_roundtrips():
    jobs = (request(), request("machine suggestion", "attached_context"))
    messages = merge_batch_messages(jobs)
    assert "Return ONLY JSON with one key, summaries" in messages[0]["content"]
    rows = json.loads(messages[1]["content"])["jobs"]
    assert rows[0]["user_spine"] is None
    assert all(f["role"] == "user_summary" for f in rows[0]["fragments"])
    assert rows[1]["user_spine"] == "User asks about orchards."
    from dataclasses import asdict
    assert restore_request(asdict(jobs[0])) == jobs[0]
    assert parse_merge_batch(response(["User planted orchards.", "Assistant suggested irrigation."]), jobs) == (
        "User planted orchards.", "Assistant suggested irrigation.")


def test_no_partial_admission_on_wrong_labels_or_oversized_output():
    jobs = (request(), request("observatory"))
    cache = SummaryMergeCache()
    bad = json.loads(response(["valid", "second"]))
    bad["summaries"][1]["label"] = "S0"
    with pytest.raises(ValueError, match="attribution"):
        cache.accept(jobs, json.dumps(bad))
    assert not cache.values
    with pytest.raises(ValueError, match="budget"):
        cache.accept(jobs, response(["valid", "too long " * 200]))
    assert not cache.values
    cache.accept(jobs, response(["orchards", "observatories"]))
    with pytest.raises(ValueError, match="changed"):
        cache.accept(jobs, response(["changed fact", "observatories"]))
    assert cache(jobs[0]) == "orchards"


def test_fitting_merge_is_exact_and_pending_requests_are_bounded():
    short = SpineSummaryRequest("user_spine", (SpineSummaryFragment("user", "today", "one"),
                                              SpineSummaryFragment("user", "today", "two")))
    assert lossless_merge(short) == "one\ntwo"
    with pytest.raises(PendingMerge):
        SummaryMergeCache()(request())
    batches = pack_merge_batches(tuple(request(str(i)) for i in range(19)))
    assert sum(map(len, batches)) == 19
    assert all(1 <= len(batch) <= 8 for batch in batches)


def test_dependency_waves_resume_independent_sources_and_preserve_order():
    class Journal:
        def __init__(self):
            self.cache = SummaryMergeCache()
            self.waves = []

        def resolve(self, pending, phase, wave):
            self.waves.append(tuple(pending))
            jobs = tuple(pending.values())
            self.cache.accept(jobs, response([r.fragments[0].summary.split()[0] for r in jobs]))
            return True

    journal = Journal()
    groups = {"source-z": (request("orchard"), request("second")), "source-a": (request("camping"),)}
    result = compile_waves(groups, lambda rows: tuple(journal.cache(r) for r in rows), journal, "fixture")
    assert result == {"source-z": ("orchard", "second"), "source-a": ("camping",)}
    assert list(result) == list(groups)
    assert [len(wave) for wave in journal.waves] == [2, 1]


def test_prepare_stops_without_publishing_a_partial_result():
    class Journal:
        cache = SummaryMergeCache()
        def resolve(self, pending, phase, wave):
            return False
    journal = Journal()
    assert compile_waves({"source": [request()]}, lambda rows: journal.cache(rows[0]), journal, "fixture") is None
