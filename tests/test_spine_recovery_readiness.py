from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tools.execute_spine_transport_recovery import require_readiness
from tools.matched_eval.artifacts import publish_sealed_json
from tools.probe_spine_gateway_readiness import READINESS_PREFLIGHT
from tests.test_spine_transport_recovery_stage import Client


NOW = datetime(2026, 9, 10, 5, tzinfo=timezone.utc)


def fixture(root, *, age=0, record=True, failed=False, preflight_body=None):
    p = deepcopy(preflight_body or READINESS_PREFLIGHT)
    preflight, _ = publish_sealed_json(root / "preflight.json", p)
    results = []
    for probe in p["probes"]:
        shas = ["unrecorded"]
        if record:
            runtime = FastCompletionRuntime(checkpoint_dir=root / probe["name"],
                prompt_population=[probe["messages"]], model=probe["model"], client=Client(),
                max_prompt_tokens=p["max_prompt_tokens"], max_new_tokens=p["max_new_tokens"],
                max_concurrency=1, retries=0, request_options=probe["request_options"],
                benchmark_provenance={"readiness_preflight_sha256": preflight.sha256, "probe_name": probe["name"]})
            try:
                shas = [r.response_journal_sha256 for r in runtime.run().unique_records]
            finally:
                runtime.close()
        results.append({"probe": probe["name"], "model": probe["model"],
            "status": "failed_or_unacknowledged" if failed else "completed",
            "observed_utc": (NOW - timedelta(seconds=age)).isoformat(), "response_journal_shas": shas})
    return publish_sealed_json(root / "report.json", {"preflight_sha256": preflight.sha256, "results": results})[0]


def test_fresh_completed_probes_replay_without_new_calls(tmp_path):
    report = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert require_readiness(report.path, now=NOW).sha256 == report.sha256
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_success_flag_without_authenticated_responses_is_rejected(tmp_path):
    report = fixture(tmp_path, record=False)
    with pytest.raises(RuntimeError):
        require_readiness(report.path, now=NOW)


def test_failed_probe_blocks_recovery(tmp_path):
    report = fixture(tmp_path, record=False, failed=True)
    with pytest.raises(ValueError, match="has not passed"):
        require_readiness(report.path, now=NOW)


@pytest.mark.parametrize("age", [-1, 301])
def test_future_or_stale_observation_blocks_recovery(tmp_path, age):
    report = fixture(tmp_path, age=age, record=False)
    with pytest.raises(ValueError, match="last five minutes"):
        require_readiness(report.path, now=NOW)


def test_changed_qwen_input_cannot_claim_synthetic_readiness(tmp_path):
    p = deepcopy(READINESS_PREFLIGHT)
    p["probes"][0]["messages"][1]["content"] = "Different input outside the declared summary probe."
    report = fixture(tmp_path, record=False, preflight_body=p)
    with pytest.raises(ValueError, match="summary-only readiness protocol"):
        require_readiness(report.path, now=NOW)
