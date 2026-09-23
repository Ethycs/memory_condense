from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.search.native_spine_merges import neutral_key
from tests.test_native_spine_exchanges import request as make_request
from tools.native_recovered_merge_seed import validate_projection
from tools.recover_native_summary_lengths import messages


def fixture():
    job = make_request(role="assistant")
    key = neutral_key(job)
    summary = "Assistant discusses the planned visit without claiming a booking."
    binding = {"request": asdict(job), "messages": messages(job)}
    request = SimpleNamespace(sha256="request", payload={"messages": messages(job),
        "original_merge_key": key, "preflight_sha256": "plan", "backend_sha256": "backend"})
    response = SimpleNamespace(sha256="response", payload={"request_sha256": "request",
        "backend_sha256": "backend", "raw_inputs_to_qwen": False, "remote_provider_calls": 0,
        "rows": [{"stopped": True, "response": canonical_json({"summary": summary})}]})
    projection = {"original_merge_key": key, "summary": summary,
        "request_sha256": "request", "response_sha256": "response"}
    return key, binding, projection, request, response


def test_projection_admits_the_actual_summary_under_the_unchanged_original_limit():
    args = fixture()
    assert validate_projection(*args, "plan", "backend") == args[2]["summary"]
    assert "transcript_date" not in args[3].payload["messages"][1]["content"]


@pytest.mark.parametrize("defect", ["over_limit", "not_stopped", "summary", "prompt", "role", "backend"])
def test_projection_rejects_changed_source_model_output_and_incomplete_or_overlong_generation(defect):
    key, binding, projection, request, response = deepcopy(fixture())
    if defect == "over_limit":
        summary = "Many separate examples. " * 100
        response.payload["rows"][0]["response"] = canonical_json({"summary": summary})
        projection["summary"] = summary
    elif defect == "not_stopped":
        response.payload["rows"][0]["stopped"] = False
    elif defect == "summary":
        projection["summary"] = "The assistant confirmed a booking."
    elif defect == "prompt":
        request.payload["messages"][1]["content"] = "Unrelated material"
    elif defect == "role":
        binding["request"]["kind"] = "user_spine"
    elif defect == "backend":
        response.payload["backend_sha256"] = "different-model"
    with pytest.raises((ValueError, TypeError)):
        validate_projection(key, binding, projection, request, response, "plan", "backend")
