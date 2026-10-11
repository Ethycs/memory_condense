import json

import numpy as np
import pytest

from memory_condense.runtime.artifacts import read
from memory_condense.runtime.semantic_summary import (
    SEMANTIC_VALIDATION_SHA256,
    SemanticLocalFirstSummaries,
    policy_root_name,
)
from memory_condense.runtime.summary_fallback import SummaryEscalationError


def messages(source="Release r18 is planned, not deployed."):
    return [
        {"role": "system", "content": "Compile source summaries."},
        {"role": "user", "content": json.dumps({"fragments": [
            {"label": "T0", "speaker": "user", "fragment": source},
        ]})},
    ]


def response(content):
    return {"content": content, "finish_reason": "stop", "request_sha256": "local"}


class Encoder:
    def embed_queries(self, texts):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "r18" in lowered:
                vectors.append([1.0, 0.05])
            elif "r19" in lowered:
                vectors.append([0.9, 0.1])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class Local:
    def __init__(self):
        self.calls = 0
        self.encoder = Encoder()

    def call(self, *args, **kwargs):
        self.calls += 1
        return response("invalid local output")

    def embedding(self):
        return self.encoder


class API:
    def __init__(self, summary):
        self.summary = summary
        self.calls = 0
        self.identity = {"provider": "openai", "model": "frontier"}

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return response(json.dumps({"atoms": [{"label": "T0", "summary": self.summary}]}))


def test_proxy_semantic_policy_accepts_paraphrase_with_exact_support(tmp_path):
    local, api = Local(), API("Release r18 remains scheduled and has not shipped.")
    runtime = SemanticLocalFirstSummaries(local, tmp_path/policy_root_name(), api)
    result = runtime.call("raw", messages(), scope="live")
    atom = json.loads(result["content"])["atoms"][0]
    assert atom["summary"] == api.summary
    assert atom["support"] == ["Release r18 is planned, not deployed."]
    assert atom["support"][0] in "Release r18 is planned, not deployed."
    request = read(next(tmp_path.rglob("request.json")))
    assert request["provider"]["summary_validation"]["sha256"] == SEMANTIC_VALIDATION_SHA256
    assert SEMANTIC_VALIDATION_SHA256[:16] in str(next(tmp_path.rglob("accepted.json")))


def test_proxy_health_exposes_semantic_policy_hash(tmp_path, monkeypatch):
    from memory_condense.runtime import sessions
    monkeypatch.setattr(sessions, "LocalRuntime", lambda *args, **kwargs: Local())
    factory = sessions.NativeProxySessions(tmp_path)
    status = factory.summary_status()
    assert status["validation_policy_sha256"] == SEMANTIC_VALIDATION_SHA256
    assert status["validation_policy"]["format"] == "semantic-summary-grounding-v1"


@pytest.mark.parametrize("summary", [
    "Release r19 remains scheduled and has not shipped.",
    "Release r18 deployed.",
])
def test_proxy_semantic_policy_rejects_changed_fact_or_status(tmp_path, summary):
    runtime = SemanticLocalFirstSummaries(Local(), tmp_path/policy_root_name(), API(summary))
    with pytest.raises(SummaryEscalationError):
        runtime.call("raw", messages(), scope="live")
    assert not list(tmp_path.rglob("accepted.json"))
