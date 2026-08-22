from __future__ import annotations

from types import SimpleNamespace

import pytest

import memory_condense.eval.recall_guarded_cumulative_synthesis_runtime as runtime_module
from memory_condense.eval.recall_guarded_cumulative_synthesis_runtime import (
    RecallGuardedCumulativeSynthesisRuntime,
)
from memory_condense.search.selectors.causal_choice_scorer import (
    QWEN_CHOICE_CHECKPOINT_SHA256,
    QWEN_CHOICE_MODEL_ID,
    QWEN_CHOICE_MODEL_REVISION,
)


class _FakeTokenIds(list[int]):
    def tolist(self):
        return list(self)


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return " ".join(message["content"] for message in messages) + " assistant"

    def __call__(self, text, **_kwargs):
        return {"input_ids": _FakeTokenIds(range(len(str(text).split())))}


class _FakeModel:
    def __init__(self, *, max_positions: int = 4096) -> None:
        self.config = SimpleNamespace(max_position_embeddings=max_positions)
        self._embedding = SimpleNamespace(
            weight=SimpleNamespace(device="cuda:0")
        )

    def get_input_embeddings(self):
        return self._embedding


class _FakeAnswerer:
    instances = []

    def __init__(self, model_dir, **kwargs) -> None:
        self.model_dir = model_dir
        self.kwargs = kwargs
        self.model = _FakeModel()
        self.tokenizer = _FakeTokenizer()
        self._torch = object()
        self.dtype_name = "float16"
        self.max_new_tokens = kwargs["max_new_tokens"]
        self.calls = 0
        self.elapsed_s = 0.0
        self.closed = False
        self.seen_limits = []
        type(self).instances.append(self)

    def __call__(self, messages):
        assert messages
        self.calls += 1
        self.seen_limits.append(self.max_new_tokens)
        return "Miss Bee Providore"

    def close(self):
        self.closed = True


class _FakeScorer:
    instances = []

    def __init__(self, model, tokenizer, **kwargs) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.calls = 0
        self.forward_passes = 0
        self.elapsed_s = 0.0
        self.last_report = None
        self.closed = False
        type(self).instances.append(self)

    def score_candidates(self, query, candidates, *, source_timestamps=None):
        self.calls += 1
        self.forward_passes += 1
        self.elapsed_s += 0.25
        self.last_report = SimpleNamespace(query=query, count=len(candidates))
        return {candidate_id: 0.75 for candidate_id in candidates}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_runtime_dependencies(monkeypatch):
    _FakeAnswerer.instances.clear()
    _FakeScorer.instances.clear()
    verified = []

    def verify(model_dir, **kwargs):
        assert not _FakeAnswerer.instances
        verified.append((model_dir, kwargs))
        return QWEN_CHOICE_CHECKPOINT_SHA256

    monkeypatch.setattr(runtime_module, "verify_local_causal_checkpoint", verify)
    monkeypatch.setattr(runtime_module, "LocalQwenAnswerer", _FakeAnswerer)
    monkeypatch.setattr(runtime_module, "CausalChoiceScorer", _FakeScorer)
    return verified


def test_runtime_verifies_then_loads_one_fp16_model_shared_with_scorer(
    fake_runtime_dependencies,
    tmp_path,
) -> None:
    runtime = RecallGuardedCumulativeSynthesisRuntime(tmp_path / "qwen")

    assert len(fake_runtime_dependencies) == 1
    _path, verification = fake_runtime_dependencies[0]
    assert verification == {
        "model_id": QWEN_CHOICE_MODEL_ID,
        "model_revision": QWEN_CHOICE_MODEL_REVISION,
        "expected_checkpoint_sha256": QWEN_CHOICE_CHECKPOINT_SHA256,
    }
    assert len(_FakeAnswerer.instances) == 1
    assert len(_FakeScorer.instances) == 1
    answerer = _FakeAnswerer.instances[0]
    scorer = _FakeScorer.instances[0]
    assert answerer.kwargs["dtype"] == "float16"
    assert answerer.kwargs["max_new_tokens"] == 2048
    assert scorer.model is answerer.model
    assert scorer.tokenizer is answerer.tokenizer
    assert scorer.kwargs["torch_module"] is answerer._torch
    assert scorer.kwargs["checkpoint_sha256"] == QWEN_CHOICE_CHECKPOINT_SHA256
    assert scorer.kwargs["candidate_tokens"] == 256
    assert scorer.kwargs["max_prompt_tokens"] == 768
    assert runtime.identity.max_position_embeddings == 4096
    assert runtime.identity.generation_do_sample is False
    assert runtime.identity.generation_thinking is False

    runtime.close()


def test_completion_override_is_bounded_reported_and_restored(
    fake_runtime_dependencies,
    tmp_path,
) -> None:
    runtime = RecallGuardedCumulativeSynthesisRuntime(
        tmp_path / "qwen",
        max_new_tokens=10,
    )
    messages = [
        {"role": "system", "content": "Use only evidence."},
        {"role": "user", "content": "Question and excerpts"},
    ]

    answer = runtime.complete(messages, max_new_tokens=7)

    assert answer == "Miss Bee Providore"
    answerer = _FakeAnswerer.instances[0]
    assert answerer.seen_limits == [7]
    assert answerer.max_new_tokens == 10
    report = runtime.last_completion_report
    assert report is not None
    assert report.input_tokens == 7
    assert report.output_tokens == 3
    assert report.max_new_tokens == 7
    assert len(report.messages_sha256) == 64
    assert len(report.completion_sha256) == 64
    assert runtime.usage.completion_calls == 1
    assert runtime.usage.completion_input_tokens == 7
    assert runtime.usage.completion_output_tokens == 3

    runtime.close()


def test_completion_refuses_context_overflow_before_generation(
    fake_runtime_dependencies,
    tmp_path,
) -> None:
    runtime = RecallGuardedCumulativeSynthesisRuntime(tmp_path / "qwen")
    messages = [{"role": "user", "content": " ".join(["x"] * 4090)}]

    with pytest.raises(ValueError, match="exceeds the pinned context limit"):
        runtime.complete(messages, max_new_tokens=8)

    assert _FakeAnswerer.instances[0].calls == 0
    assert runtime.last_completion_report is None
    runtime.close()


def test_candidate_scoring_uses_shared_scorer_and_exposes_report(
    fake_runtime_dependencies,
    tmp_path,
) -> None:
    runtime = RecallGuardedCumulativeSynthesisRuntime(tmp_path / "qwen")

    scores = runtime.score_candidates(
        "Which restaurant?",
        {"E1": "Miss Bee Providore serves nasi goreng."},
    )

    assert scores == {"E1": 0.75}
    assert runtime.last_score_report.query == "Which restaurant?"
    assert runtime.usage.score_calls == 1
    assert runtime.usage.score_forward_passes == 1
    assert runtime.usage.score_elapsed_s == 0.25
    runtime.close()


def test_close_is_idempotent_and_prevents_further_use(
    fake_runtime_dependencies,
    tmp_path,
) -> None:
    runtime = RecallGuardedCumulativeSynthesisRuntime(tmp_path / "qwen")
    runtime.close()
    runtime.close()

    assert _FakeScorer.instances[0].closed is True
    assert _FakeAnswerer.instances[0].closed is True
    with pytest.raises(RuntimeError, match="closed"):
        runtime.complete([{"role": "user", "content": "question"}])
    with pytest.raises(RuntimeError, match="closed"):
        runtime.score_candidates("question", {"E1": "evidence"})
