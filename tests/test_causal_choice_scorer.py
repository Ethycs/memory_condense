from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from memory_condense.search.selectors.causal_choice_scorer import (
    CausalChoiceScorer,
    verify_local_causal_checkpoint,
)
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn


def _result(index: int, text: str, *, role: str = "user") -> RetrievalResult:
    turn = Turn(
        turn_id=f"turn-{index}",
        source_id="source-a",
        role=role,
        text=text,
    )
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=f"chunk-{index}",
            turn_id=turn.turn_id,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=max(1, len(text.split())),
        ),
        turn=turn,
        score=1.0 - index / 10.0,
        route="source_metadata_companion",
    )


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, choices=None):
        self.choices = choices or {" A": [11], " B": [12]}
        self.templates = []

    def __call__(self, text, *, add_special_tokens):
        del add_special_tokens
        return {"input_ids": list(self.choices.get(text, [1, 2, 3]))}

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return "rendered prompt"


class _FakeModel:
    def __init__(self, row_boosts):
        self.row_boosts = list(row_boosts)
        self.offset = 0
        self.calls = []
        self.eval_called = False
        self._embedding = torch.nn.Embedding(32, 4)

    def eval(self):
        self.eval_called = True
        return self

    def get_input_embeddings(self):
        return self._embedding

    def __call__(self, *, input_ids, attention_mask, use_cache):
        assert use_cache is False
        batch, width = input_ids.shape
        boosts = self.row_boosts[self.offset : self.offset + batch]
        if len(boosts) != batch:
            raise AssertionError("fake model ran out of row scores")
        self.offset += batch
        logits = torch.zeros((batch, width, 32), dtype=torch.float32)
        for row_index, boost in enumerate(boosts):
            for position in range(width - 1):
                target = int(input_ids[row_index, position + 1])
                logits[row_index, position, target] = float(boost)
            final_position = int(attention_mask[row_index].sum().item()) - 1
            logits[row_index, final_position, 11] = float(boost)
        self.calls.append(
            {
                "shape": tuple(input_ids.shape),
                "attention": attention_mask.detach().clone(),
                "use_cache": use_cache,
            }
        )
        return SimpleNamespace(logits=logits)


class _FailingModel(_FakeModel):
    def __call__(self, **_kwargs):
        raise RuntimeError("synthetic forward failure")


def test_choice_scorer_uses_one_no_cache_forward_and_returns_role_metadata():
    tokenizer = _FakeTokenizer()
    model = _FakeModel([4.0, -3.0])
    scorer = CausalChoiceScorer(
        model,
        tokenizer,
        torch_module=torch,
        model_id="test/model",
        model_revision="revision",
        checkpoint_sha256="digest",
        max_prompt_tokens=8,
        max_workspace_tokens=64,
    )
    candidates = [
        _result(0, "I attended Billie Eilish in Philadelphia."),
        _result(1, "The assistant suggested a concert.", role="assistant"),
    ]

    evidence = scorer.score_candidates(
        "Which concerts did I attend?",
        candidates,
        source_timestamps={"source-a": "2024-03-02"},
    )

    assert model.eval_called is True
    assert len(model.calls) == 1
    assert evidence["chunk-0"].answerability > 0.5
    assert evidence["chunk-0"].value_evidence_logit > 0.0
    assert evidence["chunk-0"].role == "user"
    assert evidence["chunk-0"].inspected is True
    assert evidence["chunk-1"].answerability < 0.5
    assert evidence["chunk-1"].role == "assistant"
    assert tokenizer.templates[0][1]["enable_thinking"] is False
    user_prompt = tokenizer.templates[0][0][1]["content"]
    assistant_prompt = tokenizer.templates[1][0][1]["content"]
    assert 'Memory author conversation role: "user"' in user_prompt
    assert 'Memory author conversation role: "assistant"' in assistant_prompt
    assert "first-person pronouns" in user_prompt
    assert "assistant, not the user" in user_prompt
    assert "Source timestamp: 2024-03-02" in user_prompt
    assert scorer.last_report is not None
    assert scorer.last_report.model_id == "test/model"
    assert scorer.last_report.model_revision == "revision"
    assert scorer.last_report.checkpoint_sha256 == "digest"
    assert scorer.last_report.forward_passes == 1
    assert scorer.last_report.retained_transformer_state_bytes == 0
    assert scorer.batch_size == 8


def test_choice_scorer_microbatches_prompts_and_reports_peak_and_total_tokens():
    tokenizer = _FakeTokenizer()
    model = _FakeModel([4.0] * 5)
    scorer = CausalChoiceScorer(
        model,
        tokenizer,
        torch_module=torch,
        batch_size=2,
        max_prompt_tokens=8,
        max_workspace_tokens=64,
    )
    candidates = [_result(index, f"candidate {index}") for index in range(5)]

    evidence = scorer.score_candidates("question", candidates)

    assert len(evidence) == 5
    assert len(model.calls) == 3
    assert [call["shape"][0] for call in model.calls] == [2, 2, 1]
    assert scorer.last_report is not None
    assert scorer.last_report.forward_passes == 3
    assert scorer.last_report.workspace_tokens == 6
    assert scorer.last_report.total_sequence_tokens == 15


def test_choice_scorer_enforces_workspace_and_leaves_tail_fail_open():
    scorer = CausalChoiceScorer(
        _FakeModel([4.0]),
        _FakeTokenizer(),
        torch_module=torch,
        max_candidates=1,
        batch_size=8,
        max_prompt_tokens=8,
        max_workspace_tokens=18,
    )
    candidates = [_result(0, "answer"), _result(1, "tail")]

    evidence = scorer.score_candidates("question", candidates)

    assert scorer.batch_size == 2
    assert evidence["chunk-0"].inspected is True
    assert evidence["chunk-1"].inspected is False
    assert evidence["chunk-1"].answerability == 0.5
    assert scorer.last_report is not None
    assert scorer.last_report.workspace_tokens <= 18

    with pytest.raises(ValueError, match="cannot hold one candidate prompt"):
        CausalChoiceScorer(
            _FakeModel([]),
            _FakeTokenizer(),
            torch_module=torch,
            max_prompt_tokens=8,
            max_workspace_tokens=7,
        )


def test_choice_scorer_supports_equal_length_full_choice_sequences():
    tokenizer = _FakeTokenizer(
        {" direct": [11, 13], " indirect": [12, 14]}
    )
    scorer = CausalChoiceScorer(
        _FakeModel([3.0, 1.0]),
        tokenizer,
        torch_module=torch,
        direct_choice=" direct",
        indirect_choice=" indirect",
        max_prompt_tokens=8,
        max_workspace_tokens=64,
    )

    evidence = scorer.score_candidates("question", {"candidate": "memory"})

    assert evidence["candidate"].answerability > 0.5
    assert scorer.last_report is not None
    assert scorer.last_report.choice_sequence_tokens == (2, 2)

    with pytest.raises(ValueError, match="equal token length"):
        CausalChoiceScorer(
            _FakeModel([]),
            _FakeTokenizer({" A": [11], " B": [12, 14]}),
            torch_module=torch,
        )


def test_choice_scorer_failure_is_neutral_unless_strict():
    candidate = _result(0, "answer")
    safe = CausalChoiceScorer(
        _FailingModel([]),
        _FakeTokenizer(),
        torch_module=torch,
        max_prompt_tokens=8,
        max_workspace_tokens=64,
    )

    evidence = safe.score_candidates("question", [candidate])

    assert evidence["chunk-0"].answerability == 0.5
    assert evidence["chunk-0"].inspected is False
    assert safe.last_report is not None
    assert safe.last_report.forward_passes == 0
    assert safe.last_report.fallback_reason == (
        "RuntimeError: synthetic forward failure"
    )

    strict = CausalChoiceScorer(
        _FailingModel([]),
        _FakeTokenizer(),
        torch_module=torch,
        strict=True,
        max_prompt_tokens=8,
        max_workspace_tokens=64,
    )
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        strict.score_candidates("question", [candidate])


def test_choice_companion_selection_applies_query_role_before_answerability():
    candidates = [
        _result(0, "I framed a ticket stub."),
        _result(1, "I took a photo at the show."),
        _result(2, "I attended Billie Eilish at Wells Fargo Center."),
        _result(3, "You attended Billie Eilish.", role="assistant"),
    ]
    # Assistant has the strongest neural score, but the autobiographical query
    # requires first-person/user evidence. Among user rows, the true answer wins.
    scorer = CausalChoiceScorer(
        _FakeModel([0.0, 2.0, 3.0, 5.0]),
        _FakeTokenizer(),
        torch_module=torch,
        max_prompt_tokens=8,
        max_workspace_tokens=128,
    )

    selected = scorer.select_source_companions(
        "Which concerts did I attend?",
        {"source-a": candidates},
    )

    assert selected["source-a"] is candidates[2]
    assert scorer.last_source_companion_report is not None
    assert (
        scorer.last_source_companion_report.preferred_evidence_role == "user"
    )
    assert scorer.last_source_companion_report.selected_chunk_ids == {
        "source-a": "chunk-2"
    }
    selected_membership = (
        scorer.last_source_companion_report.selected_membership_scores
    )
    assert set(selected_membership) == {"source-a"}
    assert 0.0 < selected_membership["source-a"] < 1.0
    assert scorer.last_source_companion_report.retained_transformer_state_bytes == 0


def test_choice_companion_blends_answerability_with_surface_value(monkeypatch):
    import memory_condense.search.selectors.coverage_selector as coverage_selector_module

    generic = _result(0, "I kept a generic ticket stub.")
    answer = _result(
        1,
        "I attended Billie Eilish at Wells Fargo Center in Philadelphia.",
    )
    surface_scores = {
        generic.chunk.text: 0.25,
        answer.chunk.text: 0.75,
    }
    monkeypatch.setattr(
        coverage_selector_module,
        "_surface_value_evidence",
        lambda text, _timestamp: surface_scores[text],
    )
    scorer = CausalChoiceScorer(
        _FakeModel(
            [
                math.log(0.426 / (1.0 - 0.426)),
                math.log(0.376 / (1.0 - 0.376)),
            ]
        ),
        _FakeTokenizer(),
        torch_module=torch,
        max_prompt_tokens=8,
        max_workspace_tokens=64,
    )

    selected = scorer.select_source_companions(
        "Which concerts did I attend?",
        {"source-a": [generic, answer]},
    )

    # Neural-only ranking prefers the generic row (.426 > .376). The shared
    # global value prior reverses it: .70*.376 + .30*.75 > .70*.426 + .30*.25.
    assert selected["source-a"] is answer


def test_choice_candidate_bound_is_reported_as_fail_open_provenance():
    scorer = CausalChoiceScorer(
        _FakeModel([2.0]),
        _FakeTokenizer(),
        torch_module=torch,
        max_candidates=1,
        max_prompt_tokens=8,
        max_workspace_tokens=64,
    )
    candidates = [_result(0, "first"), _result(1, "uninspected tail")]

    evidence = scorer.score_candidates("question", candidates)

    assert evidence["chunk-0"].inspected is True
    assert evidence["chunk-1"].inspected is False
    assert scorer.last_report is not None
    assert scorer.last_report.inspected_candidates == 1
    assert scorer.last_report.input_candidates == 2
    assert scorer.last_report.fallback_reason == (
        "candidate_bound: inspected 1 of 2 candidates"
    )


def test_checkpoint_verification_is_exact_and_offline(tmp_path):
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    weights = b"safe causal weights"
    (tmp_path / "model.safetensors").write_bytes(weights)
    expected = verify_local_causal_checkpoint(
        tmp_path,
        model_id="example/test-model",
        model_revision="revision-1",
    )
    assert (
        verify_local_causal_checkpoint(
            tmp_path,
            model_id="example/test-model",
            model_revision="revision-1",
            expected_checkpoint_sha256=expected,
        )
        == expected
    )

    # Model identity and every behaviorally consumed metadata file are part of
    # the digest, not merely the safetensors bytes.
    (tmp_path / "tokenizer_config.json").write_text(
        '{"chat_template":"changed"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected.*sha256"):
        verify_local_causal_checkpoint(
            tmp_path,
            model_id="example/test-model",
            model_revision="revision-1",
            expected_checkpoint_sha256=expected,
        )


def test_checkpoint_manifest_binds_shard_index_and_revision(tmp_path):
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "shard-a.safetensors").write_bytes(b"first")
    (tmp_path / "shard-b.safetensors").write_bytes(b"second")
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "shard-a.safetensors",
            "lm_head.weight": "shard-b.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    expected = verify_local_causal_checkpoint(
        tmp_path,
        model_id="example/sharded",
        model_revision="revision-1",
    )

    with pytest.raises(ValueError, match="unexpected.*sha256"):
        verify_local_causal_checkpoint(
            tmp_path,
            model_id="example/sharded",
            model_revision="revision-2",
            expected_checkpoint_sha256=expected,
        )

    index["weight_map"]["extra.weight"] = "shard-a.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected.*sha256"):
        verify_local_causal_checkpoint(
            tmp_path,
            model_id="example/sharded",
            model_revision="revision-1",
            expected_checkpoint_sha256=expected,
        )


def test_checkpoint_manifest_rejects_shard_path_escape(tmp_path):
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    escaped = tmp_path.parent / "outside.safetensors"
    escaped.write_bytes(b"outside")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes model directory"):
        verify_local_causal_checkpoint(tmp_path)


def test_checkpoint_verification_rejects_incomplete(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        verify_local_causal_checkpoint(tmp_path)
