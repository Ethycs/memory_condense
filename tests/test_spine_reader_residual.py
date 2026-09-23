from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from tools import evaluate_spine_reader_residual as diagnostic


def population():
    rows, observations = [], []
    for ordinal in range(100):
        messages = [{"role": "system", "content": "Use the excerpts."},
                    {"role": "user", "content": f"Raw evidence {ordinal}. Question?"}]
        prediction = "original prediction"
        question = {"ordinal": ordinal, "question_id": f"q{ordinal}", "retrieval_query": "Question?"}
        call = {"arm": "as_of", "question": question, "messages": messages,
                "messages_sha256": identity_sha256(messages)}
        row = {"arm": "as_of", "ordinal": ordinal, "question_id": question["question_id"],
               "prediction_sha256": quote_sha256(prediction), "correct": ordinal < 80,
               "reference_sha256": quote_sha256("GOLD SECRET"), "verdict": "GOLD SECRET"}
        rows.extend((row, {**row, "arm": "semantic_seeds"}))
        observations.append((call, SimpleNamespace(sha256="a" * 64,
            payload={"messages": messages, "measurement": {
                "prediction": prediction, "prediction_sha256": quote_sha256(prediction)}})))
    return rows, observations


def test_all_misses_use_exact_messages_without_prior_answers_or_gold():
    rows, observations = population()
    cases = diagnostic.select_cases(rows, observations)
    assert [c["ordinal"] for c in cases] == list(range(80, 100))
    assert [c["messages"] for c in cases] == [o[0]["messages"] for o in observations[80:]]
    assert "GOLD SECRET" not in str(cases)
    assert "original prediction" not in str(cases)


@pytest.mark.parametrize("corruption", ["question", "prediction", "messages", "population", "duplicate"])
def test_corrupted_source_cannot_define_an_outbound_population(corruption):
    rows, observations = population()
    if corruption == "question":
        rows[-2]["question_id"] = "foreign"
    elif corruption == "prediction":
        rows[-2]["prediction_sha256"] = "b" * 64
    elif corruption == "messages":
        observations[-1][1].payload["messages"] = [{"role": "user", "content": "replacement"}]
    elif corruption == "population":
        rows.pop()
    else:
        rows[-1] = deepcopy(rows[-2])
    with pytest.raises(ValueError):
        diagnostic.select_cases(rows, observations)


def test_partial_reader_population_cannot_be_sealed_or_open_gold():
    cases = diagnostic.select_cases(*population())
    preflight = SimpleNamespace(sha256="a" * 64, payload={"cases": cases})
    batch = SimpleNamespace(logical_completions=["answer"], unique_records=[])
    with pytest.raises(ValueError, match="whole diagnostic answer population"):
        diagnostic.reader_payload(preflight, batch)


def test_reader_failure_prevents_reference_loading(monkeypatch, tmp_path):
    def incomplete(*args):
        raise ValueError("incomplete reader journals")
    def forbidden():
        pytest.fail("references opened before complete reader journals")
    monkeypatch.setattr(diagnostic, "answers", incomplete)
    monkeypatch.setattr(diagnostic, "load_references", forbidden)
    with pytest.raises(ValueError, match="incomplete"):
        diagnostic.judge(tmp_path)


def test_preflight_cannot_redirect_raw_packets_to_qwen(monkeypatch, tmp_path):
    from tools.matched_eval.artifacts import publish_sealed_json
    inputs = {"source_root": str(tmp_path), "cases": diagnostic.select_cases(*population())}
    monkeypatch.setattr(diagnostic, "source_inputs", lambda _: inputs)
    bad = deepcopy(diagnostic.payload(inputs))
    bad["reader_model"] = "qwen3-8b"
    publish_sealed_json(tmp_path / "preflight.json", bad)
    with pytest.raises(ValueError, match="diagnostic model"):
        diagnostic.load_preflight(tmp_path)


def test_reader_preserves_omitted_temperature_and_exact_request_population(monkeypatch, tmp_path):
    cases = diagnostic.select_cases(*population())
    preflight = SimpleNamespace(sha256="a" * 64, payload={"cases": cases})
    def capture(**kwargs):
        runtime = kwargs["runtime_factory"](None)
        try:
            assert runtime.provenance.model_dump()["request_options"] == {"timeout": 180.0}
            assert runtime.population.unique_prompt_count == len(cases)
            assert kwargs["authorized_provider_calls"] == len(cases)
            assert kwargs["enable_provider"] is False
        finally:
            runtime.close()
        return "checked"
    monkeypatch.setattr(diagnostic, "_run_exactly_authorized", capture)
    assert diagnostic.reader_batch(tmp_path, preflight, False) == "checked"
