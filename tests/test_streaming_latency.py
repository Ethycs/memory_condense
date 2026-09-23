from types import SimpleNamespace

import pytest

from memory_condense.eval.streaming_latency import measure_streaming_answer, latency_distribution


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class Stream:
    def __init__(self, clock, schedule):
        self.clock, self.schedule, self.closed = clock, schedule, False

    def __iter__(self):
        for delay, event in self.schedule:
            self.clock.value += delay
            if isinstance(event, Exception):
                raise event
            yield event

    def close(self):
        self.closed = True


def event(content=None, finish=None, **extra):
    return {"choices": [{"index": 0, "delta": {"content": content, **extra}, "finish_reason": finish}]}


def setup(schedule):
    clock = Clock()
    stream = Stream(clock, schedule)
    requests = []

    def create(**request):
        requests.append(request)
        clock.value += 2  # Buffered stream creation belongs in TTFT.
        return stream

    def prepare():
        clock.value += 3  # Actual retrieval/packing callback belongs in E2E.
        return [{"role": "user", "content": "question"}]

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return dict(client=client, model="test", prepare_prompt=prepare, clock=clock), stream, requests


def test_visible_ttft_includes_preparation_and_blocking_create_but_not_hidden_delta():
    args, stream, requests = setup([
        (1, event(role="assistant")), (2, event(reasoning_content="hidden")),
        (4, event("hello")), (5, event(" world")), (1, event(finish="stop")),
        (2, {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}}),
    ])
    result = measure_streaming_answer(**args)
    assert result["prepare_s"] == 3
    assert result["first_api_event_s"] == 3
    assert result["api_ttft_s"] == 9
    assert result["e2e_ttft_s"] == 12
    assert result["api_total_s"] == 17
    assert result["e2e_total_s"] == 20
    assert result["visible_generation_s"] == 5
    assert result["prediction"] == "hello world"
    assert result["visible_event_count"] == 2
    assert result["usage"]["completion_tokens"] == 2
    assert len(requests) == 1 and requests[0]["stream"] is True and stream.closed


@pytest.mark.parametrize("schedule", [
    [(1, event("partial"))], [(1, event(finish="stop"))],
    [(1, event("partial")), (1, RuntimeError("disconnect"))],
    [(1, event("done", "stop")), (1, event("extra"))],
])
def test_incomplete_stream_never_becomes_a_success_or_retry(schedule):
    args, stream, requests = setup(schedule)
    with pytest.raises((ValueError, RuntimeError)):
        measure_streaming_answer(**args)
    assert len(requests) == 1 and stream.closed


def test_truncation_is_explicit_and_single_visible_delta_is_detectable():
    args, stream, _ = setup([(1, event("all at once", "length"))])
    result = measure_streaming_answer(**args)
    assert result["finish_reason"] == "length"
    assert result["visible_event_count"] == 1 and result["visible_generation_s"] == 0


def test_percentile_and_empty_population():
    assert latency_distribution(list(range(1, 101)))["p95_s"] == 95
    assert latency_distribution([1, 2])["median_s"] == 1.5
    with pytest.raises(ValueError):
        latency_distribution([])
