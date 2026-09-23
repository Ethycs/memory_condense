"""Batch independent summary merges without mixing their attribution channels."""
from __future__ import annotations

import json
from collections.abc import Sequence

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy, count_tokens
from memory_condense.search.spine_summary import SPINE_SUMMARY_SYSTEM, SpineSummaryRequest, parse_spine_summary


SYSTEM = SPINE_SUMMARY_SYSTEM.replace(
    "Return ONLY JSON with one key, summary, containing a concise string within max_output_tokens.",
    "Process each labeled job independently. Return ONLY JSON with one key, summaries, containing "
    "an array of objects with exactly label and summary in input order. Aim for at most 70 words "
    "per summary, always within that job's max_output_tokens. Never mix facts between jobs.")


def lossless_merge(request: SpineSummaryRequest) -> str | None:
    if type(request) is not SpineSummaryRequest:
        raise TypeError("merge input must be a typed summary request")
    text = "\n".join(fragment.summary for fragment in request.fragments)
    return text if count_tokens(text) <= request.max_output_tokens else None


def merge_batch_messages(requests: Sequence[SpineSummaryRequest]):
    if not 1 <= len(requests) <= 8 or any(type(r) is not SpineSummaryRequest for r in requests):
        raise ValueError("a merge batch requires one to eight typed summary jobs")
    jobs = [dict(json.loads(request.messages[1]["content"]), label=f"S{i}")
            for i, request in enumerate(requests)]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": canonical_json({"jobs": jobs})}]
    if count_chat_prompt_token_proxy(messages) > 7000:
        raise ValueError("summary merge batch exceeds its prompt budget")
    return messages


def pack_merge_batches(requests: Sequence[SpineSummaryRequest]):
    batches, current = [], []
    for request in requests:
        try:
            merge_batch_messages((*current, request))
        except ValueError:
            if not current:
                raise
            batches.append(tuple(current))
            current = []
            merge_batch_messages((request,))
        current.append(request)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def parse_merge_batch(response: str, requests: Sequence[SpineSummaryRequest]) -> tuple[str, ...]:
    merge_batch_messages(requests)
    body = json.loads(response)
    if type(body) is not dict or set(body) != {"summaries"} or type(body["summaries"]) is not list:
        raise ValueError("merge response must contain only the summaries array")
    if len(body["summaries"]) != len(requests):
        raise ValueError("merge response omitted or added jobs")
    output = []
    for i, (row, request) in enumerate(zip(body["summaries"], requests, strict=True)):
        if type(row) is not dict or set(row) != {"label", "summary"} or row["label"] != f"S{i}":
            raise ValueError("merge response changed attribution or order")
        output.append(parse_spine_summary(canonical_json({"summary": row["summary"]}), request))
    return tuple(output)


class PendingMerge(Exception):
    def __init__(self, request):
        self.request = request
        super().__init__(request.prompt_sha256)


class SummaryMergeCache:
    """Stop at an unresolved dependency; independent sources can keep planning."""
    def __init__(self):
        self.values = {}

    def __call__(self, request):
        merged = lossless_merge(request)
        if merged is not None:
            return merged
        if request.prompt_sha256 not in self.values:
            raise PendingMerge(request)
        return parse_spine_summary(canonical_json({"summary": self.values[request.prompt_sha256]}), request)

    def accept(self, requests, response):
        # Validate the entire response before admitting any dependency.
        summaries = parse_merge_batch(response, requests)
        for request, summary in zip(requests, summaries, strict=True):
            old = self.values.get(request.prompt_sha256)
            if old is not None and old != summary:
                raise ValueError("an authenticated summary merge changed")
        self.values.update((r.prompt_sha256, s) for r, s in zip(requests, summaries, strict=True))
