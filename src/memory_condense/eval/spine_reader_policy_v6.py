"""Answer every requested part while preserving the evidence-only v5 rules."""
from memory_condense.eval.spine_reader_policy_v5 import SPINE_READER_SYSTEM_PROMPT_V5


_SHORTEST = (
    "Be as short as possible: reply with just the fact, name, number, or date asked for — "
    "no preamble, no explanation, no full sentences unless the question requires one."
)
if SPINE_READER_SYSTEM_PROMPT_V5.count(_SHORTEST) != 1:
    raise ValueError('v6 requires the unchanged v5 brevity instruction')

SPINE_READER_SYSTEM_PROMPT_V6 = SPINE_READER_SYSTEM_PROMPT_V5.replace(_SHORTEST, (
    "Give a concise but complete answer, usually one to four sentences. Cover every "
    "part of the question. For a list, plan, preference, requirement, explanation or "
    "multi-part question, include all relevant supported details, qualifications "
    "and constraints instead of returning only the first fact. Preserve specific "
    "names, numbers, reasons, corrections and negations when they clarify the answer. "
    "Read all relevant excerpts before answering; later statements in the same "
    "conversation can refine an earlier choice. Do not add unrelated facts from "
    "other conversations just because they share a topic. No preamble or reasoning "
    "trace is needed."
))


def complete_reader_messages(messages):
    if not messages or messages[0] != {'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V5}:
        raise ValueError('reader comparison requires the unchanged v5 baseline')
    result = [dict(message) for message in messages]
    result[0]['content'] = SPINE_READER_SYSTEM_PROMPT_V6
    if not result[-1]['content'].endswith('\nShort answer:'):
        raise ValueError('reader comparison requires the existing answer template')
    result[-1]['content'] = result[-1]['content'][:-len('Short answer:')] + 'Answer:'
    return result
