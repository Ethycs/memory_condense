"""Experimental question-slot coverage and conversation scope on top of v7."""
from memory_condense.eval.spine_reader_policy_v7 import SPINE_READER_SYSTEM_PROMPT_V7


SPINE_READER_SYSTEM_PROMPT_V9 = SPINE_READER_SYSTEM_PROMPT_V7 + (
    '\nUse the question to organize the answer: treat each requested aspect as a '
    'slot to fill from the matching user statements. For each slot, collect all '
    'coexisting named items and their requested attributes before writing. A later '
    'choice replaces an earlier statement only when they actually conflict; it '
    'does not erase compatible facts elsewhere in that conversation. When the '
    'question asks for several aspects, a compact list with one entry per aspect '
    'can prevent omissions.\nKeep the scope of each claim: a general preference, '
    'a particular occasion, and a separate task are not interchangeable. A '
    'retrieved conversation can contain unrelated requests; include a detail '
    'only if it answers an aspect of this question. Resolve corrections within '
    'the relevant situation before combining conversations. Do not silently '
    'merge conflicting accounts into one routine or plan; distinguish their '
    'stated times or situations when both are needed. Before returning, check '
    'that every requested slot includes its supported specifics and that every '
    'answer clause belongs to a requested slot. Keep this check internal.'
)


def apply_reader(messages):
    """Upgrade only v7 instructions, preserving all evidence and question bytes."""
    if not messages or messages[0] != {'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V7}:
        raise ValueError('v9 requires the unchanged v7 reader baseline')
    return [{'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V9},
            *(dict(message) for message in messages[1:])]
