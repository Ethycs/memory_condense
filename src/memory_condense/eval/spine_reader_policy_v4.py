"""Qualify events while interpreting ordered, partial conversation threads."""
from memory_condense.eval.spine_reader_policy_v3 import SPINE_READER_SYSTEM_PROMPT_V3


SPINE_READER_SYSTEM_PROMPT_V4 = SPINE_READER_SYSTEM_PROMPT_V3.replace(
    'Prefer explicit start/end events to inconsistent approximate recaps.',
    'Keep separate occurrences of the same activity or title distinct. Use an '
    'explicit reported duration for the requested completed occurrence; combine '
    'start/end boundaries only when they describe that same occurrence.'
) + (
    '\nC labels group selected excerpts from one source conversation; T labels '
    'follow its original turn order. Excerpts may omit intervening turns. Use '
    'nearby replies in that conversation to resolve descriptions and references. '
    'Keep the complete qualifying details needed to answer the question, even '
    'when the final answer is short.'
)
