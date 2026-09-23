"""Read all relevant user turns and preserve their attribution and qualifications."""
from memory_condense.eval.spine_reader_policy_v6 import SPINE_READER_SYSTEM_PROMPT_V6


SPINE_READER_SYSTEM_PROMPT_V7 = SPINE_READER_SYSTEM_PROMPT_V6.replace(
    'Give a concise but complete answer, usually one to four sentences.',
    'Give a concise, complete answer as a short paragraph or list. Preserve '
    'complete coverage rather than compressing away relevant details.'
) + (
    '\nBefore composing your answer, identify the conversation or conversations '
    'that match the question\'s full situation, entities and time scope. A shared '
    'topic alone is not enough. Read all relevant user turns in those conversations, '
    'including follow-up questions and qualifications, instead of stopping at the '
    'first matching sentence. For questions about what the user asked, wanted to '
    'learn, preferred, required or planned, cover each distinct relevant point: '
    'what, how, why, alternatives, concerns, conditions and tentative next steps. '
    'A user asking whether to do something is evidence of considering it, not '
    'evidence of doing it or committing to it; preserve that distinction in the '
    'answer. Preserve explicit rejected alternatives when reporting a correction '
    'or choice. When recalling a supplied line or passage, retain its complete '
    'relevant meaning and contrast; do not replace its central clause with an ellipsis. '
    '\nCheck the speaker for every fact about the user. Assistant turns may clarify '
    'what a user reply refers to, but their suggested quantities, choices and actions '
    'are not user facts unless the user explicitly adopts them. Do not fill a user\'s '
    'unspecified details from an assistant recommendation. State the specific names '
    'and details supplied by the relevant user turns, with their original certainty. '
    'Return only the final answer, not your inventory or checking process.'
)


def apply_reader(messages, system_prompt=SPINE_READER_SYSTEM_PROMPT_V7):
    """Replace only the reader instructions; preserve raw context and question."""
    if not messages or messages[0] != {'role': 'system', 'content': SPINE_READER_SYSTEM_PROMPT_V6}:
        raise ValueError('reader comparison requires the unchanged v6 prompt')
    if type(system_prompt) is not str or not system_prompt.strip():
        raise ValueError('reader policy requires nonempty system instructions')
    result = [dict(message) for message in messages]
    result[0]['content'] = system_prompt
    return result
