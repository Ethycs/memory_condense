"""Source-bound, role-separated summaries emitted alongside a chat answer."""
from __future__ import annotations

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens


FORMAT = 'inline-exchange-memory-v1'
SUMMARY_TOKEN_LIMIT = 128


def validate_summary(value, text):
    if type(value) is not dict or set(value) != {'summary', 'support'}:
        raise ValueError('Inline summary requires summary and support')
    summary, support = value['summary'], value['support']
    if (type(summary) is not str or not summary.strip()
            or count_tokens(summary) > SUMMARY_TOKEN_LIMIT):
        raise ValueError('Inline summary is empty or exceeds its token budget')
    if (type(support) is not list or not 1 <= len(support) <= 4
            or any(type(q) is not str or not q.strip() or q not in text
                   or count_tokens(q) > 32 for q in support)):
        raise ValueError('Inline support must quote its own source exactly')
    return dict(summary=summary, support=list(support))


def source_identity(event_id, role, text):
    return dict(event_id=event_id, role=role, text_sha256=quote_sha256(text))


def seal_record(*, input_event, output_id, answer, summaries, packet_ids, response):
    if input_event.role != 'user':
        raise ValueError('Inline exchange summaries require a user input')
    value = dict(format=FORMAT,
        input=source_identity(input_event.event_id, 'user', input_event.text),
        output=source_identity(output_id, 'assistant', answer),
        summaries={role: validate_summary(summaries[role], text)
                   for role, text in (('user', input_event.text), ('assistant', answer))},
        packet_ids=list(packet_ids), model=response.get('response_model'),
        request_sha256=response.get('request_sha256'))
    return value | dict(receipt_sha256=identity_sha256(value))


def summaries_for_rows(rows):
    """Reuse only exact, completed I/O pairs; never promote assistant to user.

    An eager compiler may already have admitted the user turn. In that case
    the assistant summary remains usable, without rewriting the earlier turn.
    Missing, edited, or malformed receipts use ordinary summarization.
    """
    by_id = {row['turn_id']: row for row in rows}
    result = {}
    for output in rows:
        metadata = output.get('metadata', {})
        record = metadata.get('inline_memory')
        if not isinstance(record, dict):
            continue
        try:
            body = {k: v for k, v in record.items() if k != 'receipt_sha256'}
            if (record['format'] != FORMAT or identity_sha256(body) != record['receipt_sha256']
                    or output['role'] != 'assistant'
                    or record['output'] != source_identity(output['turn_id'], 'assistant', output['text'])
                    or record['input']['role'] != 'user'
                    or metadata['io']['input_event_id'] != record['input']['event_id']
                    or metadata['io']['packet_ids'] != record['packet_ids']
                    or metadata['response']['content'] != output['text']
                    or set(record['summaries']) != {'user', 'assistant'}):
                continue
            assistant = validate_summary(record['summaries']['assistant'], output['text'])
            user = by_id.get(record['input']['event_id'])
            if user is not None:
                if (user['role'] != 'user' or record['input'] != source_identity(
                        user['turn_id'], 'user', user['text'])):
                    continue
                summary = validate_summary(record['summaries']['user'], user['text'])
                result.setdefault(user['turn_id'], summary | dict(receipt_sha256=record['receipt_sha256']))
            result[output['turn_id']] = assistant | dict(receipt_sha256=record['receipt_sha256'])
        except (KeyError, TypeError, ValueError):
            continue
    return result
