"""OpenAI Responses translation for memory mode; one internal message shape.

The proxy keeps a single internal message form: the chat-completions
``{role, content, tool_calls, tool_call_id}`` dict that
:func:`proxy_memory_wire.canonical_message` defines.  That form is not merely a
serialization choice.  It is the message identity ``MemoryProxy._new_messages``
deduplicates resent history by, and the dicts ``MemoryProxy._recent`` walks to
assemble the resend window. Responses is translated at the edge, with explicit
fields for custom tools, namespaces, assistant phases and image content. The
shared recall and publication pipeline remains the source of memory selection.

One consequence is worth stating because it is a feature: a conversation started
over ``/v1/chat/completions`` can be continued over ``/v1/responses`` under the
same conversation ID, because both land in the same namespace and both reduce to
the same canonical messages. Responses-specific tools and image content require
the Responses endpoint when that history is continued.

Everything here is pure: payload in, payload out.  No IO, no session state.
"""
from copy import deepcopy
import json


#: Content block types that carry prose.  ``text`` is the Codex spelling.
TEXT_BLOCKS = ('input_text', 'output_text', 'text')

#: Terminal stream events.  Each carries the whole final response object, which
#: is why assembly here needs no delta accumulation at all.
TERMINAL_EVENTS = ('response.completed', 'response.failed', 'response.incomplete')


def item_type(item):
    """The item discriminator.  ``message`` may be implied by ``role`` alone."""
    kind = item.get('type')
    if kind:
        return kind
    return 'message' if 'role' in item else 'unknown'


def input_items(payload):
    """Normalize both legal spellings of ``input`` to a typed item list."""
    value = payload.get('input')
    if isinstance(value, str):
        return [dict(type='message', role='user', content=[dict(type='input_text', text=value)])]
    return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []


def block_text(item, *, strict=False):
    content = item.get('content')
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get('type') in TEXT_BLOCKS:
            parts.append(block.get('text') or '')
        elif strict:
            raise ValueError('Memory beta supports text items; use observe mode for other content')
    return ''.join(parts)


def canonical_item(item):
    """One input item as an internal message, or ``None`` for one to drop."""
    kind = item_type(item)
    if kind == 'additional_tools':
        if item.get('role') != 'developer' or not isinstance(item.get('tools'), list):
            raise ValueError('Responses additional_tools requires developer role and a tools array')
        # Responses Lite puts tool declarations in input, not top-level tools.
        # Keep this policy item native so namespaces and schemas survive recall.
        return dict(role='developer', content='', responses_tools=deepcopy(item))
    if kind == 'reasoning':
        # Opaque and signed to an upstream response the proxy never resends.
        return None
    if kind == 'message':
        role = item.get('role')
        if role not in ('system', 'developer', 'user', 'assistant'):
            raise ValueError('Unsupported message role')
        content = item.get('content')
        images = (role == 'user' and isinstance(content, list) and any(
            isinstance(block, dict) and block.get('type') == 'input_image' for block in content))
        if images:
            if any(not isinstance(block, dict) or not (
                    block.get('type') in TEXT_BLOCKS or
                    (block.get('type') == 'input_image' and isinstance(block.get('image_url'), str)))
                   for block in content):
                raise ValueError('Memory beta supports text items and inline image URLs')
            message = dict(role=role, content=block_text(item), responses_content=deepcopy(content))
        else:
            message = dict(role=role, content=block_text(item, strict=True))
        if item.get('phase') is not None:
            message['phase'] = item['phase']
        return message
    if kind in ('function_call', 'custom_tool_call'):
        custom = kind == 'custom_tool_call'
        name, arguments = item.get('name'), item.get('input' if custom else 'arguments')
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise ValueError('A tool call requires a name and string arguments/input')
        call = dict(id=item.get('call_id') or item.get('id') or '',
                    type='custom' if custom else 'function')
        call[call['type']] = dict(name=name, **{'input' if custom else 'arguments': arguments})
        if item.get('namespace') is not None:
            call['namespace'] = item['namespace']
        return dict(role='assistant', content='',
                    tool_calls=[call])
    if kind in ('function_call_output', 'custom_tool_call_output'):
        output = item.get('output')
        message = dict(role='tool', tool_call_id=item.get('call_id') or '',
                    content=output if isinstance(output, str) else json.dumps(output, ensure_ascii=False))
        if kind == 'custom_tool_call_output':
            message['responses_type'] = kind
        return message
    if kind == 'item_reference':
        # The referenced item lives in upstream storage the proxy declines to use.
        raise ValueError('Memory mode cannot resolve an item_reference; send the item inline')
    raise ValueError('Memory beta supports text and function-call items; use observe mode for ' + kind)


def dialogue(payload):
    """Validate and normalize a request into the internal dialogue messages."""
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), str):
        raise ValueError('Memory mode requires a model and an input')
    value = payload.get('input')
    if not isinstance(value, (str, list)) or not value or (isinstance(value, str) and not value.strip()):
        raise ValueError('Memory mode requires a nonempty input')
    if payload.get('max_output_tokens') is not None and (
            type(payload['max_output_tokens']) is not int or payload['max_output_tokens'] < 1):
        raise ValueError('Output token budget must be a positive integer')
    if payload.get('background'):
        # A deferred response is retrieved later by ID from upstream storage,
        # which this endpoint deliberately does not use.
        raise ValueError('Memory mode cannot serve a background response')
    messages = [m for m in (canonical_item(i) for i in input_items(payload))
                if m is not None and m['role'] not in ('system', 'developer')]
    if not messages:
        raise ValueError('Memory mode requires a nonempty input')
    if messages[-1]['role'] not in ('user', 'tool'):
        raise ValueError('Memory mode requires a final user message or tool result')
    return messages


def policies(payload):
    """Policy items carried in ``input``; ``instructions`` is handled separately."""
    return [m for m in (canonical_item(i) for i in input_items(payload))
            if m is not None and m['role'] in ('system', 'developer')]


def policy_record(payload):
    return dict(instructions=payload.get('instructions'), messages=policies(payload))


def use_inline(payload):
    """Eligibility for the inline answer/memory envelope, by Responses names."""
    items = input_items(payload)
    if not items:
        return False
    last = items[-1]
    if item_type(last) != 'message' or last.get('role') != 'user':
        return False
    content = last.get('content')
    if isinstance(content, list) and any(
            not isinstance(b, dict) or b.get('type') not in TEXT_BLOCKS for b in content):
        return False
    if not isinstance(content, (str, list)):
        return False
    # JSON schemas, tools, reasoning and truncation already own the output
    # protocol.  `text` is the Responses spelling of `response_format`.
    if any(payload.get(k) for k in ('tools', 'text', 'reasoning', 'truncation', 'include', 'tool_choice')):
        return False
    return not any(item_type(i) != 'message' for i in items)


def input_item(message):
    """Render one internal message back as zero or more input items."""
    if 'responses_tools' in message:
        return [deepcopy(message['responses_tools'])]
    if 'responses_items' in message:
        # Keep separate assistant phases and parallel tool items in wire order.
        return deepcopy(message['responses_items'])
    role, content = message['role'], message.get('content')
    if 'responses_content' in message:
        blocks = deepcopy(message['responses_content'])
        # Recall may prepend evidence to the canonical text. Keep the image
        # bytes untouched and add only that prefix to the rendered message.
        original = block_text(dict(content=blocks))
        if content != original and isinstance(content, str) and content.endswith(original):
            prefix = content[:-len(original)] if original else content
            blocks.insert(0, dict(type='input_text', text=prefix))
        return [dict(type='message', role=role, content=blocks)]
    text = content if isinstance(content, str) else ''.join(
        b.get('text') or '' for b in content or []
        if isinstance(b, dict) and b.get('type') in TEXT_BLOCKS)
    if role in ('tool', 'function'):
        return [dict(type=message.get('responses_type', 'function_call_output'),
                     call_id=message.get('tool_call_id') or '', output=text)]
    if role == 'assistant':
        # Assistant prose replayed into `input` must use output_text; input_text
        # is rejected for that role.
        items = [dict(type='message', role='assistant', content=[dict(type='output_text', text=text)])] \
            if text.strip() else []
        if items and message.get('phase') is not None:
            items[0]['phase'] = message['phase']
        for call in message.get('tool_calls') or []:
            custom = call.get('type') == 'custom'
            function = call.get('custom' if custom else 'function') or {}
            item = dict(type='custom_tool_call' if custom else 'function_call',
                        call_id=call.get('id') or '', name=function.get('name') or '')
            field = 'input' if custom else 'arguments'
            item[field] = function.get(field) or ''
            if call.get('namespace') is not None:
                item['namespace'] = call['namespace']
            items.append(item)
        return items
    return [dict(type='message', role=role, content=[dict(type='input_text', text=text)])]


def prepare_payload(result, messages, instruction, bump):
    """Finish the upstream body: policy, the window, and no upstream state."""
    if instruction is not None:
        existing = result.get('instructions')
        result['instructions'] = (existing + '\n\n' + instruction) \
            if isinstance(existing, str) and existing.strip() else instruction
        result['max_output_tokens'] = (result.get('max_output_tokens') or 4096) + bump
    result['input'] = [item for message in messages for item in input_item(message)]
    # The proxy owns the conversation.  Server-side threading is accepted from
    # the client and dropped here, and nothing is left stored upstream.
    result.pop('previous_response_id', None)
    result.pop('conversation', None)
    result['store'] = False
    return result


def response_message(payload):
    """Collapse an ``output`` array into the internal assistant message."""
    output = payload.get('output')
    if not isinstance(output, list):
        raise ValueError('Provider did not return one assistant message')
    text, refusal, calls = [], [], []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError('Provider returned an unreadable output item')
        kind = item_type(item)
        if kind == 'reasoning':
            continue
        if kind == 'message':
            for block in item.get('content') or []:
                if not isinstance(block, dict):
                    continue
                if block.get('type') in TEXT_BLOCKS:
                    text.append(block.get('text') or '')
                elif block.get('type') == 'refusal':
                    refusal.append(block.get('refusal') or '')
        elif kind in ('function_call', 'custom_tool_call'):
            calls.extend(canonical_item(item)['tool_calls'])
        else:
            # A hosted tool ran upstream.  Its result is not an answer this
            # proxy can store, and dropping it silently would misreport the turn.
            raise ValueError('Provider returned an unsupported output item: ' + kind)
    message = dict(role='assistant', content=''.join(text))
    if calls:
        message['tool_calls'] = calls
    if refusal:
        message['refusal'] = ''.join(refusal)
    # ChatIO captures one assistant event, while Responses clients replay each
    # output item separately. Retain their boundaries for matching and resend.
    if not refusal:
        message['responses_items'] = [rendered for item in output
            if item_type(item) != 'reasoning'
            for rendered in input_item(canonical_item(item))]
    return message


def finish_reason(payload):
    """Map ``status`` onto the vocabulary the shared completion gate accepts."""
    status = payload.get('status')
    if status != 'completed':
        details = payload.get('incomplete_details')
        reason = details.get('reason') if isinstance(details, dict) else None
        return str(status) + (':' + str(reason) if reason else '')
    output = [i for i in payload.get('output') or [] if isinstance(i, dict)]
    if any(item_type(i) in ('function_call', 'custom_tool_call') for i in output):
        return 'tool_calls'
    for item in output:
        if item_type(item) == 'message' and any(
                isinstance(b, dict) and b.get('type') == 'refusal' for b in item.get('content') or []):
            return 'refusal'
    return 'stop'


def reasoning_present(payload):
    return any(isinstance(i, dict) and item_type(i) == 'reasoning' for i in payload.get('output') or [])


def public_answer(result, answer):
    """Replace the visible prose, leaving every other output item intact."""
    output = [i for i in result.get('output') or [] if isinstance(i, dict)]
    block = [dict(type='output_text', text=answer, annotations=[])]
    for item in output:
        if item_type(item) == 'message':
            item['content'] = block
            return result
    output.append(dict(type='message', id='msg_memory', role='assistant',
                       status='completed', content=block))
    result['output'] = output
    return result


def assemble_stream(events):
    """Return the terminal event's response; no delta accumulation is needed."""
    value = None
    completed_items = {}
    for event in events:
        if event.get('type') == 'response.output_item.done' and isinstance(event.get('item'), dict):
            completed_items[event['output_index']] = event['item']
        if event.get('type') in TERMINAL_EVENTS:
            value = event.get('response')
            if not isinstance(value, dict):
                raise ValueError('Provider stream ended without a response object')
    if value is None:
        raise ValueError('Provider stream ended before completion')
    if 'output' not in value and completed_items:
        # Codex can finish with a lightweight terminal envelope. Only completed
        # item events are authoritative; partial deltas must never be committed.
        if sorted(completed_items) != list(range(len(completed_items))):
            raise ValueError('Provider stream omitted completed output items')
        value = dict(value, output=[completed_items[i] for i in sorted(completed_items)])
    return deepcopy(value)


def answer_stream(payload, frame):
    """Replay one rewritten answer as a minimal, correctly ordered stream."""
    item = next((i for i in payload.get('output') or []
                 if isinstance(i, dict) and item_type(i) == 'message'), None)
    if item is None:
        raise ValueError('Rewritten response carries no assistant message')
    text = ''.join(b.get('text') or '' for b in item.get('content') or []
                   if isinstance(b, dict) and b.get('type') in TEXT_BLOCKS)
    item_id = item.get('id') or 'msg_memory'
    opening = dict(deepcopy(payload), status='in_progress', output=[])
    opening.pop('usage', None)
    part = dict(type='output_text', text=text, annotations=[])
    sequence = iter(range(1 << 30))

    def event(kind, **fields):
        return frame(dict(type=kind, sequence_number=next(sequence), **fields), kind)

    return b''.join([
        event('response.created', response=opening),
        event('response.in_progress', response=opening),
        event('response.output_item.added', output_index=0, item=dict(deepcopy(item), content=[])),
        event('response.content_part.added', item_id=item_id, output_index=0, content_index=0,
              part=dict(part, text='')),
        event('response.output_text.delta', item_id=item_id, output_index=0, content_index=0, delta=text),
        event('response.output_text.done', item_id=item_id, output_index=0, content_index=0, text=text),
        event('response.content_part.done', item_id=item_id, output_index=0, content_index=0, part=part),
        event('response.output_item.done', output_index=0, item=deepcopy(item)),
        event('response.completed', response=deepcopy(payload))])
