"""Keep one current assistant/tool round trip live; retrieve older work from memory."""
import argparse
from pathlib import Path

from tools import native_spine_engineering_session as session
from tools import native_spine_engineering_read240 as reader


def tool_cycle_messages(base, assistant_action, observation):
    messages = [*base, {'role': 'assistant', 'content': assistant_action},
                {'role': 'user', 'content': 'Tool result:\n'}]
    available = session.PROMPT_CAP - session.count_chat_prompt_token_proxy(messages) - 32
    if available < 128:
        raise ValueError('current tool round trip cannot fit the fixed request budget')
    messages[-1]['content'] += session.prefix(observation, min(session.LIVE_CAP, available))
    if session.count_chat_prompt_token_proxy(messages) > session.PROMPT_CAP:
        raise ValueError('tool round trip exceeded the fixed request budget')
    return messages


def run(root):
    if (root / 'STOP').exists():
        raise ValueError('gateway stopped; reconcile before continuing')
    read_adapter = session.old.load(root / 'read-tool-adapter.json')
    if read_adapter.payload['implementation_sha256'] != session.evaluation.digest(reader.__file__):
        raise ValueError('bound reader adapter changed')
    adapter_path = root / 'tool-cycle-adapter-v2.json'
    specification = {
        'implementation_sha256': session.evaluation.digest(__file__),
        'read_adapter_sha256': read_adapter.sha256, 'max_prompt_tokens': session.PROMPT_CAP,
        'live_policy': 'Only the immediately preceding assistant action and tool observation use normal live chat roles. Older events require memory.',
        'event_prefix_sha256s': [session.old.load(p).sha256 for p in sorted((root / 'events').glob('*.json'))],
        'reason': 'Match the working replay tool protocol; do not restart the agent from a bare user message after every tool call.'}
    if adapter_path.exists():
        adapter = session.old.load(adapter_path)
        if (adapter.payload['implementation_sha256'] != specification['implementation_sha256']
                or adapter.payload['read_adapter_sha256'] != read_adapter.sha256):
            raise ValueError('tool cycle adapter changed after activation')
    else:
        adapter = session.save(adapter_path, specification)
    original_save, original_messages = session.save, session.messages_for

    def messages_for(prompt, context, latest):
        base = original_messages(prompt, context, '')
        if not latest:
            return base
        paths = sorted((root / 'events').glob('*.json'))
        action, tool, activity = [session.payload(p) for p in paths[-3:]]
        if (action['kind'] != 'action' or tool['kind'] != 'tool' or activity['kind'] != 'activity'
                or tool['text'] != latest or len({r['step'] for r in (action, tool, activity)}) != 1):
            raise ValueError('live tool pair is not the immediately preceding recorded exchange')
        return tool_cycle_messages(base, action['text'], latest)

    def bound_save(path, value):
        if path.name == 'request.json' and 'messages' in value and 'step' in value:
            value = dict(value, tool_adapter_sha256=read_adapter.sha256, tool_cycle_adapter_sha256=adapter.sha256)
        return original_save(path, value)

    session.SYSTEM = reader.SYSTEM
    session.execute = reader.execute
    session.messages_for = messages_for
    session.save = bound_save
    session.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
