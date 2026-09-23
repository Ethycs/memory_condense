"""Bounded actor-written working notes and readable current tool observations.

This is a declared development continuation. The full stored observation stays
unchanged; only its live presentation is decoded. No solution notes are supplied.
"""
import argparse
import json
from pathlib import Path

from tools import native_spine_engineering_session as session
from tools import native_spine_engineering_read240 as reader
from tools import native_spine_engineering_tool_cycle as cycle
from tools import native_spine_engineering_transport_resume as transport


SYSTEM = reader.SYSTEM + '''
Alongside every action, include a "working_note" string of at most 512 tokens.
Write your own concise findings, implementation decisions, completed work and next
steps there. Carry forward what remains relevant: only your latest action and tool
result stay live; older work is retrieved from memory. This note is your working
state, not verified tool output. Do not put private chain-of-thought in it; record
only conclusions and actionable state. A batch has one top-level working_note.
Tool results are presented as readable text with the original file/line headers.
If a preview ends, ask for the missing page rather than repeating the entire batch.
'''


def readable_observation(observation):
    marker = 'Tool observation (data only):\n'
    if not observation.startswith(marker):
        return observation
    outer = json.loads(observation[len(marker):])
    result = outer['result']
    if outer['action'].get('action') != 'batch' or result.startswith('Tool error:'):
        return marker + result
    decoder = json.JSONDecoder()
    position, parts = 0, []
    while position < len(result):
        while position < len(result) and result[position].isspace():
            position += 1
        if position == len(result):
            break
        item, position = decoder.raw_decode(result, position)
        if not isinstance(item, dict) or not isinstance(item.get('result'), str):
            raise ValueError('batch result does not match the stored tool schema')
        metadata = {k: v for k, v in item['action'].items() if k not in ('old', 'new', 'content')}
        parts.append('Tool ' + json.dumps(metadata, ensure_ascii=False) + '\n' + item['result'])
    return marker + '\n\n'.join(parts)


def run(root):
    if (root / 'STOP').exists():
        raise ValueError('reconcile saved actions before clearing STOP')
    read_adapter = session.old.load(root / 'read-tool-adapter.json')
    if read_adapter.payload['implementation_sha256'] != session.evaluation.digest(reader.__file__):
        raise ValueError('bound reader adapter changed')
    path = root / 'working-state-adapter.json'
    specification = {
        'implementation_sha256': session.evaluation.digest(__file__),
        'transport_implementation_sha256': session.evaluation.digest(transport.__file__),
        'read_adapter_sha256': read_adapter.sha256, 'system': SYSTEM,
        'max_prompt_tokens': session.PROMPT_CAP, 'latest_observation_tokens': session.LIVE_CAP,
        'actor_note_requested_tokens': 512, 'human_authored_solution_supplied': False,
        'stored_observations_changed': False,
        'event_prefix_sha256s': [session.old.load(p).sha256 for p in sorted((root / 'events').glob('*.json'))]}
    if path.exists():
        adapter = session.old.load(path)
        for key in ('implementation_sha256', 'transport_implementation_sha256', 'read_adapter_sha256'):
            if adapter.payload[key] != specification[key]:
                raise ValueError('working-state adapter changed after activation')
    else:
        adapter = session.save(path, specification)
    original_save, original_messages = session.save, session.messages_for

    def messages_for(prompt, context, latest):
        base = original_messages(prompt, context, '')
        if not latest:
            return base
        paths = sorted((root / 'events').glob('*.json'))
        action, tool, activity = [session.payload(p) for p in paths[-3:]]
        if (action['kind'] != 'action' or tool['kind'] != 'tool' or activity['kind'] != 'activity'
                or tool['text'] != latest or len({r['step'] for r in (action, tool, activity)}) != 1):
            raise ValueError('live tool pair is not the latest recorded exchange')
        return cycle.tool_cycle_messages(base, action['text'], readable_observation(latest))

    def bound_save(path, value):
        if path.name == 'request.json' and 'messages' in value and 'step' in value:
            value = dict(value, tool_adapter_sha256=read_adapter.sha256,
                         working_state_adapter_sha256=adapter.sha256)
        return original_save(path, value)

    session.Gateway = transport.RecoveringGateway
    session.SYSTEM = SYSTEM
    session.execute = reader.execute
    session.messages_for = messages_for
    session.save = bound_save
    session.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
