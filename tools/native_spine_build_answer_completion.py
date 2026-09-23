"""Generation-only replay worker with full tool history and a 128-action ceiling."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import time

from tools import native_spine_build_replay as replay


ACTION_CAP = 128


def action_number(folder):
    for i in range(ACTION_CAP):
        if not (folder / 'actions' / f'{i:03d}.response.json').exists():
            return i
    raise ValueError('extended per-prompt action limit reached')


def answer(root, step):
    plan, rows, folder = replay.state(root, step)
    if (folder / 'complete.json').exists():
        raise ValueError('this original user prompt is already complete')
    adapter = replay.publish(root / 'completion-tool-context-adapter.json', {
        'plan_sha256': plan.sha256, 'implementation_sha256': replay.evaluation.digest(__file__),
        'policy': 'retain every recorded tool result from this current user turn',
        'action_limit': ACTION_CAP,
        'previous_adapter': replay.evaluation.binding(replay.load(root / 'extended-tool-context-adapter.json')),
        'original_user_prompts_changed': False, 'original_future_assistant_responses_used': False,
        'qa_campaign_changed': False})
    context = replay.load(folder / 'memory-context.json')
    if context.payload['history_sha256'] != replay.identity_sha256(rows):
        raise ValueError('build memory contains a different chronological prefix')
    ordinal = action_number(folder)
    content = 'Retrieved earlier conversation evidence:\n' + context.payload['text']
    recent = rows[-2:] if step else rows[-1:]
    content += '\n\nLatest live conversation messages:\n' + json.dumps(
        [{'role': r['role'], 'content': r['text']} for r in recent], ensure_ascii=False)
    content += '\n\nCurrent user prompt (respond to this):\n' + plan.payload['prompts'][step]['text']
    messages = [{'role': 'system', 'content': replay.SYSTEM.replace('36 actions', '128 actions')}, {'role': 'user', 'content': content}]
    for previous in range(ordinal):
        response = replay.load(folder / 'actions' / f'{previous:03d}.response.json')
        result = replay.load(folder / 'actions' / f'{previous:03d}.tool.json')
        messages.extend([{'role': 'assistant', 'content': response.payload['content']},
                         {'role': 'user', 'content': 'Tool result:\n' + result.payload['result']}])
    prefix = folder / 'actions' / f'{ordinal:03d}'
    request = replay.publish(prefix.with_suffix('.request.json'), {'plan_sha256': plan.sha256,
        'step': step, 'action_index': ordinal, 'context': replay.evaluation.binding(context),
        'answer_adapter': replay.evaluation.binding(adapter), 'model': replay.evaluation.MODEL,
        'messages': messages, 'max_tokens': 8192,
        'prompt_token_proxy': replay.count_chat_prompt_token_proxy(messages)})
    with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
        handle.write(request.sha256 + '\n')
    started = time.perf_counter()
    with closing(replay.evaluation.authoring._completion_client(
            'LITELLM_KEY', replay.evaluation.current.frozen.GATEWAY)) as client:
        result = client.chat.completions.create(model=replay.evaluation.MODEL, messages=messages,
            max_tokens=8192, temperature=0, timeout=240)
    choice, = result.choices
    text = choice.message.content or ''
    replay.publish(prefix.with_suffix('.response.json'), {'request_sha256': request.sha256,
        'content': text, 'finish_reason': choice.finish_reason, 'response_model': result.model,
        'usage': result.usage.model_dump() if result.usage else None,
        'elapsed_s': time.perf_counter() - started})
    if choice.finish_reason != 'stop':
        raise ValueError('build action did not stop normally')
    try:
        value = json.loads(text)
        replay.evaluation.emit(phase='build_action_ready', step=step, action_index=ordinal,
            action=value.get('action'), path=value.get('path'))
    except json.JSONDecodeError:
        replay.evaluation.emit(phase='build_recorded_non_json_reply', step=step, action_index=ordinal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if not args.enable_provider:
        parser.error('generation requires --enable-provider')
    root = args.root.resolve()
    replay.evaluation.emit(phase='build_answer_worker_ready', accepted_input='step 0..7 or quit')
    for line in sys.stdin:
        command = line.strip()
        if command == 'quit':
            break
        if command not in tuple(str(i) for i in range(8)):
            raise ValueError('worker accepts only an original replay step number')
        answer(root, int(command))
        replay.evaluation.emit(phase='build_answer_worker_waiting', step=int(command))

