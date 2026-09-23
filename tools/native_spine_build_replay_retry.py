"""Use the existing bounded neutral-summary retry prompts for build replay merges."""
import argparse
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
import time

from tools import native_spine_build_replay_memory as memory


class RetryingQwen(memory.JournaledQwen):
    def __call__(self, request):
        try:
            return super().__call__(request)
        except ValueError:
            # Only a recorded response may trigger these two compression retries.
            original = memory.replay.load(self.root / 'qwen-summary-journal' /
                (memory.neutral_key(request) + '.response.json'))
        for attempt in (1, 2):
            messages = memory.neutral_messages(request, attempt=attempt)
            prefix = self.root / 'qwen-summary-retries' / memory.neutral_key(request) / str(attempt)
            saved = memory.replay.publish(prefix.with_suffix('.request.json'), {
                'messages': messages, 'request': asdict(request), 'raw_inputs': False,
                'model': 'qwen3-8b', 'max_tokens': 768, 'attempt': attempt,
                'original_response': memory.evaluation.binding(original),
                'implementation_sha256': memory.evaluation.digest(__file__)})
            if prefix.with_suffix('.response.json').exists():
                response = memory.replay.load(prefix.with_suffix('.response.json'))
            else:
                with prefix.with_suffix('.reserved').open('x', encoding='utf-8') as handle:
                    handle.write(saved.sha256 + '\n')
                started = time.perf_counter()
                with closing(memory.evaluation.authoring._completion_client(
                        'LITELLM_KEY', memory.evaluation.current.frozen.GATEWAY)) as client:
                    result = client.chat.completions.create(model='qwen3-8b', messages=messages,
                        max_tokens=768, temperature=0, extra_body={'enable_thinking': False}, timeout=180)
                choice, = result.choices
                response = memory.replay.publish(prefix.with_suffix('.response.json'), {
                    'request_sha256': saved.sha256, 'content': choice.message.content,
                    'finish_reason': choice.finish_reason, 'response_model': result.model,
                    'elapsed_s': time.perf_counter() - started})
            if response.payload['request_sha256'] != saved.sha256:
                raise ValueError('summary retry response binding changed')
            if response.payload['finish_reason'] != 'stop':
                continue
            try:
                return memory.parse_spine_summary(response.payload['content'], request)
            except ValueError:
                continue
        raise ValueError('Qwen exhausted both recorded short-summary retries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('exchanges', 'hierarchy'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True, choices=range(8))
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    if not args.enable_provider:
        parser.error('summary generation requires --enable-provider')
    memory.JournaledQwen = RetryingQwen
    {'exchanges': memory.exchanges, 'hierarchy': memory.hierarchy}[args.phase](args.root.resolve(), args.step)
