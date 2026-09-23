"""Preserve original summaries while recording reviewed exact-quote corrections."""
import json
from pathlib import Path

from tools import native_spine_build_replay as replay
from tools import native_spine_build_replay_memory as memory


CORRECTIONS = {
    (69, 0): '**407 tests passing**, all work staged.',
    (69, 1): '**nothing in the repo binds `LLMExtractor` to a provider.**',
    (173, 0): '**9.5% recall at 222 mean context tokens — 42.6 recall points per 1k**',
    (185, 1): '**3.5× fewer tokens** (673 vs 2,384)',
    (205, 1): '**10 new tests**, 522 passing.',
}


def run(root):
    _, rows, folder = replay.state(root, 0)
    lookup = {memory.fragment_key(f): f for f in memory.fragments_for(rows)}
    accepted, repairs = 0, []
    for path in sorted((root / 'raw-summary-journal').glob('*.response.json')):
        original = replay.load(path)
        request = replay.load(path.with_name(path.name.replace('.response.json', '.request.json')))
        if original.payload['request_sha256'] != request.sha256 or original.payload['finish_reason'] != 'stop':
            raise ValueError('incomplete or mismatched raw summary response')
        batch = [lookup[k] for k in request.payload['fragment_keys']]
        response = original
        try:
            values = memory.raw_summary.parse_summaries(original.payload['content'], batch)
        except ValueError:
            payload = json.loads(original.payload['content'])
            changes = []
            for row, fragment in zip(payload['atoms'], batch, strict=True):
                for i, quote in enumerate(row['support']):
                    if quote in fragment.text and memory.count_tokens(quote) <= 32:
                        continue
                    replacement = CORRECTIONS[(fragment.turn_ordinal, i)]
                    if replacement not in fragment.text or memory.count_tokens(replacement) > 32:
                        raise ValueError('reviewed quote is not an exact bounded source slice')
                    row['support'][i] = replacement
                    changes.append({'turn_ordinal': fragment.turn_ordinal, 'support_index': i,
                        'original': quote, 'corrected': replacement})
            content = json.dumps(payload, ensure_ascii=False)
            values = memory.raw_summary.parse_summaries(content, batch)
            response = replay.publish(root / 'raw-summary-repairs' / path.name, {
                **original.payload, 'content': content, 'original_response': memory.evaluation.binding(original),
                'reviewed_quote_corrections': changes, 'summary_text_changed': False,
                'new_model_calls': 0, 'repair_implementation_sha256': memory.evaluation.digest(__file__)})
            repairs.append(memory.evaluation.binding(response))
        for fragment, value in zip(batch, values, strict=True):
            replay.publish(root / 'atomic-summaries' / f'{memory.fragment_key(fragment)}.json', {
                'key': memory.fragment_key(fragment), 'summary': value['summary'], 'support': value['support'],
                'response': memory.evaluation.binding(response), 'role': fragment.role,
                'span_text_sha256': memory.quote_sha256(fragment.text),
                'turn_text_sha256': fragment.turn_text_sha256})
            accepted += 1
    if accepted != len(lookup):
        raise ValueError('not every planned fragment has an accepted summary')
    memory.atoms_for(root, rows)
    replay.publish(folder / 'raw-summary-completion.json', {'accepted_fragments': accepted,
        'repairs': repairs, 'summary_text_changed': False, 'new_model_calls': 0})
    print(json.dumps({'accepted_fragments': accepted, 'repaired_batches': len(repairs), 'new_model_calls': 0}))


if __name__ == '__main__':
    run(Path('eval_results/native-spine-build-replay-20260916-r1').resolve())
