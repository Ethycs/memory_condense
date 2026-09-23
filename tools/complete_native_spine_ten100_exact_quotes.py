"""Apply explicitly recorded source-exact citation repairs before answering."""
import argparse
from pathlib import Path

from tools import complete_native_spine_ten100_questions as completion


def complete(root, batch, repairs):
    c = completion.campaign
    _, scope, folder = c.history(root, batch)
    artifact = c.read_sealed_json(repairs)
    if artifact.payload['scope'] != c.binding(scope):
        raise ValueError('citation repairs belong to another source scope')
    if list((folder / 'answers').glob('*.response.json')):
        raise ValueError('citation repairs must precede candidate answers')
    original = completion.exact_supports

    def exact(value, source):
        value = {**value, 'supports': [dict(s) for s in value['supports']]}
        changes = []
        for repair in artifact.payload['repairs']:
            if repair['body_sha256'] != source['body_sha256']:
                continue
            matches = [s for s in value['supports'] if s['turn_index'] == repair['turn_index']
                       and s['quote'] == repair['original_quote']]
            text = next(t['text'] for t in source['user_turns'] if t['turn_index'] == repair['turn_index'])
            if len(matches) != 1 or text.count(repair['exact_quote']) != 1:
                raise ValueError('citation repair requires unique original and exact source matches')
            matches[0]['quote'] = repair['exact_quote']
            changes.append({**repair, 'repair_artifact': c.binding(artifact),
                'repair_implementation_sha256': c.digest(__file__),
                'rule': 'explicit source-exact citation repair; question and reference unchanged'})
        validated, case_changes = original(value, source)
        return validated, changes + case_changes

    completion.exact_supports = exact
    try:
        completion.complete(root, batch)
    finally:
        completion.exact_supports = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=range(1, 11), required=True)
    parser.add_argument('--repairs', type=Path, required=True)
    args = parser.parse_args()
    complete(args.root, args.batch, args.repairs)
