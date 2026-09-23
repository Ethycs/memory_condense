"""Resume exact evidence validation with bounded prefixes of oversized quotes."""
import argparse
import json
from pathlib import Path

from tools import native_spine_engineering_session as s
from tools import native_spine_engineering_live_session as live


original_repair = s.repair_raw_support


def repair_raw_support(content, fragments):
    repaired, changes = original_repair(content, fragments)
    value = json.loads(repaired)
    if not isinstance(value, dict) or not isinstance(value.get('atoms'), list):
        return repaired, changes
    if len(value['atoms']) != len(fragments):
        return repaired, changes
    for row, fragment in zip(value['atoms'], fragments, strict=True):
        if not isinstance(row, dict) or not isinstance(row.get('support'), list):
            continue
        before = list(row['support'])
        after = []
        for quote in before:
            bounded = quote
            if isinstance(quote, str) and quote in fragment.text and s.count_tokens(quote) > 32:
                # Character slicing preserves a literal Unicode source prefix.
                # Never locate new evidence, rewrite a summary or accept a paraphrase.
                while bounded and s.count_tokens(bounded) > 32:
                    bounded = bounded[:-1]
                if not bounded.strip():
                    bounded = quote  # Strict parsing must reject invalid support.
            after.append(bounded)
        if after != before:
            row['support'] = after
            changes.append({'label': row.get('label'), 'original_support': before,
                'exact_support': after, 'fragment_sha256': s.quote_sha256(fragment.text),
                'repair': 'bounded_prefix_of_model_selected_exact_quote'})
    return json.dumps(value, ensure_ascii=False), changes


def run(root):
    s.save(root / 'quote-bound-adapter.json', {
        'implementation_sha256': s.evaluation.digest(__file__),
        'live_session_implementation_sha256': s.evaluation.digest(live.__file__),
        'repair': 'Shorten only model-selected literal support quotations above 32 tokens.',
        'summary_text_changed': False, 'actor_context_policy_changed': False,
        'generated_candidate_changed_by_operator': False})
    s.repair_raw_support = repair_raw_support
    live.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
