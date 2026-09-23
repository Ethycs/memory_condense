"""Compile each identical pending raw fragment once, retaining every occurrence."""
import argparse
from pathlib import Path

from tools import native_spine_engineering_quote_resume as quote

s = quote.s
original_pack = s.components.raw_summary.pack_batches


def pack_unique_raw_batches(fragments, *, max_atoms=8, prompt_cap=7000):
    unique = {}
    for fragment in fragments:
        unique.setdefault(s.components.fragment_key(fragment), fragment)
    return original_pack(tuple(unique.values()), max_atoms=max_atoms, prompt_cap=prompt_cap)


def run(root):
    s.save(root / 'raw-dedup-adapter.json', {
        'implementation_sha256': s.evaluation.digest(__file__),
        'quote_adapter_implementation_sha256': s.evaluation.digest(quote.__file__),
        'repair': 'Compile each pending raw summary cache key once, using its first chronological occurrence.',
        'existing_summaries_replaced': False, 'all_source_occurrences_retained': True,
        'actor_context_policy_changed': False, 'generated_candidate_changed_by_operator': False})
    s.components.raw_summary.pack_batches = pack_unique_raw_batches
    quote.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
