"""Recover exhausted Qwen compression with stricter typed summary budgets."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

from tools import native_spine_engineering_dedup_resume as dedup

s = dedup.s
OriginalCompiler = s.SummaryCompiler


class RecoveringCompiler(OriginalCompiler):
    def merge(self, request):
        try:
            return super().merge(request)
        except ValueError as error:
            if str(error) != 'bounded Qwen compression retries exhausted':
                raise
        key = s.components.neutral_key(request)
        targets = dict.fromkeys((request.max_output_tokens * 3 // 4,
                                 request.max_output_tokens // 2))
        for target in targets:
            if not 1 <= target < request.max_output_tokens:
                continue
            tightened = replace(request, max_output_tokens=target)
            response = self.gateway.call(
                'qwen', s.components.neutral_messages(tightened, attempt=0),
                max_tokens=768, typed_request=asdict(tightened), attempt=0,
                nonce={'compression_recovery': key, 'target_tokens': target})
            try:
                text = s.parse_spine_summary(response['content'], tightened)
                s.parse_spine_summary(response['content'], request)
            except ValueError:
                continue
            receipt = s.save(self.root / 'compression-recoveries' / f'{key}.json', {
                'original_request': asdict(request), 'tightened_request': asdict(tightened),
                'response': response, 'summary': text, 'raw_inputs_to_qwen': False,
                'source_fragments_unchanged': True, 'both_output_budgets_validated': True})
            s.save(self.root / 'merged-summaries' / f'{key}.json', {
                'summary': text, 'request': asdict(request), 'raw_inputs_to_qwen': False,
                'compression_recovery_sha256': receipt.sha256})
            self.merges[key] = text
            return text
        raise ValueError('stricter Qwen compression recovery exhausted')


def run(root):
    s.save(root / 'compression-budget-adapter.json', {
        'implementation_sha256': s.evaluation.digest(__file__),
        'dedup_implementation_sha256': s.evaluation.digest(dedup.__file__),
        'maximum_additional_attempts': 2, 'requested_budget_fractions': [.75, .5],
        'source_fragments_unchanged': True, 'original_output_cap_relaxed': False,
        'stricter_output_cap_enforced': True, 'actor_context_policy_changed': False,
        'introduced_after_all_actor_prompts_completed': True})
    s.SummaryCompiler = RecoveringCompiler
    dedup.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
