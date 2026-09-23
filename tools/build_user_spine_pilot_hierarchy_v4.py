"""Finish the real pilot with an explicit, audited attached-context fallback.

Reuse the r3 merge/compaction requests. If a completed compaction still exceeds
the cap, retain its longest bounded sentence prefix and mark the omission.
This lossy fallback applies only to attached context, never the user spine or
hydrated raw evidence. A single oversized sentence fails closed.
"""

import argparse
import json
from pathlib import Path
import re
import time

from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.integrity import file_sha256
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.user_spine_hierarchy import build_user_spine_hierarchy, compile_user_spine_exchanges
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_summary import parse_spine_summary
from tools.assay_user_spine_hierarchy import REPO, implementation
from tools.build_user_spine_pilot_hierarchy_v3 import MergeJournal
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


OMISSION = ' [Later attached-summary sentences omitted.]'


def bounded_attached_prefix(response, request):
    try:
        return parse_spine_summary(response, request)
    except ValueError as exc:
        if 'exceeds its output budget' not in str(exc):
            raise
    if request.kind != 'attached_context':
        raise ValueError('sentence omission is forbidden for the user spine')
    summary = json.loads(response)['summary']
    ends = [match.start() for match in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])', summary)]
    candidates = [summary[:end] + OMISSION for end in ends]
    candidates = [s for s in candidates if count_tokens(s) <= request.max_output_tokens]
    if not candidates:
        raise ValueError('no complete attached-summary sentence fits the budget')
    return parse_spine_summary(json.dumps({'summary': candidates[-1]}), request)


class FinalJournal(MergeJournal):
    def __init__(self, *args):
        super().__init__(*args)
        self.omissions = []

    def complete_messages(self, *args, **kwargs):
        self.last_completion = super().complete_messages(*args, **kwargs)
        return self.last_completion

    def summarize(self, request):
        try:
            return super().summarize(request)
        except ValueError as exc:
            if 'exceeds its output budget' not in str(exc):
                raise
        summary = bounded_attached_prefix(self.last_completion, request)
        self.omissions.append({'request_sha256': request.prompt_sha256,
            'response_sha256': quote_sha256(self.last_completion), 'retained_summary': summary,
            'original_summary_tokens': count_tokens(json.loads(self.last_completion)['summary']),
            'retained_summary_tokens': count_tokens(summary), 'kind': request.kind})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--enable-provider', action='store_true')
    args = parser.parse_args()
    root = args.output_root
    parent = read_sealed_json(root/'preflight.json')
    atoms = read_sealed_json(root/'atoms.json')
    previous = read_sealed_json(root/'hierarchy-build-preflight-r3.json')
    if parent.payload['implementation'] != implementation() or atoms.payload['preflight_sha256'] != parent.sha256:
        raise ValueError('source identity changed')
    if previous.payload['implementation_sha256'] != file_sha256(Path('tools/build_user_spine_pilot_hierarchy_v3.py')):
        raise ValueError('previous merge implementation changed')
    policy, _ = publish_sealed_json(root/'hierarchy-build-preflight-r4.json', {
        'parent_preflight_sha256': parent.sha256, 'atoms_sha256': atoms.sha256,
        'previous_policy_sha256': previous.sha256, 'implementation_sha256': file_sha256(Path(__file__)),
        'max_channel_tokens': 128, 'attached_context_overflow': 'complete sentence prefix with omission marker',
        'user_spine_omission_permitted': False, 'raw_inputs_to_qwen': False, 'query_independent': True})
    journal = FinalJournal(root, parent, args.enable_provider, previous.sha256)
    started = time.perf_counter()
    exchanges = compile_user_spine_exchanges(tuple(SectionSummary.from_dict(a) for a in atoms.payload['atoms']),
        summarize=journal.summarize, summarizer_identity=policy.sha256, max_channel_tokens=128)
    print('Loading local Qwen attention over user summaries...', flush=True)
    encoder = Qwen3PrefixEncoder(REPO/'.cache/models/Qwen3-8B', layers=6, device='cuda', dtype='float16')
    linker = QwenMemoryLinker(encoder, layer=5, max_candidates=8, max_workspace_tokens=4096)
    hierarchy = build_user_spine_hierarchy(exchanges,
        scorer=QwenAttentionHeadSurpriseScorer(linker, max_spans=8, span_token_cap=128),
        summarize=journal.summarize, summarizer_identity=policy.sha256, max_channel_tokens=128,
        max_leaf_exchanges=2, window_exchange_cap=8)
    result, _ = publish_sealed_json(root/'hierarchy-r4.json', {
        'preflight_sha256': policy.sha256, 'atoms_sha256': atoms.sha256,
        'hierarchy': hierarchy.identity_payload(), 'index_json': hierarchy.summary_index().to_json(),
        'qwen_request_artifact_shas': journal.requests, 'compactions': journal.compactions,
        'reused_singleton_summaries': journal.reused, 'attached_summary_omissions': journal.omissions, 'raw_qwen_inputs': 0})
    print(json.dumps({'hierarchy_sha256': result.sha256, 'new_provider_calls': journal.calls,
        'checkpoint_hits': journal.hits, 'elapsed_seconds': time.perf_counter()-started,
        'exchanges': len(exchanges), 'sections': len(hierarchy.sections), 'attention_windows': len(hierarchy.windows),
        'oversized_exchanges': len(hierarchy.oversized_exchange_ids), 'omissions': len(journal.omissions)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
