"""Reuse bounded singleton summaries; reserve Qwen generation for actual merges.

The raw compiler already conditions response summaries on the owning user lead.
Rewriting those singleton summaries adds work and can lose their details. The
new policy reuses them exactly, allows 128 tokens per channel, and uses Qwen for
multi-summary merges with one explicit compaction recovery if needed.
"""

import argparse
import json
from pathlib import Path
import time

from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.integrity import file_sha256
from memory_condense.modeling.qwen_prefix import Qwen3PrefixEncoder
from memory_condense.search.episodes.qwen_episode_signal import QwenAttentionHeadSurpriseScorer
from memory_condense.search.episodes.user_spine_hierarchy import build_user_spine_hierarchy, compile_user_spine_exchanges
from memory_condense.search.section_summary import SectionSummary
from memory_condense.search.spine_summary import parse_spine_summary
from tools.assay_user_spine_hierarchy import Journal, REPO, implementation
from tools.build_user_spine_pilot_hierarchy import compaction_messages
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


class MergeJournal(Journal):
    def __init__(self, root, preflight, enable_provider, policy_sha):
        super().__init__(root, preflight, enable_provider)
        self.policy_sha=policy_sha
        self.reused=self.compactions=0

    def summarize(self, request):
        if len(request.fragments)==1 and count_tokens(request.fragments[0].summary)<=request.max_output_tokens:
            self.reused+=1
            return request.fragments[0].summary
        response=self.complete_messages(request.messages,phase='summarize',cap=request.max_prompt_tokens)
        try:return parse_spine_summary(response,request)
        except ValueError as exc:
            if 'exceeds its output budget' not in str(exc):raise
        self.compactions+=1
        compact=self.complete_messages(compaction_messages(response),phase='summary_compaction_'+self.policy_sha,cap=request.max_prompt_tokens)
        return parse_spine_summary(compact,request)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--enable-provider',action='store_true')
    args=parser.parse_args()
    root=args.output_root
    parent=read_sealed_json(root/'preflight.json')
    atoms=read_sealed_json(root/'atoms.json')
    if parent.payload['implementation']!=implementation() or atoms.payload['preflight_sha256']!=parent.sha256:
        raise ValueError('source identity changed')
    policy,_=publish_sealed_json(root/'hierarchy-build-preflight-r3.json',{
        'parent_preflight_sha256':parent.sha256,'atoms_sha256':atoms.sha256,'implementation_sha256':file_sha256(Path(__file__)),
        'compaction_implementation_sha256':file_sha256(Path('tools/build_user_spine_pilot_hierarchy.py')),
        'reuse_bounded_singletons':True,'max_channel_tokens':128,'max_compaction_calls_per_merge':1,
        'attention_span_tokens':128,'attention_workspace_tokens':4096,'raw_inputs_to_qwen':False,'query_independent':True})
    journal=MergeJournal(root,parent,args.enable_provider,policy.sha256)
    started=time.perf_counter()
    exchanges=compile_user_spine_exchanges(tuple(SectionSummary.from_dict(a) for a in atoms.payload['atoms']),
        summarize=journal.summarize,summarizer_identity=policy.sha256,max_channel_tokens=128)
    print('Loading local Qwen attention over user summaries...',flush=True)
    encoder=Qwen3PrefixEncoder(REPO/'.cache/models/Qwen3-8B',layers=6,device='cuda',dtype='float16')
    linker=QwenMemoryLinker(encoder,layer=5,max_candidates=8,max_workspace_tokens=4096)
    hierarchy=build_user_spine_hierarchy(exchanges,scorer=QwenAttentionHeadSurpriseScorer(linker,max_spans=8,span_token_cap=128),
        summarize=journal.summarize,summarizer_identity=policy.sha256,max_channel_tokens=128,max_leaf_exchanges=2,window_exchange_cap=8)
    result,_=publish_sealed_json(root/'hierarchy-r3.json',{'preflight_sha256':policy.sha256,'atoms_sha256':atoms.sha256,
        'hierarchy':hierarchy.identity_payload(),'index_json':hierarchy.summary_index().to_json(),
        'qwen_request_artifact_shas':journal.requests,'compactions':journal.compactions,'reused_singleton_summaries':journal.reused,'raw_qwen_inputs':0})
    print(json.dumps({'hierarchy_sha256':result.sha256,'new_provider_calls':journal.calls,'checkpoint_hits':journal.hits,
        'elapsed_seconds':time.perf_counter()-started,'exchanges':len(exchanges),'sections':len(hierarchy.sections),
        'oversized_exchanges':len(hierarchy.oversized_exchange_ids),'reused_singletons':journal.reused,'compactions':journal.compactions},indent=2),flush=True)


if __name__=='__main__':main()
