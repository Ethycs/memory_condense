"""Read-only saved-engineering-memory dispatch check with actual CPU BGE."""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from memory_condense.application.chat_native import native_packet
from memory_condense.application.chat_session import ChatEvent
from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
from memory_condense.application.native_spine_policy import retrieval_limits
from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.application.tactic_router import BGETacticRouter
from memory_condense.persistence import native_spine_store, native_spine_parent_store
from memory_condense.runtime.artifacts import read,save,emit
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.embedding import FastEmbedBGE
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.evaluate_dialogue_neighborhood import digest,source_turns

ROOT=Path('eval_results/automatic-tactics-20261008-r1')
ORIGINAL=Path('eval_results/dialogue-neighborhood-20261006-r1')


def main():
    import argparse
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    ROOT=parser.parse_args().root
    ROOT.mkdir(parents=True,exist_ok=False)
    source=read(ORIGINAL/'source.json')
    directory=Path(source['directory'])
    for name,sha in source['files'].items():
        assert digest(directory/name)==sha
    turns=source_turns(directory)
    snapshot=native_spine_store.load(directory/'native-spine.sqlite',turns=turns)
    parents,_=native_spine_parent_store.load(directory/native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy,native_receipt=snapshot.receipt)
    embed=read(ORIGINAL/'embedding.json')
    assert digest(ORIGINAL/'episode-vectors.npy')==embed['episode_vectors_sha256']
    cases=[]
    for folder in ('dialogue-index-dump-20261006-r2','dialogue-index-extension-20261006-r2'):
        for q in read(Path('eval_results')/folder/'questions.json')['questions']:
            cases.append(dict(id=q['id'],query=q['question'],recent=[],expected='hierarchy_neighborhoods'))
    cases.extend([
        dict(id='fact',query='What port did we configure for the proxy?',recent=[],expected='cap8'),
        dict(id='ambiguous',query='Is this working?',recent=[],expected='cap8'),
        dict(id='local-followup',query='Is this working?',recent=['We are checking the return value of the single function shown in the previous message.'],expected='cap8'),
        dict(id='global-followup',query='Is this working?',recent=['We have spent months changing the architecture and measuring results. We need to assess whether the project now satisfies its original goals.'],expected='hierarchy_neighborhoods'),
    ])
    save(ROOT/'plan.json',dict(source=source,cases=cases,provider_calls=0,new_summaries=0,history_reingestions=0,
        implementation={str(p):digest(p) for p in [Path(__file__),*Path('src/memory_condense/application').glob('*tactic*.py'),
            Path('src/memory_condense/application/hierarchy_context.py'),Path('src/memory_condense/application/native_spine_retrieval.py')]}))
    started=time.perf_counter()
    encoder=FastEmbedBGE(assets=RuntimeAssets.resolve('.cache'))
    load_s=time.perf_counter()-started
    records=[]
    try:
        identity=summary_embedding_identity(encoder)
        assert identity==embed['identity']
        by_id={t.turn_id:t for t in turns}
        memory=ResidentNativeSpineMemory(snapshot.semantic,snapshot.hierarchy,encoder=encoder,load_turn=by_id.get)
        memory.router=NativeSpineUserCompletionRouter(snapshot.semantic,snapshot.hierarchy,parents)
        positions={t.turn_id:i for i,t in enumerate(turns)}
        memory.configure_dialogue(DialogueRetrievalPolicy(),np.load(ORIGINAL/'episode-vectors.npy'),
                                 embedding_identity=identity,turn_positions=positions)
        memory.configure_tactics(BGETacticRouter(encoder))
        def retrieve(query,dated,**kwargs):
            return memory.retrieve(query,dated,**retrieval_limits(memory,kwargs))
        view=SimpleNamespace(transcript=SimpleNamespace(get_turn=by_id.get),retrieve_native_spine=retrieve)
        events=[ChatEvent(t.turn_id,t.role,t.text,t.created_at.isoformat(),dict(source_id=t.source_id)) for t in turns]
        day=max(t.created_at for t in turns).strftime('%Y/%m/%d (%a)')
        for case in cases:
            started=time.perf_counter()
            packet=native_packet(view,case['query'],f'[Question asked at {day} 23:59] '+case['query'],events,recent=case['recent'])
            elapsed=time.perf_counter()-started
            trace=packet['retrieval']
            acceptable = ('hierarchy_neighborhoods','decision_history') if case['expected']=='hierarchy_neighborhoods' else (case['expected'],)
            assert trace['tactic']['selected_tactic'] in acceptable,case['id']
            assert packet['references']
            for ref in packet['references']:
                span=ref['span']; turn=by_id[span['turn_id']]
                assert turn.text[span['start_char']:span['end_char']] in packet['text']
            parity=None
            if case['id'].startswith('Q'):
                folder=ORIGINAL if int(case['id'][1:])<=15 else Path('eval_results/dialogue-index-extension-20261006-r2')
                old=read(folder/'packets'/case['id']/'closing_6k.json')
                parity=[r['span'] for r in packet['references']]==[r['span'] for r in old['spans']]
                assert parity,case['id']
                assert trace['hierarchy_context']['omitted_sections']==0
                refs=trace['hierarchy_context']['references']
                assert trace['hierarchy_context']['chronological']
                assert [r['first_turn_position'] for r in refs]==sorted(r['first_turn_position'] for r in refs)
                assert all(r['first_turn_position']==positions[r['first_turn_id']] for r in refs)
            save(ROOT/'packets'/(case['id']+'.json'),packet)
            summary=dict(id=case['id'],expected=case['expected'],selected=trace['tactic']['selected_tactic'],
                elapsed_s=elapsed,packet_tokens=trace['packet_token_count'],raw_spans=len(packet['references']),
                neighborhood_parity=parity,summary_sections=len(trace.get('hierarchy_context',{}).get('references',[])),
                summary_tokens=trace.get('hierarchy_context',{}).get('token_count',0))
            records.append(summary)
            emit(phase='checked',**summary)
    finally:
        encoder.close()
    assert all(digest(directory/name)==sha for name,sha in source['files'].items())
    save(ROOT/'report.json',dict(passed=True,model_load_s=load_s,cases=records,source_unchanged=True,
        provider_calls=0,history_reingestions=0,new_summaries=0,
        limits='Integration/packet parity on existing development memory, not a new answer-quality evaluation.'))


if __name__=='__main__':
    main()
