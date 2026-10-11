"""Development follow-up: preserve closing responses in the same neighborhoods."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import time

import numpy as np

from memory_condense.application.native_spine_policy import CAP8_LIMITS
from memory_condense.application.section_retrieval import hydrate_section_plan
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.persistence import native_spine_store,native_spine_parent_store
from memory_condense.runtime.artifacts import read,save,emit
from memory_condense.search.dialogue_neighborhood import DialogueNeighborhoodRouter
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
from tools.evaluate_dialogue_neighborhood import DEFAULT_ROOT,ACTOR_SYSTEM,source_turns,digest,api_job

ARM='closing_6k'


def prepare(root):
    source=read(root/'source.json'); directory=Path(source['directory'])
    for name,sha in source['files'].items():
        if digest(directory/name)!=sha: raise ValueError('Source archive changed')
    turns=source_turns(directory); by_id={t.turn_id:t for t in turns}
    ordinal={t.turn_id:i for i,t in enumerate(turns)}
    snapshot=native_spine_store.load(directory/'native-spine.sqlite',turns=turns)
    parents,_=native_spine_parent_store.load(directory/native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy,native_receipt=snapshot.receipt)
    native=NativeSpineUserCompletionRouter(snapshot.semantic,snapshot.hierarchy,parents)
    embeddings=read(root/'embedding.json')
    for name,key in [('episode-vectors.npy','episode_vectors_sha256'),('query-vectors.npy','query_vectors_sha256')]:
        if digest(root/name)!=embeddings[key]: raise ValueError('Cached summary/query vectors changed')
    survey=DialogueNeighborhoodRouter(native,np.load(root/'episode-vectors.npy'),embedding_identity=embeddings['identity'])
    vectors=np.load(root/'query-vectors.npy')
    options={k:v for k,v in CAP8_LIMITS.items() if k not in ('max_context_tokens','max_raw_spans')}
    asked=max(t.created_at for t in turns).strftime('%Y/%m/%d (%a)')
    questions=read(root/'questions.json')['questions']
    for q,vector in zip(questions,vectors,strict=True):
        query=q['question']; started=time.perf_counter()
        base=native.route_vector(query,f'[Question asked at {asked} 23:59] '+query,vector,
            embedding_identity=snapshot.semantic.embedding_identity,**options)
        plan,coverage=survey.route(base,vector,embedding_identity=embeddings['identity'],closing_responses=True)
        prior=read(root/'packets'/q['id']/'neighborhood_6k.json')
        assert coverage['selected_episode_ids']==prior['coverage']['selected_episode_ids']
        routing_s=time.perf_counter()-started; started=time.perf_counter()
        result=hydrate_section_plan(plan,load_turn=by_id.get,max_raw_spans=128,max_context_tokens=6144)
        pieces=[]; spans=[]
        for section in result.sections:
            for evidence in section.evidence:
                span=evidence.span; label='T'+str(ordinal[span.turn_id])
                pieces.append(f'[{label} role={span.role} chars={span.start_char}:{span.end_char}]\n'+evidence.text)
                spans.append(dict(label=label,span=span.identity_payload(),text=evidence.text))
        text='\n\n'.join(pieces)
        save(root/'packets'/q['id']/(ARM+'.json'),dict(arm=ARM,question=query,text=text,spans=spans,
            budget=6144,rendered_tokens=result.context_token_count,sent_evidence_tokens=count_tokens(text),
            routing_s=routing_s,hydration_s=time.perf_counter()-started,coverage=coverage,
            route_sha256=plan.receipt_sha256,hydration_sha256=result.receipt_sha256,
            diagnostics=dict(Counter(d.reason for d in result.diagnostics))))
    save(root/'closing-plan.json',dict(actor_calls=15,judge_calls=15,original_reference_sha256=digest(root/'private-reference.json'),
        original_questions_sha256=digest(root/'questions.json'),unchanged_selected_neighborhoods=True,
        new_embeddings=0,history_reingestions=0,reason='Initial packets retrieved activity but missed the same exchanges closing answers.',
        adaptive_development=True,implementation={p:digest(p) for p in (__file__,
            'src/memory_condense/search/dialogue_neighborhood.py')},system=ACTOR_SYSTEM))
    emit(phase='closing_packets_prepared',questions=len(questions),new_embeddings=0)


def generate(root):
    plan=read(root/'closing-plan.json')
    for p,sha in plan['implementation'].items():
        if digest(p)!=sha: raise ValueError('Closing-policy implementation changed')
    if digest(root/'private-reference.json')!=plan['original_reference_sha256']:
        raise ValueError('Reference changed during development follow-up')
    jobs=[]
    for q in read(root/'questions.json')['questions']:
        packet=read(root/'packets'/q['id']/(ARM+'.json'))
        messages=[dict(role='system',content=ACTOR_SYSTEM),dict(role='user',
            content='Archived evidence:\n'+packet['text']+'\n\nQuestion: '+q['question'])]
        jobs.append((q['id'],messages))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(api_job,root,'answer-'+qid+'-'+ARM,messages,max_tokens=1000):qid for qid,messages in jobs}
        for future in as_completed(futures):
            result=future.result()
            emit(phase='closing_answered',question=futures[future],elapsed_s=result['elapsed_s'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','generate'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    if args.phase=='prepare': prepare(args.root)
    else: generate(args.root)
