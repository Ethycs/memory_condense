"""Incremental matched assay for query-free precomputed CAV reinjection.

Reuses the frozen 2026-10-08 baseline packets and answers.  Only the new arm is
prepared/generated; each question is then judged blindly against its existing
baseline answer and the same source-bound checkpoints.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

import numpy as np

from memory_condense.application.chat_native import native_packet
from memory_condense.application.chat_session import ChatEvent
from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
from memory_condense.application.native_experiments import NativeExperimentPolicy,PreparedNativeExperiments
from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.application.native_spine_policy import retrieval_limits
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.persistence import native_spine_store,native_spine_parent_store
from memory_condense.runtime.artifacts import read,save,emit
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.characterize_native_experiments import (READER,cases,covered_quote,
    messages,job_id,validate_grades)
from tools.evaluate_dialogue_index_dump import api_job,bounded
from tools.evaluate_dialogue_neighborhood import digest,source_turns
from tools.grade_dialogue_neighborhood import JUDGE,extract_json


ORIGINAL=Path('eval_results/dialogue-neighborhood-20261006-r1')
CONTROL=Path('eval_results/dialogue-index-dump-20261006-r2')
PRIOR=Path('eval_results/native-experiment-quality-20261008-r1')
CAV=Path('eval_results/native-experiments-20261008-r1/cav/bundle.json')
FEATURE_CACHE=Path('eval_results/cav-reinjection-feature-cache-v1')
DEFAULT_ARM='cav_reinjection'


def implementation():
    files=[Path(__file__),Path('tools/characterize_native_experiments.py')]
    files += [Path('src/memory_condense')/name for name in (
        'application/chat_native.py','application/native_experiments.py',
        'application/native_spine_retrieval.py','associations/cav_reinjection.py',
        'associations/spectral_reinjection.py',
        'runtime/local.py')]
    return {str(path):digest(path) for path in files}


def packet_facts(packet,facts):
    return {fid:dict(complete=all(covered_quote(s,packet['references']) for s in fact['supports']),
        quotes=sum(covered_quote(s,packet['references']) for s in fact['supports']),
        total_quotes=len(fact['supports'])) for fid,fact in facts.items()}


def prepare(root, cav_bundle=CAV, arm=DEFAULT_ARM):
    root.mkdir(parents=True,exist_ok=False)
    source=read(ORIGINAL/'source.json'); directory=Path(source['directory'])
    if any(digest(directory/name)!=sha for name,sha in source['files'].items()):
        raise ValueError('Archived source changed')
    turns=source_turns(directory); by_id={turn.turn_id:turn for turn in turns}
    reference=read(PRIOR/'private-reference.json'); questions=cases()
    prior_plan=read(PRIOR/'plan.json'); prior_jobs=read(PRIOR/'reader-jobs.json')
    snapshot=native_spine_store.load(directory/'native-spine.sqlite',turns=turns)
    parents,_=native_spine_parent_store.load(directory/native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy,native_receipt=snapshot.receipt)
    runtime=LocalRuntime(root/'runtime',assets=RuntimeAssets.resolve('.cache'))
    records,jobs,bindings=[],{},{}
    try:
        encoder=runtime.embedding()
        memory=ResidentNativeSpineMemory(snapshot.semantic,snapshot.hierarchy,encoder=encoder,load_turn=by_id.get)
        memory.router=NativeSpineUserCompletionRouter(snapshot.semantic,snapshot.hierarchy,parents)
        identity=summary_embedding_identity(encoder); embedding=read(ORIGINAL/'embedding.json')
        if identity!=embedding['identity'] or digest(ORIGINAL/'episode-vectors.npy')!=embedding['episode_vectors_sha256']:
            raise ValueError('Dialogue vectors differ from the resident BGE identity')
        memory.configure_dialogue(DialogueRetrievalPolicy(),np.load(ORIGINAL/'episode-vectors.npy'),
            embedding_identity=identity,turn_positions={turn.turn_id:i for i,turn in enumerate(turns)})
        memory.configure_tactics(runtime.tactic_router())
        if arm=='cav_reinjection':
            policy=NativeExperimentPolicy(modules=(arm,),cav_bundle=str(Path(cav_bundle).resolve()))
            attention=runtime.attention(FEATURE_CACHE)
        elif arm=='spectral_reinjection':
            policy=NativeExperimentPolicy(modules=(arm,))
            attention=None
        else:
            raise ValueError('Unknown reinjection arm: '+arm)
        build_start=time.perf_counter()
        if arm=='cav_reinjection':
            state=attention.prepare_cav_reinjection(tuple(memory.router.atoms.values()),policy)
        else:
            from memory_condense.associations.spectral_reinjection import SpectralReinjectionIndex
            sections=memory.router.semantic.sections
            state=SpectralReinjectionIndex.build([s.section_id for s in sections],
                [s.source_id for s in sections],memory.router.semantic.address_matrix,
                degree=8,restart=.7,hops=2,direct_mix=.7,
                provenance=dict(embedding_identity=memory.router.semantic.embedding_identity))
        build_s=time.perf_counter()-build_start
        save(root/'state.json',dict(state.receipt,prepare_s=build_s))
        events=tuple(ChatEvent(t.turn_id,t.role,t.text,t.created_at.isoformat(),dict(source_id=t.source_id)) for t in turns)
        day=max(t.created_at for t in turns).strftime('%Y/%m/%d (%a)')
        def retrieve(query,dated,**kwargs):
            return memory.retrieve(query,dated,**retrieval_limits(memory,kwargs))
        view=SimpleNamespace(transcript=SimpleNamespace(get_turn=by_id.get),retrieve_native_spine=retrieve,
                             decision_revisions=())
        for question in questions:
            qid=question['id']; facts={fid:reference['facts'][fid] for fid in question['facts']}
            memory.experiments=None
            baseline_started=time.perf_counter()
            current=native_packet(view,question['question'],f'[Question asked at {day} 23:59] '+question['question'],events)
            baseline_s=time.perf_counter()-baseline_started
            prior=read(PRIOR/'packets'/qid/'baseline.json')
            # Receipts gained newer continuity/decision metadata after the prior
            # assay. Reuse its answer only when the reader-visible packet and
            # exact source pointers remain byte-for-byte equivalent.
            if current['text']!=prior['text'] or current['references']!=prior['references']:
                raise ValueError('Current baseline evidence differs from frozen prior packet: '+qid)
            baseline_key=prior_jobs['bindings'][qid]['baseline']
            if job_id(qid,messages(question['question'],prior))!=baseline_key:
                raise ValueError('Frozen baseline reader input changed')
            memory.experiments=PreparedNativeExperiments(memory.router,policy,attention=attention,
                reinjection=state if arm=='cav_reinjection' else None,
                spectral=state if arm=='spectral_reinjection' else None)
            started=time.perf_counter()
            packet=native_packet(view,question['question'],f'[Question asked at {day} 23:59] '+question['question'],events)
            elapsed=time.perf_counter()-started
            if packet['routing']!=current['routing'] or packet['hydration']!=current['hydration']:
                raise ValueError('CAV reinjection altered baseline evidence')
            exp=packet['retrieval']['experiments']; arm_receipt=exp['arms'][0]
            if arm_receipt['model_passes']!=0 or exp['query_qwen_passes']!=0:
                raise ValueError('CAV reinjection executed Qwen at query time')
            save(root/'packets'/qid/(arm+'.json'),packet)
            prompt=messages(question['question'],packet); treatment_key=job_id(qid,prompt)
            bindings[qid]={'baseline':baseline_key,arm:treatment_key}
            if treatment_key!=baseline_key:
                jobs.setdefault(treatment_key,dict(key=treatment_key,qid=qid,messages=prompt))
            records.append(dict(id=qid,category=question['category'],recall_s=elapsed,
                baseline_recall_s=baseline_s,incremental_recall_s=elapsed-baseline_s,
                packet_tokens=count_tokens(packet['text']),raw_spans=len(packet['references']),
                added_sections=len(exp.get('delivered_section_ids',[])),query_qwen_passes=exp['query_qwen_passes'],
                model_passes=arm_receipt['model_passes'],
                query_compute_cells=arm_receipt.get('query_compute_cells',0),
                query_compute_s=arm_receipt.get('query_compute_s',0),
                dense_attention_cells=arm_receipt.get('dense_attention_cells',0),
                compute_cell_ratio=arm_receipt.get('compute_cell_ratio'),status=arm_receipt['status'],
                support_quotes=packet_facts(packet,facts)))
            emit(phase='packet',id=qid,status=arm_receipt['status'],recall_s=elapsed,
                 additions=len(exp.get('delivered_section_ids',[])))
    finally:
        runtime.close()
    save(root/'questions.json',dict(questions=questions))
    save(root/'private-reference.json',reference)
    save(root/'retrieval-report.json',dict(records=records))
    save(root/'reader-jobs.json',dict(jobs=list(jobs.values()),bindings=bindings))
    plan=dict(model=prior_plan['model'],gateway=prior_plan['gateway'],reader_system=READER,
        max_output_tokens=prior_plan['max_output_tokens'],judge_output_tokens=prior_plan['judge_output_tokens'],
        prompt_limit=prior_plan['prompt_limit'],judge_prompt_limit=prior_plan['judge_prompt_limit'],
        timeout_s=prior_plan['timeout_s'],automatic_retries=0,max_concurrency=2,reasoning_effort='none',
        questions=questions,arms=('baseline',arm),experimental_arm=arm,
        actor_calls=len(jobs),judge_calls=len(questions),
        history_reingestions=0,source=source,implementation=implementation(),
        prior_root=str(PRIOR),prior_plan_sha256=digest(PRIOR/'plan.json'),
        reader_prompt_tokens=sum(sum(count_tokens(m['content']) for m in job['messages']) for job in jobs.values()),
        limitations=['One development engineering archive and one experimental retrieval arm.',
            'Publication-time state construction and warm query cost are reported separately.',
            'Baseline answers are reused byte-for-byte from the sealed 2026-10-08 assay.',
            'Same Sol reader and blinded Sol judge; single generation per distinct new input.'])
    save(root/'plan.json',plan)
    emit(phase='prepared',questions=len(questions),reader_calls=len(jobs),build_s=build_s,
         qwen_batches=state.receipt.get('build_qwen_batches',0),state_bytes=state.receipt['retained_state_bytes'])


def validate(root):
    plan=read(root/'plan.json')
    if implementation()!=plan['implementation']:
        raise ValueError('Frozen implementation changed')
    if digest(Path(plan['prior_root'])/'plan.json')!=plan['prior_plan_sha256']:
        raise ValueError('Prior assay changed')
    return plan


def generate(root):
    plan=validate(root); jobs=read(root/'reader-jobs.json')['jobs']
    def run(job):
        result=api_job(root,job['key'],job['messages'],plan=plan)
        if not result['content'].strip():
            raise ValueError('Reader returned an empty answer')
        emit(phase='answer',id=job['qid'],key=job['key'],usage=result.get('usage'))
    bounded(jobs,run,workers=plan['max_concurrency'])


def response(root,key,baseline_key):
    base=PRIOR if key==baseline_key else root
    return read(base/'api'/key/'response.json')


def grade(root):
    plan=validate(root); manifest=read(root/'reader-jobs.json'); reference=read(root/'private-reference.json')
    for question in plan['questions']:
        qid=question['id']; binding=manifest['bindings'][qid]; baseline_key=binding['baseline']
        keys=sorted(set(binding.values()),key=lambda key:hashlib.sha256(('cav-reinject-v1:'+qid+key).encode()).hexdigest())
        aliases={chr(65+i):key for i,key in enumerate(keys)}
        answers={label:response(root,key,baseline_key)['content'] for label,key in aliases.items()}
        facts={fid:reference['facts'][fid] for fid in question['facts']}
        system=JUDGE.replace('three blinded answers',str(len(keys))+' blinded answers')
        judged=validate_grades(extract_json(api_job(root,'judge-'+qid,[dict(role='system',content=system),
            dict(role='user',content=json.dumps(dict(question=question['question'],checkpoints=facts,answers=answers),
                                               ensure_ascii=False))],plan=plan,judge=True)['content']),aliases,facts)
        by_key={aliases[label]:value for label,value in judged.items()}
        save(root/'grades'/(qid+'.json'),dict(arms={arm:by_key[key] for arm,key in binding.items()},aliases=aliases))
        emit(phase='graded',id=qid,points={arm:by_key[key]['points'] for arm,key in binding.items()})


def report(root):
    plan=validate(root); rows=read(root/'retrieval-report.json')['records']
    arms=tuple(plan['arms'])
    grades={q['id']:read(root/'grades'/(q['id']+'.json'))['arms'] for q in plan['questions']}
    totals={}
    for arm in arms:
        scores=[grades[q['id']][arm] for q in plan['questions']]
        baseline=[grades[q['id']]['baseline'] for q in plan['questions']]
        deltas=[a['points']-b['points'] for a,b in zip(scores,baseline)]
        totals[arm]=dict(points=sum(s['points'] for s in scores),possible=sum(s['possible'] for s in scores),
            macro_checkpoint_pct=100*mean(s['points']/s['possible'] for s in scores),
            wins=sum(d>0 for d in deltas),ties=sum(d==0 for d in deltas),losses=sum(d<0 for d in deltas))
    treatment=dict(mean_recall_s=mean(row['recall_s'] for row in rows),
        mean_baseline_recall_s=mean(row['baseline_recall_s'] for row in rows),
        mean_incremental_recall_s=mean(row['incremental_recall_s'] for row in rows),
        mean_query_compute_s=mean(row['query_compute_s'] for row in rows),
        mean_packet_tokens=mean(row['packet_tokens'] for row in rows),
        mean_added_sections=mean(row['added_sections'] for row in rows),
        query_qwen_passes=sum(row['query_qwen_passes'] for row in rows),
        mean_query_compute_cells=mean(row['query_compute_cells'] for row in rows),
        mean_dense_attention_cells=mean(row['dense_attention_cells'] for row in rows),
        mean_compute_cell_ratio=(mean(value for row in rows
            if (value:=row['compute_cell_ratio']) is not None) if any(
                row['compute_cell_ratio'] is not None for row in rows) else None),
        complete_support_checkpoints=sum(value['complete'] for row in rows for value in row['support_quotes'].values()))
    usage=defaultdict(int)
    for path in (root/'api').glob('*/response.json'):
        for key,value in (read(path).get('usage') or {}).items():
            if type(value) is int: usage[key]+=value
    save(root/'answer-report.json',dict(totals=totals,treatment=treatment,usage=dict(usage),
        state=read(root/'state.json'),limitations=plan['limitations'],
        questions=[dict(id=q['id'],points={arm:grades[q['id']][arm]['points'] for arm in arms},
                        possible=grades[q['id']]['baseline']['possible']) for q in plan['questions']]))
    emit(phase='report',totals=totals,treatment=treatment)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','generate','grade','report'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cav-bundle',type=Path,default=CAV)
    parser.add_argument('--arm',choices=('cav_reinjection','spectral_reinjection'),
        default=DEFAULT_ARM)
    args=parser.parse_args()
    if args.phase=='prepare':
        prepare(args.root,args.cav_bundle,args.arm)
    else:
        globals()[args.phase](args.root)
