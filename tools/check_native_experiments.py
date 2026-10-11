"""Local integration assay on one already-ingested engineering archive.

No generation/API calls, no history reingestion and no claimed answer-accuracy
score. Captures candidate changes, exact hydration and real resident latency.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import numpy as np

from memory_condense.application.native_experiments import NativeExperimentPolicy, PreparedNativeExperiments, SummaryGraph
from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.application.native_spine_policy import retrieval_limits
from memory_condense.persistence import native_spine_store, native_spine_parent_store
from memory_condense.runtime.artifacts import read,save,emit
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
from tools.evaluate_dialogue_neighborhood import digest,source_turns


def decision_content(value):
    """Compare routing decisions, excluding measured per-call elapsed time."""
    if isinstance(value,dict):
        return {k:decision_content(v) for k,v in value.items() if k!='elapsed_s'}
    if isinstance(value,(list,tuple)):
        return [decision_content(v) for v in value]
    return value


def replay_accesses(directory, root, replay_root, router, turns, policy):
    """Replay observed cap8 co-accesses into a disposable DB; never invent edges."""
    from memory_condense.persistence.db import Database
    from memory_condense.associations.consolidation import LiveConsolidationStore,context_activations
    target=root/'access-replay.db'
    with sqlite3.connect((directory/'memory.db').as_uri()+'?mode=ro',uri=True) as source:
        with sqlite3.connect(target) as output:
            source.backup(output)
    db=Database(target)
    by_id={t.turn_id:t for t in turns}
    def validate(chunk_ids):
        admitted=set()
        for cid in chunk_ids:
            row=db.execute('SELECT turn_id,start_char,end_char,text FROM chunks WHERE chunk_id=?',(cid,)).fetchone()
            if row and row[0] in by_id and by_id[row[0]].text[row[1]:row[2]]==row[3]:
                admitted.add(cid)
        return admitted
    store=LiveConsolidationStore(db,native_chunk_validator=validate)
    records=[]
    try:
        for path in sorted((replay_root/'packets').glob('*/learned_associations.json')):
            packet=read(path)
            chunks=[]
            for section in packet['control_hydration']['sections']:
                for evidence in section['evidence']:
                    span=evidence['span']
                    chunks += [r[0] for r in db.execute('SELECT chunk_id FROM chunks '
                        'WHERE turn_id=? AND start_char<? AND end_char>? ORDER BY start_char,chunk_id',
                        (span['turn_id'],span['end_char'],span['start_char'])).fetchall()]
            chunks=list(dict.fromkeys(chunks))[:16]
            store.observe('assay-access:'+path.parent.name,context_activations([],chunks))
            records.append(dict(question=path.parent.name,chunks=chunks,packet_sha256=digest(path)))
        graph=SummaryGraph.capture(SimpleNamespace(_db=db),router,policy)
        save(root/'access-replay.json',dict(records=records,graph=dict(graph.receipt),
            interpretation='Actual previous packet co-accesses, once each; no success labels or fabricated repetition. Development replay, not an independent accuracy experiment.'))
        return graph
    finally:
        db.close()


def probe_qwen_learning(attention, router, root, policy):
    """Exercise scalar weighting on archived summaries without altering history."""
    from memory_condense.associations.consolidation import ConsolidationNode
    atoms=sorted(router.atoms.values(),key=lambda a:(a.spans[0].created_at,a.section_id))
    query=next(a.summary for a in reversed(atoms) if a.spans[0].role=='user')
    # This is a mechanism probe, not a claimed historical co-access observation.
    candidates=tuple(SimpleNamespace(section_id=ConsolidationNode.chunk('probe-'+str(i)).key,
        summary=a.summary) for i,a in enumerate(atoms[:8]))
    prepare_started=time.perf_counter()
    attention.prepare()
    prepare_s=time.perf_counter()-prepare_started
    start=time.perf_counter()
    scores,receipt=attention.experiment('qwen_consolidation',query,candidates,
        policy=policy,remaining_passes=policy.max_qwen_passes)
    if receipt['status']!='selected' or not scores or any(not 0<v<=1 for _,v in scores):
        raise RuntimeError('Qwen summary weighting probe failed')
    save(root/'learning-probe.json',dict(receipt=receipt,scalars=scores,
        elapsed_s=time.perf_counter()-start,model_prepare_s=prepare_s,persisted_observations=0,
        interpretation='Real Qwen scalar weighting on stored summaries; synthetic node labels; no accuracy claim.'))


def fit_cav(attention, root, dataset=Path('examples/cav/live_memory_concepts.json')):
    """Fit one explicit concept assay on the actual resident layer/precision."""
    from safetensors.torch import save_file
    from memory_condense.tooling.cav_probe import (load_concept_datasets,fit_mean_difference_cav,
        evaluate_cav,bootstrap_stability,random_label_control)
    from memory_condense.associations.cav_memory import cav_layer_key
    import torch
    attention.prepare()
    encoder=attention.scorer.linker.encoder
    dataset=Path(dataset)
    datasets=load_concept_datasets(dataset)
    texts=list(dict.fromkeys(t for d in datasets for t in d.texts))
    before=time.perf_counter()
    with attention.gate,torch.cuda.stream(attention.stream),torch.inference_mode():
        vectors=encoder.encode_layers(texts,layers=(4,),batch_size=4)[4]
        attention.stream.synchronize()
    index={t:i for i,t in enumerate(texts)}
    def take(rows):
        return vectors[[index[t] for t in rows]]
    concepts,tensors=[],{}
    for d in datasets:
        positive,negative=take(d.train_positive),take(d.train_negative)
        test_positive,test_negative=take(d.test_positive),take(d.test_negative)
        cav=fit_mean_difference_cav(positive,negative)
        metrics=evaluate_cav(cav,test_positive,test_negative)
        stability=bootstrap_stability(positive,negative,repeats=16,seed=4)
        control=random_label_control(positive,negative,test_positive,test_negative,repeats=16,seed=1004)
        concepts.append(dict(name=d.name,layers=[dict(layer=4,threshold=cav.threshold,
            test=metrics,bootstrap=stability,random_label_control=control)]))
        tensors[cav_layer_key(d.name,4)]=cav.vector.contiguous()
    root.mkdir(parents=True,exist_ok=True)
    checkpoint=encoder.checkpoint_identity.checkpoint_sha256
    save(root/'report.json',dict(model='Qwen/Qwen3-8B',checkpoint_sha256=checkpoint,
        runtime_dtype=encoder.dtype_name,concepts=concepts,dataset=str(dataset),dataset_sha256=digest(dataset),
        examples=len(texts),elapsed_s=time.perf_counter()-before,
        limitation='Controlled concept assay; not evidence of retrieval or answer improvement.'))
    save_file(tensors,str(root/'vectors.safetensors'))
    bundle=dict(checkpoint_sha256=checkpoint,layer=4,report='report.json',vectors='vectors.safetensors',
        report_sha256=digest(root/'report.json'),vectors_sha256=digest(root/'vectors.safetensors'))
    save(root/'bundle.json',bundle)
    emit(phase='cav_refit',examples=len(texts),elapsed_s=time.perf_counter()-before,
         held_out=[dict(name=c['name'],accuracy=c['layers'][0]['test']['balanced_accuracy']) for c in concepts])
    return str((root/'bundle.json').resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--questions',type=int,default=3)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--fit-cav',action='store_true')
    parser.add_argument('--cav-dataset',type=Path,
        default=Path('examples/cav/live_memory_concepts.json'))
    parser.add_argument('--cav-bundle')
    parser.add_argument('--replay-accesses',type=Path)
    parser.add_argument('--retrieval-policy',choices=('cap8','auto'),default='cap8')
    args=parser.parse_args()
    if not 1<=args.questions<=15 or args.fit_cav and not args.gpu:
        parser.error('Use 1..15 questions; fitting CAVs requires --gpu')
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    original=Path('eval_results/dialogue-neighborhood-20261006-r1')
    source=read(original/'source.json')
    directory=Path(source['directory'])
    if not all(digest(directory/name)==sha for name,sha in source['files'].items()):
        raise ValueError('Archived source changed')
    turns=source_turns(directory)
    snapshot=native_spine_store.load(directory/'native-spine.sqlite',turns=turns)
    parents,_=native_spine_parent_store.load(directory/native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy,native_receipt=snapshot.receipt)
    questions=read('eval_results/dialogue-index-dump-20261006-r2/questions.json')['questions'][:args.questions]
    runtime=LocalRuntime(root/'runtime',assets=RuntimeAssets.resolve('.cache'))
    try:
        emit(phase='loading_resident_bge',source_tokens=source['snapshot']['body_tokens'])
        encoder=runtime.embedding()
        by_id={t.turn_id:t for t in turns}
        memory=ResidentNativeSpineMemory(snapshot.semantic,snapshot.hierarchy,encoder=encoder,load_turn=by_id.get)
        memory.router=NativeSpineUserCompletionRouter(snapshot.semantic,snapshot.hierarchy,parents)
        if args.retrieval_policy=='auto':
            from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
            from memory_condense.search.summary_semantic_index import summary_embedding_identity
            identity=summary_embedding_identity(encoder)
            stored=read(original/'embedding.json')
            if identity!=stored['identity'] or digest(original/'episode-vectors.npy')!=stored['episode_vectors_sha256']:
                raise ValueError('Archived dialogue vectors differ from the admitted encoder')
            memory.configure_dialogue(DialogueRetrievalPolicy(),np.load(original/'episode-vectors.npy'),
                embedding_identity=identity,turn_positions={t.turn_id:i for i,t in enumerate(turns)})
            memory.configure_tactics(runtime.tactic_router())
        attention=runtime.attention(root/'attention') if args.gpu else None
        emit(phase='preparing_experiments',gpu=args.gpu,fit_cav=args.fit_cav)
        bundle=fit_cav(attention,root/'cav',args.cav_dataset) if args.fit_cav else args.cav_bundle
        modules=['learned_associations','heat_diffusion']
        if args.gpu:
            modules+=['qwen_summary_routing','qwen_associations','live_head_memory']
            if bundle:
                modules+=['cav_associations']
        policy=NativeExperimentPolicy(modules=tuple(modules),cav_bundle=bundle,max_qwen_passes=24)
        with sqlite3.connect((directory/'memory.db').as_uri()+'?mode=ro',uri=True) as conn:
            db=SimpleNamespace(execute=conn.execute,current_turn=lambda:conn.execute(
                'SELECT COALESCE(MAX(ordinal),0) FROM turns').fetchone()[0])
            graph=SummaryGraph.capture(SimpleNamespace(_db=db),memory.router,policy)
        if args.replay_accesses:
            graph=replay_accesses(directory,root,args.replay_accesses,memory.router,turns,policy)
        if attention is not None:
            probe_qwen_learning(attention,memory.router,root,policy)
        save(root/'plan.json',dict(source=source,questions=questions,policy=policy.identity_payload(),
            graph=dict(graph.receipt),api_calls=0,history_reingestions=0,summary_regenerations=0,
            control=args.retrieval_policy+', same archive/query/encoder; baseline hydration must match exactly.',
            limitations='Development integration assay. No answer model or accuracy score; empty learned graphs are reported as inactive evidence.'))
        records=[]
        day=max(t.created_at for t in turns).strftime('%Y/%m/%d (%a)')
        for question in questions:
            q=question['question']; dated=f'[Question asked at {day} 23:59] '+q
            memory.experiments=None
            start=time.perf_counter()
            control=memory.retrieve(q,dated,**retrieval_limits(memory,{}))
            base_s=time.perf_counter()-start
            for module in [*modules,'combined']:
                arm=policy if module=='combined' else replace(policy,modules=(module,),receipt_sha256='')
                memory.experiments=PreparedNativeExperiments(memory.router,arm,graph=graph,attention=attention)
                start=time.perf_counter()
                result=memory.retrieve(q,dated,**retrieval_limits(memory,{}))
                elapsed=time.perf_counter()-start
                if result.hydration!=control.hydration or result.routing!=control.routing:
                    raise AssertionError('Experiment altered baseline retrieval')
                if decision_content(result.tactic)!=decision_content(control.tactic) or result.hierarchy_context!=control.hierarchy_context:
                    save(root/'failure.json',dict(control_tactic=control.tactic,experiment_tactic=result.tactic,
                        hierarchy_equal=result.hierarchy_context==control.hierarchy_context))
                    raise AssertionError('Experiment altered automatic tactic or hierarchy context')
                for section in result.experiments.hydration.sections:
                    for e in section.evidence:
                        s=e.span
                        assert by_id[s.turn_id].text[s.start_char:s.end_char]==e.text
                receipt=result.experiments.receipt
                save(root/'packets'/question['id']/(module+'.json'),dict(
                    control_hydration=control.hydration.identity_payload(),experiment=receipt))
                row=dict(id=question['id'],module=module,elapsed_s=elapsed,control_s=base_s,
                    baseline_preserved=True,extra_sections=len(result.experiments.hydration.sections),
                    extra_tokens=result.experiments.hydration.context_token_count,
                    model_passes=receipt['query_qwen_passes'],
                    statuses=[a['status'] for a in receipt['arms']])
                records.append(row);emit(phase='recall',**row)
                if any(s in ('model_error','unavailable_attention') for s in row['statuses']):
                    save(root/'failed.json',dict(records=records))
                    raise RuntimeError('Stopped at an incompatible experimental arm')
        save(root/'report.json',dict(records=records,all_baselines_preserved=True,
            graph=dict(graph.receipt),accuracy_measured=False,api_calls=0))
        if not all(digest(directory/name)==sha for name,sha in source['files'].items()):
            raise ValueError('Assay modified its source archive')
    finally:
        runtime.close()


if __name__=='__main__':
    main()
