"""Local matched replay of rank and summary-weighted learning, on disposable DBs.

Uses observed baseline deliveries from the reader assay, not answer/grader data.
Short input questions are exact identity summaries; archive summaries are reused.
This tests learning and subsequent retrieval, not a full streaming ingest cycle.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import numpy as np

from memory_condense.application.chat_native import learn_native_packet, native_packet
from memory_condense.application.chat_session import ChatEvent
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
from memory_condense.application.native_experiment_learning import summary_weights
from memory_condense.application.native_experiments import NativeExperimentPolicy, PreparedNativeExperiments, SummaryGraph
from memory_condense.application.native_spine_retrieval import ResidentNativeSpineMemory
from memory_condense.application.native_spine_policy import retrieval_limits
from memory_condense.associations.consolidation import LiveConsolidationStore, context_activations
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.persistence import native_spine_store, native_spine_parent_store
from memory_condense.persistence.db import Database
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.runtime.artifacts import read, save, emit
from memory_condense.runtime.config import RuntimeAssets
from memory_condense.runtime.local import LocalRuntime
from memory_condense.search.native_spine_user_completion import NativeSpineUserCompletionRouter
from memory_condense.search.section_summary import RawSectionSpan, SectionSummary
from memory_condense.search.summary_semantic_index import summary_embedding_identity
from tools.characterize_native_experiments import ORIGINAL, packet_facts, validate
from tools.evaluate_dialogue_neighborhood import digest, source_turns


def main(root, assay):
    plan = validate(assay)
    root.mkdir(parents=True, exist_ok=False)
    source = plan['source']; directory = Path(source['directory'])
    if any(digest(directory/n) != sha for n,sha in source['files'].items()):
        raise ValueError('Source changed')
    turns = source_turns(directory); by_id = {t.turn_id:t for t in turns}
    snapshot = native_spine_store.load(directory/'native-spine.sqlite', turns=turns)
    parents,_ = native_spine_parent_store.load(directory/native_spine_parent_store.FILENAME,
        hierarchy=snapshot.hierarchy, native_receipt=snapshot.receipt)
    router = NativeSpineUserCompletionRouter(snapshot.semantic,snapshot.hierarchy,parents)
    learning_router = SimpleNamespace(atoms=dict(router.atoms))
    policy = NativeExperimentPolicy(modules=('qwen_consolidation',))
    events = [ChatEvent(t.turn_id,t.role,t.text,t.created_at.isoformat(),dict(source_id=t.source_id)) for t in turns]
    training = [q for q in plan['questions'] if q['category']!='current_state']
    heldout = [q for q in plan['questions'] if q['category']=='current_state']
    save(root/'plan.json',dict(assay_plan_sha256=digest(assay/'plan.json'),
        implementation_sha256=digest(Path(__file__)),source=source,training=[q['id'] for q in training],
        heldout=[q['id'] for q in heldout],observations_per_question=1,
        limitations=['Replay of observed baseline deliveries without success labels.',
            'Nine short input questions get identity summaries; existing archive is not reingested.',
            'Tests production learning functions and scalar persistence, not streaming publication or answer accuracy.',
            'Three held-out queries from the same development archive. No independent corpus.']))
    apps = {}
    for arm in ('rank','qwen'):
        with sqlite3.connect((directory/'memory.db').as_uri()+'?mode=ro',uri=True) as original:
            with sqlite3.connect(root/(arm+'.db')) as target:
                original.backup(target)
        db = Database(root/(arm+'.db'))
        app = SimpleNamespace(_db=db,transcript=TranscriptStore(db))
        def admitted(chunk_ids, app=app):
            result=set()
            for cid in chunk_ids:
                row=app._db.execute('SELECT turn_id,start_char,end_char,text FROM chunks WHERE chunk_id=?',(cid,)).fetchone()
                if row:
                    turn=app.transcript.get_turn(row[0])
                    if turn and turn.text[row[1]:row[2]]==row[3]: result.add(cid)
            return result
        app._consolidation = LiveConsolidationStore(db,native_chunk_validator=admitted)
        app.observe_context_access = lambda *args,app=app,**kwargs: MemoryCondenser.observe_context_access(app,*args,**kwargs)
        apps[arm]=app
    runtime = LocalRuntime(root/'runtime',assets=RuntimeAssets.resolve('.cache'))
    records=[]
    try:
        start=time.perf_counter(); attention=runtime.attention(root/'attention'); attention.prepare()
        prepare_s=time.perf_counter()-start
        for i,q in enumerate(training):
            turn_id='learning-assay:'+q['id']; cid=turn_id+':chunk'
            for app in apps.values():
                turn=app.transcript.stage('user',q['question'],source_id='learning-assay',turn_id=turn_id,
                    created_at=max(t.created_at for t in turns)+timedelta(seconds=i+1))
                app.transcript.publish_turn(turn)
                app._db.execute('INSERT INTO chunks (chunk_id,turn_id,text,start_char,end_char,token_count) VALUES (?,?,?,?,?,?)',
                    (cid,turn_id,turn.text,0,len(turn.text),count_tokens(turn.text)))
                app._db.connection.commit()
            atom=SectionSummary(turn_id+':summary',turn.source_id,turn.text,(RawSectionSpan.from_turn(turn),),
                                'assay-short-input-identity-summary')
            learning_router.atoms[atom.section_id]=atom
            events.append(ChatEvent(turn_id,'user',turn.text,turn.created_at.isoformat(),dict(source_id=turn.source_id)))
            delivered=read(assay/'packets'/q['id']/'baseline.json')
            packet=SimpleNamespace(input_event_id=turn_id,references=delivered['references'])
            weights={}; details={}; selected_chunks=[]
            def weight(chunks):
                nonlocal weights,details
                selected_chunks.extend(chunks)
                weights,details=summary_weights(apps['qwen'],packet,chunks,learning_router,attention,policy)
                if weights is None or details['status']!='selected':
                    raise RuntimeError('Qwen learning failed: '+str(details))
                return weights
            elapsed={}; updates={}
            for arm,app in apps.items():
                start=time.perf_counter()
                update=learn_native_packet(app,packet,events,access_event_id='observed:'+q['id'],
                    weight_summaries=weight if arm=='qwen' else None)
                elapsed[arm]=time.perf_counter()-start; updates[arm]=asdict(update)
                again=learn_native_packet(app,packet,events,access_event_id='observed:'+q['id'],
                    weight_summaries=(lambda chunks: weights) if arm=='qwen' else None)
                if again.created: raise ValueError('Duplicate learned twice')
            baseline=context_activations([],selected_chunks)
            row=dict(id=q['id'],receipt=details,elapsed_s=elapsed,updates=updates,
                rank_activations={n.key:v for n,v in baseline.items()},
                qwen_activations={n.key:v for n,v in weights.items()},
                changed_nodes=sum(weights[n]!=v for n,v in baseline.items()),
                delivered_packet_sha256=digest(assay/'packets'/q['id']/'baseline.json'))
            records.append(row); emit(phase='learned',id=q['id'],changed_nodes=row['changed_nodes'],elapsed_s=elapsed)
        graphs={arm:SummaryGraph.capture(app,router,policy) for arm,app in apps.items()}
        edges={arm:{(r[0],r[1]):r[2] for r in app._db.execute(
            'SELECT node_low,node_high,coactivation_mass FROM consolidation_edges').fetchall()} for arm,app in apps.items()}
        encoder=runtime.embedding()
        memory=ResidentNativeSpineMemory(snapshot.semantic,snapshot.hierarchy,encoder=encoder,load_turn=by_id.get)
        memory.router=router
        identity=summary_embedding_identity(encoder); embedding=read(ORIGINAL/'embedding.json')
        if identity!=embedding['identity']: raise ValueError('Embedding identity changed')
        memory.configure_dialogue(DialogueRetrievalPolicy(),np.load(ORIGINAL/'episode-vectors.npy'),
            embedding_identity=identity,turn_positions={t.turn_id:i for i,t in enumerate(turns)})
        memory.configure_tactics(runtime.tactic_router())
        def retrieve(query,dated,**kwargs): return memory.retrieve(query,dated,**retrieval_limits(memory,kwargs))
        view=SimpleNamespace(transcript=SimpleNamespace(get_turn=by_id.get),retrieve_native_spine=retrieve,decision_revisions=())
        reference=read(assay/'private-reference.json'); comparisons=[]
        day=max(t.created_at for t in turns).strftime('%Y/%m/%d (%a)')
        for q in heldout:
            baseline=None
            for graph_arm,module in [('baseline',None),*[(a,m) for a in apps for m in ('learned_associations','heat_diffusion')]]:
                memory.experiments=None if module is None else PreparedNativeExperiments(router,
                    NativeExperimentPolicy(modules=(module,)),graph=graphs[graph_arm])
                start=time.perf_counter()
                packet=native_packet(view,q['question'],f'[Question asked at {day} 23:59] '+q['question'],events)
                seconds=time.perf_counter()-start
                if baseline is None: baseline=packet
                elif packet['hydration']!=baseline['hydration']: raise ValueError('Baseline changed')
                label=graph_arm+('-'+module if module else '')
                save(root/'heldout'/q['id']/(label+'.json'),packet)
                comparisons.append(dict(id=q['id'],arm=label,recall_s=seconds,
                    identical_to_baseline=packet['text']==baseline['text'],
                    added_sections=packet['retrieval'].get('experiments',{}).get('delivered_section_ids',[]),
                    support_quotes=packet_facts(packet,{fid:reference['facts'][fid] for fid in q['facts']})))
        common=edges['rank'].keys() & edges['qwen'].keys()
        report=dict(model_prepare_s=prepare_s,records=records,graphs={a:dict(g.receipt) for a,g in graphs.items()},
            edge_comparison=dict(rank=len(edges['rank']),qwen=len(edges['qwen']),common=len(common),
                changed_mass=sum(abs(edges['rank'][k]-edges['qwen'][k])>1e-12 for k in common),
                mean_absolute_change=sum(abs(edges['rank'][k]-edges['qwen'][k]) for k in common)/max(1,len(common))),
            heldout=comparisons,source_unchanged=all(digest(directory/n)==sha for n,sha in source['files'].items()))
        save(root/'report.json',report); emit(phase='complete',graphs=report['graphs'],edges=report['edge_comparison'])
    finally:
        runtime.close()
        for app in apps.values(): app._db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--assay',type=Path,required=True)
    args=p.parse_args(); main(args.root,args.assay)
