"""Freeze, embed, fit on train/dev only, then evaluate one local BGE tactic head."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from memory_condense.application.tactic_router import (
    BGETacticRouter, FORMAT, LABELS, render_input,
)
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.runtime.artifacts import read, save, emit
from tools.evaluate_dialogue_neighborhood import digest
from tools.tactic_router_cases import cases

ROOT = Path('eval_results/bge-tactic-router-20261007-r1')


def validate(root):
    plan = read(root/'plan.json')
    for name, sha in plan['frozen'].items():
        if digest(root/name) != sha:
            raise ValueError('Frozen classifier input changed: '+name)
    for name, sha in plan['implementation'].items():
        if digest(name) != sha:
            raise ValueError('Classifier implementation changed: '+name)
    return plan


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    rows = cases()
    old_questions = set()
    source = Path('eval_results/scale-5m-20261003-r2/run-plan.json')
    candidates = read(source)['questions']
    for q in candidates[::5]:
        text = q['retrieval_query']
        if text in old_questions:
            continue
        old_questions.add(text)
        rows.append(dict(id='real-fact-'+q['question_id'],split='real_factual',label='focused',
            query=text,recent=[],group=q['question_id'],source=str(source)))
    for folder in ('dialogue-index-dump-20261006-r2','dialogue-index-extension-20261006-r2'):
        path = Path('eval_results')/folder/'questions.json'
        for q in read(path)['questions']:
            rows.append(dict(id='real-overview-'+q['id'],split='known_overview',label='overview',
                query=q['question'],recent=[],group=q['id'],source=str(path)))
    texts = [render_input(r['query'],r['recent'])[0] for r in rows]
    if len(set(texts)) != len(texts):
        raise ValueError('Duplicate complete inputs across frozen dataset')
    # Identical queries with different context are a deliberate paired test;
    # prohibit crossing splits even when the complete context differs.
    query_splits = {}
    for row in rows:
        query_splits.setdefault(row['query'],set()).add(row['split'])
    if any(len(v)>1 for v in query_splits.values()):
        raise ValueError('Question wording leaked across partitions')
    frozen = {}
    for split in dict.fromkeys(r['split'] for r in rows):
        name = split+'.json'
        save(root/name,dict(cases=[r for r in rows if r['split']==split]))
        frozen[name] = digest(root/name)
    save(root/'inputs.json',dict(cases=[{k:r[k] for k in ('id','query','recent')} for r in rows]))
    frozen['inputs.json'] = digest(root/'inputs.json')
    paths = [__file__,'tools/tactic_router_cases.py',
        'src/memory_condense/application/tactic_router.py','src/memory_condense/runtime/embedding.py']
    save(root/'plan.json',dict(frozen=frozen,implementation={p:digest(p) for p in paths},
        models=1,model='existing FastEmbed BGE-M3 FP32 on CPU',new_transformers=0,
        provider_calls=0,gpu_calls=0,history_reingestions=0,
        counts=dict(Counter(r['split'] for r in rows)),
        candidates=dict(C=[.1,1.,10.,100.],min_score=[0.,.5,.6,.7],min_margin=[0.,.1,.2]),
        selection='Maximize dev balanced accuracy minus 0.1 times focused-to-overview false-positive rate; ties favor fewer abstentions on specified queries, then smaller C/thresholds.',
        holdout_policy='Fit reads only train/dev labels. Freeze selected head before opening test labels. No test-guided retuning.',
        limitations=[
            'Authored scope labels are not measured answer-policy superiority.',
            'Authored test wording/context pairs held out, but same author and semantic taxonomy.',
            'Known overview questions reuse the previous development assay; not independent validation.',
            'Real factual queries are a deterministic unique subset of the existing 5M battery, not new histories.',
            'Softmax scores are not calibrated probabilities of correctness.',
            'Recent context tests use concise authored context; a separate live-buffer evaluation is required.',
        ]))
    emit(phase='prepared',counts=dict(Counter(r['split'] for r in rows)),total=len(rows))


def embed(root):
    from memory_condense.runtime.embedding import FastEmbedBGE
    from memory_condense.runtime.config import RuntimeAssets
    from memory_condense.search.summary_semantic_index import summary_embedding_identity
    validate(root)
    if (root/'embedding.json').exists():
        raise ValueError('Embeddings already frozen; reuse rather than rerun')
    rows = read(root/'inputs.json')['cases']
    started = time.perf_counter()
    encoder = FastEmbedBGE(assets=RuntimeAssets.resolve('.cache'))
    load_s = time.perf_counter()-started
    vectors, timings = [], []
    identity = summary_embedding_identity(encoder)
    try:
        for i,row in enumerate(rows):
            text, contextualized = render_input(row['query'],row['recent'])
            started = time.perf_counter()
            vectors.append(encoder.embed_query(text))
            timings.append(dict(id=row['id'],elapsed_s=time.perf_counter()-started,
                contextualized=contextualized,text_sha256=quote_sha256(text)))
            if (i+1)%20 == 0:
                emit(phase='embedded',complete=i+1,total=len(rows))
        if summary_embedding_identity(encoder) != identity:
            raise ValueError('Embedding identity changed')
    finally:
        encoder.close()
    matrix = np.asarray(vectors,dtype=np.float32)
    matrix /= np.linalg.norm(matrix,axis=1,keepdims=True)
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite embeddings')
    np.save(root/'vectors.npy',matrix)
    save(root/'embedding.json',dict(identity=identity,model_load_s=load_s,
        vectors_sha256=digest(root/'vectors.npy'),inputs_sha256=digest(root/'inputs.json'),
        timings=timings,rows=len(rows),dimension=matrix.shape[1]))
    emit(phase='embedding_complete',rows=len(rows),model_load_s=load_s,
        mean_s=float(np.mean([t['elapsed_s'] for t in timings])))


def vector_data(root):
    embedding = read(root/'embedding.json')
    if (digest(root/'vectors.npy') != embedding['vectors_sha256']
            or digest(root/'inputs.json') != embedding['inputs_sha256']):
        raise ValueError('Frozen classifier embeddings changed')
    rows = read(root/'inputs.json')['cases']
    matrix = np.load(root/'vectors.npy')
    return embedding, {r['id']:v for r,v in zip(rows,matrix,strict=True)}


def probabilities(x,weights,intercept):
    logits = x @ weights.T+intercept
    exp = np.exp(logits-logits.max(axis=1,keepdims=True))
    return exp/exp.sum(axis=1,keepdims=True)


def predictions(scores,min_score,min_margin):
    order = np.argsort(-scores,axis=1,kind='stable')
    result = order[:,0].copy()
    for i,(top,second) in enumerate(order[:,:2]):
        if scores[i,top]<min_score or scores[i,top]-scores[i,second]<min_margin:
            result[i] = 2
    return result


def metrics(y,p):
    confusion = [[int(np.sum((y==i)&(p==j))) for j in range(3)] for i in range(3)]
    recalls = [confusion[i][i]/sum(confusion[i]) for i in range(3) if sum(confusion[i])]
    n = len(y); specified = y!=2; accepted = p!=2
    return dict(n=n,correct=int(np.sum(y==p)),accuracy=float(np.mean(y==p)),
        balanced_accuracy=float(np.mean(recalls)),confusion=confusion,
        overview_false_positives=confusion[0][1],
        overview_false_positive_rate=confusion[0][1]/sum(confusion[0]) if sum(confusion[0]) else 0.,
        overview_misses=confusion[1][0]+confusion[1][2],
        abstentions=int(np.sum(p==2)),specified_abstentions=int(np.sum(specified&~accepted)),
        accepted=int(np.sum(accepted)),
        accepted_accuracy=float(np.mean(y[accepted]==p[accepted])) if np.any(accepted) else None)


def fit(root):
    from sklearn.linear_model import LogisticRegression
    plan = validate(root)
    embedding, vectors = vector_data(root)
    train, dev = (read(root/(name+'.json'))['cases'] for name in ('train','dev'))
    xt, xd = (np.asarray([vectors[r['id']] for r in rows],dtype=np.float64) for rows in (train,dev))
    yt, yd = (np.asarray([LABELS.index(r['label']) for r in rows]) for rows in (train,dev))
    trials = []
    winner = None
    for c in plan['candidates']['C']:
        classifier = LogisticRegression(C=c,class_weight='balanced',solver='lbfgs',max_iter=2000)
        classifier.fit(xt,yt)
        if tuple(classifier.classes_) != (0,1,2):
            raise ValueError('Classifier class order changed')
        weights, intercept = classifier.coef_,classifier.intercept_
        scores = probabilities(xd,weights,intercept)
        for threshold in plan['candidates']['min_score']:
            for margin in plan['candidates']['min_margin']:
                m = metrics(yd,predictions(scores,threshold,margin))
                utility = m['balanced_accuracy']-.1*m['overview_false_positive_rate']
                key = (utility,-m['specified_abstentions'],-c,-threshold,-margin)
                trial = dict(C=c,min_score=threshold,min_margin=margin,utility=utility,metrics=m)
                trials.append(trial)
                if winner is None or key>winner[0]:
                    winner = (key,trial,weights.copy(),intercept.copy())
    _,selected,weights,intercept = winner
    artifact = dict(format=FORMAT,labels=LABELS,embedding_identity=embedding['identity'],
        dimension=embedding['dimension'],weights=weights.tolist(),intercept=intercept.tolist(),
        min_score=selected['min_score'],min_margin=selected['min_margin'],
        input_format='bounded-recent-v1',max_query_tokens=512,max_context_tokens=256,
        training=dict(train_sha256=digest(root/'train.json'),dev_sha256=digest(root/'dev.json'),
            embedding_sha256=digest(root/'embedding.json'),selected_C=selected['C'],
            train_count=len(train),dev_count=len(dev),test_used_for_selection=False))
    artifact['receipt_sha256'] = identity_sha256(artifact)
    save(root/'classifier.json',artifact)
    # Nearest class centroids are an intentionally simple BGE-only baseline.
    centroids = np.asarray([xt[yt==i].mean(axis=0) for i in range(3)])
    centroids /= np.linalg.norm(centroids,axis=1,keepdims=True)
    save(root/'fit.json',dict(selected=selected,trials=trials,centroids=centroids.tolist(),
        model_sha256=digest(root/'classifier.json'),labels_read=['train','dev']))
    emit(phase='fit_complete',selected=selected,model_sha256=digest(root/'classifier.json'))


class CachedEncoder:
    def __init__(self, identity):
        value = json.loads(identity)
        self.model_name = value['model_id']
        self.model_revision = value['model_revision']
        self.checkpoint_sha256 = value['checkpoint_sha256']
        self.execution_identity = value['execution']
    def embed_query(self,_text):
        raise AssertionError('Evaluation must reuse already measured BGE vectors')


def evaluate(root):
    plan = validate(root)
    embedding,vectors = vector_data(root)
    fit_result = read(root/'fit.json')
    if digest(root/'classifier.json') != fit_result['model_sha256']:
        raise ValueError('Selected classifier changed before evaluation')
    router = BGETacticRouter(CachedEncoder(embedding['identity']),root/'classifier.json')
    centroids = np.asarray(fit_result['centroids'])
    reports, decisions = {}, []
    for split in ('test','real_factual','known_overview'):
        rows = read(root/(split+'.json'))['cases']
        y, predicted, baseline = [], [], []
        for row in rows:
            text,_ = render_input(row['query'],row['recent'])
            bound = dict(vector=vectors[row['id']],text_sha256=quote_sha256(text),
                embedding_identity=embedding['identity'])
            decision = router.prioritize(row['query'],recent=row['recent'],input_vector=bound)
            label = 'uncertain' if decision.abstained else decision.predicted_scope
            y.append(LABELS.index(row['label'])); predicted.append(LABELS.index(label))
            baseline.append(int(np.argmax(centroids @ vectors[row['id']])))
            decisions.append(dict(id=row['id'],split=split,expected=row['label'],prediction=label,
                query=row['query'],recent=row['recent'],decision=asdict(decision),correct=label==row['label']))
        y,predicted,baseline = map(np.asarray,(y,predicted,baseline))
        reports[split] = dict(linear=metrics(y,predicted),nearest_centroid=metrics(y,baseline))
    pair_rows = [r for r in decisions if r['split']=='test' and r['recent']]
    groups = {}
    for row in pair_rows:
        groups.setdefault(row['query'],[]).append(row)
    times = [t['elapsed_s'] for t in embedding['timings']]
    report = dict(model='BGE-M3 FP32 FastEmbed CPU + linear classification head',local_models=1,
        new_models_downloaded=0,provider_calls=0,history_reingestions=0,
        partitions=reports,context_pairs=dict(pairs=len(groups),
            both_correct=sum(all(r['correct'] for r in rs) for rs in groups.values()),
            individual_correct=sum(r['correct'] for r in pair_rows),total=len(pair_rows)),
        latency=dict(model_load_s=embedding['model_load_s'],embedding_mean_s=float(np.mean(times)),
            embedding_p50_s=float(np.quantile(times,.5)),embedding_p95_s=float(np.quantile(times,.95)),
            head_mean_s=float(np.mean([r['decision']['elapsed_s'] for r in decisions]))),
        selected=fit_result['selected'],classifier_sha256=digest(root/'classifier.json'),
        decisions=decisions,limitations=plan['limitations'],
        production_default_changed=False,automatic_proxy_dispatch=False)
    save(root/'report.json',report)
    emit(phase='evaluation_complete',partitions=reports,context_pairs=report['context_pairs'],latency=report['latency'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','embed','fit','evaluate'))
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    globals()[args.phase](args.root)
