"""Fit and confirm a frozen BGE query head; no raw archives or provider calls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from memory_condense.application.history_intent import FORMAT, HistoryIntent
from memory_condense.application.tactic_router import BGETacticRouter, render_input
from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.runtime.artifacts import read, save, emit
from tools.history_intent_cases import cases


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    seen = set()
    for split in ('train', 'dev', 'test'):
        rows = cases(split)
        keys = [render_input(r['query'], r['recent'])[0] for r in rows]
        if len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError('Classifier partitions overlap')
        seen.update(keys)
        save(root/(split+'.json'), dict(cases=rows))
    save(root/'plan.json', dict(C=[.1, 1., 10., 100.], thresholds=[.5, .6, .7, .8],
        selection='Minimize development false positives, then maximize recall, then smaller C/threshold',
        limitations='Authored scope labels; shared author, not answer-quality evaluation. Frozen before fitting.',
        split_hashes={s:identity_sha256(read(root/(s+'.json'))) for s in ('train','dev','test')}))


def embed(root):
    from memory_condense.runtime.embedding import FastEmbedBGE
    from memory_condense.runtime.config import RuntimeAssets
    from memory_condense.search.summary_semantic_index import summary_embedding_identity
    if (root/'embedding.json').exists():
        raise ValueError('Reuse frozen vectors')
    rows = [r for s in ('train','dev','test') for r in read(root/(s+'.json'))['cases']]
    start = time.perf_counter()
    encoder = FastEmbedBGE(assets=RuntimeAssets.resolve('.cache'))
    load_s = time.perf_counter()-start
    identity = summary_embedding_identity(encoder)
    vectors, timings = [], []
    try:
        for i, row in enumerate(rows):
            text, _ = render_input(row['query'], row['recent'])
            start = time.perf_counter()
            vectors.append(encoder.embed_query(text))
            timings.append(time.perf_counter()-start)
            if (i+1)%20 == 0:
                emit(phase='embedding_history_intent', complete=i+1, total=len(rows))
    finally:
        encoder.close()
    x = np.asarray(vectors, dtype=np.float32)
    np.save(root/'vectors.npy', x)
    save(root/'embedding.json', dict(identity=identity, ids=[r['id'] for r in rows],
        vector_sha256=__import__('hashlib').sha256(x.tobytes()).hexdigest(), timings=timings, model_load_s=load_s))


def data(root):
    plan = read(root/'plan.json')
    for split, digest in plan['split_hashes'].items():
        if identity_sha256(read(root/(split+'.json'))) != digest:
            raise ValueError('Frozen partition changed')
    info = read(root/'embedding.json')
    x = np.load(root/'vectors.npy')
    if __import__('hashlib').sha256(x.tobytes()).hexdigest() != info['vector_sha256']:
        raise ValueError('Frozen vectors changed')
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    return plan, info, dict(zip(info['ids'], x, strict=True))


def fit(root):
    from sklearn.linear_model import LogisticRegression
    plan, info, vectors = data(root)
    train, dev = [read(root/(s+'.json'))['cases'] for s in ('train','dev')]
    xt, xd = [np.asarray([vectors[r['id']] for r in rows]) for rows in (train, dev)]
    yt, yd = [np.asarray([r['label'] for r in rows]) for rows in (train, dev)]
    winner, trials = None, []
    for c in plan['C']:
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=2000).fit(xt, yt)
        scores = model.predict_proba(xd)[:,1]
        for threshold in plan['thresholds']:
            p = scores >= threshold
            fp, tp = int(sum(p & (yd==0))), int(sum(p & (yd==1)))
            key = (-fp, tp, -c, -threshold)
            if plan.get('objective') == 'balanced':
                tpr, fpr = tp / sum(yd==1), fp / sum(yd==0)
                key = (.5*(tpr+1-fpr)-.1*fpr, -fp, -c, -threshold)
            trials.append(dict(C=c, threshold=threshold, false_positives=fp, true_positives=tp))
            if winner is None or key > winner[0]:
                winner = (key, model.coef_[0].copy(), float(model.intercept_[0]), threshold, c)
    _, weights, intercept, threshold, c = winner
    artifact = dict(format=FORMAT, dimension=len(weights), embedding_identity=info['identity'],
        input_format='bounded-recent-v1', weights=weights.tolist(), intercept=intercept, threshold=threshold,
        training=dict(train_sha256=plan['split_hashes']['train'], dev_sha256=plan['split_hashes']['dev'],
                      train_count=len(train), dev_count=len(dev), selected_C=c, test_used_for_selection=False))
    artifact['receipt_sha256'] = identity_sha256(artifact)
    save(root/'classifier.json', artifact)
    save(root/'fit.json', dict(trials=trials, selected_C=c, threshold=threshold,
                             classifier_sha256=artifact['receipt_sha256']))
    emit(phase='history_fit', C=c, threshold=threshold)


def evaluate(root):
    from tools.evaluate_bge_tactic_router import CachedEncoder
    _, info, vectors = data(root)
    model = read(root/'classifier.json')
    head = HistoryIntent(model, embedding_identity=info['identity'], dimension=len(model['weights']))
    reports = {}
    for split in ('dev', 'test'):
        rows = []
        for r in read(root/(split+'.json'))['cases']:
            decision = head.classify(vectors[r['id']])
            rows.append(dict(r, decision=decision, correct=int(decision['selected'])==r['label']))
        reports[split] = dict(total=len(rows), correct=sum(r['correct'] for r in rows),
            positives=sum(r['label'] for r in rows),
            false_positives=sum(not r['label'] and r['decision']['selected'] for r in rows),
            false_negatives=sum(r['label'] and not r['decision']['selected'] for r in rows), rows=rows)
    # Existing real factual vectors are an untouched dispatch regression check,
    # not new hand-labeled history-intent ground truth.
    previous = Path('eval_results/bge-tactic-router-20261007-r2')
    changed = []
    if previous.exists():
        from tools.evaluate_bge_tactic_router import vector_data
        old_info, old_vectors = vector_data(previous)
        assert old_info['identity'] == info['identity']
        for r in read(previous/'real_factual.json')['cases']:
            d = head.classify(old_vectors[r['id']])
            if d['selected']:
                changed.append(dict(query=r['query'], decision=d))
    times = info['timings']
    report = dict(partitions=reports, factual_dispatch_changes=changed,
        embedding_mean_s=float(np.mean(times)), embedding_p95_s=float(np.quantile(times,.95)),
        provider_calls=0, history_reingestions=0, classifier_sha256=model['receipt_sha256'])
    save(root/'report.json', report)
    emit(phase='history_confirmation', partitions={s:{k:v for k,v in r.items() if k!='rows'} for s,r in reports.items()},
         factual_dispatch_changes=len(changed), embedding_mean_s=report['embedding_mean_s'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','embed','fit','evaluate'))
    parser.add_argument('--root', type=Path, default=Path('eval_results/history-intent-20261008-r1'))
    args = parser.parse_args()
    globals()[args.phase](args.root)
