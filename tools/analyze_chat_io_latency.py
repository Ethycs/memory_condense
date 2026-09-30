"""Reconcile saved live latency intervals without running ingestion or models."""
from pathlib import Path
from tools.engineering_research_gateway import read, save, emit


def analyze():
    before_root = Path('eval_results/chat-io-live-20260929-r1')
    after_root = Path('eval_results/chat-io-optimization-20260929-r1')
    before, after = (read(root/'exchange-2.json') for root in (before_root, after_root))
    timings = after['backend_timings']
    # Recovery completed before the measured exchange; exclude it explicitly.
    refreshes = [t for t in timings if t['operation']=='sync'][1:]
    if [t['history_turns'] for t in refreshes] != [5438, 5439, 5441]:
        raise ValueError('Unexpected optimized probe population')
    recall = sum(t['elapsed_s'] for t in timings if t['operation']=='recall')
    learn = sum(t['elapsed_s'] for t in timings if t['operation']=='learn')
    phases = ('compile_s', 'raw_ingest_s', 'summary_embedding_s', 'publish_s')
    input_refresh = refreshes[0]
    after_reader = after['result']['response']['elapsed_s']
    answer = {p: input_refresh[p] for p in phases}
    answer.update(recall_s=recall, reader_s=after_reader)
    answer['other_s'] = after['answer_s'] - sum(answer.values())
    full = {p: sum(t[p] for t in refreshes) for p in phases}
    full.update(recall_s=recall, learning_s=learn)
    full['other_s'] = after['full_cycle_s'] - sum(full.values())
    before_reader = before['result']['response']['elapsed_s']
    artifacts = [(p,read(p)) for p in (before_root/'live').glob('chat-*/*.json')
                 if p.name in ('ingest.json','reopened.json')]
    def one(folder_prefix, count):
        values = [v for p,v in artifacts if p.parent.name.startswith(folder_prefix) and v.get('history_turns')==count]
        if len(values)!=1:
            raise ValueError('Ambiguous baseline phase')
        return values[0]['elapsed_s']
    old_prepare = one('chat-sync-',5427) + one('chat-verify-',5427)
    old_recall = one('chat-recall-',5427)
    comparison = dict(
        input_preparation_s=[old_prepare,input_refresh['elapsed_s']],
        recall_operation_s=[old_recall,recall],
        answer_generation_s=[before_reader,after_reader],
        process_journal_coordination_s=[before['answer_s']-old_prepare-old_recall-before_reader,
            after['answer_s']-input_refresh['elapsed_s']-recall-after_reader],
        answer_total_s=[before['answer_s'],after['answer_s']],
        remaining_after_answer_s=[before['full_cycle_s']-before['answer_s'],after['full_cycle_s']-after['answer_s']],
        full_cycle_s=[before['full_cycle_s'],after['full_cycle_s']])
    result = dict(comparison=comparison, optimized_answer_breakdown=answer,
        optimized_full_cycle_breakdown=full, optimized_refreshes=refreshes,
        reader_overlaps_background_work=True, full_cycle_includes_answer=True,
        optimized_setup_and_recovery_excluded=True,
        compile_includes_summary_calls_attention_hierarchy_and_possible_model_loading=True,
        other_is_remainder_not_separately_instrumented=True,
        original_and_optimized_are_analogous_not_identical_requests=True)
    save(after_root/'latency-breakdown.json', result)
    emit(**result)
    # Bind the model calls to their sealed gateway request receipts for the
    # fresh follow-up, excluding first-exchange calls and interrupted recovery.
    bindings = {
        'raw': {
            '3b7c585fc6b3d2279ccb192dc558b038f4a87c0fad62cb2584e6268bd2dd22fe',
            '71da612b818178621ca4cd4e8ab5663002bb8af270e55c44eac625ce43a20c47',
            '5bdca041b6ade1108cacaa77b51979d807051d47346a9e04902b828198b61928'},
        'merge': {'c1388cdd90d8db388923ceb8314ef88a52151c41f7c2e59dc6a47007633f5d54'}}
    from tools.matched_eval.artifacts import read_sealed_json
    model_rows = []
    for path in (after_root/'gateway').glob('*.request.json'):
        request = read_sealed_json(path)
        job, sha = request.payload, request.sha256
        if sha not in bindings.get(job['kind'],set()):
            continue
        response = read(path.with_name(path.name.replace('.request.','.response.')))
        if response['request_sha256'] != sha:
            raise ValueError('Mismatched gateway request receipt')
        model_rows.append(dict(kind=job['kind'], request_sha256=sha, elapsed_s=response['elapsed_s']))
    if {r['request_sha256'] for r in model_rows} != set.union(*bindings.values()):
        raise ValueError('Missing bound gateway calls')
    raw_s = sum(r['elapsed_s'] for r in model_rows if r['kind']=='raw')
    merge_s = sum(r['elapsed_s'] for r in model_rows if r['kind']=='merge')
    split = dict(raw_summary_calls_s=raw_s, qwen_merge_calls_s=merge_s,
        other_compilation_s=full['compile_s']-raw_s-merge_s,
        compiler_total_s=full['compile_s'], calls=model_rows,
        other_includes_attention_model_staging_hierarchy_cpu_and_gateway_wait_overhead=True)
    save(after_root/'latency-model-breakdown.json', split)
    emit(model_split=split)


if __name__ == '__main__':
    analyze()
