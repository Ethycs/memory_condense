"""Record bounded model results and source-bound manual fidelity findings."""
from pathlib import Path
from tools.engineering_research_gateway import read,save,emit
from memory_condense.domain._discourse_identity import identity_sha256


def main():
    root=Path('eval_results/small-summarizer-20260929-r1')
    summaries=[]
    for name in ('transcript-bounded','small-bounded'):
        report=read(root/name/'report.json')
        rows=[read(root/name/f'bounded-{i}.json') for i in range(4)]
        summaries.append(dict(configuration=name,model=report['model'],valid=report['valid'],
            total=report['total'],wall_s=report['wall_s'],mean_s=report['mean_request_s'],
            new_engineering_mean_s=sum(r['elapsed_s'] for r in rows[:3])/3,
            long_saved_fragment_s=rows[3]['elapsed_s'],peak_working_set_bytes=report['memory']['peak_wset'],
            result_sha256=[identity_sha256(r) for r in rows]))
    small=[read(root/'small-bounded'/f'bounded-{i}.json') for i in range(3)]
    required=sum(len(r['identifier_coverage'][0]['required']) for r in small)
    missing=sum(len(r['identifier_coverage'][0]['missing']) for r in small)
    transcript=read(root/'transcript-bounded/bounded-3.json')
    original=read(root/'transcript-bounded/plan.json')['jobs'][3]['fragments'][0]['text']
    assert '"cluster":"saffron-842","migration_id":"arden-mig"' in original
    assert '"cluster":"cobalt-731","migration_id":"boreal-mig"' in original
    assert 'Arden deployment, use cluster saffron-842 and migration identifier boreal-mig' in transcript['parsed'][0]['summary']
    result=dict(results=summaries,scope='three novel synthetic engineering fragments plus one 2048-token saved raw fragment',
        direct_same_prompt_model_ablation=False,
        differences='Transcript uses the documented meeting template and temperature 0.3; small uses the existing raw system prompt and temperature 0. Both use CPU Q4_K_M, one slot, six threads and the same exact fragment text.',
        manual_fidelity_findings=[
            dict(model='LFM2-2.6B-Transcript',sample='bounded-3',
                finding='Summary incorrectly assigns Boreal migration boreal-mig to Arden; exact raw text assigns arden-mig to Arden.',
                response_sha256=identity_sha256(transcript)),
            dict(model='LFM2.5-1.2B-Instruct',samples=['bounded-0','bounded-1','bounded-2'],
                finding='Routing summaries are generic topic labels; exact support quotes often contain details absent from the indexed summary.',
                required_identifiers=required,missing_identifiers=missing,retained_identifiers=required-missing)],
        promotion=False,default_raw_model_unchanged=True,
        limitation='Small fidelity screen, not an end-to-end recall accuracy score or throughput guarantee.',
        hosted_control=read(root/'hosted-availability.json'),
        abandoned_configuration=read(root/'transcript/stopped.json'))
    save(root/'assessment.json',result)
    emit(**result)


if __name__=='__main__': main()
