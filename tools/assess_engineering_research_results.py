"""Offline diagnostics separating artifact quality, exact citation syntax and limits.

Never changes the frozen rubric, actor results or original strict task scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from tools.engineering_research_gateway import read, save
from tools import engineering_research_battery as battery


def accounted_usage(response):
    """Keep zero/missing gateway counters distinct from estimated token costs."""
    import tiktoken
    reported = response.get('usage') or {}
    prompt = reported.get('prompt_tokens', 0) or 0
    output = reported.get('completion_tokens', 0) or 0
    return {
        'prompt_tokens': prompt or response.get('prompt_tokens_proxy', 0),
        'completion_tokens': output or len(tiktoken.get_encoding('cl100k_base').encode(response.get('content', ''), disallowed_special=())),
        'prompt_estimated': not bool(prompt),
        'completion_estimated': not bool(output),
    }


def visible_text(text):
    return text.replace('**', '').replace('`', '')


def normalized_with_offsets(text):
    """Render emphasis/list markers and whitespace; retain exact source offsets."""
    output, offsets, position = [], [], 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^\s*(?:[-*+]|\d+[.)])\s+',line)
        i = marker.end() if marker else 0
        while i < len(line):
            math_delimiter = re.match(r'\\+[()\[\]]', line[i:])
            if math_delimiter:
                i += math_delimiter.end()
                continue
            if line.startswith('**',i):
                i += 2
                continue
            if line[i] == '`':
                i += 1
                continue
            char = ' ' if line[i].isspace() else line[i]
            if char != ' ' or not output or output[-1] != ' ':
                output.append(char)
                offsets.append(position+i)
            i += 1
        position += len(line)
    return ''.join(output), offsets


def exact_format_equivalent(quote,text):
    normalized, offsets = normalized_with_offsets(text)
    query,_ = normalized_with_offsets(quote)
    query=query.strip()
    if not query:
        return None
    start=normalized.find(query)
    if start < 0:
        return None
    return text[offsets[start]:offsets[start+len(query)-1]+1]


def recover_review(review,rubric,candidates,actor,manual_quotes=None):
    from tools.run_engineering_research_battery import parse_json,validate_review
    if review['valid']:
        return review['review'],[]
    changes=[]
    try:
        value=parse_json(review['response']['content'])
        source_ids={r['turn_id'] for r in actor['history']} | {actor['current_turn_id']}
        for label,candidate in value['candidates'].items():
            for row in candidate['criteria']:
                for index, source_id in enumerate(row.get('source_turn_ids', [])):
                    if source_id not in source_ids and source_id.startswith('family:') and source_id[7:] in source_ids:
                        changes.append(dict(criterion=row['id'],candidate=label,
                                            original_source_id=source_id,exact_source_id=source_id[7:]))
                        row['source_turn_ids'][index]=source_id[7:]
                quote=row.get('artifact_quote','')
                text=candidates[label].get(row.get('artifact_file'),'')
                if quote and quote not in text:
                    exact=exact_format_equivalent(quote,text)
                    manual=(manual_quotes or {}).get((label,row['id']))
                    if exact is None and manual and manual['original_quote']==quote and manual['artifact_file']==row.get('artifact_file'):
                        exact=manual['exact_artifact_quote']
                        if exact not in text:
                            raise ValueError('Manual evidence correction is not an exact artifact span')
                    if exact is None:
                        return None,[]
                    changes.append(dict(criterion=row['id'],candidate=label,file=row['artifact_file'],
                                        original_quote=quote,exact_artifact_quote=exact,
                                        manual_evidence_inspection=bool(manual)))
                    row['artifact_quote']=exact
        validate_review(value,rubric,candidates,actor)
        return value,changes
    except (ValueError,TypeError,KeyError):
        return None,[]


def citation_audit(actor, result):
    if actor['domain'] != 'research':
        return None
    source = {r['turn_id']:r['text'] for r in actor['history']}
    source[actor['current_turn_id']] = actor['original_request']
    try:
        claims = json.loads(result['artifacts']['claims.json'])
    except (KeyError, ValueError):
        return {'format_error': True}
    if not isinstance(claims,list):
        return {'format_error': True}
    findings=[]
    for index,claim in enumerate(claims):
        for citation in claim.get('evidence',[]):
            quote=citation.get('quote','')
            text=source.get(citation.get('turn_id'),'')
            if quote and quote in text:
                kind='exact'
            elif quote and visible_text(quote) in visible_text(text):
                kind='markdown_emphasis_only'
            else:
                kind='requires_source_review'
            findings.append(dict(claim_index=index,classification=kind,**citation))
    return {'counts':dict(Counter(f['classification'] for f in findings)), 'citations':findings,
            'semantic_entailment_certified':False, 'strict_score_changed':False}


def assess(root):
    plan=read(root/'run-plan.json')
    manual_repairs=[]
    for path in sorted((root/'diagnostics').glob('review-evidence-corrections-*.json')):
        manual_repairs.extend(read(path)['repairs'])
    adjudications={}
    for path in sorted((root/'diagnostics').glob('manual-adjudications-*.json')):
        for row in read(path)['rows']:
            key=(row['case_id'],row['arm'],row['criterion'])
            if key in adjudications:
                raise ValueError('Duplicate manual adjudication')
            adjudications[key]=row
    cases=[]
    for record in plan['cases']:
        folder=root/'cases'/record['id']
        if not (folder/'grading/reviews.json').exists():
            continue
        actor=battery.read_binding(record['actor'])
        rubric=battery.read_binding(record['rubric'])
        delivered=[]
        for path in (folder/'memory').glob('*/reopened.json'):
            if path.parent.name!='final':
                delivered.append(read(path).get('text',''))
        reviews=read(folder/'grading/reviews.json')['reviews']
        results={arm:read(folder/arm/'result.json') for arm in ('memory','full_context')}
        recovered=[]
        for review in reviews:
            candidates={label:results[arm]['artifacts'] for label,arm in review['mapping'].items()}
            corrections={}
            for repair in manual_repairs:
                if repair['case_id']==record['id'] and repair['request_sha256']==review['response']['request_sha256']:
                    result_path=folder/repair['arm']/'result.json'
                    if hashlib.sha256(result_path.read_bytes()).hexdigest()!=repair['result_sha256']:
                        raise ValueError('Manually inspected artifact changed')
                    label=next(k for k,v in review['mapping'].items() if v==repair['arm'])
                    corrections[(label,repair['criterion'])]=repair
            value,changes=recover_review(review,rubric,candidates,actor,corrections)
            recovered.append(dict(mapping=review['mapping'],review=value,quote_repairs=changes))
        arms={}
        for arm in ('memory','full_context'):
            result=results[arm]
            criteria=[]
            for criterion in rubric['criteria']:
                judgments=[]
                for review in recovered:
                    if review['review'] is not None:
                        label=next(k for k,v in review['mapping'].items() if v==arm)
                        judgments.append(next(r for r in review['review']['candidates'][label]['criteria'] if r['id']==criterion['id']))
                scores=[r['score'] for r in judgments]
                # An unrelated bad quotation must not silently erase a disagreeing
                # review and let the more favorable review determine the score.
                recorded=[]
                from tools.run_engineering_research_battery import parse_json
                for review in reviews:
                    try:
                        label=next(k for k,v in review['mapping'].items() if v==arm)
                        rows=parse_json(review['response']['content'])['candidates'][label]['criteria']
                        rows=[r for r in rows if r['id']==criterion['id']]
                        if len(rows)==1 and type(rows[0]['score']) is int and rows[0]['score'] in (0,1,2):
                            recorded.append(rows[0]['score'])
                    except (ValueError,TypeError,KeyError):
                        pass
                score=scores[0] if scores and all(v==scores[0] for v in scores+recorded) else None
                adjudication=adjudications.get((record['id'],arm,criterion['id']))
                if adjudication and hashlib.sha256((folder/arm/'result.json').read_bytes()).hexdigest()!=adjudication['result_sha256']:
                    raise ValueError('Adjudicated result changed')
                criteria.append(dict(id=criterion['id'],agreed_score=score,judgments=judgments,
                                     source_reviewed_score=adjudication['adjudicated_score'] if adjudication else score,
                                     manual_adjudication=adjudication,
                                     all_recorded_scores_including_unverified_reviews=recorded,
                                     recorded_support_delivery=[dict(turn_id=s['turn_id'],quote=s['quote'],
                                         current_request=s['turn_id']==actor['current_turn_id'],
                                         exact_quote_in_memory_evidence=any(s['quote'] in t for t in delivered),
                                         available_in_full_source=True) for s in criterion.get('supports',[])]))
            semantic='unresolved' if any(c['agreed_score'] is None for c in criteria) else (
                'met' if all(c['agreed_score']==2 for c in criteria) else 'not_fully_met')
            reviewed='unresolved' if any(c['source_reviewed_score'] is None for c in criteria) else (
                'met' if all(c['source_reviewed_score']==2 for c in criteria) else 'not_fully_met')
            citation=citation_audit(actor,result)
            structural=result['structural']['structurally_complete']
            required_files=all(name in result['artifacts'] for name in actor['deliverables'])
            mechanical=all(result[k]['exit_code']==0 for k in ('unit_tests','behavioral_checks') if k in result)
            lifecycle=arm!='memory' or bool(result.get('final_reopen'))
            arms[arm]=dict(semantic_criteria=semantic,criteria=criteria,structural=structural,
                source_reviewed_criteria=reviewed,
                required_artifacts_present=required_files,
                mechanical=mechanical,finished=result['finished'],citation_audit=citation,
                lifecycle_verified=lifecycle,
                artifact_quality_met=semantic=='met' and required_files and mechanical and result['finished'])
        cases.append(dict(case_id=record['id'],domain=record['domain'],family=record['family'],arms=arms,
            original_valid_reviews=sum(r['valid'] for r in reviews),
            format_verified_reviews=sum(r['review'] is not None for r in recovered),
            judge_quote_repairs=[c for r in recovered for c in r['quote_repairs']]))
    counts=Counter()
    usage=Counter()
    accounted=Counter()
    estimated=Counter()
    limits=[]
    models=Counter()
    errors=[]
    case_usage={}
    for path in (root/'gateway').glob('*.response.json'):
        response=read(path)
        request=read(path.with_name(path.name.replace('.response.json','.request.json')))
        kind=request['kind']
        group=request['scope'].split('/')[1] if kind=='actor' else kind
        counts[kind]+=1
        reservation_path=path.with_name(path.name.replace('.response.json','.reservation.json'))
        reservation=read(reservation_path) if reservation_path.exists() else None
        usage_response=dict(response)
        if reservation and not usage_response.get('prompt_tokens_proxy'):
            usage_response['prompt_tokens_proxy']=reservation['prompt_tokens']
        effective=accounted_usage(usage_response)
        case_id=request['scope'].split('/')[0]
        counter=case_usage.setdefault(case_id,Counter())
        counter[group+':responses']+=1
        for key in ('prompt_tokens','completion_tokens'):
            counter[group+':'+key]+=effective[key]
        if response.get('error_type'):
            errors.append(dict(scope=request['scope'],error_type=response['error_type'],
                               reserved=bool(reservation),actual_usage_unknown=True))
        for key in ('prompt_tokens','completion_tokens'):
            accounted[group+':'+key]+=effective[key]
        for key in ('prompt_estimated','completion_estimated'):
            estimated[group+':'+key]+=int(effective[key])
        if response.get('usage'):
            for key in ('prompt_tokens','completion_tokens'):
                usage[group+':'+key]+=response['usage'].get(key,0)
            if response['usage'].get('completion_tokens',0)>request['max_tokens']:
                limits.append(dict(scope=request['scope'],requested=request['max_tokens'],
                    observed=response['usage']['completion_tokens'],type='gateway_output_limit_exceeded'))
        models[response.get('response_model','error_or_unknown')]+=1
    memory_input=sum(accounted[k+':prompt_tokens'] for k in ('memory','raw','merge'))
    direct_input=accounted['full_context:prompt_tokens']
    report=dict(completed_pairs=len(cases),planned_pairs=len(plan['cases']),cases=cases,
        original_strict_reports_unchanged=True,provider_response_counts=dict(counts),provider_usage=dict(usage),
        accounted_usage=dict(accounted),estimated_usage_call_counts=dict(estimated),
        accounted_usage_by_case={k:dict(v) for k,v in case_usage.items()},provider_errors=errors,
        response_model_counts=dict(models),protocol_deviations=limits,
        quality_counts={arm:dict(Counter(c['arms'][arm]['semantic_criteria'] for c in cases)) for arm in ('memory','full_context')},
        source_reviewed_quality_counts={arm:dict(Counter(c['arms'][arm]['source_reviewed_criteria'] for c in cases)) for arm in ('memory','full_context')},
        token_accounting=dict(memory_actor_plus_ingestion_input=memory_input,direct_actor_input=direct_input,
            input_reduction_fraction=1-memory_input/direct_input if direct_input else None,
            missing_provider_usage_uses_cl100k_base_proxy=True,
            token_estimates_are_not_provider_billing_measurements=True,
            failed_reserved_calls_include_intended_prompt_payload_only=True,
            judge_usage_excluded_from_operating_comparison=True,
            includes_in_progress_case_usage=len(cases)!=len(plan['cases']),
            shared_compilation_counted_once=True),
        interpretation='Citation formatting diagnostics do not certify scientific truth or retrospectively change original strict reports. '
                       'Semantic quality, executed behavior, quotation fidelity and gateway protocol compliance are separate observations.')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    report=assess(args.run)
    path=args.run/f"assessment-{report['completed_pairs']:02d}.json"
    save(path,report)
    print(json.dumps(dict(path=str(path),completed_pairs=report['completed_pairs'],quality_counts=report['quality_counts'],
                          protocol_deviations=len(report['protocol_deviations']))))


if __name__=='__main__':
    main()
