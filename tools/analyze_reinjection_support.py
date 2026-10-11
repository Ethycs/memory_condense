"""Measure exact source-support gains from prepared experimental packets."""

import argparse
import json
from pathlib import Path

from tools.characterize_native_experiments import cases,covered_quote


PRIOR=Path('eval_results/native-experiment-quality-20261008-r1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def coverage(packet,question,reference):
    result={}
    for fid in question['facts']:
        supports=reference['facts'][fid]['supports']
        result[fid]=tuple(covered_quote(value,packet['references']) for value in supports)
    return result


def counts(rows):
    values=[flags for row in rows.values() for flags in row.values()]
    return dict(complete_checkpoints=sum(all(flags) for flags in values),
        support_quotes=sum(sum(flags) for flags in values),
        total_checkpoints=len(values),total_support_quotes=sum(len(flags) for flags in values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',action='append',required=True,metavar='NAME=ROOT')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    questions=cases(); reference=read(PRIOR/'private-reference.json')
    baseline={q['id']:coverage(read(PRIOR/'packets'/q['id']/'baseline.json'),q,reference)
              for q in questions}
    report={'baseline':counts(baseline),'arms':{}}
    for raw in args.arm:
        name,separator,root_value=raw.partition('=')
        if not separator or not name or not root_value: parser.error('--arm must be NAME=ROOT')
        root=Path(root_value); plan=read(root/'plan.json')
        arm=plan.get('experimental_arm') or next(value for value in plan['arms'] if value!='baseline')
        observed={q['id']:coverage(read(root/'packets'/q['id']/(arm+'.json')),q,reference)
                  for q in questions}
        novel=lost=0; per_question=[]
        for q in questions:
            gained=missing=0
            for fid,flags in observed[q['id']].items():
                before=baseline[q['id']][fid]
                gained+=sum(after and not old for after,old in zip(flags,before,strict=True))
                missing+=sum(old and not after for after,old in zip(flags,before,strict=True))
            novel+=gained; lost+=missing
            per_question.append(dict(id=q['id'],novel_support_quotes=gained,lost_support_quotes=missing))
        report['arms'][name]=dict(**counts(observed),novel_support_quotes=novel,
            lost_support_quotes=lost,questions=per_question)
    value=json.dumps(report,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(value+'\n',encoding='utf-8')
    print(value)


if __name__=='__main__': main()
