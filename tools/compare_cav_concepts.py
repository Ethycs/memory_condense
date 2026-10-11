"""Compare two prepared CAV-reinjection arms without reader calls."""

import argparse
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def complete_support(rows):
    return sum(value['complete'] for row in rows for value in row['support_quotes'].values())


def quote_support(rows):
    values=[value for row in rows for value in row['support_quotes'].values()]
    return sum(value['quotes'] for value in values),sum(value['total_quotes'] for value in values)


def summarize(root):
    rows=read(root/'retrieval-report.json')['records']
    quotes,total_quotes=quote_support(rows)
    return dict(recall_s=mean(row['recall_s'] for row in rows),
        baseline_recall_s=mean(row['baseline_recall_s'] for row in rows),
        incremental_recall_s=mean(row['incremental_recall_s'] for row in rows),
        selector_ms=1000*mean(row['query_compute_s'] for row in rows),
        packet_tokens=mean(row['packet_tokens'] for row in rows),
        additions=mean(row['added_sections'] for row in rows),
        complete_support_checkpoints=complete_support(rows),
        support_quotes=quotes,total_support_quotes=total_quotes,
        state=read(root/'state.json'))


def selected(root,qid):
    plan=read(root/'plan.json')
    arm=plan.get('experimental_arm') or next(value for value in plan['arms'] if value!='baseline')
    packet=read(root/'packets'/qid/(arm+'.json'))
    return packet['retrieval']['experiments']['delivered_section_ids']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--treatment',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    questions=[row['id'] for row in read(args.treatment/'retrieval-report.json')['records']]
    changes=[]
    for qid in questions:
        before=selected(args.control,qid); after=selected(args.treatment,qid)
        union=set(before)|set(after)
        changes.append(dict(id=qid,same=before==after,
            jaccard=len(set(before)&set(after))/len(union) if union else 1.,
            control=before,treatment=after))
    report=dict(control=summarize(args.control),treatment=summarize(args.treatment),
        identical_selections=sum(row['same'] for row in changes),questions=len(changes),
        mean_selection_jaccard=mean(row['jaccard'] for row in changes),changes=changes)
    value=json.dumps(report,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(value+'\n',encoding='utf-8')
    print(value)


if __name__=='__main__':
    main()
