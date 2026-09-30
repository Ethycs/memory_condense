"""Offline quote-format diagnostic; never changes the live run or its caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search import native_spine_summary as raw
from tools.engineering_research_gateway import read, save
from tools.native_spine_engineering_session import repair_raw_support


def visible_with_offsets(text):
    chars, offsets, base = [], [], 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^\s*\[L\d+\]\s?', line)
        i = marker.end() if marker else 0
        while i < len(line):
            if line.startswith('**', i):
                i += 2
                continue
            delimiter = re.match(r'\\+[()\[\]]', line[i:])
            if delimiter:
                i += delimiter.end()
                continue
            if line[i] == '`':
                i += 1
                continue
            char = ' ' if line[i].isspace() else line[i]
            if char != ' ' or not chars or chars[-1] != ' ':
                chars.append(char)
                offsets.append(base + i)
            i += 1
        base += len(line)
    return ''.join(chars), offsets


def recover_exact_span(quote, text):
    visible, offsets = visible_with_offsets(text)
    target = visible_with_offsets(quote)[0].strip()
    start = visible.find(target) if target else -1
    if start < 0 or visible.rfind(target) != start:
        return None
    end = start + len(target)
    return text[offsets[start]:offsets[end-1]+1]


def diagnose(root):
    findings = []
    for path in sorted((root/'gateway').glob('*.request.json')):
        job = read(path)
        if job['kind'] != 'raw' or job['scope'].split('/')[0] not in {'R01','R02','R03'}:
            continue
        response_path = path.with_name(path.name.replace('.request.json','.response.json'))
        if not response_path.exists():
            continue
        response = read(response_path)
        if not response.get('content'):
            continue
        entries = json.loads(job['messages'][1]['content'])['fragments']
        # Content validation fixture only. No occurrence pointers are installed.
        digest = identity_sha256(entries)
        fragments = tuple(raw.BodyFragment(digest, i, row['speaker'], 0,
            len(row['fragment']), quote_sha256(row['fragment']), row['fragment'])
            for i,row in enumerate(entries))
        try:
            original,_ = repair_raw_support(response['content'], fragments)
            raw.parse_summaries(original, fragments)
            continue
        except ValueError as exc:
            original_error = str(exc)
        value = json.loads(original)
        before_summaries = [row.get('summary') for row in value.get('atoms',[])]
        changes = []
        for row,fragment in zip(value.get('atoms',[]),fragments):
            recovered = []
            for quote in row.get('support',[]):
                exact = quote if quote in fragment.text else recover_exact_span(quote, fragment.text)
                if exact is None:
                    recovered = []
                    break
                parts = [exact] if count_tokens(exact)<=32 else [f.text for f in raw.fragment_body(
                    {'turns':[{'role':'system','text':exact}]}, token_cap=32)]
                recovered.extend(parts)
            if recovered and len(recovered)<=4 and recovered!=row['support']:
                changes.append(dict(label=row['label'],original_support=row['support'],
                    exact_support=recovered,fragment_sha256=quote_sha256(fragment.text)))
                row['support']=recovered
        try:
            raw.parse_summaries(json.dumps(value,ensure_ascii=False), fragments)
            repaired_valid, error = True, None
        except ValueError as exc:
            repaired_valid, error = False, str(exc)
        assert before_summaries == [row.get('summary') for row in value.get('atoms',[])]
        findings.append(dict(scope=job['scope'],request_sha256=response['request_sha256'],
            original_error=original_error,repaired_valid=repaired_valid,remaining_error=error,
            changes=changes,summaries_unchanged=True))
    return dict(live_policy_changed=False,caches_changed=False,new_provider_calls=0,
        scope='Saved failed raw-summary batches only; not a rerun or new task-quality score.',
        failed_batches=len(findings),format_recoverable_batches=sum(r['repaired_valid'] for r in findings),
        findings=findings)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args()
    report=diagnose(args.run)
    save(args.run/'diagnostics/raw-quote-format-repair/report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='findings'}))


if __name__=='__main__':
    main()
