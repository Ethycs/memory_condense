"""Reuse established source-bound admission on actual native occurrences.

No new provider calls. Original summaries and quote diagnostics stay intact;
every hydration pointer covers the complete input fragment at its real date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_summary import BodyFragment, body_identity, materialize, summary_messages
from memory_condense.search.spine_batch_summary import RawSummaryFragment
from memory_condense.search.spine_source_admission import admit_source_bound_summaries
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.probe_native_spine_summaries import implementation


def admit(root):
    preflight=read_sealed_json(root/"preflight.json"); p=preflight.payload
    original_result=read_sealed_json(root/"result.json")
    if p["implementation"]!=implementation() or original_result.payload["preflight_sha256"]!=preflight.sha256:
        raise ValueError("completed raw probe implementation or identity changed")
    sources_root=Path(p["sources_root"])
    sources=read_sealed_json(sources_root/"sources.json")
    if sources.sha256!=p["sources_sha256"]:raise ValueError("native source plane changed")
    bank=(sources_root/sources.payload["body_bank_path"]).resolve()
    with bank.open("rb") as handle:
        if hashlib.file_digest(handle,"sha256").hexdigest()!=p["body_bank_sha256"]:
            raise ValueError("body bank changed")
    request_rows=[]
    for binding in p["requests"]:
        request=read_sealed_json(root/binding["path"])
        if request.sha256!=binding["sha256"]:raise ValueError("raw request changed")
        fragments=tuple(BodyFragment(**f) for f in request.payload["fragments"])
        if summary_messages(fragments)!=request.payload["messages"]:raise ValueError("raw messages changed")
        runtime=FastCompletionRuntime(checkpoint_dir=root/"checkpoints"/request.sha256,
            prompt_population=[request.payload["messages"]],model=p["model"],client=None,
            max_prompt_tokens=p["max_prompt_tokens"],max_new_tokens=p["max_new_tokens"],
            max_concurrency=1,retries=0,benchmark_provenance={"native_raw_request_sha256":request.sha256})
        try:batch=runtime.run()
        finally:runtime.close()
        response=batch.logical_completions[0]
        validated=read_sealed_json(root/"validated"/f"{request.sha256}.json")
        if quote_sha256(response)!=validated.payload["response_sha256"]:
            raise ValueError("original response changed")
        request_rows.append((request,fragments,response))
    wanted={r.payload["body_sha256"] for r,_,_ in request_rows}
    occurrences={sha:[] for sha in wanted}
    for binding in sources.payload["namespaces"]:
        namespace=read_sealed_json(sources_root/binding["path"])
        if namespace.sha256!=binding["sha256"]:raise ValueError("namespace changed")
        for source in namespace.payload["sessions"]:
            if source["body_sha256"] in wanted:
                if source["occurrence_id"]!=identity_sha256({k:v for k,v in source.items() if k!="occurrence_id"}):
                    raise ValueError("source occurrence changed")
                occurrences[source["body_sha256"]].append((namespace.sha256,source))
    if any(not rows for rows in occurrences.values()):raise ValueError("orphan probe body")
    files=(Path(__file__),Path("src/memory_condense/search/spine_source_admission.py"),
           Path("src/memory_condense/search/spine_batch_summary.py"),Path("src/memory_condense/search/native_spine_summary.py"))
    policy,_=publish_sealed_json(root/"source-admission-policy.json",{
        "raw_preflight_sha256":preflight.sha256,"strict_quote_result_sha256":original_result.sha256,
        "sources_sha256":sources.sha256,"mandatory_binding":"entire authenticated input fragment",
        "generated_quotes":"unchanged diagnostics; not factual evidence",
        "summary_text_changes_allowed":False,"summary_entailment_verified":False,
        "implementation":{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}})
    cached,checks=[],[]
    with sqlite3.connect(bank.as_uri()+"?mode=ro",uri=True) as conn:
        bodies={sha:json.loads(conn.execute("SELECT body_json FROM bodies WHERE body_sha256=?",(sha,)).fetchone()[0]) for sha in wanted}
    for request,fragments,response in request_rows:
        body=bodies[request.payload["body_sha256"]]
        if body_identity(body)!=request.payload["body_sha256"]:raise ValueError("body changed")
        model_rows=json.loads(response)["atoms"]
        for occurrence_number,(namespace_sha,source) in enumerate(occurrences[request.payload["body_sha256"]]):
            materialized=[]
            for fragment,row in zip(fragments,model_rows,strict=True):
                atom=materialize({"pointer":fragment.pointer(),"summary":row["summary"]},body,
                    occurrence_id=source["occurrence_id"],created_at=source["created_at"],compiler_identity=policy.sha256)
                materialized.append(RawSummaryFragment(atom.spans[0],fragment.text))
            accepted=admit_source_bound_summaries(response,materialized,compiler_identity=policy.sha256)
            for fragment,atom,diagnostic in zip(fragments,accepted.atoms,accepted.quote_diagnostics,strict=True):
                span=atom.spans[0]
                raw=body["turns"][fragment.turn_ordinal]["text"][span.start_char:span.end_char]
                if raw!=fragment.text or quote_sha256(raw)!=span.span_text_sha256 or span.created_at!=source["created_at"]:
                    raise ValueError("actual-occurrence hydration changed raw text or timestamp")
                checks.append({"namespace_sha256":namespace_sha,"occurrence_id":source["occurrence_id"],
                    "created_at":source["created_at"],"raw_span_sha256":span.receipt_sha256,
                    "request_sha256":request.sha256,"summary_sha256":quote_sha256(atom.summary)})
                if occurrence_number==0:
                    cached.append({"request_sha256":request.sha256,"pointer":fragment.pointer(),
                        "summary":atom.summary,"generated_quote_diagnostic":diagnostic})
    summary={"body_count":len(wanted),"cached_summaries":len(cached),"verified_occurrence_atoms":len(checks),
        "source_occurrences":sum(len(v) for v in occurrences.values()),
        "bodies_with_multiple_dates":sum(len({s["created_at"] for _,s in rows})>1 for rows in occurrences.values()),
        "summaries_with_quote_diagnostics":sum(bool(c["generated_quote_diagnostic"]["failures"]) for c in cached)}
    result,_=publish_sealed_json(root/"source-bound-summaries.json",{
        "policy_sha256":policy.sha256,"raw_preflight_sha256":preflight.sha256,"sources_sha256":sources.sha256,
        "cached_summaries":cached,"occurrence_checks":checks,"summary":summary,
        "new_provider_calls":0,"replay_hits":len(request_rows),"raw_inputs_to_qwen":False,
        "summary_texts_unchanged":True,"summary_entailment_verified":False,"full100_target_eligible":False})
    print(json.dumps({"source_bound_sha256":result.sha256,**summary,"new_provider_calls":0},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,required=True)
    admit(parser.parse_args().output_root)
