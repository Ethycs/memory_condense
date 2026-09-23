"""Bounded real-data validation of occurrence-independent raw summarization."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_summary import (
    BodyFragment, body_identity, fragment_body, pack_batches, parse_summaries, summary_messages,
)
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized

MODEL="codex_sdk/gpt-5.6-terra"
GATEWAY="https://central-dev.zt:4000/v1"
FILES=("tools/probe_native_spine_summaries.py", "src/memory_condense/search/native_spine_summary.py",
       "src/memory_condense/domain/_tokenizer.py", "src/memory_condense/eval/fast_completion_runtime.py",
       "tools/run_hot_reduced30_answer_judge.py")


def implementation():
    return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}


def prepare(sources_root, root):
    sources=read_sealed_json(sources_root/"sources.json")
    complete=read_sealed_json(sources_root/"complete.json")
    verified=read_sealed_json(sources_root/"verification.json")
    if (sources.sha256!="f0c5848453552bf092a142e2d7f5c1aa402eee201ddd90be31f3166d7d700f26"
            or complete.payload["sources_sha256"]!=sources.sha256
            or verified.payload["complete_sha256"]!=complete.sha256
            or verified.payload["source_plane_question_and_gold_fields_absent"] is not True):
        raise ValueError("complete verified native source plane required")
    bank=(sources_root/sources.payload["body_bank_path"]).resolve()
    with bank.open("rb") as handle:
        if hashlib.file_digest(handle,"sha256").hexdigest()!=sources.payload["body_bank_sha256"]:
            raise ValueError("native body bank changed")
    requests=[]
    with sqlite3.connect(bank.as_uri()+"?mode=ro",uri=True) as conn:
        for body_sha,raw in conn.execute("SELECT body_sha256,body_json FROM bodies ORDER BY body_sha256"):
            body=json.loads(raw)
            if body_identity(body)!=body_sha:
                raise ValueError("body identity changed")
            for batch in pack_batches(fragment_body(body)):
                messages=summary_messages(batch)
                result,_=publish_sealed_json(root/"requests"/f"{len(requests):03}.json",{
                    "sources_sha256":sources.sha256,"body_sha256":body_sha,
                    "fragments":[asdict(f) for f in batch], "messages":messages,
                    "messages_sha256":identity_sha256(messages),"model":MODEL,
                    "prompt_token_proxy":count_chat_prompt_token_proxy(messages)})
                requests.append({"path":str(result.path.relative_to(root)),"sha256":result.sha256})
                if len(requests)==8: break
            if len(requests)==8: break
    if len(requests)!=8: raise ValueError("eight-request probe population missing")
    result,_=publish_sealed_json(root/"preflight.json",{
        "format":"native-spine-date-independent-summary-probe-v1",
        "sources_root":str(sources_root.resolve()),"sources_sha256":sources.sha256,
        "body_bank_sha256":sources.payload["body_bank_sha256"],"verification_sha256":verified.sha256,
        "selection":"first eight transcript-ordered batches, bodies ordered by content hash",
        "requests":requests,"model":MODEL,"gateway":GATEWAY,"max_new_tokens":3072,
        "max_prompt_tokens":7000,"max_concurrency":4,"retries":0,"timeout_s":180,
        "maximum_provider_calls":8,"qwen_raw_inputs":False,"question_or_gold_inputs":False,
        "occurrence_metadata_in_model_input":False,"implementation":implementation(),
        "full100_target_eligible":False})
    print(json.dumps({"preflight_sha256":result.sha256,"maximum_provider_calls":8},indent=2),flush=True)
    return result


def run(root, enable_provider):
    preflight=read_sealed_json(root/"preflight.json"); p=preflight.payload
    if p["implementation"]!=implementation() or p["model"]!=MODEL or len(p["requests"])!=8:
        raise ValueError("probe implementation or population changed")
    requests=[]
    for binding in p["requests"]:
        path=(root/binding["path"]).resolve();path.relative_to(root.resolve())
        request=read_sealed_json(path); r=request.payload
        fragments=tuple(BodyFragment(**f) for f in r["fragments"])
        if (request.sha256!=binding["sha256"] or r["sources_sha256"]!=p["sources_sha256"]
                or summary_messages(fragments)!=r["messages"]
                or identity_sha256(r["messages"])!=r["messages_sha256"]):
            raise ValueError("request attribution or content changed")
        requests.append((request,fragments))
    def one(pair):
        request,fragments=pair
        def factory(client):
            return FastCompletionRuntime(checkpoint_dir=root/"checkpoints"/request.sha256,
                prompt_population=[request.payload["messages"]], model=MODEL,client=client,
                max_prompt_tokens=p["max_prompt_tokens"],max_new_tokens=p["max_new_tokens"],
                max_concurrency=1,retries=0,benchmark_provenance={"native_raw_request_sha256":request.sha256})
        audit=factory(None)
        try: remaining=1-len(_authenticated_records(audit))
        finally: audit.close()
        batch,calls,hits,elapsed=_run_exactly_authorized(runtime_factory=factory,
            authorized_provider_calls=remaining,enable_provider=enable_provider,
            client_factory=lambda:_completion_client("LITELLM_KEY",GATEWAY).with_options(timeout=p["timeout_s"],max_retries=0))
        try:
            summaries=parse_summaries(batch.logical_completions[0],fragments)
            validation={"status":"accepted","summaries":summaries}
        except (ValueError,TypeError,KeyError) as exc:
            validation={"status":"invalid_summary","summaries":[],"error_type":type(exc).__name__,"error":str(exc)}
        result,_=publish_sealed_json(root/"validated"/f"{request.sha256}.json",{
            "preflight_sha256":preflight.sha256,"request_sha256":request.sha256,
            "response_sha256":hashlib.sha256(batch.logical_completions[0].encode()).hexdigest(),**validation})
        print(json.dumps({"request":request.path.name,"status":validation["status"],
            "summaries":len(validation["summaries"]),"new_calls":calls,"replay_hits":hits}),flush=True)
        return result,calls,hits
    with ThreadPoolExecutor(max_workers=p["max_concurrency"]) as pool:
        outputs=list(pool.map(one,requests))
    result,_=publish_sealed_json(root/"result.json",{
        "preflight_sha256":preflight.sha256,"validated_sha256s":[r.sha256 for r,_,_ in outputs],
        "accepted_batches":sum(r.payload["status"]=="accepted" for r,_,_ in outputs),
        "accepted_summaries":sum(len(r.payload["summaries"]) for r,_,_ in outputs),
        "input_fragments":sum(len(f) for _,f in requests),
        "qwen_calls":0,"raw_inputs_to_qwen":False,"full100_target_eligible":False,
        "semantic_fidelity_certified":False})
    print(json.dumps({"result_sha256":result.sha256,"accepted_batches":result.payload["accepted_batches"],
        "accepted_summaries":result.payload["accepted_summaries"],"new_calls":sum(c for _,c,_ in outputs),
        "replay_hits":sum(h for _,_,h in outputs)},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","run"))
    parser.add_argument("--sources-root",type=Path,default=Path("eval_results/native-spine-complete-sources-20260910-r1"))
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--enable-provider",action="store_true")
    args=parser.parse_args()
    if args.phase=="prepare":prepare(args.sources_root,args.output_root)
    else:run(args.output_root,args.enable_provider)
