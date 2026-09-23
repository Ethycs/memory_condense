"""Prepare and execute complete native-body summary compilation, with a probe mode.

Source and occurrence metadata stay local. Qwen cannot be the raw compiler.
Every original fragment is covered; incomplete runs cannot admit a namespace.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256, quote_sha256
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import admit, messages, pack, restore
from memory_condense.search.native_spine_summary import BodyFragment, body_identity, fragment_body
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json
from tools.run_hot_reduced30_answer_judge import _authenticated_records, _completion_client, _run_exactly_authorized

MODELS={"codex_sdk/gpt-5.6-terra","claude-haiku-4-5"}
GATEWAY="https://central-dev.zt:4000/v1"
FILES=("tools/compile_native_spine.py","src/memory_condense/search/native_spine_batch.py",
       "src/memory_condense/search/native_spine_summary.py","src/memory_condense/domain/_tokenizer.py",
       "src/memory_condense/eval/fast_completion_runtime.py","tools/run_hot_reduced30_answer_judge.py")


def implementation():
    return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}


def add_pointer(digest,fragment):
    digest.update((canonical_json(fragment.pointer())+"\n").encode("utf-8"))


def prepare(sources_root,root,models,probe_root=None):
    if not models or len(set(models))!=len(models) or any(m not in MODELS for m in models):
        raise ValueError("an explicitly supported non-Qwen raw model is required")
    if probe_root is None and len(models)!=1:
        raise ValueError("full compilation requires one declared compiler model")
    sources=read_sealed_json(sources_root/"sources.json")
    verified=read_sealed_json(sources_root/"verification.json")
    if (sources.sha256!="f0c5848453552bf092a142e2d7f5c1aa402eee201ddd90be31f3166d7d700f26"
            or verified.payload["sources_sha256"]!=sources.sha256
            or verified.payload["source_plane_question_and_gold_fields_absent"] is not True):
        raise ValueError("verified native source plane required")
    bank=(sources_root/sources.payload["body_bank_path"]).resolve()
    with bank.open("rb") as handle:
        if hashlib.file_digest(handle,"sha256").hexdigest()!=sources.payload["body_bank_sha256"]:
            raise ValueError("native bank changed")
    body_shas=set();input_digest=hashlib.sha256();output_digest=hashlib.sha256();input_count=0
    parent=None
    if probe_root is not None:
        parent=read_sealed_json(probe_root/"preflight.json")
        if parent.payload["sources_sha256"]!=sources.sha256:
            raise ValueError("probe source corpus changed")
    def inputs():
        nonlocal input_count
        if parent is not None:
            for binding in parent.payload["requests"]:
                request=read_sealed_json(probe_root/binding["path"])
                if request.sha256!=binding["sha256"]:raise ValueError("probe raw request changed")
                for value in request.payload["fragments"]:
                    fragment=BodyFragment(**value);body_shas.add(fragment.body_sha256)
                    input_count+=1;add_pointer(input_digest,fragment);yield fragment
        else:
            with sqlite3.connect(bank.as_uri()+"?mode=ro",uri=True) as conn:
                for body_sha,raw in conn.execute("SELECT body_sha256,body_json FROM bodies ORDER BY body_sha256"):
                    body=json.loads(raw)
                    if body_identity(body)!=body_sha:raise ValueError("body identity changed")
                    body_shas.add(body_sha)
                    for fragment in fragment_body(body):
                        input_count+=1;add_pointer(input_digest,fragment);yield fragment
    requests=[];output_count=0
    for index,fragments in enumerate(pack(inputs())):
        prompt=messages(fragments)
        value,_=publish_sealed_json(root/"requests"/f"{index:06}.json",{
            "sources_sha256":sources.sha256,"ordinal":index,"messages":prompt,
            "pointers":[f.pointer() for f in fragments],"messages_sha256":identity_sha256(prompt),
            "prompt_token_proxy":count_chat_prompt_token_proxy(prompt)})
        requests.append({"path":str(value.path.relative_to(root)),"sha256":value.sha256,"atoms":len(fragments)})
        for fragment in fragments:output_count+=1;add_pointer(output_digest,fragment)
        if len(requests)%256==0:print({"prepared_batches":len(requests),"bodies":len(body_shas),"fragments":output_count},flush=True)
    if not requests or input_count!=output_count or input_digest.digest()!=output_digest.digest():
        raise ValueError("complete fragment stream was not preserved")
    if parent is None and len(body_shas)!=sources.payload["body_count"]:
        raise ValueError("complete source body population missing")
    result,_=publish_sealed_json(root/"preflight.json",{
        "format":"native-spine-complete-body-summary-compiler-v1","sources_root":str(sources_root.resolve()),
        "sources_sha256":sources.sha256,"source_verification_sha256":verified.sha256,
        "body_bank_sha256":sources.payload["body_bank_sha256"],"mode":"probe" if parent else "full",
        "probe_preflight_sha256":parent.sha256 if parent else None,"models":models,"gateway":GATEWAY,
        "requests":requests,"body_count":len(body_shas),"fragment_count":input_count,
        "ordered_pointer_sha256":input_digest.hexdigest(),"max_prompt_tokens":7000,"max_new_tokens":4096,
        "max_atoms":24,"concurrency":4 if parent else 8,"retries":0,"timeout_s":240,
        "maximum_initial_provider_calls":len(requests)*len(models),"implementation":implementation(),
        "raw_inputs_to_qwen":False,"question_or_gold_inputs":False,"summary_entailment_verified":False,
        "entire_input_fragments_are_hydration_targets":True,"generated_support_quotes_requested":False})
    print({"preflight_sha256":result.sha256,"batches":len(requests),"fragments":input_count,
           "bodies":len(body_shas),"maximum_initial_provider_calls":result.payload["maximum_initial_provider_calls"]},flush=True)


def execute(root,enable_provider):
    preflight=read_sealed_json(root/"preflight.json");p=preflight.payload
    if p["implementation"]!=implementation() or any(m not in MODELS for m in p["models"]):
        raise ValueError("compiler implementation or raw models changed")
    loaded=[];digest=hashlib.sha256();count=0
    for binding in p["requests"]:
        request=read_sealed_json(root/binding["path"])
        if request.sha256!=binding["sha256"] or request.payload["sources_sha256"]!=p["sources_sha256"]:
            raise ValueError("prepared request binding changed")
        fragments=restore(request.payload)
        if identity_sha256(request.payload["messages"])!=request.payload["messages_sha256"]:
            raise ValueError("prepared prompt hash changed")
        if len(fragments)!=binding["atoms"]:raise ValueError("fragment count changed")
        for fragment in fragments:count+=1;add_pointer(digest,fragment)
        loaded.append((request,fragments))
    if count!=p["fragment_count"] or digest.hexdigest()!=p["ordered_pointer_sha256"]:
        raise ValueError("compiler must preserve every prepared fragment")
    jobs=[(pair,model) for i,pair in enumerate(loaded)
          for model in (p["models"] if i%2==0 else list(reversed(p["models"])))]
    started=time.perf_counter()
    def one(job):
        (request,fragments),model=job
        key=identity_sha256({"request_sha256":request.sha256,"model":model})
        def factory(client):
            return FastCompletionRuntime(checkpoint_dir=root/"checkpoints"/key,
                prompt_population=[request.payload["messages"]],model=model,client=client,
                max_prompt_tokens=p["max_prompt_tokens"],max_new_tokens=p["max_new_tokens"],
                max_concurrency=1,retries=0,benchmark_provenance={"native_compile_request_sha256":request.sha256})
        audit=factory(None)
        try:remaining=1-len(_authenticated_records(audit))
        finally:audit.close()
        batch,calls,hits,_=_run_exactly_authorized(runtime_factory=factory,
            authorized_provider_calls=remaining,enable_provider=enable_provider,
            client_factory=lambda:_completion_client("LITELLM_KEY",GATEWAY).with_options(timeout=p["timeout_s"],max_retries=0))
        response=batch.logical_completions[0]
        try:validation={"status":"accepted","summaries":admit(response,fragments)}
        except (ValueError,TypeError,KeyError) as exc:
            validation={"status":"invalid_summary","summaries":[],"error_type":type(exc).__name__,"error":str(exc)}
        result,_=publish_sealed_json(root/"validated"/f"{key}.json",{
            "preflight_sha256":preflight.sha256,"request_sha256":request.sha256,"model":model,
            "response_sha256":quote_sha256(response),**validation})
        print({"batch":request.payload["ordinal"],"model":model,"status":validation["status"],
               "atoms":len(validation["summaries"]),"new_calls":calls,"replay_hits":hits},flush=True)
        return result,calls,hits
    with ThreadPoolExecutor(max_workers=p["concurrency"]) as pool:outputs=list(pool.map(one,jobs))
    model_results={}
    for model in p["models"]:
        rows=[r for r,_,_ in outputs if r.payload["model"]==model]
        accepted=sum(r.payload["status"]=="accepted" for r in rows)
        atoms=sum(len(r.payload["summaries"]) for r in rows)
        complete=p["mode"]=="full" and accepted==len(loaded) and atoms==p["fragment_count"]
        model_results[model]={"accepted_batches":accepted,"batches":len(rows),"accepted_atoms":atoms,
            "complete_source_compilation":complete,"validated_sha256s":[r.sha256 for r in rows]}
    result,_=publish_sealed_json(root/"result.json",{
        "preflight_sha256":preflight.sha256,"models":model_results,"hierarchies_compiled":False,
        "raw_inputs_to_qwen":False,"summary_entailment_verified":False,"full100_target_passed":False})
    print({"result_sha256":result.sha256,"models":{m:{k:v for k,v in r.items() if k!="validated_sha256s"}
        for m,r in model_results.items()},"new_calls":sum(c for _,c,_ in outputs),
        "replay_hits":sum(h for _,_,h in outputs),"invocation_seconds":time.perf_counter()-started},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","run"))
    parser.add_argument("--sources-root",type=Path,default=Path("eval_results/native-spine-complete-sources-20260910-r1"))
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--model",action="append")
    parser.add_argument("--probe-root",type=Path)
    parser.add_argument("--enable-provider",action="store_true")
    args=parser.parse_args()
    if args.phase=="prepare":prepare(args.sources_root,args.output_root,args.model,args.probe_root)
    else:execute(args.output_root,args.enable_provider)
