"""Authenticate the completed Terra arm without retrying unavailable backends."""
import argparse
import hashlib
import json
from pathlib import Path

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from memory_condense.search.native_spine_batch import admit, restore
from tools.compile_native_spine import implementation
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def report(root,previous_root):
    preflight=read_sealed_json(root/"preflight.json");p=preflight.payload
    if p["mode"]!="probe" or p["implementation"]!=implementation():
        raise ValueError("probe mode or implementation changed")
    old=read_sealed_json(previous_root/"preflight.json")
    if old.sha256!=p["probe_preflight_sha256"]:
        raise ValueError("the compared input fragments changed")
    rows=[];prior=[];models={};pointer_hash=hashlib.sha256();old_pointer_hash=hashlib.sha256()
    for binding in p["requests"]:
        request=read_sealed_json(root/binding["path"])
        if request.sha256!=binding["sha256"]:raise ValueError("prepared batch changed")
        fragments=restore(request.payload)
        from tools.compile_native_spine import add_pointer
        for fragment in fragments:add_pointer(pointer_hash,fragment)
        for model in p["models"]:
            key=identity_sha256({"request_sha256":request.sha256,"model":model})
            directory=root/"checkpoints"/key
            responses=list(directory.glob("*.response.json"))
            if not responses:
                rows.append({"request_sha256":request.sha256,"model":model,"status":"no_completion",
                    "request_journal_count":len(list(directory.glob("*.request.json")))})
                continue
            runtime=FastCompletionRuntime(checkpoint_dir=directory,prompt_population=[request.payload["messages"]],
                model=model,client=None,max_prompt_tokens=p["max_prompt_tokens"],max_new_tokens=p["max_new_tokens"],
                max_concurrency=1,retries=0,benchmark_provenance={"native_compile_request_sha256":request.sha256})
            try:batch=runtime.run()
            finally:runtime.close()
            if len(responses)!=1:raise ValueError("ambiguous response journal")
            saved=json.loads(responses[0].read_text(encoding="utf-8"))
            validated=read_sealed_json(root/"validated"/f"{key}.json")
            atoms=admit(batch.logical_completions[0],fragments)
            if (validated.payload["response_sha256"]!=quote_sha256(batch.logical_completions[0])
                    or json.loads(json.dumps(atoms))!=validated.payload["summaries"]
                    or saved["completion"]!=batch.logical_completions[0] or saved["finish_reason"]!="stop"):
                raise ValueError("completed atom or response binding changed")
            rows.append({"request_sha256":request.sha256,"model":model,"status":"accepted",
                "validated_sha256":validated.sha256,"response_journal_sha256":saved["journal_sha256"],
                "atoms":len(atoms),"provider_elapsed_s":saved["provider_elapsed_s"],
                "output_token_proxy":saved["completion_token_proxy"]})
    from memory_condense.search.native_spine_summary import BodyFragment
    for binding in old.payload["requests"]:
        request=read_sealed_json(previous_root/binding["path"])
        if request.sha256!=binding["sha256"]:raise ValueError("original raw request changed")
        for f in request.payload["fragments"]:add_pointer(old_pointer_hash,BodyFragment(**f))
        directory=previous_root/"checkpoints"/request.sha256
        runtime=FastCompletionRuntime(checkpoint_dir=directory,prompt_population=[request.payload["messages"]],
            model=old.payload["model"],client=None,max_prompt_tokens=old.payload["max_prompt_tokens"],
            max_new_tokens=old.payload["max_new_tokens"],max_concurrency=1,retries=0,
            benchmark_provenance={"native_raw_request_sha256":request.sha256})
        try:runtime.run()
        finally:runtime.close()
        saved=json.loads(next(directory.glob("*.response.json")).read_text(encoding="utf-8"))
        prior.append({"request_sha256":request.sha256,"response_journal_sha256":saved["journal_sha256"],
            "provider_elapsed_s":saved["provider_elapsed_s"],"output_token_proxy":saved["completion_token_proxy"]})
    if pointer_hash.digest()!=old_pointer_hash.digest() or pointer_hash.hexdigest()!=p["ordered_pointer_sha256"]:
        raise ValueError("regrouping did not preserve the exact original fragments")
    for model in p["models"]:
        selected=[r for r in rows if r["model"]==model]
        accepted=[r for r in selected if r["status"]=="accepted"]
        models[model]={"requests":len(selected),"completed_requests":len(accepted),
            "accepted_atoms":sum(r["atoms"] for r in accepted),
            "summed_request_seconds":sum(r["provider_elapsed_s"] for r in accepted),
            "output_token_proxy":sum(r["output_token_proxy"] for r in accepted)}
    result,_=publish_sealed_json(root/"partial-comparison-report.json",{
        "preflight_sha256":preflight.sha256,"previous_preflight_sha256":old.sha256,
        "ordered_pointer_sha256":pointer_hash.hexdigest(),"rows":rows,"previous_rows":prior,"models":models,
        "previous_summed_request_seconds":sum(r["provider_elapsed_s"] for r in prior),
        "previous_output_token_proxy":sum(r["output_token_proxy"] for r in prior),
        "new_provider_calls":0,"unacknowledged_requests_retried":False,"benchmark_accuracy_claim":False,
        "summed_request_time_is_not_parallel_wall_time":True,
        "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print({"report_sha256":result.sha256,"models":models,
        "previous_summed_request_seconds":result.payload["previous_summed_request_seconds"],
        "previous_output_token_proxy":result.payload["previous_output_token_proxy"],"new_provider_calls":0},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--previous-root",type=Path,required=True)
    args=parser.parse_args();report(args.output_root,args.previous_root)
