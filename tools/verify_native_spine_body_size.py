"""Subtract generated source boundaries from authenticated native token totals."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path

from memory_condense.domain._tokenizer import count_tokens, tokenizer_proxy_identity
from memory_condense.ingest.loader import _parse_longmemeval_date
from tools.matched_eval.artifacts import read_sealed_json, publish_sealed_json


def verify(root, output_root):
    sources=read_sealed_json(root/"sources.json")
    complete=read_sealed_json(root/"complete.json")
    proof=read_sealed_json(root/"verification.json")
    cases=read_sealed_json(root/"evaluation-cases.json")
    if (sources.sha256!="f0c5848453552bf092a142e2d7f5c1aa402eee201ddd90be31f3166d7d700f26"
            or complete.payload["sources_sha256"]!=sources.sha256
            or proof.payload["complete_sha256"]!=complete.sha256
            or proof.payload["evaluation_cases_sha256"]!=cases.sha256
            or sources.payload["tokenizer"]!=tokenizer_proxy_identity()):
        raise ValueError("native source, evaluation or tokenizer binding changed")
    bindings={n["namespace_id"]:n for n in sources.payload["namespaces"]}
    rows=[]
    for case in cases.payload["cases"]:
        binding=bindings[case["namespace_id"]]
        namespace=read_sealed_json(root/binding["path"])
        if namespace.sha256!=binding["sha256"] or namespace.sha256!=case["namespace_sha256"]:
            raise ValueError("native namespace changed")
        asked=_parse_longmemeval_date(case["question_date"]).date()
        sessions=namespace.payload["sessions"]
        if len(sessions)!=binding["source_occurrence_count"]:
            raise ValueError("occurrence population changed")
        metadata=sum(count_tokens(s["metadata_text"]) for s in sessions)
        eligible_metadata=sum(count_tokens(s["metadata_text"]) for s in sessions
                              if datetime.fromisoformat(s["created_at"]).date()<=asked)
        total=namespace.payload["raw_token_proxy"]-metadata
        eligible=case["question_day_eligible_token_proxy"]-eligible_metadata
        if not 0<eligible<=total or case["raw_token_proxy"]!=namespace.payload["raw_token_proxy"]:
            raise ValueError("body-only token accounting inconsistent")
        rows.append({"ordinal":case["ordinal"],"namespace_sha256":namespace.sha256,
            "body_tokens":total,"through_question_day_body_tokens":eligible,
            "excluded_boundary_tokens":metadata,"excluded_eligible_boundary_tokens":eligible_metadata})
    if sorted(r["ordinal"] for r in rows)!=list(range(100)):
        raise ValueError("full100 size audit incomplete")
    summary={"minimum_body_tokens":min(r["body_tokens"] for r in rows),
        "minimum_through_question_day_body_tokens":min(r["through_question_day_body_tokens"] for r in rows),
        "total_body_tokens":sum(r["body_tokens"] for r in rows),
        "below_1m_total":[r["ordinal"] for r in rows if r["body_tokens"]<1_000_000],
        "below_1m_through_question_day":[r["ordinal"] for r in rows if r["through_question_day_body_tokens"]<1_000_000]}
    files=(Path(__file__),Path("src/memory_condense/domain/_tokenizer.py"))
    result,_=publish_sealed_json(output_root/"body-size-verification.json",{
        "sources_sha256":sources.sha256,"complete_sha256":complete.sha256,"source_proof_sha256":proof.sha256,
        "evaluation_cases_sha256":cases.sha256,"tokenizer":tokenizer_proxy_identity(),
        "method":"subtract exactly counted generated boundaries from previously verified occurrence totals",
        "body_text_retokenized":False,"generated_boundaries_included":False,"chat_framing_included":False,
        "implementation":{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "summary":summary,"rows":rows,"new_model_calls":0,"answer_accuracy_claim":False})
    print({"body_size_sha256":result.sha256,**summary},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    verify(args.sources_root,args.output_root)
