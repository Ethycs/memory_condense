"""Freeze denial, update and cross-conversation ordering probes on the same history."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import quote_sha256
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


CHALLENGES = (
    ("auction-denial", "Which older photography books had I won on eBay during the conversation about restoring my Olympus rangefinder?",
     "None; I said I had not won any of the auctions yet", "explicit denial despite a positive question premise",
     ((300, "fda7fc51da1602eb46c390e56417b1d3e95aa8dbd7bbfc848f12a0fcf4441482", (0, 4)),)),
    ("tea-update", "What tea had I actually switched to from my morning cafe coffee, rather than teas I was only considering trying?",
     "Green tea with lemon", "changed preference versus future interest",
     ((278, "23c1644104ced44320b90a7942d05825a58017db98a670e11cd85a733f63d4ba", (0, 4, 6)),)),
    ("report-order", "Which did I report first: attending a poetry reading at a local coffee shop, or buying the 1950s Olympus rangefinder?",
     "The poetry reading; it was reported on May 23, before the camera purchase report on May 25", "ordering across source conversations",
     ((200, "681a5eca50ea9bc76ed68048afc1251cb6a9599a85ef5d5cabc69f7921b53d1e", (0,)),
      (300, "fda7fc51da1602eb46c390e56417b1d3e95aa8dbd7bbfc848f12a0fcf4441482", (0,)))),
)


def prepare(root, previous, output):
    scope = read_sealed_json(root/"scope.json")
    base = read_sealed_json(previous)
    if base.payload["scope_sha256"] != scope.sha256:
        raise ValueError("challenge questions require the same cached history")
    prior_refs = bound(base.payload["references"])
    namespace = bound(scope.payload["namespace"])
    bank = Path(scope.payload["body_bank"]["path"])
    if digest(bank) != scope.payload["body_bank"]["sha256"]:
        raise ValueError("challenge source bank changed")
    # Replace one simple numeric-recall probe with three harder probes. This is
    # a separately named development population, never a comparable benchmark.
    cases = [c for c in base.payload["questions"] if c["question_id"] != "design-charity-donations"]
    refs = [r for r in prior_refs.payload["references"] if r["question_id"] != "design-charity-donations"]
    with closing(sqlite3.connect(bank.as_uri()+"?mode=ro", uri=True)) as connection:
        for name, question, answer, category, supports in CHALLENGES:
            quotations = []
            for index, body_sha, turns in supports:
                source = namespace.payload["sessions"][index]
                if source["body_sha256"] != body_sha:
                    raise ValueError("fixed challenge source changed")
                body = json.loads(connection.execute("SELECT body_json FROM bodies WHERE body_sha256=?", (body_sha,)).fetchone()[0])
                for turn_index in turns:
                    turn = body["turns"][turn_index]
                    if turn["role"] != "user":
                        raise ValueError("challenge references require actual user statements")
                    quotations.append({"source": source, "body_turn_index": turn_index,
                        "text": turn["text"], "text_sha256": quote_sha256(turn["text"])})
            case = {**scope.payload["case"], "ordinal": len(cases), "question_id": "design-"+name,
                "question": question, "reference_sha256": quote_sha256(answer), "category": category,
                "question_origin": "manually authored from cached transcripts before retrieval"}
            cases.append(case)
            refs.append({"question_id": case["question_id"], "answer": answer,
                "reference_sha256": case["reference_sha256"], "supports": quotations,
                "reference_source": "source quotations reviewed during question authoring"})
    if len(cases) != 8:
        raise ValueError("challenge set must contain exactly eight frozen questions")
    references, _ = publish_sealed_json(output/"references.json", {"scope_sha256": scope.sha256,
        "evaluation_only": True, "ingest_use_permitted": False, "references": refs})
    artifact, _ = publish_sealed_json(output/"questions.json", {"scope_sha256": scope.sha256,
        "parent_questions": binding(base), "namespace": scope.payload["namespace"],
        "actual_body_tokens": scope.payload["actual_body_tokens"], "questions": cases,
        "references": binding(references), "history_count": 1, "official_benchmark_questions": 1,
        "source_grounded_design_questions": 7, "development_set": True,
        "gold_in_retrieval_inputs": False, "general_accuracy_claim_permitted": False,
        "selection": "replace charity numeric recall with denial, update and cross-conversation ordering probes",
        "implementation_sha256": digest(__file__)})
    print({"questions_sha256": artifact.sha256, "questions": len(cases), "new_model_calls": 0, "new_histories": 0}, flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root, args.previous, args.output)
