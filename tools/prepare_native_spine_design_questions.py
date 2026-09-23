"""Freeze five source-grounded questions alongside the existing official case."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from memory_condense.domain._discourse_identity import quote_sha256
from tools.assemble_native_spine_summaries import digest
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.prepare_native_spine_design_slice import binding, bound


# Chosen by reading fixed, spaced source ordinals before routing these queries.
# The golds and quoted supports belong to evaluation only, never to ingestion.
QUESTIONS = (
    (50, "rejected-offer", "How much was the house offer that was rejected before I started asking about home inspections?",
     "$275,000", (0,), "single user fact"),
    (300, "camera-price-difference", "How much more did I pay for the 1950s Olympus rangefinder than for the rare photography manual from the used bookstore?",
     "$200 more ($250 minus $50)", (0, 4), "calculation across user turns"),
    (450, "field-guide-coverage", "What geographic area does the new field guide I got before my planned Oakwood State Park trip cover?",
     "All of North America", (0, 2, 4, 6), "reference across user turns"),
    (250, "room-service-requirements", "For my room-service review blog, what database and web interface framework did I request, how many people would write reviews, and what timing detail did I want recorded?",
     "PostgreSQL; Angular; one blogger; time from ordering to receiving room service", (0, 2, 4), "multiple user requirements"),
    (400, "charity-donations", "How much did I say I raised from friends and family at the Walk for Hunger charity event?",
     "$250", (0, 6), "single user fact with numeric distractors"),
)
EXPECTED_BODIES = {
    50: "1c49810f1c39da8ab8bce8f4c7c988d38abd2b283c32411dda6bea2d1fbbcb1c",
    300: "fda7fc51da1602eb46c390e56417b1d3e95aa8dbd7bbfc848f12a0fcf4441482",
    450: "bc4f7eca9fb8100cd2c379dc414474b2f00e4adc1cfe74791996581e2464ce99",
    250: "42176b514119ca280c98aa518a6d4c20c70979a6e38a979d27da27022d1127ea",
    400: "a656f212fe38e7779854139ee46641728969816c3e259c035a1da4a687138970",
}


def prepare(root, output):
    scope = read_sealed_json(root/"scope.json")
    namespace = bound(scope.payload["namespace"])
    bank = Path(scope.payload["body_bank"]["path"])
    if digest(bank) != scope.payload["body_bank"]["sha256"]:
        raise ValueError("question support bank changed")
    official = {**scope.payload["case"], "question_origin": "original M benchmark, previously exposed",
                "category": "single user fact"}
    questions = [official]
    references = [{"question_id": official["question_id"], "answer": "Philips LED bulb",
                   "reference_sha256": official["reference_sha256"], "supports": [],
                   "reference_source": "previously completed official pilot"}]
    if quote_sha256(references[0]["answer"]) != official["reference_sha256"]:
        raise ValueError("official reference changed")
    with closing(sqlite3.connect(bank.as_uri()+"?mode=ro", uri=True)) as connection:
        for index, name, question, answer, turn_indexes, category in QUESTIONS:
            source = namespace.payload["sessions"][index]
            if source["body_sha256"] != EXPECTED_BODIES[index]:
                raise ValueError("fixed design source ordinal changed")
            body = json.loads(connection.execute("SELECT body_json FROM bodies WHERE body_sha256=?",
                (source["body_sha256"],)).fetchone()[0])
            supports = []
            for turn_index in turn_indexes:
                turn = body["turns"][turn_index]
                if turn["role"] != "user":
                    raise ValueError("design reference must be supported by actual user statements")
                supports.append({"source": source, "body_turn_index": turn_index,
                    "text": turn["text"], "text_sha256": quote_sha256(turn["text"])})
            case = {**official, "ordinal": len(questions), "question_id": "design-"+name,
                "question": question, "reference_sha256": quote_sha256(answer), "category": category,
                "question_origin": "manually authored from cached transcripts before retrieval"}
            questions.append(case)
            references.append({"question_id": case["question_id"], "answer": answer,
                "reference_sha256": case["reference_sha256"], "supports": supports,
                "reference_source": "source quotations reviewed during question authoring"})
    reference_file, _ = publish_sealed_json(output/"references.json", {"scope_sha256": scope.sha256,
        "evaluation_only": True, "ingest_use_permitted": False, "references": references})
    cases, _ = publish_sealed_json(output/"questions.json", {"scope_sha256": scope.sha256,
        "namespace": scope.payload["namespace"], "actual_body_tokens": scope.payload["actual_body_tokens"],
        "questions": questions, "references": binding(reference_file), "history_count": 1,
        "official_benchmark_questions": 1, "source_grounded_design_questions": 5,
        "development_set": True, "gold_in_retrieval_inputs": False,
        "implementation_sha256": digest(__file__), "general_accuracy_claim_permitted": False})
    print({"questions_sha256": cases.sha256, "question_count": len(questions), "history_count": 1,
        "model_calls": 0}, flush=True)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root, args.output)
