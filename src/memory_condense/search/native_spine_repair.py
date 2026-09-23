"""Targeted native summary repairs that retain every already-valid output."""
import json

from memory_condense.domain._discourse_identity import canonical_json
from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.search.native_spine_batch import admit, messages


REPAIR_INSTRUCTION = (
    " This request repairs an overlong summary. Aim for no more than 60 words and 90 tokens; "
    "the hard maximum remains 96 words and 128 tokens. Remove generic introductory and closing "
    "phrases before dropping specific facts. Preserve names, quantities, dates, units, identifiers, "
    "negation and whether an action was proposed or completed. Output only the required JSON."
)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate model-output field")
        result[key] = value
    return result


def partition(response, fragments):
    """Only nonempty, structurally attributable over-budget atoms are repairable."""
    value = json.loads(response, object_pairs_hook=unique_object)
    if (type(value) is not dict or set(value) != {"atoms"}
            or type(value["atoms"]) is not list or len(value["atoms"]) != len(fragments)):
        raise ValueError("targeted repair requires a complete original atom list")
    valid, invalid = {}, []
    for index, (row, fragment) in enumerate(zip(value["atoms"], fragments, strict=True)):
        if (type(row) is not dict or set(row) != {"label", "summary"}
                or row["label"] != f"T{index}" or type(row["summary"]) is not str
                or not row["summary"].strip()):
            raise ValueError("targeted repair cannot infer missing text or attribution")
        # Local reindexing only permits the unchanged string to use the existing
        # one-fragment validator. No generated label becomes a source pointer.
        try:
            accepted = admit(canonical_json({"atoms": [{"label": "T0", "summary": row["summary"]}]}),
                             (fragment,))
        except ValueError:
            invalid.append(index)
        else:
            valid[index] = accepted[0]
    return valid, tuple(invalid)


def repair_messages(fragment):
    prompt = messages((fragment,))
    prompt[0]["content"] += REPAIR_INSTRUCTION
    if count_chat_prompt_token_proxy(prompt) > 7000:
        raise ValueError("complete repair fragment exceeds the prompt budget")
    return prompt


def reconcile(response, fragments, replacements):
    valid, invalid = partition(response, fragments)
    if set(replacements) != set(invalid):
        raise ValueError("repairs must cover exactly the invalid original atoms")
    output = []
    for index, fragment in enumerate(fragments):
        if index in valid:
            output.append(valid[index])
        else:
            output.append(admit(replacements[index], (fragment,))[0])
    if [row["pointer"] for row in output] != [f.pointer() for f in fragments]:
        raise ValueError("repair changed exact raw coverage")
    return tuple(output)
