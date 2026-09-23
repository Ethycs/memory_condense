"""Extend support-only syntax repair to a duplicated opening string quote."""
import json
import re

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.spine_quote_json_repair import repair_support_list_closures as legacy_repair


def _ranges(document):
    """Parse structural fields, refusing duplicate keys and ambiguous boundaries."""
    decoder = json.JSONDecoder()
    cursor = 0
    def skip():
        nonlocal cursor
        while cursor < len(document) and document[cursor].isspace():
            cursor += 1
    def literal(value):
        nonlocal cursor
        skip()
        if not document.startswith(value, cursor):
            raise ValueError("support repair requires the exact atoms schema")
        cursor += len(value)
    def decoded():
        nonlocal cursor
        skip()
        start = cursor
        value, cursor = decoder.raw_decode(document, cursor)
        return value, start, cursor
    literal("{")
    if decoded()[0] != "atoms":
        raise ValueError("support repair requires the atoms field")
    literal(":")
    literal("[")
    support, summaries = [], []
    index = 0
    skip()
    while not document.startswith("]", cursor):
        if index:
            literal(",")
        literal("{")
        for i, expected in enumerate(("label", "summary", "support")):
            if i:
                literal(",")
            if decoded()[0] != expected:
                raise ValueError("support repair requires ordered unique atom fields")
            literal(":")
            value, start, end = decoded()
            if expected == "label" and value != f"T{index}":
                raise ValueError("support repair cannot change atom labels")
            if expected == "summary":
                if type(value) is not str or not value.strip():
                    raise ValueError("support repair requires existing nonempty summaries")
                summaries.append(document[start:end])
            if expected == "support":
                if type(value) is not list or any(type(item) is not str for item in value):
                    raise ValueError("support repair requires string support arrays")
                support.append((start, end))
        literal("}")
        index += 1
        skip()
    literal("]")
    literal("}")
    skip()
    if cursor != len(document):
        raise ValueError("support repair cannot accept trailing material")
    return support, summaries


def repair_support_list_closures(response):
    try:
        return legacy_repair(response)
    except (ValueError, json.JSONDecodeError):
        pass
    # An extra unescaped quote precedes the escaped quote that begins the
    # first support string. Only this observed array-opening defect is added.
    pattern = re.compile(r'"support"\s*:\s*\[\s*""(?=\\")')
    removed = [m.end() - 1 for m in pattern.finditer(response)]
    if not removed or len(removed) > 10:
        raise ValueError("unrecognized JSON syntax fault; support repair v2 cannot apply")
    repaired = response
    for position in reversed(removed):
        repaired = repaired[:position] + repaired[position + 1:]
    json.loads(repaired)  # Combined or unrelated defects still fail closed.
    support_ranges, summary_literals = _ranges(repaired)
    restored_positions = [position - i for i, position in enumerate(removed)]
    if not all(sum(start < position < end for start, end in support_ranges) == 1
               for position in restored_positions):
        raise ValueError("duplicate-quote repair would change a non-support value")
    restored = repaired
    for position in reversed(restored_positions):
        restored = restored[:position] + '"' + restored[position:]
    if restored != response:
        raise ValueError("duplicate-quote repair modified existing response characters")
    return repaired, {"policy": "remove a duplicated opening double quote in a support array only",
        "original_response_sha256": quote_sha256(response), "repaired_response_sha256": quote_sha256(repaired),
        "original_removal_offsets": removed, "summary_literals_sha256": quote_sha256("\n".join(summary_literals)),
        "summary_texts_unchanged": True, "new_provider_calls": 0}
