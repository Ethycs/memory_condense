"""Repair one observed syntax fault in diagnostic support lists, never summaries."""
from __future__ import annotations

import json
import re

from memory_condense.domain._discourse_identity import quote_sha256


def repair_support_list_closures(response: str):
    """Restore missing or misplaced string terminators at support-list ends.

    The original must be invalid JSON. The repaired document must parse, and
    every inserted character must be inside a top-level atom's support value.
    A misplaced terminator just after the closing bracket may be moved before
    it. Summary characters never change. This does not verify support entailment.
    Any other JSON fault still fails and requires separate diagnosis.
    """
    try:
        json.loads(response)
        return response, None
    except json.JSONDecodeError:
        pass
    # Another observed serialization fault puts the string terminator after
    # the support array's closing bracket. Remove that misplaced terminator;
    # the guarded insertion path below must then prove it belongs to support.
    misplaced = re.compile(r'(\\"\s*\]\s*)"(?=\s*\}\s*(?:,\s*\{\s*"label"\s*:|\]\s*\}\s*$))')
    removed = [match.end() - 1 for match in misplaced.finditer(response)]
    if removed:
        normalized = response
        for position in reversed(removed):
            normalized = normalized[:position] + normalized[position + 1:]
        repaired, audit = repair_support_list_closures(normalized)
        if audit is None:
            raise ValueError("misplaced terminator recovery did not validate a support insertion")
        return repaired, {**audit,
            "policy": "restore missing or misplaced closing double quotes at support-list boundaries only",
            "original_response_sha256": quote_sha256(response), "normalized_response_sha256": quote_sha256(normalized),
            "removed_misplaced_terminator_offsets": removed,
            "insertion_offset_basis": "normalized response after removing misplaced terminators"}
    pattern = re.compile(r'\\"(?=\s*\]\s*\}\s*(?:,\s*\{\s*"label"\s*:|\]\s*\}\s*$))')
    positions = []
    for match in pattern.finditer(response):
        quote = match.end() - 1
        slashes = 0
        cursor = quote - 1
        while cursor >= 0 and response[cursor] == "\\":
            slashes += 1
            cursor -= 1
        if slashes % 2:
            positions.append(match.end())
    if not positions or len(positions) > 10:
        raise ValueError("unrecognized JSON syntax fault; support repair cannot apply")
    repaired = response
    for position in reversed(positions):
        repaired = repaired[:position] + '"' + repaired[position:]
    body = json.loads(repaired)
    if type(body) is not dict or set(body) != {"atoms"} or type(body["atoms"]) is not list:
        raise ValueError("support repair requires the exact atoms schema")
    # Locate actual field values by parsing the repaired top-level atom array;
    # quoted words inside summaries/support cannot impersonate structural keys.
    decoder = json.JSONDecoder()
    cursor = repaired.index("[") + 1
    support_ranges, summary_literals = [], []
    def skip(i):
        while i < len(repaired) and repaired[i].isspace():
            i += 1
        return i
    for index, atom in enumerate(body["atoms"]):
        cursor = skip(cursor)
        if index:
            if repaired[cursor] != ",":
                raise ValueError("atom array boundary changed")
            cursor = skip(cursor + 1)
        if repaired[cursor] != "{":
            raise ValueError("atom must be an object")
        cursor = skip(cursor + 1)
        fields = []
        while repaired[cursor] != "}":
            key, end = decoder.raw_decode(repaired, cursor)
            fields.append(key)
            cursor = skip(end)
            if repaired[cursor] != ":":
                raise ValueError("invalid atom field delimiter")
            start = skip(cursor + 1)
            value, end = decoder.raw_decode(repaired, start)
            if key == "support":
                support_ranges.append((start, end))
            if key == "summary":
                summary_literals.append(repaired[start:end])
            cursor = skip(end)
            if repaired[cursor] == ",":
                cursor = skip(cursor + 1)
        cursor += 1
        if fields != ["label", "summary", "support"] or atom["label"] != f"T{index}":
            raise ValueError("support syntax recovery requires ordered exact atom fields")
    inserted = [p + i for i, p in enumerate(positions)]
    if not all(sum(start < p < end for start, end in support_ranges) == 1 for p in inserted):
        raise ValueError("syntax repair would change a non-support value")
    # Removing only the inserted terminators must recover the original bytes.
    restored = repaired
    for position in reversed(inserted):
        restored = restored[:position] + restored[position + 1:]
    if restored != response:
        raise ValueError("support repair modified existing response content")
    return repaired, {"policy": "insert missing closing double quote inside support lists only",
        "original_response_sha256": quote_sha256(response), "repaired_response_sha256": quote_sha256(repaired),
        "original_insertion_offsets": positions, "summary_literals_sha256": quote_sha256("\n".join(summary_literals)),
        "summary_texts_unchanged": True, "new_provider_calls": 0}
