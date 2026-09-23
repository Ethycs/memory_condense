"""Recover duplicated support-string delimiters while preserving summary bytes."""
import difflib
import re

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.spine_quote_json_repair_v2 import _ranges
from memory_condense.search.spine_quote_json_repair_v2 import repair_support_list_closures as previous_repair


def repair_support_list_closures(response):
    try:
        return previous_repair(response)
    except ValueError:
        pass
    # These are candidate edits only. Parsing and structural support ranges
    # below must prove that every changed character belongs to support.
    positions = {m.end() - 1 for m in re.finditer(r'[\[,]\s*""(?=[^",\]\s])', response)}
    for match in re.finditer(r'""(?=\s*(?:,|\]))', response):
        cursor = match.start() - 1
        slashes = 0
        while cursor >= 0 and response[cursor] == "\\":
            slashes += 1
            cursor -= 1
        if slashes % 2:
            continue  # A valid escaped content quote followed by a terminator.
        before = response[:match.start()].rstrip()
        if not before or before[-1] in "[,":
            continue  # Preserve valid empty support strings.
        positions.add(match.end() - 1)
    if not positions or len(positions) > 60:
        raise ValueError("unrecognized JSON syntax fault; support repair v3 cannot apply")
    normalized = response
    for position in sorted(positions, reverse=True):
        normalized = normalized[:position] + normalized[position + 1:]
    repaired, inner = previous_repair(normalized)
    support, summaries = _ranges(repaired)
    changes = []
    changed_characters = 0
    for opcode, old_start, old_end, new_start, new_end in difflib.SequenceMatcher(
            None, response, repaired, autojunk=False).get_opcodes():
        if opcode == "equal":
            continue
        before, after = response[old_start:old_end], repaired[new_start:new_end]
        if any(c != '"' for c in before + after):
            raise ValueError("support delimiter repair changed a non-quote character")
        # A misplaced terminator can sit immediately after the support array;
        # no quote edit may reach a summary, field name or other atom value.
        if sum(start < new_start <= new_end <= end for start, end in support) != 1:
            raise ValueError("support delimiter repair would change a non-support value")
        changed_characters += len(before) + len(after)
        changes.append({"operation": opcode, "original_start": old_start, "original_end": old_end,
            "repaired_start": new_start, "repaired_end": new_end})
    if not changes or changed_characters > 100:
        raise ValueError("support delimiter repair exceeded its bounded quote edits")
    return repaired, {"policy": "repair duplicated support-string delimiter quotes with existing support terminator recovery",
        "original_response_sha256": quote_sha256(response), "repaired_response_sha256": quote_sha256(repaired),
        "normalization_removal_offsets": sorted(positions), "quote_edits": changes,
        "summary_literals_sha256": quote_sha256("\n".join(summaries)),
        "summary_texts_unchanged": True, "nested_terminator_repair": inner, "new_provider_calls": 0}
