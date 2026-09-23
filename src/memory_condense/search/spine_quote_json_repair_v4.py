"""Escape embedded ASCII quotes in curly-quoted diagnostic support strings."""
import json
import re

from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.search.spine_quote_json_repair_v2 import _ranges
from memory_condense.search.spine_quote_json_repair_v3 import repair_support_list_closures as previous_repair


def repair_support_list_closures(response):
    try:
        return previous_repair(response)
    except ValueError:
        pass
    # Curly quotes explicitly delimit the quoted content. Existing valid JSON
    # and all earlier supported defects take the unchanged predecessor path.
    positions = []
    for match in re.finditer(r'"“(.*?)”"(?=\s*(?:,|\]))', response, re.S):
        for position in range(match.start(1), match.end(1)):
            if response[position] != '"':
                continue
            cursor = position - 1
            while cursor >= match.start(1) and response[cursor] == "\\":
                cursor -= 1
            if (position - cursor - 1) % 2 == 0:
                positions.append(position)
    if not positions or len(positions) > 100:
        raise ValueError("unrecognized JSON syntax fault; support repair v4 cannot apply")
    repaired = response
    for position in reversed(positions):
        repaired = repaired[:position] + "\\" + repaired[position:]
    json.loads(repaired)
    support, summaries = _ranges(repaired)
    inserted = [position + i for i, position in enumerate(positions)]
    if any(sum(start < position < end for start, end in support) != 1 for position in inserted):
        raise ValueError("embedded quote escaping would change a non-support value")
    restored = repaired
    for position in reversed(inserted):
        restored = restored[:position] + restored[position + 1:]
    if restored != response:
        raise ValueError("support escaping changed existing response characters")
    return repaired, {
        "policy": "escape unescaped ASCII quotes inside curly-quoted support strings only",
        "original_response_sha256": quote_sha256(response), "repaired_response_sha256": quote_sha256(repaired),
        "original_insertion_offsets": positions, "inserted_character": "\\",
        "summary_literals_sha256": quote_sha256("\n".join(summaries)),
        "summary_texts_unchanged": True, "original_characters_preserved": True, "new_provider_calls": 0}
