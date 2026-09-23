import copy
import json

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.native_spine_summary import (
    body_identity, fragment_body, materialize, pack_batches, parse_summaries, summary_messages,
)


def body():
    return {"turns":[{"role":"user","text":"I bought a lamp yesterday."},
                     {"role":"assistant","text":"You could try a warm bulb."}]}


def response(fragments):
    return json.dumps({"atoms":[{"label":f"T{i}","summary":f"{f.role} said: {f.text}",
                                "support":[f.text]} for i,f in enumerate(fragments)]})


def test_same_body_reuses_summary_but_keeps_occurrence_dates_and_pointers_separate():
    b=body(); fragments=fragment_body(b)
    summaries=parse_summaries(response(fragments),fragments)
    atoms=[materialize(summaries[0],b,occurrence_id=occ,created_at=date,compiler_identity="test")
           for occ,date in [("a","2023-05-22T00:00:00+00:00"),("b","2024-01-03T00:00:00+00:00")]]
    assert atoms[0].summary==atoms[1].summary
    assert "yesterday" in atoms[0].summary
    assert atoms[0].spans[0].turn_id!=atoms[1].spans[0].turn_id
    assert atoms[0].spans[0].created_at!=atoms[1].spans[0].created_at
    assert atoms[0].spans[0].span_text_sha256==atoms[1].spans[0].span_text_sha256
    wire=json.loads(summary_messages(fragments)[1]["content"])
    assert all(set(f)=={"label","speaker","fragment"} for f in wire["fragments"])
    assert "2023-05-22" not in json.dumps(wire)
    assert "2024-01-03" not in json.dumps(wire)


def test_unicode_and_literal_special_tokens_have_exact_complete_raw_coverage():
    text=("  🌍 café 日本語 <|endoftext|>\n"*9)+"tail  "
    b={"turns":[{"role":"user","text":text}]}
    pieces=fragment_body(b,token_cap=9)
    assert "".join(f.text for f in pieces)==text
    assert all(count_tokens(f.text)<=9 for f in pieces)
    assert all(text[f.start_char:f.end_char]==f.text for f in pieces)
    assert pieces[0].start_char==0 and pieces[-1].end_char==len(text)
    assert all(a.end_char==b.start_char for a,b in zip(pieces,pieces[1:]))


@pytest.mark.parametrize("extra",[{"question":"what?"},{"has_answer":True},{"created_at":"2023-01-01"}])
def test_source_plane_rejects_extra_qa_or_occurrence_fields(extra):
    b=body(); b["turns"][0].update(extra)
    with pytest.raises(ValueError): body_identity(b)


def test_quotes_cannot_borrow_another_speakers_evidence():
    fragments=fragment_body(body()); value=json.loads(response(fragments))
    value["atoms"][0]["support"]=[fragments[1].text]
    with pytest.raises(ValueError,match="own fragment"):
        parse_summaries(json.dumps(value),fragments)


@pytest.mark.parametrize("mutation",["omit","reorder","timestamp"])
def test_model_output_must_preserve_all_fragment_attribution(mutation):
    fragments=fragment_body(body()); value=json.loads(response(fragments))
    if mutation=="omit": value["atoms"].pop()
    elif mutation=="reorder": value["atoms"].reverse()
    else: value["atoms"][0]["created_at"]="2024-01-01"
    with pytest.raises(ValueError): parse_summaries(json.dumps(value),fragments)


def test_changed_body_or_out_of_range_cached_pointer_cannot_hydrate():
    b=body(); fragments=fragment_body(b)
    summary=parse_summaries(response(fragments),fragments)[0]
    kwargs={"occurrence_id":"a","created_at":"2023-05-22T00:00:00+00:00","compiler_identity":"test"}
    wrong=copy.deepcopy(b);wrong["turns"][0]["text"]+=" Changed."
    with pytest.raises(ValueError,match="another body"): materialize(summary,wrong,**kwargs)
    wrong=copy.deepcopy(summary);wrong["pointer"]["end_char"]+=1
    with pytest.raises(ValueError,match="exceeds the raw turn"): materialize(wrong,b,**kwargs)


def test_packing_preserves_every_fragment_and_separates_bodies():
    first=fragment_body(body())
    second=fragment_body({"turns":[{"role":"user","text":"Different transcript."}]})
    batches=pack_batches((*first,*second),max_atoms=1)
    assert tuple(f for batch in batches for f in batch)==(*first,*second)
    assert len(batches)==3
    with pytest.raises(ValueError,match="one transcript"): summary_messages((first[0],second[0]))


def test_existing_source_admission_keeps_bad_generated_quotes_out_of_hydration():
    from memory_condense.search.spine_batch_summary import RawSummaryFragment
    from memory_condense.search.spine_source_admission import admit_source_bound_summaries
    b=body(); fragments=fragment_body(b); value=json.loads(response(fragments))
    value["atoms"][0]["support"]=['"I bought a lamp yesterday."']
    raw=[]
    for fragment,row in zip(fragments,value["atoms"],strict=True):
        atom=materialize({"pointer":fragment.pointer(),"summary":row["summary"]},b,
            occurrence_id="a",created_at="2023-05-22T00:00:00+00:00",compiler_identity="test")
        raw.append(RawSummaryFragment(atom.spans[0],fragment.text))
    result=admit_source_bound_summaries(json.dumps(value),raw,compiler_identity="test")
    assert result.quote_diagnostics[0]["failures"]==[{"quote_index":0,"reason":"quote_not_exact"}]
    span=result.atoms[0].spans[0]
    assert b["turns"][0]["text"][span.start_char:span.end_char]==fragments[0].text
    assert result.summary_entailment_verified is False
