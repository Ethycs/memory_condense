import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.native_spine_batch import admit, messages, pack, restore
from memory_condense.search.native_spine_summary import fragment_body
from tools import compile_native_spine as compiler
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json


def fragments():
    first=fragment_body({"turns":[{"role":"user","text":"I plan to buy a blue chair tomorrow."},
                                  {"role":"assistant","text":"A small chair could fit."}]})
    second=fragment_body({"turns":[{"role":"user","text":"I bought a red chair yesterday."}]})
    return (*first,*second)


def test_independent_transcripts_stay_separate_without_source_or_date_metadata():
    fs=fragments(); prompt=messages(fs)
    wire=json.loads(prompt[1]["content"])["transcripts"]
    assert [len(b["fragments"]) for b in wire]==[2,1]
    assert [f["label"] for b in wire for f in b["fragments"]]==["T0","T1","T2"]
    assert all(set(f)=={"label","speaker","fragment"} for b in wire for f in b["fragments"])
    assert all(f.body_sha256 not in prompt[1]["content"] for f in fs)
    assert restore({"messages":prompt,"pointers":[f.pointer() for f in fs]})==fs


def test_mutated_raw_input_cannot_reuse_an_original_pointer():
    fs=fragments(); p={"messages":messages(fs),"pointers":[f.pointer() for f in fs]}
    p["messages"][1]["content"]=p["messages"][1]["content"].replace("blue chair","green chair")
    with pytest.raises(ValueError):restore(p)


def test_packing_preserves_the_complete_last_partial_batch():
    fs=fragments(); batches=tuple(pack(iter(fs),max_atoms=2))
    assert [len(b) for b in batches]==[2,1]
    assert tuple(f for b in batches for f in b)==fs
    with pytest.raises(ValueError):messages((fs[0],fs[2],fs[1]))
    with pytest.raises(ValueError):messages((fs[0],fs[0]))


@pytest.mark.parametrize("mutation",["omit","duplicate_key","label","oversized","role"])
def test_model_cannot_change_fragment_population_or_provenance(mutation):
    fs=fragments(); value={"atoms":[{"label":f"T{i}","summary":"A routing summary."} for i in range(3)]}
    if mutation=="omit":value["atoms"].pop()
    if mutation=="label":value["atoms"][0]["label"]="T2"
    if mutation=="oversized":value["atoms"][0]["summary"]="word "*130
    if mutation=="role":value["atoms"][0]["role"]="assistant"
    raw=json.dumps(value)
    if mutation=="duplicate_key":raw=raw.replace('"summary":','"summary":"discarded", "summary":',1)
    with pytest.raises(ValueError):admit(raw,fs)


@pytest.mark.parametrize("mode,invalid,expected",[("full",False,True),("probe",False,False),("full",True,False)])
def test_compiler_completion_requires_every_atom_and_the_complete_source_mode(tmp_path,monkeypatch,mode,invalid,expected):
    fs=fragments(); prompt=messages(fs); digest=hashlib.sha256()
    for f in fs:compiler.add_pointer(digest,f)
    request,_=publish_sealed_json(tmp_path/"requests"/"000000.json",{
        "ordinal":0,"sources_sha256":"source","messages":prompt,"messages_sha256":identity_sha256(prompt),
        "pointers":[f.pointer() for f in fs]})
    publish_sealed_json(tmp_path/"preflight.json",{
        "models":["codex_sdk/gpt-5.6-terra"],"mode":mode,"sources_sha256":"source",
        "requests":[{"path":"requests/000000.json","sha256":request.sha256,"atoms":3}],
        "ordered_pointer_sha256":digest.hexdigest(),"fragment_count":3,
        "max_prompt_tokens":7000,"max_new_tokens":4096,"concurrency":1,"timeout_s":240,
        "implementation":compiler.implementation()})
    class Runtime:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def close(self):pass
    monkeypatch.setattr(compiler,"FastCompletionRuntime",Runtime)
    monkeypatch.setattr(compiler,"_authenticated_records",lambda runtime:[])
    def fake_run(**kwargs):
        runtime=kwargs["runtime_factory"](None)
        assert runtime.kwargs["prompt_population"]==[prompt]
        assert kwargs["authorized_provider_calls"]==1
        count=2 if invalid else 3
        response=json.dumps({"atoms":[{"label":f"T{i}","summary":"User statement."} for i in range(count)]})
        return SimpleNamespace(logical_completions=[response]),1,0,0.1
    monkeypatch.setattr(compiler,"_run_exactly_authorized",fake_run)
    compiler.execute(tmp_path,True)
    result=read_sealed_json(tmp_path/"result.json").payload
    model=result["models"]["codex_sdk/gpt-5.6-terra"]
    assert model["complete_source_compilation"] is expected
    assert model["accepted_atoms"]==(0 if invalid else 3)
    assert result["full100_target_passed"] is False


def test_full_compiler_refuses_qwen_and_multiple_raw_models_before_io(tmp_path):
    with pytest.raises(ValueError):compiler.prepare(tmp_path,tmp_path,["qwen3-8b"])
    with pytest.raises(ValueError):compiler.prepare(tmp_path,tmp_path,["codex_sdk/gpt-5.6-terra","claude-haiku-4-5"])
