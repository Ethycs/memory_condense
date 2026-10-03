"""Frozen behavioral checks for the five added engineering interfaces."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

MODULES={'E11':'tape_graph','E12':'zero_compaction','E13':'growth_plan','E14':'mix_tuner','E15':'roi_graph'}


def rejects(fn):
    try:
        fn()
    except (ValueError,TypeError):
        return
    raise AssertionError('Malformed input accepted')


def E11(m):
    text='A\nβ🙂Z'
    parts=m.segment(text,[0,2,4,5])
    assert parts==[{'id':'n0','start':0,'end':2,'text':'A\n'},
                   {'id':'n1','start':2,'end':4,'text':'β🙂'},
                   {'id':'n2','start':4,'end':5,'text':'Z'}]
    original=deepcopy(parts)
    assert m.to_graph(parts)=={'nodes':parts,'edges':[['n0','n1'],['n1','n2']]}
    assert m.reassemble(parts,['n0','n1','n2'])==text
    assert m.reassemble(parts,['n2','n0','n1'])=='ZA\nβ🙂'
    assert parts==original
    assert m.segment('',[0])==[]
    assert m.reassemble([],[])==''
    rejects(lambda:m.segment(text,[0,2,2,5]))
    rejects(lambda:m.segment(text,[1,5]))
    rejects(lambda:m.reassemble(parts,['n0','n0','n2']))
    rejects(lambda:m.reassemble(parts,['n0','n1']))
    rejects(lambda:m.reassemble(parts,['n0','n1','unknown']))


def E12(m):
    incoming=[[0,0],[2,-1],[0,0],[1e-12,0]]
    outgoing=[[7,3,8,4],[-2,1,9,-3]]
    originals=deepcopy((incoming,outgoing))
    result=m.compact_layer(incoming,outgoing)
    assert result==dict(incoming=[[2,-1],[1e-12,0]],outgoing=[[3,4],[1,-3]],kept=[1,3],removed=[0,2])
    for x in ([1,2],[-3,7],[0,0],[2.5,-8]):
        def apply(a,b):
            hidden=[sum(v*w for v,w in zip(row,x)) for row in a]
            return [sum(v*w for v,w in zip(row,hidden)) for row in b]
        assert apply(incoming,outgoing)==apply(result['incoming'],result['outgoing'])
    assert (incoming,outgoing)==originals
    assert m.compact_layer([[0],[0]],[[1,2]])==dict(incoming=[],outgoing=[[]],kept=[],removed=[0,1])
    rejects(lambda:m.compact_layer([[1,2],[3]],[[1,2]]))
    rejects(lambda:m.compact_layer([[1]],[[1,2]]))
    rejects(lambda:m.compact_layer([[float('nan')]],[[1]]))


def E13(m):
    seed={'a':2.0,'b':-4.0}
    ids=['new','b','a']
    assert m.initialize_growth(seed,ids,.2)=={'parameters':[
        {'id':'new','value':0,'lr':.2,'origin':'new'},
        {'id':'b','value':-4.0,'lr':.1,'origin':'seed'},
        {'id':'a','value':2.0,'lr':.1,'origin':'seed'}]}
    assert seed=={'a':2.0,'b':-4.0} and ids==['new','b','a']
    rejects(lambda:m.initialize_growth(seed,['a'],.1))
    rejects(lambda:m.initialize_growth(seed,['a','b','a'],.1))
    rejects(lambda:m.initialize_growth(seed,ids,0))
    rejects(lambda:m.initialize_growth(seed,ids,float('inf')))
    rejects(lambda:m.initialize_growth({'a':float('nan')},['a'],.1))


def E14(m):
    seen=[]
    def score(values):
        seen.append(list(values))
        return sum((v-1)**2 for v in values)
    left,right=[0,2],[2,0]
    result=m.select_mix(left,right,score,[1,.5,0,.5])
    assert result=={'alpha':.5,'value':[1,1],'score':0,'evaluations':[
        {'alpha':0,'score':2},{'alpha':.5,'score':0},{'alpha':1,'score':2}]}
    assert seen==[[0,2],[1,1],[2,0]] and left==[0,2] and right==[2,0]
    assert m.select_mix([0],[2],lambda _:0,[1,0])['alpha']==0
    rejects(lambda:m.select_mix([0],[1,2],score,[0]))
    rejects(lambda:m.select_mix([0],[1],score,[-.1]))
    rejects(lambda:m.select_mix([0],[1],score,[]))
    rejects(lambda:m.select_mix([0],[1],lambda _:float('nan'),[0]))


def E15(m):
    nodes=['a','b','c','d','island']
    edges=[['a','b'],['c','b'],['c','d']]
    original=deepcopy((nodes,edges))
    assert m.extract_neighborhood(nodes,edges,['b'],1)==dict(
        nodes=['a','b','c'],edges=[['a','b'],['c','b']],boundary_edges=[['c','d']])
    assert m.extract_neighborhood(nodes,edges,['b'],0)==dict(
        nodes=['b'],edges=[],boundary_edges=[['a','b'],['c','b']])
    assert m.extract_neighborhood(nodes,edges,['island'],9)==dict(nodes=['island'],edges=[],boundary_edges=[])
    assert (nodes,edges)==original
    rejects(lambda:m.extract_neighborhood(nodes,edges,['missing'],0))
    rejects(lambda:m.extract_neighborhood(nodes,edges,[],0))
    rejects(lambda:m.extract_neighborhood(nodes,edges,['a'],-1))
    rejects(lambda:m.extract_neighborhood(nodes,[['a','missing']],['a'],1))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    path=args.workspace/(MODULES[args.case]+'.py')
    spec=importlib.util.spec_from_file_location('candidate',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals()[args.case](module)
    print(json.dumps(dict(case=args.case,independent_behavioral_checks='passed')))
