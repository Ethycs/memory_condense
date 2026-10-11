"""Build an orthonormal routing basis spanning an existing CAV bank."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from memory_condense.associations.cav_memory import cav_layer_key
from memory_condense.runtime.experimental_attention import cav_bundle


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,separators=(',',':')),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('output already exists')
    source=cav_bundle(args.source,layer=json.loads(args.source.read_text())['layer'])
    source_report=json.loads(source['report'].read_text())
    names=[row['name'] for row in source_report['concepts']]; layer=source['layer']
    tensors=load_file(str(source['vectors']))
    matrix=torch.stack([tensors[cav_layer_key(name,layer)].float() for name in names])
    matrix=matrix/matrix.norm(dim=1,keepdim=True)
    _,singular,basis=torch.linalg.svd(matrix,full_matrices=False)
    # Canonicalize otherwise arbitrary SVD signs for reproducible bytes.
    for row in range(len(basis)):
        pivot=int(torch.argmax(torch.abs(basis[row])))
        if basis[row,pivot]<0: basis[row].neg_()
    args.output.mkdir(parents=True)
    component_names=[f'cav_span_component_{i+1}' for i in range(len(basis))]
    vectors=args.output/'vectors.safetensors'
    save_file({cav_layer_key(name,layer):basis[i].contiguous()
               for i,name in enumerate(component_names)},str(vectors),metadata={
                   'method':'orthonormal_svd_span','source_bundle_sha256':source['bundle_sha256']})
    energy=singular.square()
    effective=float(energy.sum().square()/energy.square().sum())
    report=args.output/'report.json'
    write(report,dict(model=source_report['model'],checkpoint_sha256=source['checkpoint_sha256'],
        runtime_dtype=source_report['runtime_dtype'],source_bundle_sha256=source['bundle_sha256'],
        source_concepts=names,transformation='orthonormal-svd-span-v1',
        singular_values=[float(value) for value in singular],effective_rank_participation=effective,
        concepts=[dict(name=name,layers=[dict(layer=layer,threshold=0.)])
                  for name in component_names],
        limitation='Routing-basis assay only; components are not independently interpretable concepts.'))
    bundle=args.output/'bundle.json'
    write(bundle,dict(checkpoint_sha256=source['checkpoint_sha256'],layer=layer,
        report=report.name,report_sha256=digest(report),vectors=vectors.name,
        vectors_sha256=digest(vectors)))
    print(bundle.resolve())


if __name__=='__main__': main()
