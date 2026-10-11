"""Screen tensorized Llama projections for multi-replica summary co-tenancy.

This is deliberately a projection-level assay.  It reads the exact deployed
GGUF tensor, dequantizes it once, finds the optimal two-core (truncated-SVD)
factorization, and stacks complete factor copies on a leading model dimension.
The packed arm is one pair of dense batched GEMMs; the control runs the same
model tensors separately.  It does not claim full-model or summary quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def factor_bytes(rows: int, columns: int, rank: int, *, copies: int, element_bytes: int = 2) -> int:
    if min(rows,columns,rank,copies,element_bytes) < 1:
        raise ValueError('Tensor dimensions, rank, copies, and element size must be positive')
    if rank > min(rows,columns):
        raise ValueError('Factor rank exceeds the matrix dimensions')
    return copies * rank * (rows + columns) * element_bytes


def maximum_rank(rows: int, columns: int, budget_bytes: int, *, copies: int,
                 element_bytes: int = 2) -> int:
    if budget_bytes < 1:
        raise ValueError('Memory budget must be positive')
    return min(rows,columns,budget_bytes // (copies * (rows + columns) * element_bytes))


def _elapsed(torch, operation, *, iterations: int) -> float:
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    started=time.perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter()-started)/iterations


def randomized_svd(torch, weight, rank: int, *, oversample: int, power_iterations: int, seed: int):
    """Return a deterministic approximate leading SVD without a full 3K SVD."""
    width=min(min(weight.shape),rank+oversample)
    generator=torch.Generator(device=weight.device).manual_seed(seed)
    omega=torch.randn(weight.shape[1],width,device=weight.device,dtype=weight.dtype,generator=generator)
    basis=weight @ omega
    for _ in range(power_iterations):
        basis=weight @ (weight.T @ basis)
    basis=torch.linalg.qr(basis,mode='reduced')[0]
    projected=basis.T @ weight
    u_small,singular,right=torch.linalg.svd(projected,full_matrices=False)
    return basis @ u_small[:,:rank],singular[:rank],right[:rank]


def assay(args) -> dict:
    sys.path.insert(0,str(args.gguf_python.resolve()))
    import numpy as np
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    reader=GGUFReader(str(args.model),'r')
    tensor=next((value for value in reader.tensors if value.name==args.tensor),None)
    if tensor is None:
        raise ValueError(f'GGUF tensor not found: {args.tensor}')
    source=np.asarray(dequantize(tensor.data,tensor.tensor_type),dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError('Assay requires one finite matrix')
    rows,columns=map(int,source.shape)
    quantized_bytes=int(tensor.n_bytes)
    feasible=maximum_rank(rows,columns,quantized_bytes,copies=args.copies,element_bytes=2)
    ranks=sorted(set(rank for rank in (*args.ranks,feasible) if 0 < rank <= min(rows,columns)))
    max_rank=max(ranks)

    torch.set_grad_enabled(False)
    weight=torch.from_numpy(source).to(args.device)
    norm_sq=float(torch.sum(weight*weight).item())
    started=time.perf_counter()
    left,singular,right=randomized_svd(torch,weight,max_rank,oversample=args.oversample,
        power_iterations=args.power_iterations,seed=args.seed)
    decomposition_s=time.perf_counter()-started
    generator=torch.Generator(device=args.device).manual_seed(args.seed+1)
    probes=torch.randn(args.copies,args.tokens,columns,device=args.device,dtype=torch.float16,
                       generator=generator)
    dense_weight=weight.to(torch.float16)
    exact=torch.matmul(probes,dense_weight.T)
    rows_out=[]
    for rank in ranks:
        # Materialize one complete factor image per model. The leading model
        # axis packs corresponding cores into dense arrays for two bmm kernels.
        # Do not replace this with expanded/shared weights: physical co-tenancy
        # and its memory/throughput trade are the subject of the assay.
        l=(left[:,:rank]*singular[:rank]).to(torch.float16)
        r=right[:rank].to(torch.float16)
        replica_l=l.unsqueeze(0).repeat(args.copies,1,1)
        replica_r=r.unsqueeze(0).repeat(args.copies,1,1)
        def packed_factorized():
            return torch.bmm(torch.bmm(probes,replica_r.transpose(1,2)),replica_l.transpose(1,2))
        def separate_factorized():
            return torch.stack([
                (probes[index] @ replica_r[index].T) @ replica_l[index].T
                for index in range(args.copies)
            ])
        predicted=packed_factorized()
        separate=separate_factorized()
        delta=(predicted.float()-exact.float())
        relative_rmse=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(exact.float()))
        cosine=float(torch.nn.functional.cosine_similarity(
            predicted.float().reshape(-1),exact.float().reshape(-1),dim=0))
        packing_delta=float(torch.max(torch.abs(predicted.float()-separate.float())).item())
        captured=min(1.0,float(torch.sum(singular[:rank]*singular[:rank]).item()/norm_sq))
        packed_ms=_elapsed(torch,packed_factorized,iterations=args.iterations)*1000
        separate_ms=_elapsed(torch,separate_factorized,iterations=args.iterations)*1000
        rows_out.append(dict(rank=rank,captured_weight_energy=captured,
            probe_relative_rmse=relative_rmse,probe_cosine=cosine,
            factor_bytes_per_replica=factor_bytes(rows,columns,rank,copies=1),
            all_replica_factor_bytes=factor_bytes(rows,columns,rank,copies=args.copies),
            replicas_vs_one_deployed_quantized_tensor=(
                factor_bytes(rows,columns,rank,copies=args.copies)/quantized_bytes),
            packed_core_shapes=[list(replica_l.shape),list(replica_r.shape)],
            packed_vs_separate_max_abs=packing_delta,
            packed_factorized_ms=packed_ms,separate_factorized_ms=separate_ms,
            packed_throughput_speedup=separate_ms/packed_ms))
        del replica_l,replica_r,predicted,separate,delta,l,r
    dense_replicas=dense_weight.unsqueeze(0).repeat(args.copies,1,1)
    def dense():
        return torch.bmm(probes,dense_replicas.transpose(1,2))
    dense_ms=_elapsed(torch,dense,iterations=args.iterations)*1000
    peak=int(torch.cuda.max_memory_allocated()) if args.device.startswith('cuda') else None
    return dict(format='tensorized-llama-cotenancy-projection-assay-v1',model=str(args.model.resolve()),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest() if args.hash_model else None,
        tensor=args.tensor,ggml_type=tensor.tensor_type.name,shape=[rows,columns],
        deployed_quantized_tensor_bytes=quantized_bytes,dense_fp16_tensor_bytes=rows*columns*2,
        copies=args.copies,tokens_per_replica=args.tokens,
        maximum_fp16_factor_rank_for_all_replicas_under_one_deployed_tensor=feasible,
        decomposition=dict(method='randomized truncated SVD (optimal two-core target)',
            oversample=args.oversample,power_iterations=args.power_iterations,seed=args.seed,
            elapsed_s=decomposition_s),dense_stacked_ms=dense_ms,arms=rows_out,
        peak_process_cuda_bytes=peak,
        packing='complete per-model factors stacked on a dense leading model axis; no weight sharing',
        boundary='projection reconstruction and packed-kernel screen; no full-model or summary-quality claim')


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--gguf-python',type=Path,required=True)
    parser.add_argument('--tensor',default='blk.0.attn_q.weight')
    parser.add_argument('--copies',type=int,default=2)
    parser.add_argument('--ranks',type=int,nargs='+',default=[64,128,256,384])
    parser.add_argument('--oversample',type=int,default=32)
    parser.add_argument('--power-iterations',type=int,default=1)
    parser.add_argument('--tokens',type=int,default=128)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--hash-model',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    return parser.parse_args()


def main():
    args=parse_args()
    if args.copies < 2 or min(args.tokens,args.iterations,args.oversample) < 1 or args.power_iterations < 0:
        raise ValueError('Invalid assay dimensions')
    result=assay(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    args.output.write_bytes(encoded)
    args.output.with_suffix(args.output.suffix+'.sha256').write_text(
        hashlib.sha256(encoded).hexdigest()+'  '+args.output.name+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
