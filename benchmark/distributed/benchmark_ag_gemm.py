# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

### This benchmark introduces overlapping intra-node AllGather and GEMM. ###

'''Bugfix first:
Triton-distributed/python/triton_dist/kernels/nvidia/allgather_gemm.py:566
```python
M = M_per_rank * ctx.num_ranks
```
should be:
```python
M = M_per_rank * num_ranks
```
'''

#TODO: further tune the performance
#TODO: support persistent gemm

import argparse
import torch
import torch.distributed as dist
import pynvshmem  
import tilelang
import tilelang.language as T
from tilelang.distributed.utils import init_distributed, dtype_map, perf_fn, CUDA_CHECK
import triton_dist
from triton_dist.kernels.nvidia.allgather_gemm import (
    create_ag_gemm_context,
    ag_gemm
)
from functools import partial
from typing import Callable

tilelang.disable_cache()


@tilelang.jit(
    out_idx=-1,
    pass_configs={"tl.disable_tma_lower": True}
)
def nonpersistent_matmut_transpose_consumer(
    rank, 
    num_ranks,
    M, 
    N_per_rank,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    threads=128
) -> Callable:
    accum_dtype = "float32"
    signal_dtype = "uint64"  # NVSHMEM requires uint64 for signal
    M_blocks, N_blocks, K_blocks = M // block_M, N_per_rank // block_N, K // block_K
    
    @T.prim_func
    def matmut_transpose(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((N_per_rank, K), dtype),  # type: ignore
        signal: T.Tensor((num_ranks), signal_dtype),  # type: ignore
        C: T.Tensor((M, N_per_rank), dtype),  # type: ignore
    ):
        with T.Kernel(M_blocks, N_blocks, threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            
            T.use_swizzle(10)

            T.signal_wait_until(
                T.address_of(signal[rank]), T.NVSHMEM_CMP_EQ, 1
            )
            for k in T.Pipelined(K_blocks, num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_shared)
                T.copy(B[by * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                
            T.copy(C_local, C[bx * block_M, by * block_N])
            
    return matmut_transpose


def overlapped_ag_gemm(
    A: torch.Tensor, 
    B: torch.Tensor,
    rank: int,
    num_ranks: int,
    persistent: bool = False,  # TODO: support persistent consumers
) -> torch.Tensor:
    """
    Overlapped AllGather-GEMM.
    Args:
        A: local input of shape (M_per_rank, K)
        B: local weight of shape (N_per_rank, K)
        rank: current rank
        num_ranks: total number of ranks
        persistent: whether to use persistent GEMM consumers
    Returns:
        Output of shape (M, N_per_rank)
    """
    
    M_per_rank, K = A.shape
    N_per_rank, _ = B.shape
    assert A.shape[1] == B.shape[1], "A and B must have the same inner dimension"
    M = M_per_rank * num_ranks
    
    # Prepare kernel and buffers
    assert persistent == False, "Persistent GEMM consumers are not supported yet"
    consumer = nonpersistent_matmut_transpose_consumer(
        rank=rank,
        num_ranks=num_ranks,
        M=M,
        N_per_rank=N_per_rank,
        K=K,
        block_M=128,
        block_N=128,
        block_K=128,
        dtype=dtype,
        threads=threads
    )
    ag_buffer = pynvshmem.nvshmem_create_tensor_list_intra_node(
        shape=[M, K],
        dtype=A.dtype,
    )
    signal_buffer = torch.zeros([num_ranks], dtype=torch.uint64, device="cuda")

    # We place copy-based AllGather and GEMM on two streams to implement inter-op comm-comp overlapping
    ag_stream = torch.cuda.Stream()
    gemm_stream = torch.cuda.current_stream()
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    gemm_stream.wait_stream(current_stream)
    
    with torch.cuda.stream(ag_stream):
        ag_buffer[rank][rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)
        pynvshmem.write_u64(
            signal_buffer[rank],
            1,
            ag_stream
        )
        pynvshmem.nvshmemx_barrier_all_on_stream(ag_stream.cuda_stream)  # Ensure visible to all ranks
        rank_orders = [(rank + i) % num_ranks for i in range(1, num_ranks)]
        for src_rank in rank_orders:
            dst = ag_buffer[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            src = ag_buffer[src_rank][src_rank * M_per_rank:(src_rank + 1) *
                                                  M_per_rank, :]
            dst.copy_(src)
            pynvshmem.write_u64(
                signal_buffer[src_rank],
                1,
                ag_stream
            )
    
    with torch.cuda.stream(gemm_stream):
        out = consumer(
            ag_buffer[rank],
            B,
            signal_buffer
        )
    
    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(gemm_stream)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--M", type=int,
        default=8192)  
    parser.add_argument("--N", type=int, default=49152)
    parser.add_argument("--K", type=int, default=12288)
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--threads", type=int, default=128, help="number of threads in a block")
    parser.add_argument("--warmup", type=int, default=5, help="number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="number of repeat iterations")
    return parser.parse_args()


if __name__ == '__main__':
    assert torch.cuda.get_device_capability()[0] >= 9, '❗This benchmark requires sm_90 or higher'
    
    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)
    assert WORLD_SIZE <= 8, "This benchmark is designed for intra-node AG-GEMM"

    args = parse_args()
    M, N, K, dtype, threads, warmup, repeat = args.M, args.N, args.K, args.dtype, args.threads, args.warmup, args.repeat
    PE_num = WORLD_SIZE
    assert M % PE_num == 0 and N % PE_num == 0, "M and N must be divisible by PE_num"
    M_per_rank, N_per_rank = M // PE_num, N // PE_num
    torch_dtype = dtype_map[dtype]
    
    
    ## Inputs: A (M_per_rank, K), B (N_per_rank, K)
    ## Outputs: ag(A0) @ B.T (M, N_per_rank)

    A = torch.randn([M_per_rank, K], dtype=torch_dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=torch_dtype, device="cuda")

    # Benchmark Torch (non-overlapped baseline)
    def torch_ag_gemm(): 
        ag_buffer = torch.empty([M, K], dtype=torch_dtype, device="cuda")
        dist.all_gather_into_tensor(ag_buffer, A, TP_GROUP)
        return ag_buffer @ B.T
    
    dist.barrier(TP_GROUP)
    torch_out, torch_t = perf_fn(torch_ag_gemm, warmup, repeat)
    print(f"rank {RANK} torch AG-GEMM avg time: {torch_t} ms")

    # Benchmark Triton-dist (overlapped)
    def triton_ag_gemm(persistent, autotune):
        return ag_gemm(
            A, B, 
            rank=RANK, 
            num_ranks=PE_num, 
            persistent=persistent, 
            autotune=autotune
        )

    dist.barrier(TP_GROUP)
    triton_ag_gemm = partial(triton_ag_gemm, persistent=False, autotune=False)
    tt_out, tt_t = perf_fn(triton_ag_gemm, warmup, repeat)
    print(f"rank {RANK} triton AG-GEMM avg time: {tt_t} ms")

    # Benchmark Tilelang-dist (overlapped)
    def tilelang_ag_gemm(persistent):
        return overlapped_ag_gemm(
            A, B, 
            rank=RANK, 
            num_ranks=PE_num, 
            persistent=persistent
        )

    # dist.barrier(TP_GROUP)
    tilelang_ag_gemm = partial(tilelang_ag_gemm, persistent=False)
    tl_out, tl_t = perf_fn(tilelang_ag_gemm, warmup, repeat)
    print(f"rank {RANK} tilelang AG-GEMM avg time: {tl_t} ms")

    # Check correctness
    assert torch.allclose(tl_out, torch_out, atol=1e-2, rtol=1e-2), f'max error: {(tl_out - torch_out).abs().max()}'
    print(f"rank {RANK} check passed.✅")

    dist.destroy_process_group()

