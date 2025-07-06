# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from inspect import getattr_static
import tilelang
import tilelang.language as T
import sys
import os
import itertools

# Add your fla repository path to sys.path
# You can set the FLA_REPO_PATH environment variable to point to your fla repository
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae
sys.path.insert(0, "/root/workspace/flash-linear-attention")
import fla
print(fla.__file__)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
import torch
from tilelang.engine.callback import register_cuda_postproc_callback

torch.random.manual_seed(0)

tilelang.disable_cache()


def prepare_input(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
):
    BS = S // chunk_size
    K = torch.rand(B, S, H, DK, dtype=input_dtype).cuda()
    W = torch.rand(B, S, H, DK, dtype=input_dtype).cuda()
    U = torch.rand(B, S, H, DV, dtype=input_dtype).cuda()
    G = torch.rand(B, S, H, dtype=gate_dtype).cuda()
    initial_state = torch.rand(B, H, DK, DV, dtype=input_dtype).cuda()
    return K, W, U, G, initial_state


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
    state_dtype,
):
    BS = S // chunk_size
    h = torch.empty(B, BS, H, DK, DV, dtype=output_dtype).cuda()
    final_state = torch.empty(B, H, DK, DV, dtype=state_dtype).cuda()
    V_new = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return h, final_state, V_new


def get_configs():
    block_DK = [32, 64]
    block_DV = [32, 64]
    # block_DV = [32, 64, 128]
    threads = [128]
    num_stages = [0, 1]
    _configs = itertools.product(block_DK, block_DV, threads, num_stages)
    configs = [{
        "block_DK": block_DK,
        "block_DV": block_DV,
        "threads": threads,
        "num_stages": num_stages,
    } for block_DK, block_DV, threads, num_stages in _configs]
    return configs


def get_best_config(
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    use_g,
    use_initial_state,
    store_final_state,
    save_new_value,
):
    block_S = chunk_size

    def tilelang_chunk_gated_delta_rule_fwd_h(
        # kernel config
        block_DK=None,
        block_DV=None,
        threads=None,
        num_stages=None,
    ):
        BS = S // block_S

        K_shape = (B, S, H, DK)
        V_shape = (B, S, H, DV)
        W_shape = (B, S, H, DK)
        U_shape = (B, S, H, DV)
        G_shape = (B, S, H)
        h_shape = (B, BS, H, DK, DV)
        initial_state_shape = (B, H, DK, DV)
        final_state_shape = (B, H, DK, DV)
        
        @T.prim_func
        def kernel(
            K: T.Tensor(K_shape, dtype=input_dtype),
            W: T.Tensor(W_shape, dtype=input_dtype),
            U: T.Tensor(U_shape, dtype=input_dtype),
            G: T.Tensor(G_shape, dtype=gate_dtype),
            initial_state: T.Tensor(initial_state_shape, dtype=input_dtype),
            h: T.Tensor(h_shape, dtype=output_dtype),
            final_state: T.Tensor(final_state_shape, dtype=state_dtype),
            V_new: T.Tensor(V_shape, dtype=output_dtype),
        ):
            with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
                bb, bh = bbh // H, bbh % H
                
                b_h_shared = T.alloc_shared((DK, block_DV), dtype=input_dtype)
                b_h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)

                U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
                U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
                W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
                V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
                V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
                K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
                G_last_local = T.alloc_local((1), dtype=gate_dtype)
                G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype)
                
                T.annotate_layout({
                    b_h_shared: tilelang.layout.make_swizzled_layout(b_h_shared),
                    U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                    W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                    V_new_shared: tilelang.layout.make_swizzled_layout(V_new_shared),
                    K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                    G_shared: tilelang.layout.make_swizzled_layout(G_shared),
                })

                if use_initial_state:
                    T.copy(initial_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], b_h_shared)
                    T.copy(b_h_shared, b_h_fragment)
                else:
                    T.clear(b_h_fragment)
                
                for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                    # Store previous result to the hidden tensor, like the epilogue
                    T.copy(b_h_shared, h[bb, i_s, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])
                    
                    # Recurrence
                    T.copy(W[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], W_shared)
                    T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)
                    
                    # U - W * S
                    T.copy(U[bb, i_s * block_S:(i_s + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV], U_shared)
                    T.copy(U_shared, U_fragment)
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        V_new_fragment[i_s2, i_v] = - V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

                    # Save V_new
                    if save_new_value:
                        T.copy(V_new_fragment, dst=V_new_shared)
                        T.copy(V_new_shared, V_new[bb, i_s * block_S:(i_s + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV])

                    # use_g
                    T.copy(K[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], K_shared)
                    if use_g:
                        G_last_local[0] = G[bb, (i_s + 1) * block_S - 1, bh]
                        for i_s2, i_v in T.Parallel(block_S, block_DV):
                            G_shared[i_s2, i_v] = G[bb, i_s * block_S + i_s2, bh]
                        for i_s2, i_v in T.Parallel(block_S, block_DV):
                            with T.If(G_last_local[0] - G_shared[i_s2, i_v] <= 0):
                                with T.Then():
                                    V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(G_last_local[0] - G_shared[i_s2, i_v])
                                with T.Else():
                                    V_new_fragment[i_s2, i_v] = 0
                        G_last_local[0] = T.exp(G_last_local[0])
                        for i_k, i_v in T.Parallel(DK, block_DV):
                            b_h_fragment[i_k, i_v] *= G_last_local[0]
                    
                    # Update intermediate results
                    T.copy(V_new_fragment, V_new_shared)
                    T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)
                    
                    T.copy(b_h_fragment, b_h_shared)

                # Save final state
                if store_final_state:
                    T.copy(b_h_fragment, final_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

        return kernel

    autotuner = tilelang.autotuner.AutoTuner.from_kernel(
        kernel=tilelang_chunk_gated_delta_rule_fwd_h,
        configs=get_configs()
    ).set_compile_args(
        out_idx=[-3, -2, -1],
        target="auto",
        # pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True},
    ).set_profile_args(
        supply_type=tilelang.TensorSupplyType.Auto,
    )
    return autotuner.run(warmup=3, rep=10)


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    use_g,
    use_initial_state,
    store_final_state,
    save_new_value,
    block_DK,
    block_DV,
    threads,
    num_stages,
):
    input_dtype_torch = getattr(torch, input_dtype)
    output_dtype_torch = getattr(torch, output_dtype)
    accum_dtype_torch = getattr(torch, accum_dtype)
    gate_dtype_torch = getattr(torch, gate_dtype)
    state_dtype_torch = getattr(torch, state_dtype)
    
    K, W, U, G, initial_state = prepare_input(
        B, S, H, DK, DV, chunk_size, input_dtype_torch, output_dtype_torch, accum_dtype_torch, gate_dtype_torch
    )
    h_ref, final_state_ref, V_new_ref = prepare_output(
        B, S, H, DK, DV, chunk_size, output_dtype_torch, state_dtype_torch
    )
    
    # fla ref
    h_ref, V_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h(
        K, W, U, G, initial_state, store_final_state, chunk_size, save_new_value
    )

    autotune_result = get_best_config(B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype, chunk_size, use_g, use_initial_state, store_final_state, save_new_value)
    print(autotune_result.config)
    best_kernel = autotune_result.kernel
    profiler = best_kernel.get_profiler()
    tilelang_latency = profiler.do_bench()
    print(f"tilelang latency: {tilelang_latency:.2f} ms")
    # print(best_kernel.get_kernel_source())


if __name__ == "__main__":
    run_test(
        B=1,
        S=32768,
        H=32,
        DK=128,
        DV=128,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=64,
        use_g=True,
        use_initial_state=True,
        store_final_state=True,
        save_new_value=True,
        block_DK=64,
        block_DV=32,
        threads=128,
        num_stages=3,
    )
