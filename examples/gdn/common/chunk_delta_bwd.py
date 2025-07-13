# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import sys

# Add your fla repository path to sys.path
# You can set the FLA_REPO_PATH environment variable to point to your fla repository
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae

sys.path.insert(0, "/root/workspace/flash-linear-attention")
import fla
print(fla.__file__)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
import torch

torch.random.manual_seed(0)
# torch.set_printoptions(profile="full")

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
    state_dtype,
):
    Q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    W = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    G = torch.randn(B, S, H, dtype=gate_dtype).cuda()
    h0 = torch.randn(B, H, DK, DV, dtype=input_dtype).cuda()
    dht = torch.randn(B, H, DK, DV, dtype=input_dtype).cuda()
    dO = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    dv = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    return Q, K, W, G, h0, dht, dO, dv


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
    gate_dtype,
    state_dtype,
):
    BS = S // chunk_size
    dh = torch.empty(B, BS, H, DK, DV, dtype=output_dtype).cuda()
    dh0 = torch.empty(B, H, DK, DV, dtype=state_dtype).cuda()
    dv2 = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return dh, dh0, dv2


def tilelang_chunk_gated_delta_rule_bwd_dhu(
    # task config
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
    scale,
    use_g=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    # kernel config
    block_DV=64,
    threads=256,
    num_stages=0,
):
    block_S = chunk_size
    # Should support cu_seqlen
    BS = S // block_S

    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    W_shape = (B, S, H, DK)
    G_shape = (B, S, H)
    h0_shape = (B, H, DK, DV)
    dht_shape = (B, H, DK, DV)
    dO_shape = (B, S, H, DV)
    dv_shape = (B, S, H, DV)

    dh_shape = (B, BS, H, DK, DV)
    dh0_shape = (B, H, DK, DV)
    dv2_shape = (B, S, H, DV)
    
    @T.prim_func
    def kernel(
        # Input
        Q: T.Tensor(Q_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        h0: T.Tensor(h0_shape, dtype=input_dtype),
        dht: T.Tensor(dht_shape, dtype=input_dtype),
        dO: T.Tensor(dO_shape, dtype=input_dtype),
        dv: T.Tensor(dv_shape, dtype=input_dtype),
        # Output
        dh: T.Tensor(dh_shape, dtype=output_dtype),
        dh0: T.Tensor(dh0_shape, dtype=state_dtype),
        dv2: T.Tensor(dv2_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            b_dh_shared = T.alloc_shared((DK, block_DV), dtype=output_dtype)
            b_dh_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            b_dh_fragment_1 = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            b_dh_fragment_2 = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            dv_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            dv_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dv_fragment_2 = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dO_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)

            Q_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)

            G_last_local = T.alloc_local((1), dtype=gate_dtype)
            G_last_local_exp = T.alloc_local((1), dtype=gate_dtype)
            G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype)
            G_fragment = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)
            G_fragment_exp = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)
            
            if use_final_state_gradient:
                T.copy(dht[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], b_dh_shared)
                T.copy(b_dh_shared, b_dh_fragment)
            else:
                T.clear(b_dh_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # The gradient should be stored in the reverse order
                i_s_inv = T.ceildiv(S, block_S) - i_s - 1

                # Store the updated dh
                T.copy(b_dh_shared, dh[bb, i_s_inv, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

                # Update dv
                T.copy(K[bb, i_s_inv * block_S:(i_s_inv + 1) * block_S, bh, 0:DK], K_shared)
                T.gemm(K_shared, b_dh_shared, dv_fragment, clear_accum=True)

                if use_g:
                    G_last_local[0] = G[bb, i_s_inv * block_S + block_S - 1, bh]
                    G_last_local_exp[0] = T.exp(G_last_local[0])
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        G_shared[i_s2, i_v] = G[bb, i_s_inv * block_S + i_s2, bh]
                    T.copy(G_shared, G_fragment)
                    T.copy(G_fragment, G_fragment_exp)
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        G_fragment_exp[i_s2, i_v] = T.exp(G_fragment[i_s2, i_v])
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        with T.If(G_last_local[0] - G_fragment_exp[i_s2, i_v] <= 0):
                            with T.Then():
                                dv_fragment[i_s2, i_v] = dv_fragment[i_s2, i_v] * T.exp(G_last_local[0] - G_fragment[i_s2, i_v])
                            with T.Else():
                                dv_fragment[i_s2, i_v] = 0
                
                T.copy(dv[bb, i_s_inv * block_S:(i_s_inv + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV], dv_shared)
                T.copy(dv_shared, dv_fragment_2)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    dv_fragment[i_s2, i_v] = dv_fragment[i_s2, i_v] + dv_fragment_2[i_s2, i_v]

                # Store the updated dv
                T.copy(dv_fragment, dv_shared)
                T.copy(dv_shared, dv2[bb, i_s_inv * block_S:(i_s_inv + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV])

                # Update dh
                T.copy(Q[bb, i_s_inv * block_S:(i_s_inv + 1) * block_S, bh, 0:DK], Q_shared)
                T.copy(W[bb, i_s_inv * block_S:(i_s_inv + 1) * block_S, bh, 0:DK], W_shared)

                if use_g:
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_dh_fragment[i_k, i_v] *= G_last_local_exp[0]
                    for i_s2, i_k in T.Parallel(block_S, DK):
                        Q_shared[i_s2, i_k] = Q_shared[i_s2, i_k] * G_fragment_exp[i_s2, i_k]
                
                # may use fragment
                for i_s2, i_k in T.Parallel(block_S, DK):
                    Q_shared[i_s2, i_k] = Q_shared[i_s2, i_k] * scale
                
                T.copy(dO[bb, i_s_inv * block_S:(i_s_inv + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV], dO_shared)
                T.gemm(Q_shared, dO_shared, b_dh_fragment, transpose_A=True)
                T.clear(b_dh_fragment_2)
                T.gemm(W_shared, dv_shared, b_dh_fragment_2, transpose_A=True)
                for i_k, i_v in T.Parallel(DK, block_DV):
                    b_dh_fragment[i_k, i_v] = b_dh_fragment[i_k, i_v] - b_dh_fragment_2[i_k, i_v]

                T.copy(b_dh_fragment, b_dh_shared)

            if use_initial_state:
                T.copy(b_dh_fragment, dh0[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])
    
    return kernel


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
    scale,
    use_g=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    Q, K, W, G, h0, dht, dO, dv = prepare_input(
        B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, output_dtype), getattr(torch, accum_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )
    dh_ref, dh0_ref, dv2_ref = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )
    dh_tilelang, dh0_tilelang, dv2_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )
    
    # fla ref
    if use_g:
        dh_ref, dh0_ref, dv2_ref = chunk_gated_delta_rule_bwd_dhu(
            Q, K, W, G, h0, dht, dO, dv, scale
        )
    else:
        dh_ref, dh0_ref, dv2_ref = chunk_gated_delta_rule_bwd_dhu(
            Q, K, W, None, h0, dht, dO, dv, scale
        )

    # tilelang
    program = tilelang_chunk_gated_delta_rule_bwd_dhu(
        B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype, chunk_size, scale, use_g, use_initial_state, use_final_state_gradient, block_DV, threads, num_stages
    )
    kernel = tilelang.compile(program)
    print(kernel.get_kernel_source())
    kernel(Q, K, W, G, h0, dht, dO, dv, dh_tilelang, dh0_tilelang, dv2_tilelang)

    # fla_time = do_bench(chunk_gated_delta_rule_bwd_dhu, Q, K, W, None, h0, dht, dO, dv, scale, chunk_size=chunk_size)
    # tilelang_time = do_bench(kernel, Q, K, W, G, h0, dht, dO, dv, dh_tilelang, dh0_tilelang, dv2_tilelang)

    # print(f"tilelang time: {tilelang_time} ms")
    # print(f"fla time: {fla_time} ms")

    # check
    try:
        torch.testing.assert_close(dh_ref, dh_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
        print("tilelang chunk gated delta rule bwd dh passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule bwd dh failed ✗")
        print(e)
    
    try:
        torch.testing.assert_close(dh0_ref, dh0_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
        print("tilelang chunk gated delta rule bwd dh0 passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule bwd dh0 failed ✗")
        print(e)
        print("dh0_ref", dh0_ref)
        print("dh0_tilelang", dh0_tilelang)
    
    try:
        torch.testing.assert_close(dv2_ref, dv2_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
        print("tilelang chunk gated delta rule bwd dv2 passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule bwd dv2 failed ✗")
        print(e)


def do_bench(fn, *args, warmup=10, rep=10, **kwargs):
    """
    Do benchmark for a function.
    """
    import time
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for i in range(warmup):
        fn(*args, **kwargs)

    start_time = time.time()
    torch.cuda.synchronize()
    for i in range(rep):
        start_event[i].record()
        fn(*args, **kwargs)
        end_event[i].record()
    torch.cuda.synchronize()
    end_time = time.time()

    # Record clocks
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )

    return times.mean().item()


if __name__ == "__main__":
    DK = 128
    run_test(
        B=1,
        S=512,
        H=4,
        DK=DK,
        DV=128,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=64,
        scale=DK ** -0.5,
        use_g=False,
        use_initial_state=True,
        use_final_state_gradient=True,
        block_DV=32,
        threads=128,
        num_stages=0,
    )
    # run_test(
    #     B=1,
    #     S=256,
    #     H=2,
    #     DK=DK,
    #     DV=128,
    #     input_dtype="bfloat16",
    #     output_dtype="bfloat16",
    #     accum_dtype="float32",
    #     gate_dtype="float32",
    #     state_dtype="float32",
    #     chunk_size=64,
    #     scale=DK ** -0.5,
    #     use_g=False,
    #     use_initial_state=True,
    #     use_final_state_gradient=True,
    #     block_DV=32,
    #     threads=128,
    #     num_stages=0,
    # )
