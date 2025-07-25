# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import sys
import os

# Add your fla repository path to sys.path
# You can set the FLA_REPO_PATH environment variable to point to your fla repository
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae

sys.path.insert(0, "/root/workspace/flash-linear-attention")
import fla
print(fla.__file__)

sys.path.insert(0, "/root/workspace/tilelang")
import tilelang
import tilelang.language as T
print(tilelang.__file__)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.utils.cumsum import chunk_local_cumsum
import torch
import torch.nn.functional as F
from tilelang.engine.callback import register_cuda_postproc_callback



torch.random.manual_seed(0)

tilelang.disable_cache()

def ref_program(q, k, w, u, g, initial_state, output_final_state, cu_seqlens, scale):
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state

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
    # torch.empty(*shape, device=device, dtype=dtype).normal_(-1.0, 1.0)
    ext = 0.1
    Q = torch.empty(B, S, H, DK, dtype=input_dtype).cuda().normal_(-ext, ext)
    Q = F.normalize(Q, dim=-1, p=2)
    K = torch.empty(B, S, H, DK, dtype=input_dtype).cuda().normal_(-ext, ext)
    K = F.normalize(K, dim=-1, p=2)
    W = torch.empty(B, S, H, DK, dtype=input_dtype).cuda().normal_(-ext, ext)
    W = F.normalize(W, dim=-1, p=2)
    U = torch.empty(B, S, H, DV, dtype=input_dtype).cuda().normal_(-ext, ext)
    U = F.normalize(U, dim=-1, p=2)
    G = torch.empty(B, S, H, dtype=gate_dtype).cuda().normal_(-ext, ext)
    G = F.logsigmoid(G)
    G = chunk_local_cumsum(G, chunk_size)
    # G = torch.ones(B, S, H, dtype=gate_dtype).cuda() * 2
    initial_state = torch.empty(B, H, DK, DV, dtype=input_dtype).cuda().normal_(-ext, ext)
    return Q, K, W, U, G, initial_state


def prepare_input_fake(
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
    Q = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    K = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    W = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    U = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    G = torch.zeros(B, S, H, dtype=gate_dtype).cuda()
    initial_state = torch.ones(B, H, DK, DV, dtype=input_dtype).cuda()
    return Q, K, W, U, G, initial_state


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


def tilelang_chunk_gated_delta_rule_chunk_o_fwd_varlen(
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
    use_g=True,
    use_initial_state=True,
    use_varlen=False,
    store_final_state=True,
    save_new_value=True,
    scale=1.0,
    cu_seqlens=None,
    chunk_num_global=None,
    # kernel config
    block_DK=64,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    block_S = chunk_size
    # Support cu_seqlens
    seq_num = len(cu_seqlens) - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    total_seqlen = cu_seqlens[-1].item()
    K_shape = (1, total_seqlen, H, DK)
    V_shape = (1, total_seqlen, H, DV)
    W_shape = (1, total_seqlen, H, DK)
    U_shape = (1, total_seqlen, H, DV)
    O_shape = (1, total_seqlen, H, DV)
    G_shape = (1, total_seqlen, H)
    h_shape = (1, chunk_num_global, H, DK, DV)
    initial_state_shape = (seq_num, H, DK, DV)
    final_state_shape = (seq_num, H, DK, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(K_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        U: T.Tensor(U_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        initial_state: T.Tensor(initial_state_shape, dtype=input_dtype),
        cu_seqlens: T.Tensor((seq_num + 1), dtype="int32"),
        seqlens: T.Tensor((seq_num), dtype="int32"),
        chunk_num_list_cumsum: T.Tensor((seq_num), dtype="int32"),
        # intermediate
        h: T.Tensor(h_shape, dtype=output_dtype),
        # output
        final_state: T.Tensor(final_state_shape, dtype=state_dtype),
        V_new: T.Tensor(V_shape, dtype=output_dtype),
        O: T.Tensor(O_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), H, seq_num, threads=threads) as (bv, bh, bs):
            Q_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            O_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)

            b_h_shared = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            b_h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)

            U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            G_last_local = T.alloc_local((1), dtype=gate_dtype)
            G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype, scope="shared")
            G_fragment = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            G_diff_local = T.alloc_fragment((block_S, block_S), dtype=gate_dtype)

            seqlen = T.alloc_local((1), dtype="int32")
            seq_start = T.alloc_local((1), dtype="int32")
            seq_end = T.alloc_local((1), dtype="int32")
            chunk_num = T.alloc_local((1), dtype="int32")
            chunk_id_offset = T.alloc_local((1), dtype="int32")
            # chunk_start = T.alloc_local((1), dtype="int32")
            # chunk_end = T.alloc_local((1), dtype="int32")
            chunk_id_global = T.alloc_local((1), dtype="int32")

            # T.no_set_max_nreg()

            T.annotate_layout({
                b_h_shared: tilelang.layout.make_swizzled_layout(b_h_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                V_new_shared: tilelang.layout.make_swizzled_layout(V_new_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                # G_shared: tilelang.layout.make_swizzled_layout(G_shared),
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                # G_diff_local: T.Fragment(G_diff_local.shape, forward_thread_fn=lambda i: i // 16 * 32 + i % 8 * 4),
            })

            # current seqlen = seqlens[bs]
            # current seq_start = cu_seqlens[bs]
            # current seq_end = cu_seqlens[bs + 1]
            # current chunk_num = ceildiv(seq_end - seq_start, block_S)
            # current chunk_id_offset = chunk_num_list_cumsum[bs]

            seqlen[0] = seqlens[bs]
            seq_start[0] = cu_seqlens[bs]
            seq_end[0] = cu_seqlens[bs + 1]
            chunk_num[0] = T.ceildiv(seqlen[0], block_S)
            chunk_id_offset[0] = chunk_num_list_cumsum[bs]

            if use_initial_state:
                T.copy(initial_state[bs, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], b_h_shared)
                T.copy(b_h_shared, b_h_fragment)
            else:
                T.clear(b_h_fragment)

            for i_s in T.Pipelined(chunk_num[0], num_stages=num_stages):
                # current chunk_start = cu_seqlens[bs] + i_s * block_S
                # current chunk_end = min(cu_seqlens[bs] + (i_s + 1) * block_S, seq_end)
                # current chunk_id_global = chunk_id_offset + i_s

                chunk_start = seq_start[0] + i_s * block_S
                chunk_end = seq_start[0] + (i_s + 1) * block_S
                cur_block_S = block_S
                if chunk_end > seq_end[0]:
                    chunk_end = seq_end[0]
                    cur_block_S = chunk_end - chunk_start
                chunk_id_global = chunk_id_offset[0] + i_s

                # Store previous result to the hidden tensor, like the epilogue
                T.copy(b_h_shared, h[0, chunk_id_global, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

                # Cast the intermediate result to input dtype
                # T.copy(b_h, b_h_input_dtype)

                # Recurrence
                # T.clear(V_new_fragment)
                for i_s2, i_k in T.Parallel(cur_block_S, DK):
                    W_shared[i_s2, i_k] = W[0, chunk_start + i_s2, bh, i_k]
                T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)

                # U - W * S
                for i_s2, i_v in T.Parallel(cur_block_S, block_DV):
                    U_shared[i_s2, i_v] = U[0, chunk_start + i_s2, bh, bv * block_DV + i_v]
                T.copy(U_shared, U_fragment)
                for i_s2, i_v in T.Parallel(cur_block_S, block_DV):
                    V_new_fragment[i_s2, i_v] = - V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

                # Save V_new
                if save_new_value:
                    T.copy(V_new_fragment, V_shared)
                    for i_s2, i_v in T.Parallel(cur_block_S, block_DV):
                        V_new[0, chunk_start + i_s2, bh, bv * block_DV + i_v] = V_shared[i_s2, i_v]

                for i_s2, i_k in T.Parallel(cur_block_S, DK):
                    K_shared[i_s2, i_k] = K[0, chunk_start + i_s2, bh, i_k]
                # use_g
                if use_g:
                    G_last_local[0] = G[0, chunk_end - 1, bh]
                    for i_s2, i_v in T.Parallel(cur_block_S, block_DV):
                        G_shared[i_s2, i_v] = G[0, chunk_start + i_s2, bh]
                    T.copy(G_shared, G_fragment)
                    for i_s2, i_v in T.Parallel(cur_block_S, block_DV):
                        # with T.If(G_last_local[0] - G_shared[i_s2, i_v] <= 0):
                        # with T.If(G_last_local[0] - G_fragment[i_s2, i_v] <= 0):
                        with T.If(G_last_local[0] - G[0, chunk_start + i_s2, bh] <= 0):
                            with T.Then():
                                # V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(G_last_local[0] - G_fragment[i_s2, i_v])
                                # V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(G_last_local[0] - G_shared[i_s2, i_v])
                                V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(G_last_local[0] - G[0, chunk_start + i_s2, bh])
                            with T.Else():
                                V_new_fragment[i_s2, i_v] = 0
                    G_last_local[0] = T.exp(G_last_local[0])
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_h_fragment[i_k, i_v] *= G_last_local[0]
                        # b_h_shared[i_k, i_v] *= G_last_local[0]

                    # T.copy(b_h_shared, b_h_fragment)

                # Update intermediate results
                T.copy(V_new_fragment, V_new_shared)
                T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)

                T.clear(A_fragment)
                T.clear(O_fragment)
                for i_s2, i_k in T.Parallel(cur_block_S, DK):
                    Q_shared[i_s2, i_k] = Q[0, chunk_start + i_s2, bh, i_k]
                T.gemm(Q_shared, b_h_shared, O_fragment)
                T.gemm(Q_shared, K_shared, A_fragment, transpose_B=True)

                T.copy(b_h_fragment, b_h_shared)

                if use_g:
                    for p_s, p_v in T.Parallel(cur_block_S, block_DV):
                        O_fragment[p_s, p_v] = O_fragment[p_s, p_v] * T.exp(G[0, chunk_start + p_s, bh])
                    for p_s1, p_s2 in T.Parallel(cur_block_S, cur_block_S):
                        G_diff_local[p_s1, p_s2] = G[0, chunk_start + p_s1, bh] - G[0, chunk_start + p_s2, bh]
                    for p_s1, p_s2 in T.Parallel(cur_block_S, cur_block_S):
                        # with T.If(G_diff_local[p_s1, p_s2] <= 0):
                        with T.If(G[0, chunk_start + p_s1, bh] - G[0, chunk_start + p_s2, bh] <= 0):
                            with T.Then():
                                # A_fragment[p_s1, p_s2] *= T.exp(G_diff_local[p_s1, p_s2])
                                A_fragment[p_s1, p_s2] *= T.exp(G[0, chunk_start + p_s1, bh] - G[0, chunk_start + p_s2, bh])
                            with T.Else():
                                A_fragment[p_s1, p_s2] = 0

                for p_s1, p_s2 in T.Parallel(cur_block_S, cur_block_S):
                    with T.If(p_s1 < p_s2):
                        with T.Then():
                            A_fragment[p_s1, p_s2] = 0

                T.copy(A_fragment, A_shared)
                T.gemm(A_shared, V_shared, O_fragment)

                for p_s, p_v in T.Parallel(cur_block_S, block_DV):
                    O_fragment[p_s, p_v] = O_fragment[p_s, p_v] * scale

                T.copy(O_fragment, O_shared)
                for i_s2, i_v in T.Parallel(cur_block_S, block_DV):
                    O[0, chunk_start + i_s2, bh, bv * block_DV + i_v] = O_shared[i_s2, i_v]

            # Save final state
            if store_final_state:
                T.copy(b_h_fragment, final_state[bs, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

    return kernel


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

    # return (end_time - start_time) * 1000 / rep
    return times.mean().item()

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
    use_g=True,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    scale=1.0,
    block_DK=64,
    block_DV=32,
    threads=128,
    num_stages=0,
    use_varlen=True,
):
    Q, K, W, U, G, initial_state = prepare_input(
        B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, output_dtype), getattr(torch, accum_dtype), getattr(torch, gate_dtype)
    )
    # cu_seqlens = torch.LongTensor([0, S//4, 2*S//4, 3*S//4, S]).cuda().to(torch.int32)
    # cu_seqlens = torch.LongTensor([0, S//2, S]).cuda().to(torch.int32)
    cu_seqlens = torch.LongTensor([0, S]).cuda().to(torch.int32)
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    total_seqlen = cu_seqlens[-1]
    seq_num = len(cu_seqlens) - 1

    if use_varlen:
        Q = Q.view(1, -1, H, DK)
        K = K.view(1, -1, H, DK)
        W = W.view(1, -1, H, DK)
        U = U.view(1, -1, H, DV)
        G = G.view(1, -1, H)
        # initial_state = torch.empty(len(seqlens), H, DK, DV, dtype=getattr(torch, input_dtype)).cuda().normal_(-0.1, 0.1)
        initial_state = torch.ones(len(seqlens), H, DK, DV, dtype=getattr(torch, input_dtype)).cuda()

    # fla ref
    if use_g:
        o_ref, final_state_ref = ref_program(Q, K, W, U, G, initial_state, output_final_state=True, cu_seqlens=cu_seqlens, scale=scale)
    else:
        o_ref, final_state_ref = ref_program(Q, K, W, U, None, initial_state, output_final_state=True, cu_seqlens=cu_seqlens, scale=scale)

    # calculate the total chunk number of cu_seqlens
    chunk_num_list = torch.LongTensor([seq_len // chunk_size + seq_len % chunk_size for seq_len in seqlens]).cuda().to(torch.int32)
    chunk_num_list_cumsum = torch.cumsum(chunk_num_list, dim=0).to(torch.int32)
    chunk_num_global = chunk_num_list_cumsum[-1].item()

    # tilelang
    h_tilelang = torch.empty(1, chunk_num_global, H, DK, DV, dtype=getattr(torch, output_dtype)).cuda()
    V_new_tilelang = torch.empty(1, total_seqlen, H, DV, dtype=getattr(torch, output_dtype)).cuda()
    o_tilelang = torch.empty(1, total_seqlen, H, DV, dtype=getattr(torch, output_dtype)).cuda()
    final_state_tilelang = torch.empty(seq_num, H, DK, DV, dtype=getattr(torch, state_dtype)).cuda()

    program = tilelang_chunk_gated_delta_rule_chunk_o_fwd_varlen(
        B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype, chunk_size, use_g, use_initial_state, use_varlen, store_final_state, save_new_value, scale, cu_seqlens, chunk_num_global, block_DK, block_DV, threads, num_stages
    )
    kernel = tilelang.compile(program)
    print(kernel.get_kernel_source())
    kernel(Q, K, W, U, G, initial_state, cu_seqlens, seqlens, chunk_num_list_cumsum, h_tilelang, final_state_tilelang, V_new_tilelang, o_tilelang)

    print("tilelang_o:", o_tilelang)
    print("o_ref:", o_ref)

    print("tilelang_final_state:", final_state_tilelang)
    print("final_state_ref:", final_state_ref)

    # fla_time = do_bench(chunk_gated_delta_rule_fwd_h, K, W, U, G, initial_state, store_final_state, chunk_size, save_new_value)
    fla_time = do_bench(ref_program, Q, K, W, U, G, initial_state, store_final_state, cu_seqlens, 1.0)
    tilelang_time = do_bench(kernel, Q, K, W, U, G, initial_state, cu_seqlens, seqlens, chunk_num_list_cumsum, h_tilelang, final_state_tilelang, V_new_tilelang, o_tilelang)

    print(f"tilelang time: {tilelang_time} ms")
    print(f"fla time: {fla_time} ms")

    # check
    try:
        torch.testing.assert_close(o_ref, o_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
        print("tilelang chunk gated delta rule fwd o passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule fwd o failed ✗")
        # print("ref h:", h_ref)
        # print("tilelang h:", h_tilelang)
        print(e)

    try:
        torch.testing.assert_close(final_state_ref, final_state_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
        print("tilelang chunk gated delta rule fwd final_state passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule fwd final_state failed ✗")
        print(e)

    # try:
    #     torch.testing.assert_close(V_new_ref, V_new_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
    #     print("tilelang chunk gated delta rule fwd V_new passed √")
    # except Exception as e:
    #     print("tilelang chunk gated delta rule fwd V_new failed ✗")
    #     # print("ref V_new:", V_new_ref)
    #     # print("tilelang V_new:", V_new_tilelang)
    #     print(e)


if __name__ == "__main__":
    run_test(
        B=1,
        S=32768,
        H=8,
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
        scale=1.0,
        block_DK=64,
        block_DV=32,
        threads=128,
        num_stages=1,
        use_varlen=True,
    )
