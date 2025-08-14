import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from tilelang import register_cuda_postproc
from tvm import tir, DataType
import itertools
import torch
import argparse

tilelang.disable_cache()

torch.manual_seed(0)
# torch.set_printoptions(profile="full")

# @register_cuda_postproc
# def tilelang_callback_cuda_postproc(code, _):
#     cuda_code = ""
#     with open("examples/dequantize_gemm/tilelang_jit_kernel_kernel_func_shared.c", "r") as f:
#     # with open("examples/dequantize_gemm/tilelang_jit_kernel_test_convert_backup.c", "r") as f:
#         cuda_code = f.read()
#     return cuda_code


def _tir_u8_to_f4_to_bf16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr,
                          dtype: str):
    assert nbit == 4
    assert dtype == "bfloat16"
    assert val.dtype == "uint8"
    mask = tir.const((1 << nbit) - 1, "uint16")
    f4 = (val >> (pos.astype("uint16") * tir.const(nbit, "uint16"))) & mask
    s = f4 >> tir.const(3, "uint16")
    e_f4 = (f4 & tir.const(6, "uint16")) >> tir.const(1, "uint16")
    # Exponential bias between f4 and bf16 is 2^(8-1) - 2^(2-1) = 126
    e_bf16 = e_f4 + tir.const(126, "uint16")
    # Scale is the exponential part, within the representation of uint8
    # To handle the overflow, we use the max function to limit the exponential part to 8 bits
    e_bf16 = T.min(e_bf16 + scale, tir.const((1 << 8) - 1, "uint16"))
    m_f4 = f4 & tir.const(1, "uint16")
    val_bf16 = tir.reinterpret("bfloat16",
                               ((((s << tir.const(8, "uint16")) | e_bf16) << tir.const(7, "uint16"))
                                | (m_f4 << tir.const(6, "uint16"))).astype("uint16"))
    return val_bf16


def _tir_u8_to_f4_to_f16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    # FIXME: Assertion will cause undefined behavior in PrimFunc.
    # assert nbit == 4
    # assert dtype == "float16"
    # assert val.dtype == "uint8"
    # e_f4 == 0 -> e_f16 = 0
    # e_f4 != 0 -> e_f16 = e_f4 + ExponentialBias(f16, f4) = e_f4 + (2^4 - 2^1) = e_f4 + 14
    # s1e2m1
    # return val
    mask = tir.const((1 << nbit) - 1, "uint16")
    f4 = (val.astype("uint8") >> (pos.astype("uint16") * tir.const(nbit, "uint16"))) & mask
    s = f4 >> tir.const(3, "uint16")
    e_f4 = (f4 & tir.const(6, "uint16")) >> tir.const(1, "uint16")
    e_f16 = e_f4 + tir.const(14, "uint16")
    m_f4 = f4 & tir.const(1, "uint16")
    m_f16 = m_f4
    val_f16 = tir.reinterpret("float16",
                              ((e_f16 | (s << tir.const(5, "uint16"))) << tir.const(10, "uint16")
                               | m_f16 << tir.const(9, "uint16")).astype("uint16"))
    # return tir.Select(e_f4 == tir.const(0, "uint32"), tir.const(0, "float16"), val_f16)
    return val_f16


def torch_convert(tensor):

    def print_bit(name, val):
        val_cpu = val.cpu().item()
        binary_repr = f'{val_cpu:032b}'
        print(name, binary_repr)

    def _convert(val, pos):
        assert val.dtype == torch.uint8
        val = val.view(torch.uint8)
        mask = (1 << 4) - 1
        f4 = ((val >> (pos * 4)) & mask).to(torch.int16)
        s = f4 >> 3
        e_f4 = (f4 & 6) >> 1
        e_f16 = e_f4 + 14
        m_f4 = f4 & 1
        m_f16 = m_f4
        val_f16 = (((e_f16 | (s << 5)) << 10) | (m_f16 << 9)) & 0xFFFF
        lower_16_bits = (val_f16 & 0xFFFF).to(torch.uint16)
        return lower_16_bits.view(torch.float16)

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.float16, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2)
    return new_tensor


def torch_convert_bit_twiddling(tensor):
    
    def print_bit(name, val):
        val_cpu = val.cpu().item()
        binary_repr = f'{val_cpu:032b}'
        print(name, binary_repr)

    def _convert(val0, val1, pos) -> torch.bfloat16:
        assert val0.dtype == torch.uint8
        assert val1.dtype == torch.uint8
        val0 = val0.view(torch.uint8)
        val1 = val1.view(torch.uint8)
        val_concat = (val0.item() << 8) | val1.item()
        mask = 0b1000000111000000
        if pos == 0:
            bf16 = val_concat & mask
        elif pos == 1:
            bf16 = (val_concat << 3) & mask
        elif pos == 2:
            bf16 = (val_concat << 6) & mask
        elif pos == 3:
            mask1 = 0b1000000000000000
            mask2 = 0b0000000110000000  
            mask3 = 0b0000000001000000
            bf16 = ((val_concat << 1) & mask1) | ((val_concat >> 3) & mask2) | ((val_concat >> 7) & mask3)
        bf16_new = torch.tensor([bf16], dtype=torch.uint16, device=val0.device).view(torch.bfloat16)
        # Add bias for change from fp4 to bf16
        bf16_new = bf16_new.item() * (2 ** 126)
        return bf16_new

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.bfloat16, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 4 * 2], tensor[i][j // 4 * 2 + 1], j % 4)
    return new_tensor  


@tilelang.jit(out_idx=[1])
def test_convert(N, K, block_N, block_K, in_dtype, num_bits=4, threads=128):
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "uint8"
    B_shape = (N, K // num_elems_per_byte)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)

    @T.prim_func
    def main(
            B: T.Tensor(B_shape, storage_dtype),
            C: T.Tensor((N, K), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                T.copy(B_shared, B_local)
                for i, j in T.Parallel(block_N, block_K):
                    B_dequantize_local[i, j] = _tir_u8_to_f4_to_bf16(
                        num_bits,
                        B_local[i, j // num_elems_per_byte],
                        j % num_elems_per_byte,
                        0,
                        dtype=in_dtype,
                    )
                T.copy(B_dequantize_local, C[bx * block_N, k * block_K])

    return main


def test_fp4_bf16_convert_close():
    N, K = 256, 256
    block_N, block_K = 64, 64
    kernel = test_convert(
        N,
        K,
        block_N,
        block_K,
        "bfloat16",
    )

    B = torch.ones(N, K // 2, dtype=torch.uint8, device="cuda").to(torch.uint8)
    tl_out = kernel(B)
    ref_out = torch_convert_bit_twiddling(B)
    print(tl_out)
    assert torch.testing.assert_close(tl_out, ref_out, rtol=0.01, atol=0.01)
    print("Pass")


def get_configs():
    block_M = [128]
    block_N = [128, 256]
    block_K = [128]
    num_stages = [2]
    threads = [256]
    splits = [1]
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads, splits))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'block_K': c[2],
        'num_stages': c[3],
        'threads': c[4],
        'split': c[5]
    } for c in _configs]
    return configs


def matmul(M, N, K, in_dtype, out_dtype, accum_dtype, source_format='uint', num_bits=4, tune=False):

    @tilelang.jit(out_idx=[-1],
                  pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def kernel_func(block_M, block_N, block_K, num_stages, threads, split=1):
        num_elems_per_byte = 8 // num_bits
        storage_dtype = "uint8"
        A_shape = (M, K)
        B_shape = (N, K // num_elems_per_byte)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)
        assert K % (block_K * split) == 0
        KK = K // split

        # Some variables for serial dequant in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // DataType(in_dtype).bits
        local_compress_size = local_size // num_elems_per_byte

        from tilelang.quantize.mxfp import get_mxfp_intrin_group
        mxfp_intrin_info = get_mxfp_intrin_group(
            out_dtype=in_dtype,
            source_format=source_format,
            source_bit=num_bits,
            storage_dtype=storage_dtype,
            with_scaling=False,
            with_zeros=False,
            use_twiddling=True,
        )
        import_source = mxfp_intrin_info["c_source"]
        func_name = mxfp_intrin_info["func_name"]
        assert import_source is not None, "mxfp_intrin_info is not found"
        assert func_name is not None, "mxfp_intrin_info is not found"
        import_source = import_source
        vectorize_dequant_size = 8

        @T.prim_func
        def main_split(
                A: T.Tensor(A_shape, in_dtype),
                B: T.Tensor(B_shape, storage_dtype),
                Ct: T.Tensor((N, M), out_dtype),
        ):
            SplitC = T.alloc_buffer([
                split, (N + block_N - 1) // block_N * block_N,
                (M + block_M - 1) // block_M * block_M
            ], out_dtype)
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), split,
                    threads=threads) as (bx, by, bz):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_fragment = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                B_dequantize_prev_fragment = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

                T.annotate_layout({
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    Ct_shared: tilelang.layout.make_swizzled_layout(Ct_shared),
                })

                T.clear(Ct_local)
                for k in T.Pipelined(K // (block_K * split), num_stages=num_stages):
                    T.copy(A[by * block_M, KK * bz + k * block_K], A_shared)
                    T.copy(B[bx * block_N, (KK * bz + k * block_K) // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_fragment[i, j] = _tir_u8_to_f4_to_bf16(
                            num_bits,
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                            0,
                            dtype=in_dtype,
                        )
                    T.copy(B_dequantize_fragment, B_dequantize_prev_fragment)
                    T.gemm(B_dequantize_prev_fragment, A_shared, Ct_local, transpose_B=True)
                T.copy(Ct_local, SplitC[bz, bx * block_N:(bx + 1) * block_N,
                                        by * block_M:(by + 1) * block_M])
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
                acc = T.alloc_fragment((block_N, block_M), out_dtype)
                T.clear(acc)
                for k in range(split):
                    for i, j in T.Parallel(block_N, block_M):
                        acc[i, j] += SplitC[k, bx * block_N + i, by * block_M + j]
                T.copy(acc, Ct[bx * block_N, by * block_M])

        @T.prim_func
        def main(
                A: T.Tensor(A_shape, in_dtype),
                B: T.Tensor(B_shape, storage_dtype),
                C: T.Tensor((M, N), out_dtype),
                # B_dequantize: T.Tensor((N, K), out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
                
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)

                T.annotate_layout({
                    C_shared: tilelang.layout.make_swizzled_layout(C_shared),
                })

                T.import_source(import_source)

                element_per_thread = block_N * block_K // threads

                T.clear(C_local)
                for k in T.Pipelined(K // block_K, num_stages=0):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

                    for i, j in T.Parallel(block_N, block_K // 8):
                        T.call_extern(
                            func_name,
                            T.address_of(B_shared[i, j * 8 // num_elems_per_byte]),
                            T.address_of(B_dequantize_shared[i, j * 8]),
                            1,
                            dtype=in_dtype,
                        )
                    T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)
                
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        if split == 1:
            return main
        else:
            return main_split

    if tune:

        @autotune(
            configs=get_configs(),
            keys=["block_M", "block_N", "block_K", "num_stages", "threads", "split"],
            warmup=10,
            rep=10)
        @tilelang.jit(out_idx=[2])
        def kernel(block_M=None,
                   block_N=None,
                   block_K=None,
                   num_stages=None,
                   threads=None,
                   split=None):
            return kernel_func(block_M, block_N, block_K, num_stages, threads, split)

        return kernel()
    else:

        def kernel(block_M, block_N, block_K, num_stages, threads, split=1):
            return kernel_func(block_M, block_N, block_K, num_stages, threads, split)

        return kernel


def ref_program(A, qB):
    dtypeC = "bfloat16"
    # B = torch_convert(qB)
    B = torch_convert_bit_twiddling(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C


def main(m=256, n=256, k=256, tune=False):
    total_flops = 2 * m * n * k

    if (not tune):
        kernel = matmul(
            m, n, k, "bfloat16", "bfloat16", "float32", num_bits=4, tune=tune)(
                block_M=256, block_N=128, block_K=128, num_stages=2, threads=256, split=1)
        # print(kernel.get_kernel_source())
        profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)
        profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
        print("All checks pass.")
        # latency = profiler.do_bench(ref_program, warmup=500)
        # print("Ref: {:.2f} ms".format(latency))
        # print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(warmup=500)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        best_result = matmul(m, n, k, "bfloat16", "bfloat16", "float32", num_bits=4, tune=tune)
        best_latency = best_result.latency
        best_config = best_result.config
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=256, help='M')
    parser.add_argument('--n', type=int, default=256, help='N')
    parser.add_argument('--k', type=int, default=256, help='K')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    main(M, N, K, args.tune)
