import tilelang.language as T
import tilelang.testing
import tilelang
import torch
from tilelang.utils.tensor import map_torch_type


def matmu_jit_kernel(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_jit_kernel(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmu_jit_kernel(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    matmul_kernel = tilelang.compile(program, out_idx=-1, execution_backend="tvm_ffi")

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)

    A = torch.randn(M, K, dtype=in_dtype).cuda()
    B = torch.randn(K, N, dtype=in_dtype).cuda()

    if trans_A:
        A = A.T
    if trans_B:
        B = B.T

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(out_dtype)
        return C

    ref_C = ref_program(A, B)
    C = matmul_kernel(A, B)

    tilelang.testing.torch_assert_close(C, ref_C, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


def test_gemm_jit_kernel_zero_dim():
    run_gemm_jit_kernel(
        512,
        1024,
        0,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        256,
        32,
        2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
