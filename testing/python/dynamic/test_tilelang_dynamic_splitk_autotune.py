# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.rasterization import NoRasterization
from tilelang import tvm as tvm
import torch
import argparse
import itertools
import pandas as pd
from tvm import DataType
import os, time

torch.manual_seed(0)
tl.disable_cache()

kernel_save_dir = "testing/python/dynamic/.log"


def tl_matmul_block_static_splitk(
    M=16384,
    N=16384,
    K=16384,
    block_M=128,
    block_N=128,
    block_K=32,
    trans_A=False,
    trans_B=False,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float32",
    num_stages=3,
    thread_num=128,
    enable_rasteration=True,
    split_k=4,
):
    splitK = K // split_k

    @T.prim_func
    def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((K, N), in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=thread_num) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(10, enable=enable_rasteration)
            if bz == 0:
                # fuse the zero initialization kernel
                for i, j in T.Parallel(block_M, block_N):
                    m, n = by * block_M + i, bx * block_N + j
                    C[m, n] = T.cast(0, out_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[bz * splitK + ko * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, bz * splitK + ko * block_K], B_shared)
                else:
                    T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)

            if DataType(out_dtype).bits == 16:
                for i, j in T.Parallel(block_M, block_N // 2):
                    m, n = by * block_M + i, bx * block_N + j * 2
                    # vectorized atomic
                    T.atomic_addx2(C[m, n], C_shared[i, j * 2])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(C[by * block_M + i, bx * block_N + j], C_shared[i, j])

    return main


def tl_matmul_block_all_dynamic_splitk(
    block_M=128,
    block_N=128,
    block_K=32,
    trans_A=False,
    trans_B=False,
    in_type="float16",
    out_type="float16",
    accum_type="float32",
    num_stages=3,
    thread_num=128,
    enable_rasteration=True,
    split_k=4,
):
    M = tvm.te.var("m")
    N = tvm.te.var("n")
    K = tvm.te.var("k")

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    C_shape = (M, N)
    splitK = K // split_k

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_type),
        B: T.Tensor(B_shape, in_type),
        C: T.Tensor(C_shape, out_type),  
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=thread_num) as (bx, by, bz):
            A_shared = T.alloc_shared(A_shared_shape, in_type)
            B_shared = T.alloc_shared(B_shared_shape, in_type)
            C_shared = T.alloc_shared(C_shape, out_type)
            C_local = T.alloc_fragment((block_M, block_N), accum_type)

            T.use_swizzle(10, enable=enable_rasteration)
            if bz == 0:
                for i, j in T.Parallel(block_M, block_N):
                    m, n = by * block_M + i, bx * block_N + j
                    C[m, n] = T.cast(0, out_type)

            T.clear(C_local)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)

            if DataType(out_type).bits == 16:
                for i, j in T.Parallel(block_M, block_N // 2):
                    m, n = by * block_M + i, bx * block_N + j * 2
                    # vectorized atomic
                    T.atomic_addx2(C[m, n], C_shared[i, j * 2])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(C[by * block_M + i, bx * block_N + j], C_shared[i, j])

    return main


def benchmark_tl_matmul_block_static(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    autotune_configs,
    autotune=False,
    pass_configs=None,
):
    df_best_config = []

    def ref_program(A, B):
        import torch
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        out = torch.matmul(A.to(torch.float), B.to(torch.float))
        out = out.to(torch.__getattribute__(out_dtype))
        return out

    import time
    start_time = time.time()
    if not autotune:
        # Purely search to find the best config
        best_latency = float("inf")
        best_config = None
        ref_latency_list = []

        for autotune_config in autotune_configs:
            correctness = True
            block_M, block_N, block_K = autotune_config["block_M"], autotune_config["block_N"], autotune_config["block_K"]
            num_stages = autotune_config["num_stages"]
            thread_num = autotune_config["thread_num"]
            enable_rasteration = autotune_config["enable_rasteration"]
            split_k = autotune_config["split_k"]

            prim_func = tl_matmul_block_static_splitk(M, N, K, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, accum_dtype, num_stages, thread_num, enable_rasteration, split_k)
            
            A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
            B = torch.rand(K, N, device="cuda", dtype=getattr(torch, in_dtype))
            C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

            try:
                kernel = tl.compile(prim_func, pass_configs=pass_configs)
                kernel(A, B, C)
                torch.cuda.synchronize()
                ref_c = ref_program(A, B)
                torch.cuda.synchronize()
                torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
                profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Normal)
                latency = profiler.do_bench(input_tensors=[A, B, C])
                print(f"Config: {autotune_config}")
                print(f"tilelang_latency: {latency} ms", flush=True)
                if latency < best_latency and correctness:
                    best_latency = latency
                    best_config = autotune_config
                ref_latency = profiler.do_bench(ref_program, input_tensors=[A, B])
                ref_latency_list.append(ref_latency)
            except Exception as e:
                correctness = False
                print(f"Config: {autotune_config} causes error: {e}", flush=True)

        print(f"Static MNK ({M}, {N}, {K}) with pass_configs {pass_configs}:")
        print(f"Best Config: {best_config}")
        print(f"tilelang_latency: {best_latency} ms")
        print(f"ref_latency average: {sum(ref_latency_list) / len(ref_latency_list)} ms")
    else:
        torch.cuda.empty_cache()
        # Use Autotuner to find the best config
        autotuner = AutoTuner.from_kernel(
            kernel=tl_matmul_block_static_splitk, configs=autotune_configs).set_compile_args(
                out_idx=[-1],
                supply_type=tl.TensorSupplyType.Auto,
                ref_prog=ref_program,
                skip_check=False,
                target="auto")
        autotune_result = autotuner.run(warmup=3, rep=20)
        print(f"Best Config: {autotune_result.config} with latency: {autotune_result.latency} ms")
        kernel = autotune_result.kernel
        best_config = autotune_result.config
        best_latency = autotune_result.latency
        A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
        B = torch.rand(K, N, device="cuda", dtype=getattr(torch, in_dtype))

        profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
        profiler_latency = profiler.do_bench(input_tensors=[A, B])
        ref_latency = profiler.do_bench(ref_program, input_tensors=[A, B])
        profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
        print(f"Static MNK ({M}, {N}, {K}) with pass_configs {pass_configs}:")
        print(f"Best Config: {best_config} tilelang_latency: {best_latency} ms")
        print(f"Best Config: {best_config} ref_latency: {ref_latency} ms")

        # Free memory for precise measurement
        del A, B
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    df_best_config = {
        "M": M if not autotune else best_config[0],
        "N": N if not autotune else best_config[1],
        "K": K if not autotune else best_config[2],
        "block_M": best_config["block_M"] if not autotune else best_config[3],
        "block_N": best_config["block_N"] if not autotune else best_config[4],
        "block_K": best_config["block_K"] if not autotune else best_config[5],
        "num_stages": best_config["num_stages"] if not autotune else best_config[-4],
        "thread_num": best_config["thread_num"] if not autotune else best_config[-3],
        "enable_rasteration": best_config["enable_rasteration"] if not autotune else best_config[-2],
        "split_k": best_config["split_k"] if not autotune else best_config[-1],
        "best_latency": best_latency,
        "TFLOPS": 2 * M * N * K / best_latency / 1e9
    }

    kernel_path = os.path.join(kernel_save_dir, f"best_config_static_splitk_{M}_{N}_{K}.cu")
    with open(kernel_path, "w") as f:
        f.write(kernel.get_kernel_source())

    return df_best_config


def benchmark_tl_matmul_block_all_dynamic(  
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    autotune_configs,
    autotune=False,
    pass_configs=None,
):
    df_best_config = []
    
    def ref_program(A, B):
        import torch
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        out = torch.matmul(A.to(torch.float), B.to(torch.float))
        out = out.to(torch.__getattribute__(out_dtype))
        return out

    import time
    start_time = time.time()
    if not autotune:
        # Purely search to find the best config
        best_latency = float("inf")
        best_config = None
        ref_latency_list = []
        error_configs = []

        for autotune_config in autotune_configs:
            correctness = True
            block_M, block_N, block_K = autotune_config["block_M"], autotune_config["block_N"], autotune_config["block_K"]
            num_stages = autotune_config["num_stages"]
            thread_num = autotune_config["thread_num"]
            enable_rasteration = autotune_config["enable_rasteration"]
            split_k = autotune_config["split_k"]

            # TODO(zhengju) num_stages = 0 will cause runtime error of "ramp lanes should be <= 4"
            # So we hard-code here to set tl.dynamic_alignment to <= 4 for num_stages = 0
            if num_stages == 0:
                pass_configs["tl.dynamic_alignment"] = min(pass_configs["tl.dynamic_alignment"], 4)

            prim_func = tl_matmul_block_all_dynamic_splitk(block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype, accum_dtype, num_stages, thread_num, enable_rasteration, split_k)

            A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
            B = torch.rand(K, N, device="cuda", dtype=getattr(torch, in_dtype))
            C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

            try:
                kernel = tl.compile(prim_func, pass_configs=pass_configs)
                kernel(A, B, C)
                torch.cuda.synchronize()
                ref_c = ref_program(A, B)
                torch.cuda.synchronize()
                profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Normal)
                latency = profiler.do_bench(input_tensors=[A, B, C])
                print(f"Config: {autotune_config}")
                print(f"tilelang_latency: {latency} ms", flush=True)
                torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
                if latency < best_latency and correctness:
                    best_latency = latency
                    best_config = autotune_config
                ref_latency = profiler.do_bench(ref_program, input_tensors=[A, B])
                ref_latency_list.append(ref_latency)
            except Exception as e:
                correctness = False
                print(f"Config: {autotune_config} causes error: {e}", flush=True)
                error_configs.append(autotune_config)

        print(f"Dynamic MNK ({M}, {N}, {K}) with pass_configs {pass_configs}:")
        print(f"Best Config: {best_config}")
        print(f"tilelang_latency: {best_latency} ms")
        print(f"torch latency average: {sum(ref_latency_list) / len(ref_latency_list)} ms")
        print(f"Error configs Number: {len(error_configs)}")

    else:
        # Add supply_prog to supply dynamic input tensors for autotuner
        from tilelang.engine.param import KernelParam
        import tvm.tir as tir
        def supply_prog(params: list[KernelParam]) -> list[torch.Tensor]:
            ins = []
            for param in params:
                shape_dim = []
                for shape in param.shape:
                    if isinstance(shape, tir.Var):
                        if shape.name == "m":
                            shape_dim.append(M)
                        elif shape.name == "n":
                            shape_dim.append(N)
                        elif shape.name == "k":
                            shape_dim.append(K)
                        else:
                            raise ValueError(f"Shape {shape.name} is not supported")
                ins.append(torch.empty(shape_dim, device="cuda", dtype=getattr(torch, in_dtype)).uniform_(-1.0, 1.0))
            return ins
        
        # Use Autotuner to find the best config
        autotuner = AutoTuner.from_kernel(
            kernel=tl_matmul_block_all_dynamic_splitk, configs=autotune_configs).set_compile_args(
                out_idx=[-1],
                supply_type=tl.TensorSupplyType.Integer,
                supply_prog=supply_prog,
                ref_prog=ref_program,
                skip_check=False,
                target="auto")
        autotune_result = autotuner.run(warmup=3, rep=20)
        print(f"Best Config: {autotune_result.config}")
        kernel = autotune_result.kernel
        best_config = autotune_result.config
        best_latency = autotune_result.latency
        
        A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
        B = torch.rand(K, N, device="cuda", dtype=getattr(torch, in_dtype))
        C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

        # kernel(A, B, C)
        # ref_c = ref_program(A, B)
        # torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)

        profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Normal)
        profiler_latency = profiler.do_bench(input_tensors=[A, B])
        ref_latency = profiler.do_bench(ref_program, input_tensors=[A, B])
        print(f"Dynamic MNK ({M}, {N}, {K}) with pass_configs {pass_configs}:")
        print(f"Best Config: {best_config} tilelang_latency: {best_latency} ms")
        print(f"Best Config: {best_config} ref_latency: {ref_latency} ms")
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    df_best_config = {
            "M": M,
            "N": N,
            "K": K,
            "block_M": best_config["block_M"] if not autotune else best_config[0],
            "block_N": best_config["block_N"] if not autotune else best_config[1],
            "block_K": best_config["block_K"] if not autotune else best_config[2],
            "num_stages": best_config["num_stages"] if not autotune else best_config[-3],
            "thread_num": best_config["thread_num"] if not autotune else best_config[-2],
            "enable_rasteration": best_config["enable_rasteration"] if not autotune else best_config[-1],
            "dynamic_tail_split": pass_configs["tl.disable_dynamic_tail_split"],
            "dynamic_alignment": pass_configs["tl.dynamic_alignment"],
            "best_latency": best_latency,
            "TFLOPS": 2 * M * N * K / best_latency / 1e9
        }
    kernel_path = os.path.join(kernel_save_dir, f"best_config_dynamic_splitk_{M}_{N}_{K}.cu")
    with open(kernel_path, "w") as f:
        f.write(kernel.get_kernel_source())

    return df_best_config


def enumerate_configs_tl_matmul_block(M, N, K, with_roller=True, autotune=False, is_static=False, is_tma=False):
    autotune_configs = []

    # Get recommended configs from BitBLAS roller or exhausted search space
    if with_roller:
        arch = CUDA("cuda")
        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype="float16",
            out_dtype="float16",
            accum_dtype="float32").with_arch(arch)
        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=3)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # block_rows, block_cols represents warp partitioning
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["M"] = M
            config["N"] = N
            config["K"] = K
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
            config["thread_num"] = block_rows * block_cols * 32
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            config["split_k"] = hint.split_k_factor
            autotune_configs.append(config)
        print(f"Found {len(autotune_configs)} configs")
    else:
        # block_M = [32]
        # block_N = [32]
        block_M = [32, 64, 128, 256]
        block_N = [32, 64, 128, 256]
        block_K = [16]
        # block_K = [32, 64, 128]
        # block_K = [16, 32, 64, 128]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        enable_rasteration = [True, False]
        split_k = [1, 2, 4, 8]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration, split_k))
        autotune_configs = [
            {
                "M": M,
                "N": N,
                "K": K,
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "enable_rasteration": c[5],
                "split_k": c[6]
            } for c in _configs
        ]
        print(f"Existing {len(autotune_configs)} configs")

    # Add constant keys into autotune_configs
    for config in autotune_configs:
        config["trans_A"] = False
        config["trans_B"] = False
        config["in_dtype"] = "float16"
        config["out_dtype"] = "float16"
        config["accum_dtype"] = "float32"
    
    # Adaptive pass configs for dynamic autotune
    factor = 1
    while N % factor == 0 and K % factor == 0:
        factor *= 2
    factor = factor // 2
    factor = min(factor, 8)
    print(f"Factor: {factor}")
    pass_configs = {
        "tl.disable_dynamic_tail_split": True if factor > 1 else False,
        "tl.dynamic_alignment": factor,
        "tl.disable_tma_lower": not is_tma,
        "tl.disable_warp_specialized": not is_tma
    }

    if is_static:
        df_best_config = benchmark_tl_matmul_block_static(M, N, K, False, False, "float16", "float16", "float32", autotune_configs, autotune, pass_configs)
    else:
        df_best_config = benchmark_tl_matmul_block_all_dynamic(M, N, K, False, False, "float16", "float16", "float32", autotune_configs, autotune, pass_configs)

    return df_best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned Dynamic MatMul Benchmark")
    parser.add_argument("--f", type=str, default=None, help="File name to load the matmul shapes")
    parser.add_argument("--m", type=int, default=-1, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=-1, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=-1, help="Matrix dimension K")
    parser.add_argument("--with_roller", type=bool, default=False, help="Whether to enable BitBLAS roller for search space")
    parser.add_argument("--autotune", type=bool, default=False, help="Whether to use autotuner to find the best config")
    parser.add_argument("--static", type=bool, default=False, help="Whether to use static kernel or dynamic kernel")
    parser.add_argument("--tma", type=bool, default=False, help="Whether to use TMA or not")
    args = parser.parse_args()
    file_path = args.f
    M = args.m
    N = args.n
    K = args.k
    with_roller = args.with_roller
    autotune = args.autotune
    is_static = args.static
    is_tma = args.tma

    # Get current time for log
    current_time = time.strftime("%m%d", time.localtime(time.time()))
    kernel_save_dir = os.path.join("testing/python/dynamic/.log", f"{'static' if is_static else 'dynamic'}_splitk_{'tma' if is_tma else 'no_tma'}_{current_time}")
    if not os.path.exists(kernel_save_dir):
        os.makedirs(kernel_save_dir)

    df_best_config_list = []

    if file_path is None:
        # TODO(zhengju): Remove this line after testing
        if M < 0 or N < 0 or K < 0:
            M, N, K = 16384, 16384, 16384
            print("Meet invalid input, running all test cases")
            # M_list = [512, 1024, 2048, 4096, 8192, 16384]
            # N_list = [512, 1024, 2048, 4096, 8192, 16384]
            # K_list = [1024, 2048, 4096, 8192, 16384]
            M_list = [16384]
            N_list = [16384]
            K_list = [16384]
            MNK_configs = list(itertools.product(M_list, N_list, K_list))
            for MNK_config in MNK_configs:
                df_best_config_list.append(enumerate_configs_tl_matmul_block(MNK_config[0], MNK_config[1], MNK_config[2], with_roller, autotune, is_static, is_tma))
        else:
            print(f"Running test case with M={M}, N={N}, K={K}")
            df_best_config_list.append(enumerate_configs_tl_matmul_block(M, N, K, with_roller, autotune, is_static, is_tma))
        df_best_config = pd.DataFrame(df_best_config_list)
        df_best_config.to_csv(os.path.join(kernel_save_dir, f"best_config_{'static' if is_static else 'dynamic'}_splitk_{M}_{N}_{K}.csv"), index=False)

    else:
        df = pd.read_csv(file_path)
        output_path = os.path.join(kernel_save_dir, f"best_config_{'static' if is_static else 'dynamic'}_splitk_matmul_config_all.csv")
        output_df = pd.read_csv(output_path)
        for index, row in output_df.iterrows():
            df_best_config_list.append({key: row[key] for key in row.keys()})
        for index, row in df.iterrows():
            M, N, K = row["M"], row["N"], row["K"]
            existing_config = output_df[(output_df['M'] == M) & (output_df['N'] == N) & (output_df['K'] == K)]
            if not existing_config.empty:
                print(f"Config for M={M}, N={N}, K={K} already exists, skipping...")
                continue
            print(f"Running test case with M={M}, N={N}, K={K}")
            df_best_config_list.append(enumerate_configs_tl_matmul_block(M, N, K, with_roller, autotune, is_static, is_tma))
            df_best_config = pd.DataFrame(df_best_config_list)
            df_best_config.to_csv(output_path, index=False)
