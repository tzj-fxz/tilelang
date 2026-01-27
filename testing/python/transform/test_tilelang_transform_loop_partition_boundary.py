import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target


def _tilelang_transform_loop_partition_boundary():
    def before():
        @T.prim_func
        def main(
            S: T.Tensor((8), T.bfloat16),
            D: T.Tensor((4, 64), T.bfloat16),
        ):
            with T.Kernel(1, threads=128):
                S_shared = T.alloc_shared((8), T.bfloat16)
                S_fragment = T.alloc_fragment((8), T.float32)
                D_shared = T.alloc_shared((4, 64), T.bfloat16)

                T.copy(S, S_shared)
                T.copy(S_shared, S_fragment)
                for k in T.serial(64):
                    for i in T.Parallel(4):
                        D_shared[i, k] = S_fragment[i]
                T.copy(D_shared, D)

        return main

    def after():
        @T.prim_func
        def main(
            S: T.Tensor((8), T.bfloat16),
            D: T.Tensor((4, 64), T.bfloat16),
        ):
            with T.Kernel(1, threads=128):
                S_shared = T.alloc_shared((8), T.bfloat16)
                S_fragment = T.alloc_fragment((8), T.float32)
                D_shared = T.alloc_shared((4, 64), T.bfloat16)

                T.copy(S, S_shared)
                T.copy(S_shared, S_fragment)
                for k in T.serial(64):
                    for i in T.Parallel(8):
                        if i < 4:
                            D_shared[i, k] = S_fragment[i]
                T.copy(D_shared, D)

        return main

    return tvm.IRModule({"main": before()}), tvm.IRModule({"main": after()})


def boundary_check():
    before, after = _tilelang_transform_loop_partition_boundary()
    target = tvm.target.Target(determine_target("auto"))
    with target:
        with tvm.transform.PassContext():
            mod = tvm.tir.transform.BindTarget(target)(before)
            mod = tilelang.transform.LayoutInference()(mod)
            mod = tilelang.transform.LowerTileOp()(mod)
            mod = tvm.tir.transform.Simplify()(mod)
        with tvm.transform.PassContext():
            ref_mod = tvm.tir.transform.BindTarget(target)(after)
            ref_mod = tilelang.transform.LayoutInference()(ref_mod)
            ref_mod = tilelang.transform.LowerTileOp()(ref_mod)
            ref_mod = tvm.tir.transform.Simplify()(ref_mod)
    assert mod["main"].script() == ref_mod["main"].script(), "mod and ref_mod are not structural equal"


def test_tilelang_transform_loop_partition_boundary():
    boundary_check()


if __name__ == "__main__":
    tilelang.testing.main()
