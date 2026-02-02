import tilelang
import tilelang.testing
import tilelang.language as T


def test_issue_1729():
    @tilelang.jit
    def get_qwq():
        @T.prim_func
        def main(A: T.Tensor[(2, 2560), T.float32], B: T.Tensor[(2, 2560), T.float32], C: T.Tensor[(2,), T.float32]):
            with T.Kernel(1, threads=256):
                A_local = T.alloc_fragment((2, 2560), T.float32)
                B_local = T.alloc_fragment((2, 2560), T.float32)
                C_local = T.alloc_fragment((2,), T.float32)

                T.annotate_layout({C_local: tilelang.layout.make_fully_replicated_layout_fragment(C_local, 256)})

                T.copy(A, A_local)
                T.copy(B, B_local)
                T.copy(C, C_local)

                for i, j in T.Parallel(2, 2560):
                    if C_local[i] >= 0:
                        B_local[i, j] = A_local[i, j]
                T.copy(B_local, B)

        return main

    kernel = get_qwq()
    code = kernel.get_kernel_source()
    assert "for (int i_2 = 0; i_2 < 20; ++i_2)" in code


if __name__ == "__main__":
    tilelang.testing.main()
