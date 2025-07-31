from aitemplate.compiler.base import IntImm, Tensor, ExecItem
from aitemplate.compiler.ops.gemm_universal import gemm_common as common

from aitemplate.compiler.tensor_accessor import TensorAccessor

from collections import OrderedDict


class gemm_sparse(common.gemm):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gemm_sparse"

        def cal_align_ab(m, n, k):
            return common.default_align_ab(k, k, self._attrs["inputs"][0].dtype())

        self._attrs["f_ab_alignment"] = cal_align_ab

    def _infer_shapes(self, a: Tensor, b: Tensor):
        return a._attrs["shape"][:-1] + [b._attrs["shape"][0]]

    def _extract_dims(self, for_profiling=False):
        # (M, K) * (N, K) = (M, N)

        # profiling always uses 2d * 2d.
        A_len = (
            2
            if for_profiling
            else len(self._attrs["input_accessors"][0].original_shapes)
        )
        return {
            "M": [
                common.DimInfo(
                    common.Source.INPUT, tensor_idx=0, dim_idx=list(range(A_len - 1))
                ),
                common.DimInfo(
                    common.Source.OUTPUT, tensor_idx=0, dim_idx=list(range(A_len - 1))
                ),
            ],
            "N": [
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=0),
                common.DimInfo(common.Source.OUTPUT, tensor_idx=0, dim_idx=A_len - 1),
            ],
            "K": [
                common.DimInfo(common.Source.INPUT, tensor_idx=0, dim_idx=A_len - 1),
                common.DimInfo(common.Source.INPUT, tensor_idx=1, dim_idx=1),
            ],
        }

    def _invert_exec_key(self, key):
        return common.gemm_inverse_key_func(key)

    def _gen_profile_cmd(self, profiler_prefix, cfg, exec_key):
        def fbuild_cmd(exec_key):
            M, N, K = self._invert_exec_key(exec_key)
            cmd = []
            cmd.append(M)  # m
            cmd.append(N)  # n
            cmd.append(K)  # k
            return cmd

        return super()._gen_profile_cmd(profiler_prefix, cfg, exec_key, fbuild_cmd)

    def _align_ab(self, a: Tensor, b_values: Tensor, b_meta: Tensor):
        ak = a._attrs["shape"][-1]
        k2 = b_values._attrs["shape"][-1]
        k4 = b_meta._attrs["shape"][-1]
        if not isinstance(ak, IntImm):
            raise RuntimeError(f"K must be static, got {ak}")
        if 2 * k2 != ak or k4*2*16 != ak:
            raise RuntimeError(
                f"Compressed B shapes must match a. "
                f"A.k={ak}, Bv.k={k2}, Bm.k={k4}"
            )

        return a, b_values, b_meta
    
    def __call__(self, A:Tensor, Bv:Tensor, Bm:Tensor) -> Tensor:
        # 1) align/validate shapes
        A, Bv, Bm = self._align_ab(A, Bv, Bm)

        # 2) register all three inputs
        self._attrs["inputs" ] = [ A, Bv, Bm ]
        self._attrs["input_accessors"] = [
            TensorAccessor(x) for x in (A, Bv, Bm)
        ]

        # 3) rest is as before:
        self._set_depth()
        self._sanity_check(A, Bv)             # sanity only checks the A,Bv pair
        output_shape = self._infer_shapes(A, Bv)
        self._extract_epilogue_alignment(output_shape)

        Y = Tensor(output_shape, src_ops={self}, dtype=A.dtype())
        self._attrs["outputs"] = [Y]
        self._attrs["output_accessors"] = [TensorAccessor(Y)]
        return Y