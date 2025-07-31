from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.ops.gemm_universal import gemm_sparse
from aitemplate.compiler.tensor_accessor import TensorAccessor



class gemm_sparse_bias(gemm_sparse):
    """GEMM Specialization: GEMM_RCR(A, B) + Bias
    A[RowMajor], B[ColMajor], Bias[RowMajor], C[RowMajor]

    This operator is equivalent to the following pytorch code:

    .. highlight:: python
    .. code-block:: python
        A = torch.randn(M, K).cuda().half()
        B = torch.randn(N, K).cuda().half()
        Bias = torch.randn(N).cuda().half()

        y = torch.nn.functional.linear(A, B, bias=Bias)
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gemm_sparse_bias"

    @staticmethod
    def is_valid_inputs(a: Tensor, b_values: Tensor, b_meta: Tensor, bias: Tensor):
        msg = ""

        bias_shapes = bias._attrs["shape"]
        if len(bias_shapes) != 1:
            msg = f"Bias should be 1D vector! Current bias shape: {bias_shapes}"
            return False, msg

        bias_shape = bias_shapes[0]
        if not isinstance(bias_shape, IntImm):
            msg = f"Bias should be fixed 1D vector! Current bias shape: {bias_shape}"
            return False, msg

        Ashape = a._attrs["shape"]
        k = Ashape[-1]
        k2 = b_values._attrs["shape"][-1]
        k4 = b_meta._attrs["shape"][-1]
        if not isinstance(k, IntImm):
            msg =  f"A.K must be static, got {k}"
            return False, msg
        if 2 * k2 != k or 4 * k4 != k:
            msg = (f"Compressed B dims mismatch: "
                   f"A.K={k}, Bc.K={k2} (need K/2), Bm.K={k4} (need K/4)")
            return False,  msg

        outshape = gemm_sparse()._infer_shapes(a, b_values)
        if outshape[-1] != bias_shape:
            msg = f"GEMM/Bias shape doesn't match! Gemm shape: {outshape}, bias shape: {bias_shape}"
            return False, msg

        return True, msg

    def _infer_shapes(self, a: Tensor, b_values: Tensor, b_meta: Tensor, bias: Tensor):
        """Infers output shapes for gemm_rcr_bas.

        Parameters
        ----------
        a : Tensor
            Input tensor A.
        b : Tensor
            Input tensor B.
        bias : Tensor
            Input tensor bias. Must be a 1D vector.

        Returns
        -------
        List[IntVar]
            Output tensor shape.
        """
        is_valid_inputs, msg = self.is_valid_inputs(a, b_values, b_meta, bias)
        if not is_valid_inputs:
            raise RuntimeError(msg)
        return super()._infer_shapes(a, b_values)

    def __call__(self, a: Tensor, b_values: Tensor, b_meta: Tensor, bias: Tensor) -> Tensor:
        a, b_values, b_meta = self._align_ab(a, b_values, b_meta)
        self._attrs["inputs"] = [a, b_values, b_meta, bias]
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        self._sanity_check(a, b_values)
        output_shape = self._infer_shapes(a, b_values, b_meta, bias)
        self._extract_epilogue_alignment(output_shape)
        
        output = Tensor(output_shape, src_ops={self}, dtype=a.dtype())
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        return output
