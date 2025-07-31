from aitemplate.compiler import ops
from aitemplate.frontend.nn.module import Module
from aitemplate.frontend.nn.parameter import Parameter
from aitemplate.testing import detect_target

class LinearSparse(Module):
    USE_CUDA = None

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        specialization=None,
        dtype="float16",
        **kwargs,
    ):
        super().__init__()
        if LinearSparse.USE_CUDA is None:
            LinearSparse.USE_CUDA = detect_target().name() == "cuda"
        # self.weight = Parameter(shape=[out_channels, in_channels], dtype=dtype)
        self.weight_comp = Parameter(
            shape=[out_channels, in_channels // 2],
            dtype=dtype,
        )
        self.weight_meta = Parameter(
            shape=[out_channels, in_channels // 2 // 16],
            dtype="uint32",
        )
        op_name = "gemm_sparse_bias" if bias else "gemm_sparse"
        if specialization is not None:
            op_name += "_" + specialization
        if bias:
            self.bias = Parameter(shape=[out_channels], dtype=dtype)
        
        op_func = getattr(ops, op_name)
        self._op_name = op_name
        self.op = op_func(**kwargs)
        self.use_bias = bias
        self.in_channels = in_channels

    def forward(self, *args):

        assert len(args) >= 1
        x = args[0]

        if not self.USE_CUDA and len(x._attrs["shape"]) != 2:
            x = ops.reshape()(x, [-1, self.in_channels])

        comp = self.weight_comp.tensor()
        meta = self.weight_meta.tensor()

        inputs = [x, comp, meta]

        if self.use_bias:
            inputs.append(self.bias.tensor())

        return self.op(*inputs)