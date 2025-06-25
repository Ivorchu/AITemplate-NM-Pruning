import sys
sys.path.insert(0, "/AITemplate/python")

import torch
import numpy as np

from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function


class PytorchModel(torch.nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = torch.nn.Linear(hidden, 4 * hidden, bias=False)
        self.dense2 = torch.nn.Linear(4 * hidden, 16 * hidden, bias=False)
        self.dense3 = torch.nn.Linear(16 * hidden, 64 * hidden, bias=False)
        self.dense4 = torch.nn.Linear(64 * hidden, 16 * hidden, bias=False)
        self.dense5 = torch.nn.Linear(16 * hidden, 4 * hidden, bias=False)
        self.dense6 = torch.nn.Linear(4 * hidden, hidden, bias=False)

    def forward(self, input):
        hidden_states = self.dense1(input)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dense3(hidden_states)
        hidden_states = self.dense4(hidden_states)
        hidden_states = self.dense5(hidden_states)
        hidden_states = self.dense6(hidden_states)
        return hidden_states
    

class DenseGemmModel(nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = nn.Linear(hidden, 4 * hidden, bias=False)
        self.dense2 = nn.Linear(4 * hidden, 16 * hidden, bias=False)
        self.dense3 = nn.Linear(16 * hidden, 64 * hidden, bias=False)
        self.dense4 = nn.Linear(64 * hidden, 16 * hidden, bias=False)
        self.dense5 = nn.Linear(16 * hidden, 4 * hidden, bias=False)
        self.dense6 = nn.Linear(4 * hidden, hidden, bias=False)

    def forward(self, input):
        hidden_states = self.dense1(input)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dense3(hidden_states)
        hidden_states = self.dense4(hidden_states)
        hidden_states = self.dense5(hidden_states)
        hidden_states = self.dense6(hidden_states)
        return hidden_states
    

class SparseGemmModel(nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = nn.LinearSparse(hidden, 4 * hidden, bias=False)
        self.dense2 = nn.LinearSparse(4 * hidden, 16 * hidden, bias=False)
        self.dense3 = nn.LinearSparse(16 * hidden, 64 * hidden, bias=False)
        self.dense4 = nn.LinearSparse(64 * hidden, 16 * hidden, bias=False)
        self.dense5 = nn.LinearSparse(16 * hidden, 4 * hidden, bias=False)
        self.dense6 = nn.LinearSparse(4 * hidden, hidden, bias=False)

    def forward(self, input):
        hidden_states = self.dense1(input)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dense3(hidden_states)
        hidden_states = self.dense4(hidden_states)
        hidden_states = self.dense5(hidden_states)
        hidden_states = self.dense6(hidden_states)
        return hidden_states
    
    
def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = {}
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        assert name in pt_params
        mapped_pt_params[ait_name] = pt_params[name]
    return mapped_pt_params


def apply_2_to_4_pruning(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            rows, cols = weight.shape

            for row in range(rows):
                for col_start in range(0, cols, 4):
                    col_end = min(col_start + 4, cols)
                    block = weight[row, col_start:col_end]

                    if block.numel() < 4:
                        continue  # skip incomplete blocks

                    # Get the 2 indices with largest magnitude
                    _, topk_indices = torch.topk(block.abs(), k=2, largest=True)
                    mask = torch.zeros_like(block)
                    mask[topk_indices] = 1.0

                    weight[row, col_start:col_end] *= mask

            print(f"Applied magnitude-based 2:4 pruning to {name}")


def compress_2_to_4(model: torch.nn.Module):
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        W = module.weight.data
        rows, cols = W.shape
        G = cols // 4

        # allocate
        Wc = torch.zeros(rows, G * 2, dtype=W.dtype, device=W.device)
        Wm = torch.zeros(rows,   G, dtype=torch.int32, device=W.device)

        for i in range(rows):
            comp_vals = []
            metas     = []
            for g in range(G):
                block = W[i, 4*g : 4*(g+1)]
                nz = (block != 0).nonzero(as_tuple=False).squeeze(1)
                if nz.numel() != 2:
                    raise RuntimeError(f"2:4 sparsity violated at row={i}, block={g}")
                p1, p2 = sorted(nz.tolist())
                # pack into nibble
                meta = (p2 << 2) | p1
                metas.append(meta)
                # pull the two values in order
                comp_vals.extend([block[p1].item(), block[p2].item()])

            Wc[i].copy_(torch.tensor(comp_vals, dtype=W.dtype, device=W.device))
            Wm[i].copy_(torch.tensor(metas,     dtype=torch.int32, device=W.device))

        module.register_buffer("weight_comp", Wc)
        module.register_buffer("weight_meta", Wm)


def assign_sparse_buffers(pt_model, ait_model):
    for (_, pt_mod), (_, ai_mod) in zip(
        pt_model.named_modules(), ait_model.named_modules()
    ):
        if isinstance(pt_mod, torch.nn.Linear) and isinstance(ai_mod, nn.LinearSparse):
            # pull out NumPy arrays
            Wc = pt_mod.weight_comp.detach().cpu().numpy()
            Wm = pt_mod.weight_meta.detach().cpu().numpy()
            # shove them into the AIT template Tensors
            ai_mod.weight_comp.tensor()._attrs["value"] = Wc   # Wc is float16
            ai_mod.weight_meta.tensor()._attrs["value"] = Wm   # now int32
            if ai_mod.use_bias:
                b = pt_mod.bias.detach().cpu().numpy().astype(np.float16)
                ai_mod.bias.tensor()._attrs["value"] = b


def map_all_constants(ait_model):
    consts = {}
    # first the parameters (weight, bias)
    for name, param in ait_model.named_parameters():
        arr = param.tensor()._attrs["value"]
        consts[name.replace(".", "_")] = arr
    # now the buffers (weight_comp, weight_meta)
    for name, buf in ait_model.named_buffers():
        arr = buf.tensor()._attrs["value"]
        consts[name.replace(".", "_")] = arr
    return consts

    
def benchmark(batch_size=1024, hidden=16):
    # create pytorch model
    pytorch_model = PytorchModel(hidden).cuda().half()

    # create pytorch input
    x = torch.randn([batch_size, hidden]).cuda().half()

    # run pt model
    pytorch_model.eval()
    apply_2_to_4_pruning(pytorch_model)
    compress_2_to_4(pytorch_model)
    y_pytorch = pytorch_model(x)

    count = 1000
    pytorch_time = benchmark_torch_function(count, pytorch_model.forward, x)
    
    # create dense model
    dense_model = DenseGemmModel(hidden)

    # create sparse model
    sparse_model = SparseGemmModel(hidden)
    assign_sparse_buffers(pytorch_model, sparse_model)

    X_dense = Tensor(
        shape = [batch_size, hidden],
        name = "X_dense",
        dtype = "float16",
        is_input = True,
    )
    Y_dense = dense_model(X_dense)
    Y_dense._attrs["is_output"] = True
    Y_dense._attrs["name"] = "Y_dense"

    X_sparse = Tensor(
        shape = [batch_size, hidden],
        name = "X_sparse",
        dtype = "float16",
        is_input = True,
    )
    Y_sparse = sparse_model(X_sparse)
    Y_sparse._attrs["is_output"] = True
    Y_sparse._attrs["name"] = "Y_sparse"

    target = detect_target()
    dense_consts = map_pt_params(dense_model, pytorch_model)
    with compile_model(
        Y_dense, target, "./tmp", "dense_model", constants=dense_consts
    ) as dense_module:
        y_dense = torch.empty([batch_size, hidden]).cuda().half()

        dense_inputs = {"X_dense": x}
        dense_outputs = {"Y_dense": y_dense}

        dense_module.run_with_tensors(dense_inputs, dense_outputs, graph_mode=True)

        if torch.allclose(y_dense, y_pytorch, atol=1e-2, rtol=1e-2):
            print("Dense model outputs were correct.")
        else:
            print("Dense model outputs were incorrect.")
        
        dense_time, _, _ = dense_module.benchmark_with_tensors(
            dense_inputs, dense_outputs, graph_mode=True, count=count
        )

    sparse_consts = map_all_constants(sparse_model)
    with compile_model(
        Y_sparse, target, "./tmp", "sparse_model", constants=sparse_consts
    ) as sparse_module:
        y_sparse = torch.empty([batch_size, hidden]).cuda().half()

        inputs = {"X_sparse": x}
        outputs = {"Y_sparse": y_sparse}

        sparse_module.run_with_tensors(inputs, outputs, graph_mode=True)

        if torch.allclose(y_sparse, y_pytorch, atol=1e-2, rtol=1e-2):
            print("Sparse model outputs were correct.")
        else:
            print("Sparse model outputs were incorrect.")

        sparse_time, _, _ = sparse_module.benchmark_with_tensors(
            inputs, outputs, graph_mode=True, count=count
        )
    
    print(f"PyTorch eager time: {pytorch_time} ms/iter")
    print(f"Dense model time: {dense_time} ms/iter")
    print(f"Sparse model time: {sparse_time} ms/iter")
        

benchmark()

