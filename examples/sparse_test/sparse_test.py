import sys
sys.path.insert(0, "/AITemplate/python")

import torch
import numpy as np

from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function

LAYER1 = 1
LAYER2 = 1
LAYER3 = 1
LAYER4 = 1
LAYER5 = 1


class PytorchModel(torch.nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = torch.nn.Linear(hidden, LAYER1 * hidden, bias=False)
        #self.dense2 = torch.nn.Linear(LAYER1 * hidden, LAYER2 * hidden, bias=False)
        # self.dense3 = torch.nn.Linear(LAYER2 * hidden, LAYER3 * hidden, bias=False)
        # self.dense4 = torch.nn.Linear(LAYER3 * hidden, LAYER4 * hidden, bias=False)
        # self.dense5 = torch.nn.Linear(LAYER4 * hidden, LAYER5 * hidden, bias=False)
        # self.dense6 = torch.nn.Linear(LAYER5 * hidden, hidden, bias=False)

    def forward(self, input):
        hidden_states = self.dense1(input)
        #hidden_states = self.dense2(hidden_states)
        # hidden_states = self.dense3(hidden_states)
        # hidden_states = self.dense4(hidden_states)
        # hidden_states = self.dense5(hidden_states)
        # hidden_states = self.dense6(hidden_states)
        return hidden_states
    

class DenseGemmModel(nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = nn.Linear(hidden, LAYER1 * hidden, bias=False)
        #self.dense2 = nn.Linear(LAYER1 * hidden, LAYER2 * hidden, bias=False)
        # self.dense3 = nn.Linear(LAYER2 * hidden, LAYER3 * hidden, bias=False)
        # self.dense4 = nn.Linear(LAYER3 * hidden, LAYER4 * hidden, bias=False)
        # self.dense5 = nn.Linear(LAYER4 * hidden, LAYER5 * hidden, bias=False)
        # self.dense6 = nn.Linear(LAYER5 * hidden, hidden, bias=False)

    def forward(self, input):
        hidden_states = self.dense1(input)
        #hidden_states = self.dense2(hidden_states)
        # hidden_states = self.dense3(hidden_states)
        # hidden_states = self.dense4(hidden_states)
        # hidden_states = self.dense5(hidden_states)
        # hidden_states = self.dense6(hidden_states)
        return hidden_states
    

class SparseGemmModel(nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = nn.LinearSparse(hidden, LAYER1 * hidden, bias=False)
        #self.dense2 = nn.LinearSparse(LAYER1 * hidden, LAYER2 * hidden, bias=False)
        # self.dense3 = nn.LinearSparse(LAYER2 * hidden, LAYER3 * hidden, bias=False)
        # self.dense4 = nn.LinearSparse(LAYER3 * hidden, LAYER4 * hidden, bias=False)
        # self.dense5 = nn.LinearSparse(LAYER4 * hidden, LAYER5 * hidden, bias=False)
        # self.dense6 = nn.LinearSparse(LAYER5 * hidden, hidden, bias=False)

    def forward(self, input):
        hidden_states = self.dense1(input)
        #hidden_states = self.dense2(hidden_states)
        # hidden_states = self.dense3(hidden_states)
        # hidden_states = self.dense4(hidden_states)
        # hidden_states = self.dense5(hidden_states)
        # hidden_states = self.dense6(hidden_states)
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


def reorder_meta(src: torch.Tensor) -> torch.Tensor:
    """
    Python replica of cutlass::reorder_meta for *any* ElementE size.

    src shape :  (M , K')
    returns    :  reordered tensor with the same shape
    """
    m_dim, k_dim   = src.shape
    elem_bytes     = src.element_size()          # 4 for int32, 2 for fp16 meta, …
    group          = 32 if elem_bytes == 2 else 16
    interweave     = 4  if elem_bytes == 2 else 2

    dst = torch.empty_like(src)

    for m in range(m_dim):
        for k in range(k_dim):
            dest_row =  (m // group) * group + (m % 8) * interweave + (m % group) // 8
            dest_col =  k

            # 2×2 Z‑to‑N block swizzle
            if (dest_row & 1) == 0 and (dest_col & 1) == 1:
                dest_row += 1
                dest_col -= 1
            elif (dest_row & 1) == 1 and (dest_col & 1) == 0:
                dest_row -= 1
                dest_col += 1

            dst[dest_row, dest_col] = src[m, k]

    return dst


'''
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

        np_array = Wm.cpu().numpy()
        np.savetxt("meta.txt", np_array, fmt="%d")
        #Wm = reorder_meta(Wm, rows, G)
        Wm = reorder_meta(Wm)
        np_array = Wm.cpu().numpy()
        np.savetxt("reorder_meta.txt", np_array, fmt="%d")

        module.register_buffer("weight_comp", Wc)
        module.register_buffer("weight_meta", Wm)
'''


def compress_2_to_4(model: torch.nn.Module):
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        W = module.weight.data
        rows, cols = W.shape
        G = cols // 4

        Wc = torch.zeros(rows, G * 2, dtype=W.dtype, device=W.device)
        # Note 1: We construct an index matrix of the element type 'uint16'.
        # Each 16-bit element can contain 8 indices. So, we only need
        # (G * 2 // 8) elements in each row.
        Wm = torch.zeros(rows, G // 8, dtype=torch.uint32, device=W.device)

        for i in range(rows):
            comp_vals = []
            metas     = []
            for g in range(G):
                block = W[i, 4*g : 4*(g+1)]
                nz = (block != 0).nonzero(as_tuple=False).squeeze(1)
                if nz.numel() != 2:
                    raise RuntimeError(f"2:4 sparsity violated at row={i}, block={g}")
                p1, p2 = sorted(nz.tolist())
                metas.append(p1)
                metas.append(p2)
                comp_vals.extend([block[p1].item(), block[p2].item()])

            Wc[i].copy_(torch.tensor(comp_vals, dtype=W.dtype, device=W.device))

            # Note 2: Pack 16 indices into one int32 element.
            for j in range(0, len(metas), 16):
                meta = 0
                for k in range(16):
                    meta |= (metas[j+k] << (k*2))
                Wm[i][j // 16] = meta

        np_array = Wm.cpu().numpy()
        np.savetxt("meta.txt", np_array, fmt="%d")
        Wm = reorder_meta(Wm)
        np_array = Wm.cpu().numpy()
        np.savetxt("reorder_meta.txt", np_array, fmt="%d")

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
    # first the parameters (weight_comp, weight_meta)
    for name, param in ait_model.named_parameters():
        arr = param.tensor()._attrs["value"]
        consts[name.replace(".", "_")] = torch.from_numpy(arr)
    # now the buffers (weight_comp, weight_meta)
    for name, buf in ait_model.named_buffers():
        arr = buf.tensor()._attrs["value"]
        consts[name.replace(".", "_")] = torch.from_numpy(arr)
    return consts

    
def benchmark(batch_size=128, hidden=128):
    # create pytorch model
    pytorch_model = PytorchModel(hidden).cuda().half()

    # create pytorch input
    x = torch.randn([batch_size, hidden]).cuda().half() * 10

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
    sparse_model.name_parameter_tensor()
    assign_sparse_buffers(pytorch_model, sparse_model)

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

        # for name, arr in sparse_consts.items():
        #     sparse_module.set_constant(name, arr) 
        sparse_module.run_with_tensors(inputs, outputs, graph_mode=True)

        print(y_sparse)
        print(y_pytorch)

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

