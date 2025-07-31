import sys
sys.path.insert(0, "/AITemplate/python")

import torch
import numpy as np

from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function


'''
/// This is needed for the sparse tensor core kernels.  The purpose
/// is to use ldmatrix to load from shared memory to the register file.
template <typename Element, typename LayoutDest, typename LayoutSrc>
void reorder_meta(TensorRef<Element, LayoutDest> dest,
                  TensorRef<Element, LayoutSrc> src,
                  cutlass::gemm::GemmCoord problem_size) {
  for (int m = 0; m < problem_size.m(); m++) {
    for (int k = 0; k < problem_size.k(); k++) {
      // First reorder the rows.
      int group = (sizeof(Element) == 2) ? 32 : 16;
      int interweave = (sizeof(Element) == 2) ? 4 : 2;

      int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
      int dest_col = k;

      // Next swizzle the 2x2 blocks from Z to N.
      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      dest.at({dest_row, dest_col}) = src.at({m, k});
    }
  }
}
'''


def reorder_meta(src: torch.Tensor) -> torch.Tensor:
    m_dim, k_dim   = src.shape
    elem_bytes     = src.element_size()
    group          = 32 if elem_bytes == 2 else 16
    interweave     = 4  if elem_bytes == 2 else 2

    dst = torch.empty_like(src)

    for m in range(m_dim):
        for k in range(k_dim):
            dest_row =  (m // group) * group + (m % 8) * interweave + (m % group) // 8
            dest_col =  k

            if (dest_row & 1) == 0 and (dest_col & 1) == 1:
                dest_row += 1
                dest_col -= 1
            elif (dest_row & 1) == 1 and (dest_col & 1) == 0:
                dest_row -= 1
                dest_col += 1

            dst[dest_row, dest_col] = src[m, k]

    return dst


class PytorchModel(torch.nn.Module):
    def __init__(self, hidden, eps: float = 1e-5):
        super().__init__()
        self.dense1 = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, input):
        hidden_states = self.dense1(input)
        return hidden_states


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
                        continue

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
                meta = (p2 << 2) | p1
                metas.append(meta)
                comp_vals.extend([block[p1].item(), block[p2].item()])

            Wc[i].copy_(torch.tensor(comp_vals, dtype=W.dtype, device=W.device))
            Wm[i].copy_(torch.tensor(metas,     dtype=torch.int32, device=W.device))

        np_array = Wm.cpu().numpy()
        np.savetxt("meta.txt", np_array, fmt="%d")
        Wm = reorder_meta(Wm)
        np_array = Wm.cpu().numpy()
        np.savetxt("reorder_meta.txt", np_array, fmt="%d")


batch_size = 128
hidden = 128

pytorch_model = PytorchModel(hidden).cuda().half()
x = torch.randn([batch_size, hidden]).cuda().half()
apply_2_to_4_pruning(pytorch_model)
compress_2_to_4(pytorch_model)