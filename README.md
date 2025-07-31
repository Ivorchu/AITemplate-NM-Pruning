# Sparse GEMM Profiler with CUTLASS and AITemplate

This project implements a custom 2:4 structured sparse GEMM (General Matrix Multiplication) profiler using NVIDIA's [CUTLASS](https://github.com/NVIDIA/cutlass) library and Meta's [AITemplate](https://github.com/facebookincubator/AITemplate) compiler framework. The profiler builds and benchmarks sparse tensor core GEMM kernels on Ampere GPUs, integrating metadata preprocessing and kernel configuration into AITemplate’s workflow.

## 📌 Key Features

- Structured 2:4 sparsity with Tensor Core acceleration
- Uses `cutlass::gemm::device::SparseGemm` for sparse kernel execution
- Metadata reordering using `cutlass::reorder_meta`
- Automatic profiler generation and kernel selection with AITemplate
- Split-K support for parallelism along the reduction (K) dimension
- Tensor inspection/debug tools via CUDA host/device memory copy

## 📁 Project Structure

```
AITemplate/
├── 3rdparty/
│   └── cutlass/                      # CUTLASS 2.x source
├── examples/sparse_test/
│   ├── sparse_test.py               # Entry point for benchmarking
│   └── [generated profiler .cu files]
├── python/aitemplate/
│   └── compiler/transform/profile/  # Profile logic and hooks
```

## 🚀 Getting Started

### 1. Build Docker Image

```bash
cd docker
./build.sh cuda
```

### 2. Launch the Container

```bash
./run.sh
```

### 3. Run Sparse Profiler

```bash
cd examples/sparse_test
python sparse_test.py
```

This will generate and run sparse GEMM profiler binaries with specific shape and split-K configs.

## 🧠 Technical Details

### CUTLASS Sparse GEMM

CUTLASS provides `SparseGemm`, a class for structured sparse matrix multiplication on NVIDIA Ampere GPUs. It requires:

- Operand A: dense activation
- Operand B: 2:4 sparse weights (split into values and metadata)
- Operand E: metadata tensor in CUTLASS-native format (reordered)

### Metadata Reordering

```cpp
cutlass::TensorRef<ElementE, cutlass::layout::RowMajor> meta_src((ElementE*)m_ptr, meta_stride);
cutlass::TensorRef<ElementE, cutlass::layout::RowMajor> meta_dst((ElementE*)m_ptr, meta_stride);
cutlass::gemm::GemmCoord meta_extent(M, N, K / 2 / kElementsPerElementE);
cutlass::reorder_meta(meta_dst, meta_src, meta_extent);
```

### Kernel Arguments Setup

```cpp
Gemm::Arguments arguments{
    cutlass::gemm::GemmCoord{M, N, K},
    {a_ptr, a_stride},
    {b_ptr, b_stride},
    {c_ptr, c_stride},
    {d_ptr, d_stride},
    {meta_ptr, meta_stride},
    {alpha, beta},
    split_k_slices
};
```

### Tensor Debugging

```cpp
std::vector<cutlass::half_t> host_B(b_size);
cudaMemcpy(host_B.data(), b_ptr, b_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);
for (int i = 0; i < b_size; ++i) {
    std::cout << __half2float(static_cast<__half>(host_B[i])) << " ";
}
```

## 📈 Profiling Output

AITemplate generates profiler binaries and stores them in:

```
/tmp/aitemplate_cache/<hash>/
```

The profiler selection uses cached results unless `force_profile()` is set to `True`.

## 🛠 Common Issues

| Issue                                     | Fix                                                                 |
|------------------------------------------|----------------------------------------------------------------------|
| `cudaFuncSetAttribute` failed            | Ensure shared memory usage < 48 KB or compile with `-maxrregcount`  |
| Illegal memory access in profiling       | Check metadata pointer and meta_stride validity                     |
| `cutlass::half_t` to `__half` conversion | Use `.raw()` or explicit cast to `__half`                           |
| Invalid tensor layout                    | Make sure B is column-major if using RCR kernel                     |

## 🙋 Author

This project is developed by **Ivor**, a Computer Engineering student at Purdue University, as part of research on structured sparsity, AI acceleration, and compiler-accelerated inference.

## 📄 License

This project is for research and educational purposes. CUTLASS and AITemplate are licensed under their respective open-source licenses.