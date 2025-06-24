import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import common_sparse
from aitemplate.backend.cuda.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


ARGS_PARSER_TEMPLATE = jinja2.Template(
    """
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);
  int64_t split_k = std::atoi(argv[4]);

  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = N;
  int64_t b_dim1 = K / 2;
  int64_t m_dim0 = N;
  int64_t m_dim1 = K / 4;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;

  int idx_A = memory_pool->AllocateTensor(a_dim0 * a_dim1, 1);
  int idx_B = memory_pool->AllocateTensor(b_dim0 * b_dim1, 1);
  int idx_M = memory_pool->AllocateTensor(m_dim0 * m_dim1, 1);
  int idx_C = memory_pool->AllocateTensor(c_dim0 * c_dim1, 1, /*is_output=*/true);

  // Now grab the raw void*’s:
  void* ptr_A = memory_pool->RequestTensorByIdx(idx_A);
  void* ptr_B = memory_pool->RequestTensorByIdx(idx_B);
  void* ptr_M = memory_pool->RequestTensorByIdx(idx_M);
  void* ptr_C = memory_pool->RequestTensorByIdx(idx_C);
"""
)

# used for real execution
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size
    split_k,                                                 // int batch_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename EpilogueOutputOp::Params epilogue

    // ---- pointers block ----
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,          // void const * ptr_A
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,          // void const * ptr_B
    (uint8_t*)(m_ptr),                                       // ptr_B_meta
    ({{elem_output_type}}*)(c_ptr) + output_offset,          // void const * ptr_C

    // ---- strides block ----
    input_a_batch_stride,                                    // int64_t batch_stride_A
    input_b_batch_stride,                                    // int64_t batch_stride_B
    metadata_stride,
    /*output_batch_stride*/ M * N,                           // int64_t batch_stride_C

    input_a_stride,                                          // typename LayoutA::Stride::LongIndex lda
    input_b_stride,                                          // typename LayoutB::Stride::LongIndex ldb
    metadata_stride,                                           // metadata stride
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldc
"""
)


PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
 """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
{% if has_tma_epilogue %}
    {
        static_cast<coord_t>(N),
        static_cast<coord_t>(M),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape

    // B values become A
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,             // ElementA const* ptr_A
    {input_b_batch_stride, cute::Int<1>{}, cute::Int<0>{}},     // StrideA dA

    // B metadata
    (uint8_t*)(metadata_ptr),                                    // ElementA_meta const* ptr
    {metadata_stride, cute::Int<1>{}, cute::Int<0>{}},           // StrideA_meta dA

    // A become B
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,             // ElementB const* ptr_B
    {input_a_batch_stride, cute::Int<1>{}, cute::Int<0>{}},     // StrideB dB

    // Epilogue (no bias)
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // thread params
        nullptr,                                                 // ElementC const* ptr_C
        {cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{}},        // StrideC dC (unused)
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD* ptr_D
        {cute::Int<1>{}, output_stride, cute::Int<0>{}},         // StrideD dD
    },
{% else %}
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape

    // A
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,             // ElementA const* ptr_A
    {input_a_batch_stride, cute::Int<1>{}, cute::Int<0>{}},     // StrideA dA

    // B values
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,             // ElementB const* ptr_B
    {input_b_batch_stride, cute::Int<1>{}, cute::Int<0>{}},     // StrideB dB

    // B metadata
    (uint8_t*)(metadata_ptr),                                   // ElementB_meta const* ptr
    {metadata_stride, cute::Int<1>{}, cute::Int<0>{}},          // StrideB_meta dB

    // Epilogue (no bias)
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // thread params
        nullptr,                                                 // ElementC const* ptr_C
        {cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{}},        // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}}          // StrideD dD
    },
{% endif %}

    M * K, N * K, N, M * N, K, K, 0, output_stride               // batch & strides
"""
)


# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size
    split_k,                                                 // int batch_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*)(a_ptr),                           // void const * ptr_A
    ({{elem_input_type}}*)(b_ptr),                           // void const * ptr_B
    ({{elem_output_type}}*)(c_ptr),                          // void const * ptr_C
    ({{elem_output_type}}*)(c_ptr) + output_offset,          // void * ptr_D
    M * K,                                                   // int64_t batch_stride_A
    N * K,                                                   // int64_t batch_stride_B
    M * N,                                                   // int64_t batch_stride_C
    M * N,                                                   // int64_t batch_stride_D
    K,                                                       // typename LayoutA::Stride::LongIndex lda
    K,                                                       // typename LayoutB::Stride::LongIndex ldb
    N,                                                       // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldd
"""
)


PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    ({{elem_input_type}}*)(a_ptr),                               // ElementA const* ptr_A
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideA dA
    ({{elem_input_type}}*)(b_ptr),                               // ElementB const* ptr_B
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideB dB
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename ThreadEpilogueOp::Params thread
        nullptr,                                                 // ElementC const* ptr_C
        {N, cute::Int<1>{}, cute::Int<0>{}},                     // StrideC dC
        ({{elem_output_type}}*)(c_ptr) + output_offset,          // ElementD const* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideD dD
    },                                                           // EpilogueArguments epilogue
"""
)


@registry.reg("cuda.gemm_sparse.config")
def gemm_sparse_config(func_attrs, dtype="float16"):
    common_sparse.make_fproc(func_attrs, RCR, include_cutlass_3x_ops=True)
    func_attrs["metadata"] = func_attrs["input_accessors"][2]
    func_attrs["metadata_stride"] = func_attrs["inputs"][2]._attrs["shape"][-1]

    import cutlass_lib

    for op in func_attrs["op_instance"].values():
        if op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x:
            # disable residual to leave more SMEM for the mainloop
            op.C.element = cutlass_lib.library.DataType.void


def common_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args_template,
    problem_args_template_cutlass_3x=None,
    metadata_ptr_arg=None,
    bias_ptr_arg=None,
    extra_code="",
):
    output_addr_calculator = common_sparse.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
        stride_dim="*b_dim0"
    )
    return common_sparse.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=src_template,
        problem_args_template=problem_args_template,
        problem_args_template_cutlass_3x=problem_args_template_cutlass_3x,
        args_parser_template=ARGS_PARSER_TEMPLATE,
        support_split_k=True,
        output_addr_calculator=output_addr_calculator,
        metadata_ptr_arg=metadata_ptr_arg,
        bias_ptr_arg=bias_ptr_arg,
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_sparse.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    return common_gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_sparse.SRC_TEMPLATE,
        problem_args_template=PROFILER_PROBLEM_ARGS_TEMPLATE,
        problem_args_template_cutlass_3x=PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X,
    )


def get_input_addr_calculator(func_attrs):
    input_a_batch_stride_dim = "M * K"
    input_a_stride_k_dim = "K"
    input_a_offset = 0
    input_b_batch_stride_dim = "N * K"
    input_b_stride_k_dim = "K"
    input_b_offset = 0

    if "input_accessors" in func_attrs:
        input_a_accessor = func_attrs["input_accessors"][0]
        input_b_accessor = func_attrs["input_accessors"][1]
        if input_a_accessor.is_from_strided_tensor:
            input_a_offset = input_a_accessor.offset
            shapes = input_a_accessor.original_shapes
            input_a_stride_k_dim = input_a_accessor.stride(len(shapes) - 2)

        if input_b_accessor.is_from_strided_tensor:
            input_b_offset = input_b_accessor.offset
            shapes = input_b_accessor.original_shapes
            input_b_stride_k_dim = input_b_accessor.stride(len(shapes) - 2)

    input_addr_calculator = common_sparse.INPUT_ADDR_CALCULATOR.render(
        input_a_batch_stride_dim=input_a_batch_stride_dim,
        input_a_stride_dim=input_a_stride_k_dim,
        input_a_offset_val=input_a_offset,
        input_b_batch_stride_dim=input_b_batch_stride_dim,
        input_b_stride_dim=input_b_stride_k_dim,
        input_b_offset_val=input_b_offset,
    )
    return input_addr_calculator


@registry.reg("cuda.gemm_sparse.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    input_addr_calculator = get_input_addr_calculator(func_attrs)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    meta_ndims = len(func_attrs["input_accessors"][2].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    problem_args_cutlass_3x = PROBLEM_ARGS_TEMPLATE_CUTLASS_3X.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return common_sparse.gen_function(
        func_attrs=func_attrs,
        src_template=common_sparse.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args,
        problem_args_cutlass_3x=problem_args_cutlass_3x,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        meta_ndims=meta_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        support_split_k=True,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common_sparse.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N",
            output_accessor=func_attrs["output_accessors"][0],
        ),
    )


@registry.reg("cuda.gemm_sparse.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    meta_ndims = len(func_attrs["input_accessors"][2].original_shapes)
    return common_sparse.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        meta_ndims=meta_ndims,
        support_split_k=True,
        has_metadata=True,
    )


@registry.reg("cuda.gemm_sparse.func_call")
def gen_function_call(func_attrs, indent="  "):
    return common_sparse.gen_function_call(
        func_attrs,
        indent=indent,
        metadata_ptr_arg = func_attrs["inputs"][2]._attrs["name"],
        metadata_stride_arg = str(func_attrs["metadata_stride"]),
    )


@registry.reg("cuda.gemm_sparse.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common_sparse.function_filter(cfg, func_attrs, ab_alignment)
