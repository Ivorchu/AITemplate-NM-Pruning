import jinja2

from aitemplate.backend import registry
from collections import OrderedDict
from aitemplate.compiler.base import ExecItem

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import common_sparse
from aitemplate.backend.cuda.gemm_universal.layout import RCR

from aitemplate.backend.cuda.gemm_universal.common_sparse import kernel_name, filter_cutlass_3x_ops
from aitemplate.backend.target import Target
from aitemplate.backend.cuda.gemm_universal.common_sparse import kernel_name

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
  int64_t m_dim1 = K / 2 / 16;
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
    // 1) problem size
    cutlass::gemm::GemmCoord{ M, N, K },

    // 2) ref_B  
    { (cutlass::half_t const*)(b_ptr) + input_b_offset,
    K / 2 },

    // 3) ref_A
    { (cutlass::half_t const*)(a_ptr) + input_a_offset,
    K },

    // 4) ref_C  
    { (cutlass::half_t*)(c_ptr) + output_offset,
    N },

    // 5) ref_D  (same as C for inplace)  
    { (cutlass::half_t*)(c_ptr) + output_offset,
    N },

    // 6) ref_E  (the 2:4 metadata)  
    { (ElementE*)(m_ptr),
    K / 2 / 8 },

    // 7) epilogue  
    { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },

    // 8) split_k  
    split_k
"""
)


PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
 """
    // CUTLASS 3.x universal adapter
    cutlass::gemm::GemmUniversalMode::kGemm,
    cutlass::gemm::GemmCoord{M,N,K},
    split_k,
    { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },

    // pointers
    ({{elem_input_type}} const*) (a_ptr) + input_a_offset,
    ({{elem_input_type}} const*) (b_ptr) + input_b_offset,
    (uint8_t*)                (m_ptr),              // metadata
    ({{elem_output_type}}*)   (c_ptr) + output_offset,

    // strides
    input_a_batch_stride,
    input_b_batch_stride,
    metadata_stride,
    output_stride,

    input_a_stride,
    input_b_stride,
    metadata_stride,
    output_stride
"""
)


# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    // 1) problem size
    cutlass::gemm::GemmCoord{ M, N, K },

    // 2) ref_B  
    { (cutlass::half_t const*)(b_ptr) + input_b_offset,
    K / 2 },

    // 3) ref_A
    { (cutlass::half_t const*)(a_ptr) + input_a_offset,
    K },

    // 4) ref_C  
    { (cutlass::half_t*)(c_ptr) + output_offset,
    N },

    // 5) ref_D  (same as C for inplace)  
    { (cutlass::half_t*)(c_ptr) + output_offset,
    N },

    // 6) ref_E  (the 2:4 metadata)  
    { (ElementE*)(m_ptr),
    K / 2 / 16 },

    // 7) epilogue  
    { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },

    // 8) split_k  
    split_k
"""
)


PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    // 1) problem size
    cutlass::gemm::GemmCoord{ M, N, K },

    // 2) ref_B  
    { (cutlass::half_t const*)(b_ptr) + input_b_offset,
    K / 2 },

    // 3) ref_A
    { (cutlass::half_t const*)(a_ptr) + input_a_offset,
    K },

    // 4) ref_C  
    { (cutlass::half_t*)(c_ptr) + output_offset,
    N },

    // 5) ref_D  (same as C for inplace)  
    { (cutlass::half_t*)(c_ptr) + output_offset,
    N },

    // 6) ref_E  (the 2:4 metadata)  
    { (ElementE*)(m_ptr),
    K / 2 / 8 },

    // 7) epilogue  
    { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },

    // 8) split_k  
    split_k                                               // EpilogueArguments epilogue
"""
)


@registry.reg("cuda.gemm_sparse.config")
def gemm_sparse_config(func_attrs, dtype="float16"):
    # 1) build the full op_instance list
    common_sparse.make_fproc(func_attrs, RCR, include_cutlass_3x_ops=False)

    # 2) your metadata plumbing
    func_attrs["metadata"] = func_attrs["input_accessors"][2]
    func_attrs["metadata_stride"] = func_attrs["inputs"][2]._attrs["shape"][-1]


def common_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args_template,
    problem_args_template_cutlass_3x=None,
    bias_ptr_arg=None,
    extra_code="",
):
    input_addr_calculator = get_input_addr_calculator(func_attrs)
    output_addr_calculator = common_sparse.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
        output_batch_stride_dim="M * N",
        output_stride_dim="N",
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
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
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
    input_b_batch_stride_dim = "N * K / 2"
    input_b_stride_k_dim = "K / 2"
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
    elem_meta_type = "ElementE"
    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        elem_meta_type=elem_meta_type,
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
        f_instance_convertor=common_sparse.sparse_gemm_instance,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common_sparse.OUTPUT_ADDR_CALCULATOR.render(
            output_batch_stride_dim="M * N",
            output_stride_dim="N",
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
    meta_ptr = func_attrs["inputs"][2]._attrs["name"]
    meta_stride = func_attrs["metadata_stride"]
    return common_sparse.gen_function_call(
        func_attrs,
        indent=indent,
        # NO trailing commas here:
        metadata_ptr_arg    = meta_ptr,
        metadata_stride_arg = str(meta_stride),
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
