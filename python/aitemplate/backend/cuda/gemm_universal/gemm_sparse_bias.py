import jinja2

from aitemplate.backend import registry

from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.cuda.gemm_universal import common_sparse, common_sparse_bias, gemm_sparse
from aitemplate.backend.cuda.gemm_universal.layout import RCR

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


EXTRA_CODE = jinja2.Template(
    """
using elem_input_type = {{elem_input_type}};
using elem_output_type = {{elem_output_type}};
using elem_metadata_type = {{elem_metadata_type}};
"""
)


# used for real execution
PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size

    { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },// Epilogue params
    split_k,                                                 // int batch_count

    // ---- pointers block ----
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,          // ptr_A
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,          // ptr_B
    (uint8_t*)(m_ptr),                                       // ptr_B_meta
    ({{elem_bias_type}}*)(bias_ptr),                         // ptr_bias ← new!
    ({{elem_output_type}}*)(c_ptr) + output_offset,          // ptr_C (output)

    // ---- strides block ----
    input_a_batch_stride,                                    // batch_stride_A
    input_b_batch_stride,                                    // batch_stride_B
    metadata_stride,                                         // batch_stride_meta
    /*output_batch_stride*/ M * N,                           // batch_stride_C

    input_a_stride,                                          // lda
    input_b_stride,                                          // ldb
    metadata_stride,                                         // ldm (meta)
    output_stride                                           // ldc
"""
)


# in case of TMA epilogue schedule, use the transposed problem to pass the
# column-major bias vector through the bias + elementwise epilogue (not residual)
PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    {
      static_cast<coord_t>(N),
      static_cast<coord_t>(M),
      static_cast<coord_t>(K),
      static_cast<coord_t>(1)
    },                                                           // ProblemShape

    // A ← B values
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,
    { input_b_batch_stride, cute::Int<1>{}, cute::Int<0>{} },

    // B metadata
    (uint8_t*)(metadata_ptr),
    { metadata_stride, cute::Int<1>{}, cute::Int<0>{} },

    // bias
    ({{elem_bias_type}}*)(bias_ptr),                             // ptr_bias ← new!

    // B ← A values
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,
    { input_a_batch_stride, cute::Int<1>{}, cute::Int<0>{} },

    // no C input (we’re fusing into D only), D ← C out
    {
      { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },
      nullptr,
      { cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{} },
      ({{elem_output_type}}*)(c_ptr) + output_offset,
      { output_stride, cute::Int<1>{}, cute::Int<0>{} }
    },

    // trailing batch & strides:
    M * K, N * K, N, M * N, K, K, 0, output_stride
"""
)


# for profiler, no need to include TensorAccessor
PROFILER_PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size

    { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },// Epilogue params
    split_k,                                                 // int batch_count

    // ---- pointers block ----
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,          // ptr_A
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,          // ptr_B
    (uint8_t*)(m_ptr),                                       // ptr_B_meta
    ({{elem_bias_type}}*)(bias_ptr),                         // ptr_bias ← new!
    ({{elem_output_type}}*)(c_ptr) + output_offset,          // ptr_C (output)

    // ---- strides block ----
    input_a_batch_stride,                                    // batch_stride_A
    input_b_batch_stride,                                    // batch_stride_B
    metadata_stride,                                         // batch_stride_meta
    /*output_batch_stride*/ M * N,                           // batch_stride_C

    input_a_stride,                                          // lda
    input_b_stride,                                          // ldb
    metadata_stride,                                         // ldm (meta)
    output_stride                                           // ldc
"""
)


# in case of TMA epilogue schedule, use the transposed problem to pass the
# column-major bias vector through the bias + elementwise epilogue (not residual)
PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    {
      static_cast<coord_t>(N),
      static_cast<coord_t>(M),
      static_cast<coord_t>(K),
      static_cast<coord_t>(1)
    },                                                           // ProblemShape

    // A ← B values
    ({{elem_input_type}}*)(b_ptr) + input_b_offset,
    { input_b_batch_stride, cute::Int<1>{}, cute::Int<0>{} },

    // B metadata
    (uint8_t*)(metadata_ptr),
    { metadata_stride, cute::Int<1>{}, cute::Int<0>{} },

    // bias
    ({{elem_bias_type}}*)(bias_ptr),                             // ptr_bias ← new!

    // B ← A values
    ({{elem_input_type}}*)(a_ptr) + input_a_offset,
    { input_a_batch_stride, cute::Int<1>{}, cute::Int<0>{} },

    // no C input (we’re fusing into D only), D ← C out
    {
      { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },
      nullptr,
      { cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{} },
      ({{elem_output_type}}*)(c_ptr) + output_offset,
      { output_stride, cute::Int<1>{}, cute::Int<0>{} }
    },

    // trailing batch & strides:
    M * K, N * K, N, M * N, K, K, 0, output_stride
"""
)


@registry.reg("cuda.gemm_sparse_bias.config")
def gemm_sparse_config(func_attrs, dtype="float16"):
    common_sparse.make_fproc(func_attrs, RCR, include_cutlass_3x_ops=True)
    func_attrs["metadata"] = func_attrs["input_accessors"][2]
    func_attrs["metadata_stride"] = func_attrs["inputs"][2]._attrs["shape"][-1]

    import cutlass_lib

    for op in func_attrs["op_instance"].values():
        if common_sparse.has_tma_epilogue(op):
            # disable residual to leave more SMEM for the mainloop
            op.C.element = cutlass_lib.library.DataType.void

            # swap the output layout to the transposed problem
            op.C.layout = cutlass_lib.library.LayoutType.ColumnMajor
            op.D.layout = cutlass_lib.library.LayoutType.ColumnMajor

            # switch to a TMA epilogue with bias
            op.epilogue_schedule = (
                cutlass_lib.library.EpilogueScheduleBiasElementwiseMapping[
                    op.epilogue_schedule
                ]
            )


@registry.reg("cuda.gemm_sparse_bias.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_metadata_type = "uint8_t"

    extra_code = EXTRA_CODE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )

    return gemm_sparse.common_gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_sparse_bias.SRC_TEMPLATE,
        problem_args_template=PROFILER_PROBLEM_ARGS_TEMPLATE,
        problem_args_template_cutlass_3x=PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
        extra_code=extra_code,
        problem_args_render_kwargs={
            "elem_input_type": elem_input_type,
            "elem_output_type": elem_output_type,
            "elem_metadata_type": elem_metadata_type,
            "has_tma_epilogue": any(
                common_sparse.has_tma_epilogue(func_attrs["op_instance"][exec_item.algo])
                for exec_item in func_attrs["exec_path"].values()
            ),
        },
    )


@registry.reg("cuda.gemm_sparse_bias.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    input_addr_calculator = gemm_sparse.get_input_addr_calculator(func_attrs)
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_metadata_type = "uint8_t"

    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        elem_metadata_type=elem_metadata_type
    )
    problem_args_cutlass_3x = PROBLEM_ARGS_TEMPLATE_CUTLASS_3X.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
        elem_metadata_type=elem_metadata_type,
        has_tma_epilogue=any(
            common_sparse.has_tma_epilogue(func_attrs["op_instance"][exec_item.algo])
            for exec_item in func_attrs["exec_path"].values()
        ),
    )
    extra_code = EXTRA_CODE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return common_sparse.gen_function(
        func_attrs=func_attrs,
        src_template=common_sparse_bias.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args,
        problem_args_cutlass_3x=problem_args_cutlass_3x,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        support_split_k=True,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=common_sparse.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N", output_accessor=func_attrs["output_accessors"][0]
        ),
        extra_code=extra_code,
    )


@registry.reg("cuda.gemm_sparse_bias.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    meta_ndims = len(func_attrs["input_accessors"][2].original_shapes)
    return common_sparse_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        meta_ndims=meta_ndims,
        support_split_k=True,
        has_metadata=True,
    )


@registry.reg("cuda.gemm_sparse_bias.func_call")
def gen_function_call(func_attrs, indent="  "):
    a, b_values, b_meta, bias = func_attrs["inputs"]
    return common_sparse.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        metadata_ptr_arg=b_meta._attrs["name"],
        metadata_stride_arg=str(func_attrs["metadata_stride"]),
        bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("cuda.gemm_sparse_bias.filter")
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