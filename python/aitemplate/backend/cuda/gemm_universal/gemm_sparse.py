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
    // 1) problem size
    cutlass::gemm::GemmCoord{ M, N, K },

    // 2) ref_B  
    { ({{elem_input_type}} const*)(b_ptr) + input_b_offset,
    input_b_batch_stride, input_b_stride },

    // 3) ref_A
    { ({{elem_input_type}} const*)(a_ptr) + input_a_offset,
    input_a_batch_stride, input_a_stride },

    // 4) ref_C  
    { ({{elem_output_type}}*)(c_ptr) + output_offset,
    M * N, output_stride },

    // 5) ref_D  (same as C for inplace)  
    { ({{elem_output_type}}*)(c_ptr) + output_offset,
    M * N, output_stride },

    // 6) ref_E  (the 2:4 metadata)  
    { ({{elem_meta_type}}*)(m_ptr) + input_m_offset,
    input_m_batch_stride, input_m_stride },

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
    cutlass::gemm::GemmCoord{ M, N, K * 2},
    
    // 3) ref_B  
    { ({{elem_input_type}} const*)(b_ptr) + input_b_offset,
    input_b_batch_stride, input_b_stride },
    
    // 2) ref_A
    { ({{elem_input_type}} const*)(a_ptr) + input_a_offset,
    input_a_batch_stride, input_a_stride },

    // 4) ref_C  
    { ({{elem_output_type}}*)(c_ptr) + output_offset,
    M * N, output_stride },

    // 5) ref_D  (same as C for inplace)  
    { ({{elem_output_type}}*)(c_ptr) + output_offset,
    M * N, output_stride },

    // 6) ref_E  (the 2:4 metadata)  
    { ({{elem_meta_type}}*)(m_ptr) + input_m_offset,
    input_m_batch_stride, input_m_stride },

    // 7) epilogue  
    { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },

    // 8) split_k  
    split_k
"""
)


PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    {
      // ProblemShape: note N↔M swap
      static_cast<coord_t>(N),
      static_cast<coord_t>(M),
      static_cast<coord_t>(K),
      static_cast<coord_t>(1)
    },

    // B values → A
    ({{elem_input_type}}*)(b_ptr),       
    { K, cute::Int<1>{}, cute::Int<0>{} },

    // B metadata
    (uint8_t*)(m_ptr),
    { metadata_stride, cute::Int<1>{}, cute::Int<0>{} },

    // A values → B
    ({{elem_input_type}}*)(a_ptr),
    { K, cute::Int<1>{}, cute::Int<0>{} },

    // Epilogue: C in unused, D out
    {
      { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },
      nullptr,
      { cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{} },
      ({{elem_output_type}}*)(c_ptr) + output_offset,
      { output_stride, cute::Int<1>{}, cute::Int<0>{} }
    },

    // trailing batch & strides
    M * K, N * K, N, M * N, K, K, 0, output_stride                                                  // EpilogueArguments epilogue
"""
)


@registry.reg("cuda.gemm_sparse.config")
def gemm_sparse_config(func_attrs, dtype="float16"):
    # 1) build the full op_instance list
    common_sparse.make_fproc(func_attrs, RCR, include_cutlass_3x_ops=True)

    # 2) your metadata plumbing
    func_attrs["metadata"] = func_attrs["input_accessors"][2]
    func_attrs["metadata_stride"] = func_attrs["inputs"][2]._attrs["shape"][-1]

    # 3) disable residual for any 3.x kernels (your existing loop)
    import cutlass_lib
    for op in func_attrs["op_instance"].values():
        if op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x:
            op.C.element = cutlass_lib.library.DataType.void

    # 4) pick exactly one algorithm from op_instance
    algo_name, algo_inst = next(iter(func_attrs["op_instance"].items()))
    func_attrs["op_instance"] = OrderedDict([(algo_name, algo_inst)])

    # 5) register an exec_path that always fires
    func_attrs["exec_path"] = OrderedDict([
      (
        "true",
        ExecItem(
          profiling_key="true",
          exec_cond="true",      # this branch will always be taken
          algo=algo_name,
        ),
      ),
    ])
    func_attrs["has_profiler"] = False


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

    import cutlass_lib

    # 1) build your cutlass‐type tuple from AIT dtype
    spec = CUDASpec()
    A_type = spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])
    B_type = spec.dtype_to_lib_type(func_attrs["inputs"][1]._attrs["dtype"])
    C_type = spec.dtype_to_lib_type(func_attrs["outputs"][0]._attrs["dtype"])
    accum = spec.dtype_to_lib_type(cutlass_lib.library.DataType.f16)
    data_type = (A_type, B_type, C_type, accum)

    # 2) reuse the *dense* GEMM tile descriptions & alignments
    td = Target.current()._kwargs["tile_descriptions"]
    ac = Target.current()._kwargs["alignment_constraints"]

    # 3) generate only Sparse‐Gemm ops via the CUTLASS Python generator
    manifest = []
    sparse_ops = cutlass_lib.generator.CreateSparseGemmOperator(
        manifest,
        layouts=[ func_attrs["layout"].cutlass_lib_layouts() ],  # RCR, RCC, etc.
        tile_descriptions=td,
        data_type=data_type,
        alignment_constraints=ac,
    )

    # 4) key them by the canonical CUTLASS name
    op_dict = OrderedDict((kernel_name(op), op) for op in sparse_ops)

    # 5) drop any 3.x kernels that don’t fit your layout/align
    op_dict, _ = filter_cutlass_3x_ops(op_dict, func_attrs)

    func_attrs["op_instance"] = op_dict

    # 6) pick exactly one algo and bake in a always‐true exec_path
    algo_name = next(iter(op_dict))
    func_attrs["exec_path"] = OrderedDict([
        ("true", ExecItem("true", "true", algo_name))
    ])

    # 7) now hand off to AIT’s built‐in sparse‐GEMM profiler generator
    return common_sparse.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        src_template=common_sparse.SRC_TEMPLATE,
        problem_args_template=common_sparse.PROFILER_PROBLEM_ARGS_TEMPLATE,
        problem_args_template_cutlass_3x=common_sparse.PROFILER_PROBLEM_ARGS_TEMPLATE_CUTLASS_3X,
        args_parser_template=common_sparse.ARGS_PARSER_TEMPLATE,
        support_split_k=True,
        output_addr_calculator=common_sparse.DEFAULT_OUTPUT_ADDR_CALCULATOR.render(
            output_batch_stride_dim="*b_dim0 * N",
            output_stride_dim="*b_dim0"
        ),
        metadata_ptr_arg="ptr_M",
        bias_ptr_arg=None,
    )

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
    input_m_batch_stride_dim = "metadata_stride"
    input_m_stride_k_dim = "metadata_stride"
    input_m_offset = 0
    input_b_batch_stride_dim = "N * K / 2"
    input_b_stride_k_dim = "K / 2"
    input_b_offset = 0

    if "input_accessors" in func_attrs:
        input_a_accessor = func_attrs["input_accessors"][0]
        input_b_accessor = func_attrs["input_accessors"][1]
        input_m_accessor = func_attrs["input_accessors"][2]
        if input_a_accessor.is_from_strided_tensor:
            input_a_offset = input_a_accessor.offset
            shapes = input_a_accessor.original_shapes
            input_a_stride_k_dim = input_a_accessor.stride(len(shapes) - 2)

        if input_b_accessor.is_from_strided_tensor:
            input_b_offset = input_b_accessor.offset
            shapes = input_b_accessor.original_shapes
            input_b_stride_k_dim = input_b_accessor.stride(len(shapes) - 2)

        if input_m_accessor.is_from_strided_tensor:
            input_m_offset = input_m_accessor.offset
            shapes = input_m_accessor.original_shapes
            input_m_stride_k_dim = input_m_accessor.stride(len(shapes) - 2)

    input_addr_calculator = common_sparse.INPUT_ADDR_CALCULATOR.render(
        input_a_batch_stride_dim=input_a_batch_stride_dim,
        input_a_stride_dim=input_a_stride_k_dim,
        input_a_offset_val=input_a_offset,
        input_m_batch_stride_dim=input_m_batch_stride_dim,
        input_m_stride_dim=input_m_stride_k_dim,
        input_m_offset_val=input_m_offset,
        input_b_batch_stride_dim=input_b_batch_stride_dim,
        input_b_stride_dim=input_b_stride_k_dim,
        input_b_offset_val=input_b_offset,
    )
    return input_addr_calculator


def gen_exec_op(func_attrs):
    import cutlass_lib

    '''
    # 1) grab our current arch & cuda version
    tgt = Target.current()
    arch = tgt._arch                                # e.g. "80"
    cuda_version = tgt._cuda_version or "11.4.2"

    # 2) build CUTLASS manifest via AIT’s gen_cutlass_ops helper
    #    this will give us manifest.operations
    from aitemplate.backend.cuda.utils import mk_cutlass_lib
    registry.get("cuda.make_cutlass_lib")(tgt.template_path())
    gen_ops = registry.get("cuda.gen_cutlass_ops")
    all_ops = gen_ops(arch, cuda_version,
                      allow_cutlass_sm90=tgt._kwargs.get("allow_cutlass_sm90", False),
                      force_cutlass_sm90=tgt._kwargs.get("force_cutlass_sm90", False))

    # 3) extract only the GemmKind.Sparse ops
    gemm_by_cfg = all_ops[cutlass_lib.library.OperationKind.Gemm]
    sparse_ops = []
    for cfg_name, ops in gemm_by_cfg.items():
        for op in ops:
            if op.gemm_kind == cutlass_lib.library.GemmKind.Sparse:
                sparse_ops.append(op)

    # 4) key by canonical name
    op_dict = OrderedDict((kernel_name(op), op) for op in sparse_ops)

    # 5) drop any 3.x kernels that don’t match our tensor-accessor alignments
    op_dict, _ = filter_cutlass_3x_ops(op_dict, func_attrs)
    # but ignore the 3.x entries entirely:
    op_dict = {k:v for k,v in op_dict.items() if v.gemm_kind != cutlass_lib.library.GemmKind.Universal3x}

    # 6) install into func_attrs
    func_attrs["op_instance"] = op_dict

    # pick exactly one algorithm and bake in an always-true guard
    algo = next(iter(op_dict.keys()))
    func_attrs["exec_path"] = OrderedDict([
        ("true", ExecItem("true", "true", algo))
    ])

    return func_attrs["exec_path"]
    '''
    '''
     # 1) Build the CUTLASS data‐type tuple
    spec = CUDASpec()
    A_type = spec.dtype_to_lib_type(func_attrs["inputs"][0]._attrs["dtype"])
    B_type = spec.dtype_to_lib_type(func_attrs["inputs"][1]._attrs["dtype"])
    C_type = spec.dtype_to_lib_type(func_attrs["outputs"][0]._attrs["dtype"])
    # accumulate in the same type as A/B
    accum = A_type
    data_type = (A_type, B_type, C_type, accum)

    # 2) Grab your target’s tile descriptions & alignments
    target = Target.current()
    td = target._kwargs["tile_descriptions"]
    ac = target._kwargs["alignment_constraints"]

    # 3) Invoke CUTLASS’s Python generator for sparse‐Gemm
    manifest_list = []
    layout = func_attrs["layout"].cutlass_lib_layouts()
    sparse_ops = cutlass_lib.generator.CreateSparseGemmOperator(
        manifest_list,
        layouts=[layout],
        tile_descriptions=td,
        data_type=data_type,
        alignment_constraints=ac,
    )

    # 4) Key them by name and drop any 3.x kernels
    op_dict = OrderedDict((kernel_name(op), op) for op in sparse_ops)
    op_dict, _ = filter_cutlass_3x_ops(op_dict, func_attrs)

    func_attrs["op_instance"] = op_dict

    # 5) Pick one of the algos as your fallback exec_path
    first_algo = next(iter(op_dict.keys()))
    func_attrs["exec_path"] = OrderedDict([
        (
            # you said you only care about M=1024,N=64,K=16
            "M == 1024 && N == 64 && K == 16",
            ExecItem(
                profiling_key="M == 1024 && N == 64 && K == 16",
                exec_cond="M == 1024 && N == 64 && K == 16",
                algo=first_algo,
            ),
        )
    ])

    # 6) Finally, stash away the metadata‐stride so your func_call template can see it
    meta_accessor = func_attrs["input_accessors"][2]
    func_attrs["metadata_stride"] = meta_accessor.stride(
        len(meta_accessor.original_shapes) - 2
    )
    '''
     # grab the CUTLASS‐generated ops from the current CUDA target
    target = Target.current()
    ops_by_kind = target._operators

    # all GEMM ops (dense + sparse + 3x) live under OperationKind.Gemm
    gemm_cfgs = ops_by_kind.get(cutlass_lib.library.OperationKind.Gemm, {})

    # flatten and pick only the sparse GEMMs
    sparse_ops = [
        op
        for cfg_ops in gemm_cfgs.values()
        for op in cfg_ops
        if op.gemm_kind == cutlass_lib.library.GemmKind.Sparse
    ]

    if not sparse_ops:
        raise RuntimeError("No sparse‐GEMM kernels found in target._operators")

    # key them by their canonical CUTLASS name
    func_attrs["op_instance"] = OrderedDict(
        (kernel_name(op), op) for op in sparse_ops
    )

    # pick the first algo for a single‐path exec
    first_algo = next(iter(func_attrs["op_instance"]))
    func_attrs["exec_path"] = OrderedDict([
        ("true", ExecItem("true", "true", first_algo))
    ])

    # compute metadata_stride from the 3rd input accessor’s last dim
    meta_acc = func_attrs["input_accessors"][2]
    last_dim = len(meta_acc.original_shapes) - 1
    func_attrs["metadata_stride"] = meta_acc.stride(last_dim)



@registry.reg("cuda.gemm_sparse.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    gen_exec_op(func_attrs)

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
