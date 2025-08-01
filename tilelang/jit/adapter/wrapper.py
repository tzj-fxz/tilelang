from abc import ABC, abstractmethod
from tilelang import tvm as tvm
from typing import Optional, List, Dict, Union, Any
from tvm import IRModule
from tvm.target import Target
from .utils import (match_declare_kernel, match_declare_kernel_cpu, is_cuda_target, is_hip_target,
                    is_cpu_target, get_annotated_mod, pythonic_expr)
import re
import logging
import textwrap

PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY = """
    cudaError_t result_{0} = cudaFuncSetAttribute({0}, cudaFuncAttributeMaxDynamicSharedMemorySize, {1});
    if (result_{0} != CUDA_SUCCESS) {{
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", {1}, cudaGetErrorString(result_{0}));
        return -1;
    }}
"""

PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY_HIP = """
    if ({1} > 65536) {{
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size for {0} to %d", {1});
        return -1;
    }}
    return 0;
"""

PREDEF_INIT_FUNC = """
#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {{
    return error_buf;
}}

extern "C" int init() {{
    error_buf[0] = '\\0';
    {0}
    return 0;
}}
"""

PREDEF_HOST_FUNC = """
extern "C" int call({}) {{
{}
\treturn 0;
}}
"""

PREDEF_HOST_FUNC_PY = """
import cuda.bindings.driver
import ctypes

_function_names = {}

def call({}):
    {}
"""

L2_PERSISTENT_MAP_CREATE_HANDLE = """
\tcudaStreamAttrValue stream_attribute;
\tsize_t init_persisting_l2_cache_size;
\tcudaDeviceGetLimit(&init_persisting_l2_cache_size, cudaLimitPersistingL2CacheSize);
"""

L2_PERSISTENT_MAP_INIT_FUNC = """
\tstream_attribute.accessPolicyWindow.hitRatio = {1};
\tstream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
\tstream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
\tcudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, {2});
\tstream_attribute.accessPolicyWindow.base_ptr = (void*)({0});
\tstream_attribute.accessPolicyWindow.num_bytes = {2};
\tcudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
"""

L2_PERSISTENT_MAP_RESET_HANDLE = """
\tstream_attribute.accessPolicyWindow.num_bytes = 0;
\tcudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
\tcudaCtxResetPersistingL2Cache();
\tcudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, init_persisting_l2_cache_size);
"""

TMA_DESC_INIT_FUNC = """
\tCUtensorMap {0};
\tCUtensorMapDataType {0}_type= (CUtensorMapDataType){1};
\tcuuint32_t {0}_tensorRank= {2};
\tvoid *{0}_globalAddress= {3};
\tcuuint64_t {0}_globalDim[{2}]= {{{4}}};
\tcuuint64_t {0}_globalStride[{2}]= {{{5}}};
\tcuuint32_t {0}_boxDim[{2}]= {{{6}}};
\tcuuint32_t {0}_elementStrides[{2}]= {{{7}}};
\tCUtensorMapInterleave {0}_interleave= (CUtensorMapInterleave){8};
\tCUtensorMapSwizzle {0}_swizzle= (CUtensorMapSwizzle){9};
\tCUtensorMapL2promotion {0}_l2Promotion= (CUtensorMapL2promotion){10};
\tCUtensorMapFloatOOBfill {0}_oobFill= (CUtensorMapFloatOOBfill){11};

\tCUresult {0}_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &{0}, {0}_type, {0}_tensorRank, {0}_globalAddress, {0}_globalDim, {0}_globalStride + 1, {0}_boxDim, {0}_elementStrides, {0}_interleave, {0}_swizzle, {0}_l2Promotion, {0}_oobFill);

\tif ({0}_result != CUDA_SUCCESS) {{
\t\tstd::stringstream ss;
\t\tss << "Error: Failed to initialize the TMA descriptor {0}";
\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
\t\treturn -1;
\t}}
"""

TMA_DESC_INIT_FUNC_PY = """
\t{0}_type = cuda.bindings.driver.CUtensorMapDataType({1})
\t{0}_tensorRank = {2}
\t{0}_globalAddress = {3}.data_ptr()
\t{0}_globalDim = [{4}]
\t{0}_globalStride = [{5}][1:]
\t{0}_boxDim = [{6}]
\t{0}_elementStrides = [{7}]
\t{0}_interleave = cuda.bindings.driver.CUtensorMapInterleave({8})
\t{0}_swizzle = cuda.bindings.driver.CUtensorMapSwizzle({9})
\t{0}_l2Promotion = cuda.bindings.driver.CUtensorMapL2promotion({10})
\t{0}_oobFill = cuda.bindings.driver.CUtensorMapFloatOOBfill({11})

\tres, {0} = cuda.bindings.driver.cuTensorMapEncodeTiled(
\t\t{0}_type,
\t\t{0}_tensorRank,
\t\t{0}_globalAddress,
\t\t{0}_globalDim,
\t\t{0}_globalStride,
\t\t{0}_boxDim,
\t\t{0}_elementStrides,
\t\t{0}_interleave,
\t\t{0}_swizzle,
\t\t{0}_l2Promotion,
\t\t{0}_oobFill,
\t)

\tif res != cuda.bindings.driver.CUresult.CUDA_SUCCESS:
\t\traise RuntimeError(f"Failed to initialize the TMA descriptor {0}: {{res}}")
"""

KERNEL_LAUNCH_FUNC_PY = """
\tres = cuda.bindings.driver.cuKernelSetAttribute(
\t\tcuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
\t\t{7},
\t\tkernels["{0}"],
\t\tcuda.bindings.driver.CUdevice({10})
\t)[0]
\tif res != cuda.bindings.driver.CUresult.CUDA_SUCCESS:
\t\traise RuntimeError(f"Failed to set max dynamic shared memory size to {7} for kernel {0}: {{res}}")

\tconfig = cuda.bindings.driver.CUlaunchConfig()
\tconfig.gridDimX = {1}
\tconfig.gridDimY = {2}
\tconfig.gridDimZ = {3}
\tconfig.blockDimX = {4}
\tconfig.blockDimY = {5}
\tconfig.blockDimZ = {6}
\tconfig.sharedMemBytes = {7}
\tconfig.hStream = stream

\targ_values = {8}
\targ_types = {9}

\tres = cuda.bindings.driver.cuLaunchKernelEx(config, kernels["{0}"], (arg_values, arg_types), 0)[0]
\tif res != cuda.bindings.driver.CUresult.CUDA_SUCCESS:
\t\traise RuntimeError(f"Failed to launch kernel {0}: {{res}}")
"""


class BaseWrapper(ABC):

    @abstractmethod
    def wrap(self, *args, **kwargs):
        raise NotImplementedError


logger = logging.getLogger(__name__)


class TLCUDASourceWrapper(object):
    _TYPE_MAP = {
        "float32": "float",
        "float16": "half_t",
        "bfloat16": "bfloat16_t",
        "float8_e4m3": "fp8_e4_t",
        "float8_e5m2": "fp8_e5_t",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int",
        "uint32": "unsigned int",
        "bool": "int8_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "uchar": "uint8_t",
    }

    backend = "tl"
    device_mod: Optional[IRModule] = None
    host_mod: Optional[IRModule] = None
    pass_configs: Optional[Dict[str, Any]] = None

    def __init__(self,
                 scheduled_ir_module: IRModule,
                 source: str,
                 target: Target,
                 device_mod: Optional[IRModule] = None,
                 host_mod: Optional[IRModule] = None,
                 pass_configs: Optional[Dict[str, Any]] = None):
        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.pass_configs = pass_configs
        self.device_mod = device_mod
        self.host_mod = host_mod
        self.function_names: Optional[str] = None
        self.dynamic_smem_buf: Optional[int] = None
        self.block_info: Union[List[int], Dict] = [1, 1, 1]
        self.grid_info: Union[List[int], Dict] = [1, 1, 1]
        self.tma_descriptor_args: Optional[Dict] = None
        self.l2_persistent_map: Optional[Dict[str, Dict]] = {}
        self.parse_source_information()
        self.srcpath: Optional[str] = None
        self.libpath: Optional[str] = None
        self.lib_code: Optional[str] = self.update_lib_code(source)

    def _pythonic_expr(self, expr: tvm.tir.PrimExpr) -> str:
        return pythonic_expr(expr, self._TYPE_MAP)

    def is_tma_descriptor_arg(self, arg_name: str) -> bool:
        return arg_name in self.prim_func.buffer_map

    def create_dispatch_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = []
        # Collect function arguments based on primary function's parameters and buffer mappings
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append({
                    "name": buffer.data.name,
                    "type": self._TYPE_MAP[buffer.dtype] + "* __restrict__",
                })
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._TYPE_MAP[param.dtype]})
            else:
                raise ValueError(
                    f"Parameter {param} is not in the buffer map of the primary function.")
        # Add dynamic symbols as integer arguments
        for dyn_sym in dynamic_symbolic_set:
            if dyn_sym not in [arg["name"] for arg in function_args]:
                function_args.append({"name": dyn_sym, "type": "int"})

        function_args.append(self.get_stream_type())

        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        def func_call_args(s, function_args, desc_name_map: Optional[Dict[str, str]] = None):
            # Extract the function call arguments matching the function definition
            def maybe_desc(name: str, matches: List[str], i: int):
                match = matches[i]
                if not (match == name + "_desc" or match.startswith(name + "_desc_")):
                    return False
                desc_decls = []
                if desc_name_map is not None:
                    desc_name_map[match] = name
                if i > 0:
                    desc_decls.append(matches[i - 1])
                if i < len(matches) - 1:
                    desc_decls.append(matches[i + 1])
                return any([decl == "CUtensorMap" for decl in desc_decls])

            pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for i, match in enumerate(matches):
                for arg in function_args:
                    if arg["name"] == match or maybe_desc(arg["name"], matches, i):
                        call_args.append(match)
            return call_args

        has_l2_persistent_map = False
        for function_name, _ in function_informations.items():
            if function_name in self.l2_persistent_map:
                has_l2_persistent_map = True
                break

        kernel_launch_code = """"""
        if has_l2_persistent_map:
            kernel_launch_code += L2_PERSISTENT_MAP_CREATE_HANDLE
        desc_name_map: Dict[str, str] = {}
        for function_name, function_info in function_informations.items():
            block_info = function_info["block_info"]
            grid_info = function_info["grid_info"]
            dynamic_smem_buf = function_info["dynamic_smem_buf"]

            # Find the location of the global kernel function in the code
            index = match_declare_kernel(code, function_name + "(")

            # Analyze the function declaration to prepare for argument extraction
            declaration = code[index:].split(";")[0]

            # Identify the start of the function body to insert arguments
            index = code.index("{", index)

            block_str = "dim3({}, {}, {})".format(
                self._pythonic_expr(block_info[0]),
                self._pythonic_expr(block_info[1]),
                self._pythonic_expr(block_info[2]),
            )
            grid_str = "dim3({}, {}, {})".format(
                self._pythonic_expr(grid_info[0]), self._pythonic_expr(grid_info[1]),
                self._pythonic_expr(grid_info[2]))
            smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf
            init_l2_persistent_map = self.generate_l2_persistent_map(function_name)
            kernel_launch_code += init_l2_persistent_map

            if self.use_cooperative_groups[function_name]:
                args_list = func_call_args(declaration, function_args, desc_name_map)
                args_array = [f"(void*)&{arg}" for arg in args_list]
                call_args = f"\tvoid* {function_name}_args[] = {{{', '.join(args_array)}}};\n"
                kernel_launch_code += call_args
                # Using cudaLaunchCooperativeKernel to launch the kernel
                kernel_launch_code += "\tTILELANG_CHECK(cudaLaunchCooperativeKernel((void*){}, {}, {}, {}, {}, stream));\n".format(
                    function_name, grid_str, block_str, function_name + "_args", smem_str)
            else:
                call_args = ", ".join(func_call_args(declaration, function_args, desc_name_map))
                kernel_launch_code += "\t{}<<<{}, {}, {}, stream>>>({});\n".format(
                    function_name, grid_str, block_str, smem_str, call_args)
                kernel_launch_code += "\tTILELANG_CHECK_LAST_ERROR(\"{}\");\n".format(function_name)
            if has_l2_persistent_map:
                kernel_launch_code += L2_PERSISTENT_MAP_RESET_HANDLE

        init_tma_descriptor_args = self.generate_tma_descriptor_args(desc_name_map)
        kernel_launch_code = init_tma_descriptor_args + kernel_launch_code

        # Wrap the kernel dispatch logic in an external C function
        host_func = PREDEF_HOST_FUNC.format(def_args, kernel_launch_code)
        return host_func

    def generate_l2_persistent_map(self, function_name: str) -> str:
        if function_name not in self.l2_persistent_map:
            return ""
        init_l2_persistent_map = ""
        for buffer_name, (hit_ratio,
                          size_in_bytes) in self.l2_persistent_map[function_name].items():
            # get persisting_l2_cache_max_size
            from tilelang.carver.arch.driver import get_persisting_l2_cache_max_size
            persisting_l2_cache_max_size = get_persisting_l2_cache_max_size()
            try:
                num_bytes = min(size_in_bytes, persisting_l2_cache_max_size)
            except Exception:
                # as size_in_bytes maybe a symbolic expression
                num_bytes = persisting_l2_cache_max_size
            init_l2_persistent_map += L2_PERSISTENT_MAP_INIT_FUNC.format(
                buffer_name, float(hit_ratio), self._pythonic_expr(num_bytes))

        return init_l2_persistent_map

    def generate_tma_descriptor_args(self, desc_name_map: Dict[str, str]) -> str:
        tma_descripter_init = ""
        if self.tma_descriptor_args is None:
            return tma_descripter_init

        for handle_name, name in desc_name_map.items():
            desc_name = name + "_desc"
            assert desc_name in self.tma_descriptor_args, f"TMA descriptor {desc_name} not found in {self.tma_descriptor_args}"
            args = self.tma_descriptor_args[desc_name]
            # Skip __tvm_tensormap_create_tiled
            if len(args) < 3:
                raise ValueError(
                    f"TMA descriptor args too short: {len(args)} elements, expected at least 3")
            _, dtype, tensor_rank, globalAddress, *remaining_args = args[1:]
            dtype = self._pythonic_expr(dtype)
            tensor_rank = int(self._pythonic_expr(tensor_rank))

            # Validate tensor_rank
            if not isinstance(tensor_rank, int) or tensor_rank <= 0:
                raise ValueError(f"Invalid tensor_rank: {tensor_rank}. Must be a positive integer")

            # Calculate required length for remaining_args
            expected_args_len = 4 * tensor_rank + 4  # 4 groups of tensor_rank size + 4 parameters
            if len(remaining_args) < expected_args_len:
                raise ValueError(f"Insufficient remaining args: got {len(remaining_args)}, "
                                 f"expected {expected_args_len} for tensor_rank {tensor_rank}")

            # Extract dimensions and strides using list slicing
            global_dim = remaining_args[:tensor_rank]
            global_stride = remaining_args[tensor_rank:2 * tensor_rank]
            box_dim = remaining_args[2 * tensor_rank:3 * tensor_rank]
            element_strides = remaining_args[3 * tensor_rank:4 * tensor_rank]

            global_dim = [self._pythonic_expr(i) for i in global_dim]
            global_stride = [self._pythonic_expr(i) for i in global_stride]
            box_dim = [self._pythonic_expr(i) for i in box_dim]
            element_strides = [self._pythonic_expr(i) for i in element_strides]

            # Extract remaining parameters
            try:
                interleave, swizzle, l2Promotion, oobFill = remaining_args[4 * tensor_rank:4 *
                                                                           tensor_rank + 4]
                interleave = self._pythonic_expr(interleave)
                swizzle = self._pythonic_expr(swizzle)
                l2Promotion = self._pythonic_expr(l2Promotion)
                oobFill = self._pythonic_expr(oobFill)
            except ValueError as e:
                raise ValueError(
                    "Failed to unpack the final 4 TMA parameters (interleave, swizzle, l2Promotion, oobFill)"
                ) from e

            tma_descripter_init += TMA_DESC_INIT_FUNC.format(handle_name, dtype, tensor_rank,
                                                             globalAddress, ",".join(global_dim),
                                                             ",".join(global_stride),
                                                             ",".join(box_dim),
                                                             ",".join(element_strides), interleave,
                                                             swizzle, l2Promotion, oobFill)
        return tma_descripter_init

    def parse_source_information(self):
        if self.device_mod is None or self.host_mod is None:
            with tvm.transform.PassContext(opt_level=3, config=self.pass_configs):
                device_mod, host_mod = get_annotated_mod(self.mod, self.target)
            self.device_mod = device_mod
            self.host_mod = host_mod
        assert (len(self.device_mod.functions)
                >= 1), "Device module should have at least one function."
        assert (len(self.host_mod.functions) == 1), "Only support one function in host module."

        block_info_map = {}
        grid_info_map = {}
        dynamic_smem_buf_map = {}
        function_names = []
        use_cooperative_groups_map = {}
        for g_var, func in self.device_mod.functions.items():
            # Default block and grid configurations
            block_info = [1, 1, 1]
            grid_info = [1, 1, 1]
            function_name = g_var.name_hint
            attrs = func.attrs
            dynamic_smem_buf = None
            use_cooperative_groups = False
            if "use_cooperative_groups" in attrs:
                use_cooperative_groups = attrs["use_cooperative_groups"]
            if "dyn_shared_memory_buf" in attrs:
                dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
            if "thread_extent" in attrs:
                # Extract block and grid sizes from thread extents
                thread_extent = attrs["thread_extent"]
                for tag, extent in thread_extent.items():
                    if "threadIdx" in tag:
                        block_info["xyz".index(tag[-1])] = extent
                    elif "blockIdx" in tag:
                        grid_info["xyz".index(tag[-1])] = extent
            # Map the extracted configurations to each function
            block_info_map[function_name] = block_info
            grid_info_map[function_name] = grid_info
            dynamic_smem_buf_map[function_name] = dynamic_smem_buf
            use_cooperative_groups_map[function_name] = use_cooperative_groups
            function_names.append(function_name)

        # Store the mappings for use in code generation
        self.block_info = block_info_map
        self.grid_info = grid_info_map
        self.dynamic_smem_buf = dynamic_smem_buf_map
        self.use_cooperative_groups = use_cooperative_groups_map

        function_names_index = {}
        for _, func in self.host_mod.functions.items():
            if "tma_descriptor_args" in func.attrs:
                self.tma_descriptor_args = func.attrs["tma_descriptor_args"]
            if "l2_persistent_map" in func.attrs:
                self.l2_persistent_map[function_name] = func.attrs["l2_persistent_map"]

            host_code = str(func)
            for function_name in function_names:
                index = host_code.index(f'T.call_packed("{function_name}"')
                function_names_index[function_name] = index
        # sort function_names
        function_names = sorted(function_names, key=lambda x: function_names_index[x])
        self.function_names = function_names

    def get_dynamic_symbolic_set(self, prim_func):
        # Determine the set of dynamic symbols used in the function
        dynamic_symbolic_set: List[str] = []
        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for dim in buffer.shape:
                    if isinstance(dim, tvm.tir.Var) and (dim.name not in dynamic_symbolic_set):
                        dynamic_symbolic_set.append(dim.name)
        return dynamic_symbolic_set

    def get_init_func(self):
        # Initialize an empty string for the CUDA function call
        call_str = """"""
        # If dynamic shared memory buffer is specified, prepare the cudaFuncSetAttribute call
        for function_name, dynamic_smem_buf in self.dynamic_smem_buf.items():
            if dynamic_smem_buf is not None:
                # Format the cudaFuncSetAttribute call for dynamic shared memory
                call_str += PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY.format(
                    function_name, dynamic_smem_buf)
        # Format the initialization function using the call_str
        init_funcs = PREDEF_INIT_FUNC.format(call_str)
        return init_funcs

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code
        # Get the function names
        function_names = self.function_names
        # Get the CUDA initialization function
        init_func = self.get_init_func()

        # Organize function information for code generation
        function_informations = {}
        for function_name in function_names:
            # Do not update function with dispatch host function
            if (function_name not in self.block_info) or (function_name not in self.grid_info):
                continue

            function_informations[function_name] = {
                "function_name": function_name,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
            }

        # Create the host function wrapper for the CUDA kernel
        host_func = self.create_dispatch_func(code, function_informations)
        # Combine the source, initialization function, and host function to form the complete library code
        lib_code = self.source + init_func + host_func
        return lib_code

    def get_stream_type(self) -> Dict[str, str]:
        return {"name": "stream=cudaStreamDefault", "type": "cudaStream_t"}

    @property
    def prim_func(self):
        if len(self.mod.get_global_vars()) == 1:
            return self.mod[self.mod.get_global_vars()[0]]
        elif "main" in self.mod:
            return self.mod["main"]
        else:
            for _, function in self.mod.functions_items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")


class TLNVRTCSourceWrapper(TLCUDASourceWrapper):
    """
    A wrapper class for the TileLang NVRTC backend.
    """

    _TYPE_MAP = {
        "float32": "ctypes.c_float",
        "float16": "ctypes.c_uint16",
        "bfloat16": "ctypes.c_uint16",
        "float8_e4m3": "ctypes.c_uint8",
        "float8_e5m2": "ctypes.c_uint8",
        "float64": "ctypes.c_double",
        "int64": "ctypes.c_int64",
        "int32": "ctypes.c_int32",
        "uint32": "ctypes.c_uint32",
        "bool": "ctypes.c_bool",
        "int8": "ctypes.c_int8",
        "uint8": "ctypes.c_uint8",
        "int16": "ctypes.c_int16",
        "uint16": "ctypes.c_uint16",
        "uchar": "ctypes.c_uint8",
    }

    def __init__(self,
                 scheduled_ir_module: IRModule,
                 source: str,
                 target: Target,
                 device_mod: Optional[IRModule] = None,
                 host_mod: Optional[IRModule] = None,
                 pass_configs: Optional[Dict[str, Any]] = None):
        super().__init__(scheduled_ir_module, source, target, device_mod, host_mod, pass_configs)

    def create_dispatch_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = [{"name": "kernels", "type": "Dict[str, cuda.bindings.driver.CUkernel]"}]
        # Collect function arguments based on primary function's parameters and buffer mappings
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append({
                    "name": buffer.data.name,
                    "type": "ctypes.c_void_p",
                })
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._TYPE_MAP[param.dtype]})
            else:
                raise ValueError(
                    f"Parameter {param} is not in the buffer map of the primary function.")
        # Add dynamic symbols as integer arguments
        for dyn_sym in dynamic_symbolic_set:
            if dyn_sym not in [arg["name"] for arg in function_args]:
                function_args.append({"name": dyn_sym, "type": "ctypes.c_int"})

        function_args.append(self.get_stream_type())

        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['name']}" for arg in function_args])

        def func_call_args(s, function_args, desc_name_map: Optional[Dict[str, str]] = None):
            # Extract the function call arguments matching the function definition
            def maybe_desc(name: str, matches: List[str], i: int):
                match = matches[i]
                if not (match == name + "_desc" or match.startswith(name + "_desc_")):
                    return False
                desc_decls = []
                if desc_name_map is not None:
                    desc_name_map[match] = name
                if i > 0:
                    desc_decls.append(matches[i - 1])
                if i < len(matches) - 1:
                    desc_decls.append(matches[i + 1])
                return any([decl == "CUtensorMap" for decl in desc_decls])

            pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for i, match in enumerate(matches):
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(
                            (f"{match}.data_ptr()" if arg["type"] == "ctypes.c_void_p" else match,
                             arg["type"]))
                    elif maybe_desc(arg["name"], matches, i):
                        call_args.append((match, "None"))
            return call_args

        desc_name_map: Dict[str, str] = {}
        device_index = 0
        kernel_launch_code = """"""
        for function_name, function_info in function_informations.items():
            block_info = function_info["block_info"]
            grid_info = function_info["grid_info"]
            dynamic_smem_buf = function_info["dynamic_smem_buf"]

            # Find the location of the global kernel function in the code
            index = match_declare_kernel(code, function_name + "(")

            # Analyze the function declaration to prepare for argument extraction
            declaration = code[index:].split(";")[0]

            # Identify the start of the function body to insert arguments
            index = code.index("{", index)
            call_args = func_call_args(declaration, function_args, desc_name_map)
            for arg_name, arg_type in call_args:
                if arg_type == "ctypes.c_void_p":
                    device_index = f"{arg_name.replace('.data_ptr()', '')}.device.index"
                    break
            arg_names = ", ".join([arg[0] for arg in call_args])
            arg_types = ", ".join([arg[1] for arg in call_args])
            smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf
            kernel_launch_code += self.generate_tma_descriptor_args(
                desc_name_map) + KERNEL_LAUNCH_FUNC_PY.format(
                    function_name, self._pythonic_expr(grid_info[0]),
                    self._pythonic_expr(grid_info[1]), self._pythonic_expr(grid_info[2]),
                    self._pythonic_expr(block_info[0]), self._pythonic_expr(block_info[1]),
                    self._pythonic_expr(
                        block_info[2]), smem_str, arg_names, arg_types, device_index)

        # Wrap the kernel dispatch logic in an external C function
        host_func = PREDEF_HOST_FUNC_PY.format(
            repr(list(function_informations.keys())), def_args, kernel_launch_code)
        return host_func

    def generate_tma_descriptor_args(self, desc_name_map: Dict[str, str]) -> str:
        tma_descripter_init = ""
        if self.tma_descriptor_args is None:
            return tma_descripter_init

        for handle_name, name in desc_name_map.items():
            desc_name = name + "_desc"
            assert desc_name in self.tma_descriptor_args, f"TMA descriptor {desc_name} not found in {self.tma_descriptor_args}"
            args = self.tma_descriptor_args[desc_name]
            # Skip __tvm_tensormap_create_tiled
            if len(args) < 3:
                raise ValueError(
                    f"TMA descriptor args too short: {len(args)} elements, expected at least 3")
            _, dtype, tensor_rank, globalAddress, *remaining_args = args[1:]

            tensor_rank = int(tensor_rank)
            # Validate tensor_rank
            if not isinstance(tensor_rank, int) or tensor_rank <= 0:
                raise ValueError(f"Invalid tensor_rank: {tensor_rank}. Must be a positive integer")

            # Calculate required length for remaining_args
            # 4 groups of tensor_rank size + 4 parameters
            expected_args_len = 4 * tensor_rank + 4
            if len(remaining_args) < expected_args_len:
                raise ValueError(f"Insufficient remaining args: got {len(remaining_args)}, "
                                 f"expected {expected_args_len} for tensor_rank {tensor_rank}")

            # Extract dimensions and strides using list slicing
            global_dim = remaining_args[:tensor_rank]
            global_stride = remaining_args[tensor_rank:2 * tensor_rank]
            box_dim = remaining_args[2 * tensor_rank:3 * tensor_rank]
            element_strides = remaining_args[3 * tensor_rank:4 * tensor_rank]

            global_dim = [str(i) for i in global_dim]
            global_stride = [str(i) for i in global_stride]
            box_dim = [str(i) for i in box_dim]
            element_strides = [str(i) for i in element_strides]

            # Extract remaining parameters
            try:
                interleave, swizzle, l2Promotion, oobFill = remaining_args[4 * tensor_rank:4 *
                                                                           tensor_rank + 4]
            except ValueError as e:
                raise ValueError(
                    "Failed to unpack the final 4 TMA parameters (interleave, swizzle, l2Promotion, oobFill)"
                ) from e

            tma_descripter_init += TMA_DESC_INIT_FUNC_PY.format(
                handle_name, dtype, tensor_rank, globalAddress,
                ", ".join(map(lambda x: f"cuda.bindings.driver.cuuint64_t({x})", global_dim)),
                ", ".join(map(lambda x: f"cuda.bindings.driver.cuuint64_t({x})", global_stride)),
                ", ".join(map(lambda x: f"cuda.bindings.driver.cuuint32_t({x})", box_dim)),
                ", ".join(map(lambda x: f"cuda.bindings.driver.cuuint32_t({x})",
                              element_strides)), interleave, swizzle, l2Promotion, oobFill)
        return tma_descripter_init

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code

        # Organize function information for code generation
        function_informations = {}
        for function_name in self.function_names:
            # Do not update function with dispatch host function
            if (function_name not in self.block_info) or (function_name not in self.grid_info):
                continue

            function_informations[function_name] = {
                "function_name": function_name,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
            }

        # Create the host function wrapper for the CUDA kernel
        self.host_func = self.create_dispatch_func(code, function_informations)
        return self.lib_code

    def get_stream_type(self) -> Dict[str, str]:
        return {"name": "stream=0", "type": "int"}


class TLHIPSourceWrapper(TLCUDASourceWrapper):
    """
    A wrapper class for the TileLang HIP backend.
    """

    _TYPE_MAP = {
        "float32": "float",
        "float16": "half_t",
        "bfloat16": "bfloat16_t",
        "float8_e4m3": "fp8_e4_t",
        "float8_e5m2": "fp8_e5_t",
        "float8_e4m3fnuz": "fp8_e4_t",
        "e4m3fnuz_float8": "fp8_e4_t",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int",
        "uint32": "unsigned int",
        "bool": "int8_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "uchar": "uint8_t",
    }

    def __init__(self,
                 scheduled_ir_module: IRModule,
                 source: str,
                 target: Target,
                 device_mod: Optional[IRModule] = None,
                 host_mod: Optional[IRModule] = None,
                 pass_configs: Optional[Dict[str, Any]] = None):
        super().__init__(scheduled_ir_module, source, target, device_mod, host_mod, pass_configs)

    def get_init_func(self):
        # Initialize an empty string for the CUDA function call
        call_str = """"""
        # If dynamic shared memory buffer is specified, prepare the cudaFuncSetAttribute call
        for function_name, dynamic_smem_buf in self.dynamic_smem_buf.items():
            if dynamic_smem_buf is not None:
                # Format the cudaFuncSetAttribute call for dynamic shared memory
                call_str += PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY_HIP.format(
                    function_name, dynamic_smem_buf)
        # Format the initialization function using the call_str
        init_funcs = PREDEF_INIT_FUNC.format(call_str)
        return init_funcs

    def get_stream_type(self) -> Dict[str, str]:
        return {"name": "stream=hipStreamDefault", "type": "hipStream_t"}


class TLCPUSourceWrapper(object):
    _TYPE_MAP = {
        "float32": "float",
        "float16": "half",
        "int32": "int32_t",
    }

    INIT_FUNC = textwrap.dedent('''
        #ifdef __cplusplus
        extern "C"
        #endif
        int32_t init() {
            return 0;
        }
    ''')

    CALL_PREFIX = textwrap.dedent("""
        #ifdef __cplusplus
        extern "C"
        #endif
        int32_t call({}) {{
          return {};
        }}
    """)

    backend = "tl"
    device_mod: Optional[IRModule] = None
    host_mod: Optional[IRModule] = None
    pass_configs: Optional[Dict[str, Any]] = None

    def __init__(self,
                 scheduled_ir_module: IRModule,
                 source: str,
                 target: Target,
                 device_mod: Optional[IRModule] = None,
                 host_mod: Optional[IRModule] = None,
                 pass_configs: Optional[Dict[str, Any]] = None):
        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.device_mod = device_mod
        self.host_mod = host_mod
        self.pass_configs = pass_configs
        self.function_names: Optional[str] = None
        self.dynamic_smem_buf: Optional[int] = None
        self.parse_source_information()
        self.srcpath: Optional[str] = None
        self.libpath: Optional[str] = None
        self.lib_code: Optional[str] = self.update_lib_code(source)

    def create_call_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = []
        # Collect function arguments based on primary function's parameters and buffer mappings
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append({
                    "name": buffer.name,
                    "type": self._TYPE_MAP[buffer.dtype] + "*",
                })
            elif isinstance(param, tvm.tir.Var):
                function_args.append({"name": param.name, "type": self._TYPE_MAP[param.dtype]})
            else:
                raise ValueError(
                    f"Parameter {param} is not in the buffer map of the primary function.")
        # Add dynamic symbols as integer arguments
        for dyn_sym in dynamic_symbolic_set:
            function_args.append({"name": dyn_sym, "type": "int"})
        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        def func_call_args(s, function_args):
            pattern = r"[,\s]*(?:\w+\s*\*+\s*\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for match in matches:
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(match)
            return call_args

        _call_str = """"""

        for function_name, _ in function_informations.items():

            # Find the location of the global kernel function in the code
            index = match_declare_kernel_cpu(code, function_name + "(")

            # Analyze the function declaration to prepare for argument extraction
            declaration = code[index:].split(";")[0]

            # Identify the start of the function body to insert arguments
            index = code.index("{", index)

            call_args = ", ".join(func_call_args(declaration, function_args))
            _call_str += "{}({})".format(function_name, call_args)

        # Wrap the kernel dispatch logic in an external C function
        host_func = self.CALL_PREFIX.format(def_args, _call_str)
        return host_func

    def parse_source_information(self):
        with tvm.transform.PassContext(opt_level=3, config=self.pass_configs):
            device_mod, host_mod = get_annotated_mod(self.mod, self.target)
        assert (len(device_mod.functions) >= 1), "Device module should have at least one function."
        assert (len(host_mod.functions) == 1), "Only support one function in host module."

        function_names = []
        for g_var, _ in device_mod.functions.items():
            function_name = g_var.name_hint
            function_names.append(function_name)

        self.function_names = function_names

    def get_dynamic_symbolic_set(self, prim_func):
        # Determine the set of dynamic symbols used in the function
        dynamic_symbolic_set: List[str] = []
        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for dim in buffer.shape:
                    if isinstance(dim, tvm.tir.Var) and (dim.name not in dynamic_symbolic_set):
                        dynamic_symbolic_set.append(dim.name)
        return dynamic_symbolic_set

    def get_cpu_init_func(self):
        init_funcs = self.INIT_FUNC
        return init_funcs

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code
        # Get the function names
        function_names = self.function_names
        # Get the CPU initialization function
        init_func = self.get_cpu_init_func()

        # Organize function information for code generation
        function_informations = {}
        for function_name in function_names:
            function_informations[function_name] = {
                "function_name": function_name,
            }

        # Create the call function wrapper for the CPU kernel
        call_func = self.create_call_func(code, function_informations)
        # Combine the source, initialization function, and call function to form the complete library code
        lib_code = self.source + init_func + call_func
        return lib_code

    @property
    def prim_func(self):
        if len(self.mod.get_global_vars()) == 1:
            return self.mod[self.mod.get_global_vars()[0]]
        elif "main" in self.mod:
            return self.mod["main"]
        else:
            for _, function in self.mod.functions_items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")


class TLWrapper(BaseWrapper):
    """
    A wrapper class for the TileLang backend.
    """
    device_mod: Optional[IRModule] = None
    host_mod: Optional[IRModule] = None
    pass_configs: Optional[Dict[str, Any]] = None
    target: Optional[Target] = None
    lib: Optional[object] = None

    def __init__(self, target: Target):
        super().__init__()
        self.scheduled_ir_module = None
        self.pass_configs = None
        self.target = target
        self.lib = None

    def assign_optimized_module(self, scheduled_ir_module: IRModule):
        self.scheduled_ir_module = scheduled_ir_module

    def assign_pass_configs(self, pass_configs: Dict[str, Any]):
        self.pass_configs = pass_configs

    def assign_host_module(self, host_mod: IRModule):
        self.host_mod = host_mod

    def assign_device_module(self, device_mod: IRModule):
        self.device_mod = device_mod

    # Get Scheduled Rt Module and return source to be compiled
    def wrap(self, c_source: str):
        assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        if is_cuda_target(self.target):
            wrapper_class = TLCUDASourceWrapper
        elif is_hip_target(self.target):
            wrapper_class = TLHIPSourceWrapper
        elif is_cpu_target(self.target):
            wrapper_class = TLCPUSourceWrapper
        else:
            raise ValueError(f"Unsupported platform: {self.arch.platform}")
        wrapper = wrapper_class(
            scheduled_ir_module=self.scheduled_ir_module,
            source=c_source,
            target=self.target,
            device_mod=self.device_mod,
            host_mod=self.host_mod,
            pass_configs=self.pass_configs)
        return wrapper.lib_code


class TLPyWrapper(TLWrapper):

    def __init__(self, target: Target):
        super().__init__(target)

    def wrap(self, c_source: str):
        # assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        if is_cuda_target(self.target):
            wrapper_class = TLNVRTCSourceWrapper
        else:
            raise ValueError(f"Unsupported platform: {self.arch.platform}")
        wrapper = wrapper_class(
            scheduled_ir_module=self.scheduled_ir_module,
            source=c_source,
            target=self.target,
            device_mod=self.device_mod,
            host_mod=self.host_mod,
            pass_configs=self.pass_configs)
        return wrapper.host_func, wrapper.function_names
