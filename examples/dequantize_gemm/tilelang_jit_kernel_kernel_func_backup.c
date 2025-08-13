#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  uchar B_local[32];
  bfloat16_t B_dequantize_local[64];
  __shared__ uint64_t _mbarrier[4];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 2);
    tl::mbarrier_init(_mbarrier[3], 2);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 64; ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 1) + 2)], (((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 65536);
        tl::tma_load<tl::CacheHintSm90::EVICT_NORMAL>(A_desc, _mbarrier[(k & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[((k & 1) * 32768)])), (k * 128), (((int)blockIdx.y) * 256));
        tl::tma_load<tl::CacheHintSm90::EVICT_NORMAL>(A_desc, _mbarrier[(k & 1)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 16384)])), ((k * 128) + 64), (((int)blockIdx.y) * 256));
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 8192);
        tl::tma_load<tl::CacheHintSm90::EVICT_NORMAL>(B_desc, _mbarrier[(k & 1)], (&(buf_dyn_shmem[(((k & 1) * 8192) + 163840)])), (k * 64), (((int)blockIdx.x) * 128));
      }
      tl::mbarrier_arrive(_mbarrier[(k & 1)]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(C_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    for (int k_1 = 0; k_1 < 64; ++k_1) {
      tl::mbarrier_wait(_mbarrier[(k_1 & 1)], ((k_1 & 3) >> 1));
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        *(uint4*)(B_local + (i_1 * 16)) = *(uint4*)(buf_dyn_shmem + (((((((k_1 & 1) * 8192) + (i_1 * 4096)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 163840));
      }
      // #pragma unroll
      // for (int i_2 = 0; i_2 < 64; ++i_2) {
      //     ushort v_ = (ushort)(((((int)((((((ushort)B_local[(i_2 >> 1)]) >> (((ushort)(i_2 & 1)) * (ushort)4)) & (ushort)15) >> (ushort)3) << (ushort)8)) | ((int)(((((((ushort)B_local[(i_2 >> 1)]) >> (((ushort)(i_2 & 1)) * (ushort)4)) & (ushort)15) & (ushort)6) >> (ushort)1) + (ushort)126))) << 7) | ((int)((((((ushort)B_local[(i_2 >> 1)]) >> (((ushort)(i_2 & 1)) * (ushort)4)) & (ushort)15) & (ushort)1) << (ushort)6)));
      //   B_dequantize_local[i_2] = (*(bfloat16_t *)(&(v_)));
      // }
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        uint32_t B_dequantize_local_vec[4];
        uint32_t tmp;
        uint32_t bias;
        uint32_t d0, d1, d2, d3, d4, d5, d6;
        asm volatile(
          // To handle the endianness issue
          "prmt.b32 %13, %4, 0, 0x0123;"
          "mov.b32 %12, 0x7e807e80;"
          "and.b32 %0, %13, 0b10000001110000001000000111000000;"
          "mul.bf16x2 %0, %0, %12;"
          "shl.b32 %1, %13, 3;"
          "and.b32 %1, %1, 0b10000001110000001000000111000000;"          
          "mul.bf16x2 %1, %1, %12;"
          "shl.b32 %2, %13, 6;"
          "and.b32 %2, %2, 0b10000001110000001000000111000000;"
          "mul.bf16x2 %2, %2, %12;"
          "shl.b32 %5, %13, 1;"
          "and.b32 %6, %5, 0b10000000000000001000000000000000;"
          "shr.b32 %7, %13, 3;"
          "and.b32 %8, %7, 0b00000001100000000000000110000000;"
          "or.b32 %9, %6, %8;"
          "shr.b32 %10, %13, 7;"
          "and.b32 %11, %10, 0b00000000010000000000000001000000;"
          "or.b32 %3, %9, %11;"
          "mul.bf16x2 %3, %3, %12;"
          :"=r"(B_dequantize_local_vec[0])
          ,"=r"(B_dequantize_local_vec[1])
          ,"=r"(B_dequantize_local_vec[2])
          ,"=r"(B_dequantize_local_vec[3])
          :"r"(*(uint32_t*)&B_local[i_2 << 2]), "r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(d4), "r"(d5), "r"(d6), "r"(bias), "r"(tmp)
        );
        for (int j = 0; j < 4; ++j) {
          // Pay attention to the big-endianness issue
          *(__nv_bfloat16*)&B_dequantize_local[(i_2 << 3) + j] = reinterpret_cast<__nv_bfloat162*>(&B_dequantize_local_vec[j])->y;
          *(__nv_bfloat16*)&B_dequantize_local[(i_2 << 3) + j + 4] = reinterpret_cast<__nv_bfloat162*>(&B_dequantize_local_vec[j])->x;
        }
      }
      cutlass::arch::NamedBarrier::sync(256, 0);
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 3) >> 1) * 8192) + ((i_3 >> 2) * 4096)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_3 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 8)) + 65536)) = *(uint4*)(B_dequantize_local + (i_3 * 8));
      }
      tl::fence_proxy_async();
      cutlass::arch::NamedBarrier::sync(256, 0);
      tl::gemm_ss<256, 128, 128, 4, 2, 0, 1, 0, 128, 128, 0, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[((k_1 & 1) * 32768)])), (&(((bfloat16_t*)buf_dyn_shmem)[65536])), (&(C_local[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 2)], 0, ((((int)threadIdx.x) % 128) == 0));
    }
    cutlass::arch::NamedBarrier::sync(256, 0);
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) >> 7) * 16384) + ((i_4 >> 2) * 4096)) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)C_local[(i_4 * 8)]), ((bfloat16_t)C_local[((i_4 * 8) + 1)])), __pack_half2(((bfloat16_t)C_local[((i_4 * 8) + 2)]), ((bfloat16_t)C_local[((i_4 * 8) + 3)])), __pack_half2(((bfloat16_t)C_local[((i_4 * 8) + 4)]), ((bfloat16_t)C_local[((i_4 * 8) + 5)])), __pack_half2(((bfloat16_t)C_local[((i_4 * 8) + 6)]), ((bfloat16_t)C_local[((i_4 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    cutlass::arch::NamedBarrier::sync(256, 0);
    if (tl::tl_shuffle_elect<256>()) {
      tl::tma_store<tl::CacheHintSm90::EVICT_NORMAL>(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 256));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store<tl::CacheHintSm90::EVICT_NORMAL>(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[16384])), ((((int)blockIdx.x) * 128) + 64), (((int)blockIdx.y) * 256));
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}


// #define ERROR_BUF_SIZE 1024
// static char error_buf[ERROR_BUF_SIZE];

// extern "C" const char* get_last_error() {
//     return error_buf;
// }

// extern "C" int init() {
//     error_buf[0] = '\0';
    
//     cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 180224);
//     if (result_main_kernel != CUDA_SUCCESS) {
//         snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 180224, cudaGetErrorString(result_main_kernel));
//         return -1;
//     }

//     return 0;
// }

// extern "C" int call(bfloat16_t* __restrict__ A, uint8_t* __restrict__ B, bfloat16_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {

// 	CUtensorMap A_desc;
// 	CUtensorMapDataType A_desc_type= (CUtensorMapDataType)9;
// 	cuuint32_t A_desc_tensorRank= 2;
// 	void *A_desc_globalAddress= A;
// 	cuuint64_t A_desc_globalDim[2]= {8192,16384};
// 	cuuint64_t A_desc_globalStride[2]= {2,16384};
// 	cuuint32_t A_desc_boxDim[2]= {64,256};
// 	cuuint32_t A_desc_elementStrides[2]= {1,1};
// 	CUtensorMapInterleave A_desc_interleave= (CUtensorMapInterleave)0;
// 	CUtensorMapSwizzle A_desc_swizzle= (CUtensorMapSwizzle)3;
// 	CUtensorMapL2promotion A_desc_l2Promotion= (CUtensorMapL2promotion)2;
// 	CUtensorMapFloatOOBfill A_desc_oobFill= (CUtensorMapFloatOOBfill)0;

// 	CUresult A_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);

// 	if (A_desc_result != CUDA_SUCCESS) {
// 		std::stringstream ss;
// 		ss << "Error: Failed to initialize the TMA descriptor A_desc";
// 		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
// 		return -1;
// 	}

// 	CUtensorMap B_desc;
// 	CUtensorMapDataType B_desc_type= (CUtensorMapDataType)0;
// 	cuuint32_t B_desc_tensorRank= 2;
// 	void *B_desc_globalAddress= B;
// 	cuuint64_t B_desc_globalDim[2]= {4096,8192};
// 	cuuint64_t B_desc_globalStride[2]= {1,4096};
// 	cuuint32_t B_desc_boxDim[2]= {64,128};
// 	cuuint32_t B_desc_elementStrides[2]= {1,1};
// 	CUtensorMapInterleave B_desc_interleave= (CUtensorMapInterleave)0;
// 	CUtensorMapSwizzle B_desc_swizzle= (CUtensorMapSwizzle)2;
// 	CUtensorMapL2promotion B_desc_l2Promotion= (CUtensorMapL2promotion)2;
// 	CUtensorMapFloatOOBfill B_desc_oobFill= (CUtensorMapFloatOOBfill)0;

// 	CUresult B_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, B_desc_globalStride + 1, B_desc_boxDim, B_desc_elementStrides, B_desc_interleave, B_desc_swizzle, B_desc_l2Promotion, B_desc_oobFill);

// 	if (B_desc_result != CUDA_SUCCESS) {
// 		std::stringstream ss;
// 		ss << "Error: Failed to initialize the TMA descriptor B_desc";
// 		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
// 		return -1;
// 	}

// 	CUtensorMap C_desc;
// 	CUtensorMapDataType C_desc_type= (CUtensorMapDataType)9;
// 	cuuint32_t C_desc_tensorRank= 2;
// 	void *C_desc_globalAddress= C;
// 	cuuint64_t C_desc_globalDim[2]= {8192,16384};
// 	cuuint64_t C_desc_globalStride[2]= {2,16384};
// 	cuuint32_t C_desc_boxDim[2]= {64,256};
// 	cuuint32_t C_desc_elementStrides[2]= {1,1};
// 	CUtensorMapInterleave C_desc_interleave= (CUtensorMapInterleave)0;
// 	CUtensorMapSwizzle C_desc_swizzle= (CUtensorMapSwizzle)3;
// 	CUtensorMapL2promotion C_desc_l2Promotion= (CUtensorMapL2promotion)2;
// 	CUtensorMapFloatOOBfill C_desc_oobFill= (CUtensorMapFloatOOBfill)0;

// 	CUresult C_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &C_desc, C_desc_type, C_desc_tensorRank, C_desc_globalAddress, C_desc_globalDim, C_desc_globalStride + 1, C_desc_boxDim, C_desc_elementStrides, C_desc_interleave, C_desc_swizzle, C_desc_l2Promotion, C_desc_oobFill);

// 	if (C_desc_result != CUDA_SUCCESS) {
// 		std::stringstream ss;
// 		ss << "Error: Failed to initialize the TMA descriptor C_desc";
// 		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
// 		return -1;
// 	}
// 	main_kernel<<<dim3(64, 64, 1), dim3(384, 1, 1), 180224, stream>>>(A_desc, B_desc, C_desc);
// 	TILELANG_CHECK_LAST_ERROR("main_kernel");

// 	return 0;
// }

