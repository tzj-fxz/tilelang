#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void kernel_kernel(float* __restrict__ G, __grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap U_desc, __grid_constant__ const CUtensorMap V_new_desc, __grid_constant__ const CUtensorMap W_desc, float* __restrict__ final_state, __grid_constant__ const CUtensorMap h_desc, __grid_constant__ const CUtensorMap initial_state_desc);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(float* __restrict__ G, __grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap U_desc, __grid_constant__ const CUtensorMap V_new_desc, __grid_constant__ const CUtensorMap W_desc, float* __restrict__ final_state, __grid_constant__ const CUtensorMap h_desc, __grid_constant__ const CUtensorMap initial_state_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float b_h_fragment[32];
  float V_new_fragment[16];
  float U_fragment[16];
  float G_last_local[1];
  float G_fragment[16];
  __shared__ uint64_t _mbarrier[13];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(initial_state_desc);
    tl::prefetch_tma_descriptor(W_desc);
    tl::prefetch_tma_descriptor(U_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(h_desc);
    tl::prefetch_tma_descriptor(V_new_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    // tl::mbarrier_init(_mbarrier[0], 1);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 128);
    tl::mbarrier_init(_mbarrier[5], 128);
    tl::mbarrier_init(_mbarrier[6], 128);
    tl::mbarrier_init(_mbarrier[7], 128);
    tl::mbarrier_init(_mbarrier[8], 128);
    tl::mbarrier_init(_mbarrier[9], 128);
    tl::mbarrier_init(_mbarrier[10], 128);
    tl::mbarrier_init(_mbarrier[11], 128);
    tl::mbarrier_init(_mbarrier[12], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[6], 8192);
      tl::tma_load(initial_state_desc, _mbarrier[6], (&(((bfloat16_t*)buf_dyn_shmem)[4096])), (((int)blockIdx.x) * 32), 0, ((int)blockIdx.y), 0);
    }
    tl::mbarrier_arrive(_mbarrier[6]);
    for (int i_s = 0; i_s < 512; ++i_s) {
      tl::mbarrier_wait(_mbarrier[3], ((i_s & 1) ^ 1));
      
      // (zhengju) use warp shuffling instructions to avoid branch in the threadIdx condition
      if (((int)threadIdx.x) == 128) {
      // if (__shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0) == 0 && cute::elect_one_sync()) {

        // (zhengju) free more threads from waiting on the barrier
        // if (threadIdx.x % 128 == 0) {
        //   tl::mbarrier_expect_tx(_mbarrier[0], 16384);
        // }
        tl::mbarrier_expect_tx(_mbarrier[0], 16384);
      
        tl::tma_load(W_desc, _mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[12288])), 0, ((int)blockIdx.y), (i_s * 64), 0);
        tl::tma_load(W_desc, _mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[16384])), 64, ((int)blockIdx.y), (i_s * 64), 0);
      }
      // if (threadIdx.x % 128 == 0) {
      //   tl::mbarrier_arrive(_mbarrier[0]);
      // }
      tl::mbarrier_arrive(_mbarrier[0]);
      tl::mbarrier_wait(_mbarrier[4], ((i_s & 1) ^ 1));
      
      // (zhengju) use warp shuffling instructions to avoid branch in the threadIdx condition
      if (((int)threadIdx.x) == 128) {
      // if (__shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0) == 0 && cute::elect_one_sync()) {
      
        tl::mbarrier_expect_tx(_mbarrier[1], 4096);
        tl::tma_load(U_desc, _mbarrier[1], (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 32), ((int)blockIdx.y), (i_s * 64), 0);
      }
      tl::mbarrier_arrive(_mbarrier[1]);
      // (zhengju) try to use try_wait code from yuqing
      // if (tl::mbarrier_try_wait(_mbarrier[5], ((i_s & 1) ^ 1)) == 0) {
      //   tl::mbarrier_wait(_mbarrier[5], ((i_s & 1) ^ 1));
      // }
      tl::mbarrier_wait(_mbarrier[5], ((i_s & 1) ^ 1));
      
      // (zhengju) use warp shuffling instructions to avoid branch in the threadIdx condition
      if (((int)threadIdx.x) == 128) {
      // if (__shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0) == 0 && cute::elect_one_sync()) {
      
        tl::mbarrier_expect_tx(_mbarrier[2], 16384);
        tl::tma_load(K_desc, _mbarrier[2], (&(((bfloat16_t*)buf_dyn_shmem)[20480])), 0, ((int)blockIdx.y), (i_s * 64), 0);
        tl::tma_load(K_desc, _mbarrier[2], (&(((bfloat16_t*)buf_dyn_shmem)[24576])), 64, ((int)blockIdx.y), (i_s * 64), 0);
      }
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        *(float4*)(((float*)buf_dyn_shmem) + ((((((i * 512) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 3584)) = make_float4(G[(((((i_s * 2048) + (i * 512)) + ((((int)threadIdx.x) >> 3) * 32)) + ((int)blockIdx.y)) - 512)], G[(((((i_s * 2048) + (i * 512)) + ((((int)threadIdx.x) >> 3) * 32)) + ((int)blockIdx.y)) - 512)], G[(((((i_s * 2048) + (i * 512)) + ((((int)threadIdx.x) >> 3) * 32)) + ((int)blockIdx.y)) - 512)], G[(((((i_s * 2048) + (i * 512)) + ((((int)threadIdx.x) >> 3) * 32)) + ((int)blockIdx.y)) - 512)]);
      }
      // tl::fence_proxy_async();
      tl::mbarrier_cp_async_arrive(_mbarrier[2]);
      tl::mbarrier_arrive(_mbarrier[2]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    tl::mbarrier_wait(_mbarrier[6], 0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      float2 __1;
      uint1 v_ = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((i_1 >> 3) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((i_1 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_1 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_1 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 4096));
      __1.x = (float)(((nv_bfloat162*)(&(v_.x)))->x);
      __1.y = (float)(((nv_bfloat162*)(&(v_.x)))->y);
      *(float2*)(b_h_fragment + (i_1 * 2)) = __1;
    }
    for (int i_s_1 = 0; i_s_1 < 512; ++i_s_1) {
      tl::syncthreads_partial(_mbarrier[7]);
      if (((int)threadIdx.x) == 0) {
        tl::tma_store(h_desc, (&(((bfloat16_t*)buf_dyn_shmem)[4096])), (((int)blockIdx.x) * 32), 0, ((int)blockIdx.y), i_s_1, 0);
        tl::tma_store_arrive();
        tl::tma_store_wait<0>();
      }
      // if (threadIdx.x % 128 == 0) {
      //   tl::mbarrier_wait(_mbarrier[0], (i_s_1 & 1));
      // }
      tl::mbarrier_wait(_mbarrier[0], (i_s_1 & 1));
      tl::gemm_ss<64, 32, 128, 4, 1, 0, 0, 1, true>((&(((bfloat16_t*)buf_dyn_shmem)[12288])), (&(((bfloat16_t*)buf_dyn_shmem)[4096])), (&(V_new_fragment[0])));
      tl::mbarrier_arrive(_mbarrier[3]);
      tl::mbarrier_wait(_mbarrier[1], (i_s_1 & 1));
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        
        // (zhengju) Use reinterpret_cast to avoid the performance loss, not too better than the original code
        uint1 v__1 = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_2 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_2 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_2 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        __nv_bfloat162 bf16_val = reinterpret_cast<__nv_bfloat162&>(v__1.x);
        float2 __2 = __bfloat1622float2(bf16_val);
        *reinterpret_cast<float2*>(U_fragment + (i_2 * 2)) = __2;
        // float2 __2;
        // uint1 v__1 = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_2 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_2 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_2 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        // __2.x = (float)(((nv_bfloat162*)(&(v__1.x)))->x);
        // __2.y = (float)(((nv_bfloat162*)(&(v__1.x)))->y);
        // *(float2*)(U_fragment + (i_2 * 2)) = __2;
      
      }
      
      // tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[4]);
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        V_new_fragment[i_3] = ((V_new_fragment[i_3] * -1.000000e+00f) + U_fragment[i_3]);
      }
      tl::syncthreads_partial(_mbarrier[8]);
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 2048)])), __pack_half2(((bfloat16_t)V_new_fragment[(i_4 * 8)]), ((bfloat16_t)V_new_fragment[((i_4 * 8) + 1)])), __pack_half2(((bfloat16_t)V_new_fragment[((i_4 * 8) + 2)]), ((bfloat16_t)V_new_fragment[((i_4 * 8) + 3)])), __pack_half2(((bfloat16_t)V_new_fragment[((i_4 * 8) + 4)]), ((bfloat16_t)V_new_fragment[((i_4 * 8) + 5)])), __pack_half2(((bfloat16_t)V_new_fragment[((i_4 * 8) + 6)]), ((bfloat16_t)V_new_fragment[((i_4 * 8) + 7)])));
      }
      tl::fence_proxy_async();
      tl::syncthreads_partial(_mbarrier[9]);
      // (zhengju) can move this store operation later to not barrier the following computation, but performance not better
      if (((int)threadIdx.x) == 0) {
        tl::tma_store(V_new_desc, (&(((bfloat16_t*)buf_dyn_shmem)[2048])), (((int)blockIdx.x) * 32), ((int)blockIdx.y), (i_s_1 * 64), 0);
        tl::tma_store_arrive();
        tl::tma_store_wait<0>();
      }
      G_last_local[0] = G[(((i_s_1 * 2048) + ((int)blockIdx.y)) + 2016)];
      // tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[2], (i_s_1 & 1));
      #pragma unroll
      for (int i_5 = 0; i_5 < 8; ++i_5) {
        *(float2*)(G_fragment + (i_5 * 2)) = *(float2*)(((float*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) >> 5) * 512) + ((i_5 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_5 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 4096));
        // float2 __3;
        // uint1 v__2 = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) >> 5) * 512) + ((i_5 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_5 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 4096));
        // __3.x = (float)(((nv_bfloat162*)(&(v__2.x)))->x);
        // __3.y = (float)(((nv_bfloat162*)(&(v__2.x)))->y);
        // *(float2*)(G_fragment + (i_5 * 2)) = __3;
      }
      // (zhengju) the V_new_desc need to be stored here the furthest, because the following memory operations are to update the smem of V_new_desc
      // The performance is not better
      // if (((int)threadIdx.x) == 0) {
      //   tl::tma_store(V_new_desc, ((&((bfloat16_t*)buf_dyn_shmem)[2048])), (((int)blockIdx.x) * 32), ((int)blockIdx.y), (i_s_1 * 64), 0);
      //   tl::tma_store_arrive();
      //   tl::tma_store_wait<0>();
      // }
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        if ((G_last_local[0] - G_fragment[i_6]) <= 0.000000e+00f) {
          V_new_fragment[i_6] = (V_new_fragment[i_6] * __expf((G_last_local[0] - G_fragment[i_6])));
        } else {
          V_new_fragment[i_6] = 0.000000e+00f;
        }
      }
      G_last_local[0] = __expf(G_last_local[0]);
      #pragma unroll
      for (int i_7 = 0; i_7 < 32; ++i_7) {
        b_h_fragment[i_7] = (b_h_fragment[i_7] * G_last_local[0]);
      }
      tl::syncthreads_partial(_mbarrier[10]);
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_8) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 2048)])), __pack_half2(((bfloat16_t)V_new_fragment[(i_8 * 8)]), ((bfloat16_t)V_new_fragment[((i_8 * 8) + 1)])), __pack_half2(((bfloat16_t)V_new_fragment[((i_8 * 8) + 2)]), ((bfloat16_t)V_new_fragment[((i_8 * 8) + 3)])), __pack_half2(((bfloat16_t)V_new_fragment[((i_8 * 8) + 4)]), ((bfloat16_t)V_new_fragment[((i_8 * 8) + 5)])), __pack_half2(((bfloat16_t)V_new_fragment[((i_8 * 8) + 6)]), ((bfloat16_t)V_new_fragment[((i_8 * 8) + 7)])));
      }
      tl::fence_proxy_async();
      tl::syncthreads_partial(_mbarrier[11]);
      tl::gemm_ss<128, 32, 64, 4, 1, 1, 0, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[20480])), (&(((bfloat16_t*)buf_dyn_shmem)[2048])), (&(b_h_fragment[0])));
      tl::mbarrier_arrive(_mbarrier[5]);
      tl::syncthreads_partial(_mbarrier[12]);
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_9 >> 1) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), __pack_half2(((bfloat16_t)b_h_fragment[(i_9 * 8)]), ((bfloat16_t)b_h_fragment[((i_9 * 8) + 1)])), __pack_half2(((bfloat16_t)b_h_fragment[((i_9 * 8) + 2)]), ((bfloat16_t)b_h_fragment[((i_9 * 8) + 3)])), __pack_half2(((bfloat16_t)b_h_fragment[((i_9 * 8) + 4)]), ((bfloat16_t)b_h_fragment[((i_9 * 8) + 5)])), __pack_half2(((bfloat16_t)b_h_fragment[((i_9 * 8) + 6)]), ((bfloat16_t)b_h_fragment[((i_9 * 8) + 7)])));
      }
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      *(float2*)(final_state + ((((((((((int)blockIdx.y) * 16384) + ((i_10 >> 3) * 8192)) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_10 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + (((int)blockIdx.x) * 32)) + (((i_10 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(b_h_fragment + (i_10 * 2));
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
    
//     cudaError_t result_kernel_kernel = cudaFuncSetAttribute(kernel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 57344);
//     if (result_kernel_kernel != CUDA_SUCCESS) {
//         snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 57344, cudaGetErrorString(result_kernel_kernel));
//         return -1;
//     }

//     return 0;
// }

// extern "C" int call(bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ W, bfloat16_t* __restrict__ U, float* __restrict__ G, bfloat16_t* __restrict__ initial_state, bfloat16_t* __restrict__ h, float* __restrict__ final_state, bfloat16_t* __restrict__ V_new, cudaStream_t stream=cudaStreamDefault) {

//         CUtensorMap K_desc;
//         CUtensorMapDataType K_desc_type= (CUtensorMapDataType)9;
//         cuuint32_t K_desc_tensorRank= 4;
//         void *K_desc_globalAddress= K;
//         cuuint64_t K_desc_globalDim[4]= {128,32,32768,1};
//         cuuint64_t K_desc_globalStride[4]= {2,256,8192,268435456};
//         cuuint32_t K_desc_boxDim[4]= {64,1,64,1};
//         cuuint32_t K_desc_elementStrides[4]= {1,1,1,1};
//         CUtensorMapInterleave K_desc_interleave= (CUtensorMapInterleave)0;
//         CUtensorMapSwizzle K_desc_swizzle= (CUtensorMapSwizzle)3;
//         CUtensorMapL2promotion K_desc_l2Promotion= (CUtensorMapL2promotion)2;
//         CUtensorMapFloatOOBfill K_desc_oobFill= (CUtensorMapFloatOOBfill)0;

//         CUresult K_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &K_desc, K_desc_type, K_desc_tensorRank, K_desc_globalAddress, K_desc_globalDim, K_desc_globalStride + 1, K_desc_boxDim, K_desc_elementStrides, K_desc_interleave, K_desc_swizzle, K_desc_l2Promotion, K_desc_oobFill);

//         if (K_desc_result != CUDA_SUCCESS) {
//                 std::stringstream ss;
//                 ss << "Error: Failed to initialize the TMA descriptor K_desc";
//                 snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
//                 return -1;
//         }

//         CUtensorMap U_desc;
//         CUtensorMapDataType U_desc_type= (CUtensorMapDataType)9;
//         cuuint32_t U_desc_tensorRank= 4;
//         void *U_desc_globalAddress= U;
//         cuuint64_t U_desc_globalDim[4]= {128,32,32768,1};
//         cuuint64_t U_desc_globalStride[4]= {2,256,8192,268435456};
//         cuuint32_t U_desc_boxDim[4]= {32,1,64,1};
//         cuuint32_t U_desc_elementStrides[4]= {1,1,1,1};
//         CUtensorMapInterleave U_desc_interleave= (CUtensorMapInterleave)0;
//         CUtensorMapSwizzle U_desc_swizzle= (CUtensorMapSwizzle)2;
//         CUtensorMapL2promotion U_desc_l2Promotion= (CUtensorMapL2promotion)2;
//         CUtensorMapFloatOOBfill U_desc_oobFill= (CUtensorMapFloatOOBfill)0;

//         CUresult U_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &U_desc, U_desc_type, U_desc_tensorRank, U_desc_globalAddress, U_desc_globalDim, U_desc_globalStride + 1, U_desc_boxDim, U_desc_elementStrides, U_desc_interleave, U_desc_swizzle, U_desc_l2Promotion, U_desc_oobFill);

//         if (U_desc_result != CUDA_SUCCESS) {
//                 std::stringstream ss;
//                 ss << "Error: Failed to initialize the TMA descriptor U_desc";
//                 snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
//                 return -1;
//         }

//         CUtensorMap V_new_desc;
//         CUtensorMapDataType V_new_desc_type= (CUtensorMapDataType)9;
//         cuuint32_t V_new_desc_tensorRank= 4;
//         void *V_new_desc_globalAddress= V_new;
//         cuuint64_t V_new_desc_globalDim[4]= {128,32,32768,1};
//         cuuint64_t V_new_desc_globalStride[4]= {2,256,8192,268435456};
//         cuuint32_t V_new_desc_boxDim[4]= {32,1,64,1};
//         cuuint32_t V_new_desc_elementStrides[4]= {1,1,1,1};
//         CUtensorMapInterleave V_new_desc_interleave= (CUtensorMapInterleave)0;
//         CUtensorMapSwizzle V_new_desc_swizzle= (CUtensorMapSwizzle)2;
//         CUtensorMapL2promotion V_new_desc_l2Promotion= (CUtensorMapL2promotion)2;
//         CUtensorMapFloatOOBfill V_new_desc_oobFill= (CUtensorMapFloatOOBfill)0;

//         CUresult V_new_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &V_new_desc, V_new_desc_type, V_new_desc_tensorRank, V_new_desc_globalAddress, V_new_desc_globalDim, V_new_desc_globalStride + 1, V_new_desc_boxDim, V_new_desc_elementStrides, V_new_desc_interleave, V_new_desc_swizzle, V_new_desc_l2Promotion, V_new_desc_oobFill);

//         if (V_new_desc_result != CUDA_SUCCESS) {
//                 std::stringstream ss;
//                 ss << "Error: Failed to initialize the TMA descriptor V_new_desc";
//                 snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
//                 return -1;
//         }

//         CUtensorMap W_desc;
//         CUtensorMapDataType W_desc_type= (CUtensorMapDataType)9;
//         cuuint32_t W_desc_tensorRank= 4;
//         void *W_desc_globalAddress= W;
//         cuuint64_t W_desc_globalDim[4]= {128,32,32768,1};
//         cuuint64_t W_desc_globalStride[4]= {2,256,8192,268435456};
//         cuuint32_t W_desc_boxDim[4]= {64,1,64,1};
//         cuuint32_t W_desc_elementStrides[4]= {1,1,1,1};
//         CUtensorMapInterleave W_desc_interleave= (CUtensorMapInterleave)0;
//         CUtensorMapSwizzle W_desc_swizzle= (CUtensorMapSwizzle)3;
//         CUtensorMapL2promotion W_desc_l2Promotion= (CUtensorMapL2promotion)2;
//         CUtensorMapFloatOOBfill W_desc_oobFill= (CUtensorMapFloatOOBfill)0;

//         CUresult W_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &W_desc, W_desc_type, W_desc_tensorRank, W_desc_globalAddress, W_desc_globalDim, W_desc_globalStride + 1, W_desc_boxDim, W_desc_elementStrides, W_desc_interleave, W_desc_swizzle, W_desc_l2Promotion, W_desc_oobFill);

//         if (W_desc_result != CUDA_SUCCESS) {
//                 std::stringstream ss;
//                 ss << "Error: Failed to initialize the TMA descriptor W_desc";
//                 snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
//                 return -1;
//         }

//         CUtensorMap h_desc;
//         CUtensorMapDataType h_desc_type= (CUtensorMapDataType)9;
//         cuuint32_t h_desc_tensorRank= 5;
//         void *h_desc_globalAddress= h;
//         cuuint64_t h_desc_globalDim[5]= {128,128,32,512,1};
//         cuuint64_t h_desc_globalStride[5]= {2,256,32768,1048576,536870912};
//         cuuint32_t h_desc_boxDim[5]= {32,128,1,1,1};
//         cuuint32_t h_desc_elementStrides[5]= {1,1,1,1,1};
//         CUtensorMapInterleave h_desc_interleave= (CUtensorMapInterleave)0;
//         CUtensorMapSwizzle h_desc_swizzle= (CUtensorMapSwizzle)2;
//         CUtensorMapL2promotion h_desc_l2Promotion= (CUtensorMapL2promotion)2;
//         CUtensorMapFloatOOBfill h_desc_oobFill= (CUtensorMapFloatOOBfill)0;

//         CUresult h_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &h_desc, h_desc_type, h_desc_tensorRank, h_desc_globalAddress, h_desc_globalDim, h_desc_globalStride + 1, h_desc_boxDim, h_desc_elementStrides, h_desc_interleave, h_desc_swizzle, h_desc_l2Promotion, h_desc_oobFill);

//         if (h_desc_result != CUDA_SUCCESS) {
//                 std::stringstream ss;
//                 ss << "Error: Failed to initialize the TMA descriptor h_desc";
//                 snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
//                 return -1;
//         }

//         CUtensorMap initial_state_desc;
//         CUtensorMapDataType initial_state_desc_type= (CUtensorMapDataType)9;
//         cuuint32_t initial_state_desc_tensorRank= 4;
//         void *initial_state_desc_globalAddress= initial_state;
//         cuuint64_t initial_state_desc_globalDim[4]= {128,128,32,1};
//         cuuint64_t initial_state_desc_globalStride[4]= {2,256,32768,1048576};
//         cuuint32_t initial_state_desc_boxDim[4]= {32,128,1,1};
//         cuuint32_t initial_state_desc_elementStrides[4]= {1,1,1,1};
//         CUtensorMapInterleave initial_state_desc_interleave= (CUtensorMapInterleave)0;
//         CUtensorMapSwizzle initial_state_desc_swizzle= (CUtensorMapSwizzle)2;
//         CUtensorMapL2promotion initial_state_desc_l2Promotion= (CUtensorMapL2promotion)2;
//         CUtensorMapFloatOOBfill initial_state_desc_oobFill= (CUtensorMapFloatOOBfill)0;

//         CUresult initial_state_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
//     &initial_state_desc, initial_state_desc_type, initial_state_desc_tensorRank, initial_state_desc_globalAddress, initial_state_desc_globalDim, initial_state_desc_globalStride + 1, initial_state_desc_boxDim, initial_state_desc_elementStrides, initial_state_desc_interleave, initial_state_desc_swizzle, initial_state_desc_l2Promotion, initial_state_desc_oobFill);

//         if (initial_state_desc_result != CUDA_SUCCESS) {
//                 std::stringstream ss;
//                 ss << "Error: Failed to initialize the TMA descriptor initial_state_desc";
//                 snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
//                 return -1;
//         }
//         kernel_kernel<<<dim3(4, 32, 1), dim3(256, 1, 1), 57344, stream>>>(G, K_desc, U_desc, V_new_desc, W_desc, final_state, h_desc, initial_state_desc);
//         TILELANG_CHECK_LAST_ERROR("kernel_kernel");

//         return 0;
// }