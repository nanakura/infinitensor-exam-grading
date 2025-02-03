#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

using bf16 =  __nv_bfloat16;
using f16 = __half;

extern "C" __global__ void gather_f32(
    float* output,
    const unsigned int* indices,
    const float* table,
    int length,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < length && dim_idx < dim) {
        unsigned int table_idx = indices[idx];
        output[idx * dim + dim_idx] = table[table_idx * dim + dim_idx];
    }
}


extern "C" __global__ void gather_f16(
    f16* output,
    const unsigned int* indices,
    const f16* table,
    int length,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < length && dim_idx < dim) {
        unsigned int table_idx = indices[idx];
        output[idx * dim + dim_idx] = table[table_idx * dim + dim_idx];
    }
}

extern "C" __global__ void gather_bf16(
    bf16* output,
    const unsigned int* indices,
    const bf16* table,
    int length,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < length && dim_idx < dim) {
        unsigned int table_idx = indices[idx];
        output[idx * dim + dim_idx] = table[table_idx * dim + dim_idx];
    }
}