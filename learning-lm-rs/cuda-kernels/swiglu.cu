#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

using bf16 =  __nv_bfloat16;
using f16 = __half;

extern "C" __global__ void swiglu_f32(
    float* y,
    const float* x,
    int num_elements
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const float x_val = x[i];
    const float sigmoid = 1.0f / (1.0f + expf(-x_val));
    const float silu = x_val * sigmoid;
    
    y[i] *= silu;
}



extern "C" __global__ void swiglu_f16(
    f16* y,
    const f16* x,
    int num_elements
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const f16 x_val = x[i];
    const f16 sigmoid = 1.0f / (1.0f + expf(-x_val));
    const f16 silu = x_val * sigmoid;
    
    y[i] *= silu;
}
extern "C" __global__ void swiglu_bf16(
    bf16* y,
    const bf16* x,
    int num_elements
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    const bf16 x_val = x[i];
    const bf16 sigmoid = 1.0f / (1.0f + expf(-x_val));
    const bf16 silu = x_val * sigmoid;
    
    y[i] *= silu;
}