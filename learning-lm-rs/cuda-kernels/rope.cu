#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

using bf16 =  __nv_bfloat16;
using f16 = __half;

extern "C" __global__ void rope_f32(
    float* data,
    int start_pos,
    float theta,
    int seq_len,
    int n_heads,
    int d
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    int head = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (tok < seq_len && head < n_heads && i < d/2) {
        int pos = start_pos + tok;
        int base_idx = tok * n_heads * d + head * d;
        
        float a = data[base_idx + i];
        float b = data[base_idx + i + d/2];
        
        float freq = (float)pos / powf(theta, (2.0f * i) / (float)d);
        float sin_val, cos_val;
        sincosf(freq, &sin_val, &cos_val);
        
        data[base_idx + i] = a * cos_val - b * sin_val;
        data[base_idx + i + d/2] = b * cos_val + a * sin_val;
    }
}


extern "C" __global__ void rope_f16(
    f16* data,
    int start_pos,
    float theta,
    int seq_len,
    int n_heads,
    int d
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    int head = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (tok < seq_len && head < n_heads && i < d/2) {
        int pos = start_pos + tok;
        int base_idx = tok * n_heads * d + head * d;
        
        f16 a = data[base_idx + i];
        f16 b = data[base_idx + i + d/2];
        
        float freq = (float)pos / powf(theta, (2.0f * i) / (float)d);
        float sin_val, cos_val;
        sincosf(freq, &sin_val, &cos_val);
        
        data[base_idx + i] = a * f16(cos_val) - b * f16(sin_val);
        data[base_idx + i + d/2] = b * f16(cos_val) + a * f16(sin_val);
    }
}
extern "C" __global__ void rope_bf16(
    bf16* data,
    int start_pos,
    float theta,
    int seq_len,
    int n_heads,
    int d
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    int head = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (tok < seq_len && head < n_heads && i < d/2) {
        int pos = start_pos + tok;
        int base_idx = tok * n_heads * d + head * d;
        
        bf16 a = data[base_idx + i];
        bf16 b = data[base_idx + i + d/2];
        
        float freq = (float)pos / powf(theta, (2.0f * i) / (float)d);
        float sin_val, cos_val;
        sincosf(freq, &sin_val, &cos_val);
        
        data[base_idx + i] = a * bf16(cos_val) - b * bf16(sin_val);
        data[base_idx + i + d/2] = b * bf16(cos_val) + a * bf16(sin_val);
    }
}