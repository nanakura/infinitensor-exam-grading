#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

using bf16 =  __nv_bfloat16;
using f16 = __half;

extern "C" __global__ void rms_norm_f32(
    float* y,
    const float* x,
    const float* w,
    float epsilon,
    int m,
    int n
) {
    extern __shared__ float shared_f32[];

    int i = blockIdx.x;
    if (i >= m) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // 计算平方和的部分和
    float sum_val = 0.0f;
    for (int j = tid; j < n; j += stride) {
        float val = x[i * n + j];
        sum_val += val * val;
    }

    // 将部分和存入共享内存
    shared_f32[tid] = sum_val;
    __syncthreads();

    // 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_f32[tid] += shared_f32[tid + s];
        }
        __syncthreads();
    }

    // 计算RMS
    float sum_squares = shared_f32[0];
    float rms = sqrtf(sum_squares / n + epsilon);

    // 归一化并应用权重
    for (int j = tid; j < n; j += stride) {
        float normalized = x[i * n + j] / rms;
        y[i * n + j] = normalized * w[j];
    }
}


extern "C" __global__ void rms_norm_f16(
    f16* y,
    const f16* x,
    const f16* w,
    f16 epsilon,
    int m,
    int n
) {
    extern __shared__ f16 shared_f16[];

    int i = blockIdx.x;
    if (i >= m) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // 计算平方和的部分和
    f16 sum_val = 0.0f;
    for (int j = tid; j < n; j += stride) {
        f16 val = x[i * n + j];
        sum_val += val * val;
    }

    // 将部分和存入共享内存
    shared_f16[tid] = sum_val;
    __syncthreads();

    // 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_f16[tid] += shared_f16[tid + s];
        }
        __syncthreads();
    }

    // 计算RMS
    f16 sum_squares = shared_f16[0];
    f16 rms = sqrtf(sum_squares / f16(n) + epsilon);

    // 归一化并应用权重
    for (int j = tid; j < n; j += stride) {
        f16 normalized = x[i * n + j] / rms;
        y[i * n + j] = normalized * w[j];
    }
}
extern "C" __global__ void rms_norm_fb16(
    bf16* y,
    const bf16* x,
    const bf16* w,
    bf16 epsilon,
    int m,
    int n
) {
    extern __shared__ bf16 shared_bf16[];

    int i = blockIdx.x;
    if (i >= m) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // 计算平方和的部分和
    bf16 sum_val = 0.0f;
    for (int j = tid; j < n; j += stride) {
        bf16 val = x[i * n + j];
        sum_val += val * val;
    }

    // 将部分和存入共享内存
    shared_bf16[tid] = sum_val;
    __syncthreads();

    // 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_bf16[tid] += shared_bf16[tid + s];
        }
        __syncthreads();
    }

    // 计算RMS
    bf16 sum_squares = shared_bf16[0];
    bf16 rms = sqrtf(sum_squares / bf16(n) + epsilon);

    // 归一化并应用权重
    for (int j = tid; j < n; j += stride) {
        bf16 normalized = x[i * n + j] / rms;
        y[i * n + j] = normalized * w[j];
    }
}