#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

using bf16 =  __nv_bfloat16;
using f16 = __half;

extern "C" __global__ void matmul_transb_f32(
    float* C,
    float beta,
    float* A,
    float* B,
    float alpha,
    int M,
    int N,
    int P
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[col * N + k];
        }
        C[row * P + col] = beta * C[row * P + col] + alpha * sum;
    }
}


extern "C" __global__ void matmul_transb_f16(
    f16* C,
    f16 beta,
    f16* A,
    f16* B,
    f16 alpha,
    int M,
    int N,
    int P
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        f16 sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[col * N + k];
        }
        C[row * P + col] = beta * C[row * P + col] + alpha * sum;
    }
}

extern "C" __global__ void matmul_transb_bf16(
    bf16* C,
    bf16 beta,
    bf16* A,
    bf16* B,
    bf16 alpha,
    int M,
    int N,
    int P
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        bf16 sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[col * N + k];
        }
        C[row * P + col] = beta * C[row * P + col] + alpha * sum;
    }
}