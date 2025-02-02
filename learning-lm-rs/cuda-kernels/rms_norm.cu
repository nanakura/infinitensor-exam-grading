#include "cuda_fp16.h"
#include "cuda_bf16.h"

using bf16 = __nv_bfloat16;
using f16 = __half;

extern "C" {
	__global__ void compute_rms_bf16(bf16* rms, const bf16* x, int M, int N, bf16 epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    bf16 sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += x[i * N + j] * x[i * N + j];
    }
    bf16 mean = sum / (bf16)N;
    rms[i] = sqrtf(mean + epsilon);
	}

	__global__ void apply_rms_norm_bf16(bf16* y, const bf16* x, const bf16* w, const bf16* rms, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int i = idx / N;
    int j = idx % N;
    y[idx] = w[j] * (x[idx] / rms[i]);
	}

	__global__ void compute_rms_f16(f16* rms, const f16* x, int M, int N, f16 epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    f16 sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += x[i * N + j] * x[i * N + j];
    }
    f16 mean = sum / (f16)N;
    rms[i] = sqrtf(mean + epsilon);
	}

	__global__ void apply_rms_norm_f16(f16* y, const f16* x, const f16* w, const f16* rms, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int i = idx / N;
    int j = idx % N;
    y[idx] = w[j] * (x[idx] / rms[i]);
	}
	__global__ void compute_rms_f32(float* rms, const float* x, int M, int N, float epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += x[i * N + j] * x[i * N + j];
    }
    float mean = sum / (float)N;
    rms[i] = sqrtf(mean + epsilon);
	}

	__global__ void apply_rms_norm_f32(float* y, const float* x, const float* w, const float* rms, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int i = idx / N;
    int j = idx % N;
    y[idx] = w[j] * (x[idx] / rms[i]);
	}
}
