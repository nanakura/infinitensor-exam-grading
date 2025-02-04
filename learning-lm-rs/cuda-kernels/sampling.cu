#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template<typename T>
__device__ T maximum(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
__device__ T exp_wrapper(T x) {
    return expf(x);
}

template<>
__device__ __half exp_wrapper(__half x) {
    return hexp(x);
}

template<>
__device__ __nv_bfloat16 exp_wrapper(__nv_bfloat16 x) {
    return hexp(x);
}

template <typename T>
__device__ void random_sample(
    T* logits,           
    unsigned int* result,   
    float top_p,            
    unsigned int top_k,
    float temperature,
    unsigned int seed,
    int size
) {
    __shared__ T max_val;
    __shared__ T sum_exp;
    __shared__ T probs[4096];
    
    T local_max = -INFINITY;
    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        local_max = maximum(local_max, logits[i]);
    }
    
    __shared__ T temp_max[256];
    temp_max[threadIdx.x] = local_max;
    __syncthreads();
    
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            temp_max[threadIdx.x] = maximum(temp_max[threadIdx.x], temp_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        max_val = temp_max[0];
        sum_exp = T(0.0f);
    }
    __syncthreads();

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        T val = exp_wrapper((logits[i] - max_val) / T(temperature));
        probs[i] = val;
        atomicAdd(&sum_exp, val);
    }
    __syncthreads();
    
    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        probs[i] /= sum_exp;
    }
    __syncthreads();
    
    if(threadIdx.x == 0) {
        int indices[4096];
        T sorted_probs[4096];
        for(int i = 0; i < size; i++) {
            indices[i] = i;
            sorted_probs[i] = probs[i];
        }
        
        int k = min((int)top_k, size);
        for(int i = 0; i < k; i++) {
            for(int j = i + 1; j < size; j++) {
                if(sorted_probs[j] > sorted_probs[i]) {
                    float temp_prob = sorted_probs[i];
                    sorted_probs[i] = sorted_probs[j];
                    sorted_probs[j] = temp_prob;
                    int temp_idx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp_idx;
                }
            }
        }
        
        T cumsum = T(0.0f);
        int last_idx = k - 1;
        if(top_p < 1.0f) {
            for(int i = 0; i < k; i++) {
                cumsum += sorted_probs[i];
                if(cumsum >= T(top_p)) {
                    last_idx = i;
                    break;
                }
            }
        }
        
        cumsum = T(0.0f);
        for(int i = 0; i <= last_idx; i++) {
            cumsum += sorted_probs[i];
        }
        for(int i = 0; i <= last_idx; i++) {
            sorted_probs[i] /= cumsum;
        }
        
        curandState state;
        curand_init(seed, 0, 0, &state);
        T rand_val = T(curand_uniform(&state));
        
        cumsum = T(0.0f);
        for(int i = 0; i <= last_idx; i++) {
            cumsum += sorted_probs[i];
            if(rand_val <= cumsum) {
                *result = indices[i];
                break;
            }
        }
        if(rand_val > cumsum) {
            *result = indices[last_idx];
        }
    }
}

#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void random_sample_##RUST_NAME( \
        TYPENAME* logits, \
        unsigned int* result, \
        float top_p, \
        unsigned int top_k, \
        float temperature, \
        unsigned int seed, \
        int size \
    ) { \
        random_sample<TYPENAME>(logits, result, top_p, top_k, temperature, seed, size); \
    } \

#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)