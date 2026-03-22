#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <functional>

// The loop body: load → rsqrtf (SFU, ~16 cycle latency) → 8 ALU ops
#define BODY(x, r, acc)              \
    (r) = rsqrtf((x));              \
    (acc) += (r);                   \
    (acc) *= 0.99f;                 \
    (acc) += (x) * 0.5f;           \
    (acc) -= (acc) * (acc) * 0.001f;\
    (acc) += __sinf((acc)) * 0.0f;  \
    (acc) *= (1.0f + (x) * 0.0001f);\
    (acc) += (r) * (x) * 0.01f;    \
    (acc) -= 0.0001f * (acc);

#define KERNEL(name, factor)                                          \
__global__ void name(const float* data, float* out, int n) {         \
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                 \
    const float* d = data + tid * n;                                 \
    float acc = 0.f;                                                 \
    _Pragma(#factor)                                                 \
    for (int i = 0; i < n; i++) {                                    \
        float x = d[i], r;                                           \
        BODY(x, r, acc);                                             \
    }                                                                \
    out[tid] = acc;                                                  \
}

KERNEL(kernel_unroll_1,  unroll 1)
KERNEL(kernel_unroll_2,  unroll 2)
KERNEL(kernel_unroll_4,  unroll 4)
KERNEL(kernel_unroll_8,  unroll 8)
KERNEL(kernel_unroll_16, unroll 16)

// ---------------------------------------------------------------------------

float time_kernel(std::function<void()> fn, int warmup = 20, int iters = 1000) {
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

bool verify(float* ref, float* test, int count, const char* name) {
    for (int i = 0; i < count; i++) {
        if (fabsf(ref[i] - test[i]) / (fabsf(test[i]) + 1e-6f) > 1e-2f) {
            printf("  %s: FAIL at [%d] ref=%.6f got=%.6f\n", name, i, ref[i], test[i]);
            return false;
        }
    }
    printf("  %s: PASS\n", name);
    return true;
}

int main() {
    const int threads = 256, blocks = 1024;
    const int total = threads * blocks;
    const int sizes[] = {64, 512};

    for (int N : sizes) {
        size_t data_bytes = (size_t)total * N * sizeof(float);
        size_t out_bytes  = (size_t)total * sizeof(float);
        printf("\n=== N=%d (%.0f MB) ===\n", N, data_bytes / (1024.0 * 1024.0));

        float* h_data = (float*)malloc(data_bytes);
        float* h_ref  = (float*)malloc(out_bytes);
        float* h_test = (float*)malloc(out_bytes);
        srand(42);
        for (size_t i = 0; i < (size_t)total * N; i++)
            h_data[i] = 0.1f + 9.9f * ((float)rand() / RAND_MAX);

        float *d_data, *d_out;
        cudaMalloc(&d_data, data_bytes);
        cudaMalloc(&d_out, out_bytes);
        cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);

        // Correctness: all kernels vs unroll_1
        kernel_unroll_1<<<blocks, threads>>>(d_data, d_out, N);
        cudaMemcpy(h_ref, d_out, out_bytes, cudaMemcpyDeviceToHost);

        auto check = [&](auto fn, const char* name) {
            fn<<<blocks, threads>>>(d_data, d_out, N);
            cudaMemcpy(h_test, d_out, out_bytes, cudaMemcpyDeviceToHost);
            verify(h_ref, h_test, total, name);
        };
        check(kernel_unroll_2,  "unroll_2");
        check(kernel_unroll_4,  "unroll_4");
        check(kernel_unroll_8,  "unroll_8");
        check(kernel_unroll_16, "unroll_16");

        // Benchmark
        printf("\n  %-12s %10s %8s\n", "Kernel", "Time(us)", "Speedup");

        auto bench = [&](auto fn, const char* name, float base) {
            float us = time_kernel([&]{ fn<<<blocks, threads>>>(d_data, d_out, N); }) * 1000.f;
            printf("  %-12s %10.2f %7.2fx\n", name, us, base > 0 ? base / us : 1.f);
            return us;
        };

        float base = bench(kernel_unroll_1,  "unroll_1",  0);
        bench(kernel_unroll_2,  "unroll_2",  base);
        bench(kernel_unroll_4,  "unroll_4",  base);
        bench(kernel_unroll_8,  "unroll_8",  base);
        bench(kernel_unroll_16, "unroll_16", base);

        cudaFree(d_data); cudaFree(d_out);
        free(h_data); free(h_ref); free(h_test);
    }
    return 0;
}
