// s08 runtime micro-benchmark: the blocked 4x3 projection helper, old
// (SymAsm=0) vs new (SymAsm=2), identical harness, same inputs on device.
// Data-independent except the QL iteration count, which sees the same
// matrices in both arms.
#include <utils/make_spd.h>
#include <cuda_tool/cuda_tool.h>
#include <random>
#include <vector>
#include <cstdio>

using F = uipc::Float;

namespace uipc::backend::cuda
{
template <int SymAsm>
__global__ void bench_kernel(const F* in, F* out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    Matrix12x12 H;
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            H(i, j) = in[((size_t)I) * 144 + i * 12 + j];
    make_spd_translation_free_4x3_blocked<1, SymAsm>(H);
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            out[((size_t)I) * 144 + i * 12 + j] = H(i, j);
}

}  // namespace uipc::backend::cuda

template <typename Launch>
static void time4(const F* d_in, F* d_out, int n, int reps, const char* tag, Launch launch)
{
    // warmup
    launch(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int r = 0; r < reps; ++r)
        launch(d_in, d_out, n);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    printf("%s: %.3f ms / %d reps = %.2f us per launch of %d mats = %.1f ns/mat\n",
           tag, ms, reps, ms * 1000.f / reps, n, ms * 1e6f / reps / n);
}

int main(int argc, char** argv)
{
    const int  n    = argc > 1 ? atoi(argv[1]) : 32768;
    const int  reps = argc > 2 ? atoi(argv[2]) : 200;
    std::mt19937_64 rng(99);
    std::normal_distribution<double> G(0.0, 1.0);

    std::vector<F> h4((size_t)n * 144);
    for(size_t i = 0; i < h4.size(); ++i)
        h4[i] = G(rng);
    F* d_in4;
    F* d_out4;
    cudaMalloc(&d_in4, h4.size() * 8);
    cudaMalloc(&d_out4, h4.size() * 8);
    cudaMemcpy(d_in4, h4.data(), h4.size() * 8, cudaMemcpyHostToDevice);

    return 0;
}
    return 0;
}
