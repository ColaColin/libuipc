#include <cstdio>
#include <cuda_runtime.h>
#define CK(e, what) do { cudaError_t e_ = (e); printf("%-55s -> %s\n", what, cudaGetErrorString(e_)); } while(0)
__global__ void dummy_kernel(int* p) { if(p) *p = 1; }

int main() {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
    printf("cc=%d.%d l2=%d maxPersisting=%d maxWindow=%d\n\n", p.major, p.minor,
           p.l2CacheSize, p.persistingL2CacheMaxSize, p.accessPolicyMaxWindowSize);

    int* buf = nullptr; cudaMalloc(&buf, 4<<20);

    // 1) window on a fresh stream, REAL device pointer, both hitProps
    cudaStream_t s; cudaStreamCreate(&s);
    cudaStreamAttrValue v{};
    v.accessPolicyWindow = { buf, size_t(1<<20), 1.0f, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming };
    CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v), "stream window (persisting, 1MB, real ptr)");
    v.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v), "stream window (normal, 1MB, real ptr)");
    v.accessPolicyWindow.num_bytes = 4096;
    CK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v), "stream window (persisting, 4KB, real ptr)");

    // 2) kernel-node attribute on a CAPTURED graph node
    dummy_kernel<<<1,1>>> (buf);
    cudaStream_t cs; cudaStreamCreate(&cs);
    cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
    dummy_kernel<<<1,1,0,cs>>>(buf);
    cudaGraph_t g; cudaStreamEndCapture(cs, &g);
    cudaGraphNode_t nodes[4]; size_t nn = 0;
    cudaGraphGetNodes(g, nodes, &nn);
    printf("\ncaptured %zu node(s)\n", nn);
    cudaKernelNodeAttrValue kv{};
    kv.accessPolicyWindow = { buf, size_t(1<<20), 1.0f, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming };
    CK(cudaGraphKernelNodeSetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &kv), "graphKernelNodeSetAttribute(window)");
    cudaGraphExec_t ex; cudaGraphInstantiate(&ex, g, 0);
    // query it back to see state
    cudaKernelNodeAttrValue qv{};
    cudaError_t qe = cudaGraphKernelNodeGetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &qv);
    printf("%-55s -> %s (base=%p bytes=%zu)\n", "graphKernelNodeGetAttribute(window)", cudaGetErrorString(qe), qv.accessPolicyWindow.base_ptr, qe==cudaSuccess?qv.accessPolicyWindow.num_bytes:0);

    // 3) setLimit anyway
    CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size_t(1<<20)), "deviceSetLimit(persisting 1MB)");
    return 0;
}
