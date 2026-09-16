#include <cstdio>
#include <cuda_runtime.h>
#define CK(e, what) do { cudaError_t e_ = (e); printf("%-58s -> %s\n", what, cudaGetErrorString(e_)); } while(0)
__global__ void dummy_kernel(int* p) { if(p) *p = 1; }
int main() {
    int* buf = nullptr; cudaMalloc(&buf, 4<<20);
    dummy_kernel<<<1,1>>>(buf);            // force module load + ctx init outside capture
    CK(cudaGetLastError(), "warmup launch");
    CK(cudaDeviceSynchronize(), "warmup sync");
    cudaStream_t cs; cudaStreamCreate(&cs);
    CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
    dummy_kernel<<<1,1,0,cs>>>(buf);
    CK(cudaGetLastError(), "launch inside capture");
    cudaGraph_t g;
    CK(cudaStreamEndCapture(cs, &g), "end capture");
    cudaGraphNode_t nodes[4]; size_t nn = 4;
    CK(cudaGraphGetNodes(g, nodes, &nn), "graphGetNodes");
    printf("captured %zu node(s)\n", nn);
    if(nn > 0) {
        cudaKernelNodeAttrValue kv{};
        kv.accessPolicyWindow = { buf, size_t(1<<20), 1.0f, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming };
        CK(cudaGraphKernelNodeSetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &kv),
           "graphKernelNodeSetAttribute(window, 1MB)");
        kv.accessPolicyWindow.num_bytes = 4096;
        CK(cudaGraphKernelNodeSetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &kv),
           "graphKernelNodeSetAttribute(window, 4KB)");
        cudaKernelNodeAttrValue qv{};
        CK(cudaGraphKernelNodeGetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &qv),
           "graphKernelNodeGetAttribute(window)");
    }
    return 0;
}
