#include <cstdio>
#include <cuda_runtime.h>
#define CK(e, what) do { cudaError_t e_ = (e); printf("%-58s -> %s\n", what, cudaGetErrorString(e_)); } while(0)
__global__ void dummy_kernel(int* p) { if(p) *p = 1; }
int main() {
    int* buf = nullptr; cudaMalloc(&buf, 4<<20);
    cudaStream_t cs; cudaStreamCreate(&cs);
    CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
    dummy_kernel<<<1,1,0,cs>>>(buf);
    cudaGraph_t g;
    CK(cudaStreamEndCapture(cs, &g), "end capture");
    cudaGraphNode_t nodes[4]; size_t nn = 0;
    cudaGraphGetNodes(g, nodes, &nn);
    printf("captured %zu node(s)\n", nn);
    if(nn > 0) {
        cudaKernelNodeAttrValue kv{};
        kv.accessPolicyWindow = { buf, size_t(1<<20), 1.0f, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming };
        CK(cudaGraphKernelNodeSetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &kv),
           "graphKernelNodeSetAttribute(window, 1MB)");
        kv.accessPolicyWindow.num_bytes = 4096;
        CK(cudaGraphKernelNodeSetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &kv),
           "graphKernelNodeSetAttribute(window, 4KB)");
    }
    // stream attr DURING capture (the mechanism this step would have used)
    cudaStream_t cs2; cudaStreamCreate(&cs2);
    CK(cudaStreamBeginCapture(cs2, cudaStreamCaptureModeThreadLocal), "begin capture #2");
    cudaStreamAttrValue v{};
    v.accessPolicyWindow = { buf, size_t(1<<20), 1.0f, cudaAccessPropertyPersisting, cudaAccessPropertyStreaming };
    CK(cudaStreamSetAttribute(cs2, cudaStreamAttributeAccessPolicyWindow, &v), "streamSetAttribute WHILE capturing");
    dummy_kernel<<<1,1,0,cs2>>>(buf);
    cudaGraph_t g2;
    CK(cudaStreamEndCapture(cs2, &g2), "end capture #2 (after failed attr set)");
    return 0;
}
