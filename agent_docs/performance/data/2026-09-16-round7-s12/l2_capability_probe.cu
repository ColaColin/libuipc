#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
    printf("name=%s cc=%d.%d\n", p.name, p.major, p.minor);
    printf("l2CacheSize=%d bytes (%.2f MB)\n", p.l2CacheSize, p.l2CacheSize/1048576.0);
    printf("persistingL2CacheMaxSize=%d bytes (%.2f MB)\n", p.persistingL2CacheMaxSize, p.persistingL2CacheMaxSize/1048576.0);
    printf("accessPolicyMaxWindowSize=%d bytes (%.2f MB)\n", p.accessPolicyMaxWindowSize, p.accessPolicyMaxWindowSize/1048576.0);
    size_t cur = 0; cudaDeviceGetLimit(&cur, cudaLimitPersistingL2CacheSize);
    printf("current persisting setaside=%zu\n", cur);
    if (p.persistingL2CacheMaxSize > 0) {
        cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)p.persistingL2CacheMaxSize);
        printf("setLimit(max) -> %s\n", cudaGetErrorString(e));
        cudaDeviceGetLimit(&cur, cudaLimitPersistingL2CacheSize);
        printf("persisting setaside now=%zu\n", cur);
    }
    // can we set a window on a fresh stream?
    cudaStream_t s; cudaError_t e1 = cudaStreamCreate(&s);
    cudaStreamAttrValue v{}; v.accessPolicyWindow.base_ptr = (void*)0x1000;
    v.accessPolicyWindow.num_bytes = 1024*1024;
    v.accessPolicyWindow.hitRatio = 0.5f;
    v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaError_t e2 = cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v);
    printf("streamSetAttribute(window) -> %s\n", cudaGetErrorString(e2));
    return 0;
}
