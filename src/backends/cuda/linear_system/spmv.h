#pragma once
#include <type_define.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
// calculate y = a * A * x + b * y
class Spmv
{
  public:
    // symmetric bcoo spmv
    void sym_spmv(Float                                a,
                  cuda_tool::CBCOOMatrixView<Float, 3> A,
                  cuda_tool::CDenseVectorView<Float>   x,
                  Float                                b,
                  cuda_tool::DenseVectorView<Float>    y);

    // reduce by key spmv
    void rbk_spmv(Float                                a,
                  cuda_tool::CBCOOMatrixView<Float, 3> A,
                  cuda_tool::CDenseVectorView<Float>   x,
                  Float                                b,
                  cuda_tool::DenseVectorView<Float>    y);

    // reduce by key symmtric spmv
    void rbk_sym_spmv(Float                                a,
                      cuda_tool::CBCOOMatrixView<Float, 3> A,
                      cuda_tool::CDenseVectorView<Float>   x,
                      Float                                b,
                      cuda_tool::DenseVectorView<Float>    y);

    // reduce by key symmetric spmv with fused dot product
    // computes y = a * A * x  AND  d_dot = x^T * (a * A * x) in a single pass
    // `stream` defaults to the legacy default stream; pass a capture stream
    // when recording a CUDA graph.
    // The triplet count is read on device (`d_triplet_count`) and the grid is
    // sized by `triplet_capacity`, so a captured graph stays valid while the
    // matrix nnz varies within the reserved capacity.
    void rbk_sym_spmv_dot(Float                                a,
                          cuda_tool::CBCOOMatrixView<Float, 3> A,
                          cuda_tool::CDenseVectorView<Float>   x,
                          Float                                b,
                          cuda_tool::DenseVectorView<Float>    y,
                          cuda_tool::VarView<Float>            d_dot,
                          cuda_tool::CDense<IndexT>            d_triplet_count,
                          SizeT                                triplet_capacity,
                          cudaStream_t stream = nullptr);

    // perf/round4 (s07) UIPC_SPMV_VERIFY=1: read back and log the max
    // differences between the chunked and the per-triplet kernel accumulated
    // since the previous call; allocates the scratch on the first call. Call
    // outside any graph capture, before the solve (once per assembly).
    void verify_report(SizeT dof_count);

    // debug fallback cpu spmv
    // very slow, only for debug
    void cpu_sym_spmv(Float                                a,
                      cuda_tool::CBCOOMatrixView<Float, 3> A,
                      cuda_tool::CDenseVectorView<Float>   x,
                      Float                                b,
                      cuda_tool::DenseVectorView<Float>    y);

  private:
    // UIPC_SPMV_VERIFY scratch: reference y / dot and the max-diff accumulators
    cuda_tool::DeviceVector<Float>              m_verify_y;
    cuda_tool::DeviceVar<Float>                 m_verify_dot;
    cuda_tool::DeviceVector<unsigned long long> m_verify_acc;
    unsigned long long                          m_verify_launches    = 0;
    double                                      m_verify_max_rel_y   = 0;
    double                                      m_verify_max_rel_dot = 0;
};
}  // namespace uipc::backend::cuda
