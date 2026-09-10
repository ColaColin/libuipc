#pragma once
#include <sim_system.h>
#include <energy_component_flags.h>
#include <global_geometry/global_vertex_manager.h>
#include <cuda_tool/cuda_tool.h>
#include <utils/offset_count_collection.h>
#include <algorithm/matrix_converter.h>
#include <dytopo_effect_system/dytopo_classify_info.h>

namespace uipc::backend::cuda
{
class DyTopoEffectReporter;
class DyTopoEffectReceiver;

class GlobalDyTopoEffectManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class GradientHessianExtentInfo
    {
      public:
        bool gradient_only() const { return m_gradient_only; }
        void gradient_count(SizeT count) noexcept { m_gradient_count = count; }
        void hessian_count(SizeT count) noexcept { m_hessian_count = count; }

      private:
        friend class Impl;
        friend class DyTopoEffectReporter;

        bool  m_gradient_only  = false;
        SizeT m_gradient_count = 0;
        SizeT m_hessian_count  = 0;
    };

    class GradientHessianInfo
    {
      public:
        bool gradient_only() const noexcept { return m_gradient_only; }
        cuda_tool::DoubletVectorView<Float, 3> gradients() const noexcept
        {
            return m_gradients;
        }
        cuda_tool::TripletMatrixView<Float, 3> hessians() const noexcept
        {
            return m_hessians;
        }


      private:
        friend class Impl;
        bool                                   m_gradient_only = false;
        cuda_tool::DoubletVectorView<Float, 3> m_gradients;
        cuda_tool::TripletMatrixView<Float, 3> m_hessians;
    };

    class EnergyExtentInfo
    {
      public:
        void energy_count(SizeT count) noexcept { m_energy_count = count; }

      private:
        friend class Impl;
        friend class DyTopoEffectLineSearchReporter;
        SizeT m_energy_count = 0;
    };

    class EnergyInfo
    {
      public:
        cuda_tool::BufferView<Float> energies() const { return m_energies; }
        bool                         is_initial() const { return m_is_initial; }

      private:
        friend class DyTopoEffectLineSearchReporter;
        cuda_tool::BufferView<Float> m_energies;
        bool                         m_is_initial = false;
    };

    using ClassifyInfo = DyTopoClassifyInfo;

    class ClassifiedDyTopoEffectInfo
    {
      public:
        cuda_tool::CDoubletVectorView<Float, 3> gradients() const noexcept
        {
            return m_gradients;
        }
        cuda_tool::CTripletMatrixView<Float, 3> hessians() const noexcept
        {
            return m_hessians;
        }

      private:
        friend class Impl;
        cuda_tool::CDoubletVectorView<Float, 3> m_gradients;
        cuda_tool::CTripletMatrixView<Float, 3> m_hessians;
    };

    class ComputeDyTopoEffectInfo
    {
      public:
        void gradient_only(bool v) noexcept { m_gradient_only = v; }
        void component_flags(EnergyComponentFlags v) noexcept
        {
            m_component_flags = v;
        }

      private:
        friend class Impl;
        bool                 m_gradient_only   = false;
        EnergyComponentFlags m_component_flags = EnergyComponentFlags::All;
    };

    class Impl
    {
      public:
        void init(WorldVisitor& world);
        void compute_dytopo_effect(ComputeDyTopoEffectInfo& info);
        void _assemble(ComputeDyTopoEffectInfo& info);
        void _convert_matrix();
        void _distribute(ComputeDyTopoEffectInfo& info);
        // perf/kernels (2026-09-10): range-select every receiver's entries straight
        // from the unsorted collected arrays (no radix sort / reduce here: the
        // GlobalLinearSystem converter sorts and reduces the raw entries anyway)
        // with a single host readback of all counts.
        void _distribute_unsorted(ComputeDyTopoEffectInfo& info);
        bool use_sorted_path = false;  // env UIPC_DYTOPO_SORTED_PATH=1 restores the old path

        SimSystemSlot<GlobalVertexManager> global_vertex_manager;

        Float reserve_ratio = 1.1;


        /***********************************************************************
        *                              Reporter                                *
        ***********************************************************************/

        SimSystemSlotCollection<DyTopoEffectReporter> dytopo_effect_reporters;
        SimSystemSlotCollection<DyTopoEffectReporter> contact_reporters;
        SimSystemSlotCollection<DyTopoEffectReporter> non_contact_reporters;

        OffsetCountCollection<IndexT> reporter_energy_offsets_counts;
        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        cuda_tool::DeviceTripletMatrix<Float, 3> collected_dytopo_effect_hessian;
        cuda_tool::DeviceDoubletVector<Float, 3> collected_dytopo_effect_gradient;

        MatrixConverter<Float, 3>             matrix_converter;
        cuda_tool::DeviceBCOOMatrix<Float, 3> sorted_dytopo_effect_hessian;
        cuda_tool::DeviceBCOOVector<Float, 3> sorted_dytopo_effect_gradient;

        /***********************************************************************
        *                               Receiver                               *
        ***********************************************************************/

        SimSystemSlotCollection<DyTopoEffectReceiver> dytopo_effect_receivers;

        cuda_tool::DeviceVar<Vector2i>  gradient_range;
        cuda_tool::DeviceBuffer<IndexT> selected_hessian;
        cuda_tool::DeviceBuffer<IndexT> selected_hessian_offsets;

        vector<cuda_tool::DeviceTripletMatrix<Float, 3>> classified_dytopo_effect_hessians;
        vector<cuda_tool::DeviceDoubletVector<Float, 3>> classified_dytopo_effect_gradients;

        // unsorted distribute: per-receiver selection flags / exclusive offsets
        // (size N+1, the last offset is the count) and one gathered count array
        vector<cuda_tool::DeviceBuffer<IndexT>> sel_gradient;
        vector<cuda_tool::DeviceBuffer<IndexT>> sel_gradient_offsets;
        vector<cuda_tool::DeviceBuffer<IndexT>> sel_hessian;
        vector<cuda_tool::DeviceBuffer<IndexT>> sel_hessian_offsets;
        cuda_tool::DeviceBuffer<IndexT>         sel_counts;
        vector<IndexT>                          h_sel_counts;

        void loose_resize_entries(cuda_tool::DeviceTripletMatrix<Float, 3>& m, SizeT size);
        void loose_resize_entries(cuda_tool::DeviceDoubletVector<Float, 3>& v, SizeT size);
        template <typename T>
        void loose_resize(cuda_tool::DeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve_discard(size * reserve_ratio);
            }
            buffer.resize_discard(size);
        }
    };

    cuda_tool::CBCOOVectorView<Float, 3> gradients() const noexcept;
    cuda_tool::CBCOOMatrixView<Float, 3> hessians() const noexcept;

    void compute_dytopo_effect(ComputeDyTopoEffectInfo& info);

  protected:
    virtual void do_build() override;

  private:
    friend class DyTopoEffectLineSearchReporter;
    void init();

    friend class SimEngine;
    // only be called by SimEngine
    void compute_dytopo_effect();

    friend class DyTopoEffectReporter;
    void add_reporter(DyTopoEffectReporter* reporter);
    friend class DyTopoEffectReceiver;
    void add_receiver(DyTopoEffectReceiver* receiver);

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
