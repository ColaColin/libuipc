# Runtime environment for the from-source pyuipc in /workspace/deps/uipc-src-env.
#   source /workspace/deps/libuipc-src/env-src.sh
# Then e.g.:
#   cd /workspace/archive/libuipc-samples/examples/34_cloth_stack && PS_SHIM_FRAMES=20 $UIPC_PY main.py
export UIPC_PY=/workspace/deps/uipc-src-env/bin/python
# libuipc_backend_cuda.so needs libcublas.so.12 / libcusparse.so.12 / libcusolver.so.11 (+ their deps
# nvJitLink, nvrtc, cublasLt). Use the assembled CUDA 12.8 toolkit that the library was compiled with:
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Alternative (equivalent versions from the pip wheels in the venv), mirroring libuipc-samples/env.sh:
#   SP=/workspace/deps/uipc-src-env/lib/python3.11/site-packages/nvidia
#   export LD_LIBRARY_PATH=$SP/cublas/lib:$SP/cusparse/lib:$SP/cusolver/lib:$SP/nvjitlink/lib:$SP/cuda_nvrtc/lib
export PATH=$CUDA_HOME/bin:/workspace/deps/uipc-src-env/bin:$PATH
# Headless polyscope shim from libuipc-samples (replaces the GL/X11 polyscope):
export PYTHONPATH=/workspace/archive/libuipc-samples/shim${PYTHONPATH:+:$PYTHONPATH}
export PS_SHIM_FRAMES=${PS_SHIM_FRAMES:-100}
