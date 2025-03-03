# LBMGL

A GPU-accelerated Lattice Boltzmann Method (LBM) simulation with real-time OpenGL visualization using CUDA-OpenGL interop

## Platform Support
- Windows
- Requires NVIDIA GPU with CUDA capability
- Tested on Windows 11 with RTX serie
- The source code should support running on Linux, but it has not been tested
- WSL is not support

## Prerequisites

### Required Software
1. **Visual Studio 2022**
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Install with "Desktop development with C++" workload

2. **NVIDIA CUDA Toolkit 12.4+**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Ensure your GPU supports CUDA
   - Install the toolkit with Visual Studio integration
   - Ensure that CUDA Toolkit 12.4 or higher is installed to be compatible with Visual Studio 2022 17.10 or higher

3. **GLEW (OpenGL Extension Wrangler)**
   - Download from: http://glew.sourceforge.net/

4. **FreeGLUT**
   - Download from: https://www.transmissionzero.co.uk/software/freeglut-devel/