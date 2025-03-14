# LBMGL

A GPU-accelerated Lattice Boltzmann Method (LBM) simulation with real-time OpenGL visualization using CUDA-OpenGL interop

## Preview Animation

Below is a real-time preview of the program's output:

Parameters: Nx=Ny=1024, Re=35000, U=0.3

[![LBM Animation](http://img.youtube.com/vi/3C0EU_5-CvM/0.jpg)](http://www.youtube.com/watch?v=3C0EU_5-CvM)


## Platform Support
- Windows
- Requires NVIDIA GPU with CUDA capability
- Tested on Windows 11 with RTX serie
- The source code should support running on Linux, but it has not been tested
- WSL is not supported

## Prerequisites

### Required Software
1. **Visual Studio 2022**
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Install with "Desktop development with C++" workload

2. **NVIDIA CUDA Toolkit 12.8**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Ensure your GPU supports CUDA
   - Install the toolkit with Visual Studio integration
   - Ensure that CUDA Toolkit 12.4 or higher is installed to be compatible with Visual Studio 2022 17.10 or higher

3. **GLEW and FreeGLUT**
   
   Install GLEW and FreeGLUT by vcpkg:

     `git clone https://github.com/Microsoft/vcpkg.git`
     
     `.\vcpkg\bootstrap-vcpkg.bat`
   
     `vcpkg integrate install`
   
     `vcpkg install glew:x64-windows`
   
     `vcpkg install freeglut:x64-windows`
   
