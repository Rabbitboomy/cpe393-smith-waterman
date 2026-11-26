# Primary build — works on Windows (Visual Studio / Ninja) and Linux/Colab.
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release -j
# Override GPU targets with e.g. -DSW_CUDA_ARCHS=native  or  -DSW_CUDA_ARCHS="86;120"
cmake_minimum_required(VERSION 3.24)
project(sw_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# One binary for every GPU we touch:
#   75 = T4 (Colab)   86 = RTX 30xx   89 = L4 (Colab) / RTX 40xx   120 = RTX 50xx (Blackwell)
# sm_120 needs CUDA >= 12.8, so it is added only when the compiler supports it.
set(SW_CUDA_ARCHS "" CACHE STRING "CUDA architectures (empty = team default list)")
if(SW_CUDA_ARCHS STREQUAL "")
  set(SW_CUDA_ARCHS 75 86 89)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND SW_CUDA_ARCHS 120)
  else()
    message(WARNING "nvcc ${CMAKE_CUDA_COMPILER_VERSION} < 12.8: no sm_120 code; this build will NOT run on RTX 50xx")
  endif()
endif()
message(STATUS "CUDA architectures: ${SW_CUDA_ARCHS}")

set(SW_CUDA_FLAGS -O3 -lineinfo)  # -lineinfo: source lines in Nsight Compute, no speed cost

add_executable(sw_cpu src/cpu/main_cpu.cpp src/cpu/cpu_sw.cpp)
target_include_directories(sw_cpu PRIVATE include src/cpu)

add_executable(sw_gpu
  src/gpu/main_gpu.cu src/gpu/gpu_common.cu
  src/gpu/gpu_v0.cu src/gpu/gpu_v1.cu src/gpu/gpu_v2.cu src/gpu/gpu_v3.cu src/gpu/gpu_v4.cu)
target_include_directories(sw_gpu PRIVATE include src/gpu)
target_compile_options(sw_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SW_CUDA_FLAGS}>)
set_target_properties(sw_gpu PROPERTIES CUDA_ARCHITECTURES "${SW_CUDA_ARCHS}")

add_executable(devinfo tools/devinfo.cu)
set_target_properties(devinfo PROPERTIES CUDA_ARCHITECTURES "${SW_CUDA_ARCHS}")

# Put all executables directly in build/bin (also on multi-config Visual Studio).
foreach(t sw_cpu sw_gpu devinfo)
  set_target_properties(${t} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
endforeach()
