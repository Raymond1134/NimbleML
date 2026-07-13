cmake_minimum_required(VERSION 3.18)
project(nimbleml_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NIMBLEML_WITH_CUDA "Build CUDA kernels" OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(NIMBLEML_SOURCES
  src/module.cpp
  src/gelu.cpp
  src/rmsnorm.cpp
  src/scatter.cpp
  src/crossentropy.cpp
  src/sdpa.cpp
  src/autograd.cpp
  src/adamw.cpp
  src/amp.cpp
  src/buffer.cpp
)

if(NIMBLEML_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    list(APPEND NIMBLEML_SOURCES src/sdpa_cuda.cu src/kernels_cuda.cu)
    add_compile_definitions(NIMBLEML_WITH_CUDA=1)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "NimbleML: CUDA enabled")
  else()
    message(WARNING "NIMBLEML_WITH_CUDA=ON but no CUDA compiler; building CPU-only")
    set(NIMBLEML_WITH_CUDA OFF)
  endif()
endif()

pybind11_add_module(nimbleml_native MODULE ${NIMBLEML_SOURCES})
target_include_directories(nimbleml_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(nimbleml_native PRIVATE /O2 /EHsc)
else()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nimbleml_native PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_compile_options(nimbleml_native PRIVATE -O3)
endif()

if(NIMBLEML_WITH_CUDA)
  # Ampere (80/86) / Ada (89) / Hopper (90) / Blackwell (100) / Blackwell Ultra (103)
  set_target_properties(nimbleml_native PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90;100;103")
  # cudart for kernels; cublas for future device GEMM helpers (CuPy already uses cuBLAS).
  target_link_libraries(nimbleml_native PRIVATE CUDA::cudart CUDA::cublas)
endif()

install(TARGETS nimbleml_native DESTINATION .)
