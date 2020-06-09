cmake_minimum_required(VERSION 2.8)

project(heat2D3D)

find_package(CUDA REQUIRED)

include_directories(
src
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS)
cuda_add_executable(
	heat2D3D
	src/2D3DheatCUDA.cu
)







