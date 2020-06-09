cmake_minimum_required(VERSION 2.8)
project(heat1D)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH} src)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_FILES src/1DheatMPI.cc)
add_executable(heat1D ${SOURCE_FILES})
