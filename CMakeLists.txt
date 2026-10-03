cmake_minimum_required(VERSION 3.20)
project(cuda_performance_profiling_labs LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
add_executable(timing_lab examples/timing_lab.cu)
