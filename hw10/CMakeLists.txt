cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project (threadIdx LANGUAGES CXX CUDA)

find_package(cxxopts REQUIRED)

add_executable( threadIdx threadIdx.cu )
# target_sources( threadIdx PRIVATE threadIdx.cu)

set(CMAKE_CUDA_ARCHITECTURES 70)

target_link_libraries(threadIdx PRIVATE cxxopts::cxxopts)

install(TARGETS threadIdx DESTINATION . )
