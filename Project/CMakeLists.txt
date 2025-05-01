cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project (matmul LANGUAGES CXX CUDA)

find_package(cxxopts REQUIRED)

add_executable( matmul matmul.cu )
# target_sources( matmul PRIVATE matmul.cu)

set(CMAKE_CUDA_ARCHITECTURES 70)

target_link_libraries(matmul PRIVATE cxxopts::cxxopts)

install(TARGETS matmul DESTINATION . )
