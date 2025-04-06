cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project (addvector LANGUAGES CXX CUDA)

find_package(cxxopts REQUIRED)

add_executable( addvector addvector.cu )
# target_sources( addvector PRIVATE addvector.cu)

set(CMAKE_CUDA_ARCHITECTURES 70)

target_link_libraries(addvector PRIVATE cxxopts::cxxopts)

install(TARGETS addvector DESTINATION . )
