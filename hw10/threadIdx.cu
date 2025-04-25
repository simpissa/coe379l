#include <iostream>
#include <math.h>
#include <chrono>
using namespace std::chrono;
using myclock = steady_clock;
#include "cxxopts.hpp"


__global__ void fillarray(int* x, size_t width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int gi = blockIdx.y * gridDim.x + blockIdx.x;
  int idx = i * width + j;
  x[idx] = threadIdx.y * blockDim.x + threadIdx.x + (gi * blockDim.x * blockDim.y);
 
}


#define CU_ASSERT( code ) { \
  cudaError_t err = code ; \
  if (err!=cudaSuccess) { \
    printf("error <<%s>> on line %d\n",cudaGetErrorString(err),__LINE__); \
    return 1; } }


int main(int argc,char **argv) {

  cxxopts::Options options
    ( "cxxopts","threadIdx" );
  options.add_options()
    ( "h,help","usage information" )
    ( "n,bx","blocksize x", cxxopts::value<size_t>()->default_value("2") )
    ( "m,by","blocksize y", cxxopts::value<size_t>()->default_value("3") )
    ( "N,gx","gridsize x", cxxopts::value<size_t>()->default_value("8") )
    ( "M,gy","gridsize y", cxxopts::value<size_t>()->default_value("12") );
  auto result = options.parse(argc, argv);
  if (result.count("help")>0) {
    std::cout << options.help() << '\n';
    return 0;
  }

  size_t bx = result["bx"].as<size_t>();
  size_t by = result["by"].as<size_t>();
  size_t gx = result["gx"].as<size_t>();
  size_t gy = result["gy"].as<size_t>();

  size_t width = bx * gx;
  size_t height = by * gy;

  size_t nbytes = width * height * sizeof(int);

  int* x;
  CU_ASSERT(cudaMallocManaged(&x, nbytes));

  dim3 block(bx, by);
  dim3 grid(gx, gy);

  fillarray<<<grid, block>>>(x, width);
  CU_ASSERT(cudaDeviceSynchronize());

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      std::cout << x[i * width + j] << '\t';
    }
    std::cout << '\n';
  }

  CU_ASSERT(cudaFree(x));  

  return 0;
}
