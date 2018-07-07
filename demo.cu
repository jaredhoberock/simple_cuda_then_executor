// nvcc -std=c++11 --expt-extended-lambda demo.cu

#include "simple_cuda_then_executor.hpp"
#include <cstdio>

int main()
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto ready = make_ready_cuda_future(stream);

  simple_cuda_then_executor ex;

  auto fut1 = ex.then_execute([] __host__ __device__ ()
  {
    printf("hello, world from continuation 1\n");
  },
  ready
  );
  
  auto fut2 = ex.then_execute([] __host__ __device__ ()
  {
    printf("hello, world from continuation 2\n");
  },
  fut1
  );

  fut2.wait();

  cudaStreamDestroy(stream);
}

