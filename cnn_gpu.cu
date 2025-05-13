// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

// Using declarations, if any...

__global__ void cnn_gpu(
    float* input,
    float* weight,
    float* bias,
    float* output)
{
  // one dimensional block array
  int index=blockIdx.x * blockDim.x + threadIdx.x;
  // three dimensional block arrays, each block correspond to an output feature map, element h,w
  // How does each block computes the range of data it operates on? 
  
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int block_z = blockIdx.z;

  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;
  int thread_z = threadIdx.z;

  for (int j=0; j  < kNum; j++){
    for (int w = 0; w < kImSize; ++w) {
      for (int p = 0; p < kKernel; ++p) {
        output(i,h,w) += weight(i,j,p,q) * input(j,h+p, w+q);
      }
    }
  }
}
