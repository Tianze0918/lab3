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
  //int index=blockIdx.x * blockDim.x + threadIdx.x;
  // three dimensional block arrays, each block correspond to an output feature map, element h,w
  // How does each block computes the range of data it operates on? 
  
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
  int block_z = blockIdx.z;

  int thread_x = threadIdx.x;
  // int thread_y = threadIdx.y;
  // int thread_z = threadIdx.z;

  int i=block_x;
  int h=block_y;
  int w=block_z;

  int j = thread_x;

  float temp1=0;
  float temp2=0;
  float temp3=0;
  float temp4=0;


  __shared__ float acc1, acc2, acc3, acc4;
  if (j==0){
    acc1=bias[i];
    acc2=bias[i];
    acc3=bias[i];
    acc4=bias[i];
  }


  if (j<kNum){


    for (int p = 0; p < kKernel; ++p) {
      for (int q = 0; q < kKernel; ++q){

        //float temp_result1 = weight(i,j,p,q) * input(j,h+p, w+q);
        // temp1+=temp_result1;
    

        float temp_result1 = weight(i,j,p,q) * input(j,h*2+p, w*2+q);
        temp1+=temp_result1;

        float temp_result2 = weight(i,j,p,q) * input(j,h*2+1+p, w*2+q);
       temp2+=temp_result2;

        float temp_result3 = weight(i,j,p,q) * input(j,h*2+p, w*2+1+q);
        temp3+=temp_result3;

        float temp_result4 = weight(i,j,p,q) * input(j,h*2+1+p, w*2+1+q);
        temp4+=temp_result4;
        
      }
    }
  }


  atomicAdd(&acc1,temp1);
  atomicAdd(&acc2,temp2);
  atomicAdd(&acc3,temp3);
  atomicAdd(&acc4,temp4);

  __syncthreads();

  if (j==0){
    // ReLU
    acc1 = max(0.f, acc1);
    acc2 = max(0.f, acc2);
    acc3 = max(0.f, acc3);
    acc4 = max(0.f, acc4);

    // Max pooling
    output(i,h,w) = max(
        max(acc1, acc2),
        max(acc3, acc4));
  }

  // I have 1024 threads per blocks, I could use each thread to compute the contribution of one input feature map to output feature maps.
  // There are 256 input feature maps, I am only using 256 threads.
}

