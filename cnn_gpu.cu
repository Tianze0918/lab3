// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"


__global__ void cnn_gpu(
     float* input,   // [kNum][kInImSize][kInImSize]
     float* weight,  // [kNum][kNum][kKernel][kKernel]
     float* bias,    // [kNum]
    float*       output)  // [kNum][outH][outW]
{
    // 1) figure out which output element this thread computes
    int i     = blockIdx.z;                                       // which filter
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;            // pooled‑col, between 0 to 112
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;            // pooled‑row, between 0 to 112
    int outH  = kImSize / 2, outW = kImSize / 2;

    // 2) input loading coordinates
    const int tileW = blockDim.x*2 + (kKernel-1);  // 36
    const int tileH = blockDim.y*2 + (kKernel-1);  // 36 
    const int tileSize = tileW * tileH;    // 1296
    int lane        = threadIdx.y * blockDim.x + threadIdx.x;   // 0…255
   



 



    // 4) Loading weights into block
    // __shared__ float weights[kNum][kKernel * kKernel];
    // float* wdst = &weights[0][0];
    // float* wsrc = &weight(i,0,0,0);
    // int stride=blockDim.y * blockDim.x;    // 256
    // int weightSize = kNum * kKernel * kKernel;
    // #pragma unroll
    // for(int x=lane; x<weightSize; x+=stride){
    //   wdst[x] = wsrc[x];
    // }
    // __syncthreads();







    


     __shared__ float inBuf[36][36];
     __shared__ float weights[kKernel * kKernel];
    // 6) Loop over 
    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;
    for (int j = 0; j < kNum; j += 1) {
      // float* w_ij = &weights[j][0];



      float* wdst = &weights[0];
      float* wsrc = &weight(i,j,0,0);
      int stride=blockDim.y * blockDim.x;    // 256
      int weightSize =  kKernel * kKernel;
      #pragma unroll
      for(int x=lane; x<weightSize; x+=stride){
        wdst[x] = wsrc[x];
      }
      __syncthreads();

      float* w_ij = &weights[0];




      int threadSize = blockDim.x * blockDim.y;
      #pragma unroll
      for (int tid = lane; tid < tileSize; tid += threadSize) {
        int row_start = tid / tileW;
        int col_start = tid % tileW;

        // Compute global input coordinates
        int input_row = 2 * blockIdx.y * blockDim.y + row_start;
        int input_col = 2 * blockIdx.x * blockDim.x + col_start;

        // Boundary check and pad with zeros
        if (input_row < kInImSize && input_col < kInImSize) {
            inBuf[row_start][col_start] = input[j * kInImSize * kInImSize 
                                              + input_row * kInImSize 
                                              + input_col];
        } else {
            inBuf[row_start][col_start] = 0.0f; // Explicit padding
        }
    }
    __syncthreads();




    #pragma unroll
    for (int p = 0; p < kKernel; ++p) {
      #pragma unroll
      for (int q = 0; q < kKernel; ++q) {
        float wval = w_ij[p * kKernel + q];     
        int row = threadIdx.y * 2 + p;
        int col = threadIdx.x * 2 + q;

        // read the four neighbors for pooling
        float v00 = inBuf[row][col];
        float v01 = inBuf[row][col + 1];
        float v10 = inBuf[row + 1][col];
        float v11 = inBuf[row + 1][col + 1];

        acc00 += wval * v00;
        acc01 += wval * v01;
        acc10 += wval * v10;
        acc11 += wval * v11;
      }
    }



    __syncthreads();
  



    }

  

    // 4) add bias, apply ReLU
    float b = bias[i];
    acc00 = fmaxf(acc00 + b, 0.f);
    acc01 = fmaxf(acc01 + b, 0.f);
    acc10 = fmaxf(acc10 + b, 0.f);
    acc11 = fmaxf(acc11 + b, 0.f);

    // 5) 2×2 max‐pool
    float pool = fmaxf(fmaxf(acc00, acc01),
                       fmaxf(acc10, acc11));

    // 6) write the one output
    int outIndex = (i * outH + h_out) * outW + w_out;
    output[outIndex] = pool;
}



