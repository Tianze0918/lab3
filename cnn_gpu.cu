// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

// //Using declarations, if any...

// __global__ void cnn_gpu(
//     float* input,
//     float* weight,
//     float* bias,
//     float* output)
// {



//   // one dimensional block array
//   //int index=blockIdx.x * blockDim.x + threadIdx.x;
//   // three dimensional block arrays, each block correspond to an output feature map, element h,w
//   // How does each block computes the range of data it operates on? 
  
//   int block_x = blockIdx.x;
//   int block_y = blockIdx.y;
//   int block_z = blockIdx.z;

//   int thread_x = threadIdx.x;
//   int thread_y = threadIdx.y;
//   int thread_z = threadIdx.z;

//   int i=block_x;
//   int h=block_y;
//   int w=block_z;

//   int j = thread_x;


//   __shared__ float acc[2][2];
//    if (j==0 && thread_y==0 && thread_z==0) {
//         acc[0][0] = 0.f;
//         acc[0][1] = 0.f;
//         acc[1][0] = 0.f;
//         acc[1][1] = 0.f;
//     }
//     __syncthreads();

  
  

//   if (j < kNum) {
//     float temp=0;
//     for (int p = 0; p < kKernel; ++p) {
//       for (int q = 0; q < kKernel; ++q){
//         temp += weight(i,j,p,q) * input(j,h*2+p + thread_y, w*2+q + thread_z);
//       }
//     }
//     atomicAdd(&acc[thread_y][thread_z], temp);
//   }
//   __syncthreads();


//   if (j==0) {
//     atomicAdd(&acc[thread_y][thread_z], bias[i]);
//   }
//   __syncthreads();


//   if (j==0 && thread_y==0 && thread_z==0){
//     // ReLU
//     float acc1 = max(0.f, acc[0][0]);
//     float acc2 = max(0.f, acc[0][1]);
//     float acc3 = max(0.f, acc[1][0]);
//     float acc4 = max(0.f, acc[1][1]);

//     // Max pooling
//     output(i,h,w) = max(
//         max(acc1, acc2),
//         max(acc3, acc4));
//   }

//   // I have 1024 threads per blocks, I could use each thread to compute the contribution of one input feature map to output feature maps.
//   // There are 256 input feature maps, I am only using 256 threads.
// }




__global__ void cnn_gpu(
     float* input,   // [kNum][kImSize][kImSize]
     float* weight,  // [kNum][kNum][kKernel][kKernel]
     float* bias,    // [kNum]
    float*       output)  // [kNum][outH][outW]
{
    // 1) figure out which output element this thread computes
    int i     = blockIdx.z;                                       // which filter
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;            // pooled‑col, between 0 to 112
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;            // pooled‑row, between 0 to 112


    // blockIdx = (7, 7, 256)
    // blockDim = (16, 16, 1)

    // compute dims
    int convH = kImSize;
    int convW = kImSize;
    int outH  = convH / 2, outW = convW / 2;


    // 2) four accumulators for the 2×2 pooling window
    float acc00 = 0.f, acc01 = 0.f, acc10 = 0.f, acc11 = 0.f;



    // Threads in same block shares output weights and nearby inputs
    // Total 16*16=256 threads, total 
    //int weight_size = kKernel * kKernel * kNum;     // 5 * 5 * 256 = 6400
    __shared__ float weights[kNum][kKernel * kKernel];

    int idx = blockIdx.x * blockDim.x + blockDim.y;
    memcpy(&weights[idx], &weight(i, idx, 0, 0), kKernel * kKernel * sizeof(float));
    __syncthreads();



    // 3) convolve across every input channel
    for (int j = 0; j < kNum; ++j) {
        // pointer to weight[i][j][0][0]
        const float* w_ij = &weight (i, j, 0, 0);
        // pointer to input[j][0][0]
        const float* in_j = &input (j, 0, 0);

        // base coords of this 2×2 pool in the conv output
        int row0 = h_out * 2;
        int col0 = w_out * 2;

        // slide the kKernel×kKernel window
        for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
                float wval = w_ij[p * kKernel + q];
                int base  = (row0 + p) * kInImSize + (col0 + q);

                // read the four neighbors for pooling
                float v00 = in_j[ base         ];
                float v01 = in_j[ base +   1    ];
                float v10 = in_j[ base + kInImSize];
                float v11 = in_j[ base + kInImSize + 1];

                acc00 += wval * v00;
                acc01 += wval * v01;
                acc10 += wval * v10;
                acc11 += wval * v11;
            }
        }
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