#include "cnn.cuh"
#include "cnn_seq.cuh"

// Sequential CNN implementation
void cnn_seq(
    const float *input,
    const float *weight,
    const float *bias,
    float *output
  ) {

  // Allocate memory on heap to avoid stack overflow.
  auto c_size = kNum * kImSize * kImSize * sizeof(float);
  // kImSize: Output feature map dimension
  // kNUM: Number of input feature maps
  float *C = static_cast<float*>(malloc(c_size));

  // Bias
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C(i,h,w) = bias[i];
      }
    }
  }

  // Convolution
  for (int i = 0; i < kNum; ++i) {
    for (int j = 0; j < kNum; ++j) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q)
              C(i,h,w) += weight(i,j,p,q) * input(j,h+p,w+q);
              // Input: j represents input layer, h+p, w+q represent convolved elements.
              //        is the input element equivalent across different layers?

              // weight: i represents output layer number, j represents input layer number, p and q represents patch matrix index
              //         is the weight matrix identical across different pairs of i, j?
                

              // Ideas:
              // 1. CUDA exploits SIMT, where every thread in same warp executes same instruction. This can be exploited with the rows of kImSize. 
              // 2. Permuting loop i and j, since second approach reuses the same input feature map across different output layers. 
              // 3. Use 3d grids/blocks for every output layer i, element h, w. This way each grid shares same c. 
          }
        }
      }
    }
  }

  // ReLU
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C(i,h,w) = max(0.f, C(i,h,w));
      }
    }
  }

  // Max pooling
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        output(i,h,w) = max(
            max(C(i, h*2, w*2  ), C(i, h*2+1, w*2  )),
            max(C(i, h*2, w*2+1), C(i, h*2+1, w*2+1)));
      }
    }
  }

  delete C;
}