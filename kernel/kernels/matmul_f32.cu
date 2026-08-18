extern "C" __global__ void naive_matmul_f32(const float* a, const float* b, float* c,
                                            unsigned int M, unsigned int N, unsigned int K) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && column < N) {
    unsigned int output_index = row * N + column;
    c[output_index] = 0.0f;

    for (unsigned int k = 0; k < K; ++k) {
      c[output_index] = c[output_index] + a[row * K + k] * b[k * N + column];
    }
  }
}
