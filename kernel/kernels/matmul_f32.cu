extern "C" __global__ void naive_matmul_f32(const float* a, const float* b, float* c,
                                            unsigned int M, unsigned int N, unsigned int K) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && column < N) {
    float acc = 0.0f;

    for (unsigned int k = 0; k < K; ++k) {
      acc += a[row * K + k] * b[k * N + column];
    }

    c[row * N + column] = acc;
  }
}
