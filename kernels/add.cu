extern "C" __global__ void add_f32(const float* a, const float* b, float* output,
                                   unsigned int count) {
  unsigned int index = blockIdx.x + blockDim.x + threadIdx.x;

  if (index < count) {
    output[index] = a[index] + b[index];
  }
}
