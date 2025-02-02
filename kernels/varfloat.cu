#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdint-gcc.h>
#include "stdio.h"

#define SIGN(val) ((0 < val) - (val < 0))


__global__ void __varfloat_cuda__(const float* in, const int N, float* out, int fraction_bits,
                                  float max_sat, float min_sat, bool round) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
      auto value = in[i];
      auto value_hex = *reinterpret_cast<const int32_t *>(&value);

      // Round to nearest even
      if (round) {
          int mant_lsb = 0x1u << (23 - fraction_bits);
          int trunc_msb = 0x1u << (23 - 1 - fraction_bits);
          int trunc_rest = trunc_msb - 1;
          if ((value_hex & trunc_msb) and (value_hex & (mant_lsb | trunc_rest))){
              value_hex += mant_lsb;
          }
      }

      // Adjust fraction bits
      int mask = ((1u << (9 + fraction_bits)) - 1) << ((23 - fraction_bits));
      value_hex &= mask;
      out[i] = *reinterpret_cast<float *>(&(value_hex));

      // Verify exponent with-in allowed range
      auto vabs = fabs(out[i]);
      if (vabs > max_sat)
          out[i] = SIGN(value) * max_sat;
      if (vabs < min_sat)
          if (vabs > min_sat / 2)
              out[i] = min_sat;
          else
              out[i] = 0.;
  }
}

#define block_count 32
#define thread_per_block 1024

// Wrapper for ATen
torch::Tensor fromFloat(torch::Tensor in, int exp_bits, int fraction_bits, bool round) {
    if (in.device().type() != torch::kCUDA)
        throw std::runtime_error("Invalid device. Only CUDA supported.");
    if (in.type().scalarType() != torch::kFloat32)
        throw std::runtime_error("Invalid dtype. Only FloatTensor supported.");

    int N = in.numel();
    auto out = at::empty_like(in);

    int one_hex = 0x3f800000 | (((0x1u << fraction_bits) - 1) << (23 - fraction_bits));
    auto one_truncated = *reinterpret_cast<float *>(&(one_hex));

    int max_exp = ((1u << (exp_bits - 1)) + 127) << 23;
    auto max_sat = *reinterpret_cast<float *>(&(max_exp)) * one_truncated;

    auto min_exp = (-((1u << (exp_bits - 1)) - 1) + 127) << 23;
    auto min_sat = *reinterpret_cast<float *>(&(min_exp));

    __varfloat_cuda__<<<block_count, thread_per_block>>>(in.data<float>(), N, out.data<float>(), fraction_bits,
                                                     max_sat, min_sat, round);

    return out;
}

torch::Tensor satBounds(int exp_bits, int fraction_bits) {
    auto out = torch::zeros(2);
    int one_hex = 0x3f800000 | (((0x1u << fraction_bits) - 1) << (23 - fraction_bits));
    auto one_truncated = *reinterpret_cast<float *>(&(one_hex));

    int max_exp = ((1u << (exp_bits - 1)) + 127) << 23;
    auto max_sat = *reinterpret_cast<float *>(&(max_exp)) * one_truncated;

    auto min_exp = (-((1u << (exp_bits - 1)) - 1) + 127) << 23;
    auto min_sat = *reinterpret_cast<float *>(&(min_exp));

    out[0] = min_sat;
    out[1] = max_sat;
    return out;
}
