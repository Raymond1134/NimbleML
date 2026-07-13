#include "nimbleml/kernels.hpp"
#include <cmath>

namespace nimbleml {

void rmsnorm_forward_cpu(const float* x, const float* gamma, float* out, float* ms, float* rms,
                         std::size_t rows, std::size_t dim, float eps) {
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t r = 0; r < static_cast<std::ptrdiff_t>(rows); ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * dim;
    float* outr = out + static_cast<std::size_t>(r) * dim;
    double acc = 0.0;
    for (std::size_t j = 0; j < dim; ++j) acc += static_cast<double>(xr[j]) * xr[j];
    const float mean_sq = static_cast<float>(acc / static_cast<double>(dim));
    const float rrms = std::sqrt(mean_sq + eps);
    ms[r] = mean_sq;
    rms[r] = rrms;
    for (std::size_t j = 0; j < dim; ++j) outr[j] = (xr[j] / rrms) * gamma[j];
  }
}

void rmsnorm_backward_cpu(const float* grad, const float* x, const float* gamma, const float* ms,
                          const float* rms, float* grad_x, float* grad_gamma, std::size_t rows,
                          std::size_t dim, float eps) {
  for (std::size_t j = 0; j < dim; ++j) grad_gamma[j] = 0.0f;
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * dim;
    const float* gr = grad + r * dim;
    float* gxr = grad_x + r * dim;
    const float rrms = rms[r];
    const float mean_sq = ms[r];
    double gms = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
      const float xhat = xr[j] / rrms;
      grad_gamma[j] += gr[j] * xhat;
      const float gxhat = gr[j] * gamma[j];
      gms += static_cast<double>(gxhat) * xr[j] * (-0.5f) * std::pow(mean_sq + eps, -1.5f);
      gxr[j] = gxhat / rrms;
    }
    const float scale = static_cast<float>((2.0 / static_cast<double>(dim)) * gms);
    for (std::size_t j = 0; j < dim; ++j) gxr[j] += scale * xr[j];
  }
}

}  // namespace nimbleml
