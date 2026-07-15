#include "nimbleml/kernels.hpp"
#include <cmath>

namespace nimbleml {

  namespace {
    constexpr float GELU_K = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float GELU_COEF = 0.044715f;
    constexpr float DU_DX_COEF = 0.134145f;
  }

  void gelu_forward_cpu(const float* x, float* out, float* tanh_u, std::size_t n) {
  #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
      const float xi = x[i];
      const float x3 = xi * xi * xi;
      const float tu = std::tanh(GELU_K * (xi + GELU_COEF * x3));
      tanh_u[i] = tu;
      out[i] = 0.5f * xi * (1.0f + tu);
    }
  }

  void gelu_backward_cpu(const float* grad, const float* x, const float* tanh_u, float* grad_x, std::size_t n) {
  #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
      const float xi = x[i];
      const float tu = tanh_u[i];
      const float du_dx = GELU_K * (1.0f + DU_DX_COEF * xi * xi);
      const float sech2 = 1.0f - tu * tu;
      const float d = 0.5f * (1.0f + tu) + 0.5f * xi * sech2 * du_dx;
      grad_x[i] = grad[i] * d;
    }
  }

}
