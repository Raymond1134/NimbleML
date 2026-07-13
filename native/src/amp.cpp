#include "nimbleml/kernels.hpp"
#include <cmath>

namespace nimbleml {

// Host helpers for GradScaler-style unscale + finite checks (CPU path / tests).
// GPU training uses CuPy in-place unscale in Python (NimbleML.utils.amp.GradScaler).

float amp_scale_grads(float** grads, const std::size_t* sizes, std::size_t n_tensors, float inv_scale) {
  double total = 0.0;
  for (std::size_t t = 0; t < n_tensors; ++t) {
    float* g = grads[t];
    for (std::size_t i = 0; i < sizes[t]; ++i) {
      g[i] *= inv_scale;
      total += static_cast<double>(g[i]) * g[i];
    }
  }
  return static_cast<float>(total);
}

bool amp_grads_finite(float** grads, const std::size_t* sizes, std::size_t n_tensors) {
  for (std::size_t t = 0; t < n_tensors; ++t) {
    const float* g = grads[t];
    for (std::size_t i = 0; i < sizes[t]; ++i) {
      if (!std::isfinite(g[i])) return false;
    }
  }
  return true;
}

}  // namespace nimbleml
