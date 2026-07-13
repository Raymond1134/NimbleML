#include "nimbleml/kernels.hpp"
#include <cmath>

namespace nimbleml {

  void adamw_step_cpu(float* param, float* grad, float* m, float* v, float* scratch, std::size_t n, loat lr,
                      float beta1, float beta2, float bias1, float bias2, float eps, float weight_decay) {
    for (std::size_t i = 0; i < n; ++i) {
      m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
      v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
      const float mhat = m[i] / bias1;
      const float vhat = v[i] / bias2;
      scratch[i] = lr * mhat / (std::sqrt(vhat) + eps);
    }
    if (weight_decay != 0.0f) {
      const float decay = 1.0f - lr * weight_decay;
      for (std::size_t i = 0; i < n; ++i) param[i] *= decay;
    }
    for (std::size_t i = 0; i < n; ++i) param[i] -= scratch[i];
  }

  float clip_grad_norm_cpu(float** grads, const std::size_t* sizes, std::size_t n_tensors, float max_norm) {
    double total = 0.0;
    for (std::size_t t = 0; t < n_tensors; ++t) {
      const float* g = grads[t];
      for (std::size_t i = 0; i < sizes[t]; ++i) total += static_cast<double>(g[i]) * g[i];
    }
    const float norm = static_cast<float>(std::sqrt(total));
    if (norm > max_norm && norm > 0.0f) {
      const float scale = max_norm / (norm + 1e-6f);
      for (std::size_t i = 0; i < n_tensors; ++i) {
        float* g = grads[t];
        for (std::size_t i = 0; i < sizes[t]; ++i) g[i] *= scale;
      }
    }
    return norm;
  }

}
