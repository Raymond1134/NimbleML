#include "nimbleml/kernels.hpp"
#include <cmath>
#include <limits>

namespace nimbleml {

void crossentropy_forward_cpu(const float* logits, const std::int64_t* labels, float* max_vals,
                              float* sum_exp, float* loss, std::size_t rows, std::size_t classes) {
  double total = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = logits + r * classes;
    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < classes; ++c) m = row[c] > m ? row[c] : m;
    max_vals[r] = m;
    double s = 0.0;
    for (std::size_t c = 0; c < classes; ++c) s += std::exp(static_cast<double>(row[c] - m));
    sum_exp[r] = static_cast<float>(s);
    const std::int64_t y = labels[r];
    const float log_prob = (row[y] - m) - std::log(static_cast<float>(s));
    total -= static_cast<double>(log_prob);
  }
  *loss = static_cast<float>(total / static_cast<double>(rows));
}

void crossentropy_backward_cpu(float grad_scale, const float* logits, const std::int64_t* labels,
                               const float* max_vals, const float* sum_exp, float* grad,
                               std::size_t rows, std::size_t classes) {
  const float inv_n = grad_scale / static_cast<float>(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = logits + r * classes;
    float* gout = grad + r * classes;
    const float m = max_vals[r];
    const float s = sum_exp[r];
    const std::int64_t y = labels[r];
    for (std::size_t c = 0; c < classes; ++c) {
      const float p = std::exp(row[c] - m) / s;
      gout[c] = inv_n * (p - (c == static_cast<std::size_t>(y) ? 1.0f : 0.0f));
    }
  }
}

}  // namespace nimbleml
