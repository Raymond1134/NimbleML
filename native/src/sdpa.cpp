#include "nimbleml/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nimbleml {

void sdpa_forward_cpu(const float* q, const float* k, const float* v, float* out, float* probs,
                      std::size_t bh, std::size_t seq, std::size_t dk, float scale, bool causal) {
  std::vector<float> scores(seq * seq);
  for (std::size_t b = 0; b < bh; ++b) {
    const float* qb = q + b * seq * dk;
    const float* kb = k + b * seq * dk;
    const float* vb = v + b * seq * dk;
    float* pb = probs + b * seq * seq;
    float* ob = out + b * seq * dk;

    for (std::size_t i = 0; i < seq; ++i) {
      float m = -std::numeric_limits<float>::infinity();
      for (std::size_t j = 0; j < seq; ++j) {
        if (causal && j > i) {
          scores[i * seq + j] = -std::numeric_limits<float>::infinity();
          continue;
        }
        double dot = 0.0;
        for (std::size_t d = 0; d < dk; ++d) {
          dot += static_cast<double>(qb[i * dk + d]) * kb[j * dk + d];
        }
        const float s = static_cast<float>(dot) * scale;
        scores[i * seq + j] = s;
        m = s > m ? s : m;
      }
      double sum = 0.0;
      for (std::size_t j = 0; j < seq; ++j) {
        float e = 0.0f;
        if (!(causal && j > i)) e = std::exp(scores[i * seq + j] - m);
        pb[i * seq + j] = e;
        sum += e;
      }
      const float inv = 1.0f / static_cast<float>(sum);
      for (std::size_t j = 0; j < seq; ++j) pb[i * seq + j] *= inv;
      for (std::size_t d = 0; d < dk; ++d) {
        double acc = 0.0;
        for (std::size_t j = 0; j < seq; ++j) acc += pb[i * seq + j] * vb[j * dk + d];
        ob[i * dk + d] = static_cast<float>(acc);
      }
    }
  }
}

void sdpa_backward_cpu(const float* grad_out, const float* q, const float* k, const float* v,
                       const float* probs, float* grad_q, float* grad_k, float* grad_v,
                       std::size_t bh, std::size_t seq, std::size_t dk, float scale) {
  std::fill(grad_q, grad_q + bh * seq * dk, 0.0f);
  std::fill(grad_k, grad_k + bh * seq * dk, 0.0f);
  std::fill(grad_v, grad_v + bh * seq * dk, 0.0f);
  std::vector<float> grad_scores(seq * seq);
  for (std::size_t b = 0; b < bh; ++b) {
    const float* qb = q + b * seq * dk;
    const float* kb = k + b * seq * dk;
    const float* vb = v + b * seq * dk;
    const float* pb = probs + b * seq * seq;
    const float* gob = grad_out + b * seq * dk;
    float* gqb = grad_q + b * seq * dk;
    float* gkb = grad_k + b * seq * dk;
    float* gvb = grad_v + b * seq * dk;

    for (std::size_t i = 0; i < seq; ++i) {
      for (std::size_t j = 0; j < seq; ++j) {
        double acc = 0.0;
        for (std::size_t d = 0; d < dk; ++d) acc += gob[i * dk + d] * vb[j * dk + d];
        // softmax backward later
        grad_scores[i * seq + j] = static_cast<float>(acc);
      }
      for (std::size_t d = 0; d < dk; ++d) {
        double acc = 0.0;
        for (std::size_t j = 0; j < seq; ++j) acc += pb[i * seq + j] * gob[i * dk + d];
        // wait grad_v accumulates over i
      }
    }
    for (std::size_t j = 0; j < seq; ++j) {
      for (std::size_t d = 0; d < dk; ++d) {
        double acc = 0.0;
        for (std::size_t i = 0; i < seq; ++i) acc += pb[i * seq + j] * gob[i * dk + d];
        gvb[j * dk + d] += static_cast<float>(acc);
      }
    }
    for (std::size_t i = 0; i < seq; ++i) {
      double dot = 0.0;
      for (std::size_t j = 0; j < seq; ++j) dot += grad_scores[i * seq + j] * pb[i * seq + j];
      for (std::size_t j = 0; j < seq; ++j) {
        const float gs = pb[i * seq + j] * (grad_scores[i * seq + j] - static_cast<float>(dot));
        for (std::size_t d = 0; d < dk; ++d) {
          gqb[i * dk + d] += gs * scale * kb[j * dk + d];
          gkb[j * dk + d] += gs * scale * qb[i * dk + d];
        }
      }
    }
  }
}

}  // namespace nimbleml
