#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nimbleml {

  inline void check(bool ok, const char* msg) {
    if (!ok) throw std::runtime_error(msg);
  }

  struct BufferF32 {
    float* data = nullptr;
    std::size_t size = 0;
    bool device = false;  // true = CUDA device pointer
  };

  struct BufferI64 {
    std::int64_t* data = nullptr;
    std::size_t size = 0;
    bool device = false;
  };

  // Host helpers
  void gelu_forward_cpu(const float* x, float* out, float* tanh_u, std::size_t n);
  void gelu_backward_cpu(const float* grad, const float* x, const float* tanh_u, float* grad_x, std::size_t n);

  void rmsnorm_forward_cpu(
    const float* x, const float* gamma, float* out, float* ms, float* rms, std::size_t rows, std::size_t dim, float eps
  );

  void rmsnorm_backward_cpu(
    const float* grad, const float* x, const float* gamma, const float* ms, const float* rms, float* grad_x,
    float* grad_gamma, std::size_t rows, std::size_t dim, float eps
  );

  void embedding_scatter_add_cpu(
    float* grad_w, const std::int64_t* ids, const float* grad_out, std::size_t n, std::size_t dim, std::size_t vocab
  );

  void crossentropy_forward_cpu(
    const float* logits, const std::int64_t* labels, float* max_vals,
    float* sum_exp, float* loss, std::size_t rows, std::size_t classes
  );

  void crossentropy_backward_cpu(
    float grad_scale, const float* logits, const std::int64_t* labels, const float* max_vals,
    const float* sum_exp, float* grad, std::size_t rows, std::size_t classes
  );

  void sdpa_forward_cpu(
    const float* q, const float* k, const float* v, float* out, float* probs,
    std::size_t bh, std::size_t seq, std::size_t dk, float scale, bool causal
  );

  void sdpa_backward_cpu(
    const float* grad_out, const float* q, const float* k, const float* v, const float* probs, float* grad_q,
    float* grad_k, float* grad_v, std::size_t bh, std::size_t seq, std::size_t dk, float scale
  );

  void adamw_step_cpu(
    float* param, float* grad, float* m, float* v, float* scratch, std::size_t n, float lr,
    float beta1, float beta2, float bias1, float bias2, float eps, float weight_decay
  );

  float clip_grad_norm_cpu(float** grads, const std::size_t* sizes, std::size_t n_tensors, float max_norm);

  #if defined(NIMBLEML_WITH_CUDA)
    void gelu_forward_cuda(const float* x, float* out, float* tanh_u, std::size_t n);
    void gelu_backward_cuda(const float* grad, const float* x, const float* tanh_u, float* grad_x, std::size_t n);

    void rmsnorm_forward_cuda(
      const float* x, const float* gamma, float* out, float* ms, float* rms, std::size_t rows, std::size_t dim, float eps
    );

    void rmsnorm_backward_cuda(
      const float* grad, const float* x, const float* gamma, const float* ms, const float* rms,
      float* grad_x, float* grad_gamma, std::size_t rows, std::size_t dim, float eps
    );

    // ``scale`` = 1/sqrt(dk) (score multiplier). ``m``/``l`` are (bh*seq) softmax stats.
    void flash_sdpa_forward_cuda(
      const float* q, const float* k, const float* v, float* out, float* m,
      float* l, std::size_t bh, std::size_t seq, std::size_t dk, float scale
    );

    void flash_sdpa_backward_cuda(
      const float* grad_out, const float* q, const float* k, const float* v, const float* m, const float* l,
      float* grad_q, float* grad_k, float* grad_v, std::size_t bh, std::size_t seq, std::size_t dk, float scale
    );

  #endif

}
