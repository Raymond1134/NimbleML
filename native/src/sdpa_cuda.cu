#include "nimbleml/kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <string>

namespace nimbleml {
namespace {

static void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

__global__ void gelu_forward_kernel(const float* x, float* out, float* tanh_u, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float k = 0.7978845608028654f;
  const float coef = 0.044715f;
  const float xi = x[i];
  const float x3 = xi * xi * xi;
  const float tu = tanhf(k * (xi + coef * x3));
  tanh_u[i] = tu;
  out[i] = 0.5f * xi * (1.0f + tu);
}

__global__ void gelu_backward_kernel(const float* grad, const float* x, const float* tanh_u,
                                     float* grad_x, std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float k = 0.7978845608028654f;
  const float du = 0.134145f;
  const float xi = x[i];
  const float tu = tanh_u[i];
  const float du_dx = k * (1.0f + du * xi * xi);
  const float sech2 = 1.0f - tu * tu;
  grad_x[i] = grad[i] * (0.5f * (1.0f + tu) + 0.5f * xi * sech2 * du_dx);
}

// One thread per (batch_head, query row). ``scale`` is the score multiplier (1/sqrt(dk)).
// Writes online-softmax stats ``m`` / ``l`` for a numerically stable device backward.
__global__ void flash_sdpa_fwd_kernel(const float* q, const float* k, const float* v, float* out,
                                      float* m_out, float* l_out, int seq, int dk, float scale) {
  const int bh = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= seq) return;

  const std::size_t base = static_cast<std::size_t>(bh) * seq * dk;
  const float* qb = q + base + static_cast<std::size_t>(i) * dk;
  float m = -1e30f;
  float l = 0.0f;

  extern __shared__ float shared[];
  float* acc = shared + threadIdx.x * dk;
  for (int d = 0; d < dk; ++d) acc[d] = 0.0f;

  for (int j = 0; j <= i; ++j) {
    const float* kb = k + base + static_cast<std::size_t>(j) * dk;
    float dot = 0.0f;
    for (int d = 0; d < dk; ++d) dot += qb[d] * kb[d];
    const float s = dot * scale;
    const float m_new = fmaxf(m, s);
    const float e1 = expf(m - m_new);
    const float e2 = expf(s - m_new);
    l = l * e1 + e2;
    for (int d = 0; d < dk; ++d) acc[d] *= e1;
    const float* vb = v + base + static_cast<std::size_t>(j) * dk;
    for (int d = 0; d < dk; ++d) acc[d] += e2 * vb[d];
    m = m_new;
  }

  const float inv = 1.0f / fmaxf(l, 1e-20f);
  float* ob = out + base + static_cast<std::size_t>(i) * dk;
  for (int d = 0; d < dk; ++d) ob[d] = acc[d] * inv;

  const std::size_t mi = static_cast<std::size_t>(bh) * seq + i;
  m_out[mi] = m;
  l_out[mi] = l;
}

// Device FA backward: one thread per query row. Recomputes causal probs from saved (m,l);
// accumulates dQ locally and atomics into dK/dV. No host staging.
__global__ void flash_sdpa_bwd_kernel(const float* grad_out, const float* q, const float* k,
                                      const float* v, const float* m_in, const float* l_in,
                                      float* grad_q, float* grad_k, float* grad_v, int seq, int dk,
                                      float scale) {
  const int bh = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= seq) return;

  const std::size_t base = static_cast<std::size_t>(bh) * seq * dk;
  const std::size_t mi = static_cast<std::size_t>(bh) * seq + i;
  const float m = m_in[mi];
  const float l = fmaxf(l_in[mi], 1e-20f);
  const float inv_l = 1.0f / l;

  const float* qb = q + base + static_cast<std::size_t>(i) * dk;
  const float* gob = grad_out + base + static_cast<std::size_t>(i) * dk;

  // First pass: D = sum_j P_ij * dP_ij where dP_ij = <do_i, v_j>
  float D = 0.0f;
  for (int j = 0; j <= i; ++j) {
    const float* kb = k + base + static_cast<std::size_t>(j) * dk;
    float dot = 0.0f;
    for (int d = 0; d < dk; ++d) dot += qb[d] * kb[d];
    const float p = expf(dot * scale - m) * inv_l;
    const float* vb = v + base + static_cast<std::size_t>(j) * dk;
    float dP = 0.0f;
    for (int d = 0; d < dk; ++d) dP += gob[d] * vb[d];
    D += p * dP;
  }

  float* gqb = grad_q + base + static_cast<std::size_t>(i) * dk;
  for (int d = 0; d < dk; ++d) gqb[d] = 0.0f;

  for (int j = 0; j <= i; ++j) {
    const float* kb = k + base + static_cast<std::size_t>(j) * dk;
    float dot = 0.0f;
    for (int d = 0; d < dk; ++d) dot += qb[d] * kb[d];
    const float p = expf(dot * scale - m) * inv_l;
    const float* vb = v + base + static_cast<std::size_t>(j) * dk;

    float dP = 0.0f;
    for (int d = 0; d < dk; ++d) dP += gob[d] * vb[d];
    const float dS = p * (dP - D);  // d(score)/d before scale; scores = dot * scale

    // dV_j += p * do_i
    float* gvb = grad_v + base + static_cast<std::size_t>(j) * dk;
    for (int d = 0; d < dk; ++d) atomicAdd(gvb + d, p * gob[d]);

    // dQ_i += dS * scale * k_j ; dK_j += dS * scale * q_i
    float* gkb = grad_k + base + static_cast<std::size_t>(j) * dk;
    for (int d = 0; d < dk; ++d) {
      const float g = dS * scale;
      gqb[d] += g * kb[d];
      atomicAdd(gkb + d, g * qb[d]);
    }
  }
}

// Real on-device RMSNorm: one block per row.
__global__ void rmsnorm_fwd_kernel(const float* x, const float* gamma, float* out, float* ms,
                                   float* rms, int rows, int dim, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + static_cast<std::size_t>(row) * dim;
  float* orow = out + static_cast<std::size_t>(row) * dim;

  // Parallel reduction of sum(x^2) within the block.
  __shared__ float shared[256];
  float local = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float v = xr[d];
    local += v * v;
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float mean_sq = shared[0] / static_cast<float>(dim);
  const float r = sqrtf(mean_sq + eps);
  if (threadIdx.x == 0) {
    ms[row] = mean_sq;
    rms[row] = r;
  }
  __syncthreads();
  const float inv = 1.0f / r;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    orow[d] = xr[d] * inv * gamma[d];
  }
}

__global__ void rmsnorm_bwd_kernel(const float* grad, const float* x, const float* gamma,
                                   const float* ms, const float* rms, float* grad_x,
                                   float* grad_gamma, int rows, int dim, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + static_cast<std::size_t>(row) * dim;
  const float* gr = grad + static_cast<std::size_t>(row) * dim;
  float* gxr = grad_x + static_cast<std::size_t>(row) * dim;
  const float r = rms[row];
  const float inv = 1.0f / r;
  const float mean_sq = ms[row];

  // row_dot = sum(grad * gamma * x) in fp32 for the ms gradient term.
  __shared__ float shared[256];
  float local = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    local += (gr[d] * gamma[d]) * xr[d];
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float row_dot = shared[0];
  const float grad_ms = row_dot * (-0.5f) * powf(mean_sq + eps, -1.5f);
  const float coef = (2.0f / static_cast<float>(dim)) * grad_ms;

  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float x_hat = xr[d] * inv;
    atomicAdd(grad_gamma + d, gr[d] * x_hat);
    gxr[d] = gr[d] * gamma[d] * inv + xr[d] * coef;
  }
}

}  // namespace

void gelu_forward_cuda(const float* x, float* out, float* tanh_u, std::size_t n) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  gelu_forward_kernel<<<blocks, threads>>>(x, out, tanh_u, n);
  check_cuda(cudaGetLastError(), "gelu_forward_cuda");
}

void gelu_backward_cuda(const float* grad, const float* x, const float* tanh_u, float* grad_x,
                        std::size_t n) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  gelu_backward_kernel<<<blocks, threads>>>(grad, x, tanh_u, grad_x, n);
  check_cuda(cudaGetLastError(), "gelu_backward_cuda");
}

void rmsnorm_forward_cuda(const float* x, const float* gamma, float* out, float* ms, float* rms,
                          std::size_t rows, std::size_t dim, float eps) {
  const int threads = 256;
  rmsnorm_fwd_kernel<<<static_cast<int>(rows), threads>>>(
      x, gamma, out, ms, rms, static_cast<int>(rows), static_cast<int>(dim), eps);
  check_cuda(cudaGetLastError(), "rmsnorm_forward_cuda");
}

void rmsnorm_backward_cuda(const float* grad, const float* x, const float* gamma, const float* ms,
                           const float* rms, float* grad_x, float* grad_gamma, std::size_t rows,
                           std::size_t dim, float eps) {
  check_cuda(cudaMemset(grad_gamma, 0, dim * sizeof(float)), "rmsnorm_bwd memset gamma");
  const int threads = 256;
  rmsnorm_bwd_kernel<<<static_cast<int>(rows), threads>>>(
      grad, x, gamma, ms, rms, grad_x, grad_gamma, static_cast<int>(rows), static_cast<int>(dim),
      eps);
  check_cuda(cudaGetLastError(), "rmsnorm_backward_cuda");
}

void flash_sdpa_forward_cuda(const float* q, const float* k, const float* v, float* out, float* m,
                             float* l, std::size_t bh, std::size_t seq, std::size_t dk,
                             float scale) {
  if (dk > 256) {
    throw std::runtime_error("flash_sdpa_forward_cuda: dk > 256 not supported");
  }
  dim3 block(32);
  dim3 grid(static_cast<unsigned>((seq + 31) / 32), static_cast<unsigned>(bh));
  const std::size_t shmem = 32 * dk * sizeof(float);
  flash_sdpa_fwd_kernel<<<grid, block, shmem>>>(q, k, v, out, m, l, static_cast<int>(seq),
                                                static_cast<int>(dk), scale);
  check_cuda(cudaGetLastError(), "flash_sdpa_forward_cuda");
}

void flash_sdpa_backward_cuda(const float* grad_out, const float* q, const float* k, const float* v,
                              const float* m, const float* l, float* grad_q, float* grad_k,
                              float* grad_v, std::size_t bh, std::size_t seq, std::size_t dk,
                              float scale) {
  const std::size_t n = bh * seq * dk;
  check_cuda(cudaMemset(grad_q, 0, n * sizeof(float)), "flash_bwd memset gq");
  check_cuda(cudaMemset(grad_k, 0, n * sizeof(float)), "flash_bwd memset gk");
  check_cuda(cudaMemset(grad_v, 0, n * sizeof(float)), "flash_bwd memset gv");
  dim3 block(32);
  dim3 grid(static_cast<unsigned>((seq + 31) / 32), static_cast<unsigned>(bh));
  flash_sdpa_bwd_kernel<<<grid, block>>>(grad_out, q, k, v, m, l, grad_q, grad_k, grad_v,
                                         static_cast<int>(seq), static_cast<int>(dk), scale);
  check_cuda(cudaGetLastError(), "flash_sdpa_backward_cuda");
}

}  // namespace nimbleml
