#include "nimbleml/kernels.hpp"
#include <cstring>

namespace nimbleml {

  void embedding_scatter_add_cpu(float* grad_w, const std::int64_t* ids, const float* grad_out,
                                std::size_t n, std::size_t dim, std::size_t /*vocab*/) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::int64_t id = ids[i];
      float* row = grad_w + static_cast<std::size_t>(id) * dim;
      const float* g = grad_out + i * dim;
      for (std::size_t j = 0; j < dim; ++j) row[j] += g[j];
    }
  }

}
