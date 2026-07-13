#pragma once
#include <functional>
#include <vector>

namespace nimbleml {

class AutogradEngine {
 public:
  int add_node(const std::vector<int>& parents, bool requires_grad, std::function<void()> backward);
  void run_backward(int root_id, bool retain_graph);
  void clear();
  std::size_t size() const;
};

AutogradEngine& engine();

float amp_scale_grads(float** grads, const std::size_t* sizes, std::size_t n_tensors, float inv_scale);
bool amp_grads_finite(float** grads, const std::size_t* sizes, std::size_t n_tensors);

}  // namespace nimbleml
