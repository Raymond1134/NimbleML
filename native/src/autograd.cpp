#include "nimbleml/autograd.hpp"
#include "nimbleml/kernels.hpp"
#include <functional>
#include <utility>
#include <vector>

namespace nimbleml {

  struct AutogradNode {
    int id = -1;
    std::vector<int> parents;
    std::function<void()> backward;
    bool requires_grad = false;
  };

  static std::vector<AutogradNode> g_nodes;

  int AutogradEngine::add_node(const std::vector<int>& parents, bool requires_grad, std::function<void()> backward) {
    AutogradNode node;
    node.id = static_cast<int>(g_nodes.size());
    node.parents = parents;
    node.requires_grad = requires_grad;
    node.backward = std::move(backward);
    g_nodes.push_back(std::move(node));
    return node.id;
  }

  void AutogradEngine::run_backward(int root_id, bool retain_graph) {
    check(root_id >= 0 && root_id < static_cast<int>(g_nodes.size()), "invalid root");
    std::vector<int> order;
    std::vector<char> seen(g_nodes.size(), 0);
    std::vector<std::pair<int, int>> stack;
    stack.emplace_back(root_id, 0);
    while (!stack.empty()) {
      auto [id, state] = stack.back();
      stack.pop_back();
      if (state == 0) {
        if (seen[id]) continue;
        seen[id] = 1;
        stack.emplace_back(id, 1);
        const auto& parents = g_nodes[id].parents;
        for (auto it = parents.rbegin(); it != parents.rend(); ++it) {
          if (*it >= 0 && !seen[*it]) stack.emplace_back(*it, 0);
        }
      } else {
        order.push_back(id);
      }
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      auto& node = g_nodes[*it];
      if (node.backward) node.backward();
      if (!retain_graph) {
        node.backward = nullptr;
        node.parents.clear();
      }
    }
  }

  void AutogradEngine::clear() { g_nodes.clear(); }

  std::size_t AutogradEngine::size() const { return g_nodes.size(); }

  static AutogradEngine g_engine;
  AutogradEngine& engine() { return g_engine; }

}
