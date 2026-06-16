"""Autograd graph profiling helpers."""
from collections import Counter

from NimbleML.utils.tensor import Tensor


def graph_stats(root: Tensor) -> dict:
    """Count autograd nodes and op types reachable from ``root``."""
    visited: set[int] = set()
    ops: Counter[str] = Counter()

    def visit(node: Tensor) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        ops[node._op or "<leaf>"] += 1
        for child in node._prev:
            visit(child)

    visit(root)
    return {
        "nodes": len(visited),
        "ops": dict(sorted(ops.items(), key=lambda item: (-item[1], item[0]))),
    }


def profile_gpt_forward(model, inputs) -> dict:
    """Run one GPT forward and return graph statistics for logits."""
    logits = model.forward(inputs)
    stats = graph_stats(logits)
    stats["logits_shape"] = logits.shape
    return stats
