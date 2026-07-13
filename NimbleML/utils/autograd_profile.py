"""Autograd graph profiling helpers."""
from __future__ import annotations
from collections import Counter
from NimbleML.utils.tensor import Tensor

# Upper bound for fused-trunk GPT train-step graphs (see test_gpt_fused_blocks_reduce_train_graph_nodes).
TRAIN_STEP_NODE_BUDGET = 80

# Fused-block train steps should stay under this count (Phase 2 fusion target).
FUSED_BLOCKS_TRAIN_NODE_BUDGET = 200


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


def profile_gpt_train_step(model, inputs, labels, *, ignore_index=None) -> dict:
    """Run ``compute_loss`` and return autograd graph statistics for the loss."""
    loss = model.compute_loss(inputs, labels, ignore_index=ignore_index)
    stats = graph_stats(loss)
    stats["loss_shape"] = loss.shape
    stats["within_budget"] = stats["nodes"] <= FUSED_BLOCKS_TRAIN_NODE_BUDGET
    return stats


def format_profile_report(stats: dict) -> str:
    """Human-readable summary for benchmark ``--profile`` output."""
    lines = [
        "Autograd train-step profile",
        f"  nodes: {stats['nodes']}",
        f"  within fused-blocks budget ({FUSED_BLOCKS_TRAIN_NODE_BUDGET}): {stats.get('within_budget')}",
        f"  fused-trunk budget ({TRAIN_STEP_NODE_BUDGET}): {stats['nodes'] <= TRAIN_STEP_NODE_BUDGET}",
    ]
    if "loss_shape" in stats:
        lines.append(f"  loss shape: {stats['loss_shape']}")
    if "logits_shape" in stats:
        lines.append(f"  logits shape: {stats['logits_shape']}")

    lines.append("  op counts:")
    for op, count in stats.get("ops", {}).items():
        lines.append(f"    {op:32} {count}")
    return "\n".join(lines)
