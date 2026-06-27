"""PyTorch reference GPT for apples-to-apples throughput comparison."""
from __future__ import annotations

from .config import ReferenceConfig


def build_torch_gpt(cfg: ReferenceConfig):
    """Return helpers dict or None if torch is unavailable."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _Norm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            if hasattr(nn, "RMSNorm"):
                self.norm = nn.RMSNorm(dim)
            else:
                self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.norm(x)

    class ReferenceGPT(nn.Module):
        """Pre-norm causal GPT with tied token embeddings (closest practical PyTorch match)."""

        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.seq, cfg.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.heads,
                dim_feedforward=cfg.d_model * cfg.ff_mult,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.layers)
            self.ln = _Norm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
            self.lm_head.weight = self.tok_emb.weight

        def forward(self, x):
            pos = torch.arange(x.size(1), device=x.device)
            h = self.tok_emb(x) + self.pos_emb(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            h = self.blocks(h, mask=mask, is_causal=True)
            return self.lm_head(self.ln(h))

    model = ReferenceGPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device=device)
    y = torch.randint(0, cfg.vocab, (cfg.batch, cfg.seq), device=device)

    def train_step() -> None:
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def reset_vram() -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_vram_mb() -> float | None:
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return None

    return {
        "torch": torch,
        "device": device,
        "train_step": train_step,
        "sync": sync,
        "reset_vram": reset_vram,
        "peak_vram_mb": peak_vram_mb,
        "tokens": cfg.tokens_per_step,
    }
