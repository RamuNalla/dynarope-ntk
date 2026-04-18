import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import precompute_freqs_cis, apply_rotary_emb
from .dynamic_ntk import precompute_freqs_cis_dynamic_ntk

class RoPEAttention(nn.Module):
    """
    A custom Causal Self-Attention layer implementing Vanilla RoPE.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # Ensure dimensions divide cleanly
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_train_len = max_train_len

        # 1. The standard Linear Projections
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len, d_model)
        bsz, seq_len, _ = x.shape

        # 1. Project to Q, K, V and split into multiple attention heads
        # Output shape: (bsz, seq_len, n_heads, head_dim)
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim)

        # 2. Precompute the Vanilla RoPE frequencies for the current sequence length
        freqs_cis = precompute_freqs_cis_dynamic_ntk(
            dim=self.head_dim, 
            seq_len=seq_len, 
            max_train_len=self.max_train_len # e.g., 512
        )

        # 3. Inject Position: Rotate Q and K in the complex plane
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 4. Transpose for PyTorch's optimized attention kernel
        # Shape becomes: (bsz, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. Scaled Dot-Product Attention (with hardware-accelerated FlashAttention if available)
        # is_causal=True ensures tokens cannot look into the future
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 6. Recombine heads and project out
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.wo(output)
