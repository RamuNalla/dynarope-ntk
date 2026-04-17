import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
import torch

from dynarope.attention import RoPEAttention


def test_rope_attention_forward_shape():
    d_model = 256
    n_heads = 8
    seq_len = 512
    batch_size = 2

    layer = RoPEAttention(d_model=d_model, n_heads=n_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    out = layer(x)

    assert out.shape == x.shape


def test_rope_attention_invalid_head_divisor():
    with pytest.raises(AssertionError, match="divisible"):
        RoPEAttention(d_model=256, n_heads=7)


def test_rope_attention_gradient_runs():
    d_model = 64
    n_heads = 4
    layer = RoPEAttention(d_model=d_model, n_heads=n_heads)
    x = torch.randn(1, 16, d_model, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
