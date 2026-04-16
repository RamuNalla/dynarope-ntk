import sys
from pathlib import Path

# Package lives under src/; put that dir on path so `dynarope` imports work.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import pytest
from dynarope.core import precompute_freqs_cis, apply_rotary_emb

def test_rotation_shapes():
    """Ensure tensor dimensions remain intact after complex rotation."""
    dim = 64
    seq_len = 128
    batch = 2
    heads = 4
    
    q = torch.randn(batch, seq_len, heads, dim)
    k = torch.randn(batch, seq_len, heads, dim)
    
    freqs = precompute_freqs_cis(dim, seq_len)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    
    assert q.shape == q_rot.shape, "Query shape mutated during rotation."
    assert k.shape == k_rot.shape, "Key shape mutated during rotation."

def test_norm_preservation():
    """
    Mathematical Invariant: RoPE is a pure rotational transformation.
    It must preserve the L2 norm (magnitude) of the embedding vectors.
    """
    dim = 64
    seq_len = 128
    
    q = torch.randn(1, seq_len, 1, dim)
    k = torch.randn(1, seq_len, 1, dim)
    
    freqs = precompute_freqs_cis(dim, seq_len)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    
    # Calculate L2 norms along the embedding dimension
    q_norm_orig = torch.linalg.vector_norm(q, dim=-1)
    q_norm_rot = torch.linalg.vector_norm(q_rot, dim=-1)
    
    # Check if the magnitudes are identical (allowing for minor floating point variance)
    assert torch.allclose(q_norm_orig, q_norm_rot, atol=1e-5), "Vector magnitude changed! RoPE should only rotate."

def test_position_zero_identity():
    """
    Mathematical Invariant: At absolute position 0, the rotation angle is 0.
    The embedding for the first token should remain strictly unchanged.
    """
    dim = 64
    seq_len = 128
    
    q = torch.randn(1, seq_len, 1, dim)
    k = torch.randn(1, seq_len, 1, dim)
    
    freqs = precompute_freqs_cis(dim, seq_len)
    q_rot, _ = apply_rotary_emb(q, k, freqs)
    
    # Check if token at index 0 (the first token in the sequence) is identical
    assert torch.allclose(q[:, 0, :, :], q_rot[:, 0, :, :], atol=1e-5), "Position 0 was mutated."