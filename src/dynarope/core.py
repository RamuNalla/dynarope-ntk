import torch

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the complex exponential rotation frequencies for the given dimensions.
    
    Args:
        dim: The dimension of the attention head (must be even).
        seq_len: The maximum sequence length.
        theta: The base frequency for the rotation (standard is 10000.0).
        
    Returns:
        Complex tensor of shape (seq_len, dim // 2).
    """
    # 1. Calculate the frequency inverse: 1 / (theta ^ (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. Create the position index tensor: [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    
    # 3. Calculate outer product to get angles for every position & dimension pair
    freqs = torch.outer(t, freqs).float()
    
    # 4. Convert to complex numbers in polar form: cos(freqs) + i * sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Physically rotates the Query and Key vectors in the complex plane.
    
    Args:
        xq: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, num_heads, head_dim)
        freqs_cis: Precomputed complex frequencies
    """
    # 1. Reshape the last dimension into pairs and cast to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Reshape freqs_cis to broadcast correctly across batch and heads
    ndim = xq_.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    
    # 3. Apply the rotation by multiplying the complex numbers
    # view_as_real splits the complex number back into pairs, flatten restores the original head_dim
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # 4. Cast back to the original dtype (crucial for mixed precision / FP16 training)
    return xq_out.type_as(xq), xk_out.type_as(xk)