import torch

def precompute_freqs_cis_dynamic_ntk(
    dim: int, 
    seq_len: int, 
    max_train_len: int = 512, 
    base: float = 10000.0
) -> torch.Tensor:
    """
    Precomputes complex exponential frequencies using Dynamic NTK-Aware Interpolation.
    
    Args:
        dim: Dimension of the attention head.
        seq_len: The target inference sequence length (e.g., 1024).
        max_train_len: The context length the model was originally trained on (e.g., 512).
        base: The original base frequency (theta).
    """
    # Dynamic Trigger: Only scale if the sequence exceeds training length
    if seq_len > max_train_len:
        # Calculate the scaling factor (alpha)
        scale = seq_len / max_train_len
        
        # The NTK Math: Scale the base frequency based on the dimension
        # theta' = theta * (scale ^ (dim / (dim - 2)))
        theta = base * (scale ** (dim / (dim - 2)))
    else:
        theta = base
        
    # Standard RoPE calculation using the new dynamically scaled theta
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    
    return torch.polar(torch.ones_like(freqs), freqs)