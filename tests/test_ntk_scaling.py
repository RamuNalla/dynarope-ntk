import sys
from pathlib import Path

# Package lives under src/; put that dir on path so `dynarope` imports work.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import pytest
from dynarope.core import precompute_freqs_cis
from dynarope.dynamic_ntk import precompute_freqs_cis_dynamic_ntk

def test_ntk_safe_zone_bypass():
    """
    Ensures that if inference length is <= training length, 
    NTK scaling is bypassed and Vanilla frequencies are returned.
    """
    dim = 64
    max_train_len = 512
    test_len = 256  # Inside the safe zone
    
    freqs_vanilla = precompute_freqs_cis(dim, test_len)
    freqs_ntk = precompute_freqs_cis_dynamic_ntk(dim, test_len, max_train_len)
    
    assert torch.allclose(freqs_vanilla, freqs_ntk), "NTK triggered prematurely inside the safe zone."

def test_ntk_dynamic_compression():
    """
    Ensures that if inference length > training length,
    the low frequencies are compressed while high frequencies are preserved.
    """
    dim = 64
    max_train_len = 512
    test_len = 1024  # Triggering a 2x scale
    
    freqs_vanilla = precompute_freqs_cis(dim, test_len)
    freqs_ntk = precompute_freqs_cis_dynamic_ntk(dim, test_len, max_train_len)
    
    # 1. High Frequencies (Local Context) should remain roughly identical
    # We check the first dimension pair (index 0)
    assert torch.allclose(freqs_vanilla[:, 0], freqs_ntk[:, 0], atol=1e-5), "NTK corrupted high-frequency local grammar."
    
    # 2. Low Frequencies (Global Context) MUST diverge and compress
    # We check the last dimension pair
    assert not torch.allclose(freqs_vanilla[:, -1], freqs_ntk[:, -1]), "NTK failed to compress low frequencies."
    
    # 3. Mathematical proof of the scale factor
    # At exactly 2x scale, the lowest frequency in NTK should be roughly half of Vanilla
    vanilla_low_freq = freqs_vanilla[1, -1].angle()  # Get the rotation angle for step 1
    ntk_low_freq = freqs_ntk[1, -1].angle()
    
    compression_ratio = ntk_low_freq / vanilla_low_freq
    assert 0.45 < compression_ratio < 0.55, f"Expected ~0.5x compression, got {compression_ratio:.2f}"