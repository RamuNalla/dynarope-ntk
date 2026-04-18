import os
import sys
from pathlib import Path
import torch
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dynarope.core import apply_rotary_emb, precompute_freqs_cis
from dynarope.dynamic_ntk import precompute_freqs_cis_dynamic_ntk

def generate_attention_heatmaps():
    print("Generating 2D Positional Attention Heatmaps...")
    
    dim = 64
    max_train_len = 512
    test_seq_len = 2048
    os.makedirs("assets", exist_ok=True)
    
    # FIX 1: Use the exact same base tensor for Q and K.
    # Setting them to pure 1s guarantees that ANY change in the dot product
    # is 100% caused by the RoPE rotational mathematics.
    base_tensor = torch.ones(1, test_seq_len, 1, dim)
    q = base_tensor.clone()
    k = base_tensor.clone()
    
    # Create Causal Mask
    mask = torch.tril(torch.ones(test_seq_len, test_seq_len)).view(1, test_seq_len, test_seq_len)
    
    def compute_raw_attention_matrix(q_rot, k_rot):
        # Compute Q * K^T
        attn_scores = torch.einsum('bqhd,bkhd->bqk', q_rot, k_rot)
        # Scale by sqrt(dim)
        attn_scores = attn_scores / math.sqrt(dim)
        # FIX 2: Do NOT apply softmax. We want to see the raw mathematical waves.
        # Set future tokens to NaN so they appear transparent/white in the plot
        attn_scores = attn_scores.masked_fill(mask == 0, float('nan'))
        return attn_scores[0].numpy()

    # --- PHASE 1: Vanilla Failure ---
    freqs_vanilla = precompute_freqs_cis(dim=dim, seq_len=test_seq_len)
    q_v, k_v = apply_rotary_emb(q, k, freqs_vanilla)
    matrix_vanilla = compute_raw_attention_matrix(q_v, k_v)
    
    # --- PHASE 2: DynaRoPE NTK Success ---
    freqs_ntk = precompute_freqs_cis_dynamic_ntk(dim=dim, seq_len=test_seq_len, max_train_len=max_train_len)
    q_n, k_n = apply_rotary_emb(q, k, freqs_ntk)
    matrix_ntk = compute_raw_attention_matrix(q_n, k_n)
    
    # --- PHASE 3: Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Note: Max possible score is dim / sqrt(dim) = 64 / 8 = 8.0
    cmap = 'viridis' # Viridis handles NaNs gracefully and shows wave intensity beautifully
    
    # Plot 1: Vanilla
    im1 = axes[0].imshow(matrix_vanilla, cmap=cmap, vmin=-2, vmax=8)
    axes[0].set_title("Vanilla RoPE: Attention Entropy (OOD Collapse)", fontsize=14)
    axes[0].set_xlabel("Key Position (Attending To)", fontsize=12)
    axes[0].set_ylabel("Query Position (Current Token)", fontsize=12)
    axes[0].axhline(y=max_train_len, color='red', linestyle='--', linewidth=2, label='Training Limit (512)')
    axes[0].axvline(x=max_train_len, color='red', linestyle='--', linewidth=2)
    axes[0].legend(loc='upper right')
    
    # Plot 2: NTK
    im2 = axes[1].imshow(matrix_ntk, cmap=cmap, vmin=-2, vmax=8)
    axes[1].set_title("DynaRoPE NTK: Preserved Geometry", fontsize=14)
    axes[1].set_xlabel("Key Position (Attending To)", fontsize=12)
    axes[1].axhline(y=max_train_len, color='red', linestyle='--', linewidth=2, label='Training Limit (512)')
    axes[1].axvline(x=max_train_len, color='red', linestyle='--', linewidth=2)
    axes[1].legend(loc='upper right')
    
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('Raw Positional Dot Product', rotation=270, labelpad=15)
    
    plt.savefig("assets/attention_heatmap_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved true geometric heatmaps to assets/attention_heatmap_comparison.png")

if __name__ == "__main__":
    generate_attention_heatmaps()