import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dynarope.core import apply_rotary_emb, precompute_freqs_cis
from dynarope.dynamic_ntk import precompute_freqs_cis_dynamic_ntk

def run_context_extrapolation_benchmark():
    print("Initializing DynaRoPE Context Scaling Benchmark...")
    
    # 1. Architecture Parameters
    dim = 64
    max_train_len = 512
    test_seq_len = 2048
    
    # 2. Ensure assets directory exists for the output graph
    os.makedirs("assets", exist_ok=True)
    
    # 3. Initialize dummy Q and K tensors (pure 1s to isolate pure positional effects)
    # Shape: (1 batch, test_seq_len, 1 head, dim)
    q = torch.ones(1, test_seq_len, 1, dim)
    k = torch.ones(1, test_seq_len, 1, dim)
    
    # --- PHASE 1: Vanilla RoPE (The Baseline Collapse) ---
    print("Running Vanilla RoPE forward pass...")
    freqs_vanilla = precompute_freqs_cis(dim=dim, seq_len=test_seq_len)
    q_vanilla, k_vanilla = apply_rotary_emb(q, k, freqs_vanilla)
    
    # Calculate Attention Scores for Token 0 vs All Tokens
    q0_vanilla = q_vanilla[0, 0, 0, :]
    k_all_vanilla = k_vanilla[0, :, 0, :]
    attn_vanilla = torch.matmul(k_all_vanilla, q0_vanilla).numpy()
    
    # --- PHASE 2: Dynamic NTK RoPE (The SOTA Fix) ---
    print("Running Dynamic NTK RoPE forward pass...")
    freqs_ntk = precompute_freqs_cis_dynamic_ntk(
        dim=dim, 
        seq_len=test_seq_len, 
        max_train_len=max_train_len
    )
    q_ntk, k_ntk = apply_rotary_emb(q, k, freqs_ntk)
    
    # Calculate Attention Scores for Token 0 vs All Tokens
    q0_ntk = q_ntk[0, 0, 0, :]
    k_all_ntk = k_ntk[0, :, 0, :]
    attn_ntk = torch.matmul(k_all_ntk, q0_ntk).numpy()
    
    # --- PHASE 3: Visualization ---
    print("Generating extrapolation graph...")
    plt.figure(figsize=(14, 6))
    
    # Plot Vanilla
    plt.plot(attn_vanilla, label="Vanilla RoPE (OOD Failure)", color='red', alpha=0.6, linestyle='--')
    
    # Plot NTK
    plt.plot(attn_ntk, label="DynaRoPE NTK (Mathematically Scaled)", color='blue', linewidth=2)
    
    # Add structural markers
    plt.axvline(x=max_train_len, color='black', linestyle=':', linewidth=2, label=f'Training Limit ({max_train_len})')
    plt.axvspan(max_train_len, test_seq_len, color='gray', alpha=0.1, label='Untrained Zone')
    
    # Graph Formatting
    plt.title(f"Context Extrapolation Benchmark: Vanilla vs Dynamic NTK ({test_seq_len} Tokens)", fontsize=14, pad=15)
    plt.xlabel("Token Distance from Target", fontsize=12)
    plt.ylabel("Positional Attention Score (Dot Product)", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the asset for the README
    save_path = "assets/context_extrapolation.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Benchmark complete. Graph saved to {save_path}")

if __name__ == "__main__":
    run_context_extrapolation_benchmark()