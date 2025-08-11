#!/usr/bin/env python3
import math
import torch
import torch.nn.functional as F

from flash_attn.cute.interface import flash_attn_func

def main():
    assert torch.cuda.is_available(), "CUDA is required"
    major_cc = torch.cuda.get_device_capability()[0]
    assert major_cc == 9, f"Expected Hopper (SM90, cc=9.x). Got cc={major_cc}. Run this on an H100/H200."

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # Problem sizes (bf16, non-causal)
    batch_size = 2
    seqlen = 128
    nheads = 6
    headdim = 128  # multiple of 8, <= 256

    # Create Q/K/V in (B, S, H, D)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    # FlashAttention (Cute SM90) forward; returns (out, lse)
    # softmax_scale defaults to 1/sqrt(D); window_size=(None, None) => non-local; causal=False
    out_flash, lse = flash_attn_func(q, k, v, causal=False)

    # PyTorch SDPA expects (B, H, S, D)
    q_pt = q.permute(0, 2, 1, 3)
    k_pt = k.permute(0, 2, 1, 3)
    v_pt = v.permute(0, 2, 1, 3)
    out_pt = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=False)
    out_pt = out_pt.permute(0, 2, 1, 3)  # back to (B, S, H, D)

    # Compare
    max_diff = (out_flash - out_pt).abs().max().item()
    mean_diff = (out_flash - out_pt).abs().mean().item()
    print(f"Max abs diff: {max_diff}")
    print(f"Mean abs diff: {mean_diff}")

if __name__ == "__main__":
    main()