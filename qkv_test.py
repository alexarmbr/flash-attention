#!/usr/bin/env python3
import math
import torch
import torch.nn.functional as F

from flash_attn.cute.interface import flash_attn_func, qkv_func

def test_qkv():
    assert torch.cuda.is_available(), "CUDA is required"
    major_cc = torch.cuda.get_device_capability()[0]
    assert major_cc == 9, (
        f"Expected Hopper (SM90, cc=9.x). Got cc={major_cc}. Run this on an H100/H200."
    )

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # Problem sizes (bf16, non-causal)
    batch_size = 2
    seqlen = 4096
    nheads = 6
    headdim = 32  # multiple of 8, <= 256

    # Create Q/K/V in (B,H, S, D)
    with torch.inference_mode():
        # q = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype)
        # k = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype)
        # v = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype)
        q = torch.ones(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype) * 3
        k = torch.ones(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype) * 3
        v = torch.ones(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype) * 3
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = qkv_func(q,k,v)

        # sychronize and check for cuda errors
        torch.cuda.synchronize()

def main():
    test_qkv()

if __name__ == "__main__":
    main()