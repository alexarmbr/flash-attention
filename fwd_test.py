#!/usr/bin/env python3
import math
import torch
import torch.nn.functional as F

from flash_attn.cute.interface import flash_attn_func

from torch.nn.attention import sdpa_kernel, SDPBackend

# benchmarking results
# constent with FA3 paper for non-causal
# slightly slower for causal


# (b,n,s,d) = (2,16,4096,128)
#   FlashAttention Cute TFLOPs/s: 659.17
#   PyTorch SDPA TFLOPs/s: 676.89

# intra_wg_overlap = False
#   FlashAttention Cute TFLOPs/s: 625.31
#   PyTorch SDPA TFLOPs/s: 676.62

# causal = True
#   FlashAttention Cute TFLOPs/s: 541.23
#   PyTorch SDPA TFLOPs/s: 475.05



def test_fwd():
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
    causal = True

    # Create Q/K/V in (B, S, H, D)
    with torch.inference_mode():
        q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

        # FlashAttention (Cute SM90) forward; returns (out, lse)
        out_flash, lse = flash_attn_func(q, k, v, causal=causal)

        # PyTorch SDPA expects (B, H, S, D)
        q_pt = q.permute(0, 2, 1, 3)
        k_pt = k.permute(0, 2, 1, 3)
        v_pt = v.permute(0, 2, 1, 3)
        out_pt = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=causal)
        out_pt = out_pt.permute(0, 2, 1, 3)  # back to (B, S, H, D)

        # Compare
        max_diff = (out_flash - out_pt).abs().max().item()
        mean_diff = (out_flash - out_pt).abs().mean().item()
        print(f"Max abs diff: {max_diff}")
        print(f"Mean abs diff: {mean_diff}")

        torch.testing.assert_close(out_flash, out_pt, rtol=5e-2, atol=5e-2)


def benchmark_fwd():
    assert torch.cuda.is_available(), "CUDA is required"
    major_cc = torch.cuda.get_device_capability()[0]
    assert major_cc == 9, (
        f"Expected Hopper (SM90, cc=9.x). Got cc={major_cc}. Run this on an H100/H200."
    )

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # Larger problem sizes for benchmarking
    batch_size = 1
    seqlen = 4096
    nheads = 16
    headdim = 128
    causal = True

    # FLOPs for attention forward (QK^T + P*V) ~ 4*B*H*S*S*D
    flops = 4.0 * batch_size * nheads * seqlen * seqlen * headdim
    if causal:
        flops /= 2.0

    with torch.inference_mode():
        q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

        # Prepare PyTorch SDPA inputs once
        q_pt = q.permute(0, 2, 1, 3)
        k_pt = k.permute(0, 2, 1, 3)
        v_pt = v.permute(0, 2, 1, 3)

        warmup = 5
        iters = 20

        # Warmup
        for _ in range(warmup):
            flash_attn_func(q, k, v, causal=causal)
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=causal)

        torch.cuda.synchronize()

        # Benchmark FlashAttention Cute
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        ms_total_flash = 0.0
        start.record()
        for _ in range(iters):
            _ = flash_attn_func(q, k, v, causal=causal)
        end.record()
        end.synchronize()
        ms_total_flash += start.elapsed_time(end)

        torch.cuda.synchronize()

        # Benchmark PyTorch SDPA
        ms_total_pt = 0.0
        start.record()
        for _ in range(iters):
            _ = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=causal)
        end.record()
        end.synchronize()
        ms_total_pt += start.elapsed_time(end)

        avg_ms_flash = ms_total_flash / iters
        avg_ms_pt = ms_total_pt / iters

        tflops_flash = flops / (avg_ms_flash * 1e-3) / 1e12
        tflops_pt = flops / (avg_ms_pt * 1e-3) / 1e12

        print(f"FlashAttention Cute TFLOPs/s: {tflops_flash:.2f}")
        print(f"PyTorch SDPA TFLOPs/s: {tflops_pt:.2f}")

def main():
    # test_fwd()
    benchmark_fwd()

if __name__ == "__main__":
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        main()