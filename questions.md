

These questions concern the Sm90 flash attention implementation in flash_attn/cute/flash_fwd.py.

## Question 1:
**In the intra_wg_overlap = True version, why does the code create smem_pipe_read_v = 
  smem_pipe_read.clone() and then immediately call smem_pipe_read.advance() before starting the gemm
  operations?**

each iteration in the `intra_wg_overlap = True` version, we are performing the Q * K^T matmul for tile `i+1` in the same loop iteration as the P * V matmul for tile `i`, i.e. the Q * K^T matmul is operating one `n` tile ahead of the P * V matmul. In order to track the `n` tile for each of these matmuls, we need two pipeline states, with the one for the Q * K^T matmul (`smem_pipe_read`) ahead of the one P * V matmul (`smem_pipe_read_v`). The rough order of operations is:
- start the Q * K^T gemm for tile `i+1`
- start the P * V gemm for tile `i`
- wait for the Q * K^T gemm to finish (P * V is still in flight)
- scoremod, mask, softmax the Q * K^T gemm result
- wait for P * V to finish
so the lower throughput instructions of the softmax are happening concurrently with the P*V gemm.


Question 2:
  Both versions call softmax.rescale_O(mma_params.acc_O, row_scale). In the overlapped version, this
   happens after warpgroup.wait_group(0) at line 2370, but in the non-overlapped version it happens
  immediately after the softmax computation. True or False: This timing difference is critical for
  correctness because the overlapped version needs to ensure the PV matmul has completed before
  rescaling the output accumulator.

Question 3:
  Looking at the warpgroup wait patterns, in the overlapped version there are calls to
  warpgroup.wait_group(1) and warpgroup.wait_group(0). What specific operations do these two wait
  calls synchronize, and why is the order wait_group(1) then wait_group(0) important?

Question 4:
    how is the swizzle value determined in the SingleTileLPTScheduler? What does the swizzling do, and why is it especially important for the causal implementation?