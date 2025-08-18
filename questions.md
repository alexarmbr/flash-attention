

These questions concern the Sm90 flash attention implementation in flash_attn/cute/flash_fwd.py.

Question 1:
  In the intra_wg_overlap = True version, why does the code create smem_pipe_read_v = 
  smem_pipe_read.clone() and then immediately call smem_pipe_read.advance() before starting the gemm
  operations?

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