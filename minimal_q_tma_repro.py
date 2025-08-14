#!/usr/bin/env python3
import math
import torch
import os

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils_basic

import cuda.bindings.driver as cuda

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

class TmaCopySm90:
    arch = 90

    def __init__(self, head_dim, dtype):
        self.head_dim = head_dim
        self.dtype = dtype
        self.m_block_size = 128
        self.num_stages = 1

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        cuda_stream: cuda.CUstream,
    ):
        self.num_threads = 256
        
        # permute mQ from (B,H,N,D) to (N,D,B,H)
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[2,3,0,1]))
        
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim
            ),
            self.dtype,
        )
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.m_block_size, self.head_dim), (0, 1)
        )

        alignment = 128
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], alignment
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]

        @cute.struct
        class SharedStorage:
            sQ: sQ_struct
            mbar_ptr_Q: mbar_ptr_Q_struct

        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        self.tma_copy_q_bytes = cute.size_in_bytes(
            mQ.element_type, cute.select(self.sQ_layout, mode=[0, 1])
        )
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q, mQ, cute.select(self.sQ_layout, mode=[0, 1]), (self.m_block_size, self.head_dim)
        )

        num_blocks_m = cute.ceil_div(cute.size(mQ.shape[2]), self.m_block_size)  # seqlen dimension
        num_heads = cute.size(mQ.shape[1])
        num_batches = cute.size(mQ.shape[3])
        
        # Grid: (num_blocks_m, num_heads, num_batches)
        grid_dim = (num_blocks_m, num_heads, num_batches)
        block_dim = (self.num_threads, 1, 1)
        smem_size = SharedStorage.size_in_bytes()

        self.kernel(
            tma_tensor_Q,
            tma_atom_Q,
            self.sQ_layout,
            SharedStorage,

        ).launch(grid=grid_dim, block=block_dim, smem=smem_size, stream=cuda_stream)

    @cute.kernel
    def kernel(
        self,
        tma_tensor_Q: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        SharedStorage: cutlass.Constexpr,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_producer = warp_idx < 4

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
        cute.arch.sync_threads()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)

        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar_ptr_Q, cnt=1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        ####
        # producer
        ####
        
        if is_producer:
            warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

            if warp_idx_in_wg == 0:
                m_block = bidx
                head_idx = bidy
                batch_idx = bidz

                # Extract the 2D slice for this batch and head
                # block_tma_Q = tma_tensor_Q[None, None, head_idx, batch_idx]
                block_tma_Q = tma_tensor_Q[None, None, batch_idx, head_idx]
                tiled_tma_Q = cute.local_tile(
                    block_tma_Q, tiler=(self.m_block_size, self.head_dim), coord=(m_block, 0)
                )
                # Group modes as in reference implementation
                sQ_grouped = cute.group_modes(sQ, 0, 2)
                tiled_tma_Q_grouped = cute.group_modes(tiled_tma_Q, 0, 2)
                    
                tQsQ, tQgQ = cpasync.tma_partition(
                    atom=tma_atom_Q,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=sQ_grouped,
                    gmem_tensor=tiled_tma_Q_grouped,
                )

                if bidx == 0 and bidy == 0 and bidz == 0 and tidx == 0:
                    cute.printf("gQ: {}", tiled_tma_Q.layout)
                    cute.printf("gQ_grouped: {}", tiled_tma_Q_grouped.layout)
                    cute.printf("sQ: {}", sQ.layout)
                    cute.printf("sQ_grouped: {}", sQ_grouped.layout)
                    cute.printf("tQgQ: {}", tQgQ.layout)
                
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr_Q, self.tma_copy_q_bytes)
                cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=mbar_ptr_Q)

                if bidx == 0 and bidy == 0 and bidz == 0 and tidx == 0:
                    cute.printf("PRODUCER: copy launched")
                    # cute.printf("sQ grouped: {}", sQ_grouped)
                    # cute.printf("tiled_tma_Q_grouped: {}", tiled_tma_Q_grouped)
                    # cute.printf("tQsQ: {}", tQsQ)
                    # cute.printf("tQgQ: {}", tQgQ)
        else:
            # Wait for phase 0 (matches original usage)
            cute.arch.mbarrier_wait(mbar_ptr_Q, 0)
            if tidx == 132 and bidx == 0 and bidy == 0 and bidz == 0:
                cute.printf("CONSUMER: Q loaded")
                cute.print_tensor(sQ)


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    major_cc = torch.cuda.get_device_capability()[0]
    assert major_cc == 9, (
        f"Expected Hopper (SM90, cc=9.x). Got cc={major_cc}. Run this on an H100/H200."
    )

    torch.manual_seed(0)
    device = "cuda"
    torch_dtype = torch.float16

    batch_size = 2
    nheads = 8
    seqlen = 4096
    headdim = 32  # multiple of 8, <= 256

    cuda_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)


    def check_alignment(tensor, alignment):
        ptr = tensor.data_ptr()
        return ptr % alignment == 0

    with torch.inference_mode():
        q = torch.ones(batch_size, nheads, seqlen, headdim, device=device, dtype=torch_dtype) * 3
        q = q.contiguous()
        
        q_tensor = from_dlpack(q.detach(), assumed_align=16)
        cute_dtype = cutlass.Float16
        
        kernel = TmaCopySm90(headdim, cute_dtype)
        kernel = cute.compile(kernel, q_tensor, cuda_stream)
        kernel(q_tensor, cuda_stream)

        torch.cuda.synchronize()


if __name__ == "__main__":
    main()

