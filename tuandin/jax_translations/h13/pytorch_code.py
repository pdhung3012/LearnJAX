"""Flash Attention v2 forward kernel — Triton kernel that tiles over Q/K
and uses the online (logsumexp) softmax trick to fuse softmax + matmul.

Source: TorchLeet llm/flash-attention.ipynb.

Note: this kernel runs on GPU only (Triton requires CUDA / HIP). The grid is
launched with (T_q, batch_size).
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr,
                     O_ptr, L_ptr,
                     stride_qb, stride_qq, stride_qd,
                     stride_kb, stride_kk, stride_kd,
                     stride_vb, stride_vk, stride_vd,
                     stride_ob, stride_ok, stride_od,
                     stride_lb, stride_lq,
                     N_q, N_k,
                     scale,
                     D: tl.constexpr,
                     BLOCK_SIZE_Q: tl.constexpr,
                     BLOCK_SIZE_K: tl.constexpr):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
                                    shape=(N_q, D), strides=(stride_qq, stride_qd),
                                    offsets=(query_tile_index * BLOCK_SIZE_Q, 0),
                                    block_shape=(BLOCK_SIZE_Q, D), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb,
                                    shape=(D, N_k), strides=(stride_kd, stride_kk),
                                    offsets=(0, 0),
                                    block_shape=(D, BLOCK_SIZE_K), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb,
                                    shape=(N_k, D), strides=(stride_vk, stride_vd),
                                    offsets=(0, 0),
                                    block_shape=(BLOCK_SIZE_K, D), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob,
                                    shape=(N_q, D), strides=(stride_ok, stride_od),
                                    offsets=(query_tile_index * BLOCK_SIZE_Q, 0),
                                    block_shape=(BLOCK_SIZE_Q, D), order=(1, 0))
    L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb,
                                    shape=(N_q,), strides=(stride_lq,),
                                    offsets=(query_tile_index * BLOCK_SIZE_Q,),
                                    block_shape=(BLOCK_SIZE_Q,), order=(0,))

    l = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    out = tl.zeros([BLOCK_SIZE_Q, D], dtype=tl.float32)
    prev_max = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    q = tl.load(Q_block_ptr).to(tl.float32)
    for _ in range(0, N_k, BLOCK_SIZE_K):
        k = tl.load(K_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)

        s = tl.dot(q, k) * scale
        curr_max = tl.maximum(prev_max, tl.max(s, axis=1))
        p = tl.math.exp(s - curr_max[:, None])

        alpha = tl.math.exp(prev_max - curr_max)
        out = out * alpha[:, None] + tl.dot(p, v)

        l = l * alpha + tl.sum(p, axis=1)
        prev_max = curr_max

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_K))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_K, 0))

    out = out / l[:, None]
    tl.store(O_block_ptr, out.to(O_ptr.dtype.element_ty))
    tl.store(L_block_ptr, (prev_max + tl.log(l)).to(L_ptr.dtype.element_ty))


def flash_attention_forward(Q, K, V, BLOCK_SIZE_Q=16, BLOCK_SIZE_K=16):
    """Wrapper that launches the Triton kernel. Q/K/V: (B, S, D)."""
    B, N_q, D = Q.shape
    N_k = K.shape[1]
    O = torch.empty_like(Q)
    L = torch.empty((B, N_q), dtype=torch.float32, device=Q.device)
    grid = (triton.cdiv(N_q, BLOCK_SIZE_Q), B)
    flash_fwd_kernel[grid](
        Q, K, V, O, L,
        *Q.stride(), *K.stride(), *V.stride(), *O.stride(), *L.stride(),
        N_q, N_k, 1.0 / math.sqrt(D),
        D=D, BLOCK_SIZE_Q=BLOCK_SIZE_Q, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return O, L


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Triton flash attention requires a CUDA GPU. Skipping.")
        raise SystemExit(0)
    B, N_q, N_k, D = 1, 64, 128, 256
    Q = torch.randn((B, N_q, D), dtype=torch.float16, device="cuda")
    K = torch.randn((B, N_k, D), dtype=torch.float16, device="cuda")
    V = torch.randn((B, N_k, D), dtype=torch.float16, device="cuda")
    O, L = flash_attention_forward(Q, K, V)
    # Reference using vanilla attention.
    scale = 1.0 / math.sqrt(D)
    scores = (Q.float() @ K.float().transpose(-2, -1)) * scale
    O_ref = (torch.softmax(scores, dim=-1) @ V.float())
    print("max abs diff:", (O.float() - O_ref).abs().max().item())
