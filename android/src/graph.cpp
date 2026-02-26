// graph.cpp — Transformer compute graph builder
//
// Clean forward pass graph — no intervention surfaces.
// Supports llama, qwen2, gemma3 architectures via ModelConfig flags.
// Uses fused QKV and gate+up weights when available (reduces op count).

#include "gguf-engine/graph.h"

#include <cmath>
#include <vector>

// --------------------------------------------------------------------------
// Build compute graph
// --------------------------------------------------------------------------
struct ggml_cgraph * build_graph(
    struct ggml_context * ctx,
    ModelState & state,
    int seq_len,
    int kv_pos,
    int kv_len,
    int n_layers,
    bool need_argmax
) {
    const ModelConfig & cfg = state.cfg;
    const int head_dim  = (int)cfg.head_dim;
    const int n_head    = (int)cfg.n_head;
    const int n_head_kv = (int)cfg.n_head_kv;
    const int n_embd_head = n_head * head_dim;
    const int n_ff      = (int)cfg.n_ff;
    const float base_scale = 1.0f / sqrtf((float)head_dim);
    const size_t f16_sz = ggml_type_size(GGML_TYPE_F16);

    const int q_out  = n_head * head_dim;
    const int kv_out = n_head_kv * head_dim;

    // Input tensors
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, seq_len);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    // Token embedding lookup
    struct ggml_tensor * cur = ggml_get_rows(ctx, state.token_embd, inp_tokens);

    // Gemma: scale embeddings by sqrt(n_embd)
    if (cfg.embd_scale) {
        cur = ggml_scale(ctx, cur, sqrtf((float)cfg.n_embd));
    }

    // Collect KV store ops
    std::vector<struct ggml_tensor *> kv_stores;

    // Transformer layers
    for (int il = 0; il < n_layers; il++) {
        const LayerWeights & lw = state.layers[il];
        struct ggml_tensor * residual = cur;

        // Attention pre-norm
        cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
        if (cfg.gemma_norm) {
            cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, lw.attn_norm));
        } else {
            cur = ggml_mul(ctx, cur, lw.attn_norm);
        }

        // QKV projections — fused or individual
        struct ggml_tensor * Q, * K, * V;
        if (lw.attn_qkv) {
            // Fused: single matmul → reshape to 3D → split with views
            // This avoids non-contiguous ggml_view_2d + ggml_reshape_3d
            struct ggml_tensor * qkv = ggml_mul_mat(ctx, lw.attn_qkv, cur);
            // qkv: [q_out + 2*kv_out, seq_len] → reshape to [head_dim, n_total_heads, seq_len]
            int n_total_heads = n_head + 2 * n_head_kv;
            qkv = ggml_reshape_3d(ctx, qkv, head_dim, n_total_heads, seq_len);
            // Split along head dimension using 3D views (zero cost)
            Q = ggml_view_3d(ctx, qkv, head_dim, n_head, seq_len,
                qkv->nb[1], qkv->nb[2], 0);
            K = ggml_view_3d(ctx, qkv, head_dim, n_head_kv, seq_len,
                qkv->nb[1], qkv->nb[2], (size_t)n_head * qkv->nb[1]);
            V = ggml_view_3d(ctx, qkv, head_dim, n_head_kv, seq_len,
                qkv->nb[1], qkv->nb[2], (size_t)(n_head + n_head_kv) * qkv->nb[1]);
        } else {
            Q = ggml_mul_mat(ctx, lw.attn_q, cur);
            K = ggml_mul_mat(ctx, lw.attn_k, cur);
            V = ggml_mul_mat(ctx, lw.attn_v, cur);
            Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, seq_len);
            K = ggml_reshape_3d(ctx, K, head_dim, n_head_kv, seq_len);
            V = ggml_reshape_3d(ctx, V, head_dim, n_head_kv, seq_len);
        }

        // Optional Q/K normalization (Gemma3)
        if (cfg.has_qk_norm && lw.q_norm && lw.k_norm) {
            Q = ggml_rms_norm(ctx, Q, cfg.rms_eps);
            if (cfg.gemma_norm) {
                Q = ggml_add(ctx, Q, ggml_mul(ctx, Q, lw.q_norm));
            } else {
                Q = ggml_mul(ctx, Q, lw.q_norm);
            }
            K = ggml_rms_norm(ctx, K, cfg.rms_eps);
            if (cfg.gemma_norm) {
                K = ggml_add(ctx, K, ggml_mul(ctx, K, lw.k_norm));
            } else {
                K = ggml_mul(ctx, K, lw.k_norm);
            }
        }

        // RoPE
        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr,
            head_dim, cfg.rope_type, (int)cfg.max_ctx,
            cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr,
            head_dim, cfg.rope_type, (int)cfg.max_ctx,
            cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // KV cache write
        struct ggml_tensor * K_perm = ggml_permute(ctx, K, 0, 2, 1, 3);
        struct ggml_tensor * V_perm = ggml_permute(ctx, V, 0, 2, 1, 3);

        size_t nb1 = head_dim * f16_sz;
        size_t nb2 = (size_t)cfg.max_ctx * head_dim * f16_sz;
        size_t offset = (size_t)kv_pos * head_dim * f16_sz;

        struct ggml_tensor * K_cache_view = ggml_view_3d(ctx, state.kv_k[il],
            head_dim, seq_len, n_head_kv, nb1, nb2, offset);
        struct ggml_tensor * V_cache_view = ggml_view_3d(ctx, state.kv_v[il],
            head_dim, seq_len, n_head_kv, nb1, nb2, offset);

        kv_stores.push_back(ggml_cpy(ctx, K_perm, K_cache_view));
        kv_stores.push_back(ggml_cpy(ctx, V_perm, V_cache_view));

        // KV cache read
        struct ggml_tensor * K_full = ggml_view_3d(ctx, state.kv_k[il],
            head_dim, kv_len, n_head_kv, nb1, nb2, 0);
        struct ggml_tensor * V_full = ggml_view_3d(ctx, state.kv_v[il],
            head_dim, kv_len, n_head_kv, nb1, nb2, 0);

        // Flash Attention
        struct ggml_tensor * Q_perm = ggml_permute(ctx, Q, 0, 2, 1, 3);
        struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx,
            Q_perm, K_full, V_full, attn_mask, base_scale, 0.0f, 0.0f);

        // flash_attn_ext output is a fresh contiguous tensor — no cont needed
        struct ggml_tensor * attn_merged = ggml_reshape_2d(ctx,
            attn_out, n_embd_head, seq_len);

        // Output projection
        cur = ggml_mul_mat(ctx, lw.attn_output, attn_merged);

        // Attention residual
        cur = ggml_add(ctx, cur, residual);

        // FFN
        struct ggml_tensor * ffn_residual = cur;

        cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
        if (cfg.gemma_norm) {
            cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, lw.ffn_norm));
        } else {
            cur = ggml_mul(ctx, cur, lw.ffn_norm);
        }

        // Gated FFN — use fused gate+up if available, else individual
        struct ggml_tensor * gate_proj, * up_proj;
        if (lw.ffn_gate_up) {
            // Fused: single matmul + cont views (cont needed for allocator buffer lifecycle)
            struct ggml_tensor * gu = ggml_mul_mat(ctx, lw.ffn_gate_up, cur);
            gate_proj = ggml_cont(ctx,
                ggml_view_2d(ctx, gu, n_ff, seq_len, gu->nb[1], 0));
            up_proj = ggml_cont(ctx,
                ggml_view_2d(ctx, gu, n_ff, seq_len, gu->nb[1],
                    n_ff * ggml_element_size(gu)));
        } else {
            gate_proj = ggml_mul_mat(ctx, lw.ffn_gate, cur);
            up_proj   = ggml_mul_mat(ctx, lw.ffn_up, cur);
        }

        struct ggml_tensor * gate = cfg.use_gelu
            ? ggml_gelu(ctx, gate_proj)
            : ggml_silu(ctx, gate_proj);
        struct ggml_tensor * gate_up = ggml_mul(ctx, gate, up_proj);
        cur = ggml_mul_mat(ctx, lw.ffn_down, gate_up);

        // FFN residual
        cur = ggml_add(ctx, cur, ffn_residual);
    }

    // Output head
    cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
    if (cfg.gemma_norm) {
        cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, state.output_norm));
    } else {
        cur = ggml_mul(ctx, cur, state.output_norm);
    }

    struct ggml_tensor * logits = ggml_mul_mat(ctx, state.output, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    // Build graph — KV stores FIRST for correct ordering
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2048, false);
    for (auto * kv_op : kv_stores) {
        ggml_build_forward_expand(graph, kv_op);
    }

    if (need_argmax) {
        struct ggml_tensor * token_id = ggml_argmax(ctx, logits);
        ggml_set_name(token_id, "token_id");
        ggml_set_output(token_id);
        ggml_build_forward_expand(graph, token_id);
    } else {
        ggml_build_forward_expand(graph, logits);
    }

    return graph;
}

// Compute context size needed for build_graph
size_t compute_ctx_size(int n_layers) {
    // ~44 tensors per layer + 20 global (fused QKV + gate_up, attn_out no cont)
    int n_tensors = n_layers * 44 + 20;
    return (size_t)n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(2048, false);
}
