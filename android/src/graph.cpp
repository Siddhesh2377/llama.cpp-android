// graph.cpp — Transformer compute graph builder
//
// Extracted from gguf-forward-test.cpp (lines 847-1126).
// Builds the full forward pass graph with optional intervention surfaces.

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
    bool need_argmax,
    const InterventionConfig * iv,
    const InterventionTensors * iv_t
) {
    const ModelConfig & cfg = state.cfg;
    const int head_dim  = (int)cfg.head_dim;
    const int n_head    = (int)cfg.n_head;
    const int n_head_kv = (int)cfg.n_head_kv;
    const int n_embd_head = n_head * head_dim;
    const int n_kv_dim    = n_head_kv * head_dim;
    const float base_scale = 1.0f / sqrtf((float)head_dim);
    const size_t f16_sz = ggml_type_size(GGML_TYPE_F16);
    const uint32_t flags = iv ? iv->flags : 0;

    // Early exit: compute fewer layers for RAG embedding extraction
    int effective_layers = n_layers;
    if ((flags & IV_EARLY_EXIT) && iv->early_exit_layer > 0
        && iv->early_exit_layer < n_layers) {
        effective_layers = iv->early_exit_layer;
    }

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
    for (int il = 0; il < effective_layers; il++) {
        const LayerWeights & lw = state.layers[il];
        struct ggml_tensor * residual = cur;

        // === Attention pre-norm ===
        cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
        if (cfg.gemma_norm) {
            cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, lw.attn_norm));
        } else {
            cur = ggml_mul(ctx, cur, lw.attn_norm);
        }

        // --- INJECTION POINT 2: NormShift (attn side) ---
        if ((flags & IV_NORM_SHIFT) && iv_t && iv_t->norm_shift[il]) {
            cur = ggml_add(ctx, cur, iv_t->norm_shift[il]);
        }

        // === QKV projections ===
        struct ggml_tensor * Q = ggml_mul_mat(ctx, lw.attn_q, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, lw.attn_k, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, lw.attn_v, cur);

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, seq_len);
        K = ggml_reshape_3d(ctx, K, head_dim, n_head_kv, seq_len);
        V = ggml_reshape_3d(ctx, V, head_dim, n_head_kv, seq_len);

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

        // === KV cache write ===
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

        // === KV cache read ===
        struct ggml_tensor * K_full = ggml_view_3d(ctx, state.kv_k[il],
            head_dim, kv_len, n_head_kv, nb1, nb2, 0);
        struct ggml_tensor * V_full = ggml_view_3d(ctx, state.kv_v[il],
            head_dim, kv_len, n_head_kv, nb1, nb2, 0);

        // === Flash Attention ===
        struct ggml_tensor * Q_perm = ggml_permute(ctx, Q, 0, 2, 1, 3);

        // --- INJECTION POINT 3: Attention Temperature ---
        // Modifies the scale param directly — zero overhead (compile-time literal)
        float scale = base_scale;
        if ((flags & IV_ATTN_TEMPERATURE) && iv->attn_temp[il] != 0.0f
            && iv->attn_temp[il] != 1.0f) {
            scale /= iv->attn_temp[il];
        }

        struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx,
            Q_perm, K_full, V_full, attn_mask, scale, 0.0f, 0.0f);

        // Reshape to [n_embd_head, seq_len]
        struct ggml_tensor * attn_merged = ggml_reshape_2d(ctx,
            ggml_cont(ctx, attn_out), n_embd_head, seq_len);

        // --- INJECTION POINT 4: Head Rescaling ---
        // Per-head scalar on attn output BEFORE output projection
        // head_scale[il] is [n_embd_head] where each head_dim block = same scalar
        if ((flags & IV_HEAD_RESCALE) && iv_t && iv_t->head_scale[il]) {
            attn_merged = ggml_mul(ctx, attn_merged, iv_t->head_scale[il]);
        }

        // Output projection
        cur = ggml_mul_mat(ctx, lw.attn_output, attn_merged);

        // --- INJECTION POINT 5: Attention Gated Residual ---
        if ((flags & IV_GATED_RESIDUAL) && iv->attn_gate[il] != 0.0f
            && iv->attn_gate[il] != 1.0f) {
            cur = ggml_scale(ctx, cur, iv->attn_gate[il]);
        }

        // Attention residual
        cur = ggml_add(ctx, cur, residual);

        // === FFN ===
        struct ggml_tensor * ffn_residual = cur;

        cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
        if (cfg.gemma_norm) {
            cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, lw.ffn_norm));
        } else {
            cur = ggml_mul(ctx, cur, lw.ffn_norm);
        }

        // --- INJECTION POINT 6: NormShift (FFN side) ---
        if ((flags & IV_NORM_SHIFT) && iv_t && iv_t->norm_shift[il]) {
            cur = ggml_add(ctx, cur, iv_t->norm_shift[il]);
        }

        // Gated FFN
        struct ggml_tensor * gate_proj = ggml_mul_mat(ctx, lw.ffn_gate, cur);
        struct ggml_tensor * gate = cfg.use_gelu
            ? ggml_gelu(ctx, gate_proj)
            : ggml_silu(ctx, gate_proj);
        struct ggml_tensor * up = ggml_mul_mat(ctx, lw.ffn_up, cur);
        struct ggml_tensor * gate_up = ggml_mul(ctx, gate, up);
        cur = ggml_mul_mat(ctx, lw.ffn_down, gate_up);

        // --- INJECTION POINT 7a: FFN Gated Residual ---
        if ((flags & IV_GATED_RESIDUAL) && iv->ffn_gate[il] != 0.0f
            && iv->ffn_gate[il] != 1.0f) {
            cur = ggml_scale(ctx, cur, iv->ffn_gate[il]);
        }

        // --- INJECTION POINT 7b: Control Vectors (post-FFN, before residual) ---
        if ((flags & IV_CONTROL_VECTORS) && iv_t && iv_t->control_vector[il]) {
            cur = ggml_add(ctx, cur, iv_t->control_vector[il]);
        }

        // FFN residual
        cur = ggml_add(ctx, cur, ffn_residual);

        // --- Activation Capture ---
        if (flags & IV_CAPTURE) {
            for (int ci = 0; ci < iv->n_capture; ci++) {
                if (iv->capture_layers[ci] == il) {
                    char name[32];
                    snprintf(name, sizeof(name), "capture_%d", il);
                    ggml_set_name(cur, name);
                    ggml_set_output(cur);
                    break;
                }
            }
        }
    }

    // === Early exit for RAG embedding extraction ===
    if ((flags & IV_EARLY_EXIT) && effective_layers < n_layers) {
        cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
        if (cfg.gemma_norm) {
            cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, state.output_norm));
        } else {
            cur = ggml_mul(ctx, cur, state.output_norm);
        }
        ggml_set_name(cur, "embedding");
        ggml_set_output(cur);

        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2048, false);
        for (auto * kv_op : kv_stores) {
            ggml_build_forward_expand(graph, kv_op);
        }
        ggml_build_forward_expand(graph, cur);
        return graph;
    }

    // === Output head ===
    cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
    if (cfg.gemma_norm) {
        cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, state.output_norm));
    } else {
        cur = ggml_mul(ctx, cur, state.output_norm);
    }

    struct ggml_tensor * logits = ggml_mul_mat(ctx, state.output, cur);

    // --- INJECTION POINT 8: Logit Bias ---
    if ((flags & IV_LOGIT_BIAS) && iv_t && iv_t->logit_bias) {
        logits = ggml_add(ctx, logits, iv_t->logit_bias);
    }

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
size_t compute_ctx_size(int n_layers, bool with_interventions) {
    // ~42 tensors per layer (base) + 7 extra for interventions + 20 global + 1 logit_bias
    int extra_per_layer = with_interventions ? 7 : 0;
    int extra_global = with_interventions ? 1 : 0;
    int n_tensors = n_layers * (42 + extra_per_layer) + 20 + extra_global;
    return (size_t)n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(2048, false);
}
