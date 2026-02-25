// stall.cpp — Stall generation during async tool execution

#include "gguf-engine/stall.h"
#include "gguf-engine/graph.h"
#include "gguf-engine/sampling.h"
#include "gguf-engine/utils.h"
#include "gguf-engine/tokenizer.h"

bool init_stall_kv(StallKV & skv, const ModelConfig & cfg, ggml_backend_t backend) {
    if (skv.allocated) return true;
    int n_kv_tensors = (int)cfg.n_layer * 2;
    size_t ctx_size = (size_t)n_kv_tensors * ggml_tensor_overhead() + 256;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    skv.ctx = ggml_init(params);
    if (!skv.ctx) return false;

    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        skv.kv_k[il] = ggml_new_tensor_3d(skv.ctx, GGML_TYPE_F16,
            cfg.head_dim, STALL_MAX_CTX, cfg.n_head_kv);
        ggml_format_name(skv.kv_k[il], "stall_k_%u", il);
        skv.kv_v[il] = ggml_new_tensor_3d(skv.ctx, GGML_TYPE_F16,
            cfg.head_dim, STALL_MAX_CTX, cfg.n_head_kv);
        ggml_format_name(skv.kv_v[il], "stall_v_%u", il);
    }

    skv.buf = ggml_backend_alloc_ctx_tensors(skv.ctx, backend);
    if (!skv.buf) {
        ggml_free(skv.ctx); skv.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(skv.buf, 0);
    skv.kv_pos = 0;
    skv.allocated = true;
    printf("  Stall KV: %.1f KB (%d ctx, %u layers)\n",
        ggml_backend_buffer_get_size(skv.buf) / 1024.0, STALL_MAX_CTX, cfg.n_layer);
    return true;
}

void free_stall_kv(StallKV & skv) {
    if (skv.buf) ggml_backend_buffer_free(skv.buf);
    if (skv.ctx) ggml_free(skv.ctx);
    skv.buf = nullptr; skv.ctx = nullptr; skv.allocated = false;
}

std::string generate_stall(
    ModelState & state,
    ggml_backend_t backend,
    ggml_gallocr_t stall_galloc,
    StallKV & skv,
    const std::string & stall_prompt,
    int max_stall_tokens,
    AsyncToolResult & tool_result)
{
    const ModelConfig & cfg = state.cfg;
    const int n_layer = (int)cfg.n_layer;
    const int n_vocab = (int)cfg.n_vocab;

    // Reset stall KV
    ggml_backend_buffer_clear(skv.buf, 0);
    skv.kv_pos = 0;

    // Create shallow copy of state with stall KV pointers
    ModelState stall_state = state;  // copies weight pointers (shared, read-only)
    for (int il = 0; il < n_layer; il++) {
        stall_state.kv_k[il] = skv.kv_k[il];
        stall_state.kv_v[il] = skv.kv_v[il];
    }
    stall_state.kv_pos = 0;

    // Tokenize stall prompt
    auto prompt_tokens = tokenize_simple(state.vocab, stall_prompt.c_str());
    if (prompt_tokens.empty()) return "";
    int seq_len = (int)prompt_tokens.size();
    if (seq_len > STALL_MAX_CTX - max_stall_tokens) {
        seq_len = STALL_MAX_CTX - max_stall_tokens;
        prompt_tokens.resize(seq_len);
    }

    // Allocator + buffers (stall graph has no interventions, but use safe size)
    size_t csz = compute_ctx_size(n_layer);
    std::vector<uint8_t> ctx_buf_stall(csz);
    std::vector<uint16_t> mask_stall;
    std::vector<float> logits_stall(n_vocab);
    SamplingParams stall_sp = { 0.6f, 20, 0.9f, 1.2f };
    std::mt19937 rng(12345);

    // Prefill stall prompt
    {
        struct ggml_init_params p = { csz, ctx_buf_stall.data(), true };
        struct ggml_context * gctx = ggml_init(p);
        struct ggml_cgraph * g = build_graph(gctx, stall_state, seq_len, 0, STALL_MAX_CTX, n_layer, false);
        if (!ggml_gallocr_alloc_graph(stall_galloc, g)) {
            ggml_free(gctx);
            return "";
        }

        // Set input tokens
        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_tokens"),
            prompt_tokens.data(), 0, seq_len * sizeof(int32_t));

        // Set positions
        {
            std::vector<int32_t> positions(seq_len);
            for (int i = 0; i < seq_len; i++) positions[i] = i;
            ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_pos"),
                positions.data(), 0, seq_len * sizeof(int32_t));
        }

        // Causal mask
        {
            mask_stall.resize((size_t)seq_len * STALL_MAX_CTX);
            uint16_t neg_inf = 0xFC00;
            for (int row = 0; row < seq_len; row++) {
                for (int col = 0; col < STALL_MAX_CTX; col++) {
                    mask_stall[(size_t)row * STALL_MAX_CTX + col] = (col <= row) ? 0 : neg_inf;
                }
            }
            ggml_backend_tensor_set(ggml_graph_get_tensor(g, "attn_mask"),
                mask_stall.data(), 0, mask_stall.size() * sizeof(uint16_t));
        }

        ggml_backend_graph_compute(backend, g);
        ggml_backend_synchronize(backend);

        // Read logits for last position
        struct ggml_tensor * logits = ggml_graph_get_tensor(g, "logits");
        if (logits) {
            size_t offset = (size_t)(seq_len - 1) * n_vocab * sizeof(float);
            ggml_backend_tensor_get(logits, logits_stall.data(), offset, n_vocab * sizeof(float));
        }
        ggml_free(gctx);
    }
    stall_state.kv_pos = seq_len;

    // Sample first token
    int32_t last_token = 0;
    {
        float max_val = logits_stall[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits_stall[i] > max_val) { max_val = logits_stall[i]; last_token = i; }
        }
    }

    // Decode loop — generate stall tokens, checking if tool is done
    std::string stall_text;
    for (int t = 0; t < max_stall_tokens; t++) {
        // Check if tool finished
        {
            std::lock_guard<std::mutex> lock(tool_result.mtx);
            if (tool_result.done) break;
        }

        if (stall_state.kv_pos >= STALL_MAX_CTX - 1) break;

        // Stop on special tokens
        if (last_token == state.eos_token) break;
        std::string tok_str = (last_token >= 0 && last_token < (int)state.vocab.size())
            ? state.vocab[last_token] : "";
        if (tok_str == "<|im_end|>" || tok_str == "<|im_start|>") break;

        // Print token
        std::string decoded = decode_token(tok_str);
        printf("%s", decoded.c_str());
        fflush(stdout);
        stall_text += decoded;

        // Decode one token
        int kv_pos = stall_state.kv_pos;
        struct ggml_init_params p = { csz, ctx_buf_stall.data(), true };
        struct ggml_context * gctx = ggml_init(p);
        struct ggml_cgraph * g = build_graph(gctx, stall_state, 1, kv_pos, STALL_MAX_CTX, n_layer, false);
        if (!ggml_gallocr_alloc_graph(stall_galloc, g)) {
            ggml_free(gctx); break;
        }

        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_tokens"),
            &last_token, 0, sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_pos"),
            &kv_pos, 0, sizeof(int32_t));
        {
            mask_stall.assign(STALL_MAX_CTX, 0);
            uint16_t neg_inf = 0xFC00;
            for (int col = kv_pos + 1; col < STALL_MAX_CTX; col++) mask_stall[col] = neg_inf;
            ggml_backend_tensor_set(ggml_graph_get_tensor(g, "attn_mask"),
                mask_stall.data(), 0, STALL_MAX_CTX * sizeof(uint16_t));
        }

        ggml_backend_graph_compute(backend, g);
        ggml_backend_synchronize(backend);

        struct ggml_tensor * logits = ggml_graph_get_tensor(g, "logits");
        if (logits) ggml_backend_tensor_get(logits, logits_stall.data(), 0, n_vocab * sizeof(float));
        ggml_free(gctx);

        stall_state.kv_pos = kv_pos + 1;

        // Greedy sample (stall text is just filler, keep it simple)
        last_token = 0;
        float max_val = logits_stall[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits_stall[i] > max_val) { max_val = logits_stall[i]; last_token = i; }
        }
    }

    return stall_text;
}
