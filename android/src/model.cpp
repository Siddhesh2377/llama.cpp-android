// model.cpp — GGUF model loading and KV cache initialization
//
// Extracted from gguf-forward-test.cpp (lines 530-845).
// Functions: load_model(), init_kv_cache()

#include "gguf-engine/model.h"
#include "gguf-engine/utils.h"

#include <cstdio>
#include <algorithm>

// --------------------------------------------------------------------------
// Step 1: Load model
// --------------------------------------------------------------------------
bool load_model(ModelState & state, const char * path, int fd, ggml_backend_t backend) {
    printf("\n=== Loading Model ===\n");
    auto t0 = Clock::now();

    // Parse GGUF
    struct ggml_context * data_ctx = nullptr;
    struct gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &data_ctx };

    struct gguf_context * gctx = nullptr;
    if (fd >= 0) {
        printf("  Loading from fd=%d (Android SAF)\n", fd);
        gctx = gguf_init_from_fd(fd, params);
    } else {
        printf("  Loading from path: %s\n", path);
        gctx = gguf_init_from_file(path, params);
    }

    if (!gctx) {
        printf("  FAIL: gguf_init failed\n");
        return false;
    }

    state.gguf_ctx = gctx;
    state.data_ctx = data_ctx;

    auto t1 = Clock::now();
    printf("  GGUF load: %.0f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    printf("  Tensors: %lld  KV pairs: %lld\n",
        (long long)gguf_get_n_tensors(gctx), (long long)gguf_get_n_kv(gctx));

    // Read architecture
    const char * arch = gguf_get_str_val(gctx, "general.architecture", "qwen2");
    snprintf(state.cfg.arch, sizeof(state.cfg.arch), "%s", arch);

    // Read hyperparameters
    ModelConfig & cfg = state.cfg;
    std::string a(arch);
    cfg.n_embd      = gguf_get_u32(gctx, (a + ".embedding_length").c_str(), 0);
    cfg.n_head       = gguf_get_u32(gctx, (a + ".attention.head_count").c_str(), 0);
    cfg.n_head_kv    = gguf_get_u32(gctx, (a + ".attention.head_count_kv").c_str(), cfg.n_head);
    cfg.n_layer      = gguf_get_u32(gctx, (a + ".block_count").c_str(), 0);
    cfg.n_ff         = gguf_get_u32(gctx, (a + ".feed_forward_length").c_str(), 0);
    cfg.n_vocab      = gguf_get_u32(gctx, (a + ".vocab_size").c_str(), 0);
    cfg.rms_eps      = gguf_get_f32_val(gctx, (a + ".attention.layer_norm_rms_epsilon").c_str(), 1e-6f);
    cfg.rope_freq_base = gguf_get_f32_val(gctx, (a + ".rope.freq_base").c_str(), 1000000.0f);
    // head_dim: prefer GGUF key, fallback to deriving from Q weight shape
    cfg.head_dim     = gguf_get_u32(gctx, (a + ".attention.key_length").c_str(), 0);
    if (cfg.head_dim == 0) {
        cfg.head_dim = cfg.n_head > 0 ? cfg.n_embd / cfg.n_head : 0;
    }
    cfg.max_ctx      = MAX_CTX;

    printf("  Arch: %s  n_embd=%u  n_head=%u(kv:%u)  n_layer=%u  n_ff=%u  n_vocab=%u\n",
        arch, cfg.n_embd, cfg.n_head, cfg.n_head_kv, cfg.n_layer, cfg.n_ff, cfg.n_vocab);
    // Detect architecture type and set flags
    if (strncmp(arch, "gemma", 5) == 0) {
        cfg.arch_type   = ARCH_GEMMA3;
        cfg.gemma_norm  = true;    // additive RMSNorm: output * (1 + weight)
        cfg.embd_scale  = true;    // scale embeddings by sqrt(n_embd)
        cfg.use_gelu    = true;    // GELUTanh instead of SiLU
        cfg.has_qk_norm = true;    // try loading Q/K norm tensors
        cfg.rope_type   = GGML_ROPE_TYPE_NEOX;
    } else {
        // Qwen2/Qwen3/default transformer
        cfg.arch_type   = ARCH_QWEN2;
        cfg.gemma_norm  = false;
        cfg.embd_scale  = false;
        cfg.use_gelu    = false;
        cfg.has_qk_norm = false;
        cfg.rope_type   = GGML_ROPE_TYPE_NEOX;
    }

    printf("  head_dim=%u  rms_eps=%.2e  rope_freq_base=%.0f  arch_type=%s\n",
        cfg.head_dim, cfg.rms_eps, cfg.rope_freq_base,
        cfg.arch_type == ARCH_GEMMA3 ? "gemma3" : "qwen");

    if (cfg.n_embd == 0 || cfg.n_layer == 0 || cfg.n_head == 0) {
        printf("  FAIL: invalid model config\n");
        return false;
    }
    if (cfg.n_layer > MAX_LAYERS) {
        printf("  FAIL: too many layers (%u > %d)\n", cfg.n_layer, MAX_LAYERS);
        return false;
    }

    // Load tokenizer
    int64_t tokens_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tokens_key >= 0) {
        size_t n = gguf_get_arr_n(gctx, tokens_key);
        state.vocab.resize(n);
        for (size_t i = 0; i < n; i++) {
            state.vocab[i] = gguf_get_arr_str(gctx, tokens_key, i);
        }
        printf("  Vocab: %zu tokens\n", n);
    }
    state.bos_token = (int)gguf_get_u32(gctx, "tokenizer.ggml.bos_token_id", 0);
    state.eos_token = (int)gguf_get_u32(gctx, "tokenizer.ggml.eos_token_id", 0);

    // If n_vocab not in GGUF, derive from tokenizer
    if (cfg.n_vocab == 0 && !state.vocab.empty()) {
        cfg.n_vocab = (uint32_t)state.vocab.size();
        printf("  [INFO] n_vocab derived from tokenizer: %u\n", cfg.n_vocab);
    }

    printf("  BOS=%d  EOS=%d\n", state.bos_token, state.eos_token);

    // Count weight tensors: 3 global + 11 per layer (9 base + 2 optional qk_norm)
    int n_weight_tensors = 3 + (int)cfg.n_layer * 11;

    // Create weight context (tensor metadata only, no data)
    size_t weight_ctx_size = (size_t)n_weight_tensors * ggml_tensor_overhead() + 256;
    struct ggml_init_params wparams = { weight_ctx_size, nullptr, true };
    state.weight_ctx = ggml_init(wparams);

    // Helper: create tensor mirroring GGUF source, or return nullptr
    auto make_weight = [&](const char * name) -> struct ggml_tensor * {
        struct ggml_tensor * src = ggml_get_tensor(data_ctx, name);
        if (!src) return nullptr;
        struct ggml_tensor * dst = nullptr;
        if (ggml_n_dims(src) == 1) {
            dst = ggml_new_tensor_1d(state.weight_ctx, src->type, src->ne[0]);
        } else {
            dst = ggml_new_tensor_2d(state.weight_ctx, src->type, src->ne[0], src->ne[1]);
        }
        ggml_set_name(dst, name);
        return dst;
    };

    // Global weights
    state.token_embd  = make_weight("token_embd.weight");
    state.output_norm = make_weight("output_norm.weight");
    state.output      = make_weight("output.weight");

    if (!state.token_embd) {
        printf("  FAIL: token_embd.weight not found\n");
        return false;
    }
    if (!state.output_norm) {
        printf("  FAIL: output_norm.weight not found\n");
        return false;
    }

    // Handle weight tying: output.weight may be shared with token_embd.weight
    bool output_tied = false;
    if (!state.output) {
        printf("  [INFO] output.weight not found, tying to token_embd.weight\n");
        // Create a tensor with the same shape as token_embd
        struct ggml_tensor * src = ggml_get_tensor(data_ctx, "token_embd.weight");
        state.output = ggml_new_tensor_2d(state.weight_ctx, src->type, src->ne[0], src->ne[1]);
        ggml_set_name(state.output, "output.weight");
        output_tied = true;
    }

    // Per-layer weights
    int missing = 0;
    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        LayerWeights & lw = state.layers[il];
        char buf[128];

        snprintf(buf, sizeof(buf), "blk.%u.attn_norm.weight", il);
        lw.attn_norm = make_weight(buf);

        snprintf(buf, sizeof(buf), "blk.%u.attn_q.weight", il);
        lw.attn_q = make_weight(buf);

        snprintf(buf, sizeof(buf), "blk.%u.attn_k.weight", il);
        lw.attn_k = make_weight(buf);

        snprintf(buf, sizeof(buf), "blk.%u.attn_v.weight", il);
        lw.attn_v = make_weight(buf);

        snprintf(buf, sizeof(buf), "blk.%u.attn_output.weight", il);
        lw.attn_output = make_weight(buf);

        // Optional Q/K norm (Gemma3)
        snprintf(buf, sizeof(buf), "blk.%u.attn_q_norm.weight", il);
        lw.q_norm = make_weight(buf);
        snprintf(buf, sizeof(buf), "blk.%u.attn_k_norm.weight", il);
        lw.k_norm = make_weight(buf);
        if (lw.q_norm && lw.k_norm && il == 0) {
            printf("  [INFO] Q/K norm tensors found\n");
            cfg.has_qk_norm = true;
        }

        snprintf(buf, sizeof(buf), "blk.%u.ffn_norm.weight", il);
        lw.ffn_norm = make_weight(buf);

        snprintf(buf, sizeof(buf), "blk.%u.ffn_gate.weight", il);
        lw.ffn_gate = make_weight(buf);

        snprintf(buf, sizeof(buf), "blk.%u.ffn_up.weight", il);
        lw.ffn_up = make_weight(buf);

        snprintf(buf, sizeof(buf), "blk.%u.ffn_down.weight", il);
        lw.ffn_down = make_weight(buf);

        if (!lw.attn_norm || !lw.attn_q || !lw.attn_k || !lw.attn_v ||
            !lw.attn_output || !lw.ffn_norm || !lw.ffn_gate || !lw.ffn_up || !lw.ffn_down) {
            printf("  [WARN] layer %u: missing tensors\n", il);
            missing++;
        }
    }

    if (missing > 0) {
        printf("  FAIL: %d layers have missing tensors\n", missing);
        return false;
    }

    // Derive head_dim from actual Q weight shape if not set from GGUF key
    // Q weight: [n_embd, n_head * head_dim] → head_dim = ne[1] / n_head
    if (state.layers[0].attn_q && cfg.n_head > 0) {
        uint32_t q_out_dim = (uint32_t)state.layers[0].attn_q->ne[1];
        uint32_t derived_head_dim = q_out_dim / cfg.n_head;
        if (derived_head_dim != cfg.head_dim) {
            printf("  [INFO] head_dim corrected: %u -> %u (from Q weight [%lld, %lld])\n",
                cfg.head_dim, derived_head_dim,
                (long long)state.layers[0].attn_q->ne[0],
                (long long)state.layers[0].attn_q->ne[1]);
            cfg.head_dim = derived_head_dim;
        }
    }

    // Allocate weight buffer on backend
    state.weight_buf = ggml_backend_alloc_ctx_tensors(state.weight_ctx, backend);
    if (!state.weight_buf) {
        printf("  FAIL: weight buffer allocation failed\n");
        return false;
    }
    ggml_backend_buffer_set_usage(state.weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    printf("  Weight buffer: %.1f MB\n",
        ggml_backend_buffer_get_size(state.weight_buf) / 1024.0 / 1024.0);

    // Copy weight data from mmap
    auto t2 = Clock::now();

    auto copy_weight = [&](struct ggml_tensor * dst, const char * name) {
        struct ggml_tensor * src = ggml_get_tensor(data_ctx, name);
        if (src && dst) {
            ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
        }
    };

    copy_weight(state.token_embd, "token_embd.weight");
    copy_weight(state.output_norm, "output_norm.weight");
    if (output_tied) {
        // Copy token_embd data to output tensor
        struct ggml_tensor * src = ggml_get_tensor(data_ctx, "token_embd.weight");
        ggml_backend_tensor_set(state.output, src->data, 0, ggml_nbytes(src));
    } else {
        copy_weight(state.output, "output.weight");
    }

    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "blk.%u.attn_norm.weight", il);   copy_weight(state.layers[il].attn_norm, buf);
        snprintf(buf, sizeof(buf), "blk.%u.attn_q.weight", il);      copy_weight(state.layers[il].attn_q, buf);
        snprintf(buf, sizeof(buf), "blk.%u.attn_k.weight", il);      copy_weight(state.layers[il].attn_k, buf);
        snprintf(buf, sizeof(buf), "blk.%u.attn_v.weight", il);      copy_weight(state.layers[il].attn_v, buf);
        snprintf(buf, sizeof(buf), "blk.%u.attn_output.weight", il); copy_weight(state.layers[il].attn_output, buf);
        snprintf(buf, sizeof(buf), "blk.%u.attn_q_norm.weight", il); copy_weight(state.layers[il].q_norm, buf);
        snprintf(buf, sizeof(buf), "blk.%u.attn_k_norm.weight", il); copy_weight(state.layers[il].k_norm, buf);
        snprintf(buf, sizeof(buf), "blk.%u.ffn_norm.weight", il);    copy_weight(state.layers[il].ffn_norm, buf);
        snprintf(buf, sizeof(buf), "blk.%u.ffn_gate.weight", il);    copy_weight(state.layers[il].ffn_gate, buf);
        snprintf(buf, sizeof(buf), "blk.%u.ffn_up.weight", il);      copy_weight(state.layers[il].ffn_up, buf);
        snprintf(buf, sizeof(buf), "blk.%u.ffn_down.weight", il);    copy_weight(state.layers[il].ffn_down, buf);
    }

    auto t3 = Clock::now();
    printf("  Weight copy: %.0f ms\n",
        std::chrono::duration<double, std::milli>(t3 - t2).count());

    printf("  PASS: Model loaded (%u layers, %u params)\n", cfg.n_layer, cfg.n_vocab);
    return true;
}

// --------------------------------------------------------------------------
// Step 2: Init KV cache
// --------------------------------------------------------------------------
bool init_kv_cache(ModelState & state, ggml_backend_t backend) {
    const ModelConfig & cfg = state.cfg;

    // Layout: [head_dim, max_ctx, n_head_kv] per layer — matches flash_attn_ext format
    int n_kv_tensors = (int)cfg.n_layer * 2;
    size_t kv_ctx_size = (size_t)n_kv_tensors * ggml_tensor_overhead() + 256;
    struct ggml_init_params kv_params = { kv_ctx_size, nullptr, true };
    state.kv_ctx = ggml_init(kv_params);

    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        state.kv_k[il] = ggml_new_tensor_3d(state.kv_ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.max_ctx, cfg.n_head_kv);
        ggml_format_name(state.kv_k[il], "kv_k_%u", il);

        state.kv_v[il] = ggml_new_tensor_3d(state.kv_ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.max_ctx, cfg.n_head_kv);
        ggml_format_name(state.kv_v[il], "kv_v_%u", il);
    }

    state.kv_buf = ggml_backend_alloc_ctx_tensors(state.kv_ctx, backend);
    if (!state.kv_buf) {
        printf("  FAIL: KV cache allocation failed\n");
        return false;
    }

    ggml_backend_buffer_clear(state.kv_buf, 0);
    state.kv_pos = 0;

    printf("  KV cache: %.1f MB (F16, %u ctx, %u layers)\n",
        ggml_backend_buffer_get_size(state.kv_buf) / 1024.0 / 1024.0,
        cfg.max_ctx, cfg.n_layer);
    return true;
}
