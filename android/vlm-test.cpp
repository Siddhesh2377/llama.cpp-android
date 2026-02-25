// vlm-test.cpp — VLM Pipeline Test & Optimization CLI
//
// Self-contained test harness for the full SmolVLM pipeline on Android.
// Tests each stage independently with detailed timing and tensor diagnostics.
//
// Target model: SmolVLM-500M-Instruct (q8_0 + q8_0.mmproj)
//   LLM:    SmolLM2-360M (llama arch, 960 embd, 30 layers, SiLU, GQA 15/5)
//   Vision: SigLIP-SO (768 embd, 12 layers, 512px, patch=16, GELU)
//   Proj:   idefics3 pixel shuffle (sf=4) + FC → 960
//
// Usage:
//   ./vlm-test --model <gguf> --mmproj <mmproj> [--threads N] [--image <path>]
//   ./vlm-test --model /sdcard/Download/SmolVLM-500M-Instruct-q8_0.gguf \
//              --mmproj /sdcard/Download/SmolVLM-500M-Instruct-q8_0.mmproj
//
// Pipeline stages tested:
//   [1] Backend init + device info
//   [2] LLM GGUF load + metadata dump
//   [3] Vision mmproj load + metadata dump
//   [4] KV cache init
//   [5] Vision encode (synthetic image → embeddings)
//   [6] Text-only LLM forward (verify basic inference)
//   [7] Full VLM pipeline (vision + text merge → decode)
//   [8] Memory + throughput summary

#include "gguf-engine/types.h"
#include "gguf-engine/model.h"
#include "gguf-engine/graph.h"
#include "gguf-engine/vision.h"
#include "gguf-engine/sampling.h"
#include "gguf-engine/tokenizer.h"
#include "gguf-engine/utils.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>

// ===========================================================================
// Helper: set CPU thread count (copied from main.cpp — local to this file)
// ===========================================================================
static void set_cpu_threads(ggml_backend_t backend, int n_threads) {
    auto * dev = ggml_backend_get_device(backend);
    if (!dev) return;
    auto * reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return;
    typedef void (*set_n_threads_fn_t)(ggml_backend_t, int);
    auto fn = (set_n_threads_fn_t)ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_cpu_set_n_threads");
    if (fn) fn(backend, n_threads);
}

// ===========================================================================
// Special token finder — search vocab for exact string match
// ===========================================================================
static int find_special_token(const std::vector<std::string> & vocab,
                               const std::string & token_str) {
    for (int i = 0; i < (int)vocab.size(); i++) {
        if (vocab[i] == token_str) return i;
    }
    return -1;
}

// ===========================================================================
// Image loading and preprocessing
// ===========================================================================
// Load image from file, resize to target_size×target_size, normalize,
// and convert to CHW planar format [W, H, C] for ggml conv2d.
//
// Returns empty vector on failure.
static std::vector<float> load_and_preprocess_image(
    const char * path, int target_size,
    const float mean[3], const float std_dev[3])
{
    int w, h, c;
    unsigned char * data = stbi_load(path, &w, &h, &c, 3); // force RGB
    if (!data) {
        printf("  FAIL: stbi_load('%s') failed: %s\n", path, stbi_failure_reason());
        return {};
    }
    printf("  Loaded image: %dx%d (channels=%d)\n", w, h, c);

    // Bilinear resize to target_size × target_size
    std::vector<float> pixels(target_size * target_size * 3);

    float sx = (float)w / (float)target_size;
    float sy = (float)h / (float)target_size;

    for (int y = 0; y < target_size; y++) {
        float src_y = (y + 0.5f) * sy - 0.5f;
        int y0 = std::max(0, (int)floorf(src_y));
        int y1 = std::min(h - 1, y0 + 1);
        float fy = src_y - y0;

        for (int x = 0; x < target_size; x++) {
            float src_x = (x + 0.5f) * sx - 0.5f;
            int x0 = std::max(0, (int)floorf(src_x));
            int x1 = std::min(w - 1, x0 + 1);
            float fx = src_x - x0;

            for (int ch = 0; ch < 3; ch++) {
                float v00 = data[(y0 * w + x0) * 3 + ch] / 255.0f;
                float v01 = data[(y0 * w + x1) * 3 + ch] / 255.0f;
                float v10 = data[(y1 * w + x0) * 3 + ch] / 255.0f;
                float v11 = data[(y1 * w + x1) * 3 + ch] / 255.0f;

                float v = v00 * (1 - fx) * (1 - fy)
                        + v01 * fx * (1 - fy)
                        + v10 * (1 - fx) * fy
                        + v11 * fx * fy;

                // Normalize: (pixel - mean) / std
                float normalized = (v - mean[ch]) / std_dev[ch];

                // CHW planar: pixels[x + y * target_size + ch * target_size * target_size]
                pixels[x + y * target_size + ch * target_size * target_size] = normalized;
            }
        }
    }

    stbi_image_free(data);
    return pixels;
}

// ===========================================================================
// Logging helpers
// ===========================================================================
#define LOG_SECTION(name) \
    printf("\n" \
           "╔══════════════════════════════════════════════════════════════╗\n" \
           "║  %-58s║\n" \
           "╚══════════════════════════════════════════════════════════════╝\n", name)

#define LOG_SUBSECTION(name) \
    printf("  ┌─ %s ─────────────────────────────\n", name)

#define LOG_KV(key, fmt, ...) printf("  │ %-28s " fmt "\n", key, ##__VA_ARGS__)
#define LOG_OK(fmt, ...)      printf("  │ ✓ " fmt "\n", ##__VA_ARGS__)
#define LOG_FAIL(fmt, ...)    printf("  │ ✗ " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)    printf("  │   " fmt "\n", ##__VA_ARGS__)
#define LOG_END()             printf("  └───────────────────────────────────────\n")

struct Timer {
    Clock::time_point t0;
    Timer() : t0(Clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
    void reset() { t0 = Clock::now(); }
};

static void log_tensor(const char * label, struct ggml_tensor * t) {
    if (!t) {
        LOG_INFO("%-20s (null)", label);
        return;
    }
    const char * type_name = ggml_type_name(t->type);
    if (ggml_n_dims(t) == 1) {
        LOG_INFO("%-20s [%5lld]                    %s  %.2f KB",
            label, (long long)t->ne[0], type_name,
            ggml_nbytes(t) / 1024.0f);
    } else if (ggml_n_dims(t) == 2) {
        LOG_INFO("%-20s [%5lld, %5lld]             %s  %.2f KB",
            label, (long long)t->ne[0], (long long)t->ne[1], type_name,
            ggml_nbytes(t) / 1024.0f);
    } else if (ggml_n_dims(t) == 3) {
        LOG_INFO("%-20s [%5lld, %5lld, %4lld]       %s  %.2f KB",
            label, (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], type_name,
            ggml_nbytes(t) / 1024.0f);
    } else {
        LOG_INFO("%-20s [%lld, %lld, %lld, %lld]  %s  %.2f KB",
            label, (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3], type_name,
            ggml_nbytes(t) / 1024.0f);
    }
}

// Dump first N float values of a tensor (for debugging)
static void dump_tensor_values(const char * label, ggml_backend_t backend,
                               struct ggml_tensor * t, int n = 8) {
    if (!t) return;
    int total = (int)ggml_nelements(t);
    n = std::min(n, total);

    std::vector<float> buf(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> f16buf(n);
        ggml_backend_tensor_get(t, f16buf.data(), 0, n * sizeof(uint16_t));
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(f16buf[i]);
    } else {
        LOG_INFO("%-20s (type %s, can't dump)", label, ggml_type_name(t->type));
        return;
    }

    printf("  │   %-20s first %d: [", label, n);
    for (int i = 0; i < n; i++) printf("%.4f%s", buf[i], i < n-1 ? ", " : "");
    printf("]\n");
}

// Check tensor for NaN/Inf
static int check_tensor_health(ggml_backend_t backend, struct ggml_tensor * t) {
    if (!t) return -1;
    int total = (int)ggml_nelements(t);
    std::vector<float> buf(total);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, total * sizeof(float));
    } else {
        return 0; // can't check non-f32
    }
    int bad = 0;
    for (int i = 0; i < total; i++) {
        if (std::isnan(buf[i]) || std::isinf(buf[i])) bad++;
    }
    return bad;
}

// ===========================================================================
// VLM-specific graph builder: starts from pre-computed embeddings
// ===========================================================================
// This mirrors build_graph() from graph.cpp but takes a 2D embedding tensor
// as input instead of token IDs. Used for the VLM prefill where vision
// embeddings are merged with text embeddings before entering the LLM.
//
// Stripped of all intervention/character-engine code.
static struct ggml_cgraph * build_vlm_prefill_graph(
    struct ggml_context * ctx,
    ModelState & state,
    int seq_len,      // total tokens (text + vision)
    int kv_pos,
    int kv_len
) {
    const ModelConfig & cfg = state.cfg;
    const int head_dim  = (int)cfg.head_dim;
    const int n_head    = (int)cfg.n_head;
    const int n_head_kv = (int)cfg.n_head_kv;
    const int n_embd_head = n_head * head_dim;
    const float base_scale = 1.0f / sqrtf((float)head_dim);
    const size_t f16_sz = ggml_type_size(GGML_TYPE_F16);

    // Input: pre-computed embeddings [n_embd, seq_len]
    struct ggml_tensor * inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
        cfg.n_embd, seq_len);
    ggml_set_name(inp_embd, "inp_embd");
    ggml_set_input(inp_embd);

    // Position indices
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // Attention mask [kv_len, seq_len]
    struct ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
        kv_len, seq_len);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    struct ggml_tensor * cur = inp_embd;

    // Gemma: scale embeddings
    if (cfg.embd_scale) {
        cur = ggml_scale(ctx, cur, sqrtf((float)cfg.n_embd));
    }

    std::vector<struct ggml_tensor *> kv_stores;

    // Transformer layers
    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        const LayerWeights & lw = state.layers[il];
        struct ggml_tensor * residual = cur;

        // Attention pre-norm
        cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
        if (cfg.gemma_norm) {
            cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, lw.attn_norm));
        } else {
            cur = ggml_mul(ctx, cur, lw.attn_norm);
        }

        // QKV projections
        struct ggml_tensor * Q = ggml_mul_mat(ctx, lw.attn_q, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, lw.attn_k, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, lw.attn_v, cur);

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, seq_len);
        K = ggml_reshape_3d(ctx, K, head_dim, n_head_kv, seq_len);
        V = ggml_reshape_3d(ctx, V, head_dim, n_head_kv, seq_len);

        // Optional Q/K norm (Gemma3)
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

        struct ggml_tensor * attn_merged = ggml_reshape_2d(ctx,
            ggml_cont(ctx, attn_out), n_embd_head, seq_len);

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

        struct ggml_tensor * gate_proj = ggml_mul_mat(ctx, lw.ffn_gate, cur);
        struct ggml_tensor * gate = cfg.use_gelu
            ? ggml_gelu(ctx, gate_proj)
            : ggml_silu(ctx, gate_proj);
        struct ggml_tensor * up = ggml_mul_mat(ctx, lw.ffn_up, cur);
        struct ggml_tensor * gate_up = ggml_mul(ctx, gate, up);
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

    // Build graph
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2048, false);
    for (auto * kv_op : kv_stores) {
        ggml_build_forward_expand(graph, kv_op);
    }
    ggml_build_forward_expand(graph, logits);

    return graph;
}

// Context size for VLM prefill graph
static size_t vlm_prefill_ctx_size(int n_layers) {
    int n_tensors = n_layers * 35 + 20;
    return (size_t)n_tensors * ggml_tensor_overhead() +
           ggml_graph_overhead_custom(2048, false);
}

// ===========================================================================
// Synthetic image generation (for pipeline testing without image loader)
// ===========================================================================
// Generates a normalized test pattern: RGB gradient with spatial variation
// Output: [image_size, image_size, 3] F32, normalized per vision model spec
static std::vector<float> generate_test_image(int image_size,
                                               const float mean[3],
                                               const float std[3]) {
    std::vector<float> pixels(image_size * image_size * 3);

    for (int y = 0; y < image_size; y++) {
        for (int x = 0; x < image_size; x++) {
            // Generate 0-1 range test pattern
            float r = (float)x / (float)(image_size - 1);              // horizontal gradient
            float g = (float)y / (float)(image_size - 1);              // vertical gradient
            float b = 0.5f + 0.5f * sinf((float)(x + y) * 0.05f);     // wave pattern

            // Normalize: (pixel - mean) / std
            float nr = (r - mean[0]) / std[0];
            float ng = (g - mean[1]) / std[1];
            float nb = (b - mean[2]) / std[2];

            // Layout: [W, H, C] — ggml conv2d expects this
            // Pixel at (x, y, c) = pixels[x + y * image_size + c * image_size * image_size]
            // But ggml_conv_2d input is [W, H, C, N] where W=ne[0], H=ne[1], C=ne[2]
            // Actually for ggml: pixels[x + y * W + c * W * H]
            int idx_r = x + y * image_size + 0 * image_size * image_size;
            int idx_g = x + y * image_size + 1 * image_size * image_size;
            int idx_b = x + y * image_size + 2 * image_size * image_size;
            pixels[idx_r] = nr;
            pixels[idx_g] = ng;
            pixels[idx_b] = nb;
        }
    }
    return pixels;
}

// ===========================================================================
// CLI argument parsing
// ===========================================================================
struct VLMTestConfig {
    const char * model_path = nullptr;
    const char * mmproj_path = nullptr;
    const char * image_path = nullptr;   // real image file (JPG/PNG), or nullptr for synthetic
    int threads = 4;
    int max_tokens = 64;
    const char * prompt = "Describe this image in detail.";
    bool verbose = false;
};

static void print_usage(const char * prog) {
    printf("VLM Pipeline Test — SmolVLM on Android\n\n");
    printf("Usage: %s --model <gguf> --mmproj <mmproj> [options]\n\n", prog);
    printf("Options:\n");
    printf("  --model PATH       LLM GGUF model file (required)\n");
    printf("  --mmproj PATH      Vision mmproj file (required)\n");
    printf("  --threads N        CPU threads (default: 4)\n");
    printf("  --max-tokens N     Max tokens to generate (default: 32)\n");
    printf("  --image PATH       Image file to test with (JPG/PNG, default: synthetic)\n");
    printf("  --prompt TEXT      User prompt (default: \"Describe this image...\")\n");
    printf("  --verbose          Extra tensor dumps\n");
    printf("\nExample:\n");
    printf("  %s --model /sdcard/Download/SmolVLM-500M-Instruct-q8_0.gguf \\\n", prog);
    printf("         --mmproj /sdcard/Download/SmolVLM-500M-Instruct-q8_0.mmproj\n");
}

static bool parse_args(int argc, char ** argv, VLMTestConfig & cfg) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            cfg.mmproj_path = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            cfg.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            cfg.image_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            cfg.prompt = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            cfg.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
    }
    if (!cfg.model_path || !cfg.mmproj_path) {
        printf("Error: --model and --mmproj are required\n\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// ===========================================================================
// Stage 1: Backend init
// ===========================================================================
static ggml_backend_t init_backend(int n_threads) {
    LOG_SECTION("STAGE 1: Backend Initialization");
    Timer t;

    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        LOG_FAIL("CPU backend init failed");
        return nullptr;
    }

    set_cpu_threads(backend, n_threads);

    LOG_SUBSECTION("CPU Backend");
    LOG_KV("backend", "%s", ggml_backend_name(backend));
    LOG_KV("threads", "%d", n_threads);
    LOG_KV("init time", "%.1f ms", t.ms());
    LOG_END();

    return backend;
}

// ===========================================================================
// Stage 2: Load LLM
// ===========================================================================
static bool load_llm(ModelState & state, const char * path, ggml_backend_t backend,
                     bool verbose) {
    LOG_SECTION("STAGE 2: LLM Model Loading");
    Timer t;

    if (!load_model(state, path, -1, backend)) {
        LOG_FAIL("LLM load failed");
        return false;
    }

    const ModelConfig & cfg = state.cfg;

    LOG_SUBSECTION("LLM Architecture");
    LOG_KV("architecture", "%s", cfg.arch);
    LOG_KV("arch_type", "%s", cfg.arch_type == ARCH_GEMMA3 ? "gemma3" : "qwen2/llama");
    LOG_KV("n_embd", "%u", cfg.n_embd);
    LOG_KV("n_head (Q/KV)", "%u / %u", cfg.n_head, cfg.n_head_kv);
    LOG_KV("head_dim", "%u", cfg.head_dim);
    LOG_KV("n_layer", "%u", cfg.n_layer);
    LOG_KV("n_ff", "%u", cfg.n_ff);
    LOG_KV("n_vocab", "%u", cfg.n_vocab);
    LOG_KV("max_ctx", "%u", cfg.max_ctx);
    LOG_KV("rms_eps", "%.2e", cfg.rms_eps);
    LOG_KV("rope_freq_base", "%.0f", cfg.rope_freq_base);
    LOG_KV("activation", "%s", cfg.use_gelu ? "GELU" : "SiLU");
    LOG_KV("norm_type", "%s", cfg.gemma_norm ? "additive (1+w)" : "standard");
    LOG_KV("embd_scale", "%s", cfg.embd_scale ? "yes" : "no");
    LOG_KV("qk_norm", "%s", cfg.has_qk_norm ? "yes" : "no");
    LOG_KV("vocab tokens", "%zu", state.vocab.size());
    LOG_KV("BOS/EOS", "%d / %d", state.bos_token, state.eos_token);
    LOG_END();

    LOG_SUBSECTION("Weight Tensors (layer 0)");
    log_tensor("token_embd", state.token_embd);
    log_tensor("output_norm", state.output_norm);
    log_tensor("output", state.output);
    log_tensor("attn_norm", state.layers[0].attn_norm);
    log_tensor("attn_q", state.layers[0].attn_q);
    log_tensor("attn_k", state.layers[0].attn_k);
    log_tensor("attn_v", state.layers[0].attn_v);
    log_tensor("attn_output", state.layers[0].attn_output);
    log_tensor("ffn_gate", state.layers[0].ffn_gate);
    log_tensor("ffn_up", state.layers[0].ffn_up);
    log_tensor("ffn_down", state.layers[0].ffn_down);
    LOG_END();

    LOG_SUBSECTION("Memory");
    LOG_KV("weight buffer", "%.1f MB",
        ggml_backend_buffer_get_size(state.weight_buf) / 1024.0 / 1024.0);
    LOG_KV("total load time", "%.0f ms", t.ms());
    LOG_END();

    return true;
}

// ===========================================================================
// Stage 3: Load Vision
// ===========================================================================
static bool load_vision(VisionModelState & vs, const char * path,
                        ggml_backend_t backend, bool verbose) {
    LOG_SECTION("STAGE 3: Vision Model Loading");
    Timer t;

    if (!load_vision_model(vs, path, backend)) {
        LOG_FAIL("Vision model load failed");
        return false;
    }

    LOG_SUBSECTION("Vision Stats");
    LOG_KV("loaded", "%s", vs.loaded ? "yes" : "no");
    LOG_KV("weight buffer", "%.1f MB",
        vs.weight_buf ? ggml_backend_buffer_get_size(vs.weight_buf) / 1024.0 / 1024.0 : 0.0);
    LOG_KV("total load time", "%.0f ms", t.ms());
    LOG_END();

    return true;
}

// ===========================================================================
// Stage 4: KV cache init
// ===========================================================================
static bool setup_kv_cache(ModelState & state, ggml_backend_t backend) {
    LOG_SECTION("STAGE 4: KV Cache Init");
    Timer t;

    if (!init_kv_cache(state, backend)) {
        LOG_FAIL("KV cache init failed");
        return false;
    }

    LOG_SUBSECTION("KV Cache");
    LOG_KV("layout", "[head_dim=%u, max_ctx=%u, n_head_kv=%u]",
        state.cfg.head_dim, state.cfg.max_ctx, state.cfg.n_head_kv);
    LOG_KV("type", "F16");
    LOG_KV("layers", "%u", state.cfg.n_layer);
    LOG_KV("buffer size", "%.1f MB",
        ggml_backend_buffer_get_size(state.kv_buf) / 1024.0 / 1024.0);
    LOG_KV("init time", "%.1f ms", t.ms());
    LOG_END();

    return true;
}

// ===========================================================================
// Stage 5: Vision encode
// ===========================================================================
static bool run_vision_encode(VisionModelState & vs, ggml_backend_t backend,
                              std::vector<float> & vision_embeds_out,
                              int & n_vision_tokens, int & vision_dim,
                              const char * image_path, bool verbose) {
    LOG_SECTION("STAGE 5: Vision Encode");

    LOG_SUBSECTION("Image Preprocessing");
    Timer t;

    // SigLIP normalization params
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std_val[3] = {0.5f, 0.5f, 0.5f};
    int image_size = 512;  // SmolVLM default

    LOG_KV("image_size", "%dx%d", image_size, image_size);
    LOG_KV("normalization", "mean=[%.1f,%.1f,%.1f] std=[%.1f,%.1f,%.1f]",
        mean[0], mean[1], mean[2], std_val[0], std_val[1], std_val[2]);

    std::vector<float> pixels;
    if (image_path) {
        LOG_KV("source", "file: %s", image_path);
        pixels = load_and_preprocess_image(image_path, image_size, mean, std_val);
        if (pixels.empty()) {
            LOG_FAIL("Image loading failed");
            LOG_END();
            return false;
        }
    } else {
        LOG_KV("source", "synthetic test pattern (gradient + wave)");
        pixels = generate_test_image(image_size, mean, std_val);
    }

    LOG_KV("pixel buffer", "%.2f MB (%d floats)",
        pixels.size() * sizeof(float) / 1024.0 / 1024.0, (int)pixels.size());
    LOG_KV("preprocess time", "%.1f ms", t.ms());
    LOG_END();

    // Build vision graph
    LOG_SUBSECTION("Vision Graph");
    t.reset();

    // Context for graph building
    // Vision: 12 layers * ~25 tensors + 20 global = ~320 tensors
    int n_tensors_est = 400;
    size_t ctx_size = (size_t)n_tensors_est * ggml_tensor_overhead() +
                      ggml_graph_overhead_custom(4096, false);
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph * graph = build_vision_graph(ctx, vs);
    if (!graph) {
        LOG_FAIL("Vision graph build failed");
        ggml_free(ctx);
        return false;
    }

    LOG_KV("graph nodes", "%d", ggml_graph_n_nodes(graph));
    LOG_KV("graph leafs", "%d", ggml_graph_n_nodes(graph));
    LOG_KV("build time", "%.1f ms", t.ms());

    // Allocate graph
    t.reset();
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        LOG_FAIL("Vision graph alloc failed");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    LOG_KV("compute buffer", "%.2f MB",
        ggml_gallocr_get_buffer_size(galloc, 0) / 1024.0 / 1024.0);
    LOG_KV("alloc time", "%.1f ms", t.ms());
    LOG_END();

    // Set inputs
    LOG_SUBSECTION("Vision Forward Pass");
    t.reset();

    struct ggml_tensor * inp_pixels = ggml_graph_get_tensor(graph, "inp_pixels");
    if (!inp_pixels) {
        LOG_FAIL("inp_pixels tensor not found in graph");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(inp_pixels, pixels.data(), 0,
        pixels.size() * sizeof(float));

    // Set attention mask (all zeros = dense attention, no masking)
    struct ggml_tensor * v_attn_mask = ggml_graph_get_tensor(graph, "v_attn_mask");
    if (v_attn_mask) {
        int mask_size = (int)ggml_nelements(v_attn_mask);
        std::vector<uint16_t> zeros(mask_size, 0);
        ggml_backend_tensor_set(v_attn_mask, zeros.data(), 0,
            mask_size * sizeof(uint16_t));
    }

    double input_ms = t.ms();
    LOG_KV("input set time", "%.1f ms", input_ms);

    // Execute vision encoder
    t.reset();
    ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    double compute_ms = t.ms();
    LOG_KV("compute time", "%.1f ms", compute_ms);

    // Extract output
    struct ggml_tensor * vision_embd = ggml_graph_get_tensor(graph, "vision_embd");
    if (!vision_embd) {
        LOG_FAIL("vision_embd output not found");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    log_tensor("vision_embd", vision_embd);

    vision_dim = (int)vision_embd->ne[0];
    n_vision_tokens = (int)vision_embd->ne[1];
    int total_floats = vision_dim * n_vision_tokens;

    vision_embeds_out.resize(total_floats);
    ggml_backend_tensor_get(vision_embd, vision_embeds_out.data(), 0,
        total_floats * sizeof(float));

    // Sanity check
    int bad = 0;
    float sum = 0, min_v = 1e9, max_v = -1e9;
    for (int i = 0; i < total_floats; i++) {
        float v = vision_embeds_out[i];
        if (std::isnan(v) || std::isinf(v)) bad++;
        sum += v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    LOG_KV("output shape", "[%d, %d]", vision_dim, n_vision_tokens);
    LOG_KV("output stats", "mean=%.4f min=%.4f max=%.4f",
        sum / total_floats, min_v, max_v);
    LOG_KV("NaN/Inf count", "%d", bad);

    if (verbose) {
        printf("  │   first 8 values: [");
        for (int i = 0; i < std::min(8, total_floats); i++)
            printf("%.4f%s", vision_embeds_out[i], i < 7 ? ", " : "");
        printf("]\n");
    }

    if (bad > 0) {
        LOG_FAIL("Vision output contains %d bad values!", bad);
    } else {
        LOG_OK("Vision encode PASS");
    }
    LOG_END();

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return bad == 0;
}

// ===========================================================================
// Stage 6: Text-only LLM forward (sanity check)
// ===========================================================================
static bool run_text_only_test(ModelState & state, ggml_backend_t backend,
                               bool verbose) {
    LOG_SECTION("STAGE 6: Text-Only LLM Forward (Sanity Check)");

    const ModelConfig & cfg = state.cfg;

    // Reset KV cache
    ggml_backend_buffer_clear(state.kv_buf, 0);
    state.kv_pos = 0;

    // Tokenize a short test prompt
    LOG_SUBSECTION("Tokenization");
    Timer t;

    std::string test_text = "Hello";
    std::vector<int32_t> tokens = tokenize_simple(state.vocab, test_text);
    if (tokens.empty()) {
        LOG_FAIL("Tokenization produced 0 tokens");
        return false;
    }

    LOG_KV("text", "\"%s\"", test_text.c_str());
    LOG_KV("tokens", "%zu", tokens.size());
    if (verbose) {
        printf("  │   token IDs: [");
        for (size_t i = 0; i < tokens.size(); i++)
            printf("%d%s", tokens[i], i < tokens.size()-1 ? ", " : "");
        printf("]\n");
    }
    LOG_END();

    // Build and run forward pass
    LOG_SUBSECTION("Forward Pass");
    t.reset();

    int seq_len = (int)tokens.size();
    int kv_pos = 0;
    int kv_len = seq_len;

    // Build graph
    size_t ctx_size = compute_ctx_size((int)cfg.n_layer);
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph * graph = build_graph(ctx, state, seq_len, kv_pos, kv_len,
        (int)cfg.n_layer, false);

    // Allocate
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        LOG_FAIL("Graph alloc failed");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    LOG_KV("graph nodes", "%d", ggml_graph_n_nodes(graph));
    LOG_KV("compute buffer", "%.2f MB",
        ggml_gallocr_get_buffer_size(galloc, 0) / 1024.0 / 1024.0);

    // Set inputs
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "inp_tokens"),
        tokens.data(), 0, seq_len * sizeof(int32_t));

    std::vector<int32_t> positions(seq_len);
    std::iota(positions.begin(), positions.end(), 0);
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "inp_pos"),
        positions.data(), 0, seq_len * sizeof(int32_t));

    std::vector<uint16_t> mask;
    build_causal_mask(mask, kv_len, seq_len, kv_pos);
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "attn_mask"),
        mask.data(), 0, kv_len * seq_len * sizeof(uint16_t));

    // Compute
    t.reset();
    ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    double fwd_ms = t.ms();

    // Read logits
    struct ggml_tensor * logits_t = ggml_graph_get_tensor(graph, "logits");
    if (!logits_t) {
        LOG_FAIL("logits tensor not found");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> logits(cfg.n_vocab);
    // Get last token's logits: offset = (seq_len-1) * n_vocab * sizeof(float)
    size_t offset = (size_t)(seq_len - 1) * cfg.n_vocab * sizeof(float);
    ggml_backend_tensor_get(logits_t, logits.data(), offset,
        cfg.n_vocab * sizeof(float));

    // Find argmax
    int best = 0;
    for (int i = 1; i < (int)cfg.n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }

    std::string next_token = (best < (int)state.vocab.size()) ? state.vocab[best] : "???";

    LOG_KV("forward time", "%.1f ms", fwd_ms);
    LOG_KV("logits shape", "[%lld, %lld]",
        (long long)logits_t->ne[0], (long long)logits_t->ne[1]);
    LOG_KV("predicted token", "'%s' (id=%d)", next_token.c_str(), best);
    LOG_KV("top logit value", "%.4f", logits[best]);

    // Check logits health
    int bad = 0;
    for (int i = 0; i < (int)cfg.n_vocab; i++) {
        if (std::isnan(logits[i]) || std::isinf(logits[i])) bad++;
    }
    if (bad > 0) {
        LOG_FAIL("Logits contain %d NaN/Inf values!", bad);
    } else {
        LOG_OK("Text-only LLM forward PASS");
    }
    LOG_END();

    state.kv_pos = kv_len;

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return bad == 0;
}

// ===========================================================================
// Stage 7: Full VLM pipeline
// ===========================================================================
static bool run_vlm_pipeline(ModelState & state, VisionModelState & vs,
                             ggml_backend_t backend,
                             const std::vector<float> & vision_embeds,
                             int n_vision_tokens, int vision_dim,
                             const VLMTestConfig & test_cfg) {
    LOG_SECTION("STAGE 7: Full VLM Pipeline");

    const ModelConfig & cfg = state.cfg;

    // Reset KV cache
    ggml_backend_buffer_clear(state.kv_buf, 0);
    state.kv_pos = 0;

    // --- Tokenize prompt with correct SmolVLM chat template ---
    LOG_SUBSECTION("Prompt Construction (SmolVLM/idefics3)");
    Timer t;

    // SmolVLM chat template (from tokenizer_config.json):
    //   <|im_start|>User:<image>{prompt}<end_of_utterance>\nAssistant:
    //
    // Where <image> expands to:
    //   <fake_token_around_image><global-img>[64 vision embeds]<fake_token_around_image>
    //
    // Token sequence:
    //   BOS + "User:" + fake_around + global_img + [vision_tokens] + fake_around + prompt + eou + "\n" + "Assistant:"

    // Find special tokens by searching vocab
    int bos_id = state.bos_token;
    int end_of_utterance_id = find_special_token(state.vocab, "<end_of_utterance>");
    int fake_token_id       = find_special_token(state.vocab, "<fake_token_around_image>");
    int global_img_id       = find_special_token(state.vocab, "<global-img>");

    LOG_KV("BOS token", "id=%d", bos_id);
    LOG_KV("<end_of_utterance>", "id=%d", end_of_utterance_id);
    LOG_KV("<fake_token_around_image>", "id=%d", fake_token_id);
    LOG_KV("<global-img>", "id=%d", global_img_id);

    // Build prefix: BOS + "User:" + <fake_token_around_image> + <global-img>
    std::vector<int32_t> prefix_tokens;
    prefix_tokens.push_back(bos_id);  // <|im_start|> = BOS for SmolLM2

    auto user_toks = tokenize_simple(state.vocab, "User:");
    prefix_tokens.insert(prefix_tokens.end(), user_toks.begin(), user_toks.end());

    if (fake_token_id >= 0) prefix_tokens.push_back(fake_token_id);
    if (global_img_id >= 0) prefix_tokens.push_back(global_img_id);

    // Vision embeddings go between prefix and suffix (n_vision_tokens tokens)

    // Build suffix: <fake_token_around_image> + prompt + <end_of_utterance> + "\n" + "Assistant:"
    std::vector<int32_t> suffix_tokens;
    if (fake_token_id >= 0) suffix_tokens.push_back(fake_token_id);

    auto prompt_toks = tokenize_simple(state.vocab, std::string(test_cfg.prompt));
    suffix_tokens.insert(suffix_tokens.end(), prompt_toks.begin(), prompt_toks.end());

    if (end_of_utterance_id >= 0) suffix_tokens.push_back(end_of_utterance_id);

    auto newline_toks = tokenize_simple(state.vocab, "\n");
    suffix_tokens.insert(suffix_tokens.end(), newline_toks.begin(), newline_toks.end());

    auto asst_toks = tokenize_simple(state.vocab, "Assistant:");
    suffix_tokens.insert(suffix_tokens.end(), asst_toks.begin(), asst_toks.end());

    int n_prefix = (int)prefix_tokens.size();
    int n_suffix = (int)suffix_tokens.size();
    int total_seq = n_prefix + n_vision_tokens + n_suffix;

    LOG_KV("prefix tokens", "%d  [BOS + \"User:\" + boundary + global_img]", n_prefix);
    LOG_KV("vision tokens", "%d  (dim=%d)", n_vision_tokens, vision_dim);
    LOG_KV("suffix tokens", "%d  [boundary + prompt + eou + newline + \"Assistant:\"]", n_suffix);
    LOG_KV("total sequence", "%d tokens", total_seq);

    // Log the token IDs for debugging
    if (test_cfg.verbose) {
        printf("  │   prefix IDs: [");
        for (int i = 0; i < n_prefix; i++) {
            printf("%d", prefix_tokens[i]);
            if (i < n_prefix-1) printf(", ");
        }
        printf("]\n");
        printf("  │   suffix IDs: [");
        for (int i = 0; i < n_suffix; i++) {
            printf("%d", suffix_tokens[i]);
            if (i < n_suffix-1) printf(", ");
        }
        printf("]\n");
        printf("  │   suffix tokens decoded: [");
        for (int i = 0; i < n_suffix; i++) {
            int id = suffix_tokens[i];
            std::string s = (id < (int)state.vocab.size()) ? state.vocab[id] : "???";
            printf("'%s'", s.c_str());
            if (i < n_suffix-1) printf(", ");
        }
        printf("]\n");
    }

    if (total_seq > (int)cfg.max_ctx) {
        LOG_FAIL("Total sequence (%d) exceeds max_ctx (%u)", total_seq, cfg.max_ctx);
        LOG_END();
        return false;
    }

    // Check dimension match
    if (vision_dim != (int)cfg.n_embd) {
        LOG_FAIL("Vision dim (%d) != LLM n_embd (%u) — projector mismatch!",
            vision_dim, cfg.n_embd);
        LOG_END();
        return false;
    }
    LOG_OK("Vision dim matches LLM n_embd (%d)", vision_dim);
    LOG_END();

    // --- Build merged embeddings on CPU ---
    LOG_SUBSECTION("Embedding Merge");
    t.reset();

    // Get text embeddings from token_embd weight
    // token_embd: [n_vocab, n_embd] in GGUF
    // ggml_get_rows does: output[i] = token_embd[tokens[i]]
    // We do this manually on CPU for the merge step
    int n_embd = (int)cfg.n_embd;
    std::vector<float> merged_embeds(total_seq * n_embd);

    // Read the full embedding table (we need random access)
    // For efficiency, only read the rows we need
    // token_embd type might be quantized — we need to dequantize
    // For simplicity, read using ggml backend tensor_get row by row
    size_t row_bytes = ggml_row_size(state.token_embd->type, n_embd);
    std::vector<uint8_t> row_buf(row_bytes);
    std::vector<float> row_f32(n_embd);

    // Fill prefix embeddings
    for (int i = 0; i < n_prefix; i++) {
        int tok = prefix_tokens[i];
        ggml_backend_tensor_get(state.token_embd, row_buf.data(),
            (size_t)tok * row_bytes, row_bytes);
        // Dequantize to F32
        ggml_get_type_traits(state.token_embd->type)->to_float(
            row_buf.data(), row_f32.data(), n_embd);
        memcpy(&merged_embeds[i * n_embd], row_f32.data(), n_embd * sizeof(float));
    }

    // Fill vision embeddings (already F32)
    for (int i = 0; i < n_vision_tokens; i++) {
        memcpy(&merged_embeds[(n_prefix + i) * n_embd],
               &vision_embeds[i * n_embd],
               n_embd * sizeof(float));
    }

    // Fill suffix embeddings
    for (int i = 0; i < n_suffix; i++) {
        int tok = suffix_tokens[i];
        ggml_backend_tensor_get(state.token_embd, row_buf.data(),
            (size_t)tok * row_bytes, row_bytes);
        ggml_get_type_traits(state.token_embd->type)->to_float(
            row_buf.data(), row_f32.data(), n_embd);
        memcpy(&merged_embeds[(n_prefix + n_vision_tokens + i) * n_embd],
               row_f32.data(), n_embd * sizeof(float));
    }

    double merge_ms = t.ms();
    LOG_KV("merged shape", "[%d, %d]", n_embd, total_seq);
    LOG_KV("merge time", "%.1f ms", merge_ms);

    // Quick stats
    float sum = 0, min_v = 1e9, max_v = -1e9;
    for (int i = 0; i < total_seq * n_embd; i++) {
        float v = merged_embeds[i];
        sum += v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    LOG_KV("embed stats", "mean=%.4f min=%.4f max=%.4f",
        sum / (total_seq * n_embd), min_v, max_v);
    LOG_OK("Embedding merge complete");
    LOG_END();

    // --- VLM Prefill ---
    LOG_SUBSECTION("VLM Prefill");
    t.reset();

    int kv_pos = 0;
    int kv_len = total_seq;

    size_t ctx_size = vlm_prefill_ctx_size((int)cfg.n_layer);
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph * graph = build_vlm_prefill_graph(ctx, state,
        total_seq, kv_pos, kv_len);

    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        LOG_FAIL("VLM prefill graph alloc failed");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    LOG_KV("graph nodes", "%d", ggml_graph_n_nodes(graph));
    LOG_KV("compute buffer", "%.2f MB",
        ggml_gallocr_get_buffer_size(galloc, 0) / 1024.0 / 1024.0);

    // Set inputs
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "inp_embd"),
        merged_embeds.data(), 0, total_seq * n_embd * sizeof(float));

    std::vector<int32_t> positions(total_seq);
    std::iota(positions.begin(), positions.end(), 0);
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "inp_pos"),
        positions.data(), 0, total_seq * sizeof(int32_t));

    std::vector<uint16_t> mask;
    build_causal_mask(mask, kv_len, total_seq, kv_pos);
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "attn_mask"),
        mask.data(), 0, kv_len * total_seq * sizeof(uint16_t));

    // Compute prefill
    t.reset();
    ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    double prefill_ms = t.ms();

    // Extract last token logits
    struct ggml_tensor * logits_t = ggml_graph_get_tensor(graph, "logits");
    std::vector<float> logits(cfg.n_vocab);
    size_t last_offset = (size_t)(total_seq - 1) * cfg.n_vocab * sizeof(float);
    ggml_backend_tensor_get(logits_t, logits.data(), last_offset,
        cfg.n_vocab * sizeof(float));

    // Sample first token (temperature=0.1 for slight diversity, rep_penalty for anti-repetition)
    SamplingParams sp = { 0.1f, 40, 0.95f, 1.2f };
    std::mt19937 rng(42);
    std::vector<int32_t> generated_tokens; // for repetition penalty
    int first_token = sample_token(logits.data(), (int)cfg.n_vocab, sp, rng);
    std::string first_raw = (first_token < (int)state.vocab.size())
        ? state.vocab[first_token] : "???";
    std::string first_str = decode_token(first_raw);

    LOG_KV("prefill time", "%.1f ms", prefill_ms);
    LOG_KV("prefill speed", "%.1f ms/token (%.1f tok/s)",
        prefill_ms / total_seq, total_seq / (prefill_ms / 1000.0));
    LOG_KV("first token", "'%s' (id=%d, logit=%.2f)",
        first_str.c_str(), first_token, logits[first_token]);
    generated_tokens.push_back(first_token);
    LOG_OK("VLM prefill PASS");
    LOG_END();

    state.kv_pos = kv_len;

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    // --- Autoregressive decode ---
    LOG_SUBSECTION("Autoregressive Decode");
    t.reset();

    // Reserve graph allocator for decode (seq_len=1)
    ggml_gallocr_t decode_galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    {
        size_t dctx_size = compute_ctx_size((int)cfg.n_layer);
        struct ggml_init_params dp = { dctx_size, nullptr, true };
        struct ggml_context * measure_ctx = ggml_init(dp);
        struct ggml_cgraph * measure_graph = build_graph(measure_ctx, state,
            1, 0, (int)cfg.max_ctx, (int)cfg.n_layer, false);
        ggml_gallocr_reserve(decode_galloc, measure_graph);
        ggml_free(measure_ctx);
    }
    LOG_KV("decode compute buf", "%.2f MB",
        ggml_gallocr_get_buffer_size(decode_galloc, 0) / 1024.0 / 1024.0);

    std::string generated;
    int last_token = first_token;
    std::vector<double> decode_times;
    size_t decode_ctx_size = compute_ctx_size((int)cfg.n_layer);
    std::vector<uint8_t> decode_ctx_buf(decode_ctx_size);

    printf("  │   Generating: %s", first_str.c_str());
    fflush(stdout);

    for (int step = 0; step < test_cfg.max_tokens; step++) {
        if (last_token == state.eos_token) break;
        if (last_token == end_of_utterance_id) break;
        if (state.kv_pos >= (int)cfg.max_ctx - 1) break;

        kv_pos = state.kv_pos;
        kv_len = kv_pos + 1;

        struct ggml_init_params dp = { decode_ctx_size, decode_ctx_buf.data(), true };
        struct ggml_context * dctx = ggml_init(dp);
        struct ggml_cgraph * dgraph = build_graph(dctx, state, 1, kv_pos, kv_len,
            (int)cfg.n_layer, false);

        if (!ggml_gallocr_alloc_graph(decode_galloc, dgraph)) {
            LOG_FAIL("Decode graph alloc failed at step %d", step);
            ggml_free(dctx);
            break;
        }

        // Set input token
        ggml_backend_tensor_set(ggml_graph_get_tensor(dgraph, "inp_tokens"),
            &last_token, 0, sizeof(int32_t));

        int32_t pos = kv_pos;
        ggml_backend_tensor_set(ggml_graph_get_tensor(dgraph, "inp_pos"),
            &pos, 0, sizeof(int32_t));

        std::vector<uint16_t> dmask;
        build_causal_mask(dmask, kv_len, 1, kv_pos);
        ggml_backend_tensor_set(ggml_graph_get_tensor(dgraph, "attn_mask"),
            dmask.data(), 0, kv_len * sizeof(uint16_t));

        Timer dt;
        ggml_backend_graph_compute(backend, dgraph);
        ggml_backend_synchronize(backend);
        double tok_ms = dt.ms();
        decode_times.push_back(tok_ms);

        // Get logits
        struct ggml_tensor * dl = ggml_graph_get_tensor(dgraph, "logits");
        ggml_backend_tensor_get(dl, logits.data(), 0, cfg.n_vocab * sizeof(float));

        // Apply repetition penalty (penalize tokens that appeared in recent context)
        if (sp.rep_penalty != 1.0f) {
            for (int32_t prev_tok : generated_tokens) {
                if (prev_tok >= 0 && prev_tok < (int)cfg.n_vocab) {
                    if (logits[prev_tok] > 0) {
                        logits[prev_tok] /= sp.rep_penalty;
                    } else {
                        logits[prev_tok] *= sp.rep_penalty;
                    }
                }
            }
        }

        last_token = sample_token(logits.data(), (int)cfg.n_vocab, sp, rng);
        generated_tokens.push_back(last_token);

        std::string raw_str = (last_token < (int)state.vocab.size())
            ? state.vocab[last_token] : "???";
        std::string tok_str = decode_token(raw_str);

        generated += tok_str;
        printf("%s", tok_str.c_str());
        fflush(stdout);

        state.kv_pos = kv_len;
        ggml_free(dctx);
    }
    printf("\n");

    double total_decode_ms = t.ms();

    // Decode stats
    if (!decode_times.empty()) {
        double avg = 0;
        for (double d : decode_times) avg += d;
        avg /= decode_times.size();

        double min_t = *std::min_element(decode_times.begin(), decode_times.end());
        double max_t = *std::max_element(decode_times.begin(), decode_times.end());

        LOG_KV("tokens generated", "%zu", decode_times.size());
        LOG_KV("total decode time", "%.1f ms", total_decode_ms);
        LOG_KV("avg ms/token", "%.1f ms", avg);
        LOG_KV("min/max ms/token", "%.1f / %.1f ms", min_t, max_t);
        LOG_KV("throughput", "%.1f tok/s", decode_times.size() / (total_decode_ms / 1000.0));
    }
    LOG_OK("VLM decode PASS");
    LOG_END();

    ggml_gallocr_free(decode_galloc);
    return true;
}

// ===========================================================================
// Stage 8: Summary
// ===========================================================================
static void print_summary(ModelState & state, VisionModelState & vs,
                          double total_ms) {
    LOG_SECTION("STAGE 8: Summary");

    LOG_SUBSECTION("Memory Usage");
    double weight_mb = state.weight_buf
        ? ggml_backend_buffer_get_size(state.weight_buf) / 1024.0 / 1024.0 : 0;
    double kv_mb = state.kv_buf
        ? ggml_backend_buffer_get_size(state.kv_buf) / 1024.0 / 1024.0 : 0;
    double vis_mb = vs.weight_buf
        ? ggml_backend_buffer_get_size(vs.weight_buf) / 1024.0 / 1024.0 : 0;

    LOG_KV("LLM weights", "%.1f MB", weight_mb);
    LOG_KV("Vision weights", "%.1f MB", vis_mb);
    LOG_KV("KV cache", "%.1f MB", kv_mb);
    LOG_KV("TOTAL", "%.1f MB", weight_mb + vis_mb + kv_mb);
    LOG_END();

    LOG_SUBSECTION("Total Time");
    LOG_KV("end-to-end", "%.1f ms (%.1f s)", total_ms, total_ms / 1000.0);
    LOG_END();

    printf("\n=== VLM Pipeline Test Complete ===\n\n");
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char ** argv) {
    printf("\n");
    printf("  ╦  ╦╦  ╔╦╗  ╔═╗┬┌─┐┌─┐┬  ┬┌┐┌┌─┐  ╔╦╗┌─┐┌─┐┌┬┐\n");
    printf("  ╚╗╔╝║  ║║║  ╠═╝│├─┘├┤ │  ││││├┤    ║ ├┤ └─┐ │ \n");
    printf("   ╚╝ ╩═╝╩ ╩  ╩  ┴┴  └─┘┴─┘┴┘└┘└─┘   ╩ └─┘└─┘ ┴ \n");
    printf("  SmolVLM-500M on Android — Raw GGML Pipeline\n\n");

    VLMTestConfig test_cfg;
    if (!parse_args(argc, argv, test_cfg)) return 1;

    Timer total;

    // Stage 1: Backend
    ggml_backend_t backend = init_backend(test_cfg.threads);
    if (!backend) return 1;

    // Stage 2: LLM
    ModelState state = {};
    if (!load_llm(state, test_cfg.model_path, backend, test_cfg.verbose)) return 1;

    // Stage 3: Vision
    VisionModelState vs = {};
    if (!load_vision(vs, test_cfg.mmproj_path, backend, test_cfg.verbose)) return 1;

    // Stage 4: KV cache
    if (!setup_kv_cache(state, backend)) return 1;

    // Dump chat template from GGUF if available
    {
        const char * chat_tmpl = gguf_get_str_val(state.gguf_ctx,
            "tokenizer.chat_template", nullptr);
        if (chat_tmpl) {
            printf("\n  Chat template from GGUF:\n  %s\n", chat_tmpl);
        } else {
            printf("\n  [INFO] No chat_template in GGUF metadata\n");
        }
    }

    // Stage 5: Vision encode
    std::vector<float> vision_embeds;
    int n_vision_tokens = 0, vision_dim = 0;
    bool vision_ok = run_vision_encode(vs, backend, vision_embeds,
        n_vision_tokens, vision_dim, test_cfg.image_path, test_cfg.verbose);

    // Stage 6: Text-only sanity check
    bool text_ok = run_text_only_test(state, backend, test_cfg.verbose);

    // Stage 7: Full VLM pipeline (only if both vision and text work)
    bool vlm_ok = false;
    if (vision_ok && text_ok) {
        vlm_ok = run_vlm_pipeline(state, vs, backend,
            vision_embeds, n_vision_tokens, vision_dim, test_cfg);
    } else {
        printf("\n  SKIPPING Stage 7: Vision=%s Text=%s\n",
            vision_ok ? "OK" : "FAIL", text_ok ? "OK" : "FAIL");
    }

    // Stage 8: Summary
    print_summary(state, vs, total.ms());

    // Cleanup
    free_vision_model(vs);
    if (state.kv_buf) ggml_backend_buffer_free(state.kv_buf);
    if (state.kv_ctx) ggml_free(state.kv_ctx);
    if (state.weight_buf) ggml_backend_buffer_free(state.weight_buf);
    if (state.weight_ctx) ggml_free(state.weight_ctx);
    if (state.data_ctx) ggml_free(state.data_ctx);
    if (state.gguf_ctx) gguf_free(state.gguf_ctx);
    ggml_backend_free(backend);

    return vlm_ok ? 0 : 1;
}
