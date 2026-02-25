// types.h — Core types, enums, and constants for gguf-engine
//
// All data structures used across the engine are defined here.
// Header-only: no .cpp file needed.

#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <chrono>

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------
#define ENGINE_VERSION "1.0.0"
#define ENGINE_NAME    "gguf-engine"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int MAX_LAYERS    = 64;
static constexpr int MAX_CTX       = 2048;

using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Architecture types
// ---------------------------------------------------------------------------
enum ArchType {
    ARCH_UNKNOWN = 0,
    ARCH_QWEN2,      // Qwen2, Qwen2.5, Qwen3 — SiLU, standard RMSNorm
    ARCH_GEMMA3,      // Gemma3 — GELUTanh, additive RMSNorm (1+w), embedding scale
};

// ---------------------------------------------------------------------------
// Model configuration (read from GGUF metadata)
// ---------------------------------------------------------------------------
struct ModelConfig {
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_ff;
    uint32_t n_vocab;
    uint32_t head_dim;
    uint32_t max_ctx;
    float    rms_eps;
    float    rope_freq_base;
    char     arch[64];
    ArchType arch_type;

    bool     gemma_norm;
    bool     embd_scale;
    bool     use_gelu;
    bool     has_qk_norm;
    int      rope_type;
};

// ---------------------------------------------------------------------------
// Layer weights (per transformer block)
// ---------------------------------------------------------------------------
struct LayerWeights {
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_q;
    struct ggml_tensor * attn_k;
    struct ggml_tensor * attn_v;
    struct ggml_tensor * attn_output;
    struct ggml_tensor * q_norm;
    struct ggml_tensor * k_norm;
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_gate;
    struct ggml_tensor * ffn_up;
    struct ggml_tensor * ffn_down;
};

// ---------------------------------------------------------------------------
// Model state (weights + KV cache + tokenizer)
// ---------------------------------------------------------------------------
struct ModelState {
    ModelConfig cfg;

    // Weights
    struct ggml_tensor * token_embd;
    struct ggml_tensor * output_norm;
    struct ggml_tensor * output;
    LayerWeights layers[MAX_LAYERS];
    struct ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;

    // KV cache
    struct ggml_tensor * kv_k[MAX_LAYERS];
    struct ggml_tensor * kv_v[MAX_LAYERS];
    struct ggml_context * kv_ctx;
    ggml_backend_buffer_t kv_buf;
    int kv_pos;

    // GGUF data (mmap'd)
    struct gguf_context * gguf_ctx;
    struct ggml_context * data_ctx;

    // Tokenizer
    std::vector<std::string> vocab;
    int bos_token;
    int eos_token;
};

// ---------------------------------------------------------------------------
// Sampling parameters
// ---------------------------------------------------------------------------
struct SamplingParams {
    float temp   = 0.0f;   // 0 = greedy (argmax)
    int   top_k  = 40;
    float top_p  = 0.95f;
    float rep_penalty = 1.0f;
};

// ---------------------------------------------------------------------------
// Vision types
// ---------------------------------------------------------------------------
struct VisionConfig {
    int n_embed = 0;
    int n_patch_height = 0;
    int n_patch_width = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_ff = 0;
    int image_size = 0;
    float rms_eps = 1e-6f;
};

struct VisionLayerWeights {
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_q;
    struct ggml_tensor * attn_k;
    struct ggml_tensor * attn_v;
    struct ggml_tensor * attn_output;
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_gate;
    struct ggml_tensor * ffn_up;
    struct ggml_tensor * ffn_down;
};

struct VisionModelState {
    VisionConfig cfg;
    std::vector<VisionLayerWeights> layers;
    struct ggml_tensor * patch_embd;
    struct ggml_tensor * pos_embd;
    struct ggml_tensor * output_norm;
    struct ggml_tensor * projection;
    struct ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    bool loaded = false;
};

