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
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <map>
#include <unordered_map>

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
static constexpr int PREFILL_CHUNK = 64;
static constexpr int STALL_MAX_CTX = 32;
static constexpr int STALL_WAIT_MS = 150;
static constexpr int FW_DIM_REDUCED = 128;
static constexpr int MAX_CAPTURE   = 8;

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
// Intervention flags (bitfield)
// ---------------------------------------------------------------------------
enum InterventionFlags : uint32_t {
    IV_NONE              = 0,
    IV_CONTROL_VECTORS   = 1 << 0,
    IV_ATTN_TEMPERATURE  = 1 << 1,
    IV_GATED_RESIDUAL    = 1 << 2,
    IV_LOGIT_BIAS        = 1 << 3,
    IV_NORM_SHIFT        = 1 << 4,
    IV_HEAD_RESCALE      = 1 << 5,
    IV_CAPTURE           = 1 << 6,
    IV_EARLY_EXIT        = 1 << 7,
};

// ---------------------------------------------------------------------------
// Intervention configuration (scalar params — zero overhead in graph)
// ---------------------------------------------------------------------------
struct InterventionConfig {
    uint32_t flags = IV_NONE;

    float attn_temp[MAX_LAYERS];
    float attn_gate[MAX_LAYERS];
    float ffn_gate[MAX_LAYERS];

    int capture_layers[MAX_CAPTURE];
    int n_capture = 0;

    int early_exit_layer = -1;

    void reset() {
        flags = IV_NONE;
        memset(attn_temp, 0, sizeof(attn_temp));
        memset(attn_gate, 0, sizeof(attn_gate));
        memset(ffn_gate, 0, sizeof(ffn_gate));
        memset(capture_layers, -1, sizeof(capture_layers));
        n_capture = 0;
        early_exit_layer = -1;
    }
};

// ---------------------------------------------------------------------------
// Intervention tensors (allocated once, updated at runtime)
// ---------------------------------------------------------------------------
struct InterventionTensors {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    struct ggml_tensor * control_vector[MAX_LAYERS] = {};
    struct ggml_tensor * norm_shift[MAX_LAYERS] = {};
    struct ggml_tensor * head_scale[MAX_LAYERS] = {};
    struct ggml_tensor * logit_bias = nullptr;

    bool allocated = false;
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
// Control vector specification
// ---------------------------------------------------------------------------
struct ControlVectorSpec {
    const char * path;
    float strength;   // -1.0 to 1.0
};

// ---------------------------------------------------------------------------
// Personality configuration (loaded from JSON)
// ---------------------------------------------------------------------------
struct PersonalityConfig {
    std::string name;
    std::string system_prompt;
    std::string user_message;

    // Per-layer intervention params
    float temp_early   = 1.3f;
    float temp_mid     = 1.0f;
    float temp_late    = 0.8f;
    float attn_gate_mid = 0.90f;
    float ffn_gate_mid  = 0.93f;
    float logit_bias_eos = -4.0f;

    // Sampling
    float sampling_temp  = 0.85f;
    int   sampling_top_k = 50;
    float sampling_top_p = 0.93f;
    float rep_penalty    = 1.20f;
    int   thinking       = 0;

    // Control vectors
    std::string cv_paths[4];
    float cv_strengths[4];
    int n_cv = 0;

    // Mood baselines
    float mood_warmth    = 0.5f;
    float mood_energy    = 0.5f;
    float mood_formality = 0.5f;

    // Stall config
    std::string stall_prompt;
    int stall_max_tokens = 20;

    // Fast weight config
    int   fw_dim_reduced = FW_DIM_REDUCED;
    float fw_gamma       = 0.95f;
    float fw_eta         = 0.01f;
    bool  fw_enabled     = false;
};

// ---------------------------------------------------------------------------
// Profile state (serializable, hot-swappable)
// ---------------------------------------------------------------------------
struct ProfileState {
    PersonalityConfig personality;
    InterventionConfig iv_config;
    SamplingParams sampling;
    bool active = false;
};

// ---------------------------------------------------------------------------
// Emotional state (3 axes, EMA update)
// ---------------------------------------------------------------------------
enum MoodAxis : uint8_t {
    MOOD_WARMTH = 0,
    MOOD_ENERGY,
    MOOD_FORMALITY,
    MOOD_COUNT
};

struct EmotionalState {
    float axes[MOOD_COUNT]     = { 0.5f, 0.5f, 0.5f };  // current values
    float baseline[MOOD_COUNT] = { 0.7f, 0.6f, 0.3f };  // persona defaults
    float alpha      = 0.85f;   // EMA decay
    float max_offset = 0.3f;    // clamp ±0.3 from baseline
    int   updates    = 0;
};

struct MoodKeyword {
    const char * word;
    MoodAxis axis;
    float delta;
};

// ---------------------------------------------------------------------------
// Fast weight associative memory
// ---------------------------------------------------------------------------
struct FastWeightMemory {
    int d_model   = 0;
    int d_reduced = FW_DIM_REDUCED;
    float gamma   = 0.95f;    // decay
    float eta     = 0.01f;    // learning rate

    std::vector<float> proj_down;   // [d_reduced × d_model]
    std::vector<float> proj_up;     // [d_model × d_reduced]
    std::vector<float> W_fast;      // [d_reduced × d_reduced]
    std::vector<float> h_reduced;   // [d_reduced] temp
    std::vector<float> recall;      // [d_reduced] temp
    std::vector<float> recall_full; // [d_model] output

    bool initialized = false;
};

// ---------------------------------------------------------------------------
// Boundary system
// ---------------------------------------------------------------------------
enum BoundaryAction {
    BOUNDARY_NONE = 0,
    BOUNDARY_STOP,
    BOUNDARY_PAUSE_RESUME,
};

struct BoundaryEvent {
    BoundaryAction action = BOUNDARY_NONE;
    int kv_pos = 0;
    int tokens_generated = 0;
    std::string text;
    std::string name;
};

struct GenerationState {
    int kv_pos = 0;
    int last_token = -1;
    int tokens_generated = 0;
    std::string full_text;
    std::mt19937 rng;
    bool initialized = false;
};

// ---------------------------------------------------------------------------
// Tool calling
// ---------------------------------------------------------------------------
struct ToolParam {
    std::string name;
    std::string type;
    std::string description;
    bool required = false;
    std::vector<std::string> enum_values;
};

struct ToolDef {
    std::string name;
    std::string description;
    std::vector<ToolParam> params;
};

struct ToolCallResult {
    std::string name;
    std::string arguments_json;
    bool valid = false;
};

// ---------------------------------------------------------------------------
// RAG types
// ---------------------------------------------------------------------------
struct RagChunk {
    int id = -1;
    std::string text;
    std::string source;
    std::vector<float> embedding;
    std::unordered_map<std::string, int> term_freq;
    int n_terms = 0;
    int64_t timestamp = 0;
};

struct KgTriple {
    std::string subject;
    std::string relation;
    std::string object;
    int source_chunk_id = -1;
};

struct MemoryEntry {
    int id = -1;
    std::string fact;
    std::string category;
    float importance = 0.5f;
    int64_t timestamp = 0;
    int access_count = 0;
    std::vector<float> embedding;
};

struct BM25Index {
    std::unordered_map<std::string, std::vector<std::pair<int, int>>> postings;
    float avg_doc_len = 0.0f;
    int n_docs = 0;
    float k1 = 1.2f;
    float b  = 0.75f;
};

struct RagState {
    std::vector<RagChunk> chunks;
    BM25Index bm25;
    std::vector<KgTriple> kg_triples;
    std::unordered_map<std::string, std::vector<int>> kg_entity_idx;
    std::vector<MemoryEntry> memories;
    int next_chunk_id = 0;
    int next_memory_id = 0;
    int n_embd = 0;   // set from model config
};

// ---------------------------------------------------------------------------
// Stall generation types
// ---------------------------------------------------------------------------
struct StallKV {
    struct ggml_tensor * kv_k[MAX_LAYERS] = {};
    struct ggml_tensor * kv_v[MAX_LAYERS] = {};
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int kv_pos = 0;
    bool allocated = false;
};

struct AsyncToolResult {
    std::mutex mtx;
    std::condition_variable cv;
    std::string result;
    bool done = false;
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

// ---------------------------------------------------------------------------
// Engine configuration (persistent, loaded from config.json)
// ---------------------------------------------------------------------------
struct EngineConfig {
    std::string model_path;
    std::string character_json;
    int threads   = 4;
    int gpu       = 0;
    int max_tokens = 256;
    int max_ctx    = MAX_CTX;
    float temp     = 0.85f;
    int top_k      = 50;
    float top_p    = 0.93f;
    float rep_penalty = 1.20f;
    bool color     = true;
    bool verbose   = false;
};
