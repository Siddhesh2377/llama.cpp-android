#pragma once

/**
 * GGMLEngine - Simplified LLM inference API for Android
 *
 * Wraps llama.cpp internals into a clean C API suitable for JNI binding.
 * Supports model loading via file path or Android SAF file descriptor.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct ggml_engine ggml_engine_t;

// Status codes
typedef enum {
    GGML_ENGINE_OK                 = 0,
    GGML_ENGINE_ERROR_LOAD_FAILED  = 1,
    GGML_ENGINE_ERROR_CONTEXT_FAIL = 2,
    GGML_ENGINE_ERROR_NO_MODEL     = 3,
    GGML_ENGINE_ERROR_TOKENIZE     = 4,
    GGML_ENGINE_ERROR_DECODE       = 5,
    GGML_ENGINE_ERROR_CANCELLED    = 6,
    GGML_ENGINE_ERROR_OUT_OF_MEM   = 7,
    GGML_ENGINE_ERROR_VLM_ENCODE   = 8,
    GGML_ENGINE_ERROR_VLM_NO_PROJ  = 9,
} ggml_engine_status;

// Engine configuration
typedef struct {
    int32_t  n_ctx;            // context size (0 = model default)
    int32_t  n_batch;          // batch size for prompt processing
    int32_t  n_threads;        // number of threads (0 = auto-detect)
    int32_t  n_threads_batch;  // threads for batch processing (0 = same as n_threads)
    bool     use_mmap;         // memory-map model file
    bool     use_mlock;        // lock model in memory
    int32_t  n_gpu_layers;     // always 0 for CPU-only
    float    rope_freq_base;   // RoPE base frequency (0 = model default)
    float    rope_freq_scale;  // RoPE frequency scale (0 = model default)
    bool     flash_attn;       // use flash attention if available
} ggml_engine_params;

// Sampling parameters
typedef struct {
    float    temperature;      // 0.0 = greedy
    int32_t  top_k;            // top-k sampling (0 = disabled)
    float    top_p;            // top-p (nucleus) sampling (1.0 = disabled)
    float    min_p;            // min-p sampling (0.0 = disabled)
    float    repeat_penalty;   // repetition penalty (1.0 = disabled)
    int32_t  repeat_last_n;   // last n tokens to penalize
    float    frequency_penalty; // frequency penalty
    float    presence_penalty;  // presence penalty
    uint32_t seed;             // random seed (0xFFFFFFFF = random)
    int32_t  n_predict;        // max tokens to generate (-1 = unlimited)
    const char * stop_sequences[8]; // up to 8 stop sequences (NULL terminated array)
    int32_t  stop_sequence_count;
} ggml_engine_sampling;

// Token callback: return false to stop generation
typedef bool (*ggml_engine_token_callback)(const char * token_text, void * user_data);

// Get default parameters
ggml_engine_params    ggml_engine_default_params(void);
ggml_engine_sampling  ggml_engine_default_sampling(void);

// Lifecycle
ggml_engine_t *     ggml_engine_create(ggml_engine_params params);
void                ggml_engine_free(ggml_engine_t * engine);

// Model loading
ggml_engine_status  ggml_engine_load_model(ggml_engine_t * engine, const char * path);
ggml_engine_status  ggml_engine_load_model_from_fd(ggml_engine_t * engine, int fd);
void                ggml_engine_unload_model(ggml_engine_t * engine);
bool                ggml_engine_is_loaded(const ggml_engine_t * engine);

// Model info - returns JSON string (caller must free with ggml_engine_free_string)
char *              ggml_engine_model_info_json(const ggml_engine_t * engine);
void                ggml_engine_free_string(char * str);

// Text generation
ggml_engine_status  ggml_engine_generate(
    ggml_engine_t * engine,
    const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback,
    void * user_data
);

// Cancel ongoing generation (thread-safe)
void                ggml_engine_cancel(ggml_engine_t * engine);

// Get full generated text from last generation (caller must free)
char *              ggml_engine_get_response(const ggml_engine_t * engine);

// Context management
void                ggml_engine_clear_context(ggml_engine_t * engine);
int32_t             ggml_engine_context_used(const ggml_engine_t * engine);
int32_t             ggml_engine_context_size(const ggml_engine_t * engine);

// Tokenization utilities
int32_t             ggml_engine_tokenize(const ggml_engine_t * engine,
                        const char * text, int32_t * tokens, int32_t max_tokens);
char *              ggml_engine_detokenize(const ggml_engine_t * engine,
                        const int32_t * tokens, int32_t n_tokens);

// Performance info
typedef struct {
    double   prompt_eval_ms;      // time to process prompt
    double   generation_ms;       // time to generate tokens
    int32_t  prompt_tokens;       // number of prompt tokens
    int32_t  generated_tokens;    // number of generated tokens
    double   prompt_tokens_per_sec;
    double   generation_tokens_per_sec;
} ggml_engine_perf;

ggml_engine_perf    ggml_engine_get_perf(const ggml_engine_t * engine);

// =====================================================
// VLM (Vision Language Model) API
// =====================================================

// Opaque VLM handle
typedef struct ggml_engine_vlm ggml_engine_vlm_t;

// VLM configuration
typedef struct {
    int32_t  n_threads;          // threads for vision encode (0 = same as engine)
    int32_t  image_min_tokens;   // -1 = model default
    int32_t  image_max_tokens;   // -1 = model default
} ggml_engine_vlm_params;

// Image data: either file bytes (width=0, height=0) or raw RGB pixels
typedef struct {
    const unsigned char * data;   // file bytes or RGB pixels
    size_t                size;   // byte count
    uint32_t              width;  // 0 = file mode (auto-detect JPEG/PNG/etc)
    uint32_t              height; // 0 = file mode
} ggml_engine_image;

// Get default VLM parameters
ggml_engine_vlm_params ggml_engine_vlm_default_params(void);

// Load vision projector (mmproj GGUF). Call after loading text model.
ggml_engine_vlm_t * ggml_engine_vlm_load(
    ggml_engine_t * engine, const char * mmproj_path,
    ggml_engine_vlm_params params);
ggml_engine_vlm_t * ggml_engine_vlm_load_from_fd(
    ggml_engine_t * engine, int fd,
    ggml_engine_vlm_params params);
void                ggml_engine_vlm_free(ggml_engine_vlm_t * vlm);
bool                ggml_engine_vlm_is_loaded(const ggml_engine_vlm_t * vlm);

// Generate from text + images. Use "<__media__>" markers in prompt for image positions.
ggml_engine_status  ggml_engine_vlm_generate(
    ggml_engine_t * engine, ggml_engine_vlm_t * vlm,
    const char * prompt,
    const ggml_engine_image * images, int32_t n_images,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);

// Encode image only (returns token count, -1 on error)
int32_t             ggml_engine_vlm_encode_image(
    ggml_engine_vlm_t * vlm, const ggml_engine_image * image);

// VLM info - returns JSON string (caller must free with ggml_engine_free_string)
char *              ggml_engine_vlm_info_json(const ggml_engine_vlm_t * vlm);

// Default image marker used in prompts
const char *        ggml_engine_vlm_default_marker(void);

// Capability queries
bool                ggml_engine_vlm_supports_vision(const ggml_engine_vlm_t * vlm);
bool                ggml_engine_vlm_supports_audio(const ggml_engine_vlm_t * vlm);

#ifdef __cplusplus
}
#endif
