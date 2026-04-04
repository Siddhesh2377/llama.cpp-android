#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_engine ggml_engine_t;

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

typedef struct {
    int32_t  n_ctx;
    int32_t  n_batch;
    int32_t  n_threads;
    int32_t  n_threads_batch;
    bool     use_mmap;
    bool     use_mlock;
    int32_t  n_gpu_layers;
    float    rope_freq_base;
    float    rope_freq_scale;
    bool     flash_attn;
} ggml_engine_params;

typedef struct {
    float    temperature;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    repeat_penalty;
    int32_t  repeat_last_n;
    float    frequency_penalty;
    float    presence_penalty;
    uint32_t seed;
    int32_t  n_predict;
    const char * stop_sequences[8];
    int32_t  stop_sequence_count;
} ggml_engine_sampling;

// Return false to stop generation
typedef bool (*ggml_engine_token_callback)(const char * token_text, void * user_data);

ggml_engine_params    ggml_engine_default_params(void);
ggml_engine_sampling  ggml_engine_default_sampling(void);

ggml_engine_t *     ggml_engine_create(ggml_engine_params params);
void                ggml_engine_free(ggml_engine_t * engine);

ggml_engine_status  ggml_engine_load_model(ggml_engine_t * engine, const char * path);
ggml_engine_status  ggml_engine_load_model_from_fd(ggml_engine_t * engine, int fd);
void                ggml_engine_unload_model(ggml_engine_t * engine);
bool                ggml_engine_is_loaded(const ggml_engine_t * engine);

// Returns JSON string, caller must free with ggml_engine_free_string
char *              ggml_engine_model_info_json(const ggml_engine_t * engine);
void                ggml_engine_free_string(char * str);

// Clears KV cache, fresh context each call
ggml_engine_status  ggml_engine_generate(
    ggml_engine_t * engine,
    const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback,
    void * user_data
);

// Appends to existing KV cache, does NOT clear it
ggml_engine_status  ggml_engine_generate_continue(
    ggml_engine_t * engine,
    const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback,
    void * user_data
);

void                ggml_engine_cancel(ggml_engine_t * engine);

// Caller must free returned string
char *              ggml_engine_get_response(const ggml_engine_t * engine);

void                ggml_engine_clear_context(ggml_engine_t * engine);
int32_t             ggml_engine_context_used(const ggml_engine_t * engine);
int32_t             ggml_engine_context_size(const ggml_engine_t * engine);
int32_t             ggml_engine_context_remaining(const ggml_engine_t * engine);

typedef struct {
    int32_t  total;
    int32_t  used;
    int32_t  remaining;
    int32_t  prompt_estimate; // -1 if no prompt
    int32_t  after_prompt;    // -1 if no prompt
} ggml_engine_context_info;

// If prompt is non-NULL, also estimates token consumption
ggml_engine_context_info ggml_engine_context_status(const ggml_engine_t * engine,
                                                     const char * prompt);

int32_t             ggml_engine_tokenize(const ggml_engine_t * engine,
                        const char * text, int32_t * tokens, int32_t max_tokens);
char *              ggml_engine_detokenize(const ggml_engine_t * engine,
                        const int32_t * tokens, int32_t n_tokens);

typedef struct {
    double   prompt_eval_ms;
    double   generation_ms;
    int32_t  prompt_tokens;
    int32_t  generated_tokens;
    double   prompt_tokens_per_sec;
    double   generation_tokens_per_sec;
} ggml_engine_perf;

ggml_engine_perf    ggml_engine_get_perf(const ggml_engine_t * engine);

// Progress callback: 0.0 to 1.0
typedef void (*ggml_engine_progress_cb)(float progress, void * user_data);

typedef struct {
    float *  data;       // n_embd floats, mean hidden state vector
    int32_t  n_embd;
    int32_t  n_tokens;
} ggml_engine_vectors;

// Extract mean hidden state vector (does not touch engine KV cache). Caller must free.
ggml_engine_vectors * ggml_engine_calc_vectors(
    ggml_engine_t * engine,
    const char * prompt,
    ggml_engine_progress_cb progress,
    void * user_data
);

void ggml_engine_free_vectors(ggml_engine_vectors * v);

// Apply control vector uniformly across layers. il_start/il_end: -1 = all layers.
bool ggml_engine_apply_vectors(
    ggml_engine_t * engine,
    const ggml_engine_vectors * vectors,
    float strength,
    int32_t il_start,
    int32_t il_end
);

void ggml_engine_clear_vectors(ggml_engine_t * engine);

typedef struct ggml_engine_vlm ggml_engine_vlm_t;

typedef struct {
    int32_t  n_threads;        // 0 = same as engine
    int32_t  image_min_tokens; // -1 = model default
    int32_t  image_max_tokens; // -1 = model default
} ggml_engine_vlm_params;

typedef struct {
    const unsigned char * data;   // file bytes or RGB pixels
    size_t                size;
    uint32_t              width;  // 0 = file mode (auto-detect format)
    uint32_t              height; // 0 = file mode
} ggml_engine_image;

ggml_engine_vlm_params ggml_engine_vlm_default_params(void);

ggml_engine_vlm_t * ggml_engine_vlm_load(
    ggml_engine_t * engine, const char * mmproj_path,
    ggml_engine_vlm_params params);
ggml_engine_vlm_t * ggml_engine_vlm_load_from_fd(
    ggml_engine_t * engine, int fd,
    ggml_engine_vlm_params params);
void                ggml_engine_vlm_free(ggml_engine_vlm_t * vlm);
bool                ggml_engine_vlm_is_loaded(const ggml_engine_vlm_t * vlm);

// Use "<__media__>" markers in prompt for image positions
ggml_engine_status  ggml_engine_vlm_generate(
    ggml_engine_t * engine, ggml_engine_vlm_t * vlm,
    const char * prompt,
    const ggml_engine_image * images, int32_t n_images,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);

// Returns token count, -1 on error
int32_t             ggml_engine_vlm_encode_image(
    ggml_engine_vlm_t * vlm, const ggml_engine_image * image);

// Caller must free with ggml_engine_free_string
char *              ggml_engine_vlm_info_json(const ggml_engine_vlm_t * vlm);

const char *        ggml_engine_vlm_default_marker(void);

bool                ggml_engine_vlm_supports_vision(const ggml_engine_vlm_t * vlm);
bool                ggml_engine_vlm_supports_audio(const ggml_engine_vlm_t * vlm);

typedef enum {
    TN_ENGINE_LOG_ERROR = 0,
    TN_ENGINE_LOG_WARN  = 1,
    TN_ENGINE_LOG_INFO  = 2,
    TN_ENGINE_LOG_DEBUG = 3,
} tn_engine_log_level;

typedef void (*tn_engine_log_callback)(tn_engine_log_level level,
                                       const char * tag,
                                       const char * msg,
                                       void * user_data);

// Pass NULL to restore default (Android logcat / stderr)
void tn_engine_set_log_callback(tn_engine_log_callback cb, void * user_data);

void tn_engine_set_log_level(tn_engine_log_level max_level);

#ifdef __cplusplus
}
#endif
