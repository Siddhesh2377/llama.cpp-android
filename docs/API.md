# Engine API Reference

Complete C API reference for the Tool-Neuron engine components. All headers are in `engine/`.

---

## GGMLEngine (`ggml-engine.h`)

Core LLM inference engine. Handles model loading, text generation, context management, tokenization, VLM support, and thread mode control.

### Types

#### `ggml_engine_t`

Opaque engine handle. Created with `ggml_engine_create()`, destroyed with `ggml_engine_free()`.

#### `ggml_engine_status`

```c
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
```

#### `ggml_engine_params`

Engine configuration. Get defaults with `ggml_engine_default_params()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_ctx` | `int32_t` | 0 | Context size (0 = model default) |
| `n_batch` | `int32_t` | 0 | Prompt batch size (0 = set by thread_mode) |
| `n_threads` | `int32_t` | 0 | Generation threads (0 = set by thread_mode) |
| `n_threads_batch` | `int32_t` | 0 | Prompt-eval threads (0 = set by thread_mode) |
| `use_mmap` | `bool` | true | Memory-map model file |
| `use_mlock` | `bool` | false | Lock model in RAM (prevents paging) |
| `n_gpu_layers` | `int32_t` | 0 | Always 0 (CPU-only build) |
| `rope_freq_base` | `float` | 0.0 | RoPE base frequency (0 = model default) |
| `rope_freq_scale` | `float` | 0.0 | RoPE frequency scale (0 = model default) |
| `flash_attn` | `bool` | true | Flash attention (reduces KV memory ~20%) |
| `thread_mode` | `int32_t` | 1 | Thread mode: 0=power_saving, 1=balanced, 2=performance, -1=manual |

**Note:** When `thread_mode >= 0`, the engine auto-configures `n_threads`, `n_threads_batch`, and `n_batch` from the big.LITTLE topology of the device. Set `thread_mode = -1` and provide explicit values to override.

#### `ggml_engine_sampling`

Sampling parameters. Get defaults with `ggml_engine_default_sampling()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | 0.7 | Sampling temperature (0.0 = greedy) |
| `top_k` | `int32_t` | 40 | Top-k sampling (0 = disabled) |
| `top_p` | `float` | 0.95 | Nucleus sampling (1.0 = disabled) |
| `min_p` | `float` | 0.05 | Min-p sampling (0.0 = disabled) |
| `repeat_penalty` | `float` | 1.1 | Repetition penalty (1.0 = disabled) |
| `repeat_last_n` | `int32_t` | 64 | Window for repetition penalty |
| `frequency_penalty` | `float` | 0.0 | Frequency penalty |
| `presence_penalty` | `float` | 0.0 | Presence penalty |
| `seed` | `uint32_t` | 0xFFFFFFFF | Random seed (0xFFFFFFFF = random) |
| `n_predict` | `int32_t` | 256 | Max tokens to generate |
| `stop_sequences` | `const char*[8]` | NULL | Up to 8 stop strings |
| `stop_sequence_count` | `int32_t` | 0 | Number of active stop strings |

#### `ggml_engine_perf`

Performance metrics from the last generation.

| Field | Type | Description |
|-------|------|-------------|
| `prompt_eval_ms` | `double` | Time to process prompt (ms) |
| `generation_ms` | `double` | Time to generate tokens (ms) |
| `prompt_tokens` | `int32_t` | Number of prompt tokens |
| `generated_tokens` | `int32_t` | Number of generated tokens |
| `prompt_tokens_per_sec` | `double` | Prompt processing speed |
| `generation_tokens_per_sec` | `double` | Generation speed |

#### `ggml_engine_context_info`

Full context window status.

| Field | Type | Description |
|-------|------|-------------|
| `total` | `int32_t` | Total context capacity |
| `used` | `int32_t` | Tokens currently in KV cache |
| `remaining` | `int32_t` | Total minus used |
| `prompt_estimate` | `int32_t` | Estimated tokens for pending prompt (-1 if no prompt given) |
| `after_prompt` | `int32_t` | Remaining after prompt (-1 if no prompt given) |

#### `ggml_engine_device_info`

Device CPU topology (read-only, populated at runtime).

| Field | Type | Description |
|-------|------|-------------|
| `n_cores_total` | `int32_t` | Total online CPU cores |
| `n_perf_cores` | `int32_t` | Performance cores (>70% max freq) |
| `n_efficiency_cores` | `int32_t` | Efficiency cores |
| `max_freq_khz` | `int32_t` | Highest core frequency (kHz) |
| `min_freq_khz` | `int32_t` | Lowest core frequency (kHz) |

#### Callback Types

```c
// Streaming token callback. Return false to stop generation.
typedef bool (*ggml_engine_token_callback)(const char * token_text, void * user_data);

// Progress callback. Reports 0.0 to 1.0.
typedef void (*ggml_engine_progress_cb)(float progress, void * user_data);
```

### Functions

#### Defaults

```c
ggml_engine_params   ggml_engine_default_params(void);
ggml_engine_sampling ggml_engine_default_sampling(void);
```

#### Lifecycle

```c
ggml_engine_t * ggml_engine_create(ggml_engine_params params);
void            ggml_engine_free(ggml_engine_t * engine);
```

#### Model Loading

```c
ggml_engine_status ggml_engine_load_model(ggml_engine_t * engine, const char * path);
ggml_engine_status ggml_engine_load_model_from_fd(ggml_engine_t * engine, int fd);
void               ggml_engine_unload_model(ggml_engine_t * engine);
bool               ggml_engine_is_loaded(const ggml_engine_t * engine);
```

`load_model_from_fd` accepts an Android SAF file descriptor. Internally resolves `/proc/self/fd/<fd>`.

#### Model Information

```c
// Returns JSON string. Caller must free with ggml_engine_free_string.
char * ggml_engine_model_info_json(const ggml_engine_t * engine);
void   ggml_engine_free_string(char * str);
```

#### Text Generation

```c
// Generate text. Clears KV cache before processing.
ggml_engine_status ggml_engine_generate(
    ggml_engine_t * engine, const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);

// Generate text. Appends to existing KV cache (multi-turn conversation).
ggml_engine_status ggml_engine_generate_continue(
    ggml_engine_t * engine, const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);

// Cancel in-progress generation. Thread-safe.
void ggml_engine_cancel(ggml_engine_t * engine);

// Get full response text from last generation. Caller must free.
char * ggml_engine_get_response(const ggml_engine_t * engine);
```

#### Context Management

```c
void    ggml_engine_clear_context(ggml_engine_t * engine);
int32_t ggml_engine_context_used(const ggml_engine_t * engine);
int32_t ggml_engine_context_size(const ggml_engine_t * engine);
int32_t ggml_engine_context_remaining(const ggml_engine_t * engine);

// Full context status. Pass NULL for prompt to skip token estimation.
ggml_engine_context_info ggml_engine_context_status(
    const ggml_engine_t * engine, const char * prompt);
```

#### Tokenization

```c
// Returns token count, or -1 on error.
int32_t ggml_engine_tokenize(const ggml_engine_t * engine,
    const char * text, int32_t * tokens, int32_t max_tokens);

// Caller must free.
char * ggml_engine_detokenize(const ggml_engine_t * engine,
    const int32_t * tokens, int32_t n_tokens);
```

#### Thread Mode (big.LITTLE-aware)

```c
// Switch thread mode at runtime. Applies immediately to the live context.
// mode: 0 = power_saving, 1 = balanced, 2 = performance
void ggml_engine_set_thread_mode(ggml_engine_t * engine, int32_t mode);
```

Thread mode controls how inference threads are distributed across CPU cores:

| Mode | Value | Generation Threads | Batch Threads | n_batch | Core Pinning |
|------|-------|--------------------|---------------|---------|--------------|
| Power Saving | 0 | 1 | E-cores only | 128 | No |
| Balanced | 1 | 2 P-cores | All P-cores | 256 | Yes |
| Performance | 2 | min(4, P-cores) | All cores | 512 | Yes |

Expose mode directly to UI as a 0-2 seekbar value. No additional mapping needed.

#### Device & Memory Queries

```c
// Read device CPU topology (reads /sys/devices/system/cpu/ on Android).
ggml_engine_device_info ggml_engine_get_device_info(void);

// Available RAM in bytes (-1 on error). Reads /proc/meminfo on Android.
int64_t ggml_engine_available_ram(void);

// Maximum model file size (bytes) that fits given available RAM and context size.
// Accounts for KV cache and OS overhead.
int64_t ggml_engine_max_model_size(int64_t available_ram, int32_t n_ctx);

// Recommended n_batch for a given model file size and current free RAM.
int32_t ggml_engine_recommend_batch(int64_t model_size_bytes);
```

#### Performance

```c
ggml_engine_perf ggml_engine_get_perf(const ggml_engine_t * engine);
```

### Usage Example

```c
#include "ggml-engine.h"

bool on_token(const char * text, void * user) {
    printf("%s", text);
    fflush(stdout);
    return true;
}

int main() {
    ggml_engine_params params = ggml_engine_default_params();
    params.n_ctx = 2048;
    params.thread_mode = 2; // performance

    ggml_engine_t * engine = ggml_engine_create(params);

    // Query device before loading to pick appropriate model size
    ggml_engine_device_info dev = ggml_engine_get_device_info();
    int64_t ram = ggml_engine_available_ram();
    int64_t max_model = ggml_engine_max_model_size(ram, 2048);
    printf("Device: %d perf cores, %d eff cores, max model: %lld MB\n",
           dev.n_perf_cores, dev.n_efficiency_cores, (long long)max_model >> 20);

    ggml_engine_load_model(engine, "model.gguf");

    ggml_engine_sampling sampling = ggml_engine_default_sampling();
    sampling.temperature = 0.7f;
    sampling.n_predict = 256;

    // First turn
    ggml_engine_generate(engine, "Hello!", sampling, on_token, NULL);

    // Multi-turn: preserve KV cache
    ggml_engine_generate_continue(engine, "Tell me more.", sampling, on_token, NULL);

    // Switch to power saving mid-session
    ggml_engine_set_thread_mode(engine, 0);

    ggml_engine_perf perf = ggml_engine_get_perf(engine);
    printf("\n%.1f t/s\n", perf.generation_tokens_per_sec);

    ggml_engine_free(engine);
}
```

---

## VLM Support (`ggml-engine.h`)

Vision-language model support. Loads a vision projector (mmproj GGUF) alongside the text model. Supports 20+ architectures. CPU-only.

### Types

#### `ggml_engine_vlm_t`

Opaque VLM handle. Created with `ggml_engine_vlm_load()`, destroyed with `ggml_engine_vlm_free()`.

#### `ggml_engine_vlm_params`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_threads` | `int32_t` | 0 | Vision encoder threads (0 = same as engine) |
| `image_min_tokens` | `int32_t` | -1 | Min image tokens (-1 = model default) |
| `image_max_tokens` | `int32_t` | -1 | Max image tokens (-1 = model default) |

#### `ggml_engine_image`

| Field | Type | Description |
|-------|------|-------------|
| `data` | `const unsigned char *` | File bytes (JPEG/PNG) or raw RGB pixels |
| `size` | `size_t` | Byte count |
| `width` | `uint32_t` | Pixel width (0 = file mode, auto-detect format) |
| `height` | `uint32_t` | Pixel height (0 = file mode) |

When `width == 0 && height == 0`, the image is loaded as a compressed file (JPEG/PNG/etc.). When `width > 0 && height > 0`, `data` must be raw RGB24 pixels.

### Functions

```c
ggml_engine_vlm_params ggml_engine_vlm_default_params(void);

// Load vision projector. Must be called after loading the text model.
ggml_engine_vlm_t * ggml_engine_vlm_load(
    ggml_engine_t * engine, const char * mmproj_path,
    ggml_engine_vlm_params params);

// Load from Android SAF file descriptor.
ggml_engine_vlm_t * ggml_engine_vlm_load_from_fd(
    ggml_engine_t * engine, int fd,
    ggml_engine_vlm_params params);

void ggml_engine_vlm_free(ggml_engine_vlm_t * vlm);
bool ggml_engine_vlm_is_loaded(const ggml_engine_vlm_t * vlm);

// Generate from text + images. Place "<__media__>" markers in prompt for image positions.
// images may be NULL if n_images == 0.
ggml_engine_status ggml_engine_vlm_generate(
    ggml_engine_t * engine, ggml_engine_vlm_t * vlm,
    const char * prompt,
    const ggml_engine_image * images, int32_t n_images,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);

// Count tokens produced by encoding one image. Returns -1 on error.
int32_t ggml_engine_vlm_encode_image(
    ggml_engine_vlm_t * vlm, const ggml_engine_image * image);

// JSON info string. Caller must free with ggml_engine_free_string.
char * ggml_engine_vlm_info_json(const ggml_engine_vlm_t * vlm);

const char * ggml_engine_vlm_default_marker(void);
bool ggml_engine_vlm_supports_vision(const ggml_engine_vlm_t * vlm);
bool ggml_engine_vlm_supports_audio(const ggml_engine_vlm_t * vlm);
```

### Usage Example

```c
#include "ggml-engine.h"

bool on_token(const char * text, void * user) { printf("%s", text); return true; }

int main() {
    ggml_engine_params params = ggml_engine_default_params();
    ggml_engine_t * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, "smolvlm-500m.gguf");

    ggml_engine_vlm_t * vlm = ggml_engine_vlm_load(
        engine, "mmproj.gguf", ggml_engine_vlm_default_params());

    FILE * f = fopen("photo.jpg", "rb");
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); rewind(f);
    unsigned char * buf = malloc(sz);
    fread(buf, 1, sz, f); fclose(f);

    ggml_engine_image img = { .data = buf, .size = sz, .width = 0, .height = 0 };
    ggml_engine_sampling s = ggml_engine_default_sampling();
    s.n_predict = 256;

    ggml_engine_vlm_generate(engine, vlm,
        "<__media__>\nDescribe this image.",
        &img, 1, s, on_token, NULL);

    free(buf);
    ggml_engine_vlm_free(vlm);
    ggml_engine_free(engine);
}
```

### Supported Architectures

LLaVA, SigLIP (Gemma3-Vision), Qwen2-VL, Qwen3-VL, Pixtral, MiniCPM-V, InternVL, CogVLM, GLM4V, Llama4, MobileNetV5 (Gemma3n-Vision), Kimi-VL, Kimi-K2.5, SmolVLM, PaddleOCR, Nemotron-V2, YouTu-VL, Whisper (audio), Conformer (audio).

---

## RAG Engine (`rag-engine.h`)

Retrieval-augmented generation with late chunking and binary-quantized embeddings. Uses a dedicated embedding model. The index is independent of the LLM — survives model swaps.

### Types

#### `rag_engine_t`

Opaque handle. Created with `rag_engine_create()`, destroyed with `rag_engine_free()`.

#### `rag_engine_params`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_threads` | `int32_t` | 0 | Encoder threads (0 = auto) |
| `chunk_size` | `int32_t` | 256 | Tokens per chunk |
| `chunk_overlap` | `int32_t` | 32 | Overlap between adjacent chunks |
| `n_dims` | `int32_t` | 256 | Matryoshka embedding dim: 768/512/256/128 |
| `top_k` | `int32_t` | 32 | BQ Hamming candidates before re-rank |
| `top_n` | `int32_t` | 5 | Final results after cosine re-rank |
| `late_chunking` | `bool` | true | Context-aware chunking (recommended) |

#### `rag_result`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `const char *` | Matched chunk text |
| `doc_id` | `const char *` | Document identifier |
| `chunk_index` | `int32_t` | Chunk index within document |
| `score` | `float` | Cosine similarity (0.0–1.0) |

### Functions

```c
// Lifecycle
rag_engine_params rag_engine_default_params(void);
rag_engine_t *    rag_engine_create(rag_engine_params params);
void              rag_engine_free(rag_engine_t * engine);

// Embedding model
int32_t rag_engine_load_model(rag_engine_t * engine, const char * path);
int32_t rag_engine_load_model_from_fd(rag_engine_t * engine, int fd);
bool    rag_engine_is_loaded(const rag_engine_t * engine);

// Indexing (returns chunk count on success, -1 on error)
int32_t rag_engine_add_document(rag_engine_t * engine,
            const char * text, const char * doc_id);
int32_t rag_engine_remove_document(rag_engine_t * engine, const char * doc_id);
void    rag_engine_clear(rag_engine_t * engine);
int32_t rag_engine_document_count(const rag_engine_t * engine);
int32_t rag_engine_chunk_count(const rag_engine_t * engine);

// Retrieval (two-stage: BQ Hamming -> cosine re-rank)
// Returns NULL if no results. Caller must free with rag_engine_free_results.
rag_result * rag_engine_query(rag_engine_t * engine,
                 const char * query, int32_t * n_results);
void         rag_engine_free_results(rag_result * results, int32_t n);

// Build prompt with retrieved context injected. Caller must free.
// Returns NULL if engine or query is NULL.
char * rag_engine_build_prompt(rag_engine_t * engine,
           const char * query, const char * user_prompt);

// Engine info as JSON. Caller must free.
char * rag_engine_info_json(const rag_engine_t * engine);
void   rag_engine_free_string(char * str);
```

### Usage Example

```c
#include "rag-engine.h"

int main() {
    rag_engine_params params = rag_engine_default_params();
    params.n_dims = 256;
    rag_engine_t * rag = rag_engine_create(params);

    rag_engine_load_model(rag, "embeddinggemma-300m-q4.gguf");

    rag_engine_add_document(rag, "Mitochondria are the powerhouses...", "biology");
    rag_engine_add_document(rag, "The French Revolution began in 1789...", "history");

    int32_t n = 0;
    rag_result * results = rag_engine_query(rag, "cell energy", &n);
    for (int i = 0; i < n; i++)
        printf("[%.3f] %s: %s\n", results[i].score, results[i].doc_id, results[i].text);
    rag_engine_free_results(results, n);

    // Inject context directly into an LLM prompt
    char * prompt = rag_engine_build_prompt(rag, "cell energy", "Explain this to me.");
    // ... pass prompt to ggml_engine_generate ...
    rag_engine_free_string(prompt);

    rag_engine_free(rag);
}
```

### How It Works

1. **Late chunking** — full document embedded with bidirectional attention, then token embeddings split into chunks. Preserves cross-chunk context lost by naive chunking.
2. **Matryoshka truncation** — 768-dim embeddings truncated to `n_dims` without retraining. 3x memory saving at 256 dims.
3. **Binary quantization** — floats thresholded to 1-bit. 32x compression. Hamming distance for O(1)-per-bit candidate search.
4. **Two-stage retrieval** — BQ Hamming finds `top_k` candidates, cosine similarity re-ranks to `top_n` final results.
5. **Sliding window** — documents longer than model context are processed in overlapping windows with averaged overlap regions.

---

## Logging

Two interfaces: the internal `tn-log.h` used by engine code, and the public callback in `ggml-engine.h` for application-level log capture.

### Internal Logging (`tn-log.h`)

```c
enum tn_log_level : int32_t {
    TN_LOG_LEVEL_ERROR = 0,
    TN_LOG_LEVEL_WARN  = 1,
    TN_LOG_LEVEL_INFO  = 2,
    TN_LOG_LEVEL_DEBUG = 3,
};

typedef void (*tn_log_callback)(enum tn_log_level level,
    const char * tag, const char * msg, void * user_data);

// Thread-safe. Callback + user_data are updated atomically as a pair.
void tn_log_set_callback(tn_log_callback cb, void * user_data);
void tn_log_set_level(enum tn_log_level max_level);
void tn_log_write(enum tn_log_level level, const char * tag, const char * fmt, ...);
```

Convenience macros (tag = `__FILE__`):

```c
TN_LOG_ERR(fmt, ...)
TN_LOG_WRN(fmt, ...)
TN_LOG_INF(fmt, ...)
TN_LOG_DBG(fmt, ...)
```

Default sink: Android logcat on Android, stderr/stdout on other platforms.

### Public Log Callback (`ggml-engine.h`)

```c
typedef enum {
    TN_ENGINE_LOG_ERROR = 0,
    TN_ENGINE_LOG_WARN  = 1,
    TN_ENGINE_LOG_INFO  = 2,
    TN_ENGINE_LOG_DEBUG = 3,
} tn_engine_log_level;

typedef void (*tn_engine_log_callback)(tn_engine_log_level level,
    const char * tag, const char * msg, void * user_data);

// Pass NULL to restore default sink.
void tn_engine_set_log_callback(tn_engine_log_callback cb, void * user_data);
void tn_engine_set_log_level(tn_engine_log_level max_level);
```

### Usage

```c
void my_logger(tn_engine_log_level level, const char * tag,
               const char * msg, void * user) {
    const char * prefix[] = { "ERR", "WRN", "INF", "DBG" };
    fprintf(stderr, "[%s] %s: %s\n", prefix[level], tag, msg);
}

tn_engine_set_log_callback(my_logger, NULL);
tn_engine_set_log_level(TN_ENGINE_LOG_INFO);
```
