# Engine API Reference

Complete C API reference for the Tool-Neuron engine components. All headers are in `engine/`.

---

## GGMLEngine (`ggml-engine.h`)

Core LLM inference engine. Handles model loading, text generation, context management, tokenization, control vectors, and VLM support.

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
| `n_batch` | `int32_t` | 512 | Batch size for prompt processing |
| `n_threads` | `int32_t` | 0 | Thread count (0 = auto-detect) |
| `n_threads_batch` | `int32_t` | 0 | Threads for batch processing (0 = same as `n_threads`) |
| `use_mmap` | `bool` | true | Memory-map model file |
| `use_mlock` | `bool` | false | Lock model in memory |
| `n_gpu_layers` | `int32_t` | 0 | Always 0 (CPU-only) |
| `rope_freq_base` | `float` | 0.0 | RoPE base frequency (0 = model default) |
| `rope_freq_scale` | `float` | 0.0 | RoPE frequency scale (0 = model default) |
| `flash_attn` | `bool` | false | Flash attention |

#### `ggml_engine_sampling`

Sampling parameters. Get defaults with `ggml_engine_default_sampling()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | 0.8 | Sampling temperature (0.0 = greedy) |
| `top_k` | `int32_t` | 40 | Top-k sampling (0 = disabled) |
| `top_p` | `float` | 0.95 | Nucleus sampling (1.0 = disabled) |
| `min_p` | `float` | 0.05 | Min-p sampling (0.0 = disabled) |
| `repeat_penalty` | `float` | 1.1 | Repetition penalty (1.0 = disabled) |
| `repeat_last_n` | `int32_t` | 64 | Window for repetition penalty |
| `frequency_penalty` | `float` | 0.0 | Frequency penalty |
| `presence_penalty` | `float` | 0.0 | Presence penalty |
| `seed` | `uint32_t` | 0xFFFFFFFF | Random seed (0xFFFFFFFF = random) |
| `n_predict` | `int32_t` | -1 | Max tokens (-1 = unlimited) |
| `stop_sequences` | `const char*[8]` | NULL | Up to 8 stop sequences |
| `stop_sequence_count` | `int32_t` | 0 | Number of stop sequences |

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

Full context window status, with optional prompt estimation.

| Field | Type | Description |
|-------|------|-------------|
| `total` | `int32_t` | Total context capacity |
| `used` | `int32_t` | Tokens currently in KV cache |
| `remaining` | `int32_t` | Total minus used |
| `prompt_estimate` | `int32_t` | Estimated tokens for pending prompt (-1 if no prompt) |
| `after_prompt` | `int32_t` | Remaining after prompt (-1 if no prompt) |

#### `ggml_engine_vectors`

Mean hidden-state vector extracted from a prompt.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `float *` | `n_embd` floats, mean hidden-state vector |
| `n_embd` | `int32_t` | Embedding dimension |
| `n_tokens` | `int32_t` | Number of tokens processed |

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

#### Model Information

```c
// Returns JSON string with model metadata. Caller must free.
char * ggml_engine_model_info_json(const ggml_engine_t * engine);
void   ggml_engine_free_string(char * str);
```

#### Text Generation

```c
// Generate text from prompt. Clears KV cache first.
ggml_engine_status ggml_engine_generate(
    ggml_engine_t * engine, const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);

// Generate text from prompt. Appends to existing KV cache (multi-turn).
ggml_engine_status ggml_engine_generate_continue(
    ggml_engine_t * engine, const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);

// Cancel ongoing generation (thread-safe).
void ggml_engine_cancel(ggml_engine_t * engine);

// Get full text from last generation. Caller must free.
char * ggml_engine_get_response(const ggml_engine_t * engine);
```

#### Context Management

```c
void    ggml_engine_clear_context(ggml_engine_t * engine);
int32_t ggml_engine_context_used(const ggml_engine_t * engine);
int32_t ggml_engine_context_size(const ggml_engine_t * engine);
int32_t ggml_engine_context_remaining(const ggml_engine_t * engine);

// Full context status with optional prompt estimation.
// Pass NULL for prompt to skip estimation.
ggml_engine_context_info ggml_engine_context_status(
    const ggml_engine_t * engine, const char * prompt);
```

#### Tokenization

```c
// Returns number of tokens, or -1 on error.
int32_t ggml_engine_tokenize(const ggml_engine_t * engine,
    const char * text, int32_t * tokens, int32_t max_tokens);

// Caller must free returned string.
char * ggml_engine_detokenize(const ggml_engine_t * engine,
    const int32_t * tokens, int32_t n_tokens);
```

#### Control Vectors

Extract and apply control vectors (representation engineering) for steering model behavior at the hidden-state level.

```c
// Extract mean hidden-state vector from a prompt. Caller must free.
ggml_engine_vectors * ggml_engine_calc_vectors(
    ggml_engine_t * engine, const char * prompt,
    ggml_engine_progress_cb progress, void * user_data);

void ggml_engine_free_vectors(ggml_engine_vectors * v);

// Apply control vector uniformly across layers.
// il_start/il_end: -1 = all layers. Returns false on failure.
bool ggml_engine_apply_vectors(
    ggml_engine_t * engine, const ggml_engine_vectors * vectors,
    float strength, int32_t il_start, int32_t il_end);

void ggml_engine_clear_vectors(ggml_engine_t * engine);
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
    return true;
}

int main() {
    ggml_engine_params params = ggml_engine_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;

    ggml_engine_t * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, "model.gguf");

    ggml_engine_sampling sampling = ggml_engine_default_sampling();
    sampling.temperature = 0.7;
    sampling.n_predict = 256;

    // First turn clears KV cache
    ggml_engine_generate(engine, "Hello, world!", sampling, on_token, NULL);

    // Continue conversation without clearing cache
    ggml_engine_generate_continue(engine, "Tell me more.", sampling, on_token, NULL);

    // Check context usage
    ggml_engine_context_info info = ggml_engine_context_status(engine, "next prompt");
    printf("\nContext: %d/%d used, prompt ~%d tokens\n",
           info.used, info.total, info.prompt_estimate);

    ggml_engine_perf perf = ggml_engine_get_perf(engine);
    printf("%.1f tokens/sec\n", perf.generation_tokens_per_sec);

    ggml_engine_free(engine);
}
```

---

## VLM Support (`ggml-engine.h`)

Vision-language model support. Loads a vision projector (mmproj GGUF) and generates text from images and text prompts. Supports 20+ VLM architectures. CPU-only.

### Types

#### `ggml_engine_vlm_t`

Opaque VLM handle. Created with `ggml_engine_vlm_load()`, destroyed with `ggml_engine_vlm_free()`.

#### `ggml_engine_vlm_params`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_threads` | `int32_t` | 0 | Threads for vision encode (0 = same as engine) |
| `image_min_tokens` | `int32_t` | -1 | Min image tokens (-1 = model default) |
| `image_max_tokens` | `int32_t` | -1 | Max image tokens (-1 = model default) |

#### `ggml_engine_image`

| Field | Type | Description |
|-------|------|-------------|
| `data` | `const unsigned char *` | File bytes (JPEG/PNG/etc.) or raw RGB pixels |
| `size` | `size_t` | Byte count |
| `width` | `uint32_t` | Pixel width (0 = file mode, auto-detect) |
| `height` | `uint32_t` | Pixel height (0 = file mode) |

### Functions

```c
ggml_engine_vlm_params ggml_engine_vlm_default_params(void);

// Load vision projector. Call after loading text model.
ggml_engine_vlm_t * ggml_engine_vlm_load(
    ggml_engine_t * engine, const char * mmproj_path,
    ggml_engine_vlm_params params);
ggml_engine_vlm_t * ggml_engine_vlm_load_from_fd(
    ggml_engine_t * engine, int fd,
    ggml_engine_vlm_params params);
void ggml_engine_vlm_free(ggml_engine_vlm_t * vlm);
bool ggml_engine_vlm_is_loaded(const ggml_engine_vlm_t * vlm);

// Generate from text + images. Use "<__media__>" markers for image positions.
ggml_engine_status ggml_engine_vlm_generate(
    ggml_engine_t * engine, ggml_engine_vlm_t * vlm,
    const char * prompt,
    const ggml_engine_image * images, int32_t n_images,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);

// Encode image only. Returns token count, -1 on error.
int32_t ggml_engine_vlm_encode_image(
    ggml_engine_vlm_t * vlm, const ggml_engine_image * image);

// VLM info as JSON. Caller must free with ggml_engine_free_string.
char * ggml_engine_vlm_info_json(const ggml_engine_vlm_t * vlm);

const char * ggml_engine_vlm_default_marker(void);
bool ggml_engine_vlm_supports_vision(const ggml_engine_vlm_t * vlm);
bool ggml_engine_vlm_supports_audio(const ggml_engine_vlm_t * vlm);
```

### Usage Example

```c
#include "ggml-engine.h"

bool on_token(const char * text, void * user) {
    printf("%s", text);
    return true;
}

int main() {
    ggml_engine_params params = ggml_engine_default_params();
    params.n_ctx = 2048;
    ggml_engine_t * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, "smolvlm-500m.gguf");

    ggml_engine_vlm_params vp = ggml_engine_vlm_default_params();
    ggml_engine_vlm_t * vlm = ggml_engine_vlm_load(engine, "mmproj.gguf", vp);

    // Load image bytes
    FILE * f = fopen("photo.jpg", "rb");
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char * buf = malloc(sz);
    fread(buf, 1, sz, f); fclose(f);

    ggml_engine_image img = { .data = buf, .size = sz, .width = 0, .height = 0 };
    ggml_engine_sampling sampling = ggml_engine_default_sampling();
    sampling.n_predict = 256;

    ggml_engine_vlm_generate(engine, vlm, "<__media__>\nDescribe this image.",
                             &img, 1, sampling, on_token, NULL);

    free(buf);
    ggml_engine_vlm_free(vlm);
    ggml_engine_free(engine);
}
```

### Supported Architectures

LLaVA, SigLIP (Gemma3-Vision), Qwen2-VL, Qwen3-VL, Pixtral, MiniCPM-V, InternVL, CogVLM, GLM4V, Llama4, MobileNetV5 (Gemma3n-Vision), Kimi-VL, Kimi-K2.5, SmolVLM, PaddleOCR, Nemotron-V2, YouTu-VL, Whisper, Conformer.

---

## RAG Engine (`rag-engine.h`)

Retrieval-augmented generation with late chunking and binary-quantized embeddings. Uses a separate embedding model. Model-agnostic: the RAG index survives LLM swaps.

### Types

#### `rag_engine_t`

Opaque handle. Created with `rag_engine_create()`, destroyed with `rag_engine_free()`.

#### `rag_engine_params`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_threads` | `int32_t` | 0 | Thread count (0 = auto) |
| `chunk_size` | `int32_t` | 256 | Tokens per chunk |
| `chunk_overlap` | `int32_t` | 32 | Overlap tokens between chunks |
| `n_dims` | `int32_t` | 256 | Matryoshka embedding dim: 768/512/256/128 |
| `top_k` | `int32_t` | 32 | BQ search candidates before re-rank |
| `top_n` | `int32_t` | 5 | Final results after cosine re-rank |
| `late_chunking` | `bool` | true | Embed full doc then chunk (context-aware) |

#### `rag_result`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `const char *` | Matched chunk text |
| `doc_id` | `const char *` | Document identifier |
| `chunk_index` | `int32_t` | Chunk index within document |
| `score` | `float` | Cosine similarity (0.0--1.0) |

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

// Indexing (returns chunk count, -1 on error)
int32_t rag_engine_add_document(rag_engine_t * engine,
            const char * text, const char * doc_id);
int32_t rag_engine_remove_document(rag_engine_t * engine, const char * doc_id);
void    rag_engine_clear(rag_engine_t * engine);
int32_t rag_engine_document_count(const rag_engine_t * engine);
int32_t rag_engine_chunk_count(const rag_engine_t * engine);

// Retrieval (two-stage: BQ Hamming -> cosine re-rank)
rag_result * rag_engine_query(rag_engine_t * engine,
                 const char * query, int32_t * n_results);
void         rag_engine_free_results(rag_result * results, int32_t n);

// Build augmented prompt with retrieved context
char * rag_engine_build_prompt(rag_engine_t * engine,
           const char * query, const char * user_prompt);

// Info as JSON. Caller must free.
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

    rag_engine_free(rag);
}
```

### How It Works

1. **Late chunking** -- full document is embedded with bidirectional attention, then token embeddings are chunked. Preserves cross-chunk context.
2. **Matryoshka truncation** -- 768-dim embeddings truncated to `n_dims` without retraining. 3x compression at 256 dims.
3. **Binary quantization** -- float embeddings thresholded to 1-bit vectors. 32x compression. Hamming distance for fast candidate search.
4. **Two-stage retrieval** -- BQ Hamming finds `top_k` candidates, cosine similarity re-ranks to `top_n` final results.
5. **Sliding window** -- documents longer than model context are processed in overlapping windows with averaged overlap regions.

---

## ToolManager (`tool-manager.h`)

Model-agnostic tool calling. Parses tool calls from model output in JSON, XML, and function-call formats. Supports multiple concurrent tool calls per response.

### Types

#### `tool_manager_t`

Opaque handle. Created with `tool_manager_create()`, destroyed with `tool_manager_free()`.

#### `tool_param_type`

```c
typedef enum {
    TOOL_PARAM_STRING,
    TOOL_PARAM_NUMBER,
    TOOL_PARAM_BOOLEAN,
    TOOL_PARAM_ARRAY,
    TOOL_PARAM_OBJECT,
} tool_param_type;
```

#### `tool_param_def`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `const char *` | Parameter name |
| `description` | `const char *` | Parameter description |
| `type` | `tool_param_type` | Data type |
| `required` | `bool` | Whether parameter is required |

#### `tool_def`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `const char *` | Tool name |
| `description` | `const char *` | Tool description |
| `params` | `tool_param_def *` | Parameter definitions |
| `n_params` | `int32_t` | Number of parameters |

#### `tool_call_result`

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `const char *` | Name of the called tool |
| `arguments_json` | `const char *` | JSON string of parsed arguments |
| `is_valid` | `bool` | Whether parsing succeeded |

#### `tool_execute_callback`

```c
typedef const char * (*tool_execute_callback)(
    const char * tool_name, const char * args_json, void * user_data);
```

### Functions

```c
// Lifecycle
tool_manager_t * tool_manager_create(void);
void             tool_manager_free(tool_manager_t * tm);

// Registration
void tool_manager_register(tool_manager_t * tm, const tool_def * tool);
void tool_manager_clear(tool_manager_t * tm);

// Generate system prompt describing available tools. Caller must free.
char * tool_manager_get_prompt(const tool_manager_t * tm);

// Parse first tool call from model output.
tool_call_result tool_manager_parse_output(
    const tool_manager_t * tm, const char * model_output);

// Parse all tool calls. Caller must free with tool_manager_free_results.
tool_call_result * tool_manager_parse_output_all(
    const tool_manager_t * tm, const char * model_output,
    int32_t * n_calls);
void tool_manager_free_results(tool_call_result * results, int32_t n_calls);

// Execution
void   tool_manager_set_callback(tool_manager_t * tm,
           tool_execute_callback cb, void * user_data);
char * tool_manager_execute(tool_manager_t * tm, const tool_call_result * call);

void tool_manager_free_string(char * str);
```

### Supported Formats

**JSON (OpenAI-style)**
```json
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
```

**XML**
```xml
<tool_call>
  <name>get_weather</name>
  <arguments>{"city": "Tokyo"}</arguments>
</tool_call>
```

**Function-call**
```
get_weather(city="Tokyo")
```

### Usage Example

```c
#include "tool-manager.h"

tool_param_def weather_params[] = {
    { "city", "City name", TOOL_PARAM_STRING, true },
};

tool_def weather_tool = {
    .name = "get_weather", .description = "Get weather for a city",
    .params = weather_params, .n_params = 1,
};

tool_manager_t * tm = tool_manager_create();
tool_manager_register(tm, &weather_tool);

char * prompt = tool_manager_get_prompt(tm);
// ... inject prompt, run generation ...

tool_call_result result = tool_manager_parse_output(tm, model_output);
if (result.is_valid)
    printf("Tool: %s  Args: %s\n", result.tool_name, result.arguments_json);

tool_manager_free_string(prompt);
tool_manager_free(tm);
```

---

## Logging

Two logging interfaces are provided: the internal logging system (`tn-log.h`) used by engine code, and the public log callback in `ggml-engine.h` for application-level log capture.

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

void tn_log_set_callback(tn_log_callback cb, void * user_data);
void tn_log_set_level(enum tn_log_level max_level);
void tn_log_write(enum tn_log_level level, const char * tag, const char * fmt, ...);
```

Convenience macros (tag defaults to `__FILE__`):

```c
TN_LOG_ERR(fmt, ...)
TN_LOG_WRN(fmt, ...)
TN_LOG_INF(fmt, ...)
TN_LOG_DBG(fmt, ...)
```

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

// Pass NULL to restore default (Android logcat / stderr).
void tn_engine_set_log_callback(tn_engine_log_callback cb, void * user_data);
void tn_engine_set_log_level(tn_engine_log_level max_level);
```

### Usage Example

```c
#include "ggml-engine.h"

void my_logger(tn_engine_log_level level, const char * tag,
               const char * msg, void * user) {
    const char * lvl[] = { "ERR", "WRN", "INF", "DBG" };
    fprintf(stderr, "[%s] %s: %s\n", lvl[level], tag, msg);
}

int main() {
    tn_engine_set_log_callback(my_logger, NULL);
    tn_engine_set_log_level(TN_ENGINE_LOG_INFO);
    // ... engine usage ...
}
```
