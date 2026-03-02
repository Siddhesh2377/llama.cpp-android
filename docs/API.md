# Engine API Reference

Complete C API reference for the four engine components. All headers are in `engine/`.

---

## GGMLEngine (`ggml-engine.h`)

Core LLM inference engine. Handles model loading, text generation, context management, and tokenization.

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

#### `ggml_engine_token_callback`

```c
typedef bool (*ggml_engine_token_callback)(const char * token_text, void * user_data);
```

Called for each generated token. Return `false` to stop generation.

### Functions

#### Defaults

```c
ggml_engine_params   ggml_engine_default_params(void);
ggml_engine_sampling ggml_engine_default_sampling(void);
```

#### Lifecycle

```c
// Create engine with configuration
ggml_engine_t * ggml_engine_create(ggml_engine_params params);

// Destroy engine and free all resources
void ggml_engine_free(ggml_engine_t * engine);
```

#### Model Loading

```c
// Load model from file path
ggml_engine_status ggml_engine_load_model(ggml_engine_t * engine, const char * path);

// Load model from file descriptor (Android SAF)
ggml_engine_status ggml_engine_load_model_from_fd(ggml_engine_t * engine, int fd);

// Unload model and free model memory
void ggml_engine_unload_model(ggml_engine_t * engine);

// Check if a model is currently loaded
bool ggml_engine_is_loaded(const ggml_engine_t * engine);
```

#### Model Information

```c
// Get model metadata as JSON string
// Includes: n_embd, n_layer, n_vocab, n_ctx, model name, metadata
// Caller must free with ggml_engine_free_string()
char * ggml_engine_model_info_json(const ggml_engine_t * engine);

// Free a string allocated by the engine
void ggml_engine_free_string(char * str);
```

#### Text Generation

```c
// Generate text from prompt with streaming callback
ggml_engine_status ggml_engine_generate(
    ggml_engine_t            * engine,
    const char               * prompt,
    ggml_engine_sampling       sampling,
    ggml_engine_token_callback callback,
    void                     * user_data
);

// Cancel ongoing generation (thread-safe, call from any thread)
void ggml_engine_cancel(ggml_engine_t * engine);

// Get full text from last generation (caller must free)
char * ggml_engine_get_response(const ggml_engine_t * engine);
```

#### Context Management

```c
// Clear KV cache and reset conversation
void ggml_engine_clear_context(ggml_engine_t * engine);

// Get number of tokens currently in context
int32_t ggml_engine_context_used(const ggml_engine_t * engine);

// Get total context capacity
int32_t ggml_engine_context_size(const ggml_engine_t * engine);
```

#### Tokenization

```c
// Tokenize text into token IDs
// Returns number of tokens, or -1 on error
int32_t ggml_engine_tokenize(
    const ggml_engine_t * engine,
    const char          * text,
    int32_t             * tokens,
    int32_t               max_tokens
);

// Convert token IDs back to text (caller must free)
char * ggml_engine_detokenize(
    const ggml_engine_t * engine,
    const int32_t       * tokens,
    int32_t               n_tokens
);
```

#### Performance

```c
// Get timing metrics from last generation
ggml_engine_perf ggml_engine_get_perf(const ggml_engine_t * engine);
```

### Usage Example

```c
#include "ggml-engine.h"

bool on_token(const char * text, void * user) {
    printf("%s", text);
    return true; // continue generating
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

    ggml_engine_generate(engine, "Hello, world!", sampling, on_token, NULL);

    ggml_engine_perf perf = ggml_engine_get_perf(engine);
    printf("\n%.1f tokens/sec\n", perf.generation_tokens_per_sec);

    ggml_engine_free(engine);
}
```

---

## VLM Engine (`ggml-engine.h` — VLM section)

Vision Language Model support. Loads a vision projector (mmproj GGUF) and generates text from images + text prompts. Supports 20+ VLM architectures including LLaVA, Qwen2-VL, Qwen3-VL, InternVL, Pixtral, Gemma3-Vision, SmolVLM, and more. CPU-only.

### Types

#### `ggml_engine_vlm_t`

Opaque VLM handle. Created with `ggml_engine_vlm_load()`, destroyed with `ggml_engine_vlm_free()`.

#### `ggml_engine_vlm_params`

VLM configuration. Get defaults with `ggml_engine_vlm_default_params()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_threads` | `int32_t` | 0 | Threads for vision encode (0 = same as engine) |
| `image_min_tokens` | `int32_t` | -1 | Min image tokens (-1 = model default) |
| `image_max_tokens` | `int32_t` | -1 | Max image tokens (-1 = model default) |

#### `ggml_engine_image`

Image data — either raw file bytes or decoded RGB pixels.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `const unsigned char *` | File bytes (JPEG/PNG/etc.) or raw RGB pixels |
| `size` | `size_t` | Byte count |
| `width` | `uint32_t` | Pixel width (0 = file mode, auto-detect format) |
| `height` | `uint32_t` | Pixel height (0 = file mode) |

### Functions

#### Defaults

```c
ggml_engine_vlm_params ggml_engine_vlm_default_params(void);
```

#### Loading

```c
// Load vision projector from file path. Call after loading text model.
ggml_engine_vlm_t * ggml_engine_vlm_load(
    ggml_engine_t * engine, const char * mmproj_path,
    ggml_engine_vlm_params params);

// Load vision projector from file descriptor (Android SAF)
ggml_engine_vlm_t * ggml_engine_vlm_load_from_fd(
    ggml_engine_t * engine, int fd,
    ggml_engine_vlm_params params);

// Free VLM resources
void ggml_engine_vlm_free(ggml_engine_vlm_t * vlm);

// Check if VLM projector is loaded
bool ggml_engine_vlm_is_loaded(const ggml_engine_vlm_t * vlm);
```

#### Generation

```c
// Generate text from prompt + images
// Use "<__media__>" markers in prompt to position images
ggml_engine_status ggml_engine_vlm_generate(
    ggml_engine_t * engine, ggml_engine_vlm_t * vlm,
    const char * prompt,
    const ggml_engine_image * images, int32_t n_images,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data);
```

#### Image Encoding

```c
// Encode image only — returns token count, -1 on error
// Useful for estimating context usage before generation
int32_t ggml_engine_vlm_encode_image(
    ggml_engine_vlm_t * vlm, const ggml_engine_image * image);
```

#### Info

```c
// Get VLM info as JSON (caller must free with ggml_engine_free_string)
// Fields: supports_vision, supports_audio, uses_mrope, uses_non_causal,
//         audio_bitrate, default_marker
char * ggml_engine_vlm_info_json(const ggml_engine_vlm_t * vlm);

// Get default image marker string ("<__media__>")
const char * ggml_engine_vlm_default_marker(void);

// Capability queries
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
    // 1. Create engine and load text model
    ggml_engine_params params = ggml_engine_default_params();
    params.n_ctx = 2048;
    ggml_engine_t * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, "smolvlm-500m.gguf");

    // 2. Load vision projector
    ggml_engine_vlm_params vlm_params = ggml_engine_vlm_default_params();
    ggml_engine_vlm_t * vlm = ggml_engine_vlm_load(engine, "mmproj.gguf", vlm_params);

    // 3. Load image from file
    FILE * f = fopen("photo.jpg", "rb");
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char * buf = malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);

    ggml_engine_image img = { .data = buf, .size = sz, .width = 0, .height = 0 };

    // 4. Generate — use <__media__> marker for image position
    ggml_engine_sampling sampling = ggml_engine_default_sampling();
    sampling.n_predict = 256;
    const char * prompt = "<__media__>\nDescribe this image in detail.";

    ggml_engine_vlm_generate(engine, vlm, prompt, &img, 1, sampling, on_token, NULL);

    // 5. Cleanup
    free(buf);
    ggml_engine_vlm_free(vlm);
    ggml_engine_free(engine);
}
```

### Supported VLM Architectures

| Architecture | Models |
|-------------|--------|
| LLaVA | LLaVA-1.5, LLaVA-1.6, BakLLaVA |
| SigLIP | Gemma3-Vision |
| Qwen2-VL | Qwen2-VL (M-RoPE support) |
| Qwen3-VL | Qwen3-VL |
| Pixtral | Pixtral, Mistral-Vision |
| MiniCPM-V | MiniCPM-V, MiniCPM-V 2.6 |
| InternVL | InternVL2, InternVL2.5 |
| CogVLM | CogVLM, CogVLM2 |
| GLM4V | GLM-4V |
| Llama4 | Llama-4-Scout, Llama-4-Maverick |
| MobileNetV5 | Gemma3n-Vision |
| Kimi-VL | Kimi-VL |
| Kimi-K2.5 | Kimi-K2.5-VL |
| SmolVLM | SmolVLM-500M, SmolVLM-2.2B |
| PaddleOCR | PaddleOCR |
| Nemotron-V2 | Nemotron-V2-VL |
| YouTu-VL | YouTu-VL |
| Whisper | Whisper audio encoder |
| Conformer | Conformer audio encoder |

---

## ToolManager (`tool-manager.h`)

Model-agnostic tool calling system. Parses tool calls from model output in JSON, XML, and function-call formats. Supports multiple concurrent tool calls per response.

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
| `params` | `tool_param_def *` | Parameter definitions array |
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
    const char * tool_name,
    const char * args_json,
    void       * user_data
);
```

Execute a tool call and return result string.

### Functions

#### Lifecycle

```c
tool_manager_t * tool_manager_create(void);
void             tool_manager_free(tool_manager_t * tm);
```

#### Registration

```c
// Register a tool definition
void tool_manager_register(tool_manager_t * tm, const tool_def * tool);

// Clear all registered tools
void tool_manager_clear(tool_manager_t * tm);
```

#### Prompt Generation

```c
// Generate system prompt describing available tools
// Caller must free with tool_manager_free_string()
char * tool_manager_get_prompt(const tool_manager_t * tm);
```

#### Parsing

```c
// Parse first tool call from model output
tool_call_result tool_manager_parse_output(
    const tool_manager_t * tm,
    const char           * model_output
);

// Parse all tool calls from model output
// Returns heap-allocated array, caller must free with tool_manager_free_results()
tool_call_result * tool_manager_parse_output_all(
    const tool_manager_t * tm,
    const char           * model_output,
    int32_t              * n_calls
);

// Free results array
void tool_manager_free_results(tool_call_result * results, int32_t n_calls);
```

#### Execution

```c
// Set callback for executing tool calls
void tool_manager_set_callback(
    tool_manager_t       * tm,
    tool_execute_callback  cb,
    void                 * user_data
);

// Execute a parsed tool call (caller must free result)
char * tool_manager_execute(tool_manager_t * tm, const tool_call_result * call);
```

#### Memory

```c
void tool_manager_free_string(char * str);
```

### Supported Formats

The parser recognizes tool calls in these formats:

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
    .name        = "get_weather",
    .description = "Get current weather for a city",
    .params      = weather_params,
    .n_params    = 1,
};

tool_manager_t * tm = tool_manager_create();
tool_manager_register(tm, &weather_tool);

// Inject tool description into prompt
char * prompt = tool_manager_get_prompt(tm);

// After generation, parse the model output
tool_call_result result = tool_manager_parse_output(tm, model_output);
if (result.is_valid) {
    printf("Tool: %s\nArgs: %s\n", result.tool_name, result.arguments_json);
}

tool_manager_free_string(prompt);
tool_manager_free(tm);
```

---

## CharacterEngine (`character-engine.h`)

Personality and behavior control via sampling parameter modulation and logit-level manipulation. Works with any model without modifying system prompts or chat templates.

### Types

#### `character_engine_t`

Opaque handle. Created with `character_engine_create()`, destroyed with `character_engine_free()`.

#### `char_mood`

```c
typedef enum {
    CHAR_MOOD_NEUTRAL    = 0,   // no modifiers
    CHAR_MOOD_HAPPY      = 1,   // +temperature, +top_p, -repetition
    CHAR_MOOD_SAD        = 2,   // -temperature, -top_p, +repetition
    CHAR_MOOD_EXCITED    = 3,   // ++temperature, ++variation
    CHAR_MOOD_CALM       = 4,   // -temperature, slight -top_p
    CHAR_MOOD_ANGRY      = 5,   // +temperature, +variation
    CHAR_MOOD_CURIOUS    = 6,   // balanced exploration
    CHAR_MOOD_CREATIVE   = 7,   // high temperature, high top_p
    CHAR_MOOD_FOCUSED    = 8,   // low temperature, low top_p (deterministic)
    CHAR_MOOD_CUSTOM     = 9,   // user-defined modifiers
} char_mood;
```

#### `char_personality`

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `name` | `const char *` | — | Character name |
| `persona` | `const char *` | — | Persona description |
| `temperature` | `float` | 0.1–2.0 | Base temperature |
| `top_p` | `float` | 0.0–1.0 | Base nucleus sampling |
| `repetition_penalty` | `float` | 1.0–2.0 | Base repetition penalty |
| `creativity` | `float` | 0.0–1.0 | Conservative to creative |
| `verbosity` | `float` | 0.0–1.0 | Terse to verbose |
| `formality` | `float` | 0.0–1.0 | Casual to formal |

#### `char_logit_bias`

| Field | Type | Description |
|-------|------|-------------|
| `token_id` | `int32_t` | Token ID to bias |
| `bias` | `float` | Positive = boost, negative = suppress |

#### `char_effective_params`

Combined result of personality + mood + biases.

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | `float` | Final temperature |
| `top_p` | `float` | Final top_p |
| `min_p` | `float` | min_p value |
| `repetition_penalty` | `float` | Final repetition penalty |
| `top_k` | `int32_t` | top_k value |
| `n_logit_biases` | `int32_t` | Number of active biases |
| `logit_biases` | `const char_logit_bias *` | Logit bias array |
| `n_suppressed` | `int32_t` | Number of suppressed tokens |
| `suppressed_tokens` | `const int32_t *` | Suppressed token IDs |
| `uncensored` | `bool` | Whether uncensored mode is active |

### Functions

#### Lifecycle

```c
character_engine_t * character_engine_create(void);
void                 character_engine_free(character_engine_t * ce);
```

#### Personality

```c
// Set character personality and base parameters
void character_engine_set_personality(
    character_engine_t     * ce,
    const char_personality * personality
);
```

#### Mood

```c
// Set predefined mood (modifies temperature, top_p, repetition_penalty)
void character_engine_set_mood(character_engine_t * ce, char_mood mood);

// Set custom mood modifiers (added to base personality values)
void character_engine_set_custom_mood(
    character_engine_t * ce,
    float temperature_mod,
    float top_p_mod,
    float rep_penalty_mod
);
```

#### Logit Biases

```c
// Add bias for a specific token (positive = boost, negative = suppress)
void character_engine_add_logit_bias(
    character_engine_t * ce,
    int32_t token_id,
    float   bias
);

// Clear all logit biases
void character_engine_clear_logit_biases(character_engine_t * ce);
```

#### Token Suppression

```c
// Suppress a token entirely (set logit to -inf)
void character_engine_suppress_token(character_engine_t * ce, int32_t token_id);

// Clear all suppressions
void character_engine_clear_suppressions(character_engine_t * ce);
```

#### Uncensored Mode

```c
// Enable/disable uncensored mode
// Suppresses refusal tokens at the logit level
void character_engine_set_uncensored(character_engine_t * ce, bool enabled);

// Check if uncensored mode is active
bool character_engine_get_uncensored(const character_engine_t * ce);

// Supply refusal token IDs (scanned from vocab after model load)
void character_engine_set_refusal_tokens(
    character_engine_t * ce,
    const int32_t      * token_ids,
    int32_t              n_tokens
);
```

#### Parameter Retrieval

```c
// Get combined effective parameters
char_effective_params character_engine_get_params(const character_engine_t * ce);

// Get persona context string for prompt injection (caller must free)
char * character_engine_get_context(const character_engine_t * ce);

void character_engine_free_string(char * str);
```

### Usage Example

```c
#include "character-engine.h"

character_engine_t * ce = character_engine_create();

char_personality persona = {
    .name        = "Atlas",
    .persona     = "A witty AI assistant with dry humor",
    .temperature = 0.9,
    .top_p       = 0.95,
    .repetition_penalty = 1.1,
    .creativity  = 0.8,
    .verbosity   = 0.5,
    .formality   = 0.3,
};

character_engine_set_personality(ce, &persona);
character_engine_set_mood(ce, CHAR_MOOD_CREATIVE);

// Get effective params to pass to engine sampling
char_effective_params ep = character_engine_get_params(ce);

// Get context string to prepend to user prompt
char * ctx = character_engine_get_context(ce);

character_engine_free_string(ctx);
character_engine_free(ce);
```
