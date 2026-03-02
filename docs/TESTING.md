# Testing

## Test CLI

The test suite is built as `llama-test-cli` from `engine/llama-test-cli.cpp`. It validates all four engine components including VLM.

### Build

```bash
# Desktop
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_COMMON=ON
cmake --build build -j$(nproc)

# Android (push to device)
# After NDK cross-compilation:
adb push build/bin/llama-test-cli /data/local/tmp/
adb shell chmod +x /data/local/tmp/llama-test-cli
```

### Run

```bash
# Full test suite (requires a GGUF model)
./llama-test-cli -m /path/to/model.gguf

# With VLM (vision) tests
./llama-test-cli -m /path/to/model.gguf --mmproj /path/to/mmproj.gguf --image /path/to/test.jpg

# Quick tests only (no model needed)
./llama-test-cli --quick

# Skip model-dependent tests
./llama-test-cli --no-model

# On Android device
adb shell /data/local/tmp/llama-test-cli -m /data/local/tmp/model.gguf

# VLM tests on Android
adb shell /data/local/tmp/llama-test-cli \
  -m /data/local/tmp/smolvlm-500m.gguf \
  --mmproj /data/local/tmp/mmproj-smolvlm-500m.gguf \
  --image /data/local/tmp/test.jpg
```

### Test Coverage

**61 tests** across 14 test groups:

| Group | Tests | Requires | What It Validates |
|-------|-------|----------|-------------------|
| Engine Lifecycle | 3 | Nothing | Default params, create/destroy, initial state |
| Model Loading | 3 | Model | File loading, loaded state, context size |
| Model Info | 4 | Model | JSON output, n_embd, n_layer, n_vocab, metadata |
| Tokenization | 3 | Model | Tokenize, token count, detokenize roundtrip |
| Text Generation | 6 | Model | Generate status, output, response getter, perf metrics |
| Tool Manager | 7 | Nothing | Create, register, prompt gen, JSON parse, XML parse, extraction |
| Character Engine | 8 | Nothing | Create, personality, mood effects, biases, suppression, context |
| Character + Generation | 5 | Model | Persona generation, custom params, mood integration, perf |
| VLM Loading | 5 | Model + mmproj | Text model load, default params, mmproj load, is_loaded |
| VLM Info | 3 | Model + mmproj | Info JSON, supports_vision, default marker |
| VLM Image Encode | 2 | Model + mmproj + image | Image file load, encode returns positive tokens |
| VLM Generation | 3 | Model + mmproj + image | Generation status OK, output not empty, perf metrics |
| VLM Error Cases | 3 | Model | Null VLM, invalid path, null is_loaded |
| Tool Call Generation | 2 | Model | Live model tool call, multi-tool parse |
| Character Moods | 6 | Model | All 6 moods with live generation, temperature ordering |
| Think/No-Think | 2 | Model (Qwen3) | `/think` mode with `<think>` tags, `/no_think` mode |

### Output

```
=== GGMLEngine Test Suite ===

--- Engine Lifecycle Tests ---
[PASS] Default params are valid
[PASS] Engine created
[PASS] Engine reports no model loaded

--- Model Loading Tests ---
[PASS] Model loaded from /data/local/tmp/qwen3-0.6b-q8_0.gguf
[PASS] Engine reports model loaded
[PASS] Context size > 0 (2048)

... (61 tests total)

=== Results: 61/61 passed ===
```

Exit code `0` = all passed, `1` = failures.

### Quick vs Full

| Mode | Flag | Tests | Time | Use Case |
|------|------|-------|------|----------|
| Full | `--all` (default) | 61 | ~10-20s | CI, release validation |
| Full + VLM | `-m ... --mmproj ... --image ...` | 61 | ~15-30s | Full with vision tests |
| Quick | `--quick` | 18 | <1s | Compile check, no model needed |
| No model | `--no-model` | 18 | <1s | Same as quick |

---

## Testing on Android Device

### Prerequisites

- Device connected via ADB
- GGUF model pushed to device

```bash
# Push a model
adb push qwen3-0.6b-q8_0.gguf /data/local/tmp/

# Push test binary (after NDK cross-compile)
adb push build/bin/llama-test-cli /data/local/tmp/
adb shell chmod +x /data/local/tmp/llama-test-cli
```

### Run on device

```bash
adb shell /data/local/tmp/llama-test-cli -m /data/local/tmp/qwen3-0.6b-q8_0.gguf
```

### Recommended test models

| Model | Size | Speed (Cortex-X3) | Purpose |
|-------|------|--------------------|---------|
| LFM2-350M | ~350 MB | 29-30 t/s | Quick smoke tests |
| SmolVLM-500M + mmproj | ~500 MB + mmproj | 28 t/s | VLM testing |
| Qwen3-0.6B Q8_0 | ~630 MB | 17-19 t/s | Primary test model (tool call, think) |
| Gemma3-1B | ~1 GB | 14 t/s | Larger model validation |

---

## What Each Test Group Validates

### Engine Lifecycle
- `ggml_engine_default_params()` returns sane values
- `ggml_engine_create()` / `ggml_engine_free()` don't crash
- Newly created engine reports `is_loaded = false`

### Model Loading
- `ggml_engine_load_model()` returns `GGML_ENGINE_OK`
- `ggml_engine_is_loaded()` returns `true` after load
- `ggml_engine_context_size()` returns value > 0

### Model Info
- `ggml_engine_model_info_json()` returns valid JSON
- JSON contains `n_embd`, `n_layer`, `n_vocab` fields
- JSON contains model metadata

### Tokenization
- `ggml_engine_tokenize()` produces tokens from text
- Token count is reasonable (> 0)
- `ggml_engine_detokenize()` roundtrips back to text

### Text Generation
- `ggml_engine_generate()` returns `GGML_ENGINE_OK`
- Callback receives non-empty token strings
- `ggml_engine_get_response()` returns accumulated text
- Performance metrics report non-zero values
- Prompt tokens and generated tokens are counted

### Tool Manager
- Tools can be registered and cleared
- `tool_manager_get_prompt()` generates tool descriptions
- JSON format tool calls are parsed correctly
- XML format tool calls are parsed correctly
- Tool name and arguments are extracted accurately

### Character Engine
- Personality can be set with all parameters
- Mood changes affect temperature (happy > neutral)
- Mood changes affect top_p (focused < happy)
- Logit biases are tracked correctly
- Token suppressions are tracked correctly
- `character_engine_get_context()` produces non-empty string

### Character + Generation (Integration)
- Persona context is prepended to prompts
- Custom personality parameters affect generation
- Mood modifiers integrate with sampling
- Generation produces output with character settings
- Performance metrics work with character engine active

### VLM Loading
- Text model loads successfully before mmproj
- Default VLM params have expected values
- mmproj GGUF loads without error
- `ggml_engine_vlm_is_loaded()` returns true after load
- VLM load fails gracefully with invalid path

### VLM Info
- `ggml_engine_vlm_info_json()` returns valid JSON with capabilities
- JSON contains `supports_vision` field (true for vision models)
- `ggml_engine_vlm_default_marker()` returns `<__media__>`

### VLM Image Encode
- Image file loads into memory
- `ggml_engine_vlm_encode_image()` returns positive token count

### VLM Generation
- `ggml_engine_vlm_generate()` returns `GGML_ENGINE_OK`
- Generated output is non-empty and describes image content
- Performance metrics report vision encode + generation timing

### VLM Error Cases
- Null VLM handle returns error status
- Invalid mmproj path returns null
- `ggml_engine_vlm_is_loaded(NULL)` returns false

### Tool Call Generation
- Live model generates valid tool call JSON from tool-injected prompt
- `tool_manager_parse_output_all()` parses multiple tool calls from single response

### Character Moods
- All 6 moods (neutral, happy, sad, angry, curious, focused) produce output with live generation
- Temperature ordering: happy > neutral > focused
- Each mood produces distinct sampling behavior

### Think/No-Think (Qwen3)
- `/think` mode generates `<think>` tags with reasoning before answer
- `/no_think` mode generates direct answer without reasoning tags
