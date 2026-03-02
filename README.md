# Tool-Neuron GGML Backend

CPU-only LLM inference engine for Android, built on llama.cpp. Runs any GGUF model on-device with zero network dependencies.

## What This Is

A production fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) stripped to the CPU backend and optimized for ARM Android devices. All non-CPU backends (CUDA, Metal, Vulkan, OpenCL, etc.) are removed. Four custom engine layers are built on top for the [Tool-Neuron](https://github.com/Siddhesh2377/ToolNeuron) Android app.

```
Kotlin SDK (gguf_lib)
    |
JNI bridge (gguf_lib.cpp)
    |
Engine layer (engine/)
  - GGMLEngine    model load/unload, streaming generation, KV cache management
  - VLM Engine    vision/audio understanding — 20+ model architectures
  - ToolManager   model-agnostic tool calling (JSON + XML + function-call formats)
  - CharacterEngine   personality, mood, uncensored mode via logit manipulation
    |
llama.cpp core (src/ + common/)
    |
ggml CPU backend (ggml/)
  - NEON, i8mm, dotprod, fp16, bf16
  - KleidiAI ARM micro-kernels
```

## Directory Structure

```
src/             llama.cpp model loading, tokenization, inference, sampling
include/         public C/C++ headers (llama.h, llama-cpp.h)
ggml/            GGML tensor library — CPU backend only, ARM optimized
common/          shared utilities (chat templates, JSON schema, sampling, ngram cache)
engine/          custom engine layer (ggml-engine, vlm, tool-manager, character-engine)
  vlm/           vision/audio encoder — CLIP, SigLIP, 20+ VLM architectures
vendor/          third-party (nlohmann/json, cpp-httplib, stb, miniaudio, sheredom)
cmake/           CMake modules (build-info, license, common)
```

## Supported Models

Any GGUF format model works. The fork preserves all compute graphs from upstream llama.cpp:

- **Text**: LLaMA, Mistral, Phi, Qwen, Gemma, DeepSeek, Command-R, and 100+ architectures
- **Vision**: SmolVLM, LLaVA, Qwen2-VL, Qwen3-VL, InternVL, Pixtral, Gemma3-Vision, and 20+ VLM architectures
- All quantization formats: Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32, IQ variants

## How It's Used

This repo is consumed as a CMake subdirectory by the `gguf_lib` Android library module:

```cmake
# in gguf_lib/src/main/cpp/CMakeLists.txt
set(LLAMA_DIR "/path/to/this/repo")
add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama)
target_link_libraries(my_jni_lib llama common)
```

The Kotlin SDK and JNI bridge live in a separate repo:
`AiSystems/gguf_lib/` — see its [CLAUDE.md](../AiSystems/gguf_lib/CLAUDE.md) for the full API reference.

## Build Configuration

When consumed via the Android NDK (through gguf_lib), these CMake variables are set:

| Variable | Value | Purpose |
|----------|-------|---------|
| `GGML_CPU` | ON | CPU backend |
| `GGML_CPU_ARM_ARCH` | `armv8.6-a+i8mm+dotprod+fp16` | Enables KleidiAI fast-path micro-kernels |
| `GGML_CPU_KLEIDIAI` | ON | ARM KleidiAI optimized kernels |
| `GGML_LTO` | ON | Link-time optimization |
| `GGML_OPENMP` | OFF | Not available on Android NDK |
| `BUILD_SHARED_LIBS` | OFF | Static link into single .so |
| `LLAMA_BUILD_COMMON` | ON | Common utilities needed by engine |
| `LLAMA_OPENSSL` | OFF | No HTTPS on Android (load from file/fd) |

### Compiler flags (set by gguf_lib CMakeLists.txt)

```
-ffp-contract=fast          FMA instruction fusion
-fno-math-errno             skip errno after math ops
-fno-signed-zeros           aggressive FP optimizations
-fno-trapping-math          ARM doesn't trap
-fvisibility=hidden         reduce .so size, eliminate PLT overhead
-fomit-frame-pointer        free x29 register
-ffunction-sections         enable --gc-sections dead code stripping
-fdata-sections             enable --gc-sections dead data stripping
```

### Linker flags

```
--gc-sections               strip unreferenced code/data
--icf=safe                  merge identical code sections
-z,max-page-size=16384      Android 15+ 16KB page support
```

## Performance

Tested on Cortex-X3 (armv9, i8mm, bf16, NEON, dotprod):

- LFM2-350M: ~29-30 t/s generation
- SmolVLM-500M: ~28 t/s text, ~22 t/s with vision
- Qwen3-0.6B Q8_0: ~17-19 t/s generation
- Gemma3-1B: ~14 t/s generation
- CPU affinity pinning to performance cores
- Thread split: decode (memory-bound) uses min(4, P-cores), batch (compute-bound) uses all P-cores
- KV cache prefix reuse across multi-turn conversations
- Automatic context shifting when context window fills
- Disk-backed system prompt cache (FNV-1a hashed filenames)
- Ngram self-speculative decoding (1.3-2x for structured output)
- Zero-copy JNI token delivery via pre-allocated byte buffers
- Stripped .so: ~4.1MB

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API.md) | Complete C API for GGMLEngine, VLM, ToolManager, CharacterEngine |
| [Build Guide](docs/BUILD.md) | CMake variables, compiler flags, NDK cross-compilation, CI/CD |
| [Architecture](docs/ARCHITECTURE.md) | Stack diagram, directory map, data flows, threading model |
| [Performance](docs/PERFORMANCE.md) | Benchmarks, ARM optimizations, KV cache, speculative decoding |
| [Supported Models](docs/MODELS.md) | Architectures, quantization formats, mobile sizing guide |
| [Testing](docs/TESTING.md) | Test CLI usage, coverage, running on device |

## License

MIT License — see [LICENSE](LICENSE).

Based on [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.
