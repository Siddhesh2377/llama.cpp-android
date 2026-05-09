# Tool-Neuron GGML Backend

CPU-only LLM/VLM inference engine for Android, built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Overview

A production fork of llama.cpp stripped to the CPU backend and optimized for ARM Android devices. All GPU backends (CUDA, Metal, Vulkan, OpenCL) have been removed. Three engine components are built on top for the [Tool-Neuron](https://github.com/Siddhesh2377/ToolNeuron) Android app.

```
Kotlin SDK (gguf_lib)
    |
JNI bridge
    |
Engine layer (engine/)
  - GGMLEngine    model load/unload, generation, KV cache, context tracking
  - ThreadEngine  big.LITTLE-aware thread mode (power_saving / balanced / performance)
  - VLM Engine    vision and audio understanding (20+ architectures)
  - RAG Engine    late chunking, binary quantized retrieval
  - Logging       callback-based, routes to Android logcat or custom handler
    |
llama.cpp core (src/ + common/)
    |
GGML CPU backend (ggml/)
  - NEON, i8mm, dotprod, fp16, bf16
  - KleidiAI ARM micro-kernels
```

## Directory Structure

```
src/             llama.cpp model loading, tokenization, inference, sampling
include/         public C/C++ headers (llama.h, llama-cpp.h)
ggml/            tensor library, CPU backend only, ARM optimized
common/          chat templates, JSON schema grammar, sampling, jinja
engine/          engine layer (ggml-engine, vlm, rag-engine, tn-log)
  vlm/           vision/audio encoders (CLIP, SigLIP, Whisper, 20+ architectures)
vendor/          nlohmann/json, stb_image, miniaudio
cmake/           build-info, license, compiler flags
docs/            API reference, architecture, build guide, benchmarks
```

## Supported Models

Any GGUF model works. All compute graphs from upstream llama.cpp are preserved.

- **Text**: LLaMA, Mistral, Phi, Qwen, Gemma, DeepSeek, Command-R, and 100+ architectures
- **Vision**: SmolVLM, LLaVA, Qwen2-VL, Qwen3-VL, InternVL, Pixtral, Gemma3-Vision, and 20+ VLM architectures
- **Audio**: Whisper, Conformer encoders
- **Quantization**: Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32, IQ variants

## Usage

This repo is consumed as a CMake subdirectory by an Android library module:

```cmake
set(LLAMA_DIR "/path/to/this/repo")
add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama)
target_link_libraries(my_jni_lib tn-engine llama common ggml)
```

All public engine headers are pure C (`extern "C"`) and safe for JNI binding.

## Build

See [docs/BUILD.md](docs/BUILD.md) for full details. Key CMake variables:

| Variable | Value | Purpose |
|----------|-------|---------|
| `GGML_CPU` | ON | CPU backend |
| `GGML_CPU_ARM_ARCH` | `armv8.6-a+i8mm+dotprod+fp16` | ARM feature flags |
| `GGML_CPU_KLEIDIAI` | ON | ARM KleidiAI micro-kernels |
| `GGML_LTO` | ON | Link-time optimization |
| `BUILD_SHARED_LIBS` | OFF | Static link into single .so |

## Thread Modes

The engine reads `/sys/devices/system/cpu/` at runtime to detect big.LITTLE core topology, then configures threads accordingly. Three modes are exposed as a 0–2 integer for a UI seekbar:

| Mode | Value | Behavior |
|------|-------|----------|
| Power Saving | 0 | 1 thread, efficiency cores, small batch — minimal battery drain |
| Balanced | 1 | 2 P-cores gen, all P-cores prompt — default |
| Performance | 2 | max 4 P-cores gen, all cores prompt, large batch |

Switch at runtime without reloading the model via `ggml_engine_set_thread_mode()`.

## Device & Memory Queries

Before loading a model, query the device to pick an appropriate size:

```c
ggml_engine_device_info dev = ggml_engine_get_device_info();
// dev.n_perf_cores, dev.n_efficiency_cores, dev.max_freq_khz

int64_t ram = ggml_engine_available_ram();
int64_t max_bytes = ggml_engine_max_model_size(ram, /*n_ctx=*/2048);
// max_bytes = budget after KV cache + 200 MB OS overhead
```

## Performance

Tested on Cortex-X3 (armv9, i8mm, bf16, NEON, dotprod):

| Model | Quant | Generation |
|-------|-------|------------|
| LFM2-350M | Q8_0 | 29-30 t/s |
| SmolVLM-500M | Q8_0 | 28 t/s text, 22 t/s with vision |
| Qwen3-0.6B | Q8_0 | 17-19 t/s |
| Gemma3-1B | Q4_K_M | 14 t/s |

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API.md) | C API for GGMLEngine, VLM, RAG, Logging |
| [Architecture](docs/ARCHITECTURE.md) | Stack diagram, directory map, data flows |
| [Build Guide](docs/BUILD.md) | CMake variables, NDK cross-compilation |
| [Performance](docs/PERFORMANCE.md) | Benchmarks, ARM optimizations, threading |
| [Models](docs/MODELS.md) | Supported architectures, quantization, sizing |

## License

MIT License -- see [LICENSE](LICENSE).

Based on [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.
