# Tool-Neuron GGML Backend

Android-optimized CPU-only LLM inference engine built on llama.cpp/ggml.

## Overview

Tool-Neuron is an offline, privacy-first LLM inference framework for Android. It strips llama.cpp down to the essential CPU backend and builds specialized engines on top:

- **GGMLEngine** - Simplified model loading, info, and text generation API
- **ToolManager** - Model-agnostic tool calling system
- **Character Engine** - Personality/mood/behavior control via logit manipulation
- **Smart KV Cache** - Memory-pressure-aware cache management for mobile

## Architecture

```
┌─────────────────────────────────────────┐
│           Kotlin/JNI SDK                │
├─────────────────────────────────────────┤
│  GGMLEngine │ ToolManager │ CharEngine  │
├─────────────────────────────────────────┤
│         llama.cpp core (src/)           │
├─────────────────────────────────────────┤
│     ggml (CPU backend, ARM optimized)   │
│     NEON / i8mm / bf16 / dot-product    │
└─────────────────────────────────────────┘
```

## Supported Models

All GGUF format models are supported. The engine preserves all compute graphs from llama.cpp including:
- LLaMA, Mistral, Phi, Qwen, Gemma, DeepSeek, and 100+ architectures
- All quantization formats (Q4_0, Q4_K_M, Q5_K_M, Q8_0, etc.)

## Building for Android

### Prerequisites
- Android NDK r27+
- CMake 3.14+

### Cross-compile for ARM64

```bash
export NDK=/path/to/android-ndk
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-android -j$(nproc)
```

### Run on device

```bash
adb push build-android/bin/llama-test-cli /data/local/tmp/
adb push model.gguf /data/local/tmp/
adb shell /data/local/tmp/llama-test-cli -m /data/local/tmp/model.gguf
```

## Project Structure

```
engine/          - GGMLEngine, ToolManager, CharacterEngine
ggml/            - Core tensor library (CPU backend only)
src/             - llama.cpp model loading and inference
common/          - Shared utilities
include/         - Public C/C++ headers
tools/           - CLI tools (quantize, completion)
examples/        - Simple usage examples
grammars/        - GBNF grammar files
```

## License

MIT License - See [LICENSE](LICENSE) for details.

Based on [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov.
