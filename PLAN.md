# Tool-Neuron Backend Build Plan

## Device: ARM64 v9 (Cortex-X3, i8mm, bf16, NEON, dot-product)

---

## Phase 1: Strip & Clean (Step 1) - DONE
- [x] Remove non-CPU backends: vulkan, opencl, cuda, metal, sycl, cann, rpc, blas, hip, musa, hexagon, virtgpu, webgpu, zdnn, zendnn
- [x] Remove non-CPU backend headers from ggml/include
- [x] Remove non-essential examples (keep: llama.android, simple, simple-chat)
- [x] Remove non-essential tools (keep: cli, quantize, completion, server)
- [x] Remove Python conversion scripts, Swift code, iOS code
- [x] Remove CI, docs, pocs, benches, media folders
- [x] Clean CMakeLists.txt to only build CPU backend + Android targets
- [x] Update README.md for Tool-Neuron project

## Phase 2: GGMLEngine (Step 2) - DONE
- [x] Create `engine/ggml-engine.h` - C API
- [x] Create `engine/ggml-engine.cpp` - Implementation
- [x] Model load/unload with path + Android SAF FileDescriptor support
- [x] Model info as JSON (arch, params, quant, context length, metadata)
- [x] Text generation with streaming callback
- [x] Auto thread detection for Android (75% of available cores)
- [x] Flash attention support, mmap, batch processing

## Phase 3: ToolManager (Step 3) - DONE
- [x] Create `engine/tool-manager.h` - Tool registration API
- [x] Create `engine/tool-manager.cpp` - Implementation
- [x] JSON-based tool schema registration
- [x] Output parsing: JSON, XML (<tool_call>), function-call formats
- [x] Model-agnostic tool calling (works without specific chat templates)
- [x] Tool execution callback system

## Phase 4: Smart KV Cache (Step 4) - DONE
- [x] Create `engine/kv-cache-manager.h` and `.cpp`
- [x] Sliding window with configurable keep_first_n
- [x] Session save/restore (binary format with TNSS magic)
- [x] Memory-pressure-aware cache sizing (configurable threshold)
- [x] Automatic context shift when approaching limit

## Phase 5: Character Engine (Step 5) - DONE
- [x] Create `engine/character-engine.h` - Character API
- [x] Create `engine/character-engine.cpp` - Implementation
- [x] 9 mood presets (happy, sad, excited, calm, angry, curious, creative, focused, custom)
- [x] Personality traits: creativity, verbosity, formality
- [x] Logit bias injection per token
- [x] Token suppression
- [x] Context generation for prompt augmentation
- [x] Public C API for JNI access

## Phase 6: LLAMA-Test-CLI (Step 6) - DONE
- [x] Android NDK cross-compilation (NDK r27d, arm64-v8a, API 28)
- [x] CLI with all engine features: lifecycle, model loading, info, tokenization, generation, character
- [x] 42 tests all passing on device
- [x] Performance: 12.8-16.0 t/s generation on Qwen3-0.6B-Q8_0

## Phase 7: JNI/Kotlin SDK (Final) - DONE
- [x] JNI bridge (`gguf_lib.cpp`) covering GGMLEngine, ToolManager, CharacterEngine
- [x] Kotlin SDK: `GGMLEngine.kt` with coroutine Flow streaming
- [x] Kotlin SDK: `ToolManager.kt` with tool registration and parsing
- [x] Kotlin SDK: `CharacterEngine.kt` with personality/mood control
- [x] Android SAF URI support for model loading
- [x] Unit tests for data classes and utilities
- [x] CMake integration pulling llama.cpp as static library
