# Build Guide

## Overview

This repo is a C/C++ library consumed via CMake. It does **not** build standalone — it's compiled as part of an Android NDK build through the `gguf_lib` module.

```
gguf_lib (Android library module)
  └── CMakeLists.txt
        └── add_subdirectory(llama.cpp)
              ├── ggml/       → libggml.a
              ├── src/        → libllama.a
              ├── common/     → libcommon.a
              └── engine/     → libtn-engine.a
                                ↓
                    All linked into libgguf_lib.so
```

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Android NDK | r27d (`27.3.13750724`) | Tested version |
| CMake | 3.31+ | Ships with Android Studio |
| C++ Standard | C++17 | Set by engine CMakeLists.txt |

---

## CMake Variables

### Required

| Variable | Value | Description |
|----------|-------|-------------|
| `CMAKE_TOOLCHAIN_FILE` | `${NDK}/build/cmake/android.toolchain.cmake` | NDK cross-compilation |
| `ANDROID_ABI` | `arm64-v8a` | Target ABI |
| `ANDROID_PLATFORM` | `android-28` | Minimum API level |

### Recommended

| Variable | Value | Description |
|----------|-------|-------------|
| `GGML_CPU` | `ON` | CPU backend (only backend available) |
| `GGML_CPU_ARM_ARCH` | `armv8.6-a+i8mm+dotprod+fp16` | ARM architecture features |
| `GGML_CPU_KLEIDIAI` | `ON` | KleidiAI ARM micro-kernels |
| `GGML_LTO` | `ON` | Link-time optimization |
| `GGML_OPENMP` | `OFF` | Not available on Android NDK |
| `BUILD_SHARED_LIBS` | `OFF` | Static libraries, linked into single .so |
| `LLAMA_BUILD_COMMON` | `ON` | Common utils needed by engine |
| `LLAMA_OPENSSL` | `OFF` | No HTTPS (models loaded from file/fd) |

---

## Standalone Build (Desktop Testing)

For running `llama-test-cli` on a desktop Linux machine:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_COMMON=ON

cmake --build build -j$(nproc)

# Run tests
./build/bin/llama-test-cli -m /path/to/model.gguf
```

---

## Android NDK Cross-Compilation

### As a CMake subdirectory (production path)

```cmake
# In your module's CMakeLists.txt
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../dev/include/llama.cpp")

set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8.6-a+i8mm+dotprod+fp16" CACHE STRING "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LTO ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama)

target_link_libraries(your_jni_lib
    tn-engine
    llama
    common
)
```

### Direct NDK build (CI / standalone .a files)

```bash
NDK_PATH="${ANDROID_HOME}/ndk/27.3.13750724"

cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE="${NDK_PATH}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_OPENSSL=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_CPU=ON \
  -DGGML_CPU_ARM_ARCH="armv8.6-a+i8mm+dotprod+fp16" \
  -DGGML_CPU_KLEIDIAI=ON

cmake --build build -j$(nproc)
```

Output: static libraries in `build/` (`libggml.a`, `libllama.a`, `libcommon.a`, `libtn-engine.a`).

---

## Compiler Flags

These are set by the consuming `gguf_lib/CMakeLists.txt` for maximum performance:

| Flag | Purpose |
|------|---------|
| `-ffp-contract=fast` | FMA instruction fusion |
| `-fno-math-errno` | Skip errno after math operations |
| `-fno-signed-zeros` | Aggressive FP optimization |
| `-fno-trapping-math` | ARM doesn't trap on FP exceptions |
| `-fvisibility=hidden` | Reduce .so size, eliminate PLT overhead |
| `-fomit-frame-pointer` | Free x29 register for computation |
| `-ffunction-sections` | Enable `--gc-sections` dead code stripping |
| `-fdata-sections` | Enable `--gc-sections` dead data stripping |

### Linker Flags

| Flag | Purpose |
|------|---------|
| `--gc-sections` | Strip unreferenced code and data |
| `--icf=safe` | Merge identical code sections |
| `-z,max-page-size=16384` | Android 15+ 16KB page size support |

### Flags NOT Used (and why)

| Flag | Reason |
|------|--------|
| `-ffast-math` | Implies `-ffinite-math-only`, breaks NaN checks in GGML |
| `-fno-exceptions` | nlohmann/json and common utilities use exceptions |
| `-fno-rtti` | Some GGML internals use `dynamic_cast` |
| `-march=native` | Cross-compiling; `GGML_CPU_ARM_ARCH` handles this |

---

## Build Targets

| Target | Type | Description |
|--------|------|-------------|
| `ggml` | Static lib | GGML tensor library (CPU backend) |
| `llama` | Static lib | Model loading, tokenization, inference, sampling |
| `common` | Static lib | Chat templates, JSON schema, sampling, ngram cache |
| `tn-engine` | Static lib | GGMLEngine, VLM Engine, ToolManager, CharacterEngine, RAG Engine |
| `llama-test-cli` | Executable | Test suite (62+ tests) |

---

## Library Sizes (arm64-v8a, Release, stripped)

| Library | Size |
|---------|------|
| Final `libgguf_lib.so` | ~4.1 MB |

The `-ffunction-sections` + `-fdata-sections` + `--gc-sections` combination strips ~27% of dead code from the final binary.

---

## CI/CD

GitHub Actions workflow at `.github/workflows/release.yml`:

- Triggers on push to `re-write`/`master`, tags `v*`, pull requests, manual dispatch
- Builds for `arm64-v8a` and `x86_64`
- Uploads `.a` + `.so` + headers as artifacts
- Creates GitHub release on version tags with `llama-cpp-android.tar.gz`

---

## ABI Notes

| ABI | Flags | KleidiAI | Notes |
|-----|-------|----------|-------|
| `arm64-v8a` | `-DGGML_CPU_ARM_ARCH=armv8.6-a+i8mm+dotprod+fp16` | ON | Production target |
| `x86_64` | (baseline) | OFF | Emulator testing only |

The `arm64-v8a` target enables:
- **i8mm**: INT8 matrix multiply (fast quantized inference)
- **dotprod**: Dot product instructions (Q4/Q8 kernels)
- **fp16**: Half-precision floating point (F16 compute)
- **KleidiAI**: ARM micro-kernels for optimized GEMM/GEMV
