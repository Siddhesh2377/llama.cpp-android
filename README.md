# ggml-mobile

Stripped-down ggml math library for mobile inference engines.

## What's here

- `ggml/` — The ggml tensor math library (NEON, OpenCL, Hexagon)
  - `ggml/include/ggml.h` — Tensor ops API (mul_mat, rope, flash_attn, rms_norm, etc.)
  - `ggml/include/gguf.h` — GGUF file format reader
  - `ggml/src/ggml-cpu/` — CPU compute kernels (ARM NEON, x86 for host testing)
  - `ggml/src/ggml-opencl/` — OpenCL backend (Adreno GPU)
  - `ggml/src/ggml-hexagon/` — Hexagon DSP backend

## What was removed

All of llama.cpp inference code. This is **just the math library** — use it to build your own inference engine.

## Build for Android

```bash
NDK=$HOME/Android/Sdk/ndk/28.0.12916984
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-27 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=ON -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON
cmake --build build-android -j$(nproc)
```

## Build for Linux (host testing)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=ON
cmake --build build -j$(nproc)
```
