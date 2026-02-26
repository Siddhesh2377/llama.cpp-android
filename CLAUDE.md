# CLAUDE.md — Project Context for Claude Code

## Project: VLM Inference Engine (Raw GGML)

A modular inference engine for GGUF-based VLMs built on raw GGML tensor ops.
Stripped-down fork of llama.cpp — no llama.cpp runtime, just ggml math library + custom SDK.

## Current Focus: Android-first VLM pipeline — SPEED OPTIMIZATION

**Target model**: SmolVLM-500M-Instruct (q8_0 + q8_0.mmproj)
- LLM: SmolLM2-360M (llama arch, 960 embd, 32 layers, SiLU, GQA 15/5)
- Vision: SigLIP-SO (768 embd, 12 layers, 512px, patch=16, GELU)
- Projector: idefics3 pixel shuffle (sf=4) + FC → 960

**Test device**: Nothing Phone 3a (Snapdragon 7s Gen 3, Adreno 810, arm64-v8a, 6GB RAM)
- CPU: 4× little (1.8 GHz) + 4× big (2.4-2.5 GHz)
- Models on device: `/sdcard/Download/SmolVLM-500M-Instruct-q8_0.{gguf,mmproj}`
- Test image: `/sdcard/Download/VLM-TEST-IMAGE.jpg` (1024×1024)
- Testing: mobile-only (no emulator), always on-device via ADB

**Goals & Status**:
- Decode: 50+ tok/s → Q5_0 achieves **47.9 tok/s** (128 tokens) with good quality
- Vision: <1s → **2450ms** (compute-bound, stuck at I8MM throughput limit)
- Prefill: Q5_0 gives **734ms** (5.8× faster than Q8_0's 4305ms)

## Architecture

```
llama.cpp-custom/
├── ggml/                    # GGML math library (CPU NEON + OpenCL backends)
├── android/                 # Engine SDK + test CLIs
│   ├── include/gguf-engine/ # SDK headers (types, model, graph, vision, sampling, tokenizer, utils)
│   ├── include/stb_image.h  # Single-header image loader
│   ├── src/                 # SDK implementations (model, graph, sampling, tokenizer, vision)
│   ├── vlm-test.cpp         # VLM pipeline test CLI (main optimization testbed)
│   └── CMakeLists.txt       # Build config
├── AutoMatrix/              # Visual ML IDE
│   ├── web/                 # Svelte frontend (dark warm theme)
│   ├── server/              # C++ backend (cpp-httplib + gguf-engine)
│   └── plugins/             # Plugin definitions (JSON)
├── research/                # Design docs
│   └── Research-VLM-Androids.md  # THE reference doc
└── CLAUDE.md                # This file
```

## Build Commands
```bash
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=~/Android/Sdk/ndk/28.2.13676358/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=ON -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH="armv8.6-a+i8mm+dotprod+fp16" \
  -DGGML_CPU_REPACK=OFF \
  -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DGGML_OPENCL_EMBED_KERNELS=ON
cmake --build build-android --target vlm-test -j$(nproc)

# Push & test (recommended config: Q5_0, 4 threads)
adb push build-android/bin/vlm-test build-android/bin/lib*.so /data/local/tmp/
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./vlm-test \
  --model /sdcard/Download/SmolVLM-500M-Instruct-q8_0.gguf \
  --mmproj /sdcard/Download/SmolVLM-500M-Instruct-q8_0.mmproj \
  --image /sdcard/Download/VLM-TEST-IMAGE.jpg --threads 4 --q5 --max-tokens 128"
```

## CLI Flags
- `--threads N`: CPU thread count (default 4, BEST for all quant types)
- `--gpu`: Direct GPU compute for LLM (faster prefill but slower decode)
- `--q4`: Runtime requantize Q8_0 → Q4_0 (48.5 tok/s, mediocre quality)
- `--q4_1`: Runtime requantize Q8_0 → Q4_1 (50.6 tok/s, BROKEN quality at 360M)
- `--q5`: Runtime requantize Q8_0 → Q5_0 (**RECOMMENDED**: 47.9 tok/s, good quality)
- `--q5_1`: Runtime requantize Q8_0 → Q5_1 (40.8 tok/s)
- `--mixed`: Attention Q8_0 + FFN Q4_0 (42.8 tok/s)
- `--mixed5`: Attention Q8_0 + FFN Q5_0 (44.6 tok/s)
- `--vlayers N`: Override vision layer count (-1 = all)
- `--image-size N`: Override vision resolution (must be multiple of patch_size)

## Key Technical Details

### SmolVLM Chat Template
```
BOS + "User:" + <fake_token_around_image>(49189) + <global-img>(49152)
+ [64 vision embeddings]
+ <fake_token_around_image>(49189) + prompt + <end_of_utterance>(49279) + "\n" + "Assistant:"
```

### VLM Pipeline Stages & Timing (Q5_0, 4 threads)
1. Vision encode (SigLIP ViT, CPU, Q8_0) → [960, 64] — **2450ms**
2. Prefill (77 tokens, CPU Q5_0, gallocr) — **734ms** (103 tok/s)
3. Decode (128 tokens, CPU Q5_0, gallocr) — **20.3ms/tok** (47.9 tok/s)

### Memory Budget
| Component | Q8_0 | Q5_0 |
|-----------|------|------|
| LLM weights | 414.9 MB | 268.5 MB |
| Vision weights | 103.7 MB | 103.7 MB |
| KV cache (F16) | 80.0 MB | 80.0 MB |
| **Total** | **598.6 MB** | **452.3 MB** |

---

## OPTIMIZATION LOG

### Decode Performance Matrix (128 tokens, 4 threads)
| Quant | Weight MB | ms/tok | tok/s | BW GB/s | Quality |
|-------|-----------|--------|-------|---------|---------|
| Q8_0 | 414.9 | 29.5 | 33.1 | 14.6 | Best |
| Mixed5 | 313.6 | 21.7 | 44.6 | 14.5 | Good |
| Q5_1 | 292.9 | 23.7 | 40.8 | 12.4 | Good |
| **Q5_0** | **268.5** | **20.3** | **47.9** | **13.8** | **Good** |
| Q4_1 | 244.1 | 19.0 | 50.6 | 13.6 | Garbage |
| Q4_0 | 219.7 | 19.8 | 48.5 | 11.1 | Mediocre |

### Thread Scaling (Q8_0 decode, 16 tokens)
| Threads | ms/tok | tok/s |
|---------|--------|-------|
| 1 | 29.5 | 33.0 |
| 2 | 29.7 | 32.8 |
| 4 | 28.5 | 34.2 |
| 8 | 55.3 | 17.8 |

Q8_0 is bandwidth-limited (single core saturates ~14 GB/s). Q5_0 is compute-limited and scales: 1t=25.7, 4t=49.7 tok/s.

### GPU vs CPU
| Mode | Prefill (77 tok) | Decode tok/s |
|------|------------------|-------------|
| CPU Q8_0 | 4305ms | 34.2 |
| CPU Q5_0 | 734ms | 47.9 |
| GPU Q8_0 | 1513ms | 23.1 |

GPU is 3× faster for prefill but 0.7× slower for decode.

### Vision Encode Investigation
| Config | Time | Notes |
|--------|------|-------|
| CPU 4t (Q8_0) | 2450ms | I8MM GEMM, ~200ms/layer |
| GPU (Q8_0) | 4928ms | All ops on Adreno |
| CPU 1t (1 layer) | 848ms | Single core baseline |
| CPU 4t (1 layer) | 248ms | 3.4× scaling |
| 384px variable res | 1078ms | Garbage output (token count changes) |

Bottleneck: compute-bound I8MM at ~18 GFLOPS/core. 12 layers × 200ms + 47ms overhead = 2450ms.

### Bugs Fixed
1. **Fused weight copy guards** (model.cpp): checked nullptr lw.attn_q/k/v instead of lw.attn_qkv
2. **OpenCL Q8_0 SOA multi-set_tensor**: calling set_tensor multiple times corrupts tensor->extra. Fix: assemble full tensor in CPU buffer, single set_tensor. Fixed in BOTH vision.cpp AND model.cpp.
3. **get_alloc_size callback**: Q8_0 SOA needs extra alignment bytes for subbuffer allocation

### Key Technical Findings
1. **Memory bandwidth ceiling**: ~14 GB/s practical on Snapdragon 7s Gen 3 (regardless of thread count 1-4)
2. **Q5_0 is the sweet spot**: 40% less bandwidth than Q8_0, good quality, near-50 tok/s
3. **Q5_0 prefill is 5.8× faster**: 734ms vs 4305ms (compute savings compound for batch=77)
4. **8 threads kills performance**: LITTLE cores are too slow, OMP spreads work there
5. **OMP affinity**: big cores only via `setenv("OMP_PLACES", "{7},{4},{5},{6}")`
6. **ne11=1 forces nrc=1**: I8MM nrows=2 is never used for GEMV decode (ggml-cpu.c:1407)
7. **Decode uses gallocr not scheduler**: 0.1ms less alloc overhead per token
8. **Q4_1 produces garbage** at 360M model size despite vec_dot being correct
9. **mmproj already stores Q8_0**: quantize=false has no effect on vision weights

### Graph Optimization Analysis
- **Fused QKV**: attn_qkv weight loaded as single tensor, split in graph build — saves 3 separate loads
- **Gate+Up fusion**: ffn_gate_up fused tensor, single matmul then split — halves FFN matmul calls
- **Flash attention**: not implemented (would help prefill, marginal for single-token decode)
- **Continuous FFN ops**: gate_up → silu → down could be fused into single kernel (not done)
- **Architecture modularity**: hardcoded llama/qwen2 arch in graph.cpp, SigLIP in vision.cpp — adding new models requires C++ edits

### Available GGML Quant Types
| Type | Bits | Block | Has I8MM vec_dot | Notes |
|------|------|-------|------------------|-------|
| F32 | 32 | 1 | No | Reference |
| F16 | 16 | 1 | No | KV cache default |
| Q8_0 | 8.5 | 32 | Yes | Best quality, BW-limited |
| Q5_1 | 5.5 | 32 | Yes (via Q8_1) | Good quality |
| Q5_0 | 5.0 | 32 | Yes | **Sweet spot** |
| Q4_1 | 4.5 | 32 | Yes (via Q8_1) | Garbage at 360M |
| Q4_0 | 4.0 | 32 | Yes | Mediocre quality |
| Q2_K | 2-3 | 256 | No | K-quant, not tested |
| Q3_K | 3-4 | 256 | No | K-quant, not tested |
| IQ4_NL | 4.0 | 32 | No | imatrix, not tested |

### Remaining Optimization Opportunities
- **Vision encode**: 2450ms bottleneck. Needs fundamental change (smaller model, tiling, etc.)
- **GPU prefill + CPU decode**: Potential hybrid for best of both worlds (needs SVM or weight copy)
- **I8MM for GEMV**: Modify alignment check to allow nrc=2 for single-token decode
- **Vocabulary pruning**: Skip output matmul for unlikely tokens

## Completed Tasks
- [x] Fix VLM chat template (correct special tokens)
- [x] Real image loading (stb_image + bilinear resize)
- [x] Strip character engine (removed unused modules)
- [x] GPU acceleration attempt (scheduler, direct GPU, vision)
- [x] Runtime requantization (Q4_0, Q4_1, Q5_0, Q5_1, mixed variants)
- [x] Thread optimization (1-8 threads, OMP affinity to big cores)
- [x] QKV weight fusion + Gate+Up fusion
- [x] ARM arch flags (armv8.6-a+i8mm+dotprod+fp16)
- [x] OpenCL Q8_0 SOA subbuffer bug fix (vision.cpp + model.cpp)
- [x] Vision encode profiling (per-layer, GPU, variable resolution)
- [x] Gallocr vs scheduler comparison for decode
- [x] Comprehensive quant/thread/GPU benchmarking
