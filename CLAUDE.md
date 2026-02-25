# CLAUDE.md — Project Context for Claude Code

## Project: VLM Inference Engine (Raw GGML)

A modular, heterogeneous inference engine for GGUF-based VLMs built on raw GGML tensor ops.
Stripped-down fork of llama.cpp — no llama.cpp runtime, just ggml math library + custom SDK.

## Current Focus: Android-first VLM pipeline

**Target model**: SmolVLM-500M-Instruct (q8_0 + q8_0.mmproj)
- LLM: SmolLM2-360M (llama arch, 960 embd, 30 layers, SiLU, GQA 15/5)
- Vision: SigLIP-SO (768 embd, 12 layers, 512px, patch=16, GELU)
- Projector: idefics3 pixel shuffle (sf=4) + FC → 960

**Test device**: Samsung A059 (Snapdragon, Adreno 810, arm64-v8a, 7.6GB RAM)
- ADB connected: `adb-001593528004300-gQg9t5._adb-tls-connect._tcp`
- Models on device: `/sdcard/Download/SmolVLM-500M-Instruct-q8_0.{gguf,mmproj}`

## Architecture

```
llama.cpp-custom/
├── ggml/                    # GGML math library (CPU NEON + OpenCL backends)
├── android/                 # Engine SDK + test CLIs
│   ├── include/gguf-engine/ # SDK headers (14 modules, keeping 5)
│   ├── src/                 # SDK implementations
│   ├── vlm-test.cpp         # NEW: VLM pipeline test CLI
│   ├── main.cpp             # Legacy character engine CLI
│   └── CMakeLists.txt       # Build config
├── research/                # Design docs
│   └── Research-VLM-Androids.md  # THE reference doc — read first!
└── CLAUDE.md                # This file
```

## Modules: Keep vs Remove

| Status | Module | Purpose |
|--------|--------|---------|
| KEEP | model | GGUF loading, weight management |
| KEEP | graph | Transformer compute graph |
| KEEP | sampling | Token sampling |
| KEEP | tokenizer | BPE tokenizer |
| KEEP | vision | SigLIP ViT + idefics3 projector |
| REMOVE | interventions | Character behavior — not needed |
| REMOVE | personality | JSON character profiles |
| REMOVE | emotional | Mood tracking |
| REMOVE | grammar | JSON state machine |
| REMOVE | boundaries | Stop strings |
| REMOVE | rag | RAG system |
| REMOVE | stall | Filler text |
| REMOVE | tui | Terminal UI |
| REMOVE | chat | Chat loop |

## Key Technical Details

### GGML Execution Pattern
```cpp
// 1. Build graph in ggml_context
ggml_cgraph* graph = build_graph(ctx, state, seq_len, kv_pos, kv_len, ...);
// 2. Allocate with gallocr
ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
ggml_gallocr_alloc_graph(galloc, graph);
// 3. Set inputs
ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "inp_tokens"), data, 0, nbytes);
// 4. Compute
ggml_backend_graph_compute(backend, graph);
ggml_backend_synchronize(backend);
// 5. Read outputs
ggml_backend_tensor_get(ggml_graph_get_tensor(graph, "logits"), buf, offset, nbytes);
// 6. Free
ggml_gallocr_free(galloc); ggml_free(ctx);
```

### KV Cache Layout
`[head_dim, max_ctx, n_head_kv]` per layer, F16, flash_attn_ext format.

### VLM Pipeline Stages
1. Image preprocess → normalized pixels [W, H, 3]
2. Vision encode (SigLIP ViT) → features [768, 1024]
3. Pixel shuffle + FC projector → vision_embeds [960, 64]
4. Embedding merge: text_embeds + vision_embeds → merged [960, total_seq]
5. LLM prefill (from embeddings) → logits + populated KV cache
6. LLM decode (autoregressive, token-based) → generated text

### Build Commands
```bash
# Host (Linux x86_64)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Android (NDK cross-compile)
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-27 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=ON -DGGML_NATIVE=OFF
cmake --build build-android -j

# Push to device and run
adb push build-android/android/vlm-test /data/local/tmp/
adb shell /data/local/tmp/vlm-test \
  --model /sdcard/Download/SmolVLM-500M-Instruct-q8_0.gguf \
  --mmproj /sdcard/Download/SmolVLM-500M-Instruct-q8_0.mmproj
```

## Platform Strategy
- **Phase 1 (current)**: Android ARM64 + CPU backend — get pipeline correct
- **Phase 2**: Android ARM64 + OpenCL Adreno — GPU acceleration
- **Phase 3**: Linux x86_64 + CPU backend — portability
- **Phase 4**: Linux + OpenCL/CUDA — desktop GPU

## Design Principles
1. MODULAR: New architectures pluggable without rewriting core
2. PLATFORM-AWARE: Same op dispatches differently per device
3. PHASE-AWARE: Prefill (GEMM) and decode (GEMV) are different problems
4. ZERO-COPY FIRST: On UMA (Android), never copy weights
5. FUSE AGGRESSIVELY: Fewer kernel launches = less overhead
6. PROFILE-DRIVEN: Every optimization must be measurable

## First Run Results (2026-02-26, CPU-only, SmolVLM-500M Q8_0)

| Stage | Time | Notes |
|-------|------|-------|
| LLM load | 755ms | 414.9 MB (q8_0, 32 layers, llama arch) |
| Vision load | 200ms | 103.7 MB (SigLIP-SO, 12 layers) |
| KV cache init | 24ms | 80 MB (F16, 2048 ctx) |
| **Vision encode** | **6172ms** | **BOTTLENECK — must move to GPU** |
| LLM prefill (82 tok) | 1030ms | 12.6 ms/tok |
| LLM decode | 40.7 ms/tok | 24.1 tok/s (greedy) |
| Total memory | 598.6 MB | Fits comfortably in 7.6 GB |

**GPU detected**: Adreno 810, OpenCL 3.0, FP16, SVM fine-grain
**Device**: SmolLM2-360M arch: llama, n_embd=960, n_head=15/5, n_layer=32, n_ff=2560

## Current State (2026-02-26)
- [x] Project study complete
- [x] Research doc reviewed (research/Research-VLM-Androids.md)
- [x] VLM test CLI created (android/vlm-test.cpp)
- [x] CMakeLists updated with vlm-test target
- [x] Build for Android and test on device
- [x] Verify vision encode produces sane outputs (no NaN, mean=-0.23)
- [x] Verify full VLM pipeline end-to-end (prefill+decode working)
- [x] Profile and identify bottlenecks (vision=6.2s, decode=40ms/tok)
- [ ] Strip character engine modules
- [ ] Clean up APIs for VLM-only use
- [ ] Move vision encode to GPU (biggest win: 6.2s → target <500ms)
- [ ] Optimize LLM decode with GPU (40ms → target <25ms)
