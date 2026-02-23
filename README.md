# llama.cpp-custom

Custom fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with a **Character Intelligence Engine** — 10 runtime intervention surfaces for controlling LLM personality, emotion, and behavior without retraining or LoRA swaps.

Built for **Android** (ARMv8/ARMv9 via NDK) and **Linux** (x86_64/aarch64).

> Base: llama.cpp commit `f64e65b45` (2026-01-18)

---

## What's Different

Standard llama.cpp treats the model as a frozen black box — you feed tokens in, get tokens out. This fork opens the transformer graph at **10 points** and lets you inject, scale, mask, and learn parameters at runtime. All interventions are RAM-only; the GGUF file is never modified.

### Intervention Surfaces

| ID | Surface | What it does | Parameters | Flash Attn |
|----|---------|-------------|-----------|------------|
| **A** | Activation Capture | Read hidden states per layer | `[n_layer][n_embd]` | Compatible |
| **C** | Attention Bias | Inject log-space bias on KQ scores pre-softmax | Token range + bias per layer | Disables per-layer |
| **D** | Head Rescaling | Per-head scalar multiplier on attention output | `[n_layer][n_head]` | Compatible |
| **E** | Attention Temperature | Per-head softmax sharpness control | `[n_layer][n_head]` | Disables per-layer |
| **G** | LayerNorm Shift | Additive offset post-normalization | `[n_layer][n_embd]` | Compatible |
| **GR** | Gated Residual | Per-layer scalar gates on attn/FFN outputs | `[n_layer]` × 2 | Compatible |
| **P4** | Hypernetwork LoRA | Rank-4 FFN LoRA for middle layers (37%-70% depth) | `[rank × n_embd]` + `[n_ff × rank]` | Compatible |
| **P5** | Sparse Masks | Per-layer FFN neuron masking | `[n_layer][n_ff]` in [0,1] | Compatible |
| **P6** | KAN-lite | Piecewise-linear spline activation overlay (8 knots) | `[n_layer][8]` | Compatible |
| **P7** | Forward Learning (SPSA) | Gradient-free parameter tuning via 2 perturbed passes | KAN coefficients | Compatible |

Plus:
- **Early Exit** — skip layers after N transformer blocks (speculative decoding)
- **Save/Load Intervention State** — persist all learnable parameters to binary file

### 28 New API Functions

```c
// Activation Capture (A)
void llama_set_capture_layer_outputs(struct llama_context * ctx, bool enable);
int  llama_get_n_captured_layers(const struct llama_context * ctx);
const float * llama_get_captured_layer_output(const struct llama_context * ctx, int layer);

// Attention Bias (C)
void llama_set_attention_bias(struct llama_context * ctx, int start, int end, float bias, int l_start, int l_end);
void llama_clear_attention_bias(struct llama_context * ctx);

// Head Rescaling (D)
void llama_set_head_scale(struct llama_context * ctx, int layer, int head, float scale);
void llama_reset_head_scales(struct llama_context * ctx);

// Attention Temperature (E)
void llama_set_attention_temperature(struct llama_context * ctx, int layer, int head, float temp);
void llama_reset_attention_temperatures(struct llama_context * ctx);

// LayerNorm Shift (G)
void llama_set_norm_offsets(struct llama_context * ctx, int layer, const float * offsets, int n_embd);
void llama_reset_norm_offsets(struct llama_context * ctx);

// Gated Residual (GR)
void llama_set_residual_gates(struct llama_context * ctx, int layer, float attn_gate, float ffn_gate);
void llama_reset_residual_gates(struct llama_context * ctx);

// Early Exit
void llama_set_early_exit_layer(struct llama_context * ctx, int layer);
void llama_reset_early_exit_layer(struct llama_context * ctx);

// Sparse Masks (P5)
void llama_set_sparse_mask(struct llama_context * ctx, int layer, const float * mask, int n_ff);
void llama_init_sparse_masks(struct llama_context * ctx, float sparsity, uint32_t seed);
void llama_reset_sparse_masks(struct llama_context * ctx);

// KAN-lite (P6)
void llama_set_kan_coefficients(struct llama_context * ctx, int layer, const float * coeffs, int n_knots);
void llama_set_kan_alpha(struct llama_context * ctx, float alpha);
void llama_reset_kan(struct llama_context * ctx);

// Hypernetwork LoRA (P4)
void llama_init_hypernetwork(struct llama_context * ctx, int rank);
void llama_set_hypernetwork_lora(struct llama_context * ctx, int layer_idx, const float * A, const float * B);
void llama_set_hypernetwork_strength(struct llama_context * ctx, float strength);
void llama_reset_hypernetwork(struct llama_context * ctx);

// Forward Learning (P7)
float llama_forward_learn_step(struct llama_context * ctx, const llama_token * tokens, int n, float lr, float noise);

// State Persistence
bool llama_save_intervention_state(struct llama_context * ctx, const char * path);
bool llama_load_intervention_state(struct llama_context * ctx, const char * path);
```

---

## Build (Android NDK)

```bash
mkdir build-android && cd build-android

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_LLAMAFILE=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)
```

This produces 7 CPU backend `.so` variants (armv8.0, armv8.2+dotprod, armv8.2+fp16, armv8.6+i8mm, armv9.0+sve, armv9.2+sme, armv9.2+sme2) that are selected at runtime.

## Build (Linux x86_64)

```bash
mkdir build && cd build

cmake .. \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_BACKEND_DL=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)
```

---

## Architecture

```
Token Input
    |
    v
[Embedding Layer]
    |
    v
for each transformer layer (0..N):
    |
    +-- [LayerNorm] --+-- (G) Norm Offset --+
    |                                        |
    +-- [Attention]                          |
    |     +-- (C) KQ Score Bias             |
    |     +-- (E) Softmax Temperature       |
    |     +-- (D) Head Output Scaling       |
    |                                        |
    +-- (GR) Attention Gate ---- [Residual Add]
    |
    +-- [LayerNorm] --+-- (G) Norm Offset --+
    |                                        |
    +-- [FFN]                                |
    |     +-- (P5) Sparse Neuron Mask       |
    |     +-- (P6) KAN Activation Overlay   |
    |     +-- (P4) Hypernetwork LoRA Add    |
    |                                        |
    +-- (GR) FFN Gate ------ [Residual Add]
    |
    +-- (A) Capture Hidden State
    |
    +-- Early Exit Check (skip remaining layers)
    |
    v
[Output Head] --> Logits
```

---

## Modified Files

Only **5 files** were changed from upstream llama.cpp:

| File | Changes |
|------|---------|
| `include/llama.h` | 28 new C API functions |
| `src/llama-context.h` | Intervention state vectors in context struct |
| `src/llama-context.cpp` | Initialization, graph param passing |
| `src/llama-graph.h` | 6 new graph input classes, param struct |
| `src/llama-graph.cpp` | Tensor filling, KAN spline eval, LoRA application |

Everything else (ggml, tokenizer, sampling, model loading) is untouched upstream.

---

## Documentation

See [`docs/`](docs/) for detailed technical documentation:

- [**Intervention Surfaces**](docs/intervention-surfaces.md) — deep dive into each of the 10 surfaces
- [**Roadmap**](docs/roadmap.md) — what's next

---

## Integration

This fork is used by:
- **[AiSystems](https://github.com/Siddhesh2377/Ai-Systems-New)** — Android SDK (JNI bindings in `ai_gguf/`)
- **[ToolNeuron](https://github.com/Siddhesh2377/ToolNeuron)** — Android app with character intelligence

JNI layer calls these C APIs through `GGUFNativeLib.kt` -> NDK -> `ai_gguf.cpp` -> `llama.h`.

---

## License

MIT (same as upstream llama.cpp)

Based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov.
