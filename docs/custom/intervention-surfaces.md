# Intervention Surfaces — Technical Reference

This document describes every runtime intervention surface added to this llama.cpp fork. All modifications live in 5 files (`llama.h`, `llama-context.h/cpp`, `llama-graph.h/cpp`) and are designed to be zero-cost when disabled — the graph builder checks for empty vectors and skips tensor creation entirely.

---

## Design Principles

1. **RAM-only** — GGUF model file is never modified. All state lives in `llama_context`.
2. **Zero-cost when disabled** — empty vectors = no graph nodes created = no overhead.
3. **Per-layer flash attention check** — only disables flash attention on layers that actually need it (C, E). Unmodified layers keep full FlashAttention speed.
4. **Graph reuse** — `can_reuse()` checks all intervention parameters. If unchanged between inferences, the previous graph topology is reused.
5. **No reallocation** — all vectors are pre-allocated at context creation time. Parameter updates are `memcpy`-level fast.

---

## A — Activation Capture

**Purpose**: Read hidden states after each transformer layer for probing, analysis, and direction vector extraction.

**API**:
```c
void llama_set_capture_layer_outputs(struct llama_context * ctx, bool enable);
int  llama_get_n_captured_layers(const struct llama_context * ctx);
const float * llama_get_captured_layer_output(const struct llama_context * ctx, int layer);
```

**How it works**:
- When enabled, the graph builder inserts `ggml_cpy` nodes after each layer's residual stream output.
- Copies go to pre-allocated `captured_layer_data[layer]` vectors of size `[n_embd]`.
- Read-only — does not affect generation.

**Use case**: Extract contrastive direction vectors for control vectors. Run positive prompt ("You are happy") and negative prompt ("You are sad") through the model, capture layer outputs, compute `mean(positive) - mean(negative)` per layer.

**Performance**: ~2% overhead when enabled (extra copy per layer). Zero when disabled.

---

## C — Attention Score Bias

**Purpose**: Boost or suppress attention to specific token ranges, per-layer.

**API**:
```c
void llama_set_attention_bias(struct llama_context * ctx, int start_pos, int end_pos, float bias, int layer_start, int layer_end);
void llama_clear_attention_bias(struct llama_context * ctx);
```

**How it works**:
- Stores `llm_attn_bias_entry` structs in `ctx->attn_biases`.
- During graph build, creates `[n_kv]` bias tensors per affected layer.
- Bias is added to KQ scores **pre-softmax** (log-space).
- `bias = 2.0` approximately 7.4x attention multiplier. `bias = -2.0` approximately 0.13x suppression.
- `layer_end = -1` means all layers.

**Flash Attention**: Disables flash attention for affected layers only.

**Use case**: Make the model pay more attention to system prompt tokens, or suppress attention to specific conversation ranges.

**Graph class**: `llm_graph_input_attn_bias`

---

## D — Head Rescaling

**Purpose**: Per-head scalar multiplier on attention output. Amplify important heads, suppress noisy ones.

**API**:
```c
void llama_set_head_scale(struct llama_context * ctx, int layer, int head, float scale);
void llama_reset_head_scales(struct llama_context * ctx);
```

**How it works**:
- Stores `head_scales[n_layer][n_head]` in context (default: all 1.0).
- During graph build, creates `[1,1,n_head,1]` scale tensors.
- Applied via `ggml_mul` on attention output before residual add.

**Values**:
- `1.0` = default (no change)
- `0.0` = ablate head (zero out its contribution)
- `2.0` = double the head's importance
- `-1.0` = reverse the head's direction

**Flash Attention**: Compatible — does not affect attention computation, only output scaling.

**Use case**: Computed from probed direction vector norms. High-importance layers get 1.0-1.5x boost, low-importance get 0.7-1.0x suppression. Middle layers left at 1.0 to preserve general capability.

**Graph class**: `llm_graph_input_head_intervention`

---

## E — Attention Temperature

**Purpose**: Per-head softmax sharpness control. Sharp = focused attention, flat = distributed attention.

**API**:
```c
void llama_set_attention_temperature(struct llama_context * ctx, int layer, int head, float temperature);
void llama_reset_attention_temperatures(struct llama_context * ctx);
```

**How it works**:
- Stores `attn_temperatures[n_layer][n_head]` (default: all 1.0).
- Computes inverse temperature `1/T` and multiplies KQ scores before softmax.
- `T < 1.0` = sharper (more focused on top tokens)
- `T > 1.0` = flatter (more distributed attention)

**Flash Attention**: Disables flash attention for layers with non-default temperatures.

**Typical values**:
- Early layers: `1.3` (sharp — pattern matching)
- Middle layers: `1.0` (default — preserve general reasoning)
- Late layers: `0.8` (flat — broader generation)

**Graph class**: `llm_graph_input_head_intervention` (shared with D)

---

## G — LayerNorm Affine Shift

**Purpose**: Per-layer additive offset post-normalization. Cheapest personality modification.

**API**:
```c
void llama_set_norm_offsets(struct llama_context * ctx, int layer, const float * offsets, int n_embd);
void llama_reset_norm_offsets(struct llama_context * ctx);
```

**How it works**:
- Stores `norm_offsets[n_layer][n_embd]` vectors.
- Applied immediately after LayerNorm, before attention/FFN computation.
- Shifts the activation distribution without changing the model weights.

**Flash Attention**: Fully compatible — zero penalty.

**Use case**: Personality tuning. Derived from control vector direction vectors projected into the normalization space.

**Graph class**: `llm_graph_input_norm_offsets`

---

## GR — Gated Residual

**Purpose**: Per-layer scalar gates on attention and FFN residual connections. Control how much each layer contributes to the output.

**API**:
```c
void llama_set_residual_gates(struct llama_context * ctx, int layer, float attn_gate, float ffn_gate);
void llama_reset_residual_gates(struct llama_context * ctx);
```

**How it works**:
- Stores `residual_attn_gates[n_layer]` and `residual_ffn_gates[n_layer]` (default: all 1.0).
- `gate = 0.0` = skip layer entirely (residual passthrough)
- `gate = 1.0` = default behavior
- `gate = 2.0` = amplify layer output
- Applied via `ggml_scale` on attention/FFN output before residual add.

**Flash Attention**: Fully compatible.

**Use case**: Dynamic layer importance based on emotional state or persona requirements. Suppress reasoning layers for creative writing, amplify them for analytical tasks.

---

## P4 — Hypernetwork FFN LoRA

**Purpose**: Runtime-trainable low-rank adaptation of FFN layers. The most expressive intervention surface.

**API**:
```c
void llama_init_hypernetwork(struct llama_context * ctx, int rank);
void llama_set_hypernetwork_lora(struct llama_context * ctx, int layer_idx, const float * A, const float * B);
void llama_set_hypernetwork_strength(struct llama_context * ctx, float strength);
void llama_reset_hypernetwork(struct llama_context * ctx);
```

**How it works**:
- Initializes rank-4 LoRA matrices for middle layers (37%-70% depth by default).
- `A` matrix: `[rank × n_embd]` — projects FFN input down to low-rank space.
- `B` matrix: `[n_ff × rank]` — projects back up to FFN output space.
- Applied as: `ffn_output += strength * (B^T @ (A^T @ ffn_input))`

**Memory**: For a 0.5B model (896 embd, 4864 ff, rank 4):
- Per target layer: `4 × 896 + 4864 × 4 = 23,040 floats = 90KB`
- ~8 target layers: `~720KB` total

**Graph class**: `llm_graph_input_hypernetwork`

---

## P5 — Dynamic Sparse Masks

**Purpose**: Per-layer FFN neuron masking. Disable specific neurons to prune model capacity or specialize behavior.

**API**:
```c
void llama_set_sparse_mask(struct llama_context * ctx, int layer, const float * mask, int n_ff);
void llama_init_sparse_masks(struct llama_context * ctx, float sparsity, uint32_t seed);
void llama_reset_sparse_masks(struct llama_context * ctx);
```

**How it works**:
- Stores `sparse_masks[n_layer][n_ff]` with values in [0.0, 1.0].
- `1.0` = neuron active (default), `0.0` = neuron disabled.
- Applied via `ggml_mul` before FFN down-projection.
- `init_sparse_masks(0.1, seed)` randomly disables 10% of neurons.

**Graph class**: `llm_graph_input_sparse_mask`

---

## P6 — KAN-lite Activation Overlay

**Purpose**: Learnable piecewise-linear activation function added on top of existing FFN activations. Inspired by Kolmogorov-Arnold Networks.

**API**:
```c
void llama_set_kan_coefficients(struct llama_context * ctx, int layer, const float * coeffs, int n_knots);
void llama_set_kan_alpha(struct llama_context * ctx, float alpha);
void llama_reset_kan(struct llama_context * ctx);
```

**How it works**:
- 8 knot points per layer on grid [-4, +3] with spacing 1.0.
- For input activation `x`:
  1. Map to grid index: `idx = clamp(floor(x - grid_min), 0, 6)`
  2. Fraction: `frac = x - grid_min - idx`
  3. Spline value: `c[idx] * (1-frac) + c[idx+1] * frac`
  4. Output: `activation + alpha * spline_value`
- `alpha = 0` disables (default). Typical range: 0.01-0.1.

**Constants**:
```c
KAN_N_KNOTS = 8
KAN_GRID_MIN = -4.0
KAN_GRID_SPACING = 1.0
```

**Graph class**: `llm_graph_input_kan`

---

## P7 — Forward-Only Learning (SPSA)

**Purpose**: Gradient-free parameter tuning using Simultaneous Perturbation Stochastic Approximation. Runs entirely on-device without backpropagation.

**API**:
```c
float llama_forward_learn_step(struct llama_context * ctx, const llama_token * tokens, int n_tokens, float learning_rate, float noise_scale);
```

**How it works**:
1. Generate random perturbation vector `delta` (±1 per parameter).
2. Run forward pass with `params + noise_scale * delta` → compute loss L+.
3. Run forward pass with `params - noise_scale * delta` → compute loss L-.
4. Estimate gradient: `g ≈ (L+ - L-) / (2 * noise_scale * delta)`.
5. Update: `params -= learning_rate * g`.
6. Returns estimated loss difference (positive = improvement).

**What it learns**: Currently updates KAN coefficients (P6). Future: sparse masks, LoRA matrices.

**Performance**: ~3MB temporary probe context. Two forward passes per step.

**Typical hyperparameters**:
- `learning_rate`: 0.001 - 0.01
- `noise_scale`: 0.01 - 0.1

---

## State Persistence

**API**:
```c
bool llama_save_intervention_state(struct llama_context * ctx, const char * path);
bool llama_load_intervention_state(struct llama_context * ctx, const char * path);
```

**Format**: Binary file with versioned header. Saves:
- KAN coefficients + alpha
- Sparse masks
- Hypernetwork LoRA matrices + strength
- Residual gates
- Head scales
- Attention temperatures
- Norm offsets

Backward-compatible: older state files are loaded with missing fields defaulting to neutral values.

---

## Memory Budget (Qwen2-0.5B: 24 layers, 896 embd, 4864 ff, 14 heads)

| Surface | Per-layer | Total | Notes |
|---------|----------|-------|-------|
| Head scales (D) | 56 B | 1.3 KB | 14 floats |
| Attn temperatures (E) | 56 B | 1.3 KB | 14 floats |
| Residual gates (GR) | 8 B | 192 B | 2 floats |
| Norm offsets (G) | 3.5 KB | 84 KB | 896 floats |
| Sparse masks (P5) | 19 KB | 456 KB | 4864 floats |
| KAN coefficients (P6) | 32 B | 768 B | 8 floats |
| Hypernetwork LoRA (P4) | 90 KB | ~720 KB | 8 target layers |
| Activation capture (A) | 3.5 KB | 84 KB | 896 floats (when enabled) |
| **Total** | | **~1.3 MB** | Negligible vs model size |
