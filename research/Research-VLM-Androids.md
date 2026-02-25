# VLM Engine Blueprint — Raw GGML Heterogeneous Inference on ARM + Linux

> **Purpose**: This is THE reference document for Claude Code (and humans) working on this project.
> It explains every layer of the stack — from GGUF binary bytes on disk to fused OpenCL kernels on Adreno.
> Read this FIRST before touching any code.

---

## Table of Contents

- [1. Project Context & Goals](#1-project-context--goals)
- [2. GGUF File Format — Complete Specification](#2-gguf-file-format--complete-specification)
- [3. GGML Tensor System — How It Works Internally](#3-ggml-tensor-system--how-it-works-internally)
- [4. GGML Compute Graph — What It Does & What We Replace](#4-ggml-compute-graph--what-it-does--what-we-replace)
- [5. GGML Operations Catalog — Every Op and Its Role in VLMs](#5-ggml-operations-catalog--every-op-and-its-role-in-vlms)
- [6. VLM Architecture — How Vision-Language Models Work End-to-End](#6-vlm-architecture--how-vision-language-models-work-end-to-end)
- [7. Quantization Formats — Block Layouts for Every Q-Type](#7-quantization-formats--block-layouts-for-every-q-type)
- [8. Memory Architecture — UMA vs Discrete, The Core Divide](#8-memory-architecture--uma-vs-discrete-the-core-divide)
- [9. Heterogeneous Compute Architecture — Our Design](#9-heterogeneous-compute-architecture--our-design)
- [10. GPU Backend — OpenCL Kernel Strategy](#10-gpu-backend--opencl-kernel-strategy)
- [11. CPU Backend — ARM NEON Kernel Strategy](#11-cpu-backend--arm-neon-kernel-strategy)
- [12. Operator Fusion — What to Fuse and Why](#12-operator-fusion--what-to-fuse-and-why)
- [13. KV Cache Architecture](#13-kv-cache-architecture)
- [14. Vision Encoder Specifics](#14-vision-encoder-specifics)
- [15. Tensor Management & Memory Pooling](#15-tensor-management--memory-pooling)
- [16. Platform Abstraction Layer](#16-platform-abstraction-layer)
- [17. Implementation Order — Phase by Phase](#17-implementation-order--phase-by-phase)
- [18. Quick Reference Tables](#18-quick-reference-tables)

---

## 1. Project Context & Goals

### What We Have

We forked `llama.cpp` and stripped it down to **raw GGML only** — no llama.cpp runtime, no `llama_context`, no
`llama_model`, no `llama_batch`, no built-in compute graph builder. We kept:

- `ggml.h / ggml.c` — core tensor library (types, allocation, op definitions)
- `ggml-backend.h / ggml-backend.c` — backend abstraction layer
- `ggml-cpu.h / ggml-cpu.c` — CPU backend with NEON/AVX kernels
- `ggml-opencl.h / ggml-opencl.c` — OpenCL backend (our primary GPU path)
- `gguf.h / gguf.c` — GGUF file parser (metadata + tensor loading)
- Quantization code (`ggml-quants.h / ggml-quants.c`)

We also have a custom Android SDK with 14 modules. We're **keeping only**:

| Keep | Module          | Why                                        |
|------|-----------------|--------------------------------------------|
| ✅    | `model`         | GGUF loading, weight management            |
| ✅    | `graph`         | Transformer forward pass (rewriting this)  |
| ✅    | `sampling`      | Token sampling (temperature, top-k, top-p) |
| ✅    | `tokenizer`     | BPE tokenizer                              |
| ✅    | `vision`        | SigLIP ViT + projector (VLM)               |
| ❌    | `interventions` | Character behavior — removed               |
| ❌    | `personality`   | JSON character profiles — removed          |
| ❌    | `emotional`     | Mood tracking — removed                    |
| ❌    | `grammar`       | JSON state machine — removed               |
| ❌    | `boundaries`    | Stop strings — simplified into sampling    |
| ❌    | `rag`           | RAG system — removed entirely              |
| ❌    | `stall`         | Filler text — removed                      |
| ❌    | `tui`           | Terminal UI — removed (separate concern)   |
| ❌    | `chat`          | Chat loop — removed (separate concern)     |

### What We're Building

A **modular, heterogeneous inference engine** for GGUF-based VLMs that:

1. Runs on **Android (ARM64 + Adreno/Mali OpenCL)** and **Linux (x86_64 + any OpenCL GPU)**
2. Has **separate optimized kernel paths** for prefill vs decode phases
3. Exploits **UMA zero-copy** on Android, handles **PCIe transfer** on desktop
4. Keeps every tensor op independently tunable per platform
5. Supports VLM architectures generically (LLaVA, Qwen-VL, Phi-Vision, SmolVLM, etc.)

### Design Principles

```
1. MODULAR: New architectures (Mamba, RWKV, future VLMs) pluggable without rewriting core
2. PLATFORM-AWARE: Same op dispatches differently on Adreno vs Mali vs desktop GPU vs CPU
3. PHASE-AWARE: Prefill and decode are treated as fundamentally different compute problems
4. ZERO-COPY FIRST: On UMA, never copy weights. On discrete, minimize PCIe transfers.
5. FUSE AGGRESSIVELY: Fewer kernel launches = less driver overhead = faster inference
6. PROFILE-DRIVEN: Every optimization must be measurable. No guessing.
```

---

## 2. GGUF File Format — Complete Specification

GGUF (GGML Universal File) is the binary format storing model weights, metadata, and tokenizer info. Understanding its
byte-level layout is critical because we mmap it directly and create GPU buffers pointing into it.

### 2.1 Overall File Structure

```
┌──────────────────────────────────────────────────┐
│                  GGUF FILE                        │
├──────────────────────────────────────────────────┤
│  HEADER (fixed)                                   │
│    ├── magic: 4 bytes = "GGUF" (0x46475547)      │
│    ├── version: uint32 = 3 (current)              │
│    ├── n_tensors: uint64                          │
│    └── n_kv: uint64 (metadata key-value pairs)    │
├──────────────────────────────────────────────────┤
│  METADATA KEY-VALUE PAIRS (n_kv entries)          │
│    Each entry:                                    │
│    ├── key_length: uint64                         │
│    ├── key_data: utf8 string                      │
│    ├── value_type: uint32 (enum)                  │
│    └── value_data: variable size                  │
├──────────────────────────────────────────────────┤
│  TENSOR INFOS (n_tensors entries)                 │
│    Each entry:                                    │
│    ├── name_length: uint64                        │
│    ├── name_data: utf8 string                     │
│    ├── n_dims: uint32                             │
│    ├── dims: uint64[n_dims]                       │
│    ├── type: uint32 (ggml_type enum)              │
│    └── offset: uint64 (from start of data section)│
├──────────────────────────────────────────────────┤
│  ALIGNMENT PADDING                                │
│    Padded to `alignment` boundary (default 32)    │
├──────────────────────────────────────────────────┤
│  TENSOR DATA (contiguous, aligned)                │
│    Raw quantized/float weight bytes               │
│    Each tensor starts at its stated offset         │
│    from the beginning of this data section         │
└──────────────────────────────────────────────────┘
```

### 2.2 Metadata Value Types

```c
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,   // uint64 length + utf8 bytes
    GGUF_TYPE_ARRAY   = 9,   // type + uint64 count + elements
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};
```

### 2.3 Critical Metadata Keys for VLMs

These are the KV pairs you MUST parse to build the compute graph:

**Architecture keys:**

```
general.architecture          → "llama", "qwen2", "phi3", etc.
general.name                  → human-readable model name
general.file_type             → quantization type used

{arch}.context_length         → max sequence length
{arch}.embedding_length       → hidden dimension (e.g. 4096)
{arch}.block_count            → number of transformer layers (e.g. 32)
{arch}.feed_forward_length    → FFN intermediate dim (e.g. 11008)
{arch}.attention.head_count   → number of attention heads (e.g. 32)
{arch}.attention.head_count_kv → number of KV heads (for GQA, e.g. 8)
{arch}.attention.layer_norm_rms_epsilon → RMSNorm epsilon
{arch}.rope.dimension_count   → RoPE dimension (usually head_dim)
{arch}.rope.freq_base         → RoPE theta base (e.g. 10000.0)
{arch}.vocab_size             → vocabulary size
```

**VLM-specific keys (varies by architecture):**

```
{arch}.vision.image_size        → input image resolution (e.g. 384)
{arch}.vision.patch_size        → ViT patch size (e.g. 14)
{arch}.vision.embedding_length  → vision hidden dim (e.g. 1152 for SigLIP)
{arch}.vision.block_count       → number of ViT layers (e.g. 27)
{arch}.vision.head_count        → ViT attention heads (e.g. 16)
{arch}.vision.projection_dim    → projection output dim (matches LLM hidden)

clip.projector_type             → "mlp", "ldp" (LDP-v2), "resampler", etc.
clip.vision.image_mean          → normalization mean [R, G, B]
clip.vision.image_std           → normalization std [R, G, B]
```

**Tokenizer keys:**

```
tokenizer.ggml.model            → "gpt2" (BPE), "llama" (SentencePiece), etc.
tokenizer.ggml.tokens           → string array of all tokens
tokenizer.ggml.token_type       → type per token (normal, control, special, etc.)
tokenizer.ggml.merges           → BPE merge rules
tokenizer.ggml.bos_token_id     → beginning of sequence token
tokenizer.ggml.eos_token_id     → end of sequence token
tokenizer.chat_template         → Jinja2 or custom chat template string
```

### 2.4 Tensor Naming Conventions

GGUF tensors follow a naming pattern that maps directly to model architecture:

**LLM tensors (per-layer, repeated `block_count` times):**

```
token_embd.weight                           → [vocab_size, hidden_dim]
blk.{i}.attn_norm.weight                    → [hidden_dim]             (RMSNorm)
blk.{i}.attn_q.weight                       → [hidden_dim, hidden_dim] (Q projection)
blk.{i}.attn_k.weight                       → [hidden_dim, kv_dim]     (K projection, smaller for GQA)
blk.{i}.attn_v.weight                       → [hidden_dim, kv_dim]     (V projection)
blk.{i}.attn_output.weight                  → [hidden_dim, hidden_dim] (output projection)
blk.{i}.ffn_norm.weight                     → [hidden_dim]             (RMSNorm before FFN)
blk.{i}.ffn_gate.weight                     → [hidden_dim, ff_dim]     (SwiGLU gate)
blk.{i}.ffn_up.weight                       → [hidden_dim, ff_dim]     (SwiGLU up)
blk.{i}.ffn_down.weight                     → [ff_dim, hidden_dim]     (SwiGLU down)
output_norm.weight                          → [hidden_dim]             (final RMSNorm)
output.weight                               → [hidden_dim, vocab_size] (LM head)
```

**Vision encoder tensors (for VLMs):**

```
v.patch_embd.weight                          → [hidden_v, C, patch, patch]  (conv2d)
v.patch_embd.bias                            → [hidden_v]
v.position_embd.weight                       → [n_patches+1, hidden_v]      (learned positions)
v.class_embd                                 → [hidden_v]                    (CLS token, if used)
v.blk.{i}.attn_norm.weight                   → [hidden_v]                   (LayerNorm)
v.blk.{i}.attn_norm.bias                     → [hidden_v]
v.blk.{i}.attn_qkv.weight                   → [hidden_v, 3*hidden_v]       (fused QKV)
v.blk.{i}.attn_qkv.bias                     → [3*hidden_v]
v.blk.{i}.attn_out.weight                   → [hidden_v, hidden_v]
v.blk.{i}.attn_out.bias                     → [hidden_v]
v.blk.{i}.ffn_norm.weight                   → [hidden_v]
v.blk.{i}.ffn_norm.bias                     → [hidden_v]
v.blk.{i}.ffn_up.weight                     → [hidden_v, ff_v]
v.blk.{i}.ffn_up.bias                       → [ff_v]
v.blk.{i}.ffn_down.weight                   → [ff_v, hidden_v]
v.blk.{i}.ffn_down.bias                     → [hidden_v]
v.post_norm.weight                           → [hidden_v]
v.post_norm.bias                             → [hidden_v]
```

**Projector tensors (bridge between vision and language):**

```
mm.0.weight                                  → [hidden_v, hidden_llm]      (linear 1)
mm.0.bias                                    → [hidden_llm]
mm.2.weight                                  → [hidden_llm, hidden_llm]    (linear 2, if MLP)
mm.2.bias                                    → [hidden_llm]
```

### 2.5 Tensor Data Section — Alignment and mmap

The tensor data section starts at the **first alignment boundary** after all tensor info entries. Default alignment is
32 bytes, but can be overridden by the `general.alignment` metadata key.

**Critical for zero-copy GPU access:**

```
data_section_start = ALIGN_UP(end_of_tensor_infos, alignment)

For each tensor:
  physical_offset = data_section_start + tensor.offset
  
  // tensor.offset is already aligned to `alignment` boundary
  // So physical_offset is guaranteed aligned to at least 32 bytes
  
  // For GPU zero-copy (CL_MEM_USE_HOST_PTR), we need PAGE alignment (4096)
  // GGUF alignment of 32 is NOT enough — must handle this at buffer creation
```

**mmap strategy:**

```c
// mmap entire file
void* file_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// For each tensor that needs GPU access:
void* tensor_data = (char*)file_data + physical_offset;

// Check alignment
if ((uintptr_t)tensor_data % 4096 != 0) {
    // NOT page-aligned — must copy to aligned buffer for zero-copy GPU
    void* aligned;
    posix_memalign(&aligned, 4096, tensor_byte_size_rounded_64);
    memcpy(aligned, tensor_data, tensor_byte_size);
    // Use aligned for CL_MEM_USE_HOST_PTR
} else {
    // Already page-aligned — true zero-copy possible
    // Use tensor_data directly for CL_MEM_USE_HOST_PTR
}
```

### 2.6 How VLM GGUF Files Differ from LLM-only GGUF

VLM GGUF files can be structured in **two ways**:

**Method A — Single file (e.g. SmolVLM, some LLaVA conversions):**
All tensors (vision + projector + LLM) in one `.gguf`. Tensor names use `v.` prefix for vision, `mm.` for projector, no
prefix for LLM.

**Method B — Split files (e.g. llava-cli style):**

- `model.gguf` — LLM weights only
- `mmproj-model.gguf` — vision encoder + projector weights only

Both are loaded into the same tensor registry. Our loader must handle both cases.

**Key difference for memory planning:**

| Component        | Typical Size (7B VLM) | GPU Needed?         | Quantization   |
|------------------|-----------------------|---------------------|----------------|
| Vision encoder   | 300-600 MB            | YES (compute-bound) | FP16 preferred |
| Projector        | 10-50 MB              | YES                 | FP16           |
| LLM weights      | 2-4 GB (Q4)           | YES                 | Q4_K_M or Q8_0 |
| LLM KV cache     | 0.5-2 GB              | YES                 | INT8           |
| Embeddings table | 50-200 MB             | Partial             | FP16 or Q8     |

---

## 3. GGML Tensor System — How It Works Internally

### 3.1 The ggml_tensor Struct

Every tensor in GGML is represented by this struct (simplified):

```c
struct ggml_tensor {
    enum ggml_type type;        // data type (F32, F16, Q4_0, Q4_K, Q8_0, etc.)
    
    int     n_dims;             // number of dimensions (1-4)
    int64_t ne[GGML_MAX_DIMS];  // number of elements per dimension [cols, rows, ...]
    size_t  nb[GGML_MAX_DIMS];  // stride in bytes per dimension
    
    enum ggml_op op;            // operation that produced this tensor (NONE for inputs)
    struct ggml_tensor * src[GGML_MAX_SRC]; // source tensors (inputs to this op)
    
    void * data;                // pointer to actual data (may be mmap'd, GPU buffer, etc.)
    
    char name[GGML_MAX_NAME];   // tensor name (matches GGUF tensor name)
    
    // ... extra fields for backend, views, etc.
};
```

### 3.2 Dimension Convention

GGML uses **column-major-ish** convention (like Fortran/BLAS), which is confusing:

```
ne[0] = number of columns (innermost, contiguous in memory)
ne[1] = number of rows
ne[2] = third dimension (e.g. batch or heads)
ne[3] = fourth dimension (e.g. batch of batches)

Strides (nb):
nb[0] = sizeof(element) or sizeof(quant_block) / block_size
nb[1] = ne[0] * nb[0]  (row stride)
nb[2] = ne[1] * nb[1]  (matrix stride)
nb[3] = ne[2] * nb[2]  (tensor stride)
```

**For a weight matrix W of shape [out_features, in_features]:**

```
ne[0] = in_features   (columns, contiguous)
ne[1] = out_features  (rows)
```

This means `W[row][col]` is stored as `data + row * nb[1] + col * nb[0]`. Row-major in the mathematical sense but ne[0]
is the inner dimension.

**Why this matters for kernels:**

- GEMV `y = W * x`: iterate over `ne[1]` (rows of W) in parallel, reduce over `ne[0]` (cols) serially. Each row is
  contiguous → good cache behavior.
- For column-major GPU layouts, you'd want to transpose at load time.

### 3.3 Tensor Views (Zero-Copy Slicing)

GGML supports **views** — tensors that share data with a source tensor but with different shape/stride/offset:

```c
// Create a view of tensor a with different shape
struct ggml_tensor * view = ggml_view_2d(ctx, a, ne0, ne1, nb1, offset);
// view->data = (char*)a->data + offset
// No data is copied — just metadata
```

**Critical for our engine:**

- KV cache uses views to access per-layer, per-head slices without copying
- Attention splits Q/K/V from a fused QKV projection using views
- Vision encoder patches can be views into the preprocessed image tensor

### 3.4 ggml_context — Memory Arena

GGML uses a bump allocator context for tensor metadata (NOT tensor data):

```c
struct ggml_init_params params = {
    .mem_size   = 256 * 1024 * 1024,  // 256 MB arena for tensor metadata
    .mem_buffer = NULL,                 // let ggml allocate
    .no_alloc   = true,                // don't allocate tensor data (we manage it)
};
struct ggml_context * ctx = ggml_init(params);
```

With `no_alloc = true`, tensor structs are allocated from the arena but `data` pointers are left NULL. We assign data
pointers ourselves (to mmap'd GGUF regions, GPU buffers, etc.).

---

## 4. GGML Compute Graph — What It Does & What We Replace

### 4.1 What the Original Compute Graph Is

In stock GGML, a compute graph is a DAG (directed acyclic graph) of tensor operations:

```c
struct ggml_cgraph {
    int n_nodes;                          // number of ops
    int n_leafs;                          // number of input tensors
    struct ggml_tensor ** nodes;          // ops in topological order
    struct ggml_tensor ** leafs;          // input tensors (weights, activations)
    struct ggml_tensor ** grads;          // gradient tensors (for training, we don't need)
    
    // Hash table for quick tensor lookup
    struct ggml_hash_set visited_hash_set;
};
```

**How llama.cpp builds it:**

```
1. Create ggml_context
2. Call ggml_new_tensor() for each intermediate result
3. Chain ops: out = ggml_mul_mat(ctx, W, x)  →  out = ggml_add(ctx, out, bias)  →  ...
4. Call ggml_build_forward_expand(graph, final_output)  →  walks backward from output, adds all dependent ops
5. Call ggml_backend_graph_compute(backend, graph)  →  executes ops in topological order
```

### 4.2 What We're Replacing and Why

**We are NOT using ggml_cgraph for execution.** Here's why:

| ggml_cgraph Limitation                                          | Our Replacement                                     |
|-----------------------------------------------------------------|-----------------------------------------------------|
| Treats every op as a separate kernel dispatch                   | We fuse ops (dequant+matmul, norm+projection, etc.) |
| No phase awareness (prefill vs decode use same kernels)         | Separate kernel paths per phase                     |
| No platform-specific dispatch (same code for Adreno, Mali, CPU) | Per-vendor kernel selection at dispatch time        |
| No heterogeneous scheduling (all ops go to one backend)         | CPU+GPU pipeline with op-level routing              |
| Graph build overhead per forward pass (~0.5ms)                  | Static graph compiled once, parameters updated      |
| No memory planning across the full graph                        | Whole-graph memory planner with buffer reuse        |

### 4.3 Our Replacement: The Execution Plan

Instead of a dynamic graph, we build a **static execution plan** at model load time:

```
ExecutionPlan {
    phases: [
        Phase::VisionEncode {
            steps: [
                FusedOp::PatchEmbed { kernel: opencl_patch_embed, device: GPU },
                FusedOp::ViTBlock { kernel: opencl_vit_block_fused, device: GPU, repeat: 27 },
                FusedOp::Projection { kernel: opencl_mlp_project, device: GPU },
            ]
        },
        Phase::Prefill {
            steps: [
                FusedOp::EmbedLookup { device: CPU },  // table lookup, not worth GPU dispatch
                FusedOp::TransformerBlock {
                    norm_q_proj:  FusedKernel { kernel: opencl_rmsnorm_gemm_q8, device: GPU },
                    kv_proj:      FusedKernel { kernel: opencl_gemm_q8, device: GPU },
                    attention:    FusedKernel { kernel: opencl_flash_attn, device: GPU },
                    out_proj:     FusedKernel { kernel: opencl_gemm_q8, device: GPU },
                    ffn:          FusedKernel { kernel: opencl_swiglu_fused_q8, device: GPU },
                    repeat: block_count
                },
                FusedOp::LMHead { kernel: opencl_gemm_q8, device: GPU },
            ]
        },
        Phase::Decode {
            steps: [
                // Same structure but different kernels!
                FusedOp::TransformerBlock {
                    norm_q_proj:  FusedKernel { kernel: opencl_rmsnorm_gemv_q4, device: GPU },
                    kv_proj:      FusedKernel { kernel: opencl_gemv_q4, device: GPU },
                    attention:    FusedKernel { kernel: opencl_flash_attn_single, device: GPU },
                    out_proj:     FusedKernel { kernel: opencl_gemv_q4, device: GPU },
                    ffn:          FusedKernel { kernel: opencl_swiglu_fused_q4, device: GPU },
                    repeat: block_count
                },
                FusedOp::RMSNorm { device: CPU },  // small, not worth GPU dispatch
                FusedOp::LMHead { kernel: opencl_gemv_q4, device: GPU },
                FusedOp::Sampling { device: CPU },  // serial, always CPU
            ]
        },
    ]
}
```

### 4.4 What We KEEP from GGML

We still use GGML for:

1. **`ggml_tensor` struct** — tensor metadata, shapes, strides, views
2. **`ggml_context`** — memory arena for tensor metadata allocation
3. **Quantization utilities** — `ggml_quantize_q4_K_M()`, `ggml_dequantize_row_q4_K()`, etc.
4. **Type system** — `ggml_type` enum, `ggml_type_size()`, `ggml_blck_size()`, etc.
5. **GGUF parser** — `gguf_init_from_file()`, metadata access functions
6. **CPU NEON kernels** — for ops that stay on CPU (norm, softmax, sampling, RoPE)

We **discard/replace**:

1. **`ggml_cgraph`** — replaced by our ExecutionPlan
2. **`ggml_backend_graph_compute()`** — replaced by our scheduler
3. **`ggml_mul_mat` and similar op-building functions** — we call kernels directly
4. **The generic backend dispatch** — replaced by platform-aware dispatch

---

## 5. GGML Operations Catalog — Every Op and Its Role in VLMs

### 5.1 Matrix Operations (THE hot path — 90%+ of compute)

| GGML Op              | Math                                 | Where in VLM                                           | Our Strategy                                                                                                                              |
|----------------------|--------------------------------------|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `GGML_OP_MUL_MAT`    | `C = A × B`                          | Q/K/V projections, FFN layers, vision encoder, LM head | **Custom fused kernels**: GEMV (Q4_K) for decode, GEMM (Q8_0/INT8) for prefill. Separate kernel per quant type × batch size × GPU vendor. |
| `GGML_OP_MUL_MAT_ID` | `C = A[id] × B` (mixture of experts) | MoE models only                                        | Skip for now unless targeting Mixtral/DBRX                                                                                                |

### 5.2 Elementwise Operations

| GGML Op         | Math                      | Where in VLM                              | Our Strategy                                                                        |
|-----------------|---------------------------|-------------------------------------------|-------------------------------------------------------------------------------------|
| `GGML_OP_ADD`   | `C = A + B`               | Bias addition, residual connections       | **Fuse into preceding matmul kernel** — never dispatch separately                   |
| `GGML_OP_MUL`   | `C = A * B` (elementwise) | SwiGLU gate multiply, RMSNorm scale       | **Fuse into FFN kernel** (SwiGLU) or **norm kernel**                                |
| `GGML_OP_SILU`  | `C = A * sigmoid(A)`      | SwiGLU activation                         | **Fuse into FFN kernel**: `output = SiLU(W_gate * x) * (W_up * x)` as single kernel |
| `GGML_OP_GELU`  | `C = 0.5*A*(1+tanh(...))` | Vision encoder FFN (CLIP/SigLIP use GELU) | Fuse into vision FFN kernel                                                         |
| `GGML_OP_SCALE` | `C = A * scalar`          | Attention score scaling by `1/sqrt(d_k)`  | **Fuse into attention kernel**                                                      |
| `GGML_OP_SQR`   | `C = A²`                  | Used inside RMSNorm                       | Fuse into RMSNorm                                                                   |

### 5.3 Normalization Operations

| GGML Op                    | Math                         | Where in VLM                                              | Our Strategy                                                                                                                        |
|----------------------------|------------------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `GGML_OP_RMS_NORM`         | `x / sqrt(mean(x²) + ε)`     | Before every attention and FFN block in LLM (LLaMA-style) | **CPU with NEON** for decode (small tensor, GPU dispatch overhead not worth it). **Fuse with next matmul** for prefill if possible. |
| `GGML_OP_NORM` (LayerNorm) | `(x - mean) / sqrt(var + ε)` | Vision encoder (ViT uses LayerNorm, not RMSNorm)          | **GPU for vision** (large batch), **CPU for LLM** if used                                                                           |

### 5.4 Attention Operations

| GGML Op                                 | Math                             | Where in VLM                                           | Our Strategy                                                                                                                                    |
|-----------------------------------------|----------------------------------|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `GGML_OP_ROPE`                          | Rotary Position Embedding        | LLM attention (not in most ViTs)                       | **CPU with NEON** — it's a per-element sin/cos multiply, not memory-heavy. Precompute sin/cos table at init.                                    |
| `GGML_OP_SOFT_MAX`                      | `softmax(x)`                     | Attention scores after QK^T                            | **Fuse into Flash Attention kernel** — never compute separately. Online softmax (log-sum-exp trick) avoids materializing full attention matrix. |
| `GGML_OP_FLASH_ATTN_EXT`                | Full Flash Attention             | The entire QK^T → scale → mask → softmax → ×V pipeline | **Single fused GPU kernel** — this is THE most important optimization for long contexts. Tiles K/V in local memory, uses online softmax.        |
| `GGML_OP_CONT`                          | Make tensor contiguous           | After transpose/permute for attention reshape          | Try to **eliminate entirely** by designing tensor layouts that don't need it                                                                    |
| `GGML_OP_TRANSPOSE` / `GGML_OP_PERMUTE` | Reshape for multi-head attention | Splitting hidden_dim → [n_heads, head_dim]             | **Zero-cost view operation** — just change strides, no data movement                                                                            |

### 5.5 Data Movement Operations

| GGML Op                       | What It Does            | Where                                            | Our Strategy                                                                                              |
|-------------------------------|-------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `GGML_OP_VIEW`                | Create view with offset | KV cache access, head splitting                  | **Free** — just pointer arithmetic                                                                        |
| `GGML_OP_RESHAPE`             | Change shape metadata   | Everywhere                                       | **Free** — no data movement                                                                               |
| `GGML_OP_CPY` / `GGML_OP_DUP` | Copy tensor data        | KV cache updates (write new K/V into cache)      | **Must optimize** — KV cache writes happen every token. Use `clEnqueueCopyBuffer` or direct kernel write. |
| `GGML_OP_GET_ROWS`            | Embedding table lookup  | Token → embedding, position → position embedding | **CPU** — random access pattern, not GPU-friendly. Small enough to be fast on CPU.                        |

### 5.6 Vision-Specific Operations

| GGML Op           | What It Does                              | Where                                                       | Our Strategy                                                                                                                |
|-------------------|-------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `GGML_OP_IM2COL`  | Convert image to column format for conv2d | ViT patch embedding (the first conv2d layer)                | **GPU kernel** — or replace with direct patch extraction kernel that combines resize + normalize + im2col                   |
| `GGML_OP_CONV_2D` | 2D convolution                            | ViT patch embedding                                         | **Single fused kernel**: image → patches → embedded. Use `image2d_t` for input on OpenCL (hardware bilinear interpolation). |
| `GGML_OP_CONCAT`  | Concatenate tensors                       | Appending CLS token, concatenating vision + text embeddings | Minimize — use pre-allocated buffers with offset writes instead                                                             |

### 5.7 Summary: What Gets Custom Kernels vs What Stays GGML

```
CUSTOM GPU KERNELS (we write these):
├── Fused dequant + GEMV (Q4_K_M) — decode hot path
├── Fused dequant + GEMM (Q8_0 / INT8) — prefill hot path
├── Flash Attention (OpenCL) — with online softmax, tiled K/V
├── Fused SwiGLU FFN — gate + up + silu + mul + down in minimal dispatches
├── Fused GELU FFN — for vision encoder
├── Vision patch embedding — image2d input → embedded patches
└── KV cache update — efficient scatter write

KEEP FROM GGML (CPU NEON, already good):
├── RMSNorm / LayerNorm — small tensor, CPU fine for decode
├── RoPE — precomputed table + elementwise apply
├── Softmax — fused into flash attention for GPU, standalone for CPU fallback
├── Embedding lookup (get_rows) — random access, CPU appropriate
├── Token sampling — inherently serial
└── All quantization/dequantization utilities — for weight preparation

ELIMINATE ENTIRELY:
├── GGML_OP_CONT — design layouts to never need it
├── Standalone GGML_OP_SCALE — always fused
├── Standalone GGML_OP_ADD for bias — always fused into matmul
└── Any op that exists just because the graph builder couldn't fuse
```

---

## 6. VLM Architecture — How Vision-Language Models Work End-to-End

### 6.1 The Universal VLM Pipeline

Every VLM (LLaVA, Qwen-VL, Phi-Vision, SmolVLM, InternVL, etc.) follows this pattern:

```
INPUT: Image (RGB pixels) + Text prompt ("What is in this image?")

┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: IMAGE PREPROCESSING (CPU or GPU)                    │
│                                                              │
│   Raw pixels → Resize to model's expected size               │
│              → Normalize (mean/std per channel)              │
│              → Convert to tensor [C, H, W] or [H, W, C]     │
│                                                              │
│   Output: image_tensor [3, 384, 384] (example for SigLIP)   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ STAGE 2: VISION ENCODER (GPU — compute-bound)                │
│                                                              │
│   ViT Patch Embedding:                                       │
│     image [3, 384, 384] → conv2d(kernel=14, stride=14)      │
│     → patches [729, 1152] (27×27 patches, 1152-dim each)    │
│     + position embeddings [729, 1152] (learned)              │
│                                                              │
│   ViT Transformer Blocks (×N, e.g. 27 for SigLIP-L):       │
│     For each block:                                          │
│       LayerNorm → QKV projection → Multi-Head Attention      │
│       → Output projection → Residual                         │
│       → LayerNorm → FFN (GELU) → Residual                   │
│                                                              │
│   Output: vision_features [729, 1152]                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ STAGE 3: PROJECTION (GPU — one-shot)                         │
│                                                              │
│   Maps vision embedding space → LLM embedding space          │
│                                                              │
│   Types:                                                     │
│   • MLP (LLaVA): Linear → GELU → Linear                     │
│     [729, 1152] → [729, 4096]                                │
│   • Pixel Shuffle + MLP (SmolVLM/Idefics3): reshapes tokens  │
│     [729, 1152] → [64, 4096] (reduces token count!)          │
│   • Cross-attention resampler (Qwen-VL): learnable queries   │
│     [729, 1152] → [256, 4096]                                │
│                                                              │
│   Output: vision_embeds [N_vision_tokens, hidden_dim]        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ STAGE 4: EMBEDDING MERGE (CPU — simple concat)               │
│                                                              │
│   text_tokens = tokenizer.encode(prompt)                     │
│   text_embeds = embedding_table[text_tokens]                 │
│                                                              │
│   merged_embeds = [                                          │
│     text_embeds[:image_placeholder_pos],                     │
│     vision_embeds,           ← injected where <image> was    │
│     text_embeds[image_placeholder_pos+1:]                    │
│   ]                                                          │
│                                                              │
│   Output: merged [total_seq_len, hidden_dim]                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ STAGE 5: LLM PREFILL (GPU — compute-bound, large batch)      │
│                                                              │
│   Process ALL merged embeddings at once (batch = seq_len)    │
│                                                              │
│   For each transformer layer:                                │
│     RMSNorm → Q,K,V = matmul(hidden, W_q/W_k/W_v)          │
│     Apply RoPE to Q, K                                       │
│     Store K, V in KV cache                                   │
│     Attention = softmax(Q @ K^T / sqrt(d)) @ V              │
│     Output projection                                        │
│     Residual connection                                      │
│     RMSNorm → FFN (SwiGLU)                                  │
│     Residual connection                                      │
│                                                              │
│   LM Head: logits = hidden @ W_vocab                         │
│   Sample first output token                                  │
│                                                              │
│   Output: first_token + populated KV cache                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ STAGE 6: LLM DECODE (GPU — bandwidth-bound, batch=1)         │
│                                                              │
│   Autoregressive: one token at a time                        │
│                                                              │
│   For each transformer layer:                                │
│     Same ops as prefill but:                                 │
│     • batch = 1 → GEMV not GEMM                             │
│     • KV cache grows by 1 entry per layer per token          │
│     • Attention is against full KV cache (growing)           │
│                                                              │
│   Repeat until EOS or max_length                             │
│                                                              │
│   Output: generated text tokens                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Compute vs Bandwidth Profile Per Stage

```
Stage           Batch Size     Op Type     Bottleneck        Time (7B, 512 tok)
────────────────────────────────────────────────────────────────────────────────
Image preprocess   1           Pixel ops   Trivial           ~5 ms
Vision encoder     729         GEMM        COMPUTE           ~200-500 ms
Projection         729→64      GEMM        COMPUTE           ~10-30 ms
Embedding merge    1           Lookup      Trivial           ~1 ms
LLM prefill        512+        GEMM        COMPUTE           ~500-2000 ms
LLM decode (×N)    1           GEMV        BANDWIDTH         ~30-80 ms/token
────────────────────────────────────────────────────────────────────────────────

TOTAL FIRST TOKEN: ~800-2500 ms (dominated by prefill)
SUBSEQUENT TOKENS: ~30-80 ms each (dominated by decode GEMV)
```

### 6.3 Architecture Variations

| VLM          | Vision Encoder         | Projector            | LLM            | Key Difference                                |
|--------------|------------------------|----------------------|----------------|-----------------------------------------------|
| LLaVA 1.5    | CLIP ViT-L/14 (336px)  | 2-layer MLP          | Vicuna/LLaMA-2 | Simple, most common                           |
| LLaVA-NeXT   | CLIP ViT-L/14 (anyres) | MLP                  | LLaMA-3/Qwen2  | Dynamic resolution, multiple crops            |
| Qwen2-VL     | ViT (variable res)     | Cross-attn resampler | Qwen2          | Compresses vision tokens significantly        |
| Phi-3 Vision | SigLIP (384px)         | MLP                  | Phi-3          | Small, fast, good quality/speed ratio         |
| SmolVLM      | SigLIP-SO (384px)      | Pixel shuffle + MLP  | SmolLM2        | Extremely small (0.5B-2B), Idefics3 projector |
| InternVL2    | InternViT (448px)      | Pixel shuffle + MLP  | InternLM2      | Large ViT, high quality                       |

**Our engine must handle:**

1. Variable vision encoder architectures (ViT with different norms, activations)
2. Different projector types (MLP, cross-attention, pixel shuffle)
3. Different image token injection methods (placeholder replacement, prefix, interleaved)
4. Different LLM architectures (LLaMA, Qwen, Phi — mainly differ in FFN structure and norm placement)

---

## 7. Quantization Formats — Block Layouts for Every Q-Type

### 7.1 Why Block Quantization

GGML quantizes weights in **blocks** — groups of consecutive values that share scale/zero-point parameters. This gives
much better quality than per-tensor quantization at minimal overhead.

### 7.2 Format Specifications

**Q4_0 — Simplest 4-bit (legacy, rarely used now):**

```c
typedef struct {
    ggml_half d;       // scale (FP16) — 2 bytes
    uint8_t   qs[16];  // 32 × 4-bit weights packed into 16 bytes
} block_q4_0;          // TOTAL: 18 bytes for 32 weights = 4.5 bits/weight

// Dequantize: weight[i] = d * (qs[i/2] >> (4*(i%2)) & 0xF) - 8)
// Note: zero point is implicitly 8 (values range 0-15, centered at 8)
```

**Q4_1 — 4-bit with explicit min (slightly better quality):**

```c
typedef struct {
    ggml_half d;       // scale (FP16) — 2 bytes
    ggml_half m;       // minimum (FP16) — 2 bytes
    uint8_t   qs[16];  // 32 × 4-bit weights — 16 bytes
} block_q4_1;          // TOTAL: 20 bytes for 32 weights = 5.0 bits/weight

// Dequantize: weight[i] = d * (qs[i/2] >> (4*(i%2)) & 0xF) + m
```

**Q8_0 — 8-bit (high quality, used for prefill and activations):**

```c
typedef struct {
    ggml_half d;       // scale (FP16) — 2 bytes
    int8_t    qs[32];  // 32 × INT8 weights — 32 bytes
} block_q8_0;          // TOTAL: 34 bytes for 32 weights = 8.5 bits/weight

// Dequantize: weight[i] = d * qs[i]
// Advantage: trivial dequant, compatible with INT8 dot product hardware
```

**Q4_K_M — The KING of mobile decode (k-quant, mixed precision):**

```c
typedef struct {
    ggml_half d;          // super-block scale (FP16) — 2 bytes
    ggml_half dmin;       // super-block minimum (FP16) — 2 bytes
    uint8_t   scales[12]; // 8 × 6-bit sub-block scales, packed — 12 bytes
    uint8_t   qs[128];    // 256 × 4-bit weights — 128 bytes
} block_q4_K;             // TOTAL: 144 bytes for 256 weights = 4.5 bits/weight

// Structure: 256 weights split into 8 groups of 32
// Each group has its own 6-bit scale and 6-bit min
// Super-block d and dmin scale the per-group values
//
// Dequantize for group g, element i:
//   scale_g = (scales[...] & 0x3F)  // 6-bit unpack, complex packing scheme
//   min_g   = (scales[...] >> 4)     // interleaved with scale bits
//   weight = d * scale_g * (qs[...] & 0xF) - dmin * min_g
```

**The 6-bit scale packing in Q4_K is the trickiest part.** Here's the exact layout:

```
scales[0..3]:  lower 6 bits of scales for groups 0-3
scales[4..7]:  lower 6 bits of scales for groups 4-7
scales[8..11]: upper 2 bits of scales for groups 0-7 (packed)
               AND lower 6 bits of mins for groups 0-3

Wait — actually the packing is:
  scales[0]:  bits [0:5] = scale[0], bits [4:7] = upper bits shared
  
  It's actually implementation-dependent. Check ggml-quants.c for the 
  exact decode_q4_K function. The kernel must match EXACTLY.
```

**Q5_K_M — 5-bit k-quant (quality between Q4_K and Q8_0):**

```c
typedef struct {
    ggml_half d;          // super-block scale — 2 bytes
    ggml_half dmin;       // super-block min — 2 bytes
    uint8_t   scales[12]; // 6-bit scales packed — 12 bytes
    uint8_t   qh[32];     // high bits of 256 5-bit weights — 32 bytes
    uint8_t   qs[128];    // low 4 bits of 256 weights — 128 bytes
} block_q5_K;             // TOTAL: 176 bytes for 256 weights = 5.5 bits/weight

// Each weight has 5 bits: 4 from qs + 1 from qh
```

**Q6_K — 6-bit (high quality, rarely used on mobile due to size):**

```c
typedef struct {
    uint8_t   ql[128];   // lower 4 bits — 128 bytes
    uint8_t   qh[64];    // upper 2 bits — 64 bytes
    int8_t    scales[16]; // INT8 scales for 16 groups — 16 bytes
    ggml_half d;          // super-block scale — 2 bytes
} block_q6_K;             // TOTAL: 210 bytes for 256 weights = 6.5625 bits/weight
```

### 7.3 Quantization Strategy Table

| Weight Category                  | Decode (batch=1) | Prefill (batch>32)                                       | Why                                                                                                                    |
|----------------------------------|------------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| LLM attention weights (Q,K,V,O)  | Q4_K_M           | Q8_0 (dual stored) or Q4_K_M (dequant to INT8 in kernel) | Decode is BW-bound → smaller is faster. Prefill is compute-bound → INT8 arithmetic is 2-4× faster than dequant-to-FP16 |
| LLM FFN weights (gate, up, down) | Q4_K_M           | Q8_0 or Q4_K_M                                           | Same reasoning. FFN is 2/3 of params → biggest impact                                                                  |
| Vision encoder weights           | FP16             | FP16                                                     | Already compute-bound, quantizing hurts quality visibly                                                                |
| Projector weights                | FP16             | FP16                                                     | Tiny, runs once per image, quality matters                                                                             |
| Token embeddings                 | Q8_0 or FP16     | N/A                                                      | Lookup table, not matmul                                                                                               |
| KV cache                         | INT8             | INT8                                                     | 2× memory reduction, negligible quality loss                                                                           |

### 7.4 Dual Quantization Storage

The ideal setup stores critical weights **twice** at different quantizations:

```
Weight tensor "blk.0.attn_q.weight":
  q4_buf: cl_mem → Q4_K_M packed (for decode GEMV)
  q8_buf: cl_mem → Q8_0 packed (for prefill GEMM)
```

Memory cost: ~1.5× base Q4 model size. If too much, store only Q4_K_M and dequantize to Q8_0 on-the-fly in the prefill
kernel (still better than dequant to FP16).

---

## 8. Memory Architecture — UMA vs Discrete, The Core Divide

### 8.1 Android UMA (Unified Memory Architecture)

```
┌─────────────────────────────────────────────────────┐
│                   PHYSICAL DRAM                       │
│              (LPDDR5/5X, 4-16 GB)                    │
│                                                       │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │ CPU      │  │ GPU      │  │ Shared/Unclaimed │  │
│   │ Allocated│  │ Allocated│  │                  │  │
│   │ Pages    │  │ Pages    │  │                  │  │
│   └────┬─────┘  └────┬─────┘  └──────────────────┘  │
│        │              │                               │
│   ┌────▼──────────────▼────┐                         │
│   │  SYSTEM LEVEL CACHE    │                         │
│   │  (SLC — 4MB on QC,     │                         │
│   │   varies on MTK/Exynos)│                         │
│   │  Shared by CPU + GPU   │                         │
│   └────────────────────────┘                         │
│        │              │                               │
│   ┌────▼────┐   ┌────▼────┐                          │
│   │ CPU L2  │   │ GPU L2  │                          │
│   │ (varies)│   │ (256KB  │                          │
│   │         │   │  Adreno)│                          │
│   └────┬────┘   └────┬────┘                          │
│   ┌────▼────┐   ┌────▼────┐                          │
│   │ CPU L1  │   │ GPU L1  │ (Adreno: 2KB/uSPTP,     │
│   │ (64KB/  │   │  IMAGE  │  read-only, image only!) │
│   │  core)  │   │  ONLY   │                          │
│   └─────────┘   └─────────┘                          │
└─────────────────────────────────────────────────────┘

KEY INSIGHT: CPU and GPU access the SAME DRAM.
No PCIe bus. No DMA copy needed.
A buffer at physical address X is readable by BOTH processors.
```

**UMA Optimization Implications:**

1. **Zero-copy is real** — `CL_MEM_USE_HOST_PTR` with page-aligned pointer means the GPU reads the exact same physical
   pages the CPU wrote. No copy. This saves 3-7 GB of memory and eliminates a multi-second copy at model load.

2. **The SLC is shared** — if the CPU produces a 4096-element RMSNorm output and it fits in SLC (~4MB), the GPU can read
   it from SLC on the next kernel dispatch without a DRAM round-trip. This makes CPU→GPU handoffs nearly free for small
   tensors.

3. **Bandwidth is shared** — CPU and GPU compete for the same DRAM bandwidth (~50-80 GB/s on LPDDR5). If both are
   hammering memory simultaneously, both slow down. Schedule carefully.

4. **No "upload" step** — unlike desktop where you `clEnqueueWriteBuffer` to copy host→device across PCIe, on UMA you
   just... use the pointer. The "upload" is literally free.

### 8.2 Desktop Discrete GPU

```
┌────────────────┐              ┌────────────────┐
│   SYSTEM RAM   │◄── PCIe ───►│   GPU VRAM     │
│   (DDR4/5)     │    Bus       │   (GDDR6/HBM)  │
│                │  16-64 GB/s  │                 │
│  CPU L3        │              │  GPU L2         │
│  CPU L2        │              │  GPU L1/SMEM    │
│  CPU L1        │              │  Registers      │
└────────────────┘              └────────────────┘

COMPLETELY SEPARATE memory spaces.
Every byte the GPU needs must be copied across PCIe.
```

**Desktop Optimization Implications:**

1. **Model weights must be uploaded** — copy GGUF weights to VRAM at load time. This is a one-time cost (~2-4 seconds
   for 7B Q4).

2. **Activations transfer is expensive** — if you compute RMSNorm on CPU, you must transfer the result to GPU via PCIe
   for the next matmul. At 16 GB/s PCIe 3.0, transferring a 4096×FP16 vector (8 KB) takes ~0.5 µs — tolerable, but adds
   up with many ops.

3. **Overlap transfer with compute** — use async copy (double buffering) to hide PCIe latency. While GPU computes layer
   N, async-copy layer N+1 weights if they don't fit in VRAM.

4. **GPU VRAM bandwidth is MUCH higher** — HBM2e at 2 TB/s vs LPDDR5 at 50 GB/s. Desktop decode is 20-40× faster per
   layer purely from bandwidth. But PCIe is the chokepoint for any CPU↔GPU data sharing.

### 8.3 The Decision Matrix: What Runs Where

```
                    UMA (Android)                    Discrete (Desktop)
                    ─────────────                    ──────────────────
Weights location    mmap'd DRAM (shared)             VRAM (copied once)
Activation flow     SLC bridge (near-zero cost)      PCIe (must minimize)
RMSNorm             CPU NEON (avoid GPU dispatch     CPU AVX (same reasoning,
                    overhead for 4096 elements)       or GPU if already there)
RoPE                CPU NEON                          CPU AVX or GPU
Softmax             Fused in FlashAttn (GPU)         Fused in FlashAttn (GPU)
GEMV (decode)       GPU OpenCL                       GPU OpenCL/CUDA
GEMM (prefill)      GPU OpenCL                       GPU OpenCL/CUDA
Sampling            CPU                              CPU
KV cache            UMA DRAM (shared access)         VRAM
```

---

## 9. Heterogeneous Compute Architecture — Our Design

### 9.1 Module Structure

```
engine/
├── include/
│   ├── engine/
│   │   ├── types.h              ← Core type definitions (TensorDesc, QuantType, etc.)
│   │   ├── gguf_loader.h        ← GGUF file parsing and tensor registry
│   │   ├── tensor_manager.h     ← Virtual tensor management, memory pools
│   │   ├── execution_plan.h     ← Static execution plan (replaces ggml_cgraph)
│   │   ├── scheduler.h          ← CPU/GPU dispatch and pipelining
│   │   ├── platform.h           ← Platform detection and capability query
│   │   │
│   │   ├── backend/
│   │   │   ├── backend.h        ← Abstract backend interface
│   │   │   ├── cpu_backend.h    ← CPU backend (NEON/AVX)
│   │   │   └── opencl_backend.h ← OpenCL backend (Adreno/Mali/desktop)
│   │   │
│   │   ├── ops/
│   │   │   ├── matmul.h         ← Matrix multiply dispatch (GEMV/GEMM)
│   │   │   ├── attention.h      ← Flash attention + KV cache ops
│   │   │   ├── normalization.h  ← RMSNorm, LayerNorm
│   │   │   ├── activation.h     ← SiLU, GELU, etc.
│   │   │   ├── rope.h           ← Rotary position embedding
│   │   │   └── vision_ops.h     ← Patch embed, image preprocess
│   │   │
│   │   ├── arch/
│   │   │   ├── transformer.h    ← Generic transformer block
│   │   │   ├── llm_decoder.h    ← LLM decoder (LLaMA/Qwen/Phi variants)
│   │   │   ├── vit_encoder.h    ← Vision transformer encoder
│   │   │   └── projector.h      ← Vision→language projectors (MLP, cross-attn, etc.)
│   │   │
│   │   ├── kv_cache.h           ← KV cache management (paged, INT8)
│   │   ├── tokenizer.h          ← BPE tokenizer
│   │   └── sampler.h            ← Token sampling
│   │
│   └── vlm.h                    ← Top-level VLM inference API
│
├── src/
│   ├── ... (implementations)
│   │
│   └── kernels/
│       ├── opencl/
│       │   ├── common.cl         ← Shared macros, dequant functions
│       │   ├── gemv_q4k.cl       ← Q4_K GEMV for decode
│       │   ├── gemm_q8.cl        ← Q8_0 GEMM for prefill
│       │   ├── gemm_fp16.cl      ← FP16 GEMM for vision encoder
│       │   ├── flash_attn.cl     ← Flash attention
│       │   ├── swiglu_fused.cl   ← Fused SwiGLU FFN
│       │   ├── gelu_fused.cl     ← Fused GELU FFN (vision)
│       │   ├── patch_embed.cl    ← Vision patch embedding
│       │   └── kv_cache_ops.cl   ← KV cache read/write
│       │
│       └── neon/
│           ├── rmsnorm.h         ← NEON RMSNorm inline functions
│           ├── layernorm.h       ← NEON LayerNorm
│           ├── rope.h            ← NEON RoPE apply
│           ├── softmax.h         ← NEON softmax (fallback)
│           └── quantize.h        ← NEON dequant utilities
│
├── platforms/
│   ├── android/
│   │   ├── AndroidManifest.xml
│   │   ├── jni/                  ← JNI bridge for Android app
│   │   └── opencl_loader.c       ← Dynamic OpenCL library loading (libOpenCL.so)
│   │
│   └── linux/
│       └── opencl_loader.c       ← Linux OpenCL ICD loading
│
└── tests/
    ├── test_gguf_loader.cpp
    ├── test_gemv_q4k.cpp
    ├── test_flash_attn.cpp
    └── bench_full_model.cpp
```

### 9.2 The Backend Interface

```c
// backend.h — every backend implements this

typedef enum {
    DEVICE_CPU,
    DEVICE_GPU_ADRENO,
    DEVICE_GPU_MALI,
    DEVICE_GPU_DESKTOP,  // Generic OpenCL (AMD, NVIDIA via OpenCL, Intel)
} DeviceType;

typedef struct Backend {
    DeviceType type;
    const char* name;

    // Lifecycle
    int  (*init)(struct Backend* self);
    void (*destroy)(struct Backend* self);

    // Buffer management
    BufferHandle (*buffer_create)(struct Backend* self, size_t size, BufferUsage usage);
    BufferHandle (*buffer_create_from_host)(struct Backend* self, void* host_ptr, size_t size);  // zero-copy on UMA
    void         (*buffer_destroy)(struct Backend* self, BufferHandle buf);
    void*        (*buffer_map)(struct Backend* self, BufferHandle buf, size_t offset, size_t size);
    void         (*buffer_unmap)(struct Backend* self, BufferHandle buf, void* mapped);

    // Kernel dispatch
    void (*matmul)(struct Backend* self, MatmulParams* params);    // dispatches GEMV or GEMM based on params->batch_size
    void (*attention)(struct Backend* self, AttentionParams* params);
    void (*normalization)(struct Backend* self, NormParams* params);
    void (*activation)(struct Backend* self, ActivationParams* params);
    void (*rope)(struct Backend* self, RoPEParams* params);
    void (*vision_patch_embed)(struct Backend* self, PatchEmbedParams* params);

    // Synchronization
    EventHandle (*get_last_event)(struct Backend* self);
    void        (*wait_event)(struct Backend* self, EventHandle event);
    void        (*sync)(struct Backend* self);  // full barrier (avoid!)

    // Platform info
    size_t (*get_max_alloc_size)(struct Backend* self);
    size_t (*get_local_mem_size)(struct Backend* self);
    int    (*get_compute_units)(struct Backend* self);
    bool   (*supports_fp16)(struct Backend* self);
    bool   (*supports_int8_dot)(struct Backend* self);
} Backend;
```

### 9.3 The Scheduler — CPU+GPU Pipeline

```c
// scheduler.h

typedef struct {
    Backend* cpu_backend;
    Backend* gpu_backend;
    DeviceType gpu_type;  // affects kernel selection within gpu_backend

    // Pre-computed device assignment per op
    DeviceType* op_device_map;  // op_device_map[op_index] = DEVICE_CPU or DEVICE_GPU_*
} Scheduler;

// The scheduling rules (hardcoded at plan build time):
//
// GPU always:
//   - All GEMV/GEMM (matmul) ops
//   - Flash Attention
//   - Vision encoder (all of it)
//   - Fused FFN blocks
//
// CPU always:
//   - RMSNorm/LayerNorm during decode (batch=1, tiny tensor)
//   - RoPE application
//   - Token embedding lookup
//   - Token sampling (argmax, top-k/p)
//   - KV cache pointer management (not the actual read/write)
//
// Phase-dependent:
//   - RMSNorm during prefill: GPU (large batch makes it worth dispatching)
//   - Attention: GPU always, but kernel selection changes (flash vs simple)
//
// Pipeline overlap:
//   - While GPU runs GEMV for layer N:
//     - CPU computes RMSNorm for layer N's output
//     - CPU prefetches layer N+1 weights via madvise (if not in cache)
//   - Use OpenCL events (not clFinish!) for fine-grained sync
```

### 9.4 Event-Based Pipelining (No Full Barriers)

```c
void decode_one_token(Scheduler* sched, Model* model, int token_id) {
    cl_event prev_event = NULL;

    // Embedding lookup (CPU)
    Tensor* hidden = cpu_embed_lookup(model->embed_table, token_id);

    for (int layer = 0; layer < model->n_layers; layer++) {
        // RMSNorm (CPU) — can start immediately, no GPU dependency
        Tensor* normed = cpu_rmsnorm(hidden, model->layers[layer].attn_norm_weight);

        // Q,K,V projections (GPU) — depends on normed being ready (it is, CPU just did it)
        cl_event qkv_event;
        gpu_fused_qkv_gemv(sched->gpu_backend,
            normed, model->layers[layer], &qkv_event);

        // RoPE (CPU) — needs QKV result, so wait for GPU
        // BUT: we can overlap with KV cache management
        wait_event(qkv_event);
        cpu_apply_rope(Q, K, model->rope_freqs, token_pos);

        // Update KV cache (GPU or CPU depending on cache location)
        gpu_kv_cache_append(sched->gpu_backend, K, V, layer, token_pos, &kv_event);

        // Flash Attention (GPU)
        cl_event attn_event;
        gpu_flash_attention(sched->gpu_backend,
            Q, model->kv_cache, layer, token_pos, &attn_event);

        // Output projection (GPU)
        cl_event oproj_event;
        gpu_gemv(sched->gpu_backend,
            attn_output, model->layers[layer].attn_output_weight, &oproj_event);

        // Residual add (CPU, or fused into output projection kernel)
        wait_event(oproj_event);
        cpu_residual_add(hidden, attn_out_projected);

        // FFN block (similar pattern: CPU norm → GPU fused SwiGLU → CPU residual)
        Tensor* ffn_normed = cpu_rmsnorm(hidden, model->layers[layer].ffn_norm_weight);
        cl_event ffn_event;
        gpu_fused_swiglu(sched->gpu_backend,
            ffn_normed, model->layers[layer], &ffn_event);
        wait_event(ffn_event);
        cpu_residual_add(hidden, ffn_output);
    }

    // Final norm + LM head
    Tensor* final_normed = cpu_rmsnorm(hidden, model->output_norm_weight);
    gpu_gemv(sched->gpu_backend, final_normed, model->output_weight, &lm_event);
    wait_event(lm_event);

    // Sampling (CPU)
    int next_token = cpu_sample(logits, sampler_params);
}
```

---

## 10. GPU Backend — OpenCL Kernel Strategy

### 10.1 Kernel Selection Matrix

```
┌─────────────────────┬─────────────────────────┬─────────────────────────┐
│ Operation            │ Adreno (6xx/7xx)        │ Mali (Valhall/Immortalis│
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ GEMV Q4_K (decode)  │ image1d_buffer_t weight  │ __global float4 weight  │
│                     │ wave64 workgroup [128,1] │ workgroup [8,4] = 32    │
│                     │ SplitK for K>2048        │ No shared mem tiling    │
│                     │ L1 cache via image reads │ Explicit vectorization  │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ GEMM Q8 (prefill)   │ Tiled [64,16,16]         │ Tiled [32,8,8]          │
│                     │ INT8 dot (cl_khr_integer │ FP16 arithmetic         │
│                     │ _dot_product)            │ Register blocking       │
│                     │ Local mem for tiles      │ Minimal local mem use   │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ Flash Attention     │ Tile K/V in local mem    │ Loop in registers       │
│                     │ (32KB available)         │ (local mem not faster)  │
│                     │ Online softmax FP32 acc  │ Online softmax FP32 acc │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ SwiGLU FFN (decode) │ 3 GEMV fused: gate+up    │ Same but smaller WG     │
│                     │ → SiLU → mul → down      │ and explicit float4     │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ Vision GEMM FP16    │ Standard tiled GEMM      │ Standard tiled GEMM     │
│                     │ half4 arithmetic         │ half4 arithmetic        │
└─────────────────────┴─────────────────────────┴─────────────────────────┘
```

### 10.2 Zero-Copy Buffer Creation on UMA

```c
BufferHandle opencl_buffer_create_from_host(Backend* self, void* host_ptr, size_t size) {
    OpenCLBackend* ocl = (OpenCLBackend*)self;

    // Check alignment requirements
    assert((uintptr_t)host_ptr % 4096 == 0);  // page-aligned
    assert(size % 64 == 0);                     // 64-byte multiple

    cl_int err;
    cl_mem buf;

    if (ocl->supports_svm && ocl->uma_mode) {
        // Best path: SVM (Adreno 6xx+, Mali G77+)
        // No cl_mem wrapper needed — pass pointer directly to kernels
        buf = NULL;  // Use SVM path in kernel dispatch
        // ... SVM-specific handling
    } else {
        // Good path: USE_HOST_PTR (universal UMA zero-copy)
        buf = clCreateBuffer(
            ocl->context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            size,
            host_ptr,
            &err
        );
    }

    // For Adreno: also create image1d_buffer_t alias for L1 caching
    cl_mem img = NULL;
    if (ocl->device_type == DEVICE_GPU_ADRENO) {
        cl_image_format fmt = { CL_RGBA, CL_HALF_FLOAT };
        cl_image_desc desc = {
            .image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER,
            .image_width = size / 8,  // 8 bytes per RGBA half4 pixel
            .buffer = buf
        };
        img = clCreateImage(ocl->context, CL_MEM_READ_ONLY, &fmt, &desc, NULL, &err);
    }

    BufferHandle handle = { .cl_buf = buf, .cl_img = img, .host_ptr = host_ptr, .size = size };
    return handle;
}
```

### 10.3 Adreno-Specific: Image1D for L1 Cache

On Adreno, the L1 texture cache (2KB per uSPTP) ONLY caches reads through `image` objects. Regular `__global` buffer
reads bypass L1 entirely. This is the single biggest micro-optimization for decode GEMV:

```c
// WITHOUT image objects (L1 miss on every read):
__kernel void gemv_bad(__global const half* weights, ...) {
    half4 w = vload4(idx, weights);  // Goes to L2/DRAM every time
}

// WITH image objects (L1 cache hit on reuse):
__kernel void gemv_good(__read_only image1d_buffer_t weights, ...) {
    half4 w = read_imageh(weights, sampler, idx);  // L1 cached!
}
```

For decode GEMV where `x` (the activation vector) is reused by every output row, loading `x` through an image object
means each element of `x` is fetched from DRAM once and then served from L1 for subsequent rows. With 4096 output rows
reading the same 4096 input elements, this is massive.

### 10.4 Runtime Kernel Compilation & Caching

```c
typedef struct {
    cl_program program;
    cl_kernel  kernel;
    uint64_t   driver_hash;  // hash of (vendor + driver_version + device_name)
} CachedKernel;

cl_kernel get_or_compile_kernel(OpenCLBackend* ocl, const char* source,
                                 const char* kernel_name, const char* build_opts) {
    // 1. Compute cache key
    uint64_t key = hash(kernel_name, build_opts, ocl->driver_hash);

    // 2. Check binary cache on disk
    char cache_path[256];
    snprintf(cache_path, sizeof(cache_path), "%s/kernel_%lx.bin", ocl->cache_dir, key);

    if (file_exists(cache_path)) {
        // Load pre-compiled binary
        size_t bin_size;
        unsigned char* binary = read_file(cache_path, &bin_size);
        cl_int status;
        cl_program program = clCreateProgramWithBinary(
            ocl->context, 1, &ocl->device, &bin_size,
            (const unsigned char**)&binary, &status, &err
        );
        if (status == CL_SUCCESS) {
            clBuildProgram(program, 1, &ocl->device, build_opts, NULL, NULL);
            return clCreateKernel(program, kernel_name, &err);
        }
        // Binary invalid (driver update?) — fall through to recompile
    }

    // 3. Compile from source
    cl_program program = clCreateProgramWithSource(ocl->context, 1, &source, NULL, &err);

    // Platform-specific build options
    char full_opts[1024];
    if (ocl->device_type == DEVICE_GPU_ADRENO) {
        snprintf(full_opts, sizeof(full_opts),
            "%s -DADRENO -DWAVE_SIZE=64 -DUSE_IMAGE_WEIGHTS", build_opts);
    } else if (ocl->device_type == DEVICE_GPU_MALI) {
        snprintf(full_opts, sizeof(full_opts),
            "%s -DMALI -DWAVE_SIZE=1 -DUSE_FLOAT4_VEC", build_opts);
    }

    clBuildProgram(program, 1, &ocl->device, full_opts, NULL, NULL);

    // 4. Save binary to cache
    size_t bin_size;
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
    unsigned char* binary = malloc(bin_size);
    clGetProgramInfo(program, CL_PROGRAM_BINARIES, bin_size, &binary, NULL);
    write_file(cache_path, binary, bin_size);

    return clCreateKernel(program, kernel_name, &err);
}
```

---

## 11. CPU Backend — ARM NEON Kernel Strategy

### 11.1 Ops That Stay on CPU

During **decode** (batch=1), these ops are too small to justify GPU dispatch overhead (~50-200 µs per kernel launch on
Android OpenCL):

| Op                   | Tensor Size   | Time on CPU NEON | GPU Dispatch Overhead | Verdict                                |
|----------------------|---------------|------------------|-----------------------|----------------------------------------|
| RMSNorm              | 4096 elements | ~2 µs            | ~100 µs               | CPU wins 50×                           |
| RoPE                 | 4096 elements | ~3 µs            | ~100 µs               | CPU wins 33×                           |
| Residual add         | 4096 elements | ~1 µs            | ~100 µs               | CPU wins 100×                          |
| Softmax (standalone) | head_dim      | ~1 µs            | ~100 µs               | CPU (but fused into flash attn on GPU) |
| Token sample         | vocab_size    | ~10 µs           | ~100 µs               | CPU wins 10×                           |

During **prefill** (batch=seq_len), norms are larger but still typically CPU-favorable unless batch > ~64.

### 11.2 RMSNorm NEON Implementation

```c
// Optimized for ARM NEON — processes 8 FP16 values per iteration
static inline void rmsnorm_fp16_neon(
    const _Float16* __restrict__ x,
    const _Float16* __restrict__ weight,
    _Float16* __restrict__ out,
    int n,
    float eps
) {
    // Pass 1: compute sum of squares
    float32x4_t sum_sq0 = vdupq_n_f32(0.0f);
    float32x4_t sum_sq1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < n; i += 8) {
        float16x8_t xv = vld1q_f16(x + i);
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(xv));
        float32x4_t hi = vcvt_f32_f16(vget_high_f16(xv));
        sum_sq0 = vfmaq_f32(sum_sq0, lo, lo);  // fused multiply-accumulate
        sum_sq1 = vfmaq_f32(sum_sq1, hi, hi);
    }

    float sum = vaddvq_f32(vaddq_f32(sum_sq0, sum_sq1));
    float scale = 1.0f / sqrtf(sum / n + eps);
    float32x4_t vscale = vdupq_n_f32(scale);

    // Pass 2: normalize and apply weight
    for (int i = 0; i < n; i += 8) {
        float16x8_t xv = vld1q_f16(x + i);
        float16x8_t wv = vld1q_f16(weight + i);

        float32x4_t lo = vmulq_f32(vcvt_f32_f16(vget_low_f16(xv)), vscale);
        float32x4_t hi = vmulq_f32(vcvt_f32_f16(vget_high_f16(xv)), vscale);

        float32x4_t wlo = vcvt_f32_f16(vget_low_f16(wv));
        float32x4_t whi = vcvt_f32_f16(vget_high_f16(wv));

        lo = vmulq_f32(lo, wlo);
        hi = vmulq_f32(hi, whi);

        float16x8_t result = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
        vst1q_f16(out + i, result);
    }
}
```

### 11.3 RoPE NEON Implementation

```c
// Precomputed at model init:
// cos_table[pos][d] = cos(pos * freq[d])
// sin_table[pos][d] = sin(pos * freq[d])
// where freq[d] = 1.0 / pow(theta, 2*d / head_dim)

static inline void apply_rope_neon(
    _Float16* __restrict__ q,   // [head_dim]
    _Float16* __restrict__ k,   // [head_dim]
    const float* __restrict__ cos_table,  // [head_dim/2]
    const float* __restrict__ sin_table,  // [head_dim/2]
    int head_dim
) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i += 4) {
        float32x4_t cos_v = vld1q_f32(cos_table + i);
        float32x4_t sin_v = vld1q_f32(sin_table + i);

        // Q: rotate pairs (q[i], q[i+half]) → (q[i]*cos - q[i+half]*sin, q[i]*sin + q[i+half]*cos)
        float32x4_t q0 = vcvt_f32_f16(vld1_f16(q + i));
        float32x4_t q1 = vcvt_f32_f16(vld1_f16(q + i + half));

        float32x4_t rq0 = vfmsq_f32(vmulq_f32(q0, cos_v), q1, sin_v);  // q0*cos - q1*sin
        float32x4_t rq1 = vfmaq_f32(vmulq_f32(q1, cos_v), q0, sin_v);  // q1*cos + q0*sin

        vst1_f16(q + i,        vcvt_f16_f32(rq0));
        vst1_f16(q + i + half, vcvt_f16_f32(rq1));

        // Same for K
        float32x4_t k0 = vcvt_f32_f16(vld1_f16(k + i));
        float32x4_t k1 = vcvt_f32_f16(vld1_f16(k + i + half));

        float32x4_t rk0 = vfmsq_f32(vmulq_f32(k0, cos_v), k1, sin_v);
        float32x4_t rk1 = vfmaq_f32(vmulq_f32(k1, cos_v), k0, sin_v);

        vst1_f16(k + i,        vcvt_f16_f32(rk0));
        vst1_f16(k + i + half, vcvt_f16_f32(rk1));
    }
}
```

---

## 12. Operator Fusion — What to Fuse and Why

### 12.1 Fusion Priority List

Every standalone kernel dispatch on Android OpenCL costs ~50-200 µs of overhead (command queue enqueue, driver
scheduling, workgroup dispatch). For a 32-layer model generating tokens at 30 tok/s, that's 33ms budget per token. If
each layer has 10 separate kernel dispatches, that's 320 dispatches × 100 µs = 32 ms of pure overhead — nearly the
entire budget!

**Tier 1 — MUST FUSE (eliminates DRAM round-trips):**

| Fusion                                               | What It Replaces                  | DRAM Saves                                                                     |
|------------------------------------------------------|-----------------------------------|--------------------------------------------------------------------------------|
| Dequant + GEMV/GEMM                                  | Separate dequant kernel + matmul  | Eliminates writing dequantized weights to DRAM (saves `N*K*2` bytes per layer) |
| QK^T + scale + mask + softmax + ×V (Flash Attention) | 5+ separate kernels               | Eliminates materializing `[seq_len, seq_len]` attention matrix                 |
| SiLU(gate) × up (SwiGLU activation)                  | Separate SiLU kernel + mul kernel | Eliminates writing intermediate activation twice                               |

**Tier 2 — SHOULD FUSE (eliminates kernel dispatch overhead):**

| Fusion                                           | Dispatches Saved Per Layer        |
|--------------------------------------------------|-----------------------------------|
| RMSNorm + Q projection (decode only)             | 1 GPU dispatch → 0 (stays on CPU) |
| gate + up as single GEMV (both share same input) | 2 dispatches → 1                  |
| Output projection + residual add                 | 2 dispatches → 1                  |
| FFN down projection + residual add               | 2 dispatches → 1                  |

**Tier 3 — NICE TO HAVE:**

| Fusion                            | Benefit                                                                                          |
|-----------------------------------|--------------------------------------------------------------------------------------------------|
| All 3 QKV projections as one GEMM | 3 → 1 dispatch (prefill only, since decode GEMV can't easily fuse different output dims for GQA) |
| Post-norm + next-layer pre-norm   | Eliminates one norm pass (only if architecture allows)                                           |

### 12.2 Fused Kernel: SwiGLU FFN (Decode Path)

The FFN in LLaMA-style models is:

```
output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

This involves 3 matmuls + 1 activation + 1 elementwise multiply. Naive implementation = 5 kernel dispatches. Our fused
version:

```
Dispatch 1: gate_out, up_out = dual_gemv(x, W_gate, W_up)
            // Single kernel reads x once, computes two GEMV outputs
Dispatch 2: intermediate = silu(gate_out) * up_out
            // Fused activation + multiply, no DRAM write for gate_out
Dispatch 3: output = gemv(intermediate, W_down) + residual
            // Fused GEMV + residual add

Total: 3 dispatches instead of 5 (or 2 if we fuse dispatches 1+2)
```

---

## 13. KV Cache Architecture

### 13.1 Memory Layout

```c
typedef struct {
    // Per-layer, per-type (K or V) cache
    // Shape: [max_seq_len, n_kv_heads, head_dim]
    // Stored as INT8 with per-head per-token scale
    struct {
        int8_t*  data;     // [max_seq_len * n_kv_heads * head_dim]
        float*   scales;   // [max_seq_len * n_kv_heads]
        // On GPU: cl_mem wrappers for both
        BufferHandle gpu_data;
        BufferHandle gpu_scales;
    } k_cache[MAX_LAYERS], v_cache[MAX_LAYERS];

    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    int current_len;     // how many tokens stored so far
} KVCache;
```

**Memory calculation (7B model, 4096 max context):**

```
INT8 KV cache:
  2 (K+V) × 32 (layers) × 4096 (tokens) × 8 (GQA heads) × 128 (head_dim) × 1 (INT8)
  = 256 MB

  + scales: 2 × 32 × 4096 × 8 × 4 (float) = 8 MB

  TOTAL: ~264 MB (vs 2 GB for FP16 KV cache — 7.6× reduction!)
```

### 13.2 Paged KV Cache

Instead of pre-allocating max_seq_len upfront:

```c
#define KV_PAGE_SIZE 16  // tokens per page

typedef struct {
    int8_t  data[KV_PAGE_SIZE][N_KV_HEADS_MAX][HEAD_DIM_MAX];
    float   scales[KV_PAGE_SIZE][N_KV_HEADS_MAX];
    bool    in_use;
    int     layer;    // which layer this page belongs to
    int     kv_type;  // 0 = K, 1 = V
    int     start_token;  // first token index in this page
} KVPage;

typedef struct {
    KVPage* pages;          // pool of pre-allocated pages
    int     n_pages;
    int*    page_table;     // page_table[token_idx / PAGE_SIZE] = page_index
    int     pages_per_layer; // max pages per layer
} PagedKVCache;
```

Benefits on UMA:

- Pages can be lazily allocated (Android's mmap with `MAP_ANONYMOUS`)
- Cold pages naturally get swapped by OS if memory pressure exists
- No fragmentation from variable-length sequences
- Easy to implement sliding window attention (just don't allocate pages beyond window)

### 13.3 KV Cache Quantization During Write

```c
// When adding new K/V entries to cache, quantize on-the-fly:

void kv_cache_append_quantized(
    KVCache* cache, int layer,
    const _Float16* k_fp16,  // [n_kv_heads, head_dim] — current token's K
    const _Float16* v_fp16,  // [n_kv_heads, head_dim] — current token's V
    int token_pos
) {
    for (int h = 0; h < cache->n_kv_heads; h++) {
        const _Float16* k_head = k_fp16 + h * cache->head_dim;
        const _Float16* v_head = v_fp16 + h * cache->head_dim;

        // Find absmax for INT8 scale
        float k_max = 0.0f, v_max = 0.0f;
        for (int d = 0; d < cache->head_dim; d++) {
            float ka = fabsf((float)k_head[d]);
            float va = fabsf((float)v_head[d]);
            if (ka > k_max) k_max = ka;
            if (va > v_max) v_max = va;
        }

        float k_scale = k_max / 127.0f;
        float v_scale = v_max / 127.0f;

        // Quantize and store
        int8_t* k_dst = cache->k_cache[layer].data +
                        (token_pos * cache->n_kv_heads + h) * cache->head_dim;
        int8_t* v_dst = cache->v_cache[layer].data +
                        (token_pos * cache->n_kv_heads + h) * cache->head_dim;

        float k_inv = (k_scale > 0) ? 127.0f / k_max : 0.0f;
        float v_inv = (v_scale > 0) ? 127.0f / v_max : 0.0f;

        for (int d = 0; d < cache->head_dim; d++) {
            k_dst[d] = (int8_t)roundf((float)k_head[d] * k_inv);
            v_dst[d] = (int8_t)roundf((float)v_head[d] * v_inv);
        }

        cache->k_cache[layer].scales[token_pos * cache->n_kv_heads + h] = k_scale;
        cache->v_cache[layer].scales[token_pos * cache->n_kv_heads + h] = v_scale;
    }
}
```

---

## 14. Vision Encoder Specifics

### 14.1 Key Differences from LLM Layers

| Aspect            | Vision Encoder (ViT)           | LLM Decoder                                  |
|-------------------|--------------------------------|----------------------------------------------|
| Normalization     | LayerNorm (has bias)           | RMSNorm (no bias)                            |
| Activation        | GELU (CLIP) or SiLU (SigLIP)   | SiLU (SwiGLU)                                |
| Attention         | Bidirectional (no causal mask) | Causal masked                                |
| Position encoding | Learned absolute               | RoPE (relative)                              |
| QKV format        | Usually fused [3*hidden]       | Usually separate Q, K, V                     |
| Bias              | Has bias in all projections    | Typically no bias (LLaMA-style)              |
| Batch size        | Fixed (n_patches, e.g. 729)    | Variable (1 for decode, seq_len for prefill) |
| Quantization      | FP16 (don't quantize)          | Q4_K_M / Q8_0                                |

### 14.2 Image Preprocessing Pipeline

```
Raw image (JPEG/PNG from camera or file)
  │
  ├── Decode (libjpeg/libpng or Android ImageDecoder API)
  │   → RGB uint8 [H_orig, W_orig, 3]
  │
  ├── Resize (bilinear interpolation)
  │   → RGB uint8 [H_target, W_target, 3]
  │   (e.g. 384×384 for SigLIP, 336×336 for CLIP)
  │
  ├── Convert to float + Normalize
  │   → FP16 [3, H, W] (CHW format for conv2d)
  │   pixel = (pixel / 255.0 - mean[c]) / std[c]
  │   mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (ImageNet)
  │
  └── Output: normalized image tensor ready for patch embedding
```

**On GPU (preferred for zero-copy camera input):**

```c
// Single kernel: RGBA uint8 → CHW FP16 normalized
// Use image2d_t input for hardware bilinear resize
__kernel void preprocess_image(
    __read_only image2d_t input_image,  // hardware resizing via sampler
    __global half* output,               // [3, H_out, W_out]
    float3 mean, float3 inv_std,
    int H_out, int W_out, int H_in, int W_in
) {
    int x = get_global_id(0);  // output column
    int y = get_global_id(1);  // output row

    // Compute source coordinate (bilinear)
    float src_x = (x + 0.5f) * (float)W_in / W_out - 0.5f;
    float src_y = (y + 0.5f) * (float)H_in / H_out - 0.5f;

    float4 pixel = read_imagef(input_image, sampler_bilinear, (float2)(src_x, src_y));

    // Normalize and write CHW
    int offset = y * W_out + x;
    output[0 * H_out * W_out + offset] = (half)((pixel.x - mean.x) * inv_std.x);
    output[1 * H_out * W_out + offset] = (half)((pixel.y - mean.y) * inv_std.y);
    output[2 * H_out * W_out + offset] = (half)((pixel.z - mean.z) * inv_std.z);
}
```

### 14.3 Patch Embedding (Conv2D with stride = patch_size)

This is just a single conv2d with `kernel_size = patch_size, stride = patch_size`:

```
Input:  [3, 384, 384]
Conv2d: weight [1152, 3, 14, 14], bias [1152]
Output: [1152, 27, 27] → reshape to [729, 1152]
        (729 patches, each 1152-dimensional)
```

Implementation: standard im2col + GEMM, or direct convolution kernel. Since this runs once per image, optimization
priority is low.

### 14.4 Vision-Language Projector Types

**MLP Projector (LLaVA-style):**

```
vision_features [N_patches, vision_dim]
  → Linear(vision_dim, llm_dim)  → GELU  → Linear(llm_dim, llm_dim)
  → projected [N_patches, llm_dim]

Token count preserved: 729 patches → 729 LLM tokens
```

**Pixel Shuffle + MLP (Idefics3/SmolVLM):**

```
vision_features [N_patches, vision_dim]
  → Reshape to [H_patches, W_patches, vision_dim]
  → Pixel unshuffle (merge 3×3 blocks): [H/3, W/3, vision_dim × 9]
  → Linear(vision_dim × 9, llm_dim)
  → projected [N_patches/9, llm_dim]

Token count REDUCED: 729 patches → 81 LLM tokens (9× compression!)
This is critical for mobile — fewer tokens = smaller KV cache = faster decode
```

**Cross-Attention Resampler (Qwen-VL):**

```
vision_features [N_patches, vision_dim]  (as keys/values)
learned_queries [N_queries, llm_dim]     (as queries, N_queries << N_patches)
  → Cross-attention: queries attend to vision features
  → projected [N_queries, llm_dim]

Token count set by N_queries: e.g. 256 tokens regardless of image resolution
```

---

## 15. Tensor Management & Memory Pooling

### 15.1 Virtual Tensor System

```c
typedef struct {
    char          name[128];       // GGUF tensor name
    ggml_type     quant_type;      // Q4_K_M, Q8_0, F16, etc.
    int           n_dims;
    int64_t       ne[4];           // shape
    size_t        nb[4];           // strides (bytes)

    // Physical storage (may have multiple representations)
    struct {
        BufferHandle  gpu_buf;     // OpenCL buffer (or zero-copy wrapper)
        BufferHandle  gpu_img;     // image1d_buffer_t alias (Adreno only)
        void*         host_ptr;    // CPU-accessible pointer (mmap or SVM)
        size_t        offset;      // offset within buffer
        size_t        size_bytes;  // total size
    } storage;

    // For dual-quantization weights (optional)
    struct {
        BufferHandle  q4_gpu_buf;   // Q4_K_M for decode
        BufferHandle  q8_gpu_buf;   // Q8_0 for prefill
    } dual_quant;

    bool is_view;                   // true if this is a view into another tensor
    int  view_src_id;               // index of source tensor if view
} VirtualTensor;
```

### 15.2 Activation Buffer Pool

Activations are temporary — they're produced by one layer and consumed by the next. We reuse buffers:

```c
typedef struct {
    BufferHandle buf;
    size_t       size;
    bool         in_use;
} PoolSlot;

typedef struct {
    PoolSlot*    slots;
    int          n_slots;
    int          max_slots;
    Backend*     backend;     // which backend allocates these
} ActivationPool;

// At model init, pre-allocate based on max activation sizes:
void pool_init(ActivationPool* pool, Backend* backend, Model* model) {
    // Compute max activation size per position in the execution plan
    // For a standard transformer decode:
    //   hidden state: batch * hidden_dim * sizeof(fp16) = 1 * 4096 * 2 = 8 KB
    //   QKV intermediate: batch * 3 * hidden_dim * sizeof(fp16) = 24 KB
    //   FFN intermediate: batch * ff_dim * sizeof(fp16) = 1 * 11008 * 2 = 22 KB
    //   Attention scores: batch * n_heads * seq_len * sizeof(fp16) = flash attn avoids this!

    // Pre-allocate a small set of buffers at max required sizes
    size_t sizes[] = {
        model->hidden_dim * sizeof(_Float16) * 2,     // 2 hidden-dim buffers (ping-pong)
        model->ff_dim * sizeof(_Float16) * 2,          // 2 FFN intermediate buffers
        model->hidden_dim * 3 * sizeof(_Float16),      // QKV combined
    };
    // ... allocate pool slots
}
```

### 15.3 Weight Memory Organization

```
Total GPU-accessible memory budget (example: 8GB phone, ~4GB available):
  ├── Vision encoder weights (FP16):  ~600 MB
  ├── LLM weights (Q4_K_M):          ~2.0 GB  (for 7B model)
  ├── LLM weights (Q8_0, optional):  ~3.5 GB  (if dual-quant, skip for <8GB devices)
  ├── KV cache (INT8):               ~264 MB  (4096 context)
  ├── Activation buffers:            ~10 MB   (reused)
  ├── Token embeddings:              ~50 MB
  └── Overhead (page tables, etc.):  ~50 MB
                                     ─────────
                                     ~3.0 GB without dual-quant
                                     ~6.5 GB with dual-quant
```

---

## 16. Platform Abstraction Layer

### 16.1 Platform Detection

```c
typedef struct {
    // Hardware
    DeviceType  gpu_type;           // ADRENO, MALI, DESKTOP, NONE
    char        gpu_name[128];      // "Adreno (TM) 730", "Mali-G715", etc.
    int         gpu_compute_units;
    size_t      gpu_max_alloc;
    size_t      gpu_local_mem_size;
    bool        gpu_supports_fp16;
    bool        gpu_supports_int8_dot;
    bool        gpu_supports_svm;
    bool        gpu_supports_images;

    // Memory
    bool        is_uma;             // true for all Android, false for desktop discrete
    size_t      total_ram;
    size_t      available_ram;

    // CPU
    int         n_big_cores;        // Cortex-A7xx / Cortex-X
    int         n_little_cores;     // Cortex-A5xx
    bool        cpu_has_fp16;       // FEAT_FP16 on ARMv8.2+
    bool        cpu_has_dotprod;    // FEAT_DotProd on ARMv8.2+
    bool        cpu_has_i8mm;       // FEAT_I8MM on ARMv8.6+ (INT8 matrix multiply)
    bool        cpu_has_sve;        // SVE/SVE2 on some ARMv9

    // Derived
    int         recommended_cpu_threads;  // typically n_big_cores
    size_t      max_model_size;           // available_ram - safety_margin
} PlatformInfo;

PlatformInfo detect_platform(void);
```

### 16.2 Tuning Parameters Per Platform

```c
typedef struct {
    // GEMV (decode) tuning
    int    gemv_local_size[2];     // workgroup size
    int    gemv_splitk_threshold;  // K above which SplitK is used
    bool   gemv_use_image_weights; // Adreno: true, Mali/desktop: false
    int    gemv_elements_per_thread; // vectorization width

    // GEMM (prefill) tuning
    int    gemm_tile_n;
    int    gemm_tile_k;
    int    gemm_tile_b;
    bool   gemm_use_int8_dot;
    int    gemm_local_size[2];

    // Flash Attention tuning
    int    flash_attn_tile_size;   // K/V tile loaded into local memory
    int    flash_attn_local_size[2];

    // Memory
    bool   use_svm;                // SVM vs USE_HOST_PTR
    int    weight_buffer_alignment; // page alignment for zero-copy
} TuningParams;

TuningParams get_tuning_params(PlatformInfo* platform) {
    TuningParams params = {};

    if (platform->gpu_type == DEVICE_GPU_ADRENO) {
        params.gemv_local_size[0] = 128;
        params.gemv_local_size[1] = 1;
        params.gemv_splitk_threshold = 2048;
        params.gemv_use_image_weights = true;
        params.gemv_elements_per_thread = 8;

        params.gemm_tile_n = 64;
        params.gemm_tile_k = 16;
        params.gemm_tile_b = 16;
        params.gemm_use_int8_dot = platform->gpu_supports_int8_dot;
        params.gemm_local_size[0] = 16;
        params.gemm_local_size[1] = 8;

        params.flash_attn_tile_size = 64;
        params.flash_attn_local_size[0] = 64;
        params.flash_attn_local_size[1] = 1;

        params.use_svm = platform->gpu_supports_svm;
        params.weight_buffer_alignment = 4096;

    } else if (platform->gpu_type == DEVICE_GPU_MALI) {
        params.gemv_local_size[0] = 8;
        params.gemv_local_size[1] = 4;  // 32 total
        params.gemv_splitk_threshold = 4096;  // less aggressive split on Mali
        params.gemv_use_image_weights = false; // no L1 benefit on Mali
        params.gemv_elements_per_thread = 4;

        params.gemm_tile_n = 32;
        params.gemm_tile_k = 8;
        params.gemm_tile_b = 8;
        params.gemm_use_int8_dot = false;  // Mali INT8 dot is different
        params.gemm_local_size[0] = 8;
        params.gemm_local_size[1] = 4;

        params.flash_attn_tile_size = 32;
        params.flash_attn_local_size[0] = 8;
        params.flash_attn_local_size[1] = 4;

        params.use_svm = false;  // Often buggy on Mali
        params.weight_buffer_alignment = 4096;

    } else {
        // Desktop OpenCL — generic defaults, should be tuned per-GPU
        params.gemv_local_size[0] = 256;
        params.gemv_local_size[1] = 1;
        // ... etc
    }

    return params;
}
```

---

## 17. Implementation Order — Phase by Phase

### Phase 0: Foundation (Week 1)

```
□ Strip character engine modules from codebase
□ Clean up build system (CMake for Android NDK + Linux)
□ Implement PlatformInfo detection
□ Implement GGUF loader that builds VirtualTensor registry
□ Implement basic OpenCL backend init (context, queue, device detection)
□ Write test: load GGUF file, print all tensor names/shapes/types
```

### Phase 1: Zero-Copy Weight Loading (Week 2)

```
□ Implement mmap-based GGUF weight access
□ Implement CL_MEM_USE_HOST_PTR buffer creation with alignment handling
□ Implement image1d_buffer_t aliases for Adreno
□ Implement SVM path (optional, with fallback)
□ Write test: create GPU buffers for all tensors, verify no copy happened
□ Benchmark: compare load time vs llama.cpp default
```

### Phase 2: Core GEMV Kernel (Week 3-4)

```
□ Write Q4_K_M dequant + GEMV kernel for Adreno (image weights, wave64)
□ Write Q4_K_M dequant + GEMV kernel for Mali (float4, small workgroup)
□ Implement SplitK variant for large K dimensions
□ Write FP16 GEMV kernel (for non-quantized tensors)
□ Write activation pool + buffer management
□ Write test: single layer forward pass (decode), verify against ggml CPU output
□ Benchmark: per-layer GEMV timing, compare to llama.cpp decode
```

### Phase 3: Transformer Decode Path (Week 5-6)

```
□ Implement CPU NEON RMSNorm
□ Implement CPU NEON RoPE
□ Implement KV cache (INT8, basic linear buffer)
□ Implement basic attention (not flash yet — correctness first)
□ Implement fused SwiGLU FFN kernel
□ Implement scheduler (CPU norm → GPU GEMV → CPU RoPE → GPU attention → ...)
□ Wire up full decode loop: embedding → N layers → LM head → sample
□ Write test: generate text from an LLM-only GGUF, verify coherent output
□ Benchmark: tokens/second decode, compare to llama.cpp
```

### Phase 4: Prefill Path (Week 7-8)

```
□ Write Q8_0 GEMM kernel (tiled, INT8 dot where available)
□ Write FP16 GEMM kernel (for vision encoder)
□ Implement phase detection in scheduler (batch size → kernel selection)
□ Implement prefill loop with batched processing
□ Write test: prefill 512 tokens, verify KV cache matches decode-one-at-a-time
□ Benchmark: time-to-first-token
```

### Phase 5: Flash Attention (Week 9-10)

```
□ Implement Flash Attention OpenCL kernel (online softmax, tiled K/V)
□ Implement causal masking within flash attention
□ Handle GQA (n_kv_heads < n_heads — broadcast K/V across query groups)
□ Write test: flash attention output matches naive attention
□ Benchmark: attention time for 1024, 2048, 4096 context lengths
```

### Phase 6: Vision Encoder (Week 11-12)

```
□ Implement image preprocessing kernel (GPU: resize + normalize)
□ Implement patch embedding (conv2d or fused kernel)
□ Implement ViT transformer block (LayerNorm + attention + GELU FFN)
□ Implement projector types (MLP, pixel shuffle + MLP)
□ Implement embedding merge (vision tokens + text tokens)
□ Wire up full VLM pipeline: image + text → answer
□ Write test: load a VLM GGUF, process image, verify description is coherent
□ Benchmark: end-to-end VLM inference time
```

### Phase 7: Optimization Pass (Week 13-14)

```
□ Implement operator fusion (SwiGLU, dequant+matmul, bias+matmul)
□ Implement paged KV cache
□ Implement dual quantization (Q4 + Q8 for same weights)
□ Implement OpenCL kernel binary caching
□ Profile with Snapdragon Profiler / ARM Streamline
□ Tune workgroup sizes per-device
□ Implement thermal monitoring + adaptive throttle
□ Final benchmark comparison vs llama.cpp / MLC LLM
```

### Phase 8: Linux Desktop Backend (Week 15-16)

```
□ Implement PCIe-aware buffer management (explicit upload)
□ Adapt kernel selection for desktop OpenCL GPUs
□ Test on AMD/NVIDIA/Intel OpenCL
□ Implement x86 AVX/AVX2 CPU backend (for norm/RoPE/sampling)
□ Full test suite passing on both Android and Linux
```

---

## 18. Quick Reference Tables

### 18.1 Operation → Device → Kernel Mapping

| Operation                   | Decode (batch=1)        | Prefill (batch>32)            |
|-----------------------------|-------------------------|-------------------------------|
| Embedding lookup            | CPU (get_rows)          | CPU (get_rows)                |
| RMSNorm                     | CPU NEON                | CPU NEON (or GPU if batch>64) |
| Q projection                | GPU: gemv_q4k           | GPU: gemm_q8                  |
| K projection                | GPU: gemv_q4k           | GPU: gemm_q8                  |
| V projection                | GPU: gemv_q4k           | GPU: gemm_q8                  |
| RoPE                        | CPU NEON                | CPU NEON                      |
| KV cache write              | GPU (scatter)           | GPU (scatter)                 |
| Attention (QK^T→softmax→×V) | GPU: flash_attn         | GPU: flash_attn               |
| Output projection           | GPU: gemv_q4k           | GPU: gemm_q8                  |
| Residual add                | Fused or CPU            | Fused or CPU                  |
| FFN RMSNorm                 | CPU NEON                | CPU NEON (or GPU)             |
| FFN gate+up                 | GPU: dual_gemv_q4k      | GPU: gemm_q8                  |
| FFN SiLU×mul                | GPU: fused with gate+up | GPU: fused                    |
| FFN down                    | GPU: gemv_q4k           | GPU: gemm_q8                  |
| Final RMSNorm               | CPU NEON                | CPU NEON                      |
| LM head                     | GPU: gemv_q4k           | GPU: gemm_q8                  |
| Sampling                    | CPU                     | CPU                           |

### 18.2 Memory Bandwidth Requirements (7B Q4_K_M, per token decode)

| Per Layer                     | Bytes Read       | Source          |
|-------------------------------|------------------|-----------------|
| attn_q (4096×4096, Q4_K)      | 9.4 MB           | Weights         |
| attn_k (4096×1024, Q4_K)      | 2.4 MB           | Weights         |
| attn_v (4096×1024, Q4_K)      | 2.4 MB           | Weights         |
| attn_output (4096×4096, Q4_K) | 9.4 MB           | Weights         |
| KV cache read                 | variable         | KV cache        |
| ffn_gate (4096×11008, Q4_K)   | 25.3 MB          | Weights         |
| ffn_up (4096×11008, Q4_K)     | 25.3 MB          | Weights         |
| ffn_down (11008×4096, Q4_K)   | 25.3 MB          | Weights         |
| **TOTAL per layer**           | **~99.5 MB**     |                 |
| **TOTAL 32 layers**           | **~3.2 GB**      |                 |
| **At 50 GB/s LPDDR5**         | **~64 ms/token** | Theoretical min |
| **At 80 GB/s LPDDR5X**        | **~40 ms/token** | Theoretical min |

These are theoretical minimums assuming 100% bandwidth utilization. Real-world: expect 60-75% utilization → 50-100
ms/token.

### 18.3 Adreno vs Mali Cheat Sheet

| Feature           | Adreno (7xx)                 | Mali (Valhall/Immortalis)               |
|-------------------|------------------------------|-----------------------------------------|
| Wave/warp size    | 64                           | 1 (thread-level)                        |
| L1 cache          | 2KB/uSPTP, IMAGE ONLY        | No separate L1 for compute              |
| Local memory      | 32KB/SP, fast                | Exists but often not faster than global |
| FP16 rate         | 2× FP32                      | 2× FP32                                 |
| INT8 dot          | Yes (Adreno 730+)            | Varies, less common                     |
| Best workgroup    | 128 or 256                   | 32 or 64                                |
| Branch divergence | Costs 2× on divergent warps  | FREE (warp size = 1)                    |
| Shared mem tiling | VERY helpful (local is fast) | Often HURTS (sync overhead > benefit)   |
| Image objects     | USE THEM (L1 cache!)         | Optional (no L1 benefit)                |
| SVM support       | 6xx+ (coarse), 7xx+ (fine)   | G77+ (often buggy)                      |

### 18.4 Quantization Bits-Per-Weight Summary

| Type   | Bits/Weight | Block Size | Best For                                 |
|--------|-------------|------------|------------------------------------------|
| Q4_0   | 4.5         | 32         | Legacy, avoid                            |
| Q4_1   | 5.0         | 32         | Legacy, avoid                            |
| Q4_K_S | 4.5         | 256        | Smaller models, max compression          |
| Q4_K_M | 4.5         | 256        | **Decode king** — best quality/bandwidth |
| Q5_K_S | 5.5         | 256        | Middle ground                            |
| Q5_K_M | 5.5         | 256        | Middle ground                            |
| Q6_K   | 6.5         | 256        | High quality, if memory allows           |
| Q8_0   | 8.5         | 32         | **Prefill king** — INT8 compute path     |
| F16    | 16.0        | 1          | Vision encoder, projector                |
| F32    | 32.0        | 1          | Never use on mobile                      |

---

## Appendix A: Key Reference Code Locations in GGML

If you need to understand the exact binary format or op semantics, look here:

```
ggml/src/ggml-quants.c          → dequantize_row_q4_K(), quantize_row_q4_K_ref()
                                   This is THE reference for Q4_K block layout and unpacking.

ggml/src/ggml.c                 → ggml_compute_forward_mul_mat()
                                   Reference CPU implementation of matmul with quantized weights.

ggml/src/ggml.c                 → ggml_compute_forward_rms_norm()
                                   Reference RMSNorm implementation.

ggml/src/ggml.c                 → ggml_compute_forward_rope()
                                   Reference RoPE implementation (check rope_type for variants).

ggml/include/ggml.h             → enum ggml_type, enum ggml_op
                                   All type and op enumerations.

ggml/include/gguf.h             → gguf_init_from_file(), gguf_get_val_*()
                                   GGUF file parsing API.
```

---

## Appendix B: Thermal & Power Budget

| Scenario                   | Typical Power | Sustainable Duration               |
|----------------------------|---------------|------------------------------------|
| GPU-only decode (30 tok/s) | 3-5 W         | 30-60s before throttle             |
| CPU+GPU heterogeneous      | 2-4 W         | 60-120s (cooler, interleaved idle) |
| Vision encode burst        | 5-7 W         | <2s (one-shot, fine)               |
| Idle between messages      | 0.1 W         | Forever                            |

**Rule of thumb:** if sustained GPU utilization > 80% for > 45 seconds, expect 20-40% clock throttle on most Android
phones. The heterogeneous approach naturally creates micro-idle periods that help with thermals.

---

*Last updated: This document. Keep it in sync with code changes.*
*Location: /home/home/CLionProjects/llama.cpp-custom/CLAUDE.md (or wherever your project lives)*