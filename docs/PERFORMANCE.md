# Performance

Optimizations specific to this CPU-only Android fork.

---

## Benchmarks

Tested on **Cortex-X3** (armv9-a, i8mm, bf16, NEON, dotprod):

| Model | Type | Quant | Prompt Eval | Generation | Context |
|-------|------|-------|-------------|------------|---------|
| LFM2-350M | Text | Q8_0 | ~500 t/s | 29-30 t/s | 2048 |
| SmolVLM-500M | VLM | Q8_0 | ~22 t/s (w/ image) | 28 t/s | 2048 |
| Qwen3-0.6B | Text | Q8_0 | ~350 t/s | 17-19 t/s | 2048 |
| Gemma3-1B | Text | Q4_K_M | ~250 t/s | 14 t/s | 2048 |
| EmbeddingGemma-300M | RAG | Q4_0 | ~25ms/query | N/A | 2048 |

---

## ARM Optimizations

### Architecture Features

Enabled via `GGML_CPU_ARM_ARCH=armv8.6-a+i8mm+dotprod+fp16`:

| Feature | Effect |
|---------|--------|
| **i8mm** | INT8 matrix multiply — accelerates Q4/Q8 quantized inference |
| **dotprod** | Dot product instructions — fast Q4_0/Q8_0 kernels |
| **fp16** | Half-precision FP — F16 compute without conversion overhead |
| **NEON** | 128-bit SIMD — baseline vector operations |
| **bf16** | BFloat16 — used by some compute kernels when available |

### KleidiAI Micro-Kernels

ARM's optimized GEMM/GEMV kernels for quantized operations. Enabled via `GGML_CPU_KLEIDIAI=ON`.

These replace the generic C implementations with hand-tuned assembly for:
- Q4_0 × F32 matrix multiply
- Q8_0 × F32 matrix multiply
- Q4_K × F32 mixed-precision GEMM

---

## Threading

### Prompt Processing (Compute-Bound)

```
n_threads_batch = all P-cores (e.g., 4 on Cortex-X3)
```

Prompt evaluation is compute-bound — more threads = faster. Uses all available performance cores.

### Token Generation (Memory-Bound)

```
n_threads = min(4, P-cores)
```

Generation is memory-bandwidth-bound (single-row matrix-vector multiply). Adding more threads increases cache contention without improving throughput.

### CPU Affinity

The JNI bridge pins threads to performance cores via `sched_setaffinity`. This prevents the scheduler from migrating inference threads to efficiency cores, which would halve throughput.

---

## KV Cache Optimizations

### Prefix Reuse

Multi-turn conversations share a common prefix (system prompt + earlier turns). The engine detects the longest common prefix and skips re-evaluating those tokens.

```
Turn 1:  [SYS][USER_1][ASST_1]          → eval all
Turn 2:  [SYS][USER_1][ASST_1][USER_2]  → skip prefix, eval USER_2 only
Turn 3:  [SYS][USER_1][ASST_1][USER_2][ASST_2][USER_3]  → skip prefix
```

Savings: typically 50-80% of prompt tokens are skipped on follow-up turns.

### Context Shifting

When the KV cache fills up (`context_used >= context_size`), the engine automatically shifts:

1. Keep first N tokens (system prompt)
2. Keep last M tokens (recent conversation)
3. Remove middle tokens from KV cache
4. Continue generation without interruption

This allows infinite conversation length without model reload.

### Disk-Backed Prompt Cache

System prompts are cached to disk using FNV-1a hashed filenames. On cold start with the same system prompt, the engine loads the cached KV state instead of re-evaluating.

```
First load:   system prompt → tokenize → eval → save to disk
Second load:  system prompt → hash match → load from disk (instant)
```

Cache location is set via `setPromptCacheDir()` in the Kotlin SDK.

---

## Speculative Decoding

Ngram-based self-speculative decoding (no draft model required):

1. **Draft**: N-gram cache predicts next K tokens based on observed patterns
2. **Verify**: Batch-evaluate all K tokens in parallel against the model
3. **Accept**: Tokens matching the model's distribution are accepted

```
Without speculation:  1 token per forward pass
With speculation:     1-4 tokens per forward pass (for structured output)
```

Performance:
- **1.3-2x speedup** for JSON, code, and repetitive text
- **No speedup** for creative/novel text (n-gram cache misses)
- **Zero overhead** when disabled (opt-in via `setSpeculativeDecoding()`)

---

## RAG Performance

### Embedding Model

EmbeddingGemma-300M Q4_0 (~265 MB), 768 native dims, 2048 context.

| Operation | Time (Cortex-X3) | Notes |
|-----------|-------------------|-------|
| Model load | ~1-2s | mmap'd, fast cold start |
| Document indexing (1 chunk) | ~50-100ms | Tokenize + encode + pool + BQ |
| Query (25 chunks indexed) | ~25ms | Encode query + Hamming search + cosine re-rank |
| Query (100 chunks indexed) | ~30ms | BQ pre-filter keeps it fast |

### Memory

| Component | Size |
|-----------|------|
| Embedding model (mmap) | ~265 MB virtual, ~50 MB resident |
| Per chunk (256 dims) | ~1 KB float + 4 words BQ = ~1.04 KB |
| 1000 chunks | ~1 MB |
| 10000 chunks | ~10 MB |

### Tuning Parameters

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `n_dims` | Lower = faster search, less accurate | 256 (good balance) |
| `chunk_size` | Smaller = more chunks, finer retrieval | 256 (default) |
| `top_k` | More BQ candidates = slower, more accurate re-rank | 32 (default) |
| `top_n` | More final results | 5 (default) |
| `late_chunking` | Better quality, slightly slower indexing | true (default) |

---

## Binary Size

| Optimization | Effect |
|--------------|--------|
| `BUILD_SHARED_LIBS=OFF` | All static libs linked into single .so |
| `-ffunction-sections` + `-fdata-sections` | Per-function/data sections |
| `--gc-sections` | Strip unreferenced sections |
| `--icf=safe` | Merge identical code sections |
| `-fvisibility=hidden` | No PLT entries for internal symbols |
| Strip (`-s`) | Remove symbol table |

Result: **~4.1 MB** stripped `.so` for arm64-v8a (down from 5.6 MB without section flags).

---

## JNI Optimizations

These are implemented in the JNI bridge (`gguf_lib.cpp`), not in this repo:

| Optimization | Description |
|--------------|-------------|
| Method ID caching | JNI method IDs cached at first call, not per-token |
| Zero-copy token delivery | `ByteBuffer` for token bytes, no JNI string allocation |
| Warm-up pass | Single BOS token decoded after model load to prime caches |
| Refusal token scan | Vocabulary scanned once at load, cached for uncensored mode |
| Sampler state preservation | Sampler rebuilt only when parameters change |
| Prompt eval progress | Float callback for prompt processing progress |

---

## Memory Usage

### Model Memory

Models are memory-mapped (`use_mmap=true` by default). The OS pages in only the weights being accessed, so a 4GB model doesn't require 4GB of free RAM.

### KV Cache Memory

Approximate KV cache size:

```
KV bytes ≈ n_ctx × n_layer × 2 × (n_embd / n_head) × n_head_kv × sizeof(type)
```

| Model | Context | KV Cache |
|-------|---------|----------|
| Qwen3-0.6B | 2048 | ~64 MB |
| Gemma3-1B | 2048 | ~128 MB |
| LLaMA-3.2-3B | 2048 | ~256 MB |

### Reducing Memory

- Use smaller context sizes (`n_ctx = 1024` instead of 2048)
- Use smaller quantization formats (Q4_0 vs Q8_0 for KV cache)
- Enable flash attention (`flash_attn = true`) for reduced KV memory
