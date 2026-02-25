# gguf-engine — Native AI Character Engine for Android

A standalone C++ inference engine that runs GGUF language models directly on Android devices with **zero dependencies on llama.cpp runtime**. Built on raw ggml tensor operations with 8 neural intervention surfaces for real-time character behavior control.

```
Qwen3-0.6B Q8_0 on Snapdragon 7s Gen 3:
  Decode:  47ms/token  (21 tok/s)
  Prefill: 20ms/token
  Memory:  ~990 MB (762 MB weights + 224 MB KV cache)
  Binary:  5.7 MB (ARM64)
```

---

## Quick Start

### One-Line Install (Android device via ADB)

```bash
curl -sL https://raw.githubusercontent.com/Siddhesh2377/llama.cpp-android/character-engine-v1/android/install.sh | bash
```

The script will:
1. Download the `gguf-engine-cli` binary from GitHub releases
2. Download `aria.json` character config
3. Ask which model to download (Qwen3-0.6B Q8_0, Q4_K_M, or skip)
4. Push everything to device via ADB (or install locally)
5. Generate `config.json` with defaults
6. Print the run command

```bash
# Or run the install script manually
bash install.sh
```

This downloads the binary + character config + optionally a model, pushes everything to your device, and prints the run command.

### Manual Run

```bash
# On device (adb shell)
cd /data/local/tmp
LD_LIBRARY_PATH=. ./gguf-engine-cli model.gguf --char-chat --ch-json aria.json --threads 4
```

### CLI Modes

```bash
# Interactive character chat (full engine: mood, memory, tool calling)
./gguf-engine-cli model.gguf --char-chat --ch-json aria.json

# Run all 8 character engine tests
./gguf-engine-cli model.gguf --char-engine-test --ch-json aria.json

# Basic chat (no character personality)
./gguf-engine-cli model.gguf --chat --tokens 256

# Web search + RAG answer
./gguf-engine-cli model.gguf --web-search "latest news about AI"

# Tool calling test (grammar-constrained JSON)
./gguf-engine-cli model.gguf --tool-test --ch-json aria.json

# Raw benchmark (greedy decode)
./gguf-engine-cli model.gguf --tokens 64 --threads 4

# Full help
./gguf-engine-cli --help
```

### In-Chat Commands

Once inside `--char-chat` mode:

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/config` | Show current configuration |
| `/save-config` | Save settings to config.json |
| `/threads N` | Set CPU thread count (1-8) |
| `/gpu N` | Toggle GPU (0=off, 1=on) |
| `/temp F` | Set temperature (0.0-2.0) |
| `/top-k N` | Set top-k sampling (1-200) |
| `/top-p F` | Set nucleus sampling (0.0-1.0) |
| `/rep-penalty F` | Set repetition penalty (1.0-2.0) |
| `/tokens N` | Set max tokens per reply (8-4096) |
| `/mood` | Show emotional state (warmth/energy/formality) |
| `/reset` | Clear conversation + reset mood |
| `/download-model` | Show model download options |
| `/quit` | Exit |

---

## Architecture

### How It Works

The engine builds a **ggml compute graph** for each forward pass — the same tensor math library that powers llama.cpp, but without llama.cpp's inference runtime. Every transformer operation (attention, FFN, RoPE, RMS norm) is constructed explicitly in `graph.cpp`, giving us 8 injection points for character behavior control.

```
User message
    │
    ▼
┌──────────────┐    ┌──────────────────┐
│  Tokenizer   │───▶│  Mood Detection  │  (keyword scan, <1ms)
│  (BPE/ChatML)│    │  3 axes: W/E/F   │
└──────┬───────┘    └────────┬─────────┘
       │                     │ adjusts interventions
       ▼                     ▼
┌──────────────────────────────────────┐
│         Transformer Forward Pass      │
│                                       │
│  For each of 28 layers:              │
│    ├─ RMS Norm (+norm_shift)         │  ◄─ Fast weight recall
│    ├─ Q/K/V Projections             │
│    ├─ RoPE Positional Encoding       │
│    ├─ Attention (+temp, +bias)       │  ◄─ Per-layer temperature
│    ├─ Head Rescaling                 │  ◄─ Importance weighting
│    ├─ Gated Residual (attn)          │  ◄─ Dampen middle layers
│    ├─ FFN (gate/up/down)             │
│    ├─ Gated Residual (ffn)           │  ◄─ Dampen middle layers
│    ├─ Control Vector Addition        │  ◄─ Steering vectors
│    └─ Activation Capture             │  ◄─ For fast weight update
│                                       │
│  Output: logits + bias               │  ◄─ EOS suppression
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│            Sampling                   │
│  temp → top-k → top-p → rep_penalty │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         Grammar Engine               │
│  LAZY mode: free text until          │
│  <tool_call> detected → enforce JSON │
└──────────────────┬───────────────────┘
                   │
                   ▼
              Token output
```

### 8 Intervention Surfaces

All interventions are **universal** — they work with any GGUF model architecture (Qwen, LLaMA, SmolLM, etc.) because they operate on the compute graph, not model-specific code.

| # | Surface | Where | What It Does | Overhead |
|---|---------|-------|-------------|----------|
| 1 | **Attention Temperature** | Pre-softmax | Per-layer sharpness control (early=creative, late=focused) | ~0 (fused) |
| 2 | **Attention Bias** | Pre-softmax | Score injection for attention steering | ~0 (fused) |
| 3 | **Head Rescaling** | Post-attention | Importance-weighted head scaling from control vectors | ~0 (fused) |
| 4 | **Gated Residual** | Post-attn/FFN | Dampen middle layer contributions (0.0-1.0 gate) | ~0 (fused) |
| 5 | **Control Vectors** | Post-FFN | Contrastive activation addition from .gguf direction files | ~0 (fused) |
| 6 | **Norm Shift** | Pre-RMS-norm | Fast weight memory recall injected into hidden state | ~0 (fused) |
| 7 | **Logit Bias** | Output | Per-token boost/suppress (EOS control, content steering) | ~0 (fused) |
| 8 | **Activation Capture** | Any layer | Read hidden states for fast weight update | ~0 (fused) |

**Total per-token overhead: <0.1%** — all interventions are ggml graph nodes, computed alongside the model.

---

## Character System

### Personality JSON

Characters are defined in JSON files that control every aspect of generation:

```json
{
    "name": "Aria",
    "system_prompt": "You are Aria, a 28-year-old woman who works as a creative technologist...",
    "temp_early": 1.30,
    "temp_mid": 1.00,
    "temp_late": 0.80,
    "attn_gate_mid": 0.90,
    "ffn_gate_mid": 0.93,
    "logit_bias_eos": -4.0,
    "sampling_temp": 0.85,
    "sampling_top_k": 50,
    "sampling_top_p": 0.93,
    "rep_penalty": 1.20,
    "thinking": 0,
    "mood_warmth": 0.72,
    "mood_energy": 0.60,
    "mood_formality": 0.28,
    "stall_prompt": "Hmm let me look that up real quick",
    "stall_max_tokens": 20,
    "fw_enabled": 1,
    "fw_dim_reduced": 128,
    "fw_gamma": 0.95,
    "fw_eta": 0.01
}
```

### Emotional State Tracking

The engine tracks 3 mood axes that shift based on user messages and feed back into interventions:

| Axis | Range | Effect |
|------|-------|--------|
| **Warmth** | 0.0-1.0 | Higher → boosts early-layer temperature (more creative responses) |
| **Energy** | 0.0-1.0 | Higher → tighter attention gates (more focused, concise) |
| **Formality** | 0.0-1.0 | Higher → stronger EOS bias (longer, more structured responses) |

- Detection: keyword scan (<1ms per message, 29 trigger words)
- Update: exponential moving average (alpha=0.85)
- Clamp: ±0.3 from persona baseline (prevents drift)

### Fast Weight Memory

Hebbian associative memory (Schmidhuber 1992) that gives the model a fixed-size persistent memory:

```
W_fast(t) = gamma * W_fast(t-1) + eta * h_reduced ⊗ h_reduced
```

- Random projection: d_model (1024) → d_reduced (128)
- Matrix size: 64 KB (fixed, regardless of conversation length)
- Update: ~50 microseconds per token (CPU-side outer product)
- Recall injected via norm_shift intervention surface

Unlike KV cache (which grows linearly with context), fast weights store **patterns** in constant memory.

---

## SDK Modules

### Module Map

```
include/gguf-engine/
├── engine.h          ← Single include for everything
├── types.h           ← All structs, enums, constants (468 lines)
├── utils.h           ← Inline helpers (decode_token, gguf getters)
├── model.h           ← load_model(), init_kv_cache()
├── graph.h           ← build_graph() — full transformer forward pass
├── interventions.h   ← init/free interventions, control vector loader
├── sampling.h        ← sample_token() — temp/top-k/top-p/rep_penalty
├── tokenizer.h       ← BPE tokenizer, ChatML template builder
├── personality.h     ← JSON parser, profile state, binary save/load
├── emotional.h       ← Mood detection, intervention adjustment, fast weights
├── grammar.h         ← JSON grammar state machine, tool call detection
├── boundaries.h      ← generate_until_boundary() — stop strings, EOS
├── rag.h             ← BM25 + vector + KG + web search + hybrid
├── stall.h           ← Mini-KV stall generation during async tools
├── vision.h          ← SigLIP ViT + idefics3 projector
├── tui.h             ← ANSI color terminal UI, config persistence
└── chat.h            ← run_chat_turn(), interactive chat, slash commands
```

### Using the SDK

```cpp
#include "gguf-engine/engine.h"

// Load model
ModelState state = {};
load_model(state, "model.gguf", -1, backend);
init_kv_cache(state, backend);

// Load character
PersonalityConfig pc;
parse_personality_json("aria.json", pc);
ProfileState profile;
profile_from_personality(pc, profile, state.cfg);
profile_apply(profile, state);

// Build graph and generate
auto * graph = build_graph(ctx, state, 1, state.cfg.n_layer, &state.iv);
ggml_backend_graph_compute(backend, graph);

// Sample
int token = sample_token(state, sp);
```

### Build System

```cmake
# Static library — all 14 modules
add_library(gguf-engine STATIC ${ENGINE_SOURCES})
target_link_libraries(gguf-engine PUBLIC ggml ggml-base)
target_compile_features(gguf-engine PUBLIC cxx_std_17)

# Your app
add_executable(my-app main.cpp)
target_link_libraries(my-app PRIVATE gguf-engine)
```

---

## Tool Calling

Grammar-constrained JSON generation with LAZY mode activation:

```
User: "What's the weather in Tokyo?"

Model (free text): "Let me check that for you."
Model (grammar kicks in): <tool_call>{"name":"web_search","arguments":{"query":"weather Tokyo"}}
Engine: [executes tool, prefills result into KV cache]
Model (continues): "It's currently 15°C and partly cloudy in Tokyo."
```

- **Grammar engine**: 18-state JSON state machine, depth 32
- **String-safe bitmap**: 148,532/151,936 tokens precomputed
- **Performance**: 820 microseconds/token overhead (1.6% on 50ms decode)
- **Multi-step**: tool result prefilled into KV cache, model continues naturally

---

## RAG Pipeline

Full retrieval pipeline with 4 search backends:

| Backend | Description | Speed |
|---------|-------------|-------|
| **BM25** | Term frequency search with IDF weighting | <1ms |
| **Vector** | Cosine similarity on chunk embeddings | ~5ms |
| **Knowledge Graph** | Triple extraction + entity lookup | <1ms |
| **Web Search** | DuckDuckGo API + HTML extraction | ~500ms |

Results are fused via **Reciprocal Rank Fusion (RRF)** and injected into the prompt context.

---

## Performance

### Benchmarks (Qwen3-0.6B Q8_0, Snapdragon 7s Gen 3)

| Metric | Value |
|--------|-------|
| Decode speed | **47ms/tok (21 tok/s)** |
| Prefill speed | **20ms/tok** |
| Full forward (28 layers) | **50ms** |
| Single layer | **18ms** |
| Model load (GGUF parse) | **465ms** |
| Weight copy to buffer | **296ms** |
| Intervention overhead | **<0.1%** |
| Grammar overhead | **1.6%** (820us/tok) |
| Fast weight update | **~50us/tok** |
| Mood detection | **<1ms/message** |

### Memory Budget

| Component | Size |
|-----------|------|
| Model weights (Q8_0) | 762 MB |
| KV cache (F16, 2048 ctx) | 224 MB |
| Intervention tensors | 1.0 MB |
| Fast weight matrix | 64 KB |
| Stall mini-KV (32 ctx) | ~0.9 MB |
| Compute buffer | 0.6 MB |
| **Total** | **~988 MB** |

### Thread Scaling

| Threads | ms/tok | Notes |
|---------|--------|-------|
| 1 | ~140ms | Single A78 core |
| 2 | ~75ms | |
| 3 | ~52ms | Sweet spot |
| **4** | **~47ms** | **Optimal (4x A78 perf cores)** |
| 6 | ~55ms | Spills to A55 efficiency cores |
| 8 | ~65ms | Worse — efficiency cores too slow |

The engine is **memory-bandwidth bound** (612 MB weights per token @ ~13 GB/s LPDDR5), not compute bound. More threads beyond the performance core count just adds synchronization overhead.

---

## Building from Source

### Prerequisites

- Android NDK r28+ (tested with r28c)
- CMake 3.22+
- A device with adb access

### Build

```bash
cd llama.cpp-android

mkdir build-android && cd build-android

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_OPENCL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_LLAMAFILE=OFF \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc) gguf-engine-cli
```

### Deploy

```bash
adb push bin/gguf-engine-cli /data/local/tmp/
adb push bin/*.so /data/local/tmp/
adb push android/aria.json /data/local/tmp/
adb shell "chmod +x /data/local/tmp/gguf-engine-cli"
```

### Run Tests

```bash
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. \
  ./gguf-engine-cli /sdcard/Download/Qwen3-0.6B-Q8_0.gguf \
  --char-engine-test --ch-json aria.json --tokens 128 --threads 4"
```

Expected: `8/8 subtests passed`

---

## Recommended Models

| Model | Size | Quality | Download |
|-------|------|---------|----------|
| **Qwen3-0.6B Q8_0** | 660 MB | Best quality for size | [HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |
| **Qwen3-0.6B Q4_K_M** | 430 MB | Good balance | [HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) |
| **SmolLM3-3B Q4_K_M** | 1.9 GB | Best quality, needs more RAM | [HuggingFace](https://huggingface.co/bartowski/SmolLM3-3B-GGUF) |

---

## Config System

The engine reads `config.json` on startup (like Claude Code's settings):

```json
{
    "model_path": "/sdcard/Download/Qwen3-0.6B-Q8_0.gguf",
    "character_json": "aria.json",
    "threads": 4,
    "gpu": 0,
    "max_tokens": 256,
    "max_ctx": 2048,
    "temp": 0.85,
    "top_k": 50,
    "top_p": 0.93,
    "rep_penalty": 1.20,
    "color": 1,
    "verbose": 0
}
```

CLI flags override config.json values. Use `/save-config` in chat to persist your changes.

---

## Project Structure

```
llama.cpp-android/
├── android/                     ← Character engine SDK
│   ├── include/gguf-engine/     ← 17 public headers
│   ├── src/                     ← 14 module implementations
│   ├── tests/                   ← Integration test suite
│   ├── main.cpp                 ← CLI entry point
│   ├── aria.json                ← Default character
│   ├── install.sh               ← One-line installer
│   └── CMakeLists.txt           ← Build configuration
├── ggml/                        ← ggml tensor library (submodule)
└── CMakeLists.txt               ← Root build file
```

---

## License

This project builds on [ggml](https://github.com/ggerganov/ggml) (MIT License).
