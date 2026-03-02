# Architecture

## Stack

```
┌─────────────────────────────────────────────────┐
│                  Kotlin SDK                      │
│              (com.dark.gguf_lib)                 │
│   GGUFNativeLib · NativeLicenseLib · Callbacks   │
├─────────────────────────────────────────────────┤
│                  JNI Bridge                       │
│               (gguf_lib.cpp)                     │
│  Thread mgmt · CPU affinity · Prompt cache ·     │
│  Zero-copy ByteBuffer · Ngram speculation        │
├─────────────────────────────────────────────────┤
│              Engine Layer (engine/)               │
│  ┌──────────┬──────────┬───────────┬───────────┐  │
│  │GGMLEngine│VLM Engine│ToolManager│CharEngine │  │
│  │Load/Gen  │Vision/   │JSON/XML/  │Mood/Logit │  │
│  │KV Cache  │Audio     │Fn parse   │Bias/Uncen │  │
│  └────┬─────┴────┬─────┴─────┬─────┴─────┬────┘  │
├─────────┼─────────────┼──────────────┼──────────┤
│         │    llama.cpp Core (src/)   │           │
│  Model loading · Tokenization · Inference ·      │
│  Sampling · Compute graphs (100+ architectures)  │
├─────────────────────────────────────────────────┤
│           Common Utilities (common/)             │
│  Chat templates · JSON schema grammar ·          │
│  Sampling chains · Ngram cache · Speculative     │
├─────────────────────────────────────────────────┤
│              GGML (ggml/)                        │
│  Tensor library · CPU backend only               │
│  NEON · i8mm · dotprod · fp16 · bf16             │
│  KleidiAI ARM micro-kernels                      │
└─────────────────────────────────────────────────┘
```

---

## Directory Map

```
llama.cpp/
├── engine/                    Custom engine layer
│   ├── ggml-engine.h/.cpp     Model lifecycle, generation, context
│   ├── ggml-engine-vlm.cpp    VLM generation (text + images)
│   ├── ggml-engine-internal.h Shared structs and generation loop
│   ├── tool-manager.h/.cpp    Tool call parsing and execution
│   ├── character-engine.h/.cpp Personality, mood, uncensored mode
│   ├── vlm/                   Vision/audio encoder (mtmd library)
│   │   ├── clip.h/.cpp        CLIP/SigLIP vision encoder (CPU-only)
│   │   ├── mtmd.h/.cpp        Multimodal tokenizer + orchestration
│   │   ├── mtmd-helper.h/.cpp Image/audio loading (stb_image, miniaudio)
│   │   ├── mtmd-audio.h/.cpp  Mel spectrogram, audio preprocessing
│   │   └── models/            20+ VLM graph builders (LLaVA, Qwen, etc.)
│   ├── llama-test-cli.cpp     61-test validation suite
│   └── CMakeLists.txt         Builds libtn-engine.a
│
├── src/                       llama.cpp core
│   ├── llama.cpp              Main implementation (~15K lines)
│   ├── llama-*.cpp            Subsystems (vocab, sampling, context, etc.)
│   └── CMakeLists.txt         Builds libllama.a
│
├── include/                   Public C headers
│   ├── llama.h                Core C API
│   └── llama-cpp.h            C++ convenience wrappers
│
├── ggml/                      GGML tensor library
│   ├── src/                   Tensor operations, CPU backend
│   │   ├── ggml.c             Core tensor library
│   │   ├── ggml-cpu/          CPU-specific kernels
│   │   │   ├── ggml-cpu-impl.h
│   │   │   ├── ggml-cpu-aarch64.cpp  ARM64 optimized paths
│   │   │   └── kleidiai/      KleidiAI micro-kernels
│   │   └── ggml-threading.cpp Thread pool
│   └── include/               GGML headers
│       ├── ggml.h
│       ├── ggml-cpu.h
│       └── ggml-backend.h
│
├── common/                    Shared utilities
│   ├── common.cpp             General utilities
│   ├── chat-template.hpp      Jinja2-style template engine
│   ├── json-schema-to-grammar.cpp  JSON schema → GBNF grammar
│   ├── sampling.cpp           Sampling chain management
│   ├── ngram-cache.cpp        N-gram cache for speculation
│   ├── speculative.cpp        Speculative decoding
│   └── CMakeLists.txt         Builds libcommon.a
│
├── vendor/                    Third-party dependencies
│   ├── nlohmann/json.hpp      JSON library (used by engine)
│   ├── cpp-httplib/           HTTP library (common/download.cpp)
│   ├── stb/                   Image loading
│   ├── miniaudio/             Audio library
│   └── sheredom/              UTF-8 utilities
│
├── cmake/                     CMake modules
│   ├── build-info.cmake       Build metadata generation
│   ├── common.cmake           Shared CMake utilities
│   └── license.cmake          License embedding
│
├── CMakeLists.txt             Root build configuration
├── LICENSE                    MIT License
└── README.md                  Project overview
```

---

## Data Flow: Text Generation

```
User prompt (string)
    │
    ▼
┌─────────────┐
│ GGMLEngine  │  ggml_engine_generate()
│             │
│ 1. Tokenize prompt via llama_tokenize()
│ 2. Check KV cache prefix (skip shared tokens)
│ 3. Batch-decode prompt tokens
│ 4. Auto-shift context if window full
│             │
│ Generation loop:
│ │ 5. Sample next token (temp, top_k, top_p, min_p, penalties)
│ │ 6. Check stop sequences
│ │ 7. Detokenize token
│ │ 8. Invoke callback(token_text, user_data)
│ │ 9. If callback returns false → stop
│ │ 10. Decode token into KV cache
│ │ 11. Loop until n_predict or EOS
│             │
│ Return perf metrics
└──────┬──────┘
       │
       ▼
  Generated text
```

---

## Data Flow: VLM Generation (Text + Image)

```
User prompt + image file(s)
    │
    ▼
┌─────────────────────┐
│ VLM Engine           │  ggml_engine_vlm_generate()
│                      │
│ 1. Load image bytes → stb_image decode → mtmd_bitmap
│ 2. mtmd_tokenize(prompt + bitmaps)
│    → Split into text chunks + image chunks
│    → Image markers "<__media__>" → image positions
│                      │
│ 3. Clear KV cache
│ 4. mtmd_helper_eval_chunks()
│    → Text chunks:  tokenize → batch decode
│    → Image chunks: CLIP/SigLIP ViT encode → embedding injection
│    → Handles M-RoPE (Qwen2-VL) and non-causal attn (Gemma3)
│ 5. Update n_past from processed chunks
│                      │
│ 6. Shared generation loop:
│    → Sample next token
│    → Check stop sequences
│    → Callback(token_text)
│    → Decode into KV cache
│    → Loop until n_predict or EOS
│                      │
│ Return perf metrics (prompt_eval includes vision encode)
└──────────────────────┘
```

---

## Data Flow: Tool Calling

```
User message + tool definitions
    │
    ▼
┌──────────────┐
│ ToolManager  │
│              │
│ 1. tool_manager_get_prompt()
│    → Generate tool description text
│    → Inject into system/user prompt
│              │
│ 2. Engine generates response
│              │
│ 3. tool_manager_parse_output()
│    → Try JSON parse: {"name": ..., "arguments": ...}
│    → Try XML parse:  <tool_call>...</tool_call>
│    → Try function parse: name(key=value)
│    → Validate against registered tool definitions
│    → Return tool_call_result
│              │
│ 4. tool_manager_execute()
│    → Invoke registered callback
│    → Return result string
│              │
│ 5. Feed result back to engine for next turn
└──────────────┘
```

---

## Data Flow: Character Engine

```
Character personality + mood
    │
    ▼
┌──────────────────┐
│ CharacterEngine   │
│                   │
│ Base parameters:  │  personality.temperature = 0.9
│ + Mood modifiers: │  CHAR_MOOD_CREATIVE → +0.3 temp, +0.1 top_p
│ + Logit biases:   │  boost token 42 by +2.0
│ + Suppressions:   │  suppress token 99 (-inf)
│ + Uncensored:     │  suppress refusal tokens
│                   │
│ → char_effective_params
│   Final temp: 1.2, top_p: 1.0, biases: [...], suppressions: [...]
│                   │
│ character_engine_get_context()
│ → "You are Atlas, a witty AI assistant..."
│   Prepended to user prompt
└───────┬──────────┘
        │
        ▼
  Pass effective params to GGMLEngine sampling
```

---

## Threading Model

On Android, the JNI bridge manages threading:

```
┌─ Prompt Processing (compute-bound) ────────────────────┐
│  Uses n_threads_batch = all performance cores            │
│  CPU affinity pinned via sched_setaffinity               │
│  Example: 4 P-cores on Cortex-X3 → 4 threads            │
└──────────────────────────────────────────────────────────┘

┌─ Token Generation (memory-bound) ──────────────────────┐
│  Uses n_threads = min(4, P-cores)                        │
│  More threads = more cache contention, slower            │
│  Bottleneck is DRAM bandwidth, not compute               │
└──────────────────────────────────────────────────────────┘
```

---

## KV Cache Management

```
Turn 1: [SYS][USER: Hello][ASST: Hi there!]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ cached

Turn 2: [SYS][USER: Hello][ASST: Hi there!][USER: How are you?]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ prefix match → skip
                                             ^^^^^^^^^^^^^^^^^ new tokens only

Context full:
  → Automatic context shifting
  → Keep first N tokens (system prompt) + last M tokens
  → Discard middle tokens from KV cache
```

---

## Speculative Decoding (Ngram)

```
Draft phase:
  N-gram cache predicts next K tokens from patterns
  (no second model needed — self-speculation)

Verify phase:
  Batch-evaluate all K draft tokens in parallel
  Accept tokens that match model's actual distribution

Result:
  1.3-2x speedup for structured/repetitive output
  Falls back gracefully for creative text
```

---

## Memory Layout

```
Model file (GGUF)
  → mmap'd into address space (if use_mmap=true)
  → Weights accessed directly from file pages
  → OS manages paging (no full load into RAM)

KV Cache
  → Allocated at context creation time
  → Size: n_ctx × n_layer × 2 × n_embd × sizeof(type)
  → For 2048 ctx, ~64-256 MB depending on model

Scratch buffers
  → Temporary compute buffers, allocated per-batch
  → Freed between generations
```
