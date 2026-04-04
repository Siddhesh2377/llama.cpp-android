# Architecture

## Stack

```
Kotlin SDK (com.dark.gguf_lib)
  |
JNI Bridge (gguf_lib.cpp)
  |
Engine Layer (engine/)
  GGMLEngine  |  VLM Engine  |  ToolManager  |  RAG Engine
  |
llama.cpp Core (src/)
  Model loading, tokenization, inference, sampling, 100+ architectures
  |
Common Utilities (common/)
  Chat templates, JSON schema grammar, sampling chains, PEG parser
  |
GGML (ggml/)
  Tensor library, CPU backend only (NEON, i8mm, dotprod, fp16, bf16, KleidiAI)
```

---

## Directory Map

```
llama.cpp/
├── engine/                       Custom engine layer
│   ├── ggml-engine.h/.cpp        Model lifecycle, generation, KV cache, context tracking
│   ├── ggml-engine-vlm.cpp       VLM generation (text + images + audio)
│   ├── ggml-engine-internal.h    Shared structs and generation loop
│   ├── rag-engine.h/.cpp         RAG: late chunking, binary quantized search, retrieval
│   ├── tool-manager.h/.cpp       Tool call parsing (JSON/XML/function) and execution
│   ├── tn-log.h/.cpp             Logging utilities
│   ├── engine-utils.h            Shared engine helper functions
│   ├── vlm/                      Vision/audio encoder (mtmd library)
│   │   ├── clip.h/.cpp           CLIP/SigLIP vision encoder (CPU-only)
│   │   ├── clip-graph.h          Vision model compute graph definitions
│   │   ├── clip-model.h          Vision model struct definitions
│   │   ├── clip-impl.h           Internal implementation details
│   │   ├── mtmd.h/.cpp           Multimodal tokenizer and orchestration
│   │   ├── mtmd-helper.h/.cpp    Image/audio loading (stb_image, miniaudio)
│   │   ├── mtmd-audio.h/.cpp     Mel spectrogram, audio preprocessing
│   │   └── models/               18 VLM graph builders
│   │       ├── llava.cpp          LLaVA
│   │       ├── qwen2vl.cpp        Qwen2-VL
│   │       ├── qwen3vl.cpp        Qwen3-VL
│   │       ├── pixtral.cpp        Pixtral
│   │       ├── internvl.cpp       InternVL
│   │       ├── minicpmv.cpp       MiniCPM-V
│   │       ├── glm4v.cpp          GLM-4V
│   │       ├── cogvlm.cpp         CogVLM
│   │       ├── siglip.cpp         SigLIP
│   │       ├── llama4.cpp         Llama 4
│   │       ├── kimivl.cpp         Kimi-VL
│   │       ├── kimik25.cpp        Kimi-K2.5
│   │       ├── mobilenetv5.cpp    MobileNetV5
│   │       ├── nemotron-v2-vl.cpp Nemotron-V2-VL
│   │       ├── paddleocr.cpp      PaddleOCR
│   │       ├── conformer.cpp      Conformer (audio)
│   │       ├── whisper-enc.cpp    Whisper encoder (audio)
│   │       └── youtuvl.cpp        YouTu-VL
│   └── CMakeLists.txt            Builds libtn-engine.a
│
├── src/                          llama.cpp core
│   ├── llama.cpp                 Main implementation
│   ├── llama-*.cpp               Subsystems (vocab, sampling, context, mmap, etc.)
│   └── CMakeLists.txt            Builds libllama.a
│
├── include/                      Public C headers
│   ├── llama.h                   Core C API
│   └── llama-cpp.h               C++ convenience wrappers
│
├── ggml/                         GGML tensor library
│   ├── src/                      Tensor operations, CPU backend
│   │   ├── ggml.c                Core tensor library
│   │   ├── ggml-cpu/             CPU-specific kernels
│   │   │   ├── ggml-cpu-aarch64.cpp  ARM64 optimized paths
│   │   │   └── kleidiai/         KleidiAI ARM micro-kernels
│   │   └── ggml-threading.cpp    Thread pool
│   └── include/                  GGML headers
│       ├── ggml.h
│       ├── ggml-cpu.h
│       └── ggml-backend.h
│
├── common/                       Shared utilities
│   ├── common.h/.cpp             General utilities
│   ├── chat.h/.cpp               Chat template rendering
│   ├── chat-parser.h/.cpp        Chat output parsing
│   ├── chat-parser-xml-toolcall.h/.cpp  XML tool call parser
│   ├── chat-peg-parser.h/.cpp    PEG-based chat parser
│   ├── jinja/                    Jinja2-style template engine
│   ├── json-schema-to-grammar.h/.cpp  JSON schema to GBNF grammar
│   ├── json-partial.h/.cpp       Partial JSON parsing
│   ├── peg-parser.h/.cpp         Generic PEG parser
│   ├── regex-partial.h/.cpp      Partial regex matching
│   ├── sampling.h/.cpp           Sampling chain management
│   ├── log.h/.cpp                Logging
│   ├── unicode.h/.cpp            Unicode utilities
│   ├── llguidance.cpp            Grammar-guided generation
│   └── CMakeLists.txt            Builds libcommon.a
│
├── vendor/                       Third-party dependencies
│   ├── nlohmann/                 JSON library (json.hpp)
│   ├── stb/                      Image loading (stb_image)
│   └── miniaudio/                Audio decoding library
│
├── cmake/                        CMake modules
│   ├── build-info.cmake          Build metadata generation
│   ├── common.cmake              Shared CMake utilities
│   └── license.cmake             License embedding
│
├── CMakeLists.txt                Root build configuration
├── LICENSE                       MIT License
└── README.md                     Project overview
```

---

## Data Flow: Text Generation

```
User prompt (string)
    |
    v
GGMLEngine — ggml_engine_generate()

  1. Tokenize prompt via llama_tokenize()
  2. Check KV cache prefix (skip shared tokens)
  3. Batch-decode prompt tokens
  4. Auto-shift context if window full

  Generation loop:
    5. Sample next token (temp, top_k, top_p, min_p, penalties)
    6. Check stop sequences
    7. Detokenize token
    8. Invoke callback(token_text, user_data)
    9. If callback returns false, stop
   10. Decode token into KV cache
   11. Loop until n_predict or EOS

  Return perf metrics
    |
    v
  Generated text
```

---

## Data Flow: VLM Generation (Text + Image)

```
User prompt + image file(s)
    |
    v
VLM Engine — ggml_engine_vlm_generate()

  1. Load image bytes via stb_image decode into mtmd_bitmap
  2. mtmd_tokenize(prompt + bitmaps)
     - Split into text chunks + image chunks
     - Image markers "<__media__>" map to image positions
  3. Clear KV cache
  4. mtmd_helper_eval_chunks()
     - Text chunks: tokenize then batch decode
     - Image chunks: CLIP/SigLIP ViT encode then embedding injection
     - Handles M-RoPE (Qwen2-VL) and non-causal attention (Gemma3)
  5. Update n_past from processed chunks
  6. Shared generation loop (same as text generation)

  Return perf metrics (prompt_eval includes vision encode time)
```

---

## Data Flow: Tool Calling

```
User message + tool definitions
    |
    v
ToolManager

  1. tool_manager_get_prompt()
     - Generate tool description text
     - Inject into system/user prompt
  2. Engine generates response
  3. tool_manager_parse_output()
     - Try JSON parse: {"name": ..., "arguments": ...}
     - Try XML parse:  <tool_call>...</tool_call>
     - Try function parse: name(key=value)
     - Validate against registered tool definitions
     - Return tool_call_result
  4. tool_manager_execute()
     - Invoke registered callback
     - Return result string
  5. Feed result back to engine for next turn
```

---

## Data Flow: RAG (Retrieval-Augmented Generation)

```
Documents + embedding model
    |
    v
RAG Engine

  Indexing:
    1. Load embedding model (e.g. EmbeddingGemma-300M Q4)
    2. rag_engine_add_document(text, doc_id)
       - Tokenize document
       - Late chunking: encode full doc with bidirectional attention
       - Split token embeddings into chunks (256 tokens, 32 overlap)
       - Mean pool + Matryoshka truncate (768 to 256 dims)
       - L2 normalize to float embedding
       - Binary quantize to 1-bit BQ vector
       - Store both float + BQ per chunk

  Query:
    3. rag_engine_query(query)
       - Embed query (same pipeline)
       - Stage 1: BQ Hamming distance to top_k candidates (fast)
       - Stage 2: Cosine similarity re-rank to top_n results (accurate)
       - Return ranked rag_result array

  Prompt building:
    4. rag_engine_build_prompt(query, user_prompt)
       - Query and retrieve top results
       - Format: "Context:\n[chunk1]\n[chunk2]\n...\n\nuser_prompt"
       - Pass to GGMLEngine for generation
```

---

## Threading Model

On Android, the JNI bridge manages threading:

**Prompt processing (compute-bound):**
Uses n_threads_batch equal to all performance cores. CPU affinity pinned via sched_setaffinity. Example: 4 P-cores on Cortex-X3 means 4 threads.

**Token generation (memory-bound):**
Uses n_threads = min(4, P-cores). More threads increases cache contention. The bottleneck is DRAM bandwidth, not compute.

---

## KV Cache Management

```
Turn 1: [SYS][USER: Hello][ASST: Hi there!]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ cached

Turn 2: [SYS][USER: Hello][ASST: Hi there!][USER: How are you?]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ prefix match, skip
                                             ^^^^^^^^^^^^^^^^^ new tokens only

Context full:
  - Automatic context shifting
  - Keep first N tokens (system prompt) + last M tokens
  - Discard middle tokens from KV cache
```

---

## Memory Layout

**Model file (GGUF):**
Memory-mapped into address space when use_mmap is true. Weights are accessed directly from file pages. The OS manages paging so there is no full load into RAM.

**KV Cache:**
Allocated at context creation time. Size is n_ctx * n_layer * 2 * n_embd * sizeof(type). For a 2048 context window, typically 64-256 MB depending on model size.

**Scratch buffers:**
Temporary compute buffers allocated per batch. Freed between generations.
