// main.cpp — CLI entry point for gguf-engine
//
// Extracts the main() function and CLI help system from the monolith.
// Links against the gguf-engine SDK modules for all functionality.

#include "gguf-engine/engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Test functions (defined in tests/test_engine.cpp)
// ---------------------------------------------------------------------------
extern bool test_load(ModelState & state, const char * path, int fd,
                      ggml_backend_t backend);

extern bool test_single_layer(ModelState & state, ggml_backend_t backend);

extern bool test_full_forward(ModelState & state, ggml_backend_t backend);

extern bool test_decode(ModelState & state, ggml_backend_t backend,
                        int max_tokens, const SamplingParams & sp,
                        const std::vector<int32_t> & prompt_tokens);

extern bool test_hybrid_decode(ModelState & state, ggml_backend_t cpu_backend,
                               ggml_backend_t gpu_backend, int max_tokens);

extern bool test_character_engine(ModelState & state, ggml_backend_t backend,
                                  int max_tokens);

extern bool test_tool_calling(ModelState & state, ggml_backend_t backend,
                              int max_tokens, const char * json_path);

extern bool test_profile_system(ModelState & state, ggml_backend_t backend,
                                int max_tokens, const char * json_path);

extern bool test_async_boundaries(ModelState & state, ggml_backend_t backend,
                                  int max_tokens, const char * json_path);

extern bool test_rag(ModelState & state, ggml_backend_t backend,
                     int max_tokens, const char * rag_file,
                     const char * rag_query);

extern bool test_web_rag(ModelState & state, ggml_backend_t backend,
                         int max_tokens, const char * query);

extern bool test_web_fetch_rag(ModelState & state, ggml_backend_t backend,
                               int max_tokens, const char * url);

extern bool test_full_character_engine(ModelState & state, ggml_backend_t backend,
                                       int max_tokens, const char * json_path);

extern bool chat_bench(ModelState & state, ggml_backend_t backend,
                       int max_tokens);

// ---------------------------------------------------------------------------
// Helper: set CPU thread count via backend registry (dynamic backend safe)
// ---------------------------------------------------------------------------
static void set_cpu_threads(ggml_backend_t backend, int n_threads) {
    auto * dev = ggml_backend_get_device(backend);
    if (!dev) return;
    auto * reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return;
    typedef void (*set_n_threads_fn_t)(ggml_backend_t, int);
    auto fn = (set_n_threads_fn_t)ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_cpu_set_n_threads");
    if (fn) fn(backend, n_threads);
}

// ---------------------------------------------------------------------------
// Help / usage
// ---------------------------------------------------------------------------
void print_version() {
    printf("%s v%s — Native AI Character Engine\n", ENGINE_NAME, ENGINE_VERSION);
    printf("Built with ggml (standalone, no llama.cpp runtime)\n");
}

void print_help(const char * prog) {
    print_version();
    printf("\nUSAGE\n");
    printf("  %s <model.gguf> [MODE] [OPTIONS]\n\n", prog);

    printf("MODES (pick one)\n");
    printf("  --char-chat              Interactive chat with full character engine\n");
    printf("  --char-engine-test       Run all character engine subsystem tests\n");
    printf("  --chat                   Interactive chat (basic prompt, no character)\n");
    printf("  --chat-skills            Interactive chat with skills-enhanced prompt\n");
    printf("  --chat-bench             Automated skills benchmark (3 tiers x 8 questions)\n");
    printf("  --web-search \"QUERY\"     Search the web + generate RAG answer\n");
    printf("  --web-fetch URL          Fetch a webpage + RAG summarize\n");
    printf("  --tool-test              Grammar-constrained tool calling test\n");
    printf("  --profile-test           Profile state system test\n");
    printf("  --boundary-test          Async boundary system test\n");
    printf("  --rag-test               Offline RAG system test\n");
    printf("  --vlm-test               Vision-language model test\n");
    printf("  (none)                   Full benchmark suite (forward, decode, hybrid)\n");

    printf("\nCHARACTER OPTIONS\n");
    printf("  --ch-json FILE           Character personality JSON file\n");
    printf("                           Defines name, system prompt, mood baselines,\n");
    printf("                           temperature profiles, control vectors, and more.\n");
    printf("                           See aria.json for reference format.\n");

    printf("\nMODEL OPTIONS\n");
    printf("  --tokens N               Max tokens to generate (default: auto per mode)\n");
    printf("  --threads N              CPU threads (default: 4, recommended: 3-4)\n");
    printf("  --gpu                    Enable hybrid CPU/GPU compute\n");
    printf("  --fd N                   Load model from file descriptor (Android SAF)\n");

    printf("\nSAMPLING OPTIONS\n");
    printf("  --temp F                 Temperature (0 = greedy, default: 0)\n");
    printf("  --top-k N               Top-K sampling (default: 40)\n");
    printf("  --top-p F               Nucleus sampling threshold (default: 0.95)\n");
    printf("  --rep-penalty F          Repetition penalty (default: 1.0)\n");

    printf("\nRAG / VLM OPTIONS\n");
    printf("  --rag-file FILE          Index a text file for RAG retrieval\n");
    printf("  --rag-query TEXT         Custom query for RAG test\n");
    printf("  --mmproj FILE            Vision projector GGUF for --vlm-test\n");
    printf("  --prompt TEXT            Custom prompt for raw decode mode\n");

    printf("\nINFO\n");
    printf("  -h, --help               Show this help message\n");
    printf("  -v, --version            Show version\n");

    printf("\nEXAMPLES\n");
    printf("  # Chat with Aria (28yo creative technologist character)\n");
    printf("  %s model.gguf --char-chat --ch-json aria.json --threads 4\n\n", prog);
    printf("  # Run character engine tests\n");
    printf("  %s model.gguf --char-engine-test --ch-json aria.json\n\n", prog);
    printf("  # Interactive chat with web search tool calling\n");
    printf("  %s model.gguf --chat --tokens 256 --temp 0.8\n\n", prog);
    printf("  # Search the web and get an AI answer\n");
    printf("  %s model.gguf --web-search \"latest news about AI\"\n\n", prog);
    printf("  # Raw benchmark (greedy decode, no chat)\n");
    printf("  %s model.gguf --tokens 64 --threads 4\n\n", prog);
    printf("  # VLM: image understanding\n");
    printf("  %s model.gguf --vlm-test --mmproj projector.gguf\n\n", prog);

    printf("CHARACTER JSON FORMAT (aria.json)\n");
    printf("  {\n");
    printf("    \"name\": \"Aria\",\n");
    printf("    \"system_prompt\": \"You are Aria, a 28-year-old ...\",\n");
    printf("    \"temp_early\": 1.30,  \"temp_mid\": 1.00,  \"temp_late\": 0.80,\n");
    printf("    \"attn_gate_mid\": 0.90, \"ffn_gate_mid\": 0.93,\n");
    printf("    \"logit_bias_eos\": -4.0,\n");
    printf("    \"sampling_temp\": 0.85, \"sampling_top_k\": 50, \"sampling_top_p\": 0.93,\n");
    printf("    \"rep_penalty\": 1.20,\n");
    printf("    \"mood_warmth\": 0.72, \"mood_energy\": 0.60, \"mood_formality\": 0.28,\n");
    printf("    \"stall_prompt\": \"Hmm let me look that up\",\n");
    printf("    \"fw_enabled\": 1\n");
    printf("  }\n\n");

    printf("CHARACTER ENGINE FEATURES\n");
    printf("  - 3-band attention temperature (early/mid/late layers)\n");
    printf("  - Gated residuals (attention + FFN dampening in middle layers)\n");
    printf("  - Control vector steering (per-layer activation addition from .gguf)\n");
    printf("  - Adaptive head rescaling (importance-weighted attention heads)\n");
    printf("  - Emotional state tracking (keyword-based mood -> intervention offset)\n");
    printf("  - Fast weight associative memory (Hebbian, ~50us/token overhead)\n");
    printf("  - Stall generation during async tool calls (natural filler text)\n");
    printf("  - Grammar-constrained tool calling (JSON schema enforcement)\n");
    printf("  - Web search + RAG pipeline (DuckDuckGo + chunked retrieval)\n");
    printf("  - Role name replacement (character name in ChatML template)\n");
}

bool is_help_flag(const char * arg) {
    return strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0
        || strcmp(arg, "-help") == 0 || strcmp(arg, "--h") == 0;
}

bool is_version_flag(const char * arg) {
    return strcmp(arg, "-v") == 0 || strcmp(arg, "--version") == 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    // Check for help/version before anything else
    for (int i = 1; i < argc; i++) {
        if (is_help_flag(argv[i])) { print_help(argv[0]); return 0; }
        if (is_version_flag(argv[i])) { print_version(); return 0; }
    }

    // Load .config/config.json if present
    EngineConfig engine_config;
    if (load_engine_config(engine_config, ".config/config.json")) {
        printf("  Loaded .config/config.json\n");
    }

    const char * model_path = engine_config.model_path.empty() ? nullptr : engine_config.model_path.c_str();
    int model_fd = -1;
    bool use_gpu = engine_config.gpu;
    int max_tokens = engine_config.max_tokens > 32 ? engine_config.max_tokens : 32;
    int n_threads = engine_config.threads;
    SamplingParams sp;
    sp.temp = engine_config.temp;
    sp.top_k = engine_config.top_k;
    sp.top_p = engine_config.top_p;
    sp.rep_penalty = engine_config.rep_penalty;
    tui::color_enabled = engine_config.color;
    const char * prompt = nullptr;
    const char * ch_json = engine_config.character_json.empty() ? nullptr : engine_config.character_json.c_str();
    bool tool_test = false;
    bool profile_test = false;
    bool boundary_test = false;
    bool vlm_test = false;
    const char * mmproj_path = nullptr;
    bool rag_test = false;
    const char * rag_file = nullptr;
    const char * rag_query = nullptr;
    const char * web_search_query = nullptr;
    const char * web_fetch_url = nullptr;
    bool chat_mode = false;
    bool chat_skills = false;
    bool chat_bench_mode = false;
    bool char_engine_test = false;
    bool char_chat_mode = false;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (is_help_flag(argv[i]) || is_version_flag(argv[i])) {
            continue; // already handled
        } else if (strcmp(argv[i], "--fd") == 0 && i + 1 < argc) {
            model_fd = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            sp.temp = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            sp.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            sp.top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            sp.rep_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--ch-json") == 0 && i + 1 < argc) {
            ch_json = argv[++i];
        } else if (strcmp(argv[i], "--tool-test") == 0) {
            tool_test = true;
        } else if (strcmp(argv[i], "--profile-test") == 0) {
            profile_test = true;
        } else if (strcmp(argv[i], "--boundary-test") == 0) {
            boundary_test = true;
        } else if (strcmp(argv[i], "--vlm-test") == 0) {
            vlm_test = true;
        } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (strcmp(argv[i], "--rag-test") == 0) {
            rag_test = true;
        } else if (strcmp(argv[i], "--rag-file") == 0 && i + 1 < argc) {
            rag_file = argv[++i];
        } else if (strcmp(argv[i], "--rag-query") == 0 && i + 1 < argc) {
            rag_query = argv[++i];
        } else if (strcmp(argv[i], "--web-search") == 0 && i + 1 < argc) {
            web_search_query = argv[++i];
        } else if (strcmp(argv[i], "--web-fetch") == 0 && i + 1 < argc) {
            web_fetch_url = argv[++i];
        } else if (strcmp(argv[i], "--chat") == 0) {
            chat_mode = true;
        } else if (strcmp(argv[i], "--chat-skills") == 0) {
            chat_skills = true;
        } else if (strcmp(argv[i], "--chat-bench") == 0) {
            chat_bench_mode = true;
        } else if (strcmp(argv[i], "--char-engine-test") == 0) {
            char_engine_test = true;
        } else if (strcmp(argv[i], "--char-chat") == 0) {
            char_chat_mode = true;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            fprintf(stderr, "Run '%s --help' for usage information.\n", argv[0]);
            return 1;
        } else {
            model_path = argv[i];
        }
    }

    if (!model_path && model_fd < 0) {
        fprintf(stderr, "Error: no model specified.\n\n");
        fprintf(stderr, "Usage: %s <model.gguf> [MODE] [OPTIONS]\n", argv[0]);
        fprintf(stderr, "Run '%s --help' for full usage information.\n", argv[0]);
        return 1;
    }

    print_version();
    printf("-------------------------------------\n");
    if (model_path) printf("  Model:   %s\n", model_path);
    if (model_fd >= 0) printf("  Model:   fd=%d\n", model_fd);
    printf("  Threads: %d   GPU: %s   Tokens: %d\n", n_threads, use_gpu ? "yes" : "no", max_tokens);
    if (sp.temp > 0) {
        printf("  Sampling: temp=%.2f top_k=%d top_p=%.2f rep=%.2f\n",
            sp.temp, sp.top_k, sp.top_p, sp.rep_penalty);
    } else {
        printf("  Sampling: greedy (argmax)\n");
    }
    if (ch_json) printf("  Character: %s\n", ch_json);
    printf("-------------------------------------\n");

    // Init backends
    ggml_backend_load_all();

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        printf("FAIL: no CPU backend\n");
        return 1;
    }
    set_cpu_threads(cpu_backend, n_threads);
    printf("CPU backend: %s (%d threads)\n", ggml_backend_name(cpu_backend), n_threads);

    ggml_backend_t gpu_backend = nullptr;
    if (use_gpu) {
        gpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (gpu_backend) {
            printf("GPU backend: %s\n", ggml_backend_name(gpu_backend));
        } else {
            printf("GPU backend: not available\n");
            use_gpu = false;
        }
    }

    // Use CPU for weights + compute (GPU for hybrid test only)
    ggml_backend_t primary_backend = cpu_backend;

    ModelState state = {};
    std::vector<int32_t> prompt_tokens;
    int pass = 0, fail = 0;

    // Test A: Load model
    if (test_load(state, model_path, model_fd, primary_backend)) pass++; else { fail++; goto cleanup; }

    if (char_engine_test) {
        // Full character engine test
        if (max_tokens == 32) max_tokens = 128;
        if (test_full_character_engine(state, primary_backend, max_tokens, ch_json)) pass++; else fail++;
    } else if (char_chat_mode) {
        // Interactive character chat (all systems active)
        if (max_tokens == 32) max_tokens = 256;
        if (model_path) engine_config.model_path = model_path;
        if (ch_json) engine_config.character_json = ch_json;
        if (interactive_character_chat(state, primary_backend, max_tokens, ch_json, engine_config)) pass++; else fail++;
    } else if (chat_bench_mode) {
        // Skills benchmark
        if (max_tokens == 32) max_tokens = 256;
        if (chat_bench(state, primary_backend, max_tokens)) pass++; else fail++;
    } else if (chat_mode || chat_skills) {
        // Interactive chat
        if (max_tokens == 32) max_tokens = 256;
        int tier = chat_skills ? 2 : 1;
        if (interactive_chat(state, primary_backend, max_tokens, tier)) pass++; else fail++;
    } else if (web_search_query) {
        // Web search + RAG
        if (max_tokens == 32) max_tokens = 128;
        if (test_web_rag(state, primary_backend, max_tokens, web_search_query)) pass++; else fail++;
    } else if (web_fetch_url) {
        // Web fetch + RAG
        if (max_tokens == 32) max_tokens = 128;
        if (test_web_fetch_rag(state, primary_backend, max_tokens, web_fetch_url)) pass++; else fail++;
    } else if (rag_test) {
        // RAG system test
        if (max_tokens == 32) max_tokens = 64;
        if (test_rag(state, primary_backend, max_tokens, rag_file, rag_query)) pass++; else fail++;
    } else if (vlm_test) {
        // VLM test — vision encoder + LLM decode (TODO: extract test)
        printf("  VLM test not yet extracted to modular SDK. Use monolith.\n");
        fail++;
    } else if (boundary_test) {
        // Async boundary system test
        if (max_tokens == 32) max_tokens = 256;
        if (test_async_boundaries(state, primary_backend, max_tokens, ch_json)) pass++; else fail++;
    } else if (profile_test) {
        // Profile state system test
        if (max_tokens == 32) max_tokens = 64;
        if (test_profile_system(state, primary_backend, max_tokens, ch_json)) pass++; else fail++;
    } else if (tool_test) {
        // Tool calling test mode
        if (max_tokens == 32) max_tokens = 128; // default longer for tool calling
        if (test_tool_calling(state, primary_backend, max_tokens, ch_json)) pass++; else fail++;
    } else if (ch_json) {
        // Character chat with JSON but no specific mode — run interactive
        if (max_tokens == 32) max_tokens = 256;
        if (model_path) engine_config.model_path = model_path;
        engine_config.character_json = ch_json;
        if (interactive_character_chat(state, primary_backend, max_tokens, ch_json, engine_config)) pass++; else fail++;
    } else {
        // Full test suite mode
        // Tokenize prompt if provided (must be after model load for vocab)
        if (prompt && state.vocab.size() > 0) {
            prompt_tokens.clear();
            auto toks = tokenize_simple(state.vocab, prompt);
            for (int t : toks) prompt_tokens.push_back((int32_t)t);
            printf("\n  Tokenized prompt: %zu tokens", prompt_tokens.size());
            if (!prompt_tokens.empty()) {
                printf(" [");
                for (int i = 0; i < std::min((int)prompt_tokens.size(), 8); i++) {
                    if (i > 0) printf(", ");
                    printf("%d", prompt_tokens[i]);
                }
                if ((int)prompt_tokens.size() > 8) printf(", ...");
                printf("]");
            }
            printf("\n");
        }

        // Test E: Single layer
        if (test_single_layer(state, primary_backend)) pass++; else fail++;

        // Test F: Full forward
        if (test_full_forward(state, primary_backend)) pass++; else fail++;

        // Test G: Autoregressive decode (CPU)
        if (test_decode(state, primary_backend, max_tokens, sp, prompt_tokens)) pass++; else fail++;

        // Test I: Character Intelligence Engine v2
        if (test_character_engine(state, primary_backend, max_tokens)) pass++; else fail++;

        // Test H: Hybrid CPU/GPU decode
        if (use_gpu && gpu_backend) {
            if (test_hybrid_decode(state, cpu_backend, gpu_backend, max_tokens)) pass++; else fail++;
        }
    }

cleanup:
    printf("\n=====================================\n");
    printf("Results: %d passed, %d failed\n", pass, fail);

    // Cleanup
    if (state.kv_buf) ggml_backend_buffer_free(state.kv_buf);
    if (state.kv_ctx) ggml_free(state.kv_ctx);
    if (state.weight_buf) ggml_backend_buffer_free(state.weight_buf);
    if (state.weight_ctx) ggml_free(state.weight_ctx);
    if (state.data_ctx) ggml_free(state.data_ctx);
    if (state.gguf_ctx) gguf_free(state.gguf_ctx);
    if (gpu_backend) ggml_backend_free(gpu_backend);
    ggml_backend_free(cpu_backend);

    return fail > 0 ? 1 : 0;
}
