// chat.cpp — Interactive chat modes
//
// Extracted from gguf-forward-test.cpp (lines ~7812-9374).
// Contains:
//   - run_chat_turn():               single prefill+decode turn with tool calling
//   - interactive_chat():            basic interactive chat with tiered prompts
//   - handle_chat_command():         slash command processing
//   - interactive_character_chat():  full character engine loop
//
// Local helpers (not exported):
//   - execute_web_search_tool():     DuckDuckGo search via ToolCallResult
//   - prompt tier constants

#include "gguf-engine/chat.h"
#include "gguf-engine/engine.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// System prompt tiers
// ---------------------------------------------------------------------------
const char * PROMPT_TIER0 = "You are a helpful assistant.";

const char * PROMPT_TIER1 = "You are a helpful assistant with web search. "
    "Use the web_search tool when you need current information.";

const char * PROMPT_TIER2 =
    "You are a precise AI assistant with web search capability. Follow these rules strictly:\n\n"
    "## When to Search\n"
    "- ALWAYS search for: current events, versions, dates, statistics, people, companies, recent news\n"
    "- ALWAYS search for: medical/health questions, legal questions, scientific claims\n"
    "- ALWAYS search for: code library versions, API documentation, error messages\n"
    "- NEVER search for: greetings, opinions, math, logic puzzles, creative writing\n\n"
    "## How to Answer\n"
    "- Search FIRST, then synthesize a concise answer (1-3 sentences)\n"
    "- Say \"According to search results, ...\" when citing web data\n"
    "- If search returns nothing useful, say \"I couldn't find current information on that\"\n"
    "- For code questions: search for docs/examples, then write code based on findings\n\n"
    "## Format\n"
    "- Use plain text, be concise\n"
    "- Never make up facts, dates, or statistics";

// ---------------------------------------------------------------------------
// Local helper: execute web_search tool via real DuckDuckGo search
// ---------------------------------------------------------------------------
static std::string execute_web_search_tool(const ToolCallResult & call) {
    std::string query = extract_json_string(call.arguments_json, "query");
    if (query.empty()) return "{\"error\": \"missing query parameter\"}";

    int num_results = 3;
    std::string nr_str = extract_json_string(call.arguments_json, "num_results");
    if (!nr_str.empty()) num_results = std::max(1, std::min(5, atoi(nr_str.c_str())));

    // Use the RAG module's web search
    std::string result = execute_web_search(query);
    if (result.empty()) {
        return "{\"error\": \"search failed\"}";
    }
    return result;
}

// ---------------------------------------------------------------------------
// Local helper: build a single web_search ToolDef
// ---------------------------------------------------------------------------
static std::vector<ToolDef> make_web_search_tools() {
    std::vector<ToolDef> tools(1);
    tools[0].name = "web_search";
    tools[0].description = "Search the web for current information";

    ToolParam p0;
    p0.name = "query";
    p0.type = "string";
    p0.description = "The search query";
    p0.required = true;
    tools[0].params.push_back(p0);

    ToolParam p1;
    p1.name = "num_results";
    p1.type = "integer";
    p1.description = "Number of results (1-5)";
    p1.required = false;
    tools[0].params.push_back(p1);

    return tools;
}

// ==========================================================================
// run_chat_turn — single prefill+decode turn with grammar-constrained
//                 tool calling and multi-round tool execution
// ==========================================================================
ChatTurnStats run_chat_turn(
    ModelState & state, ggml_backend_t backend, int max_tokens,
    GrammarEngine & grammar, ggml_gallocr_t galloc,
    InterventionConfig & iv, InterventionTensors & iv_t,
    const SamplingParams & sp, std::vector<float> & logits_buf,
    size_t ctx_size, std::vector<uint8_t> & ctx_buf,
    std::vector<uint16_t> & mask,
    const std::vector<int32_t> & turn_tokens,
    const std::string & assistant_name)
{
    const ModelConfig & cfg = state.cfg;
    const int n_layer = (int)cfg.n_layer;
    const int n_vocab = (int)cfg.n_vocab;
    bool need_argmax = (sp.temp <= 0.0f);
    std::mt19937 rng((uint32_t)state.kv_pos); // seed from position for variety

    ChatTurnStats stats = {};

    // Find stop tokens
    int32_t im_end_id = -1, im_start_id = -1;
    for (int id = 0; id < (int)state.vocab.size(); id++) {
        if (state.vocab[id] == "<|im_end|>") im_end_id = id;
        if (state.vocab[id] == "<|im_start|>") im_start_id = id;
    }

    // --- Prefill turn tokens ---
    auto prefill_tokens = [&](const std::vector<int32_t> & tokens) -> bool {
        int n = (int)tokens.size();
        int processed = 0;
        while (processed < n) {
            int chunk = std::min(PREFILL_CHUNK, n - processed);
            int kv_pos = state.kv_pos;
            int kv_len = kv_pos + chunk;
            if (kv_len > (int)cfg.max_ctx) return false;

            struct ggml_init_params p = { ctx_size, ctx_buf.data(), true };
            struct ggml_context * ctx = ggml_init(p);
            bool last = (processed + chunk >= n);
            struct ggml_cgraph * g = build_graph(ctx, state, chunk, kv_pos, kv_len,
                n_layer, last ? need_argmax : false, &iv, &iv_t);
            if (!ggml_gallocr_alloc_graph(galloc, g)) { ggml_free(ctx); return false; }

            ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_tokens"),
                &tokens[processed], 0, chunk * sizeof(int32_t));
            std::vector<int32_t> pos(chunk);
            for (int i = 0; i < chunk; i++) pos[i] = kv_pos + i;
            ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_pos"),
                pos.data(), 0, chunk * sizeof(int32_t));
            build_causal_mask(mask, kv_len, chunk, kv_pos);
            ggml_backend_tensor_set(ggml_graph_get_tensor(g, "attn_mask"),
                mask.data(), 0, (size_t)kv_len * chunk * sizeof(uint16_t));

            ggml_backend_graph_compute(backend, g);
            ggml_backend_synchronize(backend);

            if (last && !need_argmax) {
                size_t off = (size_t)(chunk - 1) * n_vocab * sizeof(float);
                ggml_backend_tensor_get(ggml_graph_get_tensor(g, "logits"),
                    logits_buf.data(), off, n_vocab * sizeof(float));
            }
            state.kv_pos = kv_len;
            processed += chunk;
            ggml_free(ctx);
        }
        return true;
    };

    // --- Decode one token ---
    auto decode_one = [&](int32_t input_token) -> int32_t {
        int kv_pos = state.kv_pos;
        int kv_len = kv_pos + 1;
        if (kv_len > (int)cfg.max_ctx) return -1;

        struct ggml_init_params p = { ctx_size, ctx_buf.data(), true };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_cgraph * g = build_graph(ctx, state, 1, kv_pos, kv_len,
            n_layer, need_argmax, &iv, &iv_t);
        if (!ggml_gallocr_alloc_graph(galloc, g)) { ggml_free(ctx); return -1; }

        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_tokens"),
            &input_token, 0, sizeof(int32_t));
        int32_t pos = kv_pos;
        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_pos"),
            &pos, 0, sizeof(int32_t));
        std::fill(mask.begin(), mask.begin() + kv_len, (uint16_t)0);
        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "attn_mask"),
            mask.data(), 0, kv_len * sizeof(uint16_t));

        ggml_backend_graph_compute(backend, g);
        ggml_backend_synchronize(backend);

        ggml_backend_tensor_get(ggml_graph_get_tensor(g, "logits"),
            logits_buf.data(), 0, n_vocab * sizeof(float));

        if (sp.rep_penalty != 1.0f) {
            if (logits_buf[input_token] > 0) logits_buf[input_token] /= sp.rep_penalty;
            else logits_buf[input_token] *= sp.rep_penalty;
        }

        grammar.apply_mask(logits_buf.data(), state.vocab);

        int32_t tid = sample_token(logits_buf.data(), n_vocab, sp, rng);
        state.kv_pos = kv_len;
        ggml_free(ctx);
        return tid;
    };

    // Prefill
    auto t_pf0 = Clock::now();
    if (!prefill_tokens(turn_tokens)) {
        printf("[prefill failed]\n");
        return stats;
    }
    auto t_pf1 = Clock::now();
    stats.prefill_tokens = (int)turn_tokens.size();
    stats.prefill_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();

    int32_t last_token = sample_token(logits_buf.data(), n_vocab, sp, rng);

    // Decode loop with tool calling
    int max_rounds = 3;
    for (int round = 0; round < max_rounds; round++) {
        // Print + advance first token
        if (last_token >= 0 && last_token < (int)state.vocab.size()
            && last_token != state.eos_token && last_token != im_end_id
            && last_token != im_start_id) {
            std::string ts = decode_token(state.vocab[last_token]);
            printf("%s", ts.c_str());
            fflush(stdout);
            stats.response_text += ts;
            grammar.advance(last_token, state.vocab[last_token]);
        }

        for (int t = 0; t < max_tokens; t++) {
            if (grammar.is_tool_call_ready()) break;
            if (last_token == state.eos_token || last_token == im_start_id) break;
            if (last_token == im_end_id && !grammar.is_active()) break;

            auto t0 = Clock::now();
            int32_t tid = decode_one(last_token);
            auto t1 = Clock::now();
            stats.decode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (tid < 0) break;

            if (tid < (int)state.vocab.size()) {
                std::string ts = decode_token(state.vocab[tid]);
                printf("%s", ts.c_str());
                fflush(stdout);
                stats.response_text += ts;
                grammar.advance(tid, state.vocab[tid]);
            }
            last_token = tid;
            stats.decode_tokens++;
        }

        // Tool call handling
        if (grammar.is_tool_call_ready()) {
            ToolCallResult tcr = parse_tool_call_json(grammar.detector.json_buffer);
            if (!tcr.valid) break;

            stats.tool_calls++;
            printf("\n  [TOOL: %s(%s)]\n", tcr.name.c_str(), tcr.arguments_json.c_str());

            // Execute real web search
            auto t_search0 = Clock::now();
            std::string result;
            if (tcr.name == "web_search") {
                result = execute_web_search_tool(tcr);
                stats.used_search = true;
            } else {
                result = "{\"error\": \"unknown tool: " + tcr.name + "\"}";
            }
            auto t_search1 = Clock::now();
            stats.search_ms += std::chrono::duration<double, std::milli>(t_search1 - t_search0).count();

            // Truncate result if too long (save context space)
            if (result.size() > 800) result = result.substr(0, 800) + "...]}";

            printf("  [Result: %zu chars, %.0fms]\n  ", result.size(), stats.search_ms);
            fflush(stdout);

            auto result_tokens = build_tool_result_tokens(state.vocab, result, assistant_name);
            prefill_tokens(result_tokens);
            last_token = sample_token(logits_buf.data(), n_vocab, sp, rng);
            grammar.reset_for_next_round();
            continue;
        }

        break; // normal stop
    }

    stats.grammar_constrained = grammar.tokens_constrained;
    printf("\n");
    return stats;
}

// ==========================================================================
// interactive_chat — basic interactive chat with tiered system prompts
// ==========================================================================
bool interactive_chat(ModelState & state, ggml_backend_t backend,
    int max_tokens, int prompt_tier)
{
    printf("\n========================================\n");
    printf("Interactive Chat (Tier %d prompt)\n", prompt_tier);
    printf("========================================\n");

    const ModelConfig & cfg = state.cfg;
    const int n_layer = (int)cfg.n_layer;
    const int n_vocab = (int)cfg.n_vocab;

    // Define web_search tool
    auto tools = make_web_search_tools();

    // Select system prompt
    const char * base_prompt = (prompt_tier == 0) ? PROMPT_TIER0 :
                               (prompt_tier == 1) ? PROMPT_TIER1 : PROMPT_TIER2;
    std::string tool_system = build_tool_system_prompt(base_prompt, tools);

    // Interventions
    InterventionTensors iv_t;
    if (!init_interventions(iv_t, cfg, backend)) { printf("FAIL: interventions\n"); return false; }

    InterventionConfig iv;
    iv.reset();
    iv.flags = IV_ATTN_TEMPERATURE | IV_GATED_RESIDUAL | IV_LOGIT_BIAS;
    for (int il = 0; il < n_layer; il++) {
        float t = (float)il / (float)std::max(n_layer - 1, 1);
        if (t < 0.33f) iv.attn_temp[il] = 1.2f;
        else if (t < 0.66f) iv.attn_temp[il] = 1.0f;
        else iv.attn_temp[il] = 0.85f;
    }
    for (int il = 0; il < n_layer; il++) {
        float t = (float)il / (float)std::max(n_layer - 1, 1);
        if (t > 0.25f && t < 0.75f) { iv.attn_gate[il] = 0.92f; iv.ffn_gate[il] = 0.95f; }
        else { iv.attn_gate[il] = 1.0f; iv.ffn_gate[il] = 1.0f; }
    }
    std::vector<float> bias(n_vocab, 0.0f);
    bias[state.eos_token] = -3.0f;
    // Think suppression handled by /no_think prompt syntax (more reliable than logit bias)
    ggml_backend_tensor_set(iv_t.logit_bias, bias.data(), 0, n_vocab * sizeof(float));

    // Grammar
    GrammarEngine grammar;
    grammar.init(state.vocab);

    // Sampling
    SamplingParams sp;
    sp.temp = 0.7f; sp.top_k = 40; sp.top_p = 0.9f; sp.rep_penalty = 1.1f;
    bool need_argmax = false;

    // Allocator
    ggml_backend_buffer_clear(state.kv_buf, 0);
    state.kv_pos = 0;

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    {
        size_t csz = compute_ctx_size(n_layer, true);
        struct ggml_init_params p = { csz, nullptr, true };
        struct ggml_context * mctx = ggml_init(p);
        struct ggml_cgraph * mg = build_graph(mctx, state, PREFILL_CHUNK, 0,
            (int)cfg.max_ctx, n_layer, need_argmax, &iv, &iv_t);
        ggml_gallocr_reserve(galloc, mg);
        ggml_free(mctx);
    }

    size_t ctx_size = compute_ctx_size(n_layer, true);
    std::vector<uint8_t> ctx_buf(ctx_size);
    std::vector<uint16_t> mask((size_t)cfg.max_ctx * PREFILL_CHUNK, 0);
    std::vector<float> logits_buf(n_vocab);

    // First turn: read input, build full ChatML, prefill
    printf("\nType your message (or /quit to exit):\n\n");
    char input[2048];
    bool first_turn = true;
    int total_tool_calls = 0;
    int total_turns = 0;

    while (true) {
        printf("You: ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        // Strip newline
        size_t len = strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r')) input[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(input, "/quit") == 0) break;

        std::string user_input(input);
        // Append /no_think for Qwen3
        user_input += " /no_think";

        std::vector<int32_t> turn_tokens;
        if (first_turn) {
            turn_tokens = build_chat_tokens(state.vocab, tool_system, user_input);
            first_turn = false;
        } else {
            turn_tokens = build_turn_tokens(state.vocab, user_input);
        }

        grammar.detector.reset();
        grammar.grammar.reset();
        grammar.tokens_constrained = 0;

        printf("Assistant: ");
        fflush(stdout);

        auto stats = run_chat_turn(state, backend, max_tokens,
            grammar, galloc, iv, iv_t, sp, logits_buf,
            ctx_size, ctx_buf, mask, turn_tokens);

        total_tool_calls += stats.tool_calls;
        total_turns++;

        printf("[Stats] prefill: %d tok %.0fms | decode: %d tok @ %.0fms/tok | "
               "tools: %d (search: %.0fms) | grammar: %d constrained | kv: %d/%d\n\n",
            stats.prefill_tokens, stats.prefill_ms,
            stats.decode_tokens, stats.decode_tokens > 0 ? stats.decode_ms / stats.decode_tokens : 0.0,
            stats.tool_calls, stats.search_ms,
            stats.grammar_constrained,
            state.kv_pos, (int)cfg.max_ctx);

        if (state.kv_pos > (int)cfg.max_ctx - 200) {
            printf("[Context nearly full (%d/%d), resetting]\n", state.kv_pos, (int)cfg.max_ctx);
            ggml_backend_buffer_clear(state.kv_buf, 0);
            state.kv_pos = 0;
            first_turn = true;
        }
    }

    printf("\n=== Session Summary ===\n");
    printf("  Turns: %d | Tool calls: %d\n", total_turns, total_tool_calls);

    ggml_gallocr_free(galloc);
    free_interventions(iv_t);
    return true;
}

// ==========================================================================
// handle_chat_command — /command processing for interactive chat modes
// ==========================================================================
bool handle_chat_command(
    const char * input, EngineConfig & ec, SamplingParams & sp,
    EmotionalState & mood, ModelState & state, PersonalityConfig & pc,
    int & max_tokens)
{
    if (input[0] != '/') return false;
    std::string cmd(input + 1); std::string arg;
    size_t eq = cmd.find('='), sp2 = cmd.find(' ');
    size_t sep = std::min(eq, sp2);
    if (sep != std::string::npos) { arg = cmd.substr(sep + 1); cmd = cmd.substr(0, sep); }
    for (auto & ch : cmd) ch = tolower(ch);

    if (cmd == "quit" || cmd == "exit" || cmd == "q") return false;
    else if (cmd == "help" || cmd == "h" || cmd == "?") {
        tui::banner("Chat Commands");
        printf("  /help              Show this help\n");
        printf("  /config            Show current configuration\n");
        printf("  /save-config       Save config to .config/config.json\n");
        printf("  /threads N         Set thread count (1-8)\n");
        printf("  /gpu N             Set GPU (0=off, 1=on)\n");
        printf("  /temp F            Set temperature (0.0-2.0)\n");
        printf("  /top-k N           Set top-k (1-200)\n");
        printf("  /top-p F           Set top-p (0.0-1.0)\n");
        printf("  /rep-penalty F     Set repetition penalty (1.0-2.0)\n");
        printf("  /tokens N          Set max tokens per reply\n");
        printf("  /mood              Show emotional state\n");
        printf("  /reset             Reset conversation\n");
        printf("  /download-model    Download a GGUF model\n");
        printf("  /quit              Exit\n\n");
    }
    else if (cmd == "config") {
        tui::banner("Current Configuration");
        tui::status("Model", ec.model_path.c_str());
        tui::status("Character", ec.character_json.c_str());
        tui::status_int("Threads", ec.threads);
        tui::status("GPU", ec.gpu ? "on" : "off");
        tui::status_int("Max tokens", max_tokens);
        tui::status_float("Temperature", sp.temp);
        tui::status_int("Top-K", sp.top_k);
        tui::status_float("Top-P", sp.top_p);
        tui::status_float("Rep penalty", sp.rep_penalty);
        printf("  %-16s%d / %d\n\n", "Context", state.kv_pos, (int)state.cfg.max_ctx);
    }
    else if (cmd == "save-config") {
        ec.temp = sp.temp; ec.top_k = sp.top_k; ec.top_p = sp.top_p;
        ec.rep_penalty = sp.rep_penalty; ec.max_tokens = max_tokens;
        // Ensure .config/ directory exists
        mkdir(".config", 0755);
        if (save_engine_config(ec, ".config/config.json")) tui::success("Config saved to .config/config.json");
        else tui::error("Failed to save .config/config.json");
    }
    else if (cmd == "threads" && !arg.empty()) {
        int n = atoi(arg.c_str());
        if (n >= 1 && n <= 8) { ec.threads = n; tui::success("Threads set"); }
        else tui::error("Threads must be 1-8");
    }
    else if (cmd == "gpu" && !arg.empty()) {
        ec.gpu = atoi(arg.c_str()) != 0;
        tui::success(ec.gpu ? "GPU: on (next model load)" : "GPU: off (next model load)");
    }
    else if (cmd == "temp" && !arg.empty()) {
        float v = (float)atof(arg.c_str());
        if (v >= 0.0f && v <= 2.0f) { sp.temp = v; ec.temp = v; tui::success("Temperature updated"); }
        else tui::error("Temperature must be 0.0-2.0");
    }
    else if ((cmd == "top-k" || cmd == "topk") && !arg.empty()) {
        int v = atoi(arg.c_str());
        if (v >= 1 && v <= 200) { sp.top_k = v; ec.top_k = v; tui::success("Top-K updated"); }
        else tui::error("Top-K must be 1-200");
    }
    else if ((cmd == "top-p" || cmd == "topp") && !arg.empty()) {
        float v = (float)atof(arg.c_str());
        if (v > 0.0f && v <= 1.0f) { sp.top_p = v; ec.top_p = v; tui::success("Top-P updated"); }
        else tui::error("Top-P must be 0.0-1.0");
    }
    else if ((cmd == "rep-penalty" || cmd == "rep") && !arg.empty()) {
        float v = (float)atof(arg.c_str());
        if (v >= 1.0f && v <= 2.0f) { sp.rep_penalty = v; ec.rep_penalty = v; tui::success("Rep penalty updated"); }
        else tui::error("Rep penalty must be 1.0-2.0");
    }
    else if (cmd == "tokens" && !arg.empty()) {
        int v = atoi(arg.c_str());
        if (v >= 8 && v <= 4096) { max_tokens = v; ec.max_tokens = v; tui::success("Max tokens updated"); }
        else tui::error("Tokens must be 8-4096");
    }
    else if (cmd == "mood") {
        tui::banner("Emotional State");
        char buf[64];
        snprintf(buf, 64, "%.2f (baseline: %.2f)", mood.axes[MOOD_WARMTH], mood.baseline[MOOD_WARMTH]);
        tui::status("Warmth", buf);
        snprintf(buf, 64, "%.2f (baseline: %.2f)", mood.axes[MOOD_ENERGY], mood.baseline[MOOD_ENERGY]);
        tui::status("Energy", buf);
        snprintf(buf, 64, "%.2f (baseline: %.2f)", mood.axes[MOOD_FORMALITY], mood.baseline[MOOD_FORMALITY]);
        tui::status("Formality", buf);
        printf("  %-16s%d updates\n\n", "Updates", mood.updates);
    }
    else if (cmd == "reset") {
        ggml_backend_buffer_clear(state.kv_buf, 0);
        state.kv_pos = 0;
        memcpy(mood.axes, mood.baseline, sizeof(mood.axes));
        mood.updates = 0;
        tui::success("Conversation reset (KV cache cleared, mood reset)");
    }
    else if (cmd == "download-model") {
        tui::banner("Available Models");
        printf("  1. Qwen3-0.6B-Q8_0     (660 MB)  Best quality, slower\n");
        printf("  2. Qwen3-0.6B-Q4_K_M   (430 MB)  Good balance\n");
        printf("  3. SmolLM3-3B-Q4_K_M   (1.9 GB)  Best quality, needs more RAM\n\n");
        printf("  Download with curl:\n");
        printf("  \033[36m");
        printf("  curl -L -o model.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q8_0.gguf\n");
        printf("\033[0m\n");
    }
    else tui::error("Unknown command. Type /help for available commands.");
    return true;
}

// ==========================================================================
// interactive_character_chat — full engine with web search tool
// ==========================================================================
bool interactive_character_chat(
    ModelState & state, ggml_backend_t backend, int max_tokens, const char * json_path,
    EngineConfig & engine_config)
{
    const ModelConfig & cfg = state.cfg;
    const int n_layer = (int)cfg.n_layer;
    const int n_vocab = (int)cfg.n_vocab;

    // Load personality
    PersonalityConfig pc;
    if (json_path && parse_personality_json(json_path, pc)) {
        printf("  Character: %s\n", pc.name.c_str());
    } else {
        pc.name = "Aria";
        pc.system_prompt = "You are Aria, a 28-year-old woman who works as a creative technologist. "
            "You're warm, witty, and genuinely curious. You speak naturally like a smart friend. "
            "Never say 'As an AI'. Share opinions honestly.";
        pc.temp_early = 1.30f; pc.temp_mid = 1.0f; pc.temp_late = 0.80f;
        pc.attn_gate_mid = 0.90f; pc.ffn_gate_mid = 0.93f;
        pc.logit_bias_eos = -4.0f;
        pc.sampling_temp = 0.85f; pc.sampling_top_k = 50;
        pc.sampling_top_p = 0.93f; pc.rep_penalty = 1.20f;
        pc.mood_warmth = 0.72f; pc.mood_energy = 0.60f; pc.mood_formality = 0.28f;
    }

    // Build profile + interventions
    ProfileState profile;
    profile_from_personality(pc, profile, (int)cfg.n_layer);
    InterventionTensors iv_t;
    init_interventions(iv_t, cfg, backend);
    InterventionConfig iv;
    SamplingParams sp;
    profile_apply(profile, iv, iv_t, cfg, backend);
    sp = profile.sampling;

    // Emotional state
    EmotionalState mood;
    mood.baseline[MOOD_WARMTH] = pc.mood_warmth;
    mood.baseline[MOOD_ENERGY] = pc.mood_energy;
    mood.baseline[MOOD_FORMALITY] = pc.mood_formality;
    memcpy(mood.axes, mood.baseline, sizeof(mood.axes));

    // Fast weights
    FastWeightMemory fw;
    if (pc.fw_enabled) init_fast_weights(fw, (int)cfg.n_embd);

    // Tool definition
    auto tools = make_web_search_tools();
    std::string system_prompt = build_tool_system_prompt(pc.system_prompt, tools);

    // Grammar engine
    GrammarEngine grammar;
    grammar.init(state.vocab);

    // Allocator (reserve with PREFILL_CHUNK for chunked prefill)
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool need_argmax = (sp.temp <= 0.0f);
    {
        size_t csz = compute_ctx_size(n_layer, true);
        struct ggml_init_params p = { csz, nullptr, true };
        struct ggml_context * mctx = ggml_init(p);
        struct ggml_cgraph * mg = build_graph(mctx, state, PREFILL_CHUNK, 0, (int)cfg.max_ctx,
            n_layer, need_argmax, &iv, &iv_t);
        ggml_gallocr_reserve(galloc, mg);
        ggml_free(mctx);
    }

    size_t ctx_size = compute_ctx_size(n_layer, true);
    std::vector<uint8_t> ctx_buf(ctx_size);
    std::vector<uint16_t> mask;
    std::vector<float> logits_buf(n_vocab);

    // TUI welcome
    printf("%s%s", tui::BOLD, tui::CYAN);
    printf("\n  +======================================+\n");
    printf("  |  Chat with %-25s |\n", pc.name.c_str());
    printf("  +======================================+\n");
    printf("%s", tui::RESET);
    printf("%s", tui::DIM);
    printf("  Type /help for commands, /quit to exit\n\n");
    printf("%s", tui::RESET);

    bool first_turn = true;
    char input[2048];

    while (true) {
        printf("%s%s", tui::GREEN, tui::BOLD);
        printf("You: ");
        printf("%s", tui::RESET);
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;

        // Strip newline
        size_t ilen = strlen(input);
        while (ilen > 0 && (input[ilen-1] == '\n' || input[ilen-1] == '\r')) input[--ilen] = '\0';
        if (ilen == 0) continue;
        if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0 || strcmp(input, "/q") == 0) break;

        // Handle slash commands
        if (input[0] == '/') {
            handle_chat_command(input, engine_config, sp, mood, state, pc, max_tokens);
            continue;
        }

        std::string user_input(input);

        // Detect mood from user input
        detect_mood_keywords(mood, user_input);
        apply_mood_to_interventions(mood, iv, cfg, pc);

        // Append /no_think if thinking disabled
        if (pc.thinking == 0) user_input += " /no_think";

        // Build tokens
        std::vector<int32_t> turn_tokens;
        if (first_turn) {
            turn_tokens = build_chat_tokens(state.vocab, system_prompt, user_input, pc.name);
            first_turn = false;
        } else {
            turn_tokens = build_turn_tokens(state.vocab, user_input, pc.name);
        }

        // Reset grammar
        grammar.detector.reset();
        grammar.grammar.reset();

        printf("%s: ", pc.name.c_str());
        fflush(stdout);

        auto stats = run_chat_turn(state, backend, max_tokens, grammar, galloc,
            iv, iv_t, sp, logits_buf, ctx_size, ctx_buf, mask, turn_tokens, pc.name);

        printf("\n[Mood W:%.2f E:%.2f F:%.2f | %d tok @ %.1fms/tok",
            mood.axes[MOOD_WARMTH], mood.axes[MOOD_ENERGY], mood.axes[MOOD_FORMALITY],
            stats.decode_tokens,
            stats.decode_tokens > 0 ? stats.decode_ms / stats.decode_tokens : 0);
        if (stats.tool_calls > 0) printf(" | %d tool calls", stats.tool_calls);
        printf(" | kv=%d/%d]\n\n", state.kv_pos, (int)cfg.max_ctx);

        // Context management
        if (state.kv_pos > (int)cfg.max_ctx - 200) {
            printf("[Context nearly full (%d/%d), resetting]\n", state.kv_pos, (int)cfg.max_ctx);
            ggml_backend_buffer_clear(state.kv_buf, 0);
            state.kv_pos = 0;
            first_turn = true;
        }
    }

    ggml_gallocr_free(galloc);
    free_interventions(iv_t);
    return true;
}
