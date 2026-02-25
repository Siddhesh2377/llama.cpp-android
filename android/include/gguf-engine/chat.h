// chat.h — Interactive chat modes
//
// Contains:
//   - ChatTurnStats: per-turn statistics
//   - run_chat_turn(): single prefill+decode turn with tool calling
//   - interactive_chat(): basic interactive chat with tiered prompts
//   - handle_chat_command(): slash command processing
//   - interactive_character_chat(): full character engine loop

#pragma once

#include "types.h"
#include "grammar.h"

// ---------------------------------------------------------------------------
// Per-turn generation statistics
// ---------------------------------------------------------------------------
struct ChatTurnStats {
    int prefill_tokens;
    double prefill_ms;
    int decode_tokens;
    double decode_ms;
    int tool_calls;
    double search_ms;
    int grammar_constrained;
    bool used_search;
    std::string response_text;
};

// ---------------------------------------------------------------------------
// System prompt tiers
// ---------------------------------------------------------------------------
extern const char * PROMPT_TIER0;
extern const char * PROMPT_TIER1;
extern const char * PROMPT_TIER2;

// ---------------------------------------------------------------------------
// Run a single chat turn: prefill + decode + tool call handling
// ---------------------------------------------------------------------------
ChatTurnStats run_chat_turn(
    ModelState & state, ggml_backend_t backend, int max_tokens,
    GrammarEngine & grammar, ggml_gallocr_t galloc,
    InterventionConfig & iv, InterventionTensors & iv_t,
    const SamplingParams & sp, std::vector<float> & logits_buf,
    size_t ctx_size, std::vector<uint8_t> & ctx_buf,
    std::vector<uint16_t> & mask,
    const std::vector<int32_t> & turn_tokens,
    const std::string & assistant_name = "assistant");

// ---------------------------------------------------------------------------
// Interactive chat loop (basic, with tiered prompts)
// ---------------------------------------------------------------------------
bool interactive_chat(ModelState & state, ggml_backend_t backend,
    int max_tokens, int prompt_tier);

// ---------------------------------------------------------------------------
// Chat command handler (returns true if command was handled)
// ---------------------------------------------------------------------------
bool handle_chat_command(
    const char * input, EngineConfig & ec, SamplingParams & sp,
    EmotionalState & mood, ModelState & state, PersonalityConfig & pc,
    int & max_tokens);

// ---------------------------------------------------------------------------
// Interactive character chat — full engine with web search tool
// ---------------------------------------------------------------------------
bool interactive_character_chat(
    ModelState & state, ggml_backend_t backend, int max_tokens,
    const char * json_path, EngineConfig & engine_config);
