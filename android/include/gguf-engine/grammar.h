// grammar.h — Grammar-constrained tool calling engine

#pragma once

#include "types.h"

// ---------------------------------------------------------------------------
// JSON grammar state machine (18 states, depth 32)
// ---------------------------------------------------------------------------
enum JsonState {
    JS_VALUE_START = 0,
    JS_OBJECT_KEY_OR_END,
    JS_COLON,
    JS_OBJECT_VALUE,
    JS_OBJECT_COMMA_OR_END,
    JS_ARRAY_VALUE_OR_END,
    JS_ARRAY_COMMA_OR_END,
    JS_STRING,
    JS_STRING_ESCAPE,
    JS_STRING_U1,
    JS_STRING_U2,
    JS_STRING_U3,
    JS_STRING_U4,
    JS_NUMBER_INT,
    JS_NUMBER_FRAC,
    JS_NUMBER_EXP,
    JS_LITERAL,
    JS_DONE,
};

enum JsonContainer { JC_OBJECT, JC_ARRAY };

struct JSONGrammarState {
    JsonState state = JS_VALUE_START;
    JsonContainer stack[32];
    int depth = 0;
    bool parsing_key = false;

    char literal_target[8];
    int literal_pos = 0;
    int literal_len = 0;

    void reset();
    bool is_complete() const;
    void pop_container();
    void end_value();
    void end_string();
    void end_number();
    bool accept(char c);
};

// ---------------------------------------------------------------------------
// Token classification
// ---------------------------------------------------------------------------
enum TokenClassType {
    TC_NORMAL = 0,
    TC_OPENING_BRACE,
    TC_CLOSING_BRACE,
    TC_OPENING_BRACKET,
    TC_CLOSING_BRACKET,
    TC_COLON,
    TC_COMMA,
    TC_QUOTE,
    TC_WHITESPACE,
    TC_DIGIT,
    TC_BOOL_NULL,
    TC_MINUS,
};

// ---------------------------------------------------------------------------
// Tool call detection
// ---------------------------------------------------------------------------
enum ToolCallPhase {
    TCP_IDLE = 0,
    TCP_DETECTED,
    TCP_PARSING,
    TCP_VALID,
    TCP_INVALID,
};

struct ToolCallDetector {
    ToolCallPhase phase = TCP_IDLE;
    int search_pos = 0;
    std::string accumulated;   // full accumulated text (for end-tag detection)
    std::string json_buffer;   // JSON content only (between <tool_call> tags)

    void reset();
    void feed(const std::string & text);
    bool is_ready() const;
};

// ---------------------------------------------------------------------------
// Grammar engine (LAZY mode: idle during free text, activates on <tool_call>)
// ---------------------------------------------------------------------------
struct GrammarEngine {
    ToolCallDetector detector;
    JSONGrammarState grammar;

    // Precomputed token masks
    std::vector<uint8_t> safe_string_bitmap;  // tokens safe inside JSON strings
    int n_vocab = 0;
    bool initialized = false;

    // Stats
    int tokens_free = 0;
    int tokens_constrained = 0;
    double grammar_mask_us = 0;

    // Initialize with vocabulary
    void init(const std::vector<std::string> & vocab);

    // Apply grammar mask to logits (CPU-side)
    void apply_mask(float * logits, const std::vector<std::string> & vocab);

    // Advance grammar + detector with chosen token
    void advance(int32_t token_id, const std::string & raw_vocab_entry);

    // Check if a complete tool call JSON is ready
    bool is_tool_call_ready() const;

    // Check if grammar is actively constraining
    bool is_active() const;

    // Reset state
    void reset();

    // Reset for next tool call round (keeps vocabulary data)
    void reset_for_next_round();

    // Print grammar statistics
    void print_stats() const;
};

// ---------------------------------------------------------------------------
// Tool system prompt and parsing
// ---------------------------------------------------------------------------

// Build system prompt with tool definitions in Qwen3 XML format
std::string build_tool_system_prompt(const std::string & base_prompt,
                                     const std::vector<ToolDef> & tools);

// Parse tool call JSON from generated text
ToolCallResult parse_tool_call_json(const std::string & text);

// Extract a JSON string value by key
std::string extract_json_string(const std::string & json, const std::string & key);
