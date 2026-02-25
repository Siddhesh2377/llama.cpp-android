// grammar.cpp — Grammar-constrained tool calling engine
//
// LAZY grammar: model generates freely, grammar activates on <tool_call>,
// forces valid JSON, then forces </tool_call>. Multi-step support.
// CPU-side logit masking (~200us per constrained token, <0.5% overhead).

#include "gguf-engine/grammar.h"
#include "gguf-engine/utils.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

// ==========================================================================
// Component 1: Tool Definitions + System Prompt Builder
// ==========================================================================

std::string build_tool_system_prompt(const std::string & base_prompt,
                                      const std::vector<ToolDef> & tools) {
    std::string p = base_prompt;
    p += "\n\n# Tools\n\n";
    p += "You may call one or more functions to assist with the user query.\n";
    p += "You are provided with function signatures within <tools></tools> XML tags:\n";
    p += "<tools>\n";

    for (int i = 0; i < (int)tools.size(); i++) {
        const ToolDef & t = tools[i];
        p += "{\"type\":\"function\",\"function\":{\"name\":\"";
        p += t.name;
        p += "\",\"description\":\"";
        p += t.description;
        p += "\",\"parameters\":{\"type\":\"object\",\"properties\":{";
        for (int j = 0; j < (int)t.params.size(); j++) {
            if (j > 0) p += ",";
            p += "\"";
            p += t.params[j].name;
            p += "\":{\"type\":\"";
            p += t.params[j].type;
            p += "\",\"description\":\"";
            p += t.params[j].description;
            p += "\"";
            if (!t.params[j].enum_values.empty()) {
                p += ",\"enum\":[";
                for (int e = 0; e < (int)t.params[j].enum_values.size(); e++) {
                    if (e > 0) p += ",";
                    p += "\"";
                    p += t.params[j].enum_values[e];
                    p += "\"";
                }
                p += "]";
            }
            p += "}";
        }
        p += "},\"required\":[";
        bool first_req = true;
        for (int j = 0; j < (int)t.params.size(); j++) {
            if (t.params[j].required) {
                if (!first_req) p += ",";
                p += "\"";
                p += t.params[j].name;
                p += "\"";
                first_req = false;
            }
        }
        p += "]}}}\n";
    }

    p += "</tools>\n\n";
    p += "For each function call, return a json object with function name and arguments ";
    p += "within <tool_call></tool_call> XML tags:\n";
    p += "<tool_call>\n";
    p += "{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}\n";
    p += "</tool_call>";

    return p;
}

// ==========================================================================
// Component 2: JSON Grammar State Machine
// ==========================================================================

static constexpr int JSON_MAX_DEPTH = 32;

static bool is_ws(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static bool is_hex(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

void JSONGrammarState::reset() {
    state = JS_VALUE_START;
    depth = 0;
    parsing_key = false;
    literal_pos = 0;
    literal_len = 0;
}

bool JSONGrammarState::is_complete() const {
    return state == JS_DONE;
}

void JSONGrammarState::pop_container() {
    depth--;
    if (depth <= 0) {
        depth = 0;
        state = JS_DONE;
        return;
    }
    state = (stack[depth - 1] == JC_OBJECT) ? JS_OBJECT_COMMA_OR_END
                                              : JS_ARRAY_COMMA_OR_END;
}

void JSONGrammarState::end_value() {
    if (depth <= 0) {
        state = JS_DONE;
        return;
    }
    if (stack[depth - 1] == JC_OBJECT) {
        state = JS_OBJECT_COMMA_OR_END;
    } else {
        state = JS_ARRAY_COMMA_OR_END;
    }
}

void JSONGrammarState::end_string() {
    if (depth > 0 && stack[depth - 1] == JC_OBJECT && parsing_key) {
        parsing_key = false;
        state = JS_COLON;
        return;
    }
    end_value();
}

void JSONGrammarState::end_number() {
    end_value();
}

bool JSONGrammarState::accept(char c) {
    switch (state) {
    case JS_VALUE_START:
        if (is_ws(c)) return true;
        if (c == '{') {
            if (depth >= JSON_MAX_DEPTH) return false;
            stack[depth] = JC_OBJECT;
            depth++;
            state = JS_OBJECT_KEY_OR_END;
            return true;
        }
        if (c == '[') {
            if (depth >= JSON_MAX_DEPTH) return false;
            stack[depth] = JC_ARRAY;
            depth++;
            state = JS_ARRAY_VALUE_OR_END;
            return true;
        }
        if (c == '"') { state = JS_STRING; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            state = JS_NUMBER_INT;
            return true;
        }
        if (c == 't') { strncpy(literal_target, "true",  8); literal_pos = 1; literal_len = 4; state = JS_LITERAL; return true; }
        if (c == 'f') { strncpy(literal_target, "false", 8); literal_pos = 1; literal_len = 5; state = JS_LITERAL; return true; }
        if (c == 'n') { strncpy(literal_target, "null",  8); literal_pos = 1; literal_len = 4; state = JS_LITERAL; return true; }
        return false;

    case JS_OBJECT_KEY_OR_END:
        if (is_ws(c)) return true;
        if (c == '}') { pop_container(); return true; }
        if (c == '"') {
            parsing_key = true;
            state = JS_STRING;
            return true;
        }
        return false;

    case JS_COLON:
        if (is_ws(c)) return true;
        if (c == ':') { state = JS_VALUE_START; return true; }
        return false;

    case JS_OBJECT_VALUE:
        // Alias for JS_VALUE_START after colon — delegate
        state = JS_VALUE_START;
        return accept(c);

    case JS_OBJECT_COMMA_OR_END:
        if (is_ws(c)) return true;
        if (c == ',') { state = JS_OBJECT_KEY_OR_END; return true; }
        if (c == '}') { pop_container(); return true; }
        return false;

    case JS_ARRAY_VALUE_OR_END:
        if (is_ws(c)) return true;
        if (c == ']') { pop_container(); return true; }
        state = JS_VALUE_START;
        return accept(c);

    case JS_ARRAY_COMMA_OR_END:
        if (is_ws(c)) return true;
        if (c == ',') { state = JS_ARRAY_VALUE_OR_END; return true; }
        if (c == ']') { pop_container(); return true; }
        return false;

    case JS_STRING:
        if (c == '"') { end_string(); return true; }
        if (c == '\\') { state = JS_STRING_ESCAPE; return true; }
        if ((unsigned char)c >= 0x20) return true;
        return false;

    case JS_STRING_ESCAPE:
        if (c == '"' || c == '\\' || c == '/' || c == 'b' ||
            c == 'f' || c == 'n' || c == 'r' || c == 't') {
            state = JS_STRING;
            return true;
        }
        if (c == 'u') { state = JS_STRING_U1; return true; }
        return false;

    case JS_STRING_U1: if (is_hex(c)) { state = JS_STRING_U2; return true; } return false;
    case JS_STRING_U2: if (is_hex(c)) { state = JS_STRING_U3; return true; } return false;
    case JS_STRING_U3: if (is_hex(c)) { state = JS_STRING_U4; return true; } return false;
    case JS_STRING_U4: if (is_hex(c)) { state = JS_STRING;    return true; } return false;

    case JS_NUMBER_INT:
        if (c >= '0' && c <= '9') return true;
        if (c == '.') { state = JS_NUMBER_FRAC; return true; }
        if (c == 'e' || c == 'E') { state = JS_NUMBER_EXP; return true; }
        // Number ended — re-process char in parent context
        end_number();
        return accept(c);

    case JS_NUMBER_FRAC:
        if (c >= '0' && c <= '9') return true;
        if (c == 'e' || c == 'E') { state = JS_NUMBER_EXP; return true; }
        end_number();
        return accept(c);

    case JS_NUMBER_EXP:
        if (c == '+' || c == '-' || (c >= '0' && c <= '9')) return true;
        end_number();
        return accept(c);

    case JS_LITERAL:
        if (literal_pos < literal_len && c == literal_target[literal_pos]) {
            literal_pos++;
            if (literal_pos >= literal_len) { end_value(); return true; }
            return true;
        }
        return false;

    case JS_DONE:
        if (is_ws(c)) return true;
        return false;
    }
    return false;
}

// ==========================================================================
// Component 3: Get valid bytes for grammar state (used by apply_mask)
// ==========================================================================

static void get_valid_bytes(const JSONGrammarState & grammar, bool * valid) {
    memset(valid, 0, 256);

    switch (grammar.state) {
    case JS_VALUE_START:
        valid[(uint8_t)'{'] = valid[(uint8_t)'['] = valid[(uint8_t)'"'] = true;
        valid[(uint8_t)'-'] = true;
        for (int d = '0'; d <= '9'; d++) valid[d] = true;
        valid[(uint8_t)'t'] = valid[(uint8_t)'f'] = valid[(uint8_t)'n'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_OBJECT_KEY_OR_END:
        valid[(uint8_t)'"'] = valid[(uint8_t)'}'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_COLON:
        valid[(uint8_t)':'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_OBJECT_VALUE:
        // Same as JS_VALUE_START
        valid[(uint8_t)'{'] = valid[(uint8_t)'['] = valid[(uint8_t)'"'] = true;
        valid[(uint8_t)'-'] = true;
        for (int d = '0'; d <= '9'; d++) valid[d] = true;
        valid[(uint8_t)'t'] = valid[(uint8_t)'f'] = valid[(uint8_t)'n'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_OBJECT_COMMA_OR_END:
        valid[(uint8_t)','] = valid[(uint8_t)'}'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_ARRAY_VALUE_OR_END:
        valid[(uint8_t)'{'] = valid[(uint8_t)'['] = valid[(uint8_t)'"'] = valid[(uint8_t)']'] = true;
        valid[(uint8_t)'-'] = true;
        for (int d = '0'; d <= '9'; d++) valid[d] = true;
        valid[(uint8_t)'t'] = valid[(uint8_t)'f'] = valid[(uint8_t)'n'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_ARRAY_COMMA_OR_END:
        valid[(uint8_t)','] = valid[(uint8_t)']'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_STRING:
        for (int b = 0x20; b < 256; b++) valid[b] = true;
        break;
    case JS_STRING_ESCAPE:
        valid[(uint8_t)'"'] = valid[(uint8_t)'\\'] = valid[(uint8_t)'/'] = true;
        valid[(uint8_t)'b'] = valid[(uint8_t)'f'] = valid[(uint8_t)'n'] = true;
        valid[(uint8_t)'r'] = valid[(uint8_t)'t'] = valid[(uint8_t)'u'] = true;
        break;
    case JS_STRING_U1: case JS_STRING_U2: case JS_STRING_U3: case JS_STRING_U4:
        for (int d = '0'; d <= '9'; d++) valid[d] = true;
        for (int d = 'a'; d <= 'f'; d++) valid[d] = true;
        for (int d = 'A'; d <= 'F'; d++) valid[d] = true;
        break;
    case JS_NUMBER_INT:
        for (int d = '0'; d <= '9'; d++) valid[d] = true;
        valid[(uint8_t)'.'] = valid[(uint8_t)'e'] = valid[(uint8_t)'E'] = true;
        // Number-ending chars
        valid[(uint8_t)','] = valid[(uint8_t)'}'] = valid[(uint8_t)']'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_NUMBER_FRAC:
        for (int d = '0'; d <= '9'; d++) valid[d] = true;
        valid[(uint8_t)'e'] = valid[(uint8_t)'E'] = true;
        valid[(uint8_t)','] = valid[(uint8_t)'}'] = valid[(uint8_t)']'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_NUMBER_EXP:
        valid[(uint8_t)'+'] = valid[(uint8_t)'-'] = true;
        for (int d = '0'; d <= '9'; d++) valid[d] = true;
        valid[(uint8_t)','] = valid[(uint8_t)'}'] = valid[(uint8_t)']'] = true;
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    case JS_LITERAL:
        if (grammar.literal_pos < grammar.literal_len)
            valid[(uint8_t)grammar.literal_target[grammar.literal_pos]] = true;
        break;
    case JS_DONE:
        valid[(uint8_t)' '] = valid[(uint8_t)'\n'] = valid[(uint8_t)'\r'] = valid[(uint8_t)'\t'] = true;
        break;
    }
}

// ==========================================================================
// Component 4: Tool Call Detector
// ==========================================================================

static constexpr const char * TOOL_OPEN_TAG  = "<tool_call>";
static constexpr int TOOL_OPEN_LEN  = 11;

void ToolCallDetector::reset() {
    phase = TCP_IDLE;
    search_pos = 0;
    accumulated.clear();
    json_buffer.clear();
}

void ToolCallDetector::feed(const std::string & text) {
    for (char c : text) {
        accumulated += c;

        switch (phase) {
        case TCP_IDLE:
            if (c == TOOL_OPEN_TAG[search_pos]) {
                search_pos++;
                if (search_pos >= TOOL_OPEN_LEN) {
                    phase = TCP_DETECTED;
                    search_pos = 0;
                }
            } else {
                search_pos = (c == '<') ? 1 : 0;
            }
            break;

        case TCP_DETECTED:
        case TCP_PARSING:
        case TCP_VALID:
        case TCP_INVALID:
            break;
        }
    }
}

bool ToolCallDetector::is_ready() const {
    return phase == TCP_DETECTED;
}

// ==========================================================================
// Component 5: Grammar Engine (Orchestrator)
// ==========================================================================

void GrammarEngine::init(const std::vector<std::string> & vocab) {
    n_vocab = (int)vocab.size();

    // Build safe_string_bitmap: tokens that are safe inside JSON strings
    // (no " or \ or control chars)
    safe_string_bitmap.resize(n_vocab, 0);
    int n_safe = 0;
    for (int id = 0; id < n_vocab; id++) {
        std::string decoded = decode_token(vocab[id]);
        if (decoded.empty()) continue;
        bool safe = true;
        for (char c : decoded) {
            if (c == '"' || c == '\\' || (uint8_t)c < 0x20) {
                safe = false;
                break;
            }
        }
        if (safe) {
            safe_string_bitmap[id] = 1;
            n_safe++;
        }
    }

    detector.reset();
    grammar.reset();
    initialized = true;

    printf("  GrammarEngine: %d/%d tokens string-safe\n", n_safe, n_vocab);
}

void GrammarEngine::apply_mask(float * logits, const std::vector<std::string> & vocab) {
    if (!is_active()) return;

    auto t0 = Clock::now();

    bool valid_bytes[256];
    get_valid_bytes(grammar, valid_bytes);

    const bool in_string = (grammar.state == JS_STRING);
    const int nv = (int)vocab.size();

    if (in_string) {
        // STRING PATH: most tokens valid. Only mask invalid ones.
        for (int id = 0; id < nv; id++) {
            std::string decoded = decode_token(vocab[id]);
            if (decoded.empty()) {
                logits[id] = -INFINITY;
                continue;
            }
            // First byte check
            if (!valid_bytes[(uint8_t)decoded[0]]) {
                logits[id] = -INFINITY;
                continue;
            }
            // Multi-byte tokens that are NOT string-safe need full validation
            if (decoded.size() > 1 && !safe_string_bitmap[id]) {
                JSONGrammarState test = grammar;
                bool valid = true;
                for (size_t i = 0; i < decoded.size(); i++) {
                    if (!test.accept(decoded[i])) { valid = false; break; }
                }
                if (!valid) logits[id] = -INFINITY;
            }
        }
    } else {
        // NON-STRING PATH: few valid tokens. Mask all, unmask valid.
        struct VT { int32_t tid; float val; };
        std::vector<VT> valid_list;
        valid_list.reserve(2048);

        for (int id = 0; id < nv; id++) {
            std::string decoded = decode_token(vocab[id]);
            if (decoded.empty()) continue;
            if (!valid_bytes[(uint8_t)decoded[0]]) continue;

            if (decoded.size() == 1) {
                valid_list.push_back({id, logits[id]});
                continue;
            }
            // Multi-byte: full validation
            JSONGrammarState test = grammar;
            bool valid = true;
            for (size_t i = 0; i < decoded.size(); i++) {
                if (!test.accept(decoded[i])) { valid = false; break; }
            }
            if (valid) valid_list.push_back({id, logits[id]});
        }

        for (int i = 0; i < nv; i++) logits[i] = -INFINITY;
        for (const auto & v : valid_list) logits[v.tid] = v.val;
    }

    auto t1 = Clock::now();
    grammar_mask_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    tokens_constrained++;
}

bool GrammarEngine::is_tool_call_ready() const {
    return detector.phase == TCP_VALID
        || (detector.phase == TCP_DETECTED && grammar.is_complete())
        || (detector.phase == TCP_PARSING && grammar.is_complete());
}

bool GrammarEngine::is_active() const {
    return detector.phase == TCP_DETECTED || detector.phase == TCP_PARSING;
}

void GrammarEngine::reset() {
    detector.reset();
    grammar.reset();
}

void GrammarEngine::advance(int32_t token_id, const std::string & raw_vocab_entry) {
    std::string decoded = decode_token(raw_vocab_entry);

    // If grammar active (detected tool call, parsing JSON), advance state
    if (detector.phase == TCP_DETECTED || detector.phase == TCP_PARSING) {
        for (char c : decoded) {
            grammar.accept(c);
        }
        detector.json_buffer += decoded;
        // Check if JSON complete
        if (grammar.is_complete()) {
            detector.phase = TCP_VALID;
        }
    }

    // Always feed detector (for IDLE -> DETECTED transition)
    if (detector.phase == TCP_IDLE) {
        detector.feed(decoded);
        // If just transitioned to DETECTED, init grammar
        if (detector.phase == TCP_DETECTED) {
            grammar.reset();
            // Feed any JSON chars already accumulated from the same token
            for (char c : detector.json_buffer) {
                grammar.accept(c);
            }
        }
        tokens_free++;
    }
}

void GrammarEngine::reset_for_next_round() {
    detector.reset();
    grammar.reset();
}

void GrammarEngine::print_stats() const {
    printf("  Grammar stats:\n");
    printf("    Free tokens: %d\n", tokens_free);
    printf("    Constrained tokens: %d\n", tokens_constrained);
    if (tokens_constrained > 0) {
        printf("    Avg mask time: %.1f us/token\n",
            grammar_mask_us / tokens_constrained);
    }
}

// ==========================================================================
// Component 6: Tool Call Parsing
// ==========================================================================

ToolCallResult parse_tool_call_json(const std::string & json) {
    ToolCallResult result;
    result.valid = false;

    size_t start = json.find_first_not_of(" \t\n\r");
    size_t end = json.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return result;
    std::string trimmed = json.substr(start, end - start + 1);

    // Find "name" and extract value
    size_t name_key = trimmed.find("\"name\"");
    if (name_key == std::string::npos) return result;
    size_t colon = trimmed.find(':', name_key + 6);
    if (colon == std::string::npos) return result;
    size_t name_start = trimmed.find('"', colon + 1);
    if (name_start == std::string::npos) return result;
    name_start++;
    size_t name_end = trimmed.find('"', name_start);
    if (name_end == std::string::npos) return result;
    result.name = trimmed.substr(name_start, name_end - name_start);

    // Find "arguments" and extract object
    size_t args_key = trimmed.find("\"arguments\"");
    if (args_key == std::string::npos) return result;
    size_t args_brace = trimmed.find('{', args_key + 11);
    if (args_brace == std::string::npos) return result;

    int brace_depth = 0;
    size_t args_end_pos = args_brace;
    for (size_t i = args_brace; i < trimmed.size(); i++) {
        if (trimmed[i] == '{') brace_depth++;
        if (trimmed[i] == '}') {
            brace_depth--;
            if (brace_depth == 0) { args_end_pos = i; break; }
        }
    }

    result.arguments_json = trimmed.substr(args_brace, args_end_pos - args_brace + 1);
    result.valid = true;
    return result;
}

std::string extract_json_string(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos);
    if (pos == std::string::npos) return "";
    pos++; // skip opening quote
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            result += json[pos];
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}
