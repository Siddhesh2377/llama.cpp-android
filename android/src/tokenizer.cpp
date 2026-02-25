// tokenizer.cpp — BPE tokenization and ChatML token construction

#include "gguf-engine/tokenizer.h"
#include "gguf-engine/utils.h"

#include <cstring>
#include <unordered_map>

// Simple BPE tokenizer — greedy longest-match against vocabulary
std::vector<int> tokenize_simple(const std::vector<std::string> & vocab,
                                 const std::string & text) {
    // Build reverse lookup: token_string -> token_id
    std::unordered_map<std::string, int32_t> token_map;
    for (int id = 0; id < (int)vocab.size(); id++) {
        // First occurrence wins (lower ID = higher priority for BPE)
        if (token_map.find(vocab[id]) == token_map.end()) {
            token_map[vocab[id]] = id;
        }
    }

    std::vector<int> tokens;
    int len = (int)text.size();
    int pos = 0;

    while (pos < len) {
        int best_len = 0;
        int best_id = -1;

        // Greedy longest match (up to 32 chars)
        int max_try = std::min(32, len - pos);
        for (int l = max_try; l >= 1; l--) {
            std::string candidate(text.c_str() + pos, l);
            auto it = token_map.find(candidate);
            if (it != token_map.end()) {
                best_len = l;
                best_id = it->second;
                break;
            }
        }

        if (best_id >= 0) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            // Skip unknown byte
            pos++;
        }
    }
    return tokens;
}

// Build ChatML token sequence: system + user + assistant prefix
// Format: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
std::vector<int> build_chat_tokens(
    const std::vector<std::string> & vocab,
    const std::string & system_prompt,
    const std::string & user_message,
    const std::string & assistant_name
) {
    // Build token lookup
    std::unordered_map<std::string, int32_t> token_map;
    for (int id = 0; id < (int)vocab.size(); id++) {
        if (token_map.find(vocab[id]) == token_map.end()) {
            token_map[vocab[id]] = id;
        }
    }

    // Find special tokens
    int32_t im_start = -1, im_end = -1;
    auto it_s = token_map.find("<|im_start|>");
    auto it_e = token_map.find("<|im_end|>");
    if (it_s != token_map.end()) im_start = it_s->second;
    if (it_e != token_map.end()) im_end = it_e->second;

    printf("  ChatML tokens: <|im_start|>=%d  <|im_end|>=%d\n", im_start, im_end);

    std::vector<int> tokens;

    if (im_start >= 0 && im_end >= 0) {
        // Qwen/ChatML format:
        // <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
        auto tokenize_text = [&](const std::string & text) {
            auto toks = tokenize_simple(vocab, text);
            tokens.insert(tokens.end(), toks.begin(), toks.end());
        };

        tokens.push_back(im_start);
        tokenize_text("system\n" + system_prompt);
        tokens.push_back(im_end);
        tokenize_text("\n");

        tokens.push_back(im_start);
        tokenize_text("user\n" + user_message);
        tokens.push_back(im_end);
        tokenize_text("\n");

        tokens.push_back(im_start);
        tokenize_text(assistant_name + "\n");
    } else {
        // Fallback: plain text
        printf("  WARNING: ChatML special tokens not found, using plain prompt\n");
        auto toks = tokenize_simple(vocab, (system_prompt + "\n\n" + user_message + "\n"));
        tokens = toks;
    }

    return tokens;
}

// Build multi-turn user tokens (not first turn — appended to existing KV cache)
// Format: <|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
std::vector<int> build_turn_tokens(const std::vector<std::string> & vocab,
    const std::string & user_message,
    const std::string & assistant_name)
{
    std::unordered_map<std::string, int32_t> token_map;
    for (int id = 0; id < (int)vocab.size(); id++) {
        if (token_map.find(vocab[id]) == token_map.end())
            token_map[vocab[id]] = id;
    }

    int32_t im_start = -1, im_end = -1;
    auto it_s = token_map.find("<|im_start|>");
    auto it_e = token_map.find("<|im_end|>");
    if (it_s != token_map.end()) im_start = it_s->second;
    if (it_e != token_map.end()) im_end = it_e->second;

    std::vector<int> tokens;
    if (im_start < 0 || im_end < 0) return tokens;

    auto add_text = [&](const std::string & text) {
        auto toks = tokenize_simple(vocab, text);
        tokens.insert(tokens.end(), toks.begin(), toks.end());
    };

    // <|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
    tokens.push_back(im_end);
    add_text("\n");
    tokens.push_back(im_start);
    add_text("user\n" + user_message);
    tokens.push_back(im_end);
    add_text("\n");
    tokens.push_back(im_start);
    add_text(assistant_name + "\n");

    return tokens;
}

// Build tool result tokens for injection after tool execution
// Format: <|im_end|>\n<|im_start|>tool\n{result}<|im_end|>\n<|im_start|>assistant\n
std::vector<int> build_tool_result_tokens(
    const std::vector<std::string> & vocab,
    const std::string & result,
    const std::string & assistant_name
) {
    std::unordered_map<std::string, int32_t> token_map;
    for (int id = 0; id < (int)vocab.size(); id++) {
        if (token_map.find(vocab[id]) == token_map.end())
            token_map[vocab[id]] = id;
    }

    int32_t im_start = -1, im_end = -1;
    auto it_s = token_map.find("<|im_start|>");
    auto it_e = token_map.find("<|im_end|>");
    if (it_s != token_map.end()) im_start = it_s->second;
    if (it_e != token_map.end()) im_end = it_e->second;

    std::vector<int> tokens;
    if (im_start < 0 || im_end < 0) return tokens;

    auto add_text = [&](const std::string & text) {
        auto toks = tokenize_simple(vocab, text);
        tokens.insert(tokens.end(), toks.begin(), toks.end());
    };

    tokens.push_back(im_end);
    add_text("\n");
    tokens.push_back(im_start);
    add_text("tool\n" + result);
    tokens.push_back(im_end);
    add_text("\n");
    tokens.push_back(im_start);
    add_text(assistant_name + "\n");

    return tokens;
}
