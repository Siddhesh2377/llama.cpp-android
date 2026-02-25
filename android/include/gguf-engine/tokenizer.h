// tokenizer.h — BPE tokenization and ChatML token construction

#pragma once

#include "types.h"

// Simple BPE tokenizer (vocabulary lookup)
std::vector<int> tokenize_simple(const std::vector<std::string> & vocab,
                                 const std::string & text);

// Build ChatML token sequence: system + user + assistant prefix
std::vector<int> build_chat_tokens(const std::vector<std::string> & vocab,
                                   const std::string & system_prompt,
                                   const std::string & user_message,
                                   const std::string & assistant_name = "assistant");

// Build turn tokens for multi-turn: append user->assistant exchange
std::vector<int> build_turn_tokens(const std::vector<std::string> & vocab,
                                   const std::string & user_message,
                                   const std::string & assistant_name = "assistant");

// Build tool result tokens for injection
std::vector<int> build_tool_result_tokens(const std::vector<std::string> & vocab,
                                          const std::string & result,
                                          const std::string & assistant_name = "assistant");
