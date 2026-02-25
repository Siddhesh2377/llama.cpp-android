// stall.h — Stall generation during async tool execution

#pragma once

#include "types.h"

// Initialize mini KV cache for stall generation (32 ctx)
bool init_stall_kv(StallKV & skv, const ModelConfig & cfg, ggml_backend_t backend);

// Free stall KV resources
void free_stall_kv(StallKV & skv);

// Generate stall text while waiting for async tool result
std::string generate_stall(ModelState & state, ggml_backend_t backend,
    ggml_gallocr_t stall_galloc, StallKV & skv,
    const std::string & prompt, int max_tokens,
    AsyncToolResult & tool_result);
