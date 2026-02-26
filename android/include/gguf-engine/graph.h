// graph.h — Transformer compute graph builder

#pragma once

#include "types.h"

// Build transformer forward pass graph (token-based input)
struct ggml_cgraph * build_graph(
    struct ggml_context * ctx,
    ModelState & state,
    int seq_len,
    int kv_pos,
    int kv_len,
    int n_layers,
    bool need_argmax = false
);

// Compute context size needed for build_graph
size_t compute_ctx_size(int n_layers);
