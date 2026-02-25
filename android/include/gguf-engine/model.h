// model.h — GGUF model loading and KV cache initialization

#pragma once

#include "types.h"

// Load model weights and tokenizer from GGUF file
bool load_model(ModelState & state, const char * path, int fd, ggml_backend_t backend);

// Initialize KV cache (F16, flash_attn_ext layout)
bool init_kv_cache(ModelState & state, ggml_backend_t backend);
