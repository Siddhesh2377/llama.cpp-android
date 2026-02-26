// model.h — GGUF model loading and KV cache initialization

#pragma once

#include "types.h"

// Load model weights and tokenizer from GGUF file
// quant_type: target quantization for 2D weights (GGML_TYPE_COUNT = keep original)
// quant_ffn: separate quantization for FFN weights (GGML_TYPE_COUNT = use quant_type)
bool load_model(ModelState & state, const char * path, int fd, ggml_backend_t backend,
                ggml_type quant_type = GGML_TYPE_COUNT,
                ggml_type quant_ffn = GGML_TYPE_COUNT);

// Initialize KV cache (F16, flash_attn_ext layout)
bool init_kv_cache(ModelState & state, ggml_backend_t backend);
