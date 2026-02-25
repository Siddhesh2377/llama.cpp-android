// utils.h — Utility functions for gguf-engine
//
// Small helpers: token decoding, causal mask, validation, GGUF key access.
// Header-only (inline functions).

#pragma once

#include "types.h"
#include <cmath>

// ---------------------------------------------------------------------------
// Token decoding (BPE → printable)
// ---------------------------------------------------------------------------
inline std::string decode_token(const std::string & tok) {
    std::string out;
    out.reserve(tok.size());
    for (size_t i = 0; i < tok.size(); ) {
        if (i + 1 < tok.size() && (uint8_t)tok[i] == 0xC4 && (uint8_t)tok[i+1] == 0xA0) {
            out += ' '; i += 2;
        } else if (i + 1 < tok.size() && (uint8_t)tok[i] == 0xC4 && (uint8_t)tok[i+1] == 0x8A) {
            out += '\n'; i += 2;
        } else {
            out += tok[i]; i++;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// GGUF key helpers
// ---------------------------------------------------------------------------
inline uint32_t gguf_get_u32(struct gguf_context * ctx, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    enum gguf_type type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_UINT32) return gguf_get_val_u32(ctx, id);
    if (type == GGUF_TYPE_INT32)  return (uint32_t)gguf_get_val_i32(ctx, id);
    if (type == GGUF_TYPE_UINT64) return (uint32_t)gguf_get_val_u64(ctx, id);
    if (type == GGUF_TYPE_INT64)  return (uint32_t)gguf_get_val_i64(ctx, id);
    return fallback;
}

inline float gguf_get_f32_val(struct gguf_context * ctx, const char * key, float fallback) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    enum gguf_type type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(ctx, id);
    if (type == GGUF_TYPE_FLOAT64) return (float)gguf_get_val_f64(ctx, id);
    return fallback;
}

inline const char * gguf_get_str_val(struct gguf_context * ctx, const char * key, const char * fallback) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return fallback;
    return gguf_get_val_str(ctx, id);
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------
inline int count_bad(const float * data, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (std::isnan(data[i]) || std::isinf(data[i])) bad++;
    }
    return bad;
}

inline bool all_zero(const float * data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] != 0.0f) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Causal mask construction (F16)
// ---------------------------------------------------------------------------
inline void build_causal_mask(std::vector<uint16_t> & mask, int kv_len, int seq_len, int kv_pos) {
    mask.resize(kv_len * seq_len);
    const uint16_t neg_inf = 0xFC00;
    for (int q = 0; q < seq_len; q++) {
        int max_attend = kv_pos + q;
        for (int k = 0; k < kv_len; k++) {
            mask[q * kv_len + k] = (k <= max_attend) ? 0 : neg_inf;
        }
    }
}
