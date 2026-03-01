#pragma once

/**
 * Smart KV Cache Manager for Android
 *
 * Provides memory-pressure-aware KV cache management with:
 * - Session save/restore for conversation continuity
 * - Sliding window with priority eviction
 * - Automatic cache trimming on low memory
 * - Context shifting for long conversations
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kv_cache_manager kv_cache_manager_t;

// Forward declare - user provides the engine
typedef struct ggml_engine ggml_engine_t;

// Cache management strategy
typedef enum {
    KV_STRATEGY_FIFO,           // remove oldest tokens first
    KV_STRATEGY_SLIDING_WINDOW, // keep system prompt + recent window
    KV_STRATEGY_PRIORITY,       // keep high-importance tokens
} kv_cache_strategy;

// Configuration
typedef struct {
    kv_cache_strategy strategy;
    int32_t  max_cache_tokens;       // max tokens to keep (0 = use full context)
    int32_t  keep_first_n;           // always keep first N tokens (system prompt)
    int32_t  sliding_window_size;    // for sliding window strategy
    float    memory_pressure_threshold; // 0.0-1.0, auto-trim when used% exceeds this
    bool     enable_auto_trim;       // auto-trim when approaching limit
} kv_cache_config;

// Create / destroy
kv_cache_manager_t * kv_cache_manager_create(kv_cache_config config);
void                 kv_cache_manager_free(kv_cache_manager_t * mgr);

// Attach to an engine (must be called after engine loads a model)
void                 kv_cache_manager_attach(kv_cache_manager_t * mgr, ggml_engine_t * engine);

// Check cache state
int32_t              kv_cache_manager_tokens_used(const kv_cache_manager_t * mgr);
int32_t              kv_cache_manager_tokens_capacity(const kv_cache_manager_t * mgr);
float                kv_cache_manager_usage_ratio(const kv_cache_manager_t * mgr);

// Manual cache management
bool                 kv_cache_manager_trim(kv_cache_manager_t * mgr, int32_t n_keep);
void                 kv_cache_manager_clear(kv_cache_manager_t * mgr);

// Context shift: when cache is full, shift to make room
// Returns number of tokens freed
int32_t              kv_cache_manager_shift(kv_cache_manager_t * mgr);

// Session save/restore
bool                 kv_cache_manager_save_session(const kv_cache_manager_t * mgr, const char * path);
bool                 kv_cache_manager_load_session(kv_cache_manager_t * mgr, const char * path);

// Get default config
kv_cache_config      kv_cache_manager_default_config(void);

#ifdef __cplusplus
}
#endif
