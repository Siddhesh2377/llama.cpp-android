#include "kv-cache-manager.h"
#include "ggml-engine.h"
#include "llama.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

// Access engine internals
struct ggml_engine;

struct kv_cache_manager {
    kv_cache_config      config;
    ggml_engine_t      * engine     = nullptr;
    struct llama_context * ctx      = nullptr;
    llama_memory_t       memory     = nullptr;
    int32_t              capacity   = 0;
};

kv_cache_config kv_cache_manager_default_config(void) {
    kv_cache_config cfg = {};
    cfg.strategy = KV_STRATEGY_SLIDING_WINDOW;
    cfg.max_cache_tokens = 0;        // use full context
    cfg.keep_first_n = 4;            // keep BOS/system prompt tokens
    cfg.sliding_window_size = 512;   // keep last 512 tokens
    cfg.memory_pressure_threshold = 0.9f;
    cfg.enable_auto_trim = true;
    return cfg;
}

kv_cache_manager_t * kv_cache_manager_create(kv_cache_config config) {
    auto * mgr = new kv_cache_manager();
    mgr->config = config;
    return mgr;
}

void kv_cache_manager_free(kv_cache_manager_t * mgr) {
    delete mgr;
}

void kv_cache_manager_attach(kv_cache_manager_t * mgr, ggml_engine_t * engine) {
    if (!mgr || !engine) return;
    mgr->engine = engine;

    // We need access to the llama_context. Since our engine is opaque,
    // we use the public API to get context info.
    mgr->capacity = ggml_engine_context_size(engine);
    if (mgr->config.max_cache_tokens > 0 && mgr->config.max_cache_tokens < mgr->capacity) {
        mgr->capacity = mgr->config.max_cache_tokens;
    }
}

int32_t kv_cache_manager_tokens_used(const kv_cache_manager_t * mgr) {
    if (!mgr || !mgr->engine) return 0;
    return ggml_engine_context_used(mgr->engine);
}

int32_t kv_cache_manager_tokens_capacity(const kv_cache_manager_t * mgr) {
    if (!mgr) return 0;
    return mgr->capacity;
}

float kv_cache_manager_usage_ratio(const kv_cache_manager_t * mgr) {
    if (!mgr || mgr->capacity <= 0) return 0.0f;
    return (float)kv_cache_manager_tokens_used(mgr) / (float)mgr->capacity;
}

bool kv_cache_manager_trim(kv_cache_manager_t * mgr, int32_t n_keep) {
    if (!mgr || !mgr->engine) return false;

    int32_t used = ggml_engine_context_used(mgr->engine);
    if (used <= n_keep) return true;  // nothing to trim

    // Use the engine's clear + we'd need to re-process the kept tokens
    // For now, we clear and the caller should re-feed important context
    ggml_engine_clear_context(mgr->engine);
    return true;
}

void kv_cache_manager_clear(kv_cache_manager_t * mgr) {
    if (!mgr || !mgr->engine) return;
    ggml_engine_clear_context(mgr->engine);
}

int32_t kv_cache_manager_shift(kv_cache_manager_t * mgr) {
    if (!mgr || !mgr->engine) return 0;

    int32_t used = ggml_engine_context_used(mgr->engine);
    int32_t cap = mgr->capacity;

    if (used < cap * mgr->config.memory_pressure_threshold) {
        return 0;  // no need to shift yet
    }

    // Calculate how much to free
    int32_t target = cap / 2;  // free to 50% capacity
    int32_t to_free = used - target;

    if (to_free <= 0) return 0;

    // For sliding window: keep first N (system prompt) + last window_size
    int32_t keep_first = mgr->config.keep_first_n;
    int32_t keep_last = mgr->config.sliding_window_size;

    // we need to clear and re-process. return how many we'd free.
    // actual re-processing would need the original tokens which we don't store.
    // Signal to the caller how much space they'd gain.
    ggml_engine_clear_context(mgr->engine);

    return used;  // freed all tokens
}

bool kv_cache_manager_save_session(const kv_cache_manager_t * mgr, const char * path) {
    if (!mgr || !mgr->engine || !path) return false;

    // Use llama's built-in session save/load
    // We need access to the llama_context - access through the engine
    // For now, save the context state as a binary file
    // The engine would need to expose this, so we use a simplified approach

    FILE * f = fopen(path, "wb");
    if (!f) return false;

    // Write a header
    int32_t magic = 0x544E5353;  // "TNSS" - Tool Neuron Session Save
    int32_t version = 1;
    int32_t tokens_used = ggml_engine_context_used(mgr->engine);

    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&tokens_used, sizeof(tokens_used), 1, f);

    // In a full implementation, we'd serialize the KV cache state here
    // For now, we save the metadata so we know what to expect on restore

    fclose(f);
    return true;
}

bool kv_cache_manager_load_session(kv_cache_manager_t * mgr, const char * path) {
    if (!mgr || !mgr->engine || !path) return false;

    FILE * f = fopen(path, "rb");
    if (!f) return false;

    int32_t magic, version, tokens_used;
    if (fread(&magic, sizeof(magic), 1, f) != 1 ||
        fread(&version, sizeof(version), 1, f) != 1 ||
        fread(&tokens_used, sizeof(tokens_used), 1, f) != 1) {
        fclose(f);
        return false;
    }

    if (magic != 0x544E5353 || version != 1) {
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}
