#pragma once

/**
 * GGMLEngine internal header - shared between ggml-engine.cpp and ggml-engine-vlm.cpp
 *
 * Contains the ggml_engine struct definition and the shared generation loop.
 */

#include "ggml-engine.h"
#include "llama.h"
#include "common.h"
#include "sampling.h"

#include <string>
#include <vector>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdio>

struct ggml_engine {
    ggml_engine_params   params;
    struct llama_model  * model   = nullptr;
    struct llama_context * ctx    = nullptr;
    const struct llama_vocab * vocab = nullptr;

    // generation state
    std::string           response;
    std::atomic<bool>     cancelled{false};
    ggml_engine_perf      perf{};

    // context tracking
    int32_t               n_past = 0;
};

// Internal helper: duplicate string with malloc
static inline char * strdup_alloc(const std::string & s) {
    char * p = (char *)malloc(s.size() + 1);
    if (p) {
        memcpy(p, s.c_str(), s.size() + 1);
    }
    return p;
}

/**
 * Shared generation loop: sets up sampler, runs autoregressive decode from current n_past.
 * Called by both ggml_engine_generate() (after text prompt processing)
 * and ggml_engine_vlm_generate() (after multimodal chunk processing).
 *
 * Expects engine->ctx to have logits ready at position (n_past - 1).
 */
static inline ggml_engine_status ggml_engine_generate_loop(
    ggml_engine_t * engine,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback,
    void * user_data
) {
    const int n_ctx = llama_n_ctx(engine->ctx);

    // setup sampler
    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (sampling.repeat_penalty != 1.0f || sampling.frequency_penalty != 0.0f || sampling.presence_penalty != 0.0f) {
        llama_sampler_chain_add(smpl,
            llama_sampler_init_penalties(
                sampling.repeat_last_n,
                sampling.repeat_penalty,
                sampling.frequency_penalty,
                sampling.presence_penalty));
    }

    if (sampling.top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(sampling.top_k));
    }
    if (sampling.top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(sampling.top_p, 1));
    }
    if (sampling.min_p > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(sampling.min_p, 1));
    }
    if (sampling.temperature > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(sampling.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(sampling.seed));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    // generate tokens
    int64_t t_gen_start = llama_time_us();
    int n_generated = 0;
    int max_tokens = sampling.n_predict > 0 ? sampling.n_predict : n_ctx - engine->n_past;
    engine->response.reserve(max_tokens * 4);

    struct llama_batch batch = llama_batch_init(engine->params.n_batch, 0, 1);

    while (n_generated < max_tokens) {
        if (engine->cancelled.load()) {
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            return GGML_ENGINE_ERROR_CANCELLED;
        }

        llama_token new_token = llama_sampler_sample(smpl, engine->ctx, -1);

        // check for end of generation
        if (llama_vocab_is_eog(engine->vocab, new_token)) {
            break;
        }

        // convert token to text
        char buf[256];
        int n = llama_token_to_piece(engine->vocab, new_token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            n = 0;
        }
        std::string piece(buf, n);

        engine->response += piece;
        n_generated++;

        // check stop sequences — windowed search
        bool should_stop = false;
        for (int s = 0; s < sampling.stop_sequence_count && s < 8; s++) {
            if (!sampling.stop_sequences[s]) continue;
            size_t stop_len = strlen(sampling.stop_sequences[s]);
            if (stop_len == 0 || engine->response.size() < stop_len) continue;

            size_t window = stop_len + (size_t)n;
            size_t from = engine->response.size() > window ? engine->response.size() - window : 0;
            size_t pos = engine->response.find(sampling.stop_sequences[s], from);
            if (pos != std::string::npos) {
                engine->response.erase(pos);
                should_stop = true;
                break;
            }
        }

        if (should_stop) break;

        // callback
        if (callback) {
            if (!callback(piece.c_str(), user_data)) {
                break;
            }
        }

        // prepare next batch
        common_batch_clear(batch);
        common_batch_add(batch, new_token, engine->n_past, {0}, true);

        if (llama_decode(engine->ctx, batch) != 0) {
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            return GGML_ENGINE_ERROR_DECODE;
        }
        engine->n_past++;
    }

    int64_t t_gen_end = llama_time_us();
    engine->perf.generation_ms = (t_gen_end - t_gen_start) / 1000.0;
    engine->perf.generated_tokens = n_generated;

    if (engine->perf.prompt_eval_ms > 0) {
        engine->perf.prompt_tokens_per_sec =
            engine->perf.prompt_tokens / (engine->perf.prompt_eval_ms / 1000.0);
    }
    if (engine->perf.generation_ms > 0) {
        engine->perf.generation_tokens_per_sec =
            engine->perf.generated_tokens / (engine->perf.generation_ms / 1000.0);
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);

    return GGML_ENGINE_OK;
}
