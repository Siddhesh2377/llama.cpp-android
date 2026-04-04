#include "ggml-engine-internal.h"
#include "tn-log.h"

#include <sstream>
#include <thread>

#ifdef __ANDROID__
#include <sys/sysinfo.h>
#include <unistd.h>
#include <sched.h>
#endif

static int detect_optimal_threads() {
#ifdef __ANDROID__
    int n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    int n_threads = (n_cpus * 3) / 4;
    return n_threads > 0 ? n_threads : 1;
#else
    int n = std::thread::hardware_concurrency();
    return n > 0 ? n : 4;
#endif
}

ggml_engine_params ggml_engine_default_params(void) {
    ggml_engine_params p{};
    p.n_ctx            = 0;
    p.n_batch          = 512;
    p.n_threads        = 0;
    p.n_threads_batch  = 0;
    p.use_mmap         = true;
    p.use_mlock        = false;
    p.n_gpu_layers     = 0;
    p.rope_freq_base   = 0.0f;
    p.rope_freq_scale  = 0.0f;
    p.flash_attn       = true;
    return p;
}

ggml_engine_sampling ggml_engine_default_sampling(void) {
    ggml_engine_sampling s{};
    s.temperature       = 0.7f;
    s.top_k             = 40;
    s.top_p             = 0.95f;
    s.min_p             = 0.05f;
    s.repeat_penalty    = 1.1f;
    s.repeat_last_n     = 64;
    s.frequency_penalty = 0.0f;
    s.presence_penalty  = 0.0f;
    s.seed              = 0xFFFFFFFF;
    s.n_predict         = 256;
    s.stop_sequence_count = 0;
    memset(s.stop_sequences, 0, sizeof(s.stop_sequences));
    return s;
}

ggml_engine_t * ggml_engine_create(ggml_engine_params params) {
    llama_backend_init();

    auto * engine = new ggml_engine();
    engine->params = params;

    if (engine->params.n_threads <= 0) {
        engine->params.n_threads = detect_optimal_threads();
    }
    if (engine->params.n_threads_batch <= 0) {
        engine->params.n_threads_batch = engine->params.n_threads;
    }

    return engine;
}

void ggml_engine_free(ggml_engine_t * engine) {
    if (!engine) return;
    ggml_engine_unload_model(engine);
    delete engine;
}

ggml_engine_status ggml_engine_load_model(ggml_engine_t * engine, const char * path) {
    if (!engine || !path) return GGML_ENGINE_ERROR_LOAD_FAILED;

    ggml_engine_unload_model(engine);

    auto mparams = llama_model_default_params();
    mparams.use_mmap  = engine->params.use_mmap;
    mparams.use_mlock = engine->params.use_mlock;

    engine->model = llama_model_load_from_file(path, mparams);
    if (!engine->model) {
        return GGML_ENGINE_ERROR_LOAD_FAILED;
    }

    engine->vocab = llama_model_get_vocab(engine->model);

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = engine->params.n_ctx > 0 ? engine->params.n_ctx : 0;
    cparams.n_batch         = engine->params.n_batch;
    cparams.n_threads       = engine->params.n_threads;
    cparams.n_threads_batch = engine->params.n_threads_batch;
    if (engine->params.flash_attn) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    if (engine->params.rope_freq_base > 0.0f) {
        cparams.rope_freq_base = engine->params.rope_freq_base;
    }
    if (engine->params.rope_freq_scale > 0.0f) {
        cparams.rope_freq_scale = engine->params.rope_freq_scale;
    }

    engine->ctx = llama_init_from_model(engine->model, cparams);
    if (!engine->ctx) {
        llama_model_free(engine->model);
        engine->model = nullptr;
        engine->vocab = nullptr;
        return GGML_ENGINE_ERROR_CONTEXT_FAIL;
    }

    engine->n_past = 0;
    return GGML_ENGINE_OK;
}

ggml_engine_status ggml_engine_load_model_from_fd(ggml_engine_t * engine, int fd) {
    if (!engine || fd < 0) return GGML_ENGINE_ERROR_LOAD_FAILED;

    // Android SAF: access file via /proc/self/fd/<fd>
    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", fd);

    return ggml_engine_load_model(engine, fd_path);
}

void ggml_engine_unload_model(ggml_engine_t * engine) {
    if (!engine) return;

    if (engine->ctx) {
        llama_free(engine->ctx);
        engine->ctx = nullptr;
    }
    if (engine->model) {
        llama_model_free(engine->model);
        engine->model = nullptr;
    }
    engine->vocab = nullptr;
    engine->n_past = 0;
    engine->response.clear();
}

bool ggml_engine_is_loaded(const ggml_engine_t * engine) {
    return engine && engine->model && engine->ctx;
}

char * ggml_engine_model_info_json(const ggml_engine_t * engine) {
    if (!engine || !engine->model) return strdup_alloc("{}");

    const auto * model = engine->model;
    const auto * vocab = engine->vocab;

    char desc[256] = {0};
    llama_model_desc(model, desc, sizeof(desc));

    std::ostringstream json;
    json << "{\n";
    json << "  \"description\": \"" << desc << "\",\n";
    json << "  \"size_bytes\": " << llama_model_size(model) << ",\n";
    json << "  \"n_params\": " << llama_model_n_params(model) << ",\n";
    json << "  \"n_embd\": " << llama_model_n_embd(model) << ",\n";
    json << "  \"n_layer\": " << llama_model_n_layer(model) << ",\n";
    json << "  \"n_head\": " << llama_model_n_head(model) << ",\n";
    json << "  \"n_head_kv\": " << llama_model_n_head_kv(model) << ",\n";
    json << "  \"n_ctx_train\": " << llama_model_n_ctx_train(model) << ",\n";
    json << "  \"n_vocab\": " << llama_vocab_n_tokens(vocab) << ",\n";
    json << "  \"has_encoder\": " << (llama_model_has_encoder(model) ? "true" : "false") << ",\n";
    json << "  \"has_decoder\": " << (llama_model_has_decoder(model) ? "true" : "false") << ",\n";
    json << "  \"is_recurrent\": " << (llama_model_is_recurrent(model) ? "true" : "false") << ",\n";

    if (engine->ctx) {
        json << "  \"n_ctx\": " << llama_n_ctx(engine->ctx) << ",\n";
        json << "  \"n_batch\": " << llama_n_batch(engine->ctx) << ",\n";
    }

    json << "  \"metadata\": {\n";
    int n_meta = llama_model_meta_count(model);
    for (int i = 0; i < n_meta; i++) {
        char key[256] = {0};
        char val[512] = {0};
        llama_model_meta_key_by_index(model, i, key, sizeof(key));
        llama_model_meta_val_str_by_index(model, i, val, sizeof(val));

        std::string escaped_val;
        for (const char * p = val; *p; p++) {
            if (*p == '"') escaped_val += "\\\"";
            else if (*p == '\\') escaped_val += "\\\\";
            else if (*p == '\n') escaped_val += "\\n";
            else escaped_val += *p;
        }

        json << "    \"" << key << "\": \"" << escaped_val << "\"";
        if (i < n_meta - 1) json << ",";
        json << "\n";
    }
    json << "  }\n";
    json << "}";

    return strdup_alloc(json.str());
}

void ggml_engine_free_string(char * str) {
    free(str);
}

static ggml_engine_status ggml_engine_generate_impl(
    ggml_engine_t * engine,
    const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback,
    void * user_data,
    bool clear_kv
) {
    if (!engine || !engine->model || !engine->ctx) {
        return GGML_ENGINE_ERROR_NO_MODEL;
    }

    engine->cancelled.store(false);
    engine->response.clear();
    memset(&engine->perf, 0, sizeof(engine->perf));

    const int n_ctx = llama_n_ctx(engine->ctx);

    std::vector<llama_token> tokens = common_tokenize(engine->vocab, prompt, true, true);
    int n_prompt = (int)tokens.size();

    if (n_prompt < 1) {
        return GGML_ENGINE_ERROR_TOKENIZE;
    }

    if (clear_kv) {
        llama_memory_t mem = llama_get_memory(engine->ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
        engine->n_past = 0;
    }

    int remaining = n_ctx - engine->n_past;
    if (n_prompt > remaining) {
        return GGML_ENGINE_ERROR_OUT_OF_MEM;
    }

    int space_after_prompt = remaining - n_prompt;
    if (sampling.n_predict < 0) {
        sampling.n_predict = space_after_prompt;
    } else {
        sampling.n_predict = std::min(sampling.n_predict, space_after_prompt);
    }

    int64_t t_prompt_start = llama_time_us();

    struct llama_batch batch = llama_batch_init(engine->params.n_batch, 0, 1);

    for (int i = 0; i < n_prompt; i += engine->params.n_batch) {
        int n_eval = std::min(engine->params.n_batch, n_prompt - i);

        common_batch_clear(batch);
        for (int j = 0; j < n_eval; j++) {
            common_batch_add(batch, tokens[i + j], engine->n_past + j, {0}, (i + j == n_prompt - 1));
        }

        if (llama_decode(engine->ctx, batch) != 0) {
            llama_batch_free(batch);
            return GGML_ENGINE_ERROR_DECODE;
        }
        engine->n_past += n_eval;
    }

    llama_batch_free(batch);

    int64_t t_prompt_end = llama_time_us();
    engine->perf.prompt_eval_ms = (t_prompt_end - t_prompt_start) / 1000.0;
    engine->perf.prompt_tokens = n_prompt;

    return ggml_engine_generate_loop(engine, sampling, callback, user_data);
}

ggml_engine_status ggml_engine_generate(
    ggml_engine_t * engine,
    const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback,
    void * user_data
) {
    return ggml_engine_generate_impl(engine, prompt, sampling, callback, user_data, true);
}

ggml_engine_status ggml_engine_generate_continue(
    ggml_engine_t * engine,
    const char * prompt,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback,
    void * user_data
) {
    return ggml_engine_generate_impl(engine, prompt, sampling, callback, user_data, false);
}

void ggml_engine_cancel(ggml_engine_t * engine) {
    if (engine) {
        engine->cancelled.store(true);
    }
}

char * ggml_engine_get_response(const ggml_engine_t * engine) {
    if (!engine) return strdup_alloc("");
    return strdup_alloc(engine->response);
}

void ggml_engine_clear_context(ggml_engine_t * engine) {
    if (!engine || !engine->ctx) return;

    llama_memory_t mem = llama_get_memory(engine->ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    engine->n_past = 0;
    engine->response.clear();
}

int32_t ggml_engine_context_used(const ggml_engine_t * engine) {
    if (!engine) return 0;
    return engine->n_past;
}

int32_t ggml_engine_context_size(const ggml_engine_t * engine) {
    if (!engine || !engine->ctx) return 0;
    return llama_n_ctx(engine->ctx);
}

int32_t ggml_engine_context_remaining(const ggml_engine_t * engine) {
    if (!engine || !engine->ctx) return 0;
    return llama_n_ctx(engine->ctx) - engine->n_past;
}

ggml_engine_context_info ggml_engine_context_status(const ggml_engine_t * engine,
                                                     const char * prompt) {
    ggml_engine_context_info info = {};
    if (!engine || !engine->ctx) return info;

    info.total     = llama_n_ctx(engine->ctx);
    info.used      = engine->n_past;
    info.remaining = info.total - info.used;

    if (prompt && engine->vocab) {
        std::vector<llama_token> tokens = common_tokenize(engine->vocab, prompt, true, true);
        info.prompt_estimate = (int32_t)tokens.size();
        info.after_prompt    = info.remaining - info.prompt_estimate;
        if (info.after_prompt < 0) info.after_prompt = 0;
    } else {
        info.prompt_estimate = -1;
        info.after_prompt    = -1;
    }

    return info;
}

int32_t ggml_engine_tokenize(const ggml_engine_t * engine,
                              const char * text, int32_t * tokens, int32_t max_tokens) {
    if (!engine || !engine->vocab || !text || !tokens) return -1;

    std::vector<llama_token> tok = common_tokenize(engine->vocab, text, false, false);
    int32_t n = (int32_t)tok.size();
    int32_t copy = n < max_tokens ? n : max_tokens;
    memcpy(tokens, tok.data(), copy * sizeof(int32_t));
    return n;
}

char * ggml_engine_detokenize(const ggml_engine_t * engine,
                               const int32_t * tokens, int32_t n_tokens) {
    if (!engine || !engine->vocab || !tokens || n_tokens <= 0) {
        return strdup_alloc("");
    }

    std::string result;
    for (int32_t i = 0; i < n_tokens; i++) {
        char buf[256];
        int n = llama_token_to_piece(engine->vocab, tokens[i], buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }
    }
    return strdup_alloc(result);
}

ggml_engine_vectors * ggml_engine_calc_vectors(
    ggml_engine_t * engine,
    const char * prompt,
    ggml_engine_progress_cb progress,
    void * user_data
) {
    if (!engine || !engine->model || !prompt) return nullptr;

    if (progress) progress(0.05f, user_data);

    // Temporary embedding context, does not touch main KV cache
    auto cparams = llama_context_default_params();
    cparams.n_ctx           = engine->params.n_ctx > 0 ? engine->params.n_ctx : 2048;
    cparams.n_batch         = engine->params.n_batch;
    cparams.n_threads       = engine->params.n_threads;
    cparams.n_threads_batch = engine->params.n_threads_batch;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_MEAN;

    struct llama_context * emb_ctx = llama_init_from_model(engine->model, cparams);
    if (!emb_ctx) return nullptr;

    if (progress) progress(0.10f, user_data);

    std::vector<llama_token> tokens = common_tokenize(engine->vocab, prompt, true, true);
    int n_tokens = (int)tokens.size();
    if (n_tokens < 1) {
        llama_free(emb_ctx);
        return nullptr;
    }

    int n_ctx = llama_n_ctx(emb_ctx);
    if (n_tokens > n_ctx) {
        n_tokens = n_ctx;
        tokens.resize(n_tokens);
    }

    if (progress) progress(0.15f, user_data);

    struct llama_batch batch = llama_batch_init(engine->params.n_batch, 0, 1);
    int n_past = 0;

    for (int i = 0; i < n_tokens; i += engine->params.n_batch) {
        int n_eval = std::min(engine->params.n_batch, n_tokens - i);

        common_batch_clear(batch);
        for (int j = 0; j < n_eval; j++) {
            common_batch_add(batch, tokens[i + j], n_past + j, {0}, true);
        }

        if (llama_decode(emb_ctx, batch) != 0) {
            llama_batch_free(batch);
            llama_free(emb_ctx);
            return nullptr;
        }
        n_past += n_eval;

        float p = 0.15f + 0.70f * ((float)(i + n_eval) / n_tokens);
        if (progress) progress(p, user_data);
    }

    llama_batch_free(batch);

    int n_embd = llama_model_n_embd(engine->model);
    const float * embd = llama_get_embeddings_seq(emb_ctx, 0);

    // Fallback to last token if pooling didn't produce output
    if (!embd) {
        embd = llama_get_embeddings_ith(emb_ctx, -1);
    }

    if (!embd) {
        llama_free(emb_ctx);
        return nullptr;
    }

    if (progress) progress(0.90f, user_data);

    auto * result = (ggml_engine_vectors *)malloc(sizeof(ggml_engine_vectors));
    if (!result) {
        llama_free(emb_ctx);
        return nullptr;
    }

    result->n_embd   = n_embd;
    result->n_tokens  = n_tokens;
    result->data      = (float *)malloc(n_embd * sizeof(float));
    if (!result->data) {
        free(result);
        llama_free(emb_ctx);
        return nullptr;
    }
    memcpy(result->data, embd, n_embd * sizeof(float));

    llama_free(emb_ctx);

    if (progress) progress(1.0f, user_data);
    return result;
}

void ggml_engine_free_vectors(ggml_engine_vectors * v) {
    if (!v) return;
    free(v->data);
    free(v);
}

bool ggml_engine_apply_vectors(
    ggml_engine_t * engine,
    const ggml_engine_vectors * vectors,
    float strength,
    int32_t il_start,
    int32_t il_end
) {
    if (!engine || !engine->ctx || !engine->model || !vectors || !vectors->data) return false;

    int n_embd  = llama_model_n_embd(engine->model);
    int n_layer = llama_model_n_layer(engine->model);

    if (vectors->n_embd != n_embd) return false;

    if (il_start < 0) il_start = 0;
    if (il_end   < 0) il_end   = n_layer - 1;
    if (il_end >= n_layer) il_end = n_layer - 1;
    if (il_start > il_end) return false;

    int n_apply = il_end - il_start + 1;

    // Replicate vector across target layers, scaled by strength
    std::vector<float> cv_data(n_apply * n_embd);
    for (int l = 0; l < n_apply; l++) {
        for (int i = 0; i < n_embd; i++) {
            cv_data[l * n_embd + i] = vectors->data[i] * strength;
        }
    }

    int32_t rc = llama_set_adapter_cvec(
        engine->ctx,
        cv_data.data(),
        cv_data.size(),
        n_embd,
        il_start,
        il_end
    );

    return rc == 0;
}

void ggml_engine_clear_vectors(ggml_engine_t * engine) {
    if (!engine || !engine->ctx) return;

    int n_layer = llama_model_n_layer(engine->model);
    llama_set_adapter_cvec(engine->ctx, nullptr, 0, 0, 0, n_layer - 1);
}

ggml_engine_perf ggml_engine_get_perf(const ggml_engine_t * engine) {
    if (!engine) {
        return {};
    }
    return engine->perf;
}

void tn_engine_set_log_callback(tn_engine_log_callback cb, void * user_data) {
    tn_log_set_callback(reinterpret_cast<tn_log_callback>(cb), user_data);
}

void tn_engine_set_log_level(tn_engine_log_level max_level) {
    tn_log_set_level(static_cast<tn_log_level>(max_level));
}
