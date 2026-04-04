#include "rag-engine.h"
#include "llama.h"
#include "common.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <new>

static char * rag_strdup(const char * s) {
    if (!s) return nullptr;
    size_t len = strlen(s);
    char * d = (char *)malloc(len + 1);
    if (d) memcpy(d, s, len + 1);
    return d;
}

struct rag_chunk {
    std::string              text;
    std::string              doc_id;
    int32_t                  chunk_index;
    std::vector<float>       embedding;
    std::vector<uint64_t>    bq_vector;
};

struct rag_document {
    std::string              doc_id;
    int32_t                  first_chunk;
    int32_t                  n_chunks;
};

struct rag_engine {
    rag_engine_params        params;

    llama_model            * model   = nullptr;
    llama_context          * ctx     = nullptr;
    const llama_vocab      * vocab   = nullptr;
    int32_t                  n_embd  = 0;

    std::vector<rag_chunk>       chunks;
    std::vector<rag_document>    documents;
    std::unordered_map<std::string, int32_t> doc_index;
};

rag_engine_params rag_engine_default_params(void) {
    return {
        /* n_threads      */ 0,
        /* chunk_size     */ 256,
        /* chunk_overlap  */ 32,
        /* n_dims         */ 256,
        /* top_k          */ 32,
        /* top_n          */ 5,
        /* late_chunking  */ true,
    };
}

rag_engine_t * rag_engine_create(rag_engine_params params) {
    auto * engine = new (std::nothrow) rag_engine();
    if (!engine) return nullptr;
    engine->params = params;
    return engine;
}

void rag_engine_free(rag_engine_t * engine) {
    if (!engine) return;
    if (engine->ctx)   llama_free(engine->ctx);
    if (engine->model) llama_model_free(engine->model);
    delete engine;
}

bool rag_engine_is_loaded(const rag_engine_t * engine) {
    return engine && engine->model && engine->ctx;
}

int32_t rag_engine_document_count(const rag_engine_t * engine) {
    return engine ? (int32_t)engine->documents.size() : 0;
}

int32_t rag_engine_chunk_count(const rag_engine_t * engine) {
    return engine ? (int32_t)engine->chunks.size() : 0;
}

static int32_t rag_load_model_impl(rag_engine_t * engine, llama_model * model) {
    if (!model) return -1;

    if (engine->ctx)   { llama_free(engine->ctx); engine->ctx = nullptr; }
    if (engine->model) { llama_model_free(engine->model); engine->model = nullptr; }

    engine->model = model;
    engine->vocab = llama_model_get_vocab(model);
    engine->n_embd = llama_model_n_embd(model);

    // POOLING_TYPE_NONE for raw per-token embeddings
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx        = 2048;
    ctx_params.n_batch      = 2048;
    ctx_params.n_ubatch     = 2048;
    ctx_params.n_threads    = engine->params.n_threads > 0 ? engine->params.n_threads : 4;
    ctx_params.embeddings   = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;

    engine->ctx = llama_init_from_model(model, ctx_params);
    if (!engine->ctx) {
        llama_model_free(engine->model);
        engine->model = nullptr;
        return -1;
    }

    return 0;
}

int32_t rag_engine_load_model(rag_engine_t * engine, const char * path) {
    if (!engine || !path) return -1;

    auto model_params = llama_model_default_params();
    model_params.use_mmap = true;
    llama_model * model = llama_model_load_from_file(path, model_params);

    return rag_load_model_impl(engine, model);
}

int32_t rag_engine_load_model_from_fd(rag_engine_t * engine, int fd) {
    if (!engine || fd < 0) return -1;

    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", fd);

    auto model_params = llama_model_default_params();
    model_params.use_mmap = true;
    llama_model * model = llama_model_load_from_file(fd_path, model_params);

    return rag_load_model_impl(engine, model);
}

static std::vector<llama_token> rag_tokenize(const rag_engine_t * engine, const char * text) {
    return common_tokenize(engine->vocab, text, true, false);
}

// Encode tokens and extract raw per-token embeddings [n_tokens x n_embd]
static std::vector<float> rag_encode_tokens(
    rag_engine_t * engine,
    const llama_token * tokens, int32_t n_tokens
) {
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = true;
    }
    batch.n_tokens = n_tokens;

    if (llama_encode(engine->ctx, batch) != 0) {
        llama_batch_free(batch);
        return {};
    }

    std::vector<float> embeddings(n_tokens * engine->n_embd);
    for (int32_t i = 0; i < n_tokens; i++) {
        const float * emb = llama_get_embeddings_ith(engine->ctx, i);
        if (!emb) {
            llama_batch_free(batch);
            return {};
        }
        memcpy(embeddings.data() + i * engine->n_embd, emb, engine->n_embd * sizeof(float));
    }

    llama_batch_free(batch);
    return embeddings;
}

// Mean-pool token embeddings [start, end), truncate to n_dims (Matryoshka), L2-normalize
static std::vector<float> rag_mean_pool_and_truncate(
    const float * token_embeddings, int32_t n_embd,
    int32_t start, int32_t end, int32_t n_dims
) {
    int32_t dims = std::min(n_dims, n_embd);
    std::vector<float> result(dims, 0.0f);
    int32_t count = end - start;
    if (count <= 0) return result;

    for (int32_t i = start; i < end; i++) {
        const float * tok = token_embeddings + i * n_embd;
        for (int32_t d = 0; d < dims; d++) {
            result[d] += tok[d];
        }
    }

    float inv = 1.0f / (float)count;
    for (int32_t d = 0; d < dims; d++) {
        result[d] *= inv;
    }

    // L2 normalize
    float norm = 0.0f;
    for (int32_t d = 0; d < dims; d++) {
        norm += result[d] * result[d];
    }
    norm = sqrtf(norm);
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / norm;
        for (int32_t d = 0; d < dims; d++) {
            result[d] *= inv_norm;
        }
    }

    return result;
}

// Embed a short text (query)
static std::vector<float> rag_embed_text(rag_engine_t * engine, const char * text) {
    auto tokens = rag_tokenize(engine, text);
    if (tokens.empty()) return {};

    auto tok_embs = rag_encode_tokens(engine, tokens.data(), (int32_t)tokens.size());
    if (tok_embs.empty()) return {};

    return rag_mean_pool_and_truncate(
        tok_embs.data(), engine->n_embd,
        0, (int32_t)tokens.size(), engine->params.n_dims
    );
}

// Float vector -> binary vector (threshold at 0)
static std::vector<uint64_t> rag_bq_quantize(const float * vec, int32_t n_dims) {
    int32_t n_words = (n_dims + 63) / 64;
    std::vector<uint64_t> bq(n_words, 0);

    for (int32_t i = 0; i < n_dims; i++) {
        if (vec[i] > 0.0f) {
            bq[i / 64] |= (1ULL << (i % 64));
        }
    }
    return bq;
}

static int32_t rag_hamming_distance(
    const uint64_t * a, const uint64_t * b, int32_t n_words
) {
    int32_t dist = 0;
    for (int32_t i = 0; i < n_words; i++) {
        dist += __builtin_popcountll(a[i] ^ b[i]);
    }
    return dist;
}

// Dot product of L2-normalized vectors = cosine similarity
static float rag_cosine_similarity(const float * a, const float * b, int32_t n_dims) {
    float dot = 0.0f;
    for (int32_t i = 0; i < n_dims; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

// BQ coarse search: top_k candidates by Hamming distance
static std::vector<std::pair<int32_t, int32_t>> rag_bq_search(
    const std::vector<uint64_t> & query_bq,
    const std::vector<rag_chunk> & chunks,
    int32_t top_k
) {
    int32_t n_words = (int32_t)query_bq.size();
    std::vector<std::pair<int32_t, int32_t>> distances;
    distances.reserve(chunks.size());

    for (int32_t i = 0; i < (int32_t)chunks.size(); i++) {
        int32_t d = rag_hamming_distance(
            query_bq.data(), chunks[i].bq_vector.data(), n_words);
        distances.push_back({d, i});
    }

    int32_t k = std::min(top_k, (int32_t)distances.size());
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
    distances.resize(k);
    return distances;
}

// Late chunking: encode full document with sliding windows, then chunk token embeddings
static int32_t rag_index_document_late(
    rag_engine_t * engine, const char * text, const char * doc_id
) {
    auto tokens = rag_tokenize(engine, text);
    if (tokens.empty()) return -1;

    int32_t n_tokens = (int32_t)tokens.size();
    int32_t ctx_window = 2048;
    int32_t window_overlap = 256;

    std::vector<float> all_embs(n_tokens * engine->n_embd, 0.0f);
    std::vector<int32_t> emb_counts(n_tokens, 0);

    for (int32_t start = 0; start < n_tokens; start += (ctx_window - window_overlap)) {
        int32_t end = std::min(start + ctx_window, n_tokens);
        int32_t window_len = end - start;

        auto window_embs = rag_encode_tokens(engine, tokens.data() + start, window_len);
        if (window_embs.empty()) return -1;

        for (int32_t i = 0; i < window_len; i++) {
            int32_t gi = start + i;
            for (int32_t d = 0; d < engine->n_embd; d++) {
                all_embs[gi * engine->n_embd + d] += window_embs[i * engine->n_embd + d];
            }
            emb_counts[gi]++;
        }

        if (end >= n_tokens) break;
    }

    // Average overlapping window regions
    for (int32_t i = 0; i < n_tokens; i++) {
        if (emb_counts[i] > 1) {
            float inv = 1.0f / (float)emb_counts[i];
            for (int32_t d = 0; d < engine->n_embd; d++) {
                all_embs[i * engine->n_embd + d] *= inv;
            }
        }
    }

    int32_t chunk_size    = engine->params.chunk_size;
    int32_t chunk_overlap = engine->params.chunk_overlap;
    int32_t first_chunk   = (int32_t)engine->chunks.size();
    int32_t n_chunks      = 0;

    for (int32_t tok_start = 0; tok_start < n_tokens; tok_start += (chunk_size - chunk_overlap)) {
        int32_t tok_end = std::min(tok_start + chunk_size, n_tokens);

        auto embedding = rag_mean_pool_and_truncate(
            all_embs.data(), engine->n_embd,
            tok_start, tok_end, engine->params.n_dims
        );
        auto bq = rag_bq_quantize(embedding.data(), engine->params.n_dims);

        std::string chunk_text;
        for (int32_t t = tok_start; t < tok_end; t++) {
            chunk_text += common_token_to_piece(engine->ctx, tokens[t]);
        }

        rag_chunk chunk;
        chunk.text        = std::move(chunk_text);
        chunk.doc_id      = doc_id;
        chunk.chunk_index = n_chunks;
        chunk.embedding   = std::move(embedding);
        chunk.bq_vector   = std::move(bq);

        engine->chunks.push_back(std::move(chunk));
        n_chunks++;

        if (tok_end >= n_tokens) break;
    }

    rag_document doc;
    doc.doc_id      = doc_id;
    doc.first_chunk = first_chunk;
    doc.n_chunks    = n_chunks;

    engine->doc_index[doc_id] = (int32_t)engine->documents.size();
    engine->documents.push_back(std::move(doc));

    return n_chunks;
}

// Naive fallback: chunk first, embed each chunk independently
static int32_t rag_index_document_naive(
    rag_engine_t * engine, const char * text, const char * doc_id
) {
    auto tokens = rag_tokenize(engine, text);
    if (tokens.empty()) return -1;

    int32_t n_tokens      = (int32_t)tokens.size();
    int32_t chunk_size    = engine->params.chunk_size;
    int32_t chunk_overlap = engine->params.chunk_overlap;
    int32_t first_chunk   = (int32_t)engine->chunks.size();
    int32_t n_chunks      = 0;

    for (int32_t tok_start = 0; tok_start < n_tokens; tok_start += (chunk_size - chunk_overlap)) {
        int32_t tok_end = std::min(tok_start + chunk_size, n_tokens);
        int32_t window_len = tok_end - tok_start;

        auto tok_embs = rag_encode_tokens(engine, tokens.data() + tok_start, window_len);
        if (tok_embs.empty()) return -1;

        auto embedding = rag_mean_pool_and_truncate(
            tok_embs.data(), engine->n_embd,
            0, window_len, engine->params.n_dims
        );
        auto bq = rag_bq_quantize(embedding.data(), engine->params.n_dims);

        std::string chunk_text;
        for (int32_t t = tok_start; t < tok_end; t++) {
            chunk_text += common_token_to_piece(engine->ctx, tokens[t]);
        }

        rag_chunk chunk;
        chunk.text        = std::move(chunk_text);
        chunk.doc_id      = doc_id;
        chunk.chunk_index = n_chunks;
        chunk.embedding   = std::move(embedding);
        chunk.bq_vector   = std::move(bq);

        engine->chunks.push_back(std::move(chunk));
        n_chunks++;

        if (tok_end >= n_tokens) break;
    }

    rag_document doc;
    doc.doc_id      = doc_id;
    doc.first_chunk = first_chunk;
    doc.n_chunks    = n_chunks;

    engine->doc_index[doc_id] = (int32_t)engine->documents.size();
    engine->documents.push_back(std::move(doc));

    return n_chunks;
}

int32_t rag_engine_add_document(rag_engine_t * engine,
    const char * text, const char * doc_id)
{
    if (!engine || !text || !doc_id) return -1;
    if (!rag_engine_is_loaded(engine)) return -1;

    if (engine->doc_index.count(doc_id)) {
        rag_engine_remove_document(engine, doc_id);
    }

    if (engine->params.late_chunking) {
        return rag_index_document_late(engine, text, doc_id);
    } else {
        return rag_index_document_naive(engine, text, doc_id);
    }
}

int32_t rag_engine_remove_document(rag_engine_t * engine, const char * doc_id) {
    if (!engine || !doc_id) return -1;

    auto it = engine->doc_index.find(doc_id);
    if (it == engine->doc_index.end()) return -1;

    int32_t doc_idx = it->second;
    const auto & doc = engine->documents[doc_idx];

    engine->chunks.erase(
        engine->chunks.begin() + doc.first_chunk,
        engine->chunks.begin() + doc.first_chunk + doc.n_chunks
    );

    for (auto & d : engine->documents) {
        if (d.first_chunk > doc.first_chunk) {
            d.first_chunk -= doc.n_chunks;
        }
    }

    engine->documents.erase(engine->documents.begin() + doc_idx);
    engine->doc_index.erase(it);

    // Rebuild doc_index after erasure
    engine->doc_index.clear();
    for (int32_t i = 0; i < (int32_t)engine->documents.size(); i++) {
        engine->doc_index[engine->documents[i].doc_id] = i;
    }

    return 0;
}

void rag_engine_clear(rag_engine_t * engine) {
    if (!engine) return;
    engine->chunks.clear();
    engine->documents.clear();
    engine->doc_index.clear();
}

rag_result * rag_engine_query(rag_engine_t * engine,
    const char * query, int32_t * n_results)
{
    if (!engine || !query || !n_results) return nullptr;
    if (!rag_engine_is_loaded(engine)) return nullptr;
    if (engine->chunks.empty()) { *n_results = 0; return nullptr; }

    auto query_emb = rag_embed_text(engine, query);
    if (query_emb.empty()) return nullptr;

    // BQ coarse search -> top_k candidates
    auto query_bq = rag_bq_quantize(query_emb.data(), engine->params.n_dims);
    auto candidates = rag_bq_search(query_bq, engine->chunks, engine->params.top_k);

    if (candidates.empty()) { *n_results = 0; return nullptr; }

    // Cosine re-rank -> top_n
    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(candidates.size());
    for (auto & [hamming_dist, chunk_idx] : candidates) {
        float score = rag_cosine_similarity(
            query_emb.data(),
            engine->chunks[chunk_idx].embedding.data(),
            engine->params.n_dims
        );
        scored.push_back({score, chunk_idx});
    }

    std::sort(scored.begin(), scored.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });

    int32_t n = std::min(engine->params.top_n, (int32_t)scored.size());

    auto * results = (rag_result *)malloc(n * sizeof(rag_result));
    if (!results) return nullptr;

    for (int32_t i = 0; i < n; i++) {
        auto & [score, chunk_idx] = scored[i];
        const auto & chunk = engine->chunks[chunk_idx];
        results[i].text        = rag_strdup(chunk.text.c_str());
        results[i].doc_id      = rag_strdup(chunk.doc_id.c_str());
        results[i].chunk_index = chunk.chunk_index;
        results[i].score       = score;
    }

    *n_results = n;
    return results;
}

void rag_engine_free_results(rag_result * results, int32_t n) {
    if (!results) return;
    for (int32_t i = 0; i < n; i++) {
        free((void *)results[i].text);
        free((void *)results[i].doc_id);
    }
    free(results);
}

char * rag_engine_build_prompt(rag_engine_t * engine,
    const char * query, const char * user_prompt)
{
    if (!engine || !query) return nullptr;

    int32_t n_results = 0;
    auto * results = rag_engine_query(engine, query, &n_results);

    std::string prompt;
    if (n_results > 0 && results) {
        prompt += "Use the following context to answer the question.\n\n";
        prompt += "Context:\n";
        for (int32_t i = 0; i < n_results; i++) {
            prompt += "---\n[";
            prompt += results[i].doc_id;
            prompt += "]\n";
            prompt += results[i].text;
            prompt += "\n";
        }
        prompt += "---\n\n";
        rag_engine_free_results(results, n_results);
    }

    prompt += (user_prompt ? user_prompt : query);

    return rag_strdup(prompt.c_str());
}

char * rag_engine_info_json(const rag_engine_t * engine) {
    if (!engine) return nullptr;

    char buf[1024];
    snprintf(buf, sizeof(buf),
        "{"
        "\"loaded\":%s,"
        "\"n_dims\":%d,"
        "\"chunk_size\":%d,"
        "\"chunk_overlap\":%d,"
        "\"late_chunking\":%s,"
        "\"top_k\":%d,"
        "\"top_n\":%d,"
        "\"n_documents\":%d,"
        "\"n_chunks\":%d,"
        "\"n_embd\":%d,"
        "\"model_n_ctx\":%d"
        "}",
        rag_engine_is_loaded(engine) ? "true" : "false",
        engine->params.n_dims,
        engine->params.chunk_size,
        engine->params.chunk_overlap,
        engine->params.late_chunking ? "true" : "false",
        engine->params.top_k,
        engine->params.top_n,
        (int32_t)engine->documents.size(),
        (int32_t)engine->chunks.size(),
        engine->n_embd,
        engine->ctx ? (int32_t)llama_n_ctx(engine->ctx) : 0
    );

    return rag_strdup(buf);
}

void rag_engine_free_string(char * str) {
    free(str);
}
