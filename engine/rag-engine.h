#pragma once

/**
 * RAG Engine - Context-preserving retrieval for Android
 *
 * Uses EmbeddingGemma-300M (Q4, ~50MB) with late chunking for
 * context-aware embeddings and binary quantization for compact storage.
 * Model-agnostic: index survives LLM swaps.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rag_engine rag_engine_t;

typedef struct {
    int32_t  n_threads;           // 0 = auto
    int32_t  chunk_size;          // tokens per chunk (default 256)
    int32_t  chunk_overlap;       // overlap tokens (default 32)
    int32_t  n_dims;              // Matryoshka dim: 768/512/256/128 (default 256)
    int32_t  top_k;               // BQ search candidates (default 32)
    int32_t  top_n;               // final results after re-rank (default 5)
    bool     late_chunking;       // embed full doc then chunk (default true)
} rag_engine_params;

typedef struct {
    const char * text;
    const char * doc_id;
    int32_t      chunk_index;
    float        score;
} rag_result;

// Lifecycle
rag_engine_params  rag_engine_default_params(void);
rag_engine_t *     rag_engine_create(rag_engine_params params);
void               rag_engine_free(rag_engine_t * engine);

// Embedding model
int32_t  rag_engine_load_model(rag_engine_t * engine, const char * path);
int32_t  rag_engine_load_model_from_fd(rag_engine_t * engine, int fd);
bool     rag_engine_is_loaded(const rag_engine_t * engine);

// Indexing
int32_t  rag_engine_add_document(rag_engine_t * engine,
             const char * text, const char * doc_id);
int32_t  rag_engine_remove_document(rag_engine_t * engine,
             const char * doc_id);
void     rag_engine_clear(rag_engine_t * engine);
int32_t  rag_engine_document_count(const rag_engine_t * engine);
int32_t  rag_engine_chunk_count(const rag_engine_t * engine);

// Retrieval
rag_result * rag_engine_query(rag_engine_t * engine,
                 const char * query, int32_t * n_results);
void         rag_engine_free_results(rag_result * results, int32_t n);

// Convenience: query + format into augmented prompt
char *       rag_engine_build_prompt(rag_engine_t * engine,
                 const char * query, const char * user_prompt);

// Info
char *       rag_engine_info_json(const rag_engine_t * engine);
void         rag_engine_free_string(char * str);

#ifdef __cplusplus
}
#endif
