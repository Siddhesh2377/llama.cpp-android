// rag.h — Retrieval-Augmented Generation (embedding, BM25, KG, web search)

#pragma once

#include "types.h"

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------
struct ggml_cgraph * build_embedding_graph(
    struct ggml_context * ctx, ModelState & state,
    int seq_len, int n_layers, InterventionConfig * iv);

// ---------------------------------------------------------------------------
// BM25
// ---------------------------------------------------------------------------
std::vector<std::string> rag_tokenize_terms(const std::string & text);
void compute_bm25_stats(RagState & rag);
std::vector<std::pair<int, float>> bm25_search(const RagState & rag,
    const std::string & query, int top_k);

// ---------------------------------------------------------------------------
// Vector search
// ---------------------------------------------------------------------------
float cosine_sim(const std::vector<float> & a, const std::vector<float> & b);
std::vector<std::pair<int, float>> vector_search(const RagState & rag,
    const std::vector<float> & query_emb, int top_k);

// ---------------------------------------------------------------------------
// Knowledge graph
// ---------------------------------------------------------------------------
void extract_triples(const std::string & text, int chunk_id,
    std::vector<KgTriple> & triples);
void rag_extract_kg(RagState & rag);
std::vector<int> kg_search(const RagState & rag, const std::string & query);

// ---------------------------------------------------------------------------
// Hybrid search
// ---------------------------------------------------------------------------
std::vector<std::pair<int, float>> hybrid_search(RagState & rag,
    const std::string & query, const std::vector<float> & query_emb, int top_k);

// ---------------------------------------------------------------------------
// Chunk embedding
// ---------------------------------------------------------------------------
void rag_embed_chunks(RagState & rag, ModelState & state,
    ggml_backend_t backend, ggml_gallocr_t galloc);

// ---------------------------------------------------------------------------
// Web integration
// ---------------------------------------------------------------------------
std::string url_encode(const std::string & str);
std::string strip_html(const std::string & html);
std::string http_fetch(const std::string & url);
std::string fetch_and_extract(const std::string & url);
std::string execute_web_search(const std::string & query);

// ---------------------------------------------------------------------------
// Ingestion and indexing
// ---------------------------------------------------------------------------
void rag_ingest_chunks(RagState & rag, const std::string & text,
    const std::string & source);
void bm25_add(BM25Index & idx, int chunk_id,
    const std::unordered_map<std::string, int> & term_freq, int n_terms);

// ---------------------------------------------------------------------------
// Batch embedding
// ---------------------------------------------------------------------------
bool compute_embeddings(ModelState & state, ggml_backend_t backend,
    const std::vector<std::string> & texts,
    std::vector<std::vector<float>> & out_embeddings,
    ggml_gallocr_t galloc = nullptr);

// ---------------------------------------------------------------------------
// RRF fusion
// ---------------------------------------------------------------------------
std::vector<int> rrf_fuse(
    const std::vector<std::pair<int, float>> & bm25_results,
    const std::vector<std::pair<int, float>> & vec_results,
    const std::vector<std::pair<int, float>> & kg_results_scored,
    int top_k, float k = 60.0f);

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
bool rag_save(const RagState & rag, const char * path);
bool rag_load(RagState & rag, const char * path);
