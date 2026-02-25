// rag.cpp — Retrieval-Augmented Generation (embedding, BM25, KG, web search)
//
// Extracted from gguf-forward-test.cpp (lines 5832-7805).
// Self-embedding (same GGUF model embeds AND generates), triple-path retrieval
// (BM25 + vector + KG), extractive compression, live memory, web search.
// All C++, no external dependencies beyond curl (via popen).

#include "gguf-engine/rag.h"
#include "gguf-engine/graph.h"
#include "gguf-engine/utils.h"
#include "gguf-engine/tokenizer.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cctype>

// =========================================================================
// File-local helpers
// =========================================================================
namespace {

// Current time in seconds since epoch
float rag_now_sec() {
    auto now = Clock::now();
    return (float)std::chrono::duration<double>(now.time_since_epoch()).count();
}

// Compute context size for embedding graph (no KV stores, no interventions)
size_t compute_emb_ctx_size(int n_layers) {
    // ~35 tensors per layer (no KV cache ops) + 15 global (embedding, mean, norm)
    int n_tensors = n_layers * 35 + 15;
    return (size_t)n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(2048, false);
}

// Compute embedding for a single token sequence, store result in out_embedding
bool compute_one_embedding(ModelState & state, ggml_backend_t backend,
    const std::vector<int32_t> & tokens, std::vector<float> & out_embedding,
    ggml_gallocr_t galloc)
{
    const int n_embd = (int)state.cfg.n_embd;
    const int n_layers = (int)state.cfg.n_layer;
    int seq_len = (int)tokens.size();
    if (seq_len == 0) return false;
    if (seq_len > (int)state.cfg.max_ctx) seq_len = (int)state.cfg.max_ctx;

    size_t ctx_size = compute_emb_ctx_size(n_layers);
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_cgraph * graph = build_embedding_graph(ctx, state, seq_len, n_layers, nullptr);

    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        ggml_free(ctx);
        return false;
    }

    // Set tokens
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "inp_tokens"),
        tokens.data(), 0, seq_len * sizeof(int32_t));

    // Set positions [0, 1, 2, ...]
    std::vector<int32_t> positions(seq_len);
    for (int i = 0; i < seq_len; i++) positions[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "inp_pos"),
        positions.data(), 0, seq_len * sizeof(int32_t));

    // Bidirectional mask: all zeros
    std::vector<uint16_t> mask(seq_len * seq_len, 0);
    ggml_backend_tensor_set(ggml_graph_get_tensor(graph, "attn_mask"),
        mask.data(), 0, seq_len * seq_len * sizeof(uint16_t));

    ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);

    // Read hidden states [n_embd, seq_len] and mean pool on CPU
    struct ggml_tensor * hs_t = ggml_graph_get_tensor(graph, "hidden_states");
    std::vector<float> hidden(n_embd * seq_len);
    ggml_backend_tensor_get(hs_t, hidden.data(), 0, n_embd * seq_len * sizeof(float));

    // Mean pool: average over seq_len dimension
    out_embedding.assign(n_embd, 0.0f);
    for (int s = 0; s < seq_len; s++) {
        for (int e = 0; e < n_embd; e++) {
            out_embedding[e] += hidden[s * n_embd + e];
        }
    }
    float inv_seq = 1.0f / (float)seq_len;
    for (int e = 0; e < n_embd; e++) out_embedding[e] *= inv_seq;

    // L2 normalize
    float norm = 0.0f;
    for (int i = 0; i < n_embd; i++) norm += out_embedding[i] * out_embedding[i];
    norm = sqrtf(norm + 1e-12f);
    for (int i = 0; i < n_embd; i++) out_embedding[i] /= norm;

    ggml_free(ctx);
    return true;
}

} // anonymous namespace (part 1)

// Batch embed multiple texts (public, declared in rag.h)
bool compute_embeddings(ModelState & state, ggml_backend_t backend,
    const std::vector<std::string> & texts,
    std::vector<std::vector<float>> & out_embeddings,
    ggml_gallocr_t galloc)
{
    const int n_layers = (int)state.cfg.n_layer;
    out_embeddings.resize(texts.size());

    // Reserve galloc from worst-case (max chunk size = 256 tokens)
    int max_seq = 256;
    bool own_galloc = false;
    if (!galloc) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        own_galloc = true;
    }
    {
        size_t ctx_size = compute_emb_ctx_size(n_layers);
        struct ggml_init_params p = { ctx_size, nullptr, true };
        struct ggml_context * measure_ctx = ggml_init(p);
        struct ggml_cgraph * measure_graph = build_embedding_graph(measure_ctx, state, max_seq, n_layers, nullptr);
        ggml_gallocr_reserve(galloc, measure_graph);
        ggml_free(measure_ctx);
    }

    for (size_t i = 0; i < texts.size(); i++) {
        std::vector<int> tokens_int = tokenize_simple(state.vocab, texts[i]);
        // Convert to int32_t
        std::vector<int32_t> tokens(tokens_int.begin(), tokens_int.end());
        if (tokens.empty()) {
            out_embeddings[i].assign((int)state.cfg.n_embd, 0.0f);
            continue;
        }
        // Truncate to max_seq
        if ((int)tokens.size() > max_seq) tokens.resize(max_seq);

        if (!compute_one_embedding(state, backend, tokens, out_embeddings[i], galloc)) {
            printf("  WARNING: embedding failed for text %zu\n", i);
            out_embeddings[i].assign((int)state.cfg.n_embd, 0.0f);
        }
    }

    if (own_galloc) ggml_gallocr_free(galloc);
    return true;
}

namespace { // anonymous namespace (part 2)

// Chunk text on sentence boundaries with overlap
std::vector<std::string> chunk_text(const std::string & text,
    int target_words = 128, int overlap_words = 32)
{
    // Split into sentences
    std::vector<std::string> sentences;
    std::string cur;
    for (size_t i = 0; i < text.size(); i++) {
        cur += text[i];
        bool is_boundary = false;
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            // Sentence end if followed by space/newline/EOF
            if (i + 1 >= text.size() || text[i + 1] == ' ' || text[i + 1] == '\n')
                is_boundary = true;
        } else if (text[i] == '\n' && i + 1 < text.size() && text[i + 1] == '\n') {
            is_boundary = true;
        }
        if (is_boundary) {
            // Trim
            size_t start = cur.find_first_not_of(" \n\r\t");
            if (start != std::string::npos) {
                sentences.push_back(cur.substr(start));
            }
            cur.clear();
        }
    }
    if (!cur.empty()) {
        size_t start = cur.find_first_not_of(" \n\r\t");
        if (start != std::string::npos) sentences.push_back(cur.substr(start));
    }

    // Merge sentences into chunks of ~target_words
    std::vector<std::string> chunks;
    std::string chunk;
    int word_count = 0;
    std::vector<int> sentence_word_counts;

    for (size_t si = 0; si < sentences.size(); si++) {
        int sw = (int)rag_tokenize_terms(sentences[si]).size();
        sentence_word_counts.push_back(sw);

        if (word_count + sw > target_words && word_count > 0) {
            chunks.push_back(chunk);

            // Start new chunk with overlap from previous sentences
            chunk.clear();
            word_count = 0;
            // Walk backwards to get ~overlap_words
            int ow = 0;
            int back = (int)si - 1;
            while (back >= 0 && ow + sentence_word_counts[back] <= overlap_words) {
                ow += sentence_word_counts[back];
                back--;
            }
            back++;
            for (int b = back; b < (int)si; b++) {
                if (!chunk.empty()) chunk += " ";
                chunk += sentences[b];
                word_count += sentence_word_counts[b];
            }
        }

        if (!chunk.empty()) chunk += " ";
        chunk += sentences[si];
        word_count += sw;
    }
    if (!chunk.empty()) chunks.push_back(chunk);

    // Fallback: if no sentence boundaries found, split by word count
    if (chunks.empty() && !text.empty()) {
        chunks.push_back(text.substr(0, std::min((int)text.size(), target_words * 6)));
    }

    return chunks;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public functions declared in rag.h
// ---------------------------------------------------------------------------

// Add a document to BM25 index
void bm25_add(BM25Index & idx, int chunk_id,
    const std::unordered_map<std::string, int> & term_freq, int n_terms)
{
    // Update avg doc len
    float total = idx.avg_doc_len * idx.n_docs + (float)n_terms;
    idx.n_docs++;
    idx.avg_doc_len = total / idx.n_docs;

    // Add postings
    for (auto & [term, freq] : term_freq) {
        idx.postings[term].push_back({chunk_id, freq});
    }
}

// Ingest text: chunk -> compute term freqs -> add to BM25
void rag_ingest_chunks(RagState & rag, const std::string & text,
    const std::string & source)
{
    auto chunk_texts = chunk_text(text);
    int64_t now = (int64_t)rag_now_sec();

    for (auto & ct : chunk_texts) {
        RagChunk chunk;
        chunk.id = (int)rag.chunks.size();
        chunk.text = ct;
        chunk.source = source;
        chunk.timestamp = now;

        auto terms = rag_tokenize_terms(ct);
        chunk.n_terms = (int)terms.size();
        for (auto & t : terms) chunk.term_freq[t]++;

        bm25_add(rag.bm25, chunk.id, chunk.term_freq, chunk.n_terms);
        rag.chunks.push_back(std::move(chunk));
    }
}

// Reciprocal Rank Fusion: combine ranked lists
std::vector<int> rrf_fuse(
    const std::vector<std::pair<int, float>> & bm25_results,
    const std::vector<std::pair<int, float>> & vec_results,
    const std::vector<std::pair<int, float>> & kg_results_scored,
    int top_k, float k)
{
    std::unordered_map<int, float> fused;

    auto add_ranks = [&](const std::vector<std::pair<int, float>> & results, float weight) {
        for (int rank = 0; rank < (int)results.size(); rank++) {
            fused[results[rank].first] += weight / (k + rank + 1);
        }
    };

    add_ranks(bm25_results, 1.0f);
    add_ranks(vec_results, 1.0f);
    add_ranks(kg_results_scored, 0.5f); // KG gets slightly lower weight

    std::vector<std::pair<int, float>> sorted(fused.begin(), fused.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });

    std::vector<int> result;
    for (int i = 0; i < std::min(top_k, (int)sorted.size()); i++) {
        result.push_back(sorted[i].first);
    }
    return result;
}

namespace { // internal helpers continued

// DuckDuckGo HTML search (internal helper)
bool web_search_ddg(const std::string & query, int max_results,
    std::vector<std::string> & out_titles,
    std::vector<std::string> & out_snippets,
    std::vector<std::string> & out_urls)
{
    std::string url = "https://html.duckduckgo.com/html/?q=" + url_encode(query);
    std::string html = http_fetch(url);

    if (html.empty()) {
        printf("  WARNING: DuckDuckGo fetch failed\n");
        return false;
    }

    printf("  Searching: %s\n", query.c_str());

    // Parse results: look for result__a (title+url) and result__snippet
    size_t pos = 0;
    int count = 0;
    while (count < max_results && pos < html.size()) {
        // Find result link: <a rel="nofollow" class="result__a" href="..."
        size_t link_pos = html.find("class=\"result__a\"", pos);
        if (link_pos == std::string::npos) break;

        // Extract href
        size_t href_start = html.rfind("href=\"", link_pos);
        std::string href;
        if (href_start != std::string::npos && href_start > link_pos - 200) {
            href_start += 6;
            size_t href_end = html.find('"', href_start);
            if (href_end != std::string::npos) {
                href = html.substr(href_start, href_end - href_start);
            }
        }

        // Extract title (text between > and </a>)
        size_t title_start = html.find('>', link_pos);
        std::string title;
        if (title_start != std::string::npos) {
            title_start++;
            size_t title_end = html.find("</a>", title_start);
            if (title_end != std::string::npos) {
                title = strip_html(html.substr(title_start, title_end - title_start));
            }
        }

        // Find snippet: <a class="result__snippet"
        size_t snip_pos = html.find("class=\"result__snippet\"", link_pos);
        std::string snippet;
        if (snip_pos != std::string::npos && snip_pos < link_pos + 2000) {
            size_t snip_start = html.find('>', snip_pos);
            if (snip_start != std::string::npos) {
                snip_start++;
                size_t snip_end = html.find("</a>", snip_start);
                if (snip_end != std::string::npos) {
                    snippet = strip_html(html.substr(snip_start, snip_end - snip_start));
                }
            }
        }

        if (!title.empty() || !snippet.empty()) {
            out_titles.push_back(title);
            out_snippets.push_back(snippet);
            out_urls.push_back(href);
            count++;
        }

        pos = (snip_pos != std::string::npos) ? snip_pos + 20 : link_pos + 20;
    }

    printf("  Found %d results\n", count);
    return count > 0;
}

} // anonymous namespace (helpers continued)

// =========================================================================
// Section 1: Self-Embedding Engine
// =========================================================================
// Uses the same GGUF model with a BIDIRECTIONAL attention mask (all zeros)
// and NO KV cache writes. Mean-pools hidden states -> L2-normalized embedding.

struct ggml_cgraph * build_embedding_graph(
    struct ggml_context * ctx, ModelState & state, int seq_len, int n_layers,
    InterventionConfig * /*iv*/)
{
    const ModelConfig & cfg = state.cfg;
    const int head_dim  = (int)cfg.head_dim;
    const int n_head    = (int)cfg.n_head;
    const int n_head_kv = (int)cfg.n_head_kv;
    const int n_embd_head = n_head * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Input tensors
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // Bidirectional mask: all zeros (every token attends to every token)
    struct ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, seq_len, seq_len);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    // Token embedding lookup
    struct ggml_tensor * cur = ggml_get_rows(ctx, state.token_embd, inp_tokens);
    if (cfg.embd_scale) {
        cur = ggml_scale(ctx, cur, sqrtf((float)cfg.n_embd));
    }

    // Transformer layers -- NO KV cache writes (embedding mode)
    for (int il = 0; il < n_layers; il++) {
        const LayerWeights & lw = state.layers[il];
        struct ggml_tensor * residual = cur;

        // Attention pre-norm
        cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
        if (cfg.gemma_norm) {
            cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, lw.attn_norm));
        } else {
            cur = ggml_mul(ctx, cur, lw.attn_norm);
        }

        // QKV
        struct ggml_tensor * Q = ggml_mul_mat(ctx, lw.attn_q, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, lw.attn_k, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, lw.attn_v, cur);

        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, seq_len);
        K = ggml_reshape_3d(ctx, K, head_dim, n_head_kv, seq_len);
        V = ggml_reshape_3d(ctx, V, head_dim, n_head_kv, seq_len);

        if (cfg.has_qk_norm && lw.q_norm && lw.k_norm) {
            Q = ggml_rms_norm(ctx, Q, cfg.rms_eps);
            if (cfg.gemma_norm) {
                Q = ggml_add(ctx, Q, ggml_mul(ctx, Q, lw.q_norm));
            } else {
                Q = ggml_mul(ctx, Q, lw.q_norm);
            }
            K = ggml_rms_norm(ctx, K, cfg.rms_eps);
            if (cfg.gemma_norm) {
                K = ggml_add(ctx, K, ggml_mul(ctx, K, lw.k_norm));
            } else {
                K = ggml_mul(ctx, K, lw.k_norm);
            }
        }

        // RoPE
        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr,
            head_dim, cfg.rope_type, (int)cfg.max_ctx,
            cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr,
            head_dim, cfg.rope_type, (int)cfg.max_ctx,
            cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Self-attention (no KV cache -- K,V are from current sequence only)
        struct ggml_tensor * Q_perm = ggml_permute(ctx, Q, 0, 2, 1, 3);
        struct ggml_tensor * K_perm = ggml_permute(ctx, K, 0, 2, 1, 3);
        struct ggml_tensor * V_perm = ggml_permute(ctx, V, 0, 2, 1, 3);

        struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx,
            Q_perm, K_perm, V_perm, attn_mask, scale, 0.0f, 0.0f);

        struct ggml_tensor * attn_merged = ggml_reshape_2d(ctx,
            ggml_cont(ctx, attn_out), n_embd_head, seq_len);

        cur = ggml_mul_mat(ctx, lw.attn_output, attn_merged);
        cur = ggml_add(ctx, cur, residual);

        // FFN
        struct ggml_tensor * ffn_residual = cur;
        cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
        if (cfg.gemma_norm) {
            cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, lw.ffn_norm));
        } else {
            cur = ggml_mul(ctx, cur, lw.ffn_norm);
        }

        struct ggml_tensor * gate_proj = ggml_mul_mat(ctx, lw.ffn_gate, cur);
        struct ggml_tensor * gate = cfg.use_gelu
            ? ggml_gelu(ctx, gate_proj)
            : ggml_silu(ctx, gate_proj);
        struct ggml_tensor * up = ggml_mul_mat(ctx, lw.ffn_up, cur);
        cur = ggml_mul_mat(ctx, lw.ffn_down, ggml_mul(ctx, gate, up));
        cur = ggml_add(ctx, cur, ffn_residual);
    }

    // Final norm
    cur = ggml_rms_norm(ctx, cur, cfg.rms_eps);
    if (cfg.gemma_norm) {
        cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, state.output_norm));
    } else {
        cur = ggml_mul(ctx, cur, state.output_norm);
    }

    // Output raw hidden states [n_embd, seq_len] -- mean pooling done on CPU
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2048, false);
    ggml_build_forward_expand(graph, cur);
    return graph;
}

// =========================================================================
// Section 2: BM25
// =========================================================================

// Lowercase + split on whitespace/punctuation -> terms
std::vector<std::string> rag_tokenize_terms(const std::string & text) {
    std::vector<std::string> terms;
    std::string cur;
    for (char c : text) {
        if (std::isalnum((unsigned char)c)) {
            cur += (char)std::tolower((unsigned char)c);
        } else {
            if (!cur.empty()) { terms.push_back(cur); cur.clear(); }
        }
    }
    if (!cur.empty()) terms.push_back(cur);
    return terms;
}

// Build BM25 postings index from all chunks in RagState
void compute_bm25_stats(RagState & rag) {
    rag.bm25 = {}; // reset
    for (const auto & c : rag.chunks) {
        bm25_add(rag.bm25, c.id, c.term_freq, c.n_terms);
    }
}

// BM25 search
std::vector<std::pair<int, float>> bm25_search(
    const RagState & rag, const std::string & query, int top_k)
{
    const BM25Index & idx = rag.bm25;
    auto query_terms = rag_tokenize_terms(query);
    std::unordered_map<int, float> scores;

    for (const auto & term : query_terms) {
        auto it = idx.postings.find(term);
        if (it == idx.postings.end()) continue;

        const auto & posting = it->second;
        // IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        float df = (float)posting.size();
        float idf = logf((idx.n_docs - df + 0.5f) / (df + 0.5f) + 1.0f);

        for (auto & [chunk_id, tf] : posting) {
            // Need doc length -- approximate from tf sum (stored in chunk)
            float dl = idx.avg_doc_len; // approximation; exact needs chunk lookup
            float tf_f = (float)tf;
            float tf_component = (tf_f * (idx.k1 + 1.0f)) /
                (tf_f + idx.k1 * (1.0f - idx.b + idx.b * dl / std::max(idx.avg_doc_len, 1.0f)));
            scores[chunk_id] += idf * tf_component;
        }
    }

    // Sort by score
    std::vector<std::pair<int, float>> results(scores.begin(), scores.end());
    std::sort(results.begin(), results.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });
    if ((int)results.size() > top_k) results.resize(top_k);
    return results;
}

// =========================================================================
// Section 3: Vector Search
// =========================================================================

// Cosine similarity (vectors assumed L2-normalized -> just dot product)
float cosine_sim(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); i++) dot += a[i] * b[i];
    return dot;
}

// Vector search: flat cosine similarity over all chunks
std::vector<std::pair<int, float>> vector_search(
    const RagState & rag, const std::vector<float> & query_embd, int top_k)
{
    std::vector<std::pair<int, float>> scores;
    scores.reserve(rag.chunks.size());
    for (const auto & chunk : rag.chunks) {
        if (chunk.embedding.empty()) continue;
        float sim = cosine_sim(query_embd, chunk.embedding);
        scores.push_back({chunk.id, sim});
    }

    // Partial sort for top-K
    if ((int)scores.size() > top_k) {
        std::nth_element(scores.begin(), scores.begin() + top_k, scores.end(),
            [](const auto & a, const auto & b) { return a.second > b.second; });
        scores.resize(top_k);
    }
    std::sort(scores.begin(), scores.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });
    return scores;
}

// =========================================================================
// Section 4: Knowledge Graph
// =========================================================================

// Extract knowledge graph triples (pattern-based)
void extract_triples(const std::string & text, int source_chunk_id,
    std::vector<KgTriple> & triples)
{
    // Patterns: "X is Y", "X is a Y", "X works at Y", "X lives in Y",
    //           "X likes Y", "X was created by Y", "X is the capital of Y"
    struct Pattern {
        const char * regex_like; // simplified: "X <relation> Y" where we match keyword
        const char * relation;
    };
    static const Pattern patterns[] = {
        { " is the capital of ", "capital_of" },
        { " was created by ",    "created_by" },
        { " was founded by ",    "founded_by" },
        { " works at ",          "works_at" },
        { " lives in ",          "lives_in" },
        { " is from ",           "from" },
        { " is a ",              "is_a" },
        { " is an ",             "is_a" },
        { " likes ",             "likes" },
        { " loves ",             "loves" },
    };

    // Process sentence by sentence
    std::string sentence;
    for (size_t i = 0; i <= text.size(); i++) {
        char c = (i < text.size()) ? text[i] : '.';
        if (c == '.' || c == '!' || c == '?' || c == '\n') {
            if (sentence.size() > 5) {
                for (const auto & pat : patterns) {
                    size_t pos = sentence.find(pat.regex_like);
                    if (pos == std::string::npos) continue;

                    // Subject = text before pattern (last ~3 words)
                    std::string before = sentence.substr(0, pos);
                    std::string after = sentence.substr(pos + strlen(pat.regex_like));

                    // Trim subject to last meaningful phrase
                    auto sub_terms = rag_tokenize_terms(before);
                    if (sub_terms.empty()) continue;
                    int start = std::max(0, (int)sub_terms.size() - 4);
                    std::string subject;
                    for (int s = start; s < (int)sub_terms.size(); s++) {
                        if (!subject.empty()) subject += " ";
                        subject += sub_terms[s];
                    }

                    // Object = first ~4 words after pattern
                    auto obj_terms = rag_tokenize_terms(after);
                    if (obj_terms.empty()) continue;
                    int end = std::min(4, (int)obj_terms.size());
                    std::string object;
                    for (int o = 0; o < end; o++) {
                        if (!object.empty()) object += " ";
                        object += obj_terms[o];
                    }

                    KgTriple triple;
                    triple.subject = subject;
                    triple.relation = pat.relation;
                    triple.object = object;
                    triple.source_chunk_id = source_chunk_id;

                    triples.push_back(triple);
                }
            }
            sentence.clear();
        } else {
            sentence += c;
        }
    }
}

// Extract triples from all chunks
void rag_extract_kg(RagState & rag) {
    rag.kg_triples.clear();
    rag.kg_entity_idx.clear();

    for (const auto & chunk : rag.chunks) {
        extract_triples(chunk.text, chunk.id, rag.kg_triples);
    }

    // Rebuild entity index
    for (int i = 0; i < (int)rag.kg_triples.size(); i++) {
        rag.kg_entity_idx[rag.kg_triples[i].subject].push_back(i);
        rag.kg_entity_idx[rag.kg_triples[i].object].push_back(i);
    }
}

// KG search: find chunk IDs related to query entities
std::vector<int> kg_search(const RagState & rag, const std::string & query) {
    auto terms = rag_tokenize_terms(query);
    std::unordered_map<int, float> chunk_scores;

    // Try multi-word entity matches first, then single terms
    for (const auto & [entity, triple_ids] : rag.kg_entity_idx) {
        // Check if query contains this entity
        bool match = false;
        auto entity_terms = rag_tokenize_terms(entity);
        if (entity_terms.empty()) continue;

        // Simple containment check
        for (const auto & et : entity_terms) {
            for (const auto & qt : terms) {
                if (et == qt) { match = true; break; }
            }
            if (match) break;
        }

        if (match) {
            for (int tidx : triple_ids) {
                if (tidx < 0 || tidx >= (int)rag.kg_triples.size()) continue;
                int cid = rag.kg_triples[tidx].source_chunk_id;
                chunk_scores[cid] += 1.0f;

                // Also boost chunks containing the other entity in the triple
                const auto & triple = rag.kg_triples[tidx];
                // Find chunks containing subject/object
                auto check = [&](const std::string & ent) {
                    auto it2 = rag.kg_entity_idx.find(ent);
                    if (it2 != rag.kg_entity_idx.end()) {
                        for (int ti2 : it2->second) {
                            if (ti2 < (int)rag.kg_triples.size())
                                chunk_scores[rag.kg_triples[ti2].source_chunk_id] += 0.5f;
                        }
                    }
                };
                check(triple.subject);
                check(triple.object);
            }
        }
    }

    // Sort by score and return chunk IDs only
    std::vector<std::pair<int, float>> sorted(chunk_scores.begin(), chunk_scores.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });

    std::vector<int> result;
    for (const auto & [cid, _] : sorted) {
        result.push_back(cid);
    }
    return result;
}

// =========================================================================
// Section 5: Hybrid Search
// =========================================================================

// Full hybrid retrieval pipeline (BM25 + vector + KG with RRF)
std::vector<std::pair<int, float>> hybrid_search(RagState & rag,
    const std::string & query, const std::vector<float> & query_emb, int top_k)
{
    // 1. BM25 search
    auto bm25_results = bm25_search(rag, query, 20);

    // 2. Vector search
    auto vec_results = vector_search(rag, query_emb, 20);

    // 3. KG search -- convert to scored pairs for RRF
    auto kg_ids = kg_search(rag, query);
    std::vector<std::pair<int, float>> kg_scored;
    for (int i = 0; i < (int)kg_ids.size() && i < 10; i++) {
        kg_scored.push_back({kg_ids[i], 1.0f / (float)(i + 1)});
    }

    // 4. RRF fusion
    auto fused_ids = rrf_fuse(bm25_results, vec_results, kg_scored, top_k);

    // Build result with scores from fused ranking
    std::vector<std::pair<int, float>> result;
    for (int i = 0; i < (int)fused_ids.size(); i++) {
        // Score is reciprocal rank position
        result.push_back({fused_ids[i], 1.0f / (float)(i + 1)});
    }
    return result;
}

// =========================================================================
// Section 6: Chunk Embedding
// =========================================================================

// Embed all chunks that don't have embeddings yet
void rag_embed_chunks(RagState & rag, ModelState & state,
    ggml_backend_t backend, ggml_gallocr_t galloc)
{
    std::vector<std::string> texts;
    std::vector<int> indices; // which chunks need embedding

    for (size_t i = 0; i < rag.chunks.size(); i++) {
        if (rag.chunks[i].embedding.empty()) {
            texts.push_back(rag.chunks[i].text);
            indices.push_back((int)i);
        }
    }
    if (texts.empty()) return;

    std::vector<std::vector<float>> embeddings;
    compute_embeddings(state, backend, texts, embeddings, galloc);

    for (size_t i = 0; i < indices.size(); i++) {
        rag.chunks[indices[i]].embedding = std::move(embeddings[i]);
    }
}

// =========================================================================
// Section 7: Web Integration
// =========================================================================

// URL-encode a string
std::string url_encode(const std::string & s) {
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += (char)c;
        } else if (c == ' ') {
            out += '+';
        } else {
            char buf[4];
            snprintf(buf, sizeof(buf), "%%%02X", c);
            out += buf;
        }
    }
    return out;
}

// Strip HTML tags from text
std::string strip_html(const std::string & html) {
    std::string out;
    out.reserve(html.size());
    bool in_tag = false;
    bool in_script = false;
    bool in_style = false;

    for (size_t i = 0; i < html.size(); i++) {
        if (html[i] == '<') {
            in_tag = true;
            // Check for <script or <style
            if (i + 7 < html.size()) {
                std::string tag7 = html.substr(i, 7);
                for (auto & c : tag7) c = (char)std::tolower((unsigned char)c);
                if (tag7 == "<script") in_script = true;
                if (tag7.substr(0, 6) == "<style") in_style = true;
            }
            if (i + 8 < html.size()) {
                std::string tag8 = html.substr(i, 8);
                for (auto & c : tag8) c = (char)std::tolower((unsigned char)c);
                if (tag8 == "</script") in_script = false;
                if (tag8.substr(0, 7) == "</style") in_style = false;
            }
            continue;
        }
        if (html[i] == '>') {
            in_tag = false;
            if (!in_script && !in_style) out += ' ';
            continue;
        }
        if (!in_tag && !in_script && !in_style) {
            out += html[i];
        }
    }

    // Collapse whitespace
    std::string clean;
    bool last_space = false;
    for (char c : out) {
        if (c == ' ' || c == '\t' || c == '\r') {
            if (!last_space) { clean += ' '; last_space = true; }
        } else if (c == '\n') {
            if (!last_space) { clean += '\n'; last_space = true; }
        } else {
            clean += c;
            last_space = false;
        }
    }

    // Decode common HTML entities
    std::string decoded;
    decoded.reserve(clean.size());
    for (size_t i = 0; i < clean.size(); i++) {
        if (clean[i] == '&') {
            if (clean.compare(i, 4, "&lt;") == 0)   { decoded += '<'; i += 3; }
            else if (clean.compare(i, 4, "&gt;") == 0)   { decoded += '>'; i += 3; }
            else if (clean.compare(i, 5, "&amp;") == 0)  { decoded += '&'; i += 4; }
            else if (clean.compare(i, 6, "&quot;") == 0) { decoded += '"'; i += 5; }
            else if (clean.compare(i, 6, "&apos;") == 0) { decoded += '\''; i += 5; }
            else if (clean.compare(i, 6, "&nbsp;") == 0) { decoded += ' '; i += 5; }
            else decoded += clean[i];
        } else {
            decoded += clean[i];
        }
    }
    return decoded;
}

// Fetch URL via curl (popen) -- handles HTTPS, follows redirects
std::string http_fetch(const std::string & url) {
    // Sanitize URL: reject dangerous characters for shell
    for (char c : url) {
        if (c == '\'' || c == '`' || c == '$' || c == ';' || c == '|') {
            printf("  WARNING: rejected URL with unsafe char '%c'\n", c);
            return "";
        }
    }

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "curl -sL --max-time 10 -H 'User-Agent: Mozilla/5.0' '%s' 2>/dev/null",
        url.c_str());

    FILE * fp = popen(cmd, "r");
    if (!fp) return "";

    std::string out_body;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) {
        out_body += buf;
    }
    int status = pclose(fp);
    if (status != 0 || out_body.empty()) return "";
    return out_body;
}

// Fetch a web page and extract plain text
std::string fetch_and_extract(const std::string & url) {
    std::string html = http_fetch(url);
    if (html.empty()) return "";
    std::string text = strip_html(html);
    // Truncate to reasonable size
    if (text.size() > 10000) text.resize(10000);
    return text;
}

// Execute web search: search DuckDuckGo and return JSON results
std::string execute_web_search(const std::string & query) {
    if (query.empty()) return "{\"error\": \"missing query parameter\"}";

    std::vector<std::string> titles, snippets, urls;
    if (!web_search_ddg(query, 3, titles, snippets, urls)) {
        return "{\"error\": \"search failed\"}";
    }

    // Build JSON result
    std::string json = "{\"results\": [";
    for (size_t i = 0; i < titles.size(); i++) {
        if (i > 0) json += ", ";
        // Escape quotes in strings
        auto esc = [](const std::string & s) {
            std::string out;
            for (char c : s) {
                if (c == '"') out += "\\\"";
                else if (c == '\\') out += "\\\\";
                else if (c == '\n') out += " ";
                else out += c;
            }
            return out;
        };
        json += "{\"title\": \"" + esc(titles[i]) + "\", "
                "\"snippet\": \"" + esc(snippets[i]) + "\", "
                "\"url\": \"" + esc(urls[i]) + "\"}";
    }
    json += "]}";
    return json;
}

// =========================================================================
// Section 8: Persistence (RAGS binary format)
// =========================================================================

bool rag_save(const RagState & rag, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;

    // Header
    const char magic[4] = {'R','A','G','S'};
    uint32_t version = 1;
    fwrite(magic, 1, 4, f);
    fwrite(&version, 4, 1, f);
    uint32_t n_embd = rag.chunks.empty() ? 0 : (uint32_t)rag.chunks[0].embedding.size();
    fwrite(&n_embd, 4, 1, f);

    // Helper lambdas
    auto write_str = [&](const std::string & s) {
        uint32_t len = (uint32_t)s.size();
        fwrite(&len, 4, 1, f);
        fwrite(s.data(), 1, len, f);
    };

    // Chunks
    uint32_t n_chunks = (uint32_t)rag.chunks.size();
    fwrite(&n_chunks, 4, 1, f);
    for (const auto & c : rag.chunks) {
        int32_t id = c.id;
        fwrite(&id, 4, 1, f);
        write_str(c.text);
        write_str(c.source);
        int64_t ts = c.timestamp;
        fwrite(&ts, 8, 1, f);
        int32_t nt = c.n_terms;
        fwrite(&nt, 4, 1, f);
        // Embedding
        uint32_t emb_sz = (uint32_t)c.embedding.size();
        fwrite(&emb_sz, 4, 1, f);
        if (emb_sz > 0) fwrite(c.embedding.data(), sizeof(float), emb_sz, f);
        // Term freq
        uint32_t tf_sz = (uint32_t)c.term_freq.size();
        fwrite(&tf_sz, 4, 1, f);
        for (const auto & [term, freq] : c.term_freq) {
            write_str(term);
            int32_t fr = freq;
            fwrite(&fr, 4, 1, f);
        }
    }

    // KG triples
    uint32_t n_triples = (uint32_t)rag.kg_triples.size();
    fwrite(&n_triples, 4, 1, f);
    for (const auto & t : rag.kg_triples) {
        write_str(t.subject);
        write_str(t.relation);
        write_str(t.object);
        int32_t scid = t.source_chunk_id;
        fwrite(&scid, 4, 1, f);
    }

    // Memories
    uint32_t n_memories = (uint32_t)rag.memories.size();
    fwrite(&n_memories, 4, 1, f);
    for (const auto & m : rag.memories) {
        int32_t id = m.id;
        fwrite(&id, 4, 1, f);
        write_str(m.fact);
        write_str(m.category);
        float imp = m.importance;
        fwrite(&imp, 4, 1, f);
        int64_t ts = m.timestamp;
        fwrite(&ts, 8, 1, f);
        int32_t ac = m.access_count;
        fwrite(&ac, 4, 1, f);
        uint32_t emb_sz = (uint32_t)m.embedding.size();
        fwrite(&emb_sz, 4, 1, f);
        if (emb_sz > 0) fwrite(m.embedding.data(), sizeof(float), emb_sz, f);
    }

    fclose(f);
    return true;
}

bool rag_load(RagState & rag, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "RAGS", 4) != 0) { fclose(f); return false; }

    uint32_t version;
    fread(&version, 4, 1, f);
    if (version != 1) { fclose(f); return false; }

    uint32_t n_embd;
    fread(&n_embd, 4, 1, f);
    // n_embd stored in header for reference but not used in RagState directly

    auto read_str = [&](std::string & s) {
        uint32_t len;
        fread(&len, 4, 1, f);
        s.resize(len);
        if (len > 0) fread(&s[0], 1, len, f);
    };

    // Chunks
    uint32_t n_chunks;
    fread(&n_chunks, 4, 1, f);
    rag.chunks.resize(n_chunks);
    rag.bm25 = {}; // rebuild BM25 index
    for (uint32_t i = 0; i < n_chunks; i++) {
        auto & c = rag.chunks[i];
        int32_t id;
        fread(&id, 4, 1, f);
        c.id = id;
        read_str(c.text);
        read_str(c.source);
        fread(&c.timestamp, 8, 1, f);
        int32_t nt;
        fread(&nt, 4, 1, f);
        c.n_terms = nt;
        uint32_t emb_sz;
        fread(&emb_sz, 4, 1, f);
        c.embedding.resize(emb_sz);
        if (emb_sz > 0) fread(c.embedding.data(), sizeof(float), emb_sz, f);
        uint32_t tf_sz;
        fread(&tf_sz, 4, 1, f);
        for (uint32_t j = 0; j < tf_sz; j++) {
            std::string term;
            read_str(term);
            int32_t fr;
            fread(&fr, 4, 1, f);
            c.term_freq[term] = fr;
        }
        // Rebuild BM25 index
        bm25_add(rag.bm25, c.id, c.term_freq, c.n_terms);
    }

    // KG triples
    uint32_t n_triples;
    fread(&n_triples, 4, 1, f);
    rag.kg_triples.resize(n_triples);
    rag.kg_entity_idx.clear();
    for (uint32_t i = 0; i < n_triples; i++) {
        auto & t = rag.kg_triples[i];
        read_str(t.subject);
        read_str(t.relation);
        read_str(t.object);
        int32_t scid;
        fread(&scid, 4, 1, f);
        t.source_chunk_id = scid;
        rag.kg_entity_idx[t.subject].push_back((int)i);
        rag.kg_entity_idx[t.object].push_back((int)i);
    }

    // Memories
    uint32_t n_memories;
    fread(&n_memories, 4, 1, f);
    rag.memories.resize(n_memories);
    for (uint32_t i = 0; i < n_memories; i++) {
        auto & m = rag.memories[i];
        int32_t id;
        fread(&id, 4, 1, f);
        m.id = id;
        read_str(m.fact);
        read_str(m.category);
        fread(&m.importance, 4, 1, f);
        fread(&m.timestamp, 8, 1, f);
        int32_t ac;
        fread(&ac, 4, 1, f);
        m.access_count = ac;
        uint32_t emb_sz;
        fread(&emb_sz, 4, 1, f);
        m.embedding.resize(emb_sz);
        if (emb_sz > 0) fread(m.embedding.data(), sizeof(float), emb_sz, f);
    }

    fclose(f);
    return true;
}
