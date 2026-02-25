// boundaries.cpp — Async generation boundary system
//
// Extracted from gguf-forward-test.cpp (lines 3318-3494).
// Generates tokens until a boundary condition is met:
//   stop strings, EOS, tool call (grammar), max tokens, context full.

#include "gguf-engine/boundaries.h"
#include "gguf-engine/graph.h"
#include "gguf-engine/sampling.h"
#include "gguf-engine/utils.h"

#include <cstdio>

// Generate tokens until a boundary condition is met.
// Boundaries: stop strings, token limit, EOS, tool call (via grammar).
// Returns BoundaryEvent describing what happened.
BoundaryEvent generate_until_boundary(
    ModelState & state,
    ggml_backend_t backend,
    ggml_gallocr_t galloc,
    int max_tokens,
    const SamplingParams & sp,
    const InterventionConfig * iv,
    const InterventionTensors * iv_t,
    GrammarEngine * grammar,
    const std::vector<std::string> & stop_strings,
    GenerationState & gen_state,
    std::vector<float> & logit_buf,
    std::vector<uint16_t> & mask_buf,
    std::vector<int32_t> & pos_buf,
    bool stream)
{
    const ModelConfig & cfg = state.cfg;
    const int n_layer = (int)cfg.n_layer;
    const int n_vocab = (int)cfg.n_vocab;
    bool need_argmax = (sp.temp <= 0.0f);
    size_t ctx_size = compute_ctx_size(n_layer, iv != nullptr);
    std::vector<uint8_t> ctx_buf(ctx_size);

    // Find stop tokens
    int32_t eos_id = state.eos_token;
    int32_t im_end_id = -1, im_start_id = -1;
    for (int id = 0; id < (int)state.vocab.size(); id++) {
        if (state.vocab[id] == "<|im_end|>") im_end_id = id;
        if (state.vocab[id] == "<|im_start|>") im_start_id = id;
    }

    // Stop string accumulators
    std::vector<std::string> stop_accum(stop_strings.size());

    BoundaryEvent event;
    event.action = BOUNDARY_NONE;

    for (int t = 0; t < max_tokens; t++) {
        // Check grammar tool call ready (before decode)
        if (grammar && grammar->is_tool_call_ready()) {
            event.action = BOUNDARY_PAUSE_RESUME;
            event.name = "tool_call";
            event.kv_pos = state.kv_pos;
            event.tokens_generated = gen_state.tokens_generated;
            event.text = gen_state.full_text;
            return event;
        }

        // Check EOS / special tokens
        if (gen_state.last_token == eos_id || gen_state.last_token == im_start_id) {
            event.action = BOUNDARY_STOP;
            event.name = "eos";
            break;
        }
        if (gen_state.last_token == im_end_id && !(grammar && grammar->is_active())) {
            event.action = BOUNDARY_STOP;
            event.name = "im_end";
            break;
        }

        // Decode one token
        int kv_pos = state.kv_pos;
        int kv_len = kv_pos + 1;
        if (kv_len > (int)cfg.max_ctx) {
            event.action = BOUNDARY_STOP;
            event.name = "ctx_full";
            break;
        }

        struct ggml_init_params p = { ctx_size, ctx_buf.data(), true };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_cgraph * g = build_graph(ctx, state, 1, kv_pos, kv_len,
            n_layer, need_argmax, iv, iv_t);
        if (!ggml_gallocr_alloc_graph(galloc, g)) {
            ggml_free(ctx);
            event.action = BOUNDARY_STOP;
            event.name = "alloc_fail";
            break;
        }

        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_tokens"),
            &gen_state.last_token, 0, sizeof(int32_t));
        int32_t pos = kv_pos;
        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "inp_pos"),
            &pos, 0, sizeof(int32_t));
        std::fill(mask_buf.begin(), mask_buf.begin() + kv_len, (uint16_t)0);
        ggml_backend_tensor_set(ggml_graph_get_tensor(g, "attn_mask"),
            mask_buf.data(), 0, kv_len * sizeof(uint16_t));

        ggml_backend_graph_compute(backend, g);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(ggml_graph_get_tensor(g, "logits"),
            logit_buf.data(), 0, n_vocab * sizeof(float));

        // Rep penalty
        if (sp.rep_penalty != 1.0f) {
            if (logit_buf[gen_state.last_token] > 0)
                logit_buf[gen_state.last_token] /= sp.rep_penalty;
            else
                logit_buf[gen_state.last_token] *= sp.rep_penalty;
        }

        // Grammar masking
        if (grammar) grammar->apply_mask(logit_buf.data(), state.vocab);

        int32_t tid = sample_token(logit_buf.data(), n_vocab, sp, gen_state.rng);
        state.kv_pos = kv_len;
        ggml_free(ctx);

        // Print + accumulate
        if (tid >= 0 && tid < (int)state.vocab.size()) {
            std::string ts = decode_token(state.vocab[tid]);
            if (stream) {
                printf("%s", ts.c_str());
                fflush(stdout);
            }
            gen_state.full_text += ts;

            // Advance grammar
            if (grammar) grammar->advance(tid, state.vocab[tid]);

            // Check stop strings
            for (size_t si = 0; si < stop_strings.size(); si++) {
                stop_accum[si] += ts;
                if (stop_accum[si].find(stop_strings[si]) != std::string::npos) {
                    event.action = BOUNDARY_STOP;
                    event.name = "stop_string";
                    event.kv_pos = state.kv_pos;
                    event.tokens_generated = gen_state.tokens_generated + 1;
                    event.text = gen_state.full_text;
                    gen_state.last_token = tid;
                    gen_state.tokens_generated++;
                    return event;
                }
            }
        }

        gen_state.last_token = tid;
        gen_state.tokens_generated++;
    }

    // Token limit reached
    if (event.action == BOUNDARY_NONE) {
        event.action = BOUNDARY_STOP;
        event.name = "token_limit";
    }
    event.kv_pos = state.kv_pos;
    event.tokens_generated = gen_state.tokens_generated;
    event.text = gen_state.full_text;
    return event;
}
