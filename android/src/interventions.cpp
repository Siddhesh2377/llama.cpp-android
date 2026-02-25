// interventions.cpp — Intervention tensor allocation and control vector loading

#include "gguf-engine/interventions.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Allocate intervention tensors on the given backend
// ---------------------------------------------------------------------------
bool init_interventions(InterventionTensors & iv_t, const ModelConfig & cfg,
                        ggml_backend_t backend) {
    if (iv_t.allocated) return true;

    int n_embd = (int)cfg.n_embd;
    int n_embd_head = (int)(cfg.n_head * cfg.head_dim);
    int n_vocab = (int)cfg.n_vocab;
    int n_layer = (int)cfg.n_layer;

    // Count tensors: per-layer (cv + norm_shift + head_scale) * n_layer + logit_bias
    int n_tensors = n_layer * 3 + 1;
    size_t ctx_size = (size_t)n_tensors * ggml_tensor_overhead() + 256;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    iv_t.ctx = ggml_init(params);
    if (!iv_t.ctx) return false;

    for (int il = 0; il < n_layer; il++) {
        iv_t.control_vector[il] = ggml_new_tensor_1d(iv_t.ctx, GGML_TYPE_F32, n_embd);
        ggml_format_name(iv_t.control_vector[il], "cv_%d", il);

        iv_t.norm_shift[il] = ggml_new_tensor_1d(iv_t.ctx, GGML_TYPE_F32, n_embd);
        ggml_format_name(iv_t.norm_shift[il], "ns_%d", il);

        iv_t.head_scale[il] = ggml_new_tensor_1d(iv_t.ctx, GGML_TYPE_F32, n_embd_head);
        ggml_format_name(iv_t.head_scale[il], "hs_%d", il);
    }

    iv_t.logit_bias = ggml_new_tensor_1d(iv_t.ctx, GGML_TYPE_F32, n_vocab);
    ggml_set_name(iv_t.logit_bias, "logit_bias");

    iv_t.buf = ggml_backend_alloc_ctx_tensors(iv_t.ctx, backend);
    if (!iv_t.buf) {
        ggml_free(iv_t.ctx);
        iv_t.ctx = nullptr;
        return false;
    }

    // Zero-fill all intervention tensors (inactive by default)
    ggml_backend_buffer_clear(iv_t.buf, 0);

    // Set head_scale to 1.0 (identity) by default
    std::vector<float> ones(n_embd_head, 1.0f);
    for (int il = 0; il < n_layer; il++) {
        ggml_backend_tensor_set(iv_t.head_scale[il], ones.data(), 0,
            n_embd_head * sizeof(float));
    }

    iv_t.allocated = true;
    printf("  Intervention tensors: %.2f KB (%d layers)\n",
        ggml_backend_buffer_get_size(iv_t.buf) / 1024.0, n_layer);
    return true;
}

// ---------------------------------------------------------------------------
// Free intervention tensors
// ---------------------------------------------------------------------------
void free_interventions(InterventionTensors & iv_t) {
    if (iv_t.buf) ggml_backend_buffer_free(iv_t.buf);
    if (iv_t.ctx) ggml_free(iv_t.ctx);
    iv_t.buf = nullptr;
    iv_t.ctx = nullptr;
    iv_t.allocated = false;
}

// ---------------------------------------------------------------------------
// Control vector GGUF loader
// ---------------------------------------------------------------------------
bool load_control_vectors(InterventionTensors & iv_t, InterventionConfig & iv,
                          const ModelConfig & cfg, const ControlVectorSpec * specs, int n_specs) {
    if (!iv_t.allocated || n_specs <= 0) return false;
    int n_layer = (int)cfg.n_layer;
    int n_embd  = (int)cfg.n_embd;

    // Accumulator: [n_layer * n_embd] floats, zero-initialized
    std::vector<float> accumulated((size_t)n_layer * n_embd, 0.0f);
    int vectors_loaded = 0;

    for (int s = 0; s < n_specs; s++) {
        struct gguf_init_params gip = { /*.no_alloc =*/ false, /*.ctx =*/ nullptr };
        struct gguf_context * gctx = gguf_init_from_file(specs[s].path, gip);
        if (!gctx) {
            printf("  WARN: could not open CV file: %s\n", specs[s].path);
            continue;
        }

        int64_t n_tensors = gguf_get_n_tensors(gctx);
        float strength = specs[s].strength;
        int loaded_this = 0;

        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(gctx, i);
            std::string tname(name);
            // Expect tensor names like "direction.0", "direction.1", ...
            if (tname.rfind("direction.", 0) != 0) continue;
            int layer_id = std::stoi(tname.substr(10));
            if (layer_id < 0 || layer_id >= n_layer) continue;

            // Read tensor data via file offset
            size_t tensor_offset = gguf_get_tensor_offset(gctx, i);
            size_t data_offset = gguf_get_data_offset(gctx);

            FILE * f = fopen(specs[s].path, "rb");
            if (!f) continue;
            std::vector<float> tensor_data(n_embd);
            fseek(f, (long)(data_offset + tensor_offset), SEEK_SET);
            size_t read = fread(tensor_data.data(), sizeof(float), n_embd, f);
            fclose(f);
            if ((int)read != n_embd) continue;

            // Accumulate with strength scaling
            size_t base = (size_t)layer_id * n_embd;
            for (int k = 0; k < n_embd; k++) {
                accumulated[base + k] += strength * tensor_data[k];
            }
            loaded_this++;
        }
        gguf_free(gctx);
        if (loaded_this > 0) vectors_loaded++;
        printf("  CV [%d]: %s (strength=%.2f, %d layers)\n", s, specs[s].path, strength, loaded_this);
    }

    if (vectors_loaded == 0) return false;

    // Upload accumulated vectors to intervention tensors
    int layers_active = 0;
    for (int il = 0; il < n_layer; il++) {
        const float * layer_data = &accumulated[(size_t)il * n_embd];
        // Check if nonzero
        float norm = 0.0f;
        for (int k = 0; k < n_embd; k++) norm += layer_data[k] * layer_data[k];
        if (norm > 1e-12f) {
            ggml_backend_tensor_set(iv_t.control_vector[il], layer_data, 0, n_embd * sizeof(float));
            layers_active++;
        }
    }
    iv.flags |= IV_CONTROL_VECTORS;
    printf("  Control vectors: %d files, %d/%d layers active\n", vectors_loaded, layers_active, n_layer);
    return true;
}

// ---------------------------------------------------------------------------
// Compute per-head importance from accumulated control vectors
// ---------------------------------------------------------------------------
void compute_head_importance(InterventionTensors & iv_t, InterventionConfig & iv,
                             const ModelConfig & cfg, const float * accumulated) {
    int n_layer = (int)cfg.n_layer;
    int n_embd  = (int)cfg.n_embd;
    int n_head  = (int)cfg.n_head;
    int head_dim = (int)cfg.head_dim;
    int n_embd_head = n_head * head_dim;

    std::vector<float> head_scales(n_embd_head);

    for (int il = 0; il < n_layer; il++) {
        const float * cv = &accumulated[(size_t)il * n_embd];

        // Compute L2 norm per head (using first n_embd values, heads map to head_dim chunks)
        std::vector<float> head_norms(n_head, 0.0f);
        for (int h = 0; h < n_head && h * head_dim < n_embd; h++) {
            float norm = 0.0f;
            for (int d = 0; d < head_dim && h * head_dim + d < n_embd; d++) {
                float v = cv[h * head_dim + d];
                norm += v * v;
            }
            head_norms[h] = sqrtf(norm);
        }

        // Find 25th and 75th percentile thresholds
        std::vector<float> sorted_norms = head_norms;
        std::sort(sorted_norms.begin(), sorted_norms.end());
        float p25 = sorted_norms[n_head / 4];
        float p75 = sorted_norms[3 * n_head / 4];

        // Assign scales: top 25% → 1.3x, bottom 25% → 0.7x, middle → 1.0x
        for (int h = 0; h < n_head; h++) {
            float s = 1.0f;
            if (head_norms[h] >= p75)      s = 1.3f;
            else if (head_norms[h] <= p25) s = 0.7f;
            // Fill head_dim consecutive values with same scalar
            for (int d = 0; d < head_dim; d++) {
                head_scales[(size_t)h * head_dim + d] = s;
            }
        }

        ggml_backend_tensor_set(iv_t.head_scale[il], head_scales.data(), 0,
            n_embd_head * sizeof(float));
    }
    iv.flags |= IV_HEAD_RESCALE;
    printf("  Head importance: %d heads/layer, rescale 0.7x/1.0x/1.3x\n", n_head);
}
