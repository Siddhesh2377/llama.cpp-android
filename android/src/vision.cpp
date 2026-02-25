// vision.cpp — Vision-Language Model (VLM) support
//
// Contains the full vision encoder pipeline: GGUF loading, SigLIP ViT graph
// builder with idefics3 pixel shuffle projector, and cleanup.
//
// NOTE: The monolith uses detailed vision structs (VisionConfig with patch_size,
// scale_factor, proj_dim, etc.) that are richer than the current types.h
// VisionModelState. These are defined here in an anonymous namespace until
// types.h is updated to match. The three exported functions operate on the
// internal types and are accessed through the wrapper API in vision.h.

#include "gguf-engine/vision.h"
#include "gguf-engine/utils.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Internal types — full-fidelity vision structs from the monolith
// ---------------------------------------------------------------------------
namespace {

static constexpr int VLM_MAX_LAYERS = 32;

struct VCfg {
    uint32_t n_embd;        // 768 (SigLIP-B/16)
    uint32_t n_head;        // 12
    uint32_t n_layer;       // 12
    uint32_t n_ff;          // 3072
    uint32_t patch_size;    // 16
    uint32_t image_size;    // 512
    uint32_t proj_dim;      // LLM n_embd (960 for SmolLM2-360M)
    uint32_t scale_factor;  // 4 (idefics3 pixel shuffle)
    float    norm_eps;      // 1e-6
    int      head_dim;      // 64
    float    image_mean[3];
    float    image_std[3];
};

struct VLayerW {
    ggml_tensor * ln_1_w, * ln_1_b;
    ggml_tensor * attn_q_w, * attn_q_b;
    ggml_tensor * attn_k_w, * attn_k_b;
    ggml_tensor * attn_v_w, * attn_v_b;
    ggml_tensor * attn_out_w, * attn_out_b;
    ggml_tensor * ln_2_w, * ln_2_b;
    ggml_tensor * ffn_up_w, * ffn_up_b;
    ggml_tensor * ffn_down_w, * ffn_down_b;
};

struct VState {
    VCfg vcfg;
    ggml_tensor * patch_embd_w;   // [16, 16, 3, 768] conv2d kernel
    ggml_tensor * patch_embd_b;   // [768]
    ggml_tensor * pos_embd;       // [768, 1024]
    ggml_tensor * post_ln_w;      // [768]
    ggml_tensor * post_ln_b;      // [768]
    ggml_tensor * proj_w;         // [12288, proj_dim] idefics3 FC
    VLayerW layers[VLM_MAX_LAYERS];
    ggml_context * weight_ctx;
    ggml_backend_buffer_t weight_buf;
    gguf_context * gguf_ctx;
    ggml_context * data_ctx;
    bool loaded;
    bool ffn_needs_transpose;
};

// File-level singleton — load_vision_model populates this, build/free use it
static VState g_vs = {};

} // anonymous namespace

// ---------------------------------------------------------------------------
// load_vision_model
// ---------------------------------------------------------------------------
bool load_vision_model(VisionModelState & vs, const char * path, ggml_backend_t backend) {
    printf("\n=== Loading Vision Model ===\n");
    auto t0 = Clock::now();

    ggml_context * data_ctx = nullptr;
    gguf_init_params params = { false, &data_ctx };
    gguf_context * gctx = gguf_init_from_file(path, params);
    if (!gctx) { printf("  FAIL: gguf_init failed for mmproj\n"); return false; }

    g_vs.gguf_ctx = gctx;
    g_vs.data_ctx = data_ctx;

    VCfg & vc = g_vs.vcfg;
    vc.n_embd      = gguf_get_u32(gctx, "clip.vision.embedding_length", 768);
    vc.n_head      = gguf_get_u32(gctx, "clip.vision.attention.head_count", 12);
    vc.n_layer     = gguf_get_u32(gctx, "clip.vision.block_count", 12);
    vc.n_ff        = gguf_get_u32(gctx, "clip.vision.feed_forward_length", 3072);
    vc.patch_size  = gguf_get_u32(gctx, "clip.vision.patch_size", 16);
    vc.image_size  = gguf_get_u32(gctx, "clip.vision.image_size", 512);
    vc.proj_dim    = gguf_get_u32(gctx, "clip.vision.projection_dim", 960);
    vc.scale_factor= gguf_get_u32(gctx, "clip.vision.projector.scale_factor", 4);
    vc.norm_eps    = gguf_get_f32_val(gctx, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    vc.head_dim    = (int)(vc.n_embd / vc.n_head);

    // Image normalization
    int64_t mean_id = gguf_find_key(gctx, "clip.vision.image_mean");
    int64_t std_id  = gguf_find_key(gctx, "clip.vision.image_std");
    for (int i = 0; i < 3; i++) {
        vc.image_mean[i] = (mean_id >= 0 && (int)gguf_get_arr_n(gctx, mean_id) > i)
            ? ((const float *)gguf_get_arr_data(gctx, mean_id))[i] : 0.5f;
        vc.image_std[i] = (std_id >= 0 && (int)gguf_get_arr_n(gctx, std_id) > i)
            ? ((const float *)gguf_get_arr_data(gctx, std_id))[i] : 0.5f;
    }

    printf("  Vision: n_embd=%u n_head=%u n_layer=%u n_ff=%u\n",
        vc.n_embd, vc.n_head, vc.n_layer, vc.n_ff);
    printf("  Image: %ux%u patch=%u scale=%u proj_dim=%u\n",
        vc.image_size, vc.image_size, vc.patch_size, vc.scale_factor, vc.proj_dim);

    if (vc.n_layer > VLM_MAX_LAYERS) {
        printf("  FAIL: too many vision layers (%u)\n", vc.n_layer);
        return false;
    }

    // Detect FFN weight convention before creating tensors
    // Q8_0 mmproj: weights in ggml convention [in, out] -> ffn_up.ne[0] == n_embd
    // F16 mmproj:  weights in PyTorch convention [out, in] -> ffn_up.ne[0] == n_ff
    g_vs.ffn_needs_transpose = false;
    {
        ggml_tensor * test_ffn = ggml_get_tensor(data_ctx, "v.blk.0.ffn_up.weight");
        if (test_ffn && test_ffn->ne[0] != (int64_t)vc.n_embd) {
            printf("  Note: FFN weights in PyTorch convention, will transpose during load\n");
            g_vs.ffn_needs_transpose = true;
        }
    }

    // Count tensors: 6 global + 16 per layer
    int n_tensors = 6 + (int)vc.n_layer * 16;
    size_t ctx_size = (size_t)n_tensors * ggml_tensor_overhead() + 256;
    ggml_init_params wparams = { ctx_size, nullptr, true };
    g_vs.weight_ctx = ggml_init(wparams);

    auto mw = [&](const char * name, bool transpose = false) -> ggml_tensor * {
        ggml_tensor * src = ggml_get_tensor(data_ctx, name);
        if (!src) return nullptr;
        ggml_tensor * dst = nullptr;
        int nd = ggml_n_dims(src);
        if (nd == 1)      dst = ggml_new_tensor_1d(g_vs.weight_ctx, src->type, src->ne[0]);
        else if (nd == 2) {
            if (transpose)
                dst = ggml_new_tensor_2d(g_vs.weight_ctx, src->type, src->ne[1], src->ne[0]);
            else
                dst = ggml_new_tensor_2d(g_vs.weight_ctx, src->type, src->ne[0], src->ne[1]);
        }
        else if (nd == 3) dst = ggml_new_tensor_3d(g_vs.weight_ctx, src->type, src->ne[0], src->ne[1], src->ne[2]);
        else              dst = ggml_new_tensor_4d(g_vs.weight_ctx, src->type, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        ggml_set_name(dst, name);
        return dst;
    };

    // Global tensors
    g_vs.patch_embd_w = mw("v.patch_embd.weight");
    g_vs.patch_embd_b = mw("v.patch_embd.bias");
    g_vs.pos_embd     = mw("v.position_embd.weight");
    g_vs.post_ln_w    = mw("v.post_ln.weight");
    g_vs.post_ln_b    = mw("v.post_ln.bias");
    g_vs.proj_w       = mw("mm.model.fc.weight");

    if (!g_vs.patch_embd_w || !g_vs.pos_embd || !g_vs.proj_w) {
        printf("  FAIL: missing critical vision tensors\n");
        return false;
    }

    // Per-layer
    char buf[128];
    for (uint32_t il = 0; il < vc.n_layer; il++) {
        VLayerW & lw = g_vs.layers[il];
        snprintf(buf, sizeof(buf), "v.blk.%u.ln1.weight", il);   lw.ln_1_w = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.ln1.bias", il);     lw.ln_1_b = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_q.weight", il);  lw.attn_q_w = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_q.bias", il);    lw.attn_q_b = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_k.weight", il);  lw.attn_k_w = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_k.bias", il);    lw.attn_k_b = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_v.weight", il);  lw.attn_v_w = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_v.bias", il);    lw.attn_v_b = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_out.weight", il); lw.attn_out_w = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_out.bias", il);   lw.attn_out_b = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.ln2.weight", il);   lw.ln_2_w = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.ln2.bias", il);     lw.ln_2_b = mw(buf);
        bool tp = g_vs.ffn_needs_transpose;
        snprintf(buf, sizeof(buf), "v.blk.%u.ffn_up.weight", il);   lw.ffn_up_w = mw(buf, tp);
        snprintf(buf, sizeof(buf), "v.blk.%u.ffn_up.bias", il);     lw.ffn_up_b = mw(buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.ffn_down.weight", il); lw.ffn_down_w = mw(buf, tp);
        snprintf(buf, sizeof(buf), "v.blk.%u.ffn_down.bias", il);   lw.ffn_down_b = mw(buf);
    }

    // Allocate + copy
    g_vs.weight_buf = ggml_backend_alloc_ctx_tensors(g_vs.weight_ctx, backend);
    if (!g_vs.weight_buf) { printf("  FAIL: vision weight alloc failed\n"); return false; }
    ggml_backend_buffer_set_usage(g_vs.weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    auto cw = [&](ggml_tensor * dst, const char * name, bool transpose = false) {
        ggml_tensor * src = ggml_get_tensor(data_ctx, name);
        if (!src || !dst) return;
        if (transpose && ggml_n_dims(src) == 2 && src->type == GGML_TYPE_F16) {
            // Transpose F16 2D weight: [ne0, ne1] -> [ne1, ne0]
            int64_t ne0 = src->ne[0], ne1 = src->ne[1];
            std::vector<uint16_t> transposed(ne0 * ne1);
            const uint16_t * s = (const uint16_t *)src->data;
            for (int64_t j = 0; j < ne1; j++)
                for (int64_t i = 0; i < ne0; i++)
                    transposed[j + i * ne1] = s[i + j * ne0];
            ggml_backend_tensor_set(dst, transposed.data(), 0, ne0 * ne1 * sizeof(uint16_t));
        } else {
            ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
        }
    };

    cw(g_vs.patch_embd_w, "v.patch_embd.weight");
    cw(g_vs.patch_embd_b, "v.patch_embd.bias");
    cw(g_vs.pos_embd, "v.position_embd.weight");
    cw(g_vs.post_ln_w, "v.post_ln.weight");
    cw(g_vs.post_ln_b, "v.post_ln.bias");
    cw(g_vs.proj_w, "mm.model.fc.weight");

    for (uint32_t il = 0; il < vc.n_layer; il++) {
        VLayerW & lw = g_vs.layers[il];
        snprintf(buf, sizeof(buf), "v.blk.%u.ln1.weight", il);   cw(lw.ln_1_w, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.ln1.bias", il);     cw(lw.ln_1_b, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_q.weight", il);  cw(lw.attn_q_w, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_q.bias", il);    cw(lw.attn_q_b, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_k.weight", il);  cw(lw.attn_k_w, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_k.bias", il);    cw(lw.attn_k_b, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_v.weight", il);  cw(lw.attn_v_w, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_v.bias", il);    cw(lw.attn_v_b, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_out.weight", il); cw(lw.attn_out_w, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.attn_out.bias", il);   cw(lw.attn_out_b, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.ln2.weight", il);   cw(lw.ln_2_w, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.ln2.bias", il);     cw(lw.ln_2_b, buf);
        // FFN weights: transpose if in PyTorch convention [out, in] -> ggml [in, out]
        snprintf(buf, sizeof(buf), "v.blk.%u.ffn_up.weight", il);   cw(lw.ffn_up_w, buf, g_vs.ffn_needs_transpose);
        snprintf(buf, sizeof(buf), "v.blk.%u.ffn_up.bias", il);     cw(lw.ffn_up_b, buf);
        snprintf(buf, sizeof(buf), "v.blk.%u.ffn_down.weight", il); cw(lw.ffn_down_w, buf, g_vs.ffn_needs_transpose);
        snprintf(buf, sizeof(buf), "v.blk.%u.ffn_down.bias", il);   cw(lw.ffn_down_b, buf);
        // When PyTorch convention: biases are associated with wrong weight names, swap
        if (g_vs.ffn_needs_transpose) std::swap(lw.ffn_up_b, lw.ffn_down_b);
    }

    auto t1 = Clock::now();
    printf("  Weight buffer: %.1f MB\n",
        ggml_backend_buffer_get_size(g_vs.weight_buf) / 1024.0 / 1024.0);
    printf("  Loaded in %.0f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    g_vs.loaded = true;

    // Mirror into the public struct
    vs.weight_ctx = g_vs.weight_ctx;
    vs.weight_buf = g_vs.weight_buf;
    vs.loaded = true;
    return true;
}

// ---------------------------------------------------------------------------
// build_vision_graph — SigLIP ViT + idefics3 pixel shuffle projector
// ---------------------------------------------------------------------------
// Input: [image_size, image_size, 3, 1] F32 normalized pixels
// Output: [proj_dim, n_patches_out] F32 embeddings ready for LLM
struct ggml_cgraph * build_vision_graph(ggml_context * ctx, VisionModelState & vs) {
    (void)vs;  // we use the internal g_vs
    const VCfg & vc = g_vs.vcfg;
    // conv2d output: floor((image_size - patch_size) / patch_size) + 1
    const int n_patches = (int)((vc.image_size - vc.patch_size) / vc.patch_size) + 1;
    const int n_patches_total = n_patches * n_patches;
    const int n_embd = (int)vc.n_embd;                          // 768
    const int n_head = (int)vc.n_head;                          // 12
    const int head_dim = vc.head_dim;                           // 64
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int sf = (int)vc.scale_factor;                        // 4
    const int n_out = n_patches_total / (sf * sf);              // 64

    // Input pixels [W, H, 3, 1]
    ggml_tensor * pixels = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        (int)vc.image_size, (int)vc.image_size, 3, 1);
    ggml_set_name(pixels, "inp_pixels");
    ggml_set_input(pixels);

    // Patch embedding: conv2d -> [n_patches_w, n_patches_h, 768]
    ggml_tensor * cur = ggml_conv_2d(ctx, g_vs.patch_embd_w, pixels,
        (int)vc.patch_size, (int)vc.patch_size, 0, 0, 1, 1);
    // cur: [n_patches, n_patches, 768, 1]

    // Reshape to [768, n_patches_total] for transformer
    cur = ggml_reshape_2d(ctx, cur, n_embd, n_patches_total);
    // Note: conv2d output is [W, H, C] -> reshape treats W*H as seq, C as embd
    // Actually ggml conv2d output: [out_w, out_h, n_embd, 1]
    // reshape_2d: [n_embd, n_patches_total] -- but we need [n_embd, seq]
    // The conv2d output ne[0]=out_w, ne[1]=out_h, ne[2]=n_embd
    // So reshape to [out_w*out_h, n_embd] then permute
    cur = ggml_reshape_2d(ctx, cur, n_patches * n_patches, n_embd);
    cur = ggml_permute(ctx, cur, 1, 0, 2, 3); // -> [n_embd, n_patches_total]
    cur = ggml_cont(ctx, cur);

    // Add patch embedding bias
    if (g_vs.patch_embd_b) {
        cur = ggml_add(ctx, cur, g_vs.patch_embd_b);
    }

    // Add position embeddings
    if (g_vs.pos_embd) {
        cur = ggml_add(ctx, cur, g_vs.pos_embd);
    }

    // Dense attention mask (all zeros = attend everywhere)
    ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
        n_patches_total, n_patches_total);
    ggml_set_name(attn_mask, "v_attn_mask");
    ggml_set_input(attn_mask);

    // Transformer blocks
    for (uint32_t il = 0; il < vc.n_layer; il++) {
        const VLayerW & lw = g_vs.layers[il];
        ggml_tensor * residual = cur;

        // LayerNorm 1 (NOT RMSNorm -- has bias)
        cur = ggml_norm(ctx, cur, vc.norm_eps);
        cur = ggml_mul(ctx, cur, lw.ln_1_w);
        if (lw.ln_1_b) cur = ggml_add(ctx, cur, lw.ln_1_b);

        // Q, K, V projections (separate, not fused)
        ggml_tensor * Q = ggml_mul_mat(ctx, lw.attn_q_w, cur);
        if (lw.attn_q_b) Q = ggml_add(ctx, Q, lw.attn_q_b);
        ggml_tensor * K = ggml_mul_mat(ctx, lw.attn_k_w, cur);
        if (lw.attn_k_b) K = ggml_add(ctx, K, lw.attn_k_b);
        ggml_tensor * V = ggml_mul_mat(ctx, lw.attn_v_w, cur);
        if (lw.attn_v_b) V = ggml_add(ctx, V, lw.attn_v_b);

        // Reshape to multi-head: [head_dim, n_head, seq]
        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, n_patches_total);
        K = ggml_reshape_3d(ctx, K, head_dim, n_head, n_patches_total);
        V = ggml_reshape_3d(ctx, V, head_dim, n_head, n_patches_total);

        // Permute to [head_dim, seq, n_head] for flash_attn_ext
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        // Dense attention (all-zeros mask -> no masking)
        ggml_tensor * attn_out = ggml_flash_attn_ext(ctx,
            Q, K, V, attn_mask, scale, 0.0f, 0.0f);

        // Reshape back to [n_embd, seq]
        attn_out = ggml_cont(ctx, attn_out);
        attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, n_patches_total);

        // Output projection
        cur = ggml_mul_mat(ctx, lw.attn_out_w, attn_out);
        if (lw.attn_out_b) cur = ggml_add(ctx, cur, lw.attn_out_b);

        // Residual 1
        cur = ggml_add(ctx, cur, residual);
        residual = cur;

        // LayerNorm 2
        cur = ggml_norm(ctx, cur, vc.norm_eps);
        cur = ggml_mul(ctx, cur, lw.ln_2_w);
        if (lw.ln_2_b) cur = ggml_add(ctx, cur, lw.ln_2_b);

        // FFN: up -> GELU -> down (weights pre-transposed during load if needed)
        cur = ggml_mul_mat(ctx, lw.ffn_up_w, cur);
        if (lw.ffn_up_b) cur = ggml_add(ctx, cur, lw.ffn_up_b);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_mul_mat(ctx, lw.ffn_down_w, cur);
        if (lw.ffn_down_b) cur = ggml_add(ctx, cur, lw.ffn_down_b);

        // Residual 2
        cur = ggml_add(ctx, cur, residual);
    }

    // Post-LayerNorm
    if (g_vs.post_ln_w) {
        cur = ggml_norm(ctx, cur, vc.norm_eps);
        cur = ggml_mul(ctx, cur, g_vs.post_ln_w);
        if (g_vs.post_ln_b) cur = ggml_add(ctx, cur, g_vs.post_ln_b);
    }

    // idefics3 pixel shuffle: [n_embd, h*w] -> [n_embd*sf*sf, (h/sf)*(w/sf)]
    // Reshape to spatial: [n_embd, w, h]
    cur = ggml_reshape_3d(ctx, cur, n_embd, n_patches, n_patches);
    // Reshape to [n_embd, sf, w/sf, sf, h/sf] via [n_embd*sf, w/sf, sf*h/sf]
    // Simpler: reshape [n_embd, sf, w/sf, h] then permute
    // Actually: pixel shuffle groups sf x sf spatial neighbors and stacks channels
    // [C, H, W] -> [C*sf*sf, H/sf, W/sf]
    // In ggml terms (col-major): cur is [n_embd, n_patches_w, n_patches_h]
    // We need: [n_embd*sf*sf, n_patches_w/sf, n_patches_h/sf]
    int pw = n_patches / sf;  // 8
    int ph = n_patches / sf;  // 8
    // Reshape: [n_embd, sf, pw, sf, ph] -- but ggml max 4d
    // Step 1: [n_embd, sf, pw, n_patches_h] = [768, 4, 8, 32]
    cur = ggml_reshape_4d(ctx, cur, n_embd, sf, pw, n_patches);
    // permute to [n_embd, pw, sf, n_patches_h] = put pw before sf
    cur = ggml_permute(ctx, cur, 0, 2, 1, 3);
    cur = ggml_cont(ctx, cur);
    // Now [n_embd, pw, sf, n_patches_h]
    // Reshape: [n_embd, pw, sf*sf, ph]: need [768, 8, 16, 8] but sf*n_patches_h=128 != sf*sf*ph=128 ok
    // Wait: sf * n_patches_h = 4*32 = 128, and sf*sf*ph = 16*8=128. So:
    cur = ggml_reshape_4d(ctx, cur, n_embd, pw, sf * sf, ph);
    // [768, 8, 16, 8]
    // Now permute to [n_embd*sf*sf, pw, ph]: merge dim0 and dim2
    // permute(0,2,1,3) -> [n_embd, sf*sf, pw, ph] = [768, 16, 8, 8]
    cur = ggml_permute(ctx, cur, 0, 2, 1, 3);
    cur = ggml_cont(ctx, cur);
    // Reshape to [n_embd*sf*sf, pw*ph] = [12288, 64]
    cur = ggml_reshape_2d(ctx, cur, n_embd * sf * sf, pw * ph);

    // Projector FC: [12288, proj_dim] x [12288, n_out] -> [proj_dim, n_out]
    cur = ggml_mul_mat(ctx, g_vs.proj_w, cur);

    ggml_set_name(cur, "vision_embd");
    ggml_set_output(cur);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, cur);
    return graph;
}

// ---------------------------------------------------------------------------
// free_vision_model
// ---------------------------------------------------------------------------
void free_vision_model(VisionModelState & vs) {
    if (g_vs.weight_buf) ggml_backend_buffer_free(g_vs.weight_buf);
    if (g_vs.weight_ctx) ggml_free(g_vs.weight_ctx);
    if (g_vs.data_ctx) ggml_free(g_vs.data_ctx);
    if (g_vs.gguf_ctx) gguf_free(g_vs.gguf_ctx);
    g_vs.loaded = false;
    g_vs.weight_buf = nullptr;
    g_vs.weight_ctx = nullptr;
    g_vs.data_ctx = nullptr;
    g_vs.gguf_ctx = nullptr;

    vs.weight_buf = nullptr;
    vs.weight_ctx = nullptr;
    vs.loaded = false;
}
