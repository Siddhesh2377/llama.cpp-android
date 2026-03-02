#include "ggml-engine-internal.h"
#include "vlm/mtmd.h"
#include "vlm/mtmd-helper.h"
#include "vlm/clip.h"

#include <sstream>
#include <cstring>

struct ggml_engine_vlm {
    mtmd_context       * mtmd_ctx = nullptr;
    ggml_engine_t      * engine   = nullptr;  // non-owning back-reference
    ggml_engine_vlm_params params;
};

// ----- VLM API implementation -----

ggml_engine_vlm_params ggml_engine_vlm_default_params(void) {
    ggml_engine_vlm_params p{};
    p.n_threads        = 0;   // same as engine
    p.image_min_tokens = -1;  // model default
    p.image_max_tokens = -1;  // model default
    return p;
}

ggml_engine_vlm_t * ggml_engine_vlm_load(
    ggml_engine_t * engine, const char * mmproj_path,
    ggml_engine_vlm_params params)
{
    if (!engine || !engine->model || !mmproj_path) return nullptr;

    auto mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = false;  // CPU only
    mtmd_params.n_threads = params.n_threads > 0 ? params.n_threads : engine->params.n_threads_batch;
    mtmd_params.print_timings = false;
    mtmd_params.warmup = true;

    if (engine->params.flash_attn) {
        mtmd_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    if (params.image_min_tokens >= 0) {
        mtmd_params.image_min_tokens = params.image_min_tokens;
    }
    if (params.image_max_tokens >= 0) {
        mtmd_params.image_max_tokens = params.image_max_tokens;
    }

    mtmd_context * mtmd_ctx = mtmd_init_from_file(mmproj_path, engine->model, mtmd_params);
    if (!mtmd_ctx) {
        return nullptr;
    }

    auto * vlm = new ggml_engine_vlm();
    vlm->mtmd_ctx = mtmd_ctx;
    vlm->engine   = engine;
    vlm->params   = params;

    return vlm;
}

ggml_engine_vlm_t * ggml_engine_vlm_load_from_fd(
    ggml_engine_t * engine, int fd,
    ggml_engine_vlm_params params)
{
    if (fd < 0) return nullptr;
    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", fd);
    return ggml_engine_vlm_load(engine, fd_path, params);
}

void ggml_engine_vlm_free(ggml_engine_vlm_t * vlm) {
    if (!vlm) return;
    if (vlm->mtmd_ctx) {
        mtmd_free(vlm->mtmd_ctx);
    }
    delete vlm;
}

bool ggml_engine_vlm_is_loaded(const ggml_engine_vlm_t * vlm) {
    return vlm && vlm->mtmd_ctx;
}

ggml_engine_status ggml_engine_vlm_generate(
    ggml_engine_t * engine, ggml_engine_vlm_t * vlm,
    const char * prompt,
    const ggml_engine_image * images, int32_t n_images,
    ggml_engine_sampling sampling,
    ggml_engine_token_callback callback, void * user_data)
{
    if (!engine || !engine->model || !engine->ctx) {
        return GGML_ENGINE_ERROR_NO_MODEL;
    }
    if (!vlm || !vlm->mtmd_ctx) {
        return GGML_ENGINE_ERROR_VLM_NO_PROJ;
    }

    engine->cancelled.store(false);
    engine->response.clear();
    memset(&engine->perf, 0, sizeof(engine->perf));

    // create bitmaps from images
    std::vector<mtmd_bitmap *> bitmaps;
    for (int32_t i = 0; i < n_images; i++) {
        mtmd_bitmap * bmp = nullptr;
        if (images[i].width == 0 || images[i].height == 0) {
            // file mode: auto-detect format from bytes
            bmp = mtmd_helper_bitmap_init_from_buf(vlm->mtmd_ctx, images[i].data, images[i].size);
        } else {
            // raw RGB mode
            bmp = mtmd_bitmap_init(images[i].width, images[i].height, images[i].data);
        }
        if (!bmp) {
            for (auto * b : bitmaps) mtmd_bitmap_free(b);
            return GGML_ENGINE_ERROR_VLM_ENCODE;
        }
        bitmaps.push_back(bmp);
    }

    // build const pointer array for mtmd_tokenize
    std::vector<const mtmd_bitmap *> bitmap_ptrs(bitmaps.begin(), bitmaps.end());

    // tokenize prompt + images into chunks
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text;
    input_text.text = prompt;
    input_text.add_special = true;
    input_text.parse_special = true;

    int32_t tok_result = mtmd_tokenize(vlm->mtmd_ctx, chunks,
        &input_text, bitmap_ptrs.data(), bitmap_ptrs.size());

    // free bitmaps - tokenize has processed them
    for (auto * b : bitmaps) mtmd_bitmap_free(b);

    if (tok_result != 0) {
        mtmd_input_chunks_free(chunks);
        return GGML_ENGINE_ERROR_TOKENIZE;
    }

    // clear KV cache
    llama_memory_t mem = llama_get_memory(engine->ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    engine->n_past = 0;

    // process all chunks: text decoding + image encoding + embedding injection
    int64_t t_prompt_start = llama_time_us();

    llama_pos new_n_past = 0;
    int32_t eval_result = mtmd_helper_eval_chunks(
        vlm->mtmd_ctx, engine->ctx, chunks,
        0,      // n_past
        0,      // seq_id
        engine->params.n_batch,
        true,   // logits_last
        &new_n_past);

    mtmd_input_chunks_free(chunks);

    if (eval_result != 0) {
        return GGML_ENGINE_ERROR_VLM_ENCODE;
    }

    engine->n_past = new_n_past;

    int64_t t_prompt_end = llama_time_us();
    engine->perf.prompt_eval_ms = (t_prompt_end - t_prompt_start) / 1000.0;
    engine->perf.prompt_tokens = engine->n_past;

    // shared generation loop handles sampling + autoregressive decoding
    return ggml_engine_generate_loop(engine, sampling, callback, user_data);
}

int32_t ggml_engine_vlm_encode_image(
    ggml_engine_vlm_t * vlm, const ggml_engine_image * image)
{
    if (!vlm || !vlm->mtmd_ctx || !image) return -1;

    // create bitmap
    mtmd_bitmap * bmp = nullptr;
    if (image->width == 0 || image->height == 0) {
        bmp = mtmd_helper_bitmap_init_from_buf(vlm->mtmd_ctx, image->data, image->size);
    } else {
        bmp = mtmd_bitmap_init(image->width, image->height, image->data);
    }
    if (!bmp) return -1;

    // tokenize with a simple marker prompt to get token count
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text;
    const char * marker = mtmd_default_marker();
    input_text.text = marker;
    input_text.add_special = false;
    input_text.parse_special = true;

    const mtmd_bitmap * bmp_ptr = bmp;
    int32_t tok_result = mtmd_tokenize(vlm->mtmd_ctx, chunks,
        &input_text, &bmp_ptr, 1);

    mtmd_bitmap_free(bmp);

    if (tok_result != 0) {
        mtmd_input_chunks_free(chunks);
        return -1;
    }

    // count image tokens from chunks
    int32_t n_image_tokens = 0;
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            n_image_tokens += (int32_t)mtmd_input_chunk_get_n_tokens(chunk);
        }
    }

    mtmd_input_chunks_free(chunks);
    return n_image_tokens;
}

char * ggml_engine_vlm_info_json(const ggml_engine_vlm_t * vlm) {
    if (!vlm || !vlm->mtmd_ctx) return strdup_alloc("{}");

    std::ostringstream json;
    json << "{\n";
    json << "  \"supports_vision\": " << (mtmd_support_vision(vlm->mtmd_ctx) ? "true" : "false") << ",\n";
    json << "  \"supports_audio\": " << (mtmd_support_audio(vlm->mtmd_ctx) ? "true" : "false") << ",\n";
    json << "  \"uses_mrope\": " << (mtmd_decode_use_mrope(vlm->mtmd_ctx) ? "true" : "false") << ",\n";
    json << "  \"uses_non_causal\": " << (mtmd_decode_use_non_causal(vlm->mtmd_ctx) ? "true" : "false") << ",\n";
    json << "  \"audio_bitrate\": " << mtmd_get_audio_bitrate(vlm->mtmd_ctx) << ",\n";
    json << "  \"default_marker\": \"" << mtmd_default_marker() << "\"\n";
    json << "}";

    return strdup_alloc(json.str());
}

const char * ggml_engine_vlm_default_marker(void) {
    return mtmd_default_marker();
}

bool ggml_engine_vlm_supports_vision(const ggml_engine_vlm_t * vlm) {
    if (!vlm || !vlm->mtmd_ctx) return false;
    return mtmd_support_vision(vlm->mtmd_ctx);
}

bool ggml_engine_vlm_supports_audio(const ggml_engine_vlm_t * vlm) {
    if (!vlm || !vlm->mtmd_ctx) return false;
    return mtmd_support_audio(vlm->mtmd_ctx);
}
