// vision.h — Vision-Language Model (VLM) support

#pragma once

#include "types.h"

// Load vision model from GGUF file
// quantize: convert F16/F32 2D weights to Q8_0 for I8MM GEMM acceleration
bool load_vision_model(VisionModelState & vstate, const char * path,
                       ggml_backend_t backend, bool quantize = false);

// Build vision transformer graph (uses internal state populated by load_vision_model)
// n_layers_override: -1 = use all layers, >=0 = use that many layers
struct ggml_cgraph * build_vision_graph(
    struct ggml_context * ctx, VisionModelState & vstate, int n_layers_override = -1);

// Override the internal image size for variable resolution (must be multiple of patch_size)
void set_vision_image_size(uint32_t image_size);

// Free vision model resources
void free_vision_model(VisionModelState & vstate);
