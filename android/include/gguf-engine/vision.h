// vision.h — Vision-Language Model (VLM) support

#pragma once

#include "types.h"

// Load vision model from GGUF file
bool load_vision_model(VisionModelState & vstate, const char * path,
                       ggml_backend_t backend);

// Build vision transformer graph (uses internal state populated by load_vision_model)
struct ggml_cgraph * build_vision_graph(
    struct ggml_context * ctx, VisionModelState & vstate);

// Free vision model resources
void free_vision_model(VisionModelState & vstate);
