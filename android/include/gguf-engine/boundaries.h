// boundaries.h — Async generation boundary system

#pragma once

#include "types.h"
#include "grammar.h"

// Generate tokens until a boundary is hit (stop string, EOS, tool call, max tokens)
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
    bool stream = true
);
