// interventions.h — Intervention tensor allocation and control vector loading

#pragma once

#include "types.h"

// Allocate intervention tensors on backend
bool init_interventions(InterventionTensors & iv_t, const ModelConfig & cfg,
                        ggml_backend_t backend);

// Free intervention tensors
void free_interventions(InterventionTensors & iv_t);

// Load control vectors from GGUF files, accumulate into intervention tensors
bool load_control_vectors(InterventionTensors & iv_t, InterventionConfig & iv,
                          const ModelConfig & cfg, const ControlVectorSpec * specs, int n_specs);

// Compute per-head importance from accumulated control vectors
void compute_head_importance(InterventionTensors & iv_t, InterventionConfig & iv,
                             const ModelConfig & cfg, const float * accumulated);
