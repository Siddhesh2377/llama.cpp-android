// emotional.h — Emotional state tracking and fast weight memory

#pragma once

#include "types.h"

// Keyword-based mood detection (EMA update, clamped to baseline ± 0.3)
void detect_mood_keywords(EmotionalState & es, const std::string & text);

// Apply emotional state to intervention parameters
void apply_mood_to_interventions(const EmotionalState & es, InterventionConfig & iv,
                                 const ModelConfig & cfg, const PersonalityConfig & base_pc);

// Initialize fast weight memory (random projection + Hebbian matrix)
void init_fast_weights(FastWeightMemory & fw, int d_model, uint32_t seed = 42);

// Hebbian fast weight step: W(t) = gamma*W(t-1) + eta*(v x v)
void fast_weight_step(FastWeightMemory & fw, const float * h_full);
