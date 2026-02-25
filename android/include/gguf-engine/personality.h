// personality.h — Personality system (JSON parsing, profile state)

#pragma once

#include "types.h"

// Parse personality from JSON file
bool parse_personality_json(const char * path, PersonalityConfig & pc);

// Convert PersonalityConfig to intervention configuration
void profile_from_personality(const PersonalityConfig & pc, ProfileState & ps,
                              int n_layer);

// Apply profile to active intervention system
void profile_apply(const ProfileState & ps, InterventionConfig & iv,
                   InterventionTensors & iv_t, const ModelConfig & cfg,
                   ggml_backend_t backend);

// Save profile state to binary file
bool profile_save(const ProfileState & ps, const char * path);

// Load profile state from binary file
bool profile_load(ProfileState & ps, const char * path);
