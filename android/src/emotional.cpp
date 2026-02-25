// emotional.cpp — Emotional state tracking and fast weight memory
//
// Keyword-based mood detection with EMA smoothing,
// intervention parameter adjustment, and Hopfield-style fast weight memory.

#include "gguf-engine/emotional.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Mood keyword table (file-local)
// ---------------------------------------------------------------------------
namespace {

static const MoodKeyword MOOD_KEYWORDS[] = {
    // Warmth
    {"thanks",     MOOD_WARMTH,  +0.15f},
    {"thank you",  MOOD_WARMTH,  +0.20f},
    {"please",     MOOD_WARMTH,  +0.10f},
    {"love",       MOOD_WARMTH,  +0.20f},
    {"hate",       MOOD_WARMTH,  -0.20f},
    {"angry",      MOOD_WARMTH,  -0.15f},
    {"sorry",      MOOD_WARMTH,  +0.10f},
    {"great",      MOOD_WARMTH,  +0.10f},
    {"terrible",   MOOD_WARMTH,  -0.15f},
    {"awesome",    MOOD_WARMTH,  +0.15f},
    {"annoyed",    MOOD_WARMTH,  -0.10f},
    {"frustrated", MOOD_WARMTH,  -0.15f},
    {"happy",      MOOD_WARMTH,  +0.15f},
    {"sad",        MOOD_WARMTH,  -0.10f},
    {"wonderful",  MOOD_WARMTH,  +0.15f},
    {"awful",      MOOD_WARMTH,  -0.15f},
    // Energy
    {"urgent",     MOOD_ENERGY,  +0.20f},
    {"asap",       MOOD_ENERGY,  +0.20f},
    {"quick",      MOOD_ENERGY,  +0.15f},
    {"relax",      MOOD_ENERGY,  -0.15f},
    {"chill",      MOOD_ENERGY,  -0.15f},
    {"excited",    MOOD_ENERGY,  +0.20f},
    {"bored",      MOOD_ENERGY,  -0.10f},
    {"help",       MOOD_ENERGY,  +0.10f},
    {"emergency",  MOOD_ENERGY,  +0.25f},
    {"slow",       MOOD_ENERGY,  -0.10f},
    // Formality
    {"sir",        MOOD_FORMALITY, +0.15f},
    {"dear",       MOOD_FORMALITY, +0.10f},
    {"lol",        MOOD_FORMALITY, -0.20f},
    {"lmao",       MOOD_FORMALITY, -0.20f},
    {"haha",       MOOD_FORMALITY, -0.15f},
    {"yo",         MOOD_FORMALITY, -0.15f},
    {"hey",        MOOD_FORMALITY, -0.10f},
    {"bruh",       MOOD_FORMALITY, -0.20f},
    {"please",     MOOD_FORMALITY, +0.05f},
};
static constexpr int N_MOOD_KEYWORDS = sizeof(MOOD_KEYWORDS) / sizeof(MOOD_KEYWORDS[0]);

} // anonymous namespace

// ---------------------------------------------------------------------------
// Keyword-based mood detection (EMA update, clamped to baseline +/- max_offset)
// ---------------------------------------------------------------------------
void detect_mood_keywords(EmotionalState & es, const std::string & text) {
    // Lowercase copy
    std::string lower = text;
    for (auto & c : lower) c = (char)tolower((unsigned char)c);

    float deltas[MOOD_COUNT] = {};
    for (int i = 0; i < N_MOOD_KEYWORDS; i++) {
        if (lower.find(MOOD_KEYWORDS[i].word) != std::string::npos) {
            deltas[MOOD_KEYWORDS[i].axis] += MOOD_KEYWORDS[i].delta;
        }
    }

    // EMA update + clamp to baseline ± max_offset
    for (int a = 0; a < MOOD_COUNT; a++) {
        if (deltas[a] != 0.0f) {
            float target = es.axes[a] + deltas[a];
            es.axes[a] = es.alpha * es.axes[a] + (1.0f - es.alpha) * target;
            float lo = es.baseline[a] - es.max_offset;
            float hi = es.baseline[a] + es.max_offset;
            es.axes[a] = std::max(lo, std::min(hi, es.axes[a]));
        }
    }
    es.updates++;
}

// ---------------------------------------------------------------------------
// Apply emotional state to intervention parameters
// ---------------------------------------------------------------------------
void apply_mood_to_interventions(const EmotionalState & es, InterventionConfig & iv,
                                 const ModelConfig & cfg, const PersonalityConfig & base_pc) {
    int n_layer = (int)cfg.n_layer;

    // Warmth offset → temperature profile shift
    float warmth_off = es.axes[MOOD_WARMTH] - es.baseline[MOOD_WARMTH];
    float temp_early_adj = base_pc.temp_early + warmth_off * 0.3f;
    float temp_late_adj  = base_pc.temp_late  - warmth_off * 0.2f;

    // Energy offset → gate strength adjustment
    float energy_off = es.axes[MOOD_ENERGY] - es.baseline[MOOD_ENERGY];
    float gate_adj = energy_off * 0.1f;

    for (int il = 0; il < n_layer; il++) {
        float t = (float)il / (float)std::max(n_layer - 1, 1);
        // Temperature bands
        if (t < 0.33f)      iv.attn_temp[il] = temp_early_adj;
        else if (t < 0.66f) iv.attn_temp[il] = base_pc.temp_mid;
        else                iv.attn_temp[il] = temp_late_adj;
        // Gate bands (middle 50% of layers)
        if (t > 0.25f && t < 0.75f) {
            iv.attn_gate[il] = std::max(0.8f, std::min(1.0f, base_pc.attn_gate_mid + gate_adj));
            iv.ffn_gate[il]  = std::max(0.8f, std::min(1.0f, base_pc.ffn_gate_mid + gate_adj));
        } else {
            iv.attn_gate[il] = 1.0f;
            iv.ffn_gate[il]  = 1.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Fast Weight Associative Memory — Hopfield-style (Schmidhuber 1992)
// ---------------------------------------------------------------------------
void init_fast_weights(FastWeightMemory & fw, int d_model, uint32_t seed) {
    fw.d_model = d_model;
    fw.d_reduced = FW_DIM_REDUCED;

    // Xavier-init random projection
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)fw.d_reduced));

    fw.proj_down.resize((size_t)fw.d_reduced * d_model);
    fw.proj_up.resize((size_t)d_model * fw.d_reduced);
    for (size_t i = 0; i < fw.proj_down.size(); i++) {
        fw.proj_down[i] = dist(rng);
    }
    // proj_up = transpose(proj_down)
    for (int i = 0; i < fw.d_reduced; i++) {
        for (int j = 0; j < d_model; j++) {
            fw.proj_up[(size_t)j * fw.d_reduced + i] = fw.proj_down[(size_t)i * d_model + j];
        }
    }

    fw.W_fast.assign((size_t)fw.d_reduced * fw.d_reduced, 0.0f);
    fw.h_reduced.resize(fw.d_reduced);
    fw.recall.resize(fw.d_reduced);
    fw.recall_full.resize(d_model);
    fw.initialized = true;

    printf("  Fast weights: %dD → %dD, matrix %.1f KB\n",
        d_model, fw.d_reduced,
        (float)fw.d_reduced * fw.d_reduced * sizeof(float) / 1024.0f);
}

void fast_weight_step(FastWeightMemory & fw, const float * h_full) {
    if (!fw.initialized) return;

    // 1. Project down: h_reduced = proj_down × h_full
    for (int i = 0; i < fw.d_reduced; i++) {
        float sum = 0.0f;
        const float * row = &fw.proj_down[(size_t)i * fw.d_model];
        for (int j = 0; j < fw.d_model; j++) {
            sum += row[j] * h_full[j];
        }
        fw.h_reduced[i] = sum;
    }

    // 2. Recall: recall = W_fast × h_reduced
    for (int i = 0; i < fw.d_reduced; i++) {
        float sum = 0.0f;
        const float * row = &fw.W_fast[(size_t)i * fw.d_reduced];
        for (int j = 0; j < fw.d_reduced; j++) {
            sum += row[j] * fw.h_reduced[j];
        }
        fw.recall[i] = sum;
    }

    // 3. Hebbian write: W_fast = γ·W_fast + η·(h_reduced ⊗ h_reduced)
    for (int i = 0; i < fw.d_reduced; i++) {
        float * row = &fw.W_fast[(size_t)i * fw.d_reduced];
        float eta_hi = fw.eta * fw.h_reduced[i];
        for (int j = 0; j < fw.d_reduced; j++) {
            row[j] = fw.gamma * row[j] + eta_hi * fw.h_reduced[j];
        }
    }

    // 4. Project recall back up: recall_full = proj_up × recall
    for (int i = 0; i < fw.d_model; i++) {
        float sum = 0.0f;
        const float * row = &fw.proj_up[(size_t)i * fw.d_reduced];
        for (int j = 0; j < fw.d_reduced; j++) {
            sum += row[j] * fw.recall[j];
        }
        fw.recall_full[i] = sum;
    }
}
