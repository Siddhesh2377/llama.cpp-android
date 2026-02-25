// personality.cpp — Personality JSON parsing, profile state, persistence

#include "gguf-engine/personality.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Profile binary format constants
// ---------------------------------------------------------------------------
static constexpr uint32_t PROF_MAGIC   = 0x464F5250; // "PROF"
static constexpr uint32_t PROF_VERSION = 1;

// ---------------------------------------------------------------------------
// Parse personality from JSON file
// ---------------------------------------------------------------------------
bool parse_personality_json(const char * path, PersonalityConfig & pc) {
    FILE * f = fopen(path, "rb");
    if (!f) { printf("  Cannot open: %s\n", path); return false; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string json((size_t)len, '\0');
    fread(&json[0], 1, (size_t)len, f);
    fclose(f);

    // Extract string value for a key (handles simple escaped chars)
    auto get_str = [&](const char * key) -> std::string {
        std::string needle = std::string("\"") + key + "\"";
        size_t pos = json.find(needle);
        if (pos == std::string::npos) return "";
        pos = json.find('"', pos + needle.size());
        if (pos == std::string::npos) return "";
        pos++; // skip opening quote
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                if (json[pos] == 'n') result += '\n';
                else if (json[pos] == 't') result += '\t';
                else if (json[pos] == '"') result += '"';
                else if (json[pos] == '\\') result += '\\';
                else result += json[pos];
            } else {
                result += json[pos];
            }
            pos++;
        }
        return result;
    };

    auto get_float = [&](const char * key, float def) -> float {
        std::string needle = std::string("\"") + key + "\"";
        size_t pos = json.find(needle);
        if (pos == std::string::npos) return def;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return def;
        return (float)atof(json.c_str() + pos + 1);
    };

    auto get_int = [&](const char * key, int def) -> int {
        std::string needle = std::string("\"") + key + "\"";
        size_t pos = json.find(needle);
        if (pos == std::string::npos) return def;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return def;
        return atoi(json.c_str() + pos + 1);
    };

    pc.name           = get_str("name");
    pc.system_prompt   = get_str("system_prompt");
    pc.user_message    = get_str("user_message");
    pc.temp_early      = get_float("temp_early", pc.temp_early);
    pc.temp_mid        = get_float("temp_mid", pc.temp_mid);
    pc.temp_late       = get_float("temp_late", pc.temp_late);
    pc.attn_gate_mid   = get_float("attn_gate_mid", pc.attn_gate_mid);
    pc.ffn_gate_mid    = get_float("ffn_gate_mid", pc.ffn_gate_mid);
    pc.logit_bias_eos  = get_float("logit_bias_eos", pc.logit_bias_eos);
    pc.sampling_temp   = get_float("sampling_temp", pc.sampling_temp);
    pc.sampling_top_k  = get_int("sampling_top_k", pc.sampling_top_k);
    pc.sampling_top_p  = get_float("sampling_top_p", pc.sampling_top_p);
    pc.rep_penalty     = get_float("rep_penalty", pc.rep_penalty);
    pc.thinking        = get_int("thinking", pc.thinking);

    // Control vectors (up to 4)
    for (int i = 0; i < 4; i++) {
        char key_path[32], key_str[32];
        snprintf(key_path, sizeof(key_path), "cv_path_%d", i);
        snprintf(key_str, sizeof(key_str), "cv_strength_%d", i);
        std::string p = get_str(key_path);
        if (!p.empty()) {
            pc.cv_paths[pc.n_cv] = p;
            pc.cv_strengths[pc.n_cv] = get_float(key_str, 1.0f);
            pc.n_cv++;
        }
    }

    // Mood baselines
    pc.mood_warmth    = get_float("mood_warmth", pc.mood_warmth);
    pc.mood_energy    = get_float("mood_energy", pc.mood_energy);
    pc.mood_formality = get_float("mood_formality", pc.mood_formality);

    // Stall config
    std::string sp = get_str("stall_prompt");
    if (!sp.empty()) pc.stall_prompt = sp;
    pc.stall_max_tokens = get_int("stall_max_tokens", pc.stall_max_tokens);

    // Fast weight config
    pc.fw_dim_reduced = get_int("fw_dim_reduced", pc.fw_dim_reduced);
    pc.fw_gamma       = get_float("fw_gamma", pc.fw_gamma);
    pc.fw_eta         = get_float("fw_eta", pc.fw_eta);
    pc.fw_enabled     = get_int("fw_enabled", 1) != 0;

    if (pc.name.empty()) {
        printf("  Missing \"name\" in personality JSON\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Build a profile from PersonalityConfig + layer count
// ---------------------------------------------------------------------------
void profile_from_personality(const PersonalityConfig & pc, ProfileState & ps,
                              int n_layer) {
    ps.personality = pc;
    ps.active = true;

    // Sampling
    ps.sampling.temp = pc.sampling_temp;
    ps.sampling.top_k = pc.sampling_top_k;
    ps.sampling.top_p = pc.sampling_top_p;
    ps.sampling.rep_penalty = pc.rep_penalty;

    // Intervention config: temperature profile + gated residuals + logit bias
    ps.iv_config.reset();
    ps.iv_config.flags = IV_ATTN_TEMPERATURE | IV_GATED_RESIDUAL | IV_LOGIT_BIAS;

    for (int il = 0; il < n_layer; il++) {
        float t = (float)il / (float)std::max(n_layer - 1, 1);
        if (t < 0.33f)      ps.iv_config.attn_temp[il] = pc.temp_early;
        else if (t < 0.66f) ps.iv_config.attn_temp[il] = pc.temp_mid;
        else                ps.iv_config.attn_temp[il] = pc.temp_late;
    }
    for (int il = 0; il < n_layer; il++) {
        float t = (float)il / (float)std::max(n_layer - 1, 1);
        if (t > 0.25f && t < 0.75f) {
            ps.iv_config.attn_gate[il] = pc.attn_gate_mid;
            ps.iv_config.ffn_gate[il]  = pc.ffn_gate_mid;
        } else {
            ps.iv_config.attn_gate[il] = 1.0f;
            ps.iv_config.ffn_gate[il]  = 1.0f;
        }
    }

    // Enable control vectors flag if paths are specified
    if (pc.n_cv > 0) {
        ps.iv_config.flags |= IV_CONTROL_VECTORS;
    }
}

// ---------------------------------------------------------------------------
// Apply profile to intervention tensors + sampling params
// ---------------------------------------------------------------------------
void profile_apply(const ProfileState & ps, InterventionConfig & iv,
                   InterventionTensors & iv_t, const ModelConfig & cfg,
                   ggml_backend_t backend) {
    // Copy scalar intervention config
    iv = ps.iv_config;

    // Upload logit bias tensor if allocated
    if ((iv.flags & IV_LOGIT_BIAS) && iv_t.logit_bias) {
        std::vector<float> bias((size_t)cfg.n_vocab, 0.0f);
        // EOS bias from personality (applied at vocab level)
        // Note: caller is responsible for setting the EOS token bias
        // in the bias vector if needed, since EOS token ID is model-specific
        ggml_backend_tensor_set(iv_t.logit_bias, bias.data(), 0,
            cfg.n_vocab * sizeof(float));
    }

    // Attn temperature, gates are used as scalars in build_graph() — no tensor upload needed.
    // They're read directly from InterventionConfig during graph construction.
    // Only logit_bias needs tensor upload (handled above).
}

// ---------------------------------------------------------------------------
// Save profile to binary file
// ---------------------------------------------------------------------------
bool profile_save(const ProfileState & ps, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) { printf("  Cannot write: %s\n", path); return false; }

    // Header
    uint32_t magic = PROF_MAGIC;
    uint32_t version = PROF_VERSION;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);

    // Name (64 bytes, null-padded)
    char name[64] = {};
    snprintf(name, sizeof(name), "%s", ps.personality.name.c_str());
    fwrite(name, 64, 1, f);

    // Intervention config (fixed size)
    fwrite(&ps.iv_config, sizeof(InterventionConfig), 1, f);

    // Sampling params (fixed size)
    fwrite(&ps.sampling, sizeof(SamplingParams), 1, f);

    // Active flag
    uint8_t flags = (uint8_t)ps.active;
    fwrite(&flags, 1, 1, f);

    fclose(f);
    printf("  Profile saved: %s (%zu bytes)\n", path,
        72 + sizeof(InterventionConfig) + sizeof(SamplingParams) + 1);
    return true;
}

// ---------------------------------------------------------------------------
// Load profile from binary file
// ---------------------------------------------------------------------------
bool profile_load(ProfileState & ps, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { printf("  Cannot read: %s\n", path); return false; }

    uint32_t magic;
    fread(&magic, 4, 1, f);
    if (magic != PROF_MAGIC) {
        printf("  Invalid profile magic: 0x%08X\n", magic);
        fclose(f);
        return false;
    }

    uint32_t version;
    fread(&version, 4, 1, f);

    char name[64];
    fread(name, 64, 1, f);
    ps.personality.name = name;

    fread(&ps.iv_config, sizeof(InterventionConfig), 1, f);
    fread(&ps.sampling, sizeof(SamplingParams), 1, f);

    uint8_t flags;
    fread(&flags, 1, 1, f);
    ps.active = (flags & 1) != 0;

    fclose(f);
    printf("  Profile loaded: %s (name=%s)\n", path, name);
    return true;
}
