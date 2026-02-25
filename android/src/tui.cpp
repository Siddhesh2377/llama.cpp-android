// tui.cpp — Terminal UI helpers, engine configuration, and chat commands

#include "gguf-engine/tui.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

// ---------------------------------------------------------------------------
// TUI namespace — ANSI color terminal helpers
// ---------------------------------------------------------------------------
namespace tui {

bool color_enabled = true;

void c(const char * color) { if (color_enabled) printf("%s", color); }
void reset() { c(RESET); }

void divider() { c(DIM); printf("─────────────────────────────────────────\n"); reset(); }

void progress_bar(const char * label, float pct, int width) {
    int filled = (int)(pct * width);
    c(CYAN); printf("  %s ", label); c(WHITE); printf("[");
    c(GREEN);
    for (int i = 0; i < width; i++) printf(i < filled ? "█" : " ");
    c(WHITE); printf("] "); c(YELLOW);
    printf("%.0f%%\n", pct * 100.0f); reset();
}

void status(const char * key, const char * val) {
    c(DIM); printf("  %-16s", key); reset(); printf("%s\n", val);
}
void status_int(const char * key, int val) {
    c(DIM); printf("  %-16s", key); reset(); printf("%d\n", val);
}
void status_float(const char * key, float val) {
    c(DIM); printf("  %-16s", key); reset(); printf("%.2f\n", val);
}
void banner(const char * text) {
    c(BOLD); c(CYAN); printf("\n  %s\n", text); reset(); divider();
}
void error(const char * msg) { c(RED); printf("  Error: %s\n", msg); reset(); }
void success(const char * msg) { c(GREEN); printf("  %s\n", msg); reset(); }
void info(const char * msg) { c(DIM); printf("  %s\n", msg); reset(); }

} // namespace tui

// ---------------------------------------------------------------------------
// EngineConfig — persistent runtime configuration
// ---------------------------------------------------------------------------
bool load_engine_config(EngineConfig & ec, const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 16384) { fclose(f); return false; }
    std::string json(sz, '\0');
    fread(&json[0], 1, sz, f); fclose(f);

    auto get_str = [&](const char * key, char * dst, size_t maxlen) {
        std::string search = std::string("\"") + key + "\"";
        size_t pos = json.find(search); if (pos == std::string::npos) return;
        pos = json.find(':', pos); if (pos == std::string::npos) return;
        size_t q1 = json.find('"', pos + 1); if (q1 == std::string::npos) return;
        size_t q2 = json.find('"', q1 + 1); if (q2 == std::string::npos) return;
        size_t len = std::min(q2 - q1 - 1, maxlen - 1);
        memcpy(dst, json.c_str() + q1 + 1, len); dst[len] = '\0';
    };
    auto get_int = [&](const char * key, int def) -> int {
        std::string s = std::string("\"") + key + "\"";
        size_t p = json.find(s); if (p == std::string::npos) return def;
        p = json.find(':', p); if (p == std::string::npos) return def;
        return atoi(json.c_str() + p + 1);
    };
    auto get_float = [&](const char * key, float def) -> float {
        std::string s = std::string("\"") + key + "\"";
        size_t p = json.find(s); if (p == std::string::npos) return def;
        p = json.find(':', p); if (p == std::string::npos) return def;
        return (float)atof(json.c_str() + p + 1);
    };

    // EngineConfig in types.h uses std::string, monolith uses char arrays.
    // Bridge: parse into temp char arrays, then assign to ec's std::string fields.
    char tmp_model_path[512] = {};
    char tmp_character_json[256] = {};
    get_str("model_path", tmp_model_path, sizeof(tmp_model_path));
    get_str("character_json", tmp_character_json, sizeof(tmp_character_json));
    ec.model_path = tmp_model_path;
    ec.character_json = tmp_character_json;
    ec.threads = get_int("threads", ec.threads);
    ec.gpu = get_int("gpu", ec.gpu) != 0;
    ec.max_tokens = get_int("max_tokens", ec.max_tokens);
    ec.max_ctx = get_int("max_ctx", ec.max_ctx);
    ec.temp = get_float("temp", ec.temp);
    ec.top_k = get_int("top_k", ec.top_k);
    ec.top_p = get_float("top_p", ec.top_p);
    ec.rep_penalty = get_float("rep_penalty", ec.rep_penalty);
    ec.color = get_int("color", ec.color ? 1 : 0) != 0;
    ec.verbose = get_int("verbose", ec.verbose ? 1 : 0) != 0;
    return true;
}

bool save_engine_config(const EngineConfig & ec, const char * path) {
    FILE * f = fopen(path, "w"); if (!f) return false;
    fprintf(f, "{\n");
    fprintf(f, "    \"model_path\": \"%s\",\n", ec.model_path.c_str());
    fprintf(f, "    \"character_json\": \"%s\",\n", ec.character_json.c_str());
    fprintf(f, "    \"threads\": %d,\n", ec.threads);
    fprintf(f, "    \"gpu\": %d,\n", ec.gpu ? 1 : 0);
    fprintf(f, "    \"max_tokens\": %d,\n", ec.max_tokens);
    fprintf(f, "    \"max_ctx\": %d,\n", ec.max_ctx);
    fprintf(f, "    \"temp\": %.2f,\n", ec.temp);
    fprintf(f, "    \"top_k\": %d,\n", ec.top_k);
    fprintf(f, "    \"top_p\": %.2f,\n", ec.top_p);
    fprintf(f, "    \"rep_penalty\": %.2f,\n", ec.rep_penalty);
    fprintf(f, "    \"color\": %d,\n", ec.color ? 1 : 0);
    fprintf(f, "    \"verbose\": %d\n", ec.verbose ? 1 : 0);
    fprintf(f, "}\n"); fclose(f); return true;
}

// handle_chat_command is defined in chat.cpp
