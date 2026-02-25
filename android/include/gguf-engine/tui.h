// tui.h — Terminal UI helpers and engine configuration

#pragma once

#include "types.h"

// ---------------------------------------------------------------------------
// TUI color helpers
// ---------------------------------------------------------------------------
namespace tui {
    inline const char * RESET   = "\033[0m";
    inline const char * BOLD    = "\033[1m";
    inline const char * DIM     = "\033[2m";
    inline const char * RED     = "\033[31m";
    inline const char * GREEN   = "\033[32m";
    inline const char * YELLOW  = "\033[33m";
    inline const char * BLUE    = "\033[34m";
    inline const char * MAGENTA = "\033[35m";
    inline const char * CYAN    = "\033[36m";
    inline const char * WHITE   = "\033[37m";

    extern bool color_enabled;

    void c(const char * color);
    void reset();
    void banner(const char * text);
    void divider();
    void error(const char * msg);
    void success(const char * msg);
    void info(const char * msg);
    void status(const char * key, const char * val);
    void status_int(const char * key, int val);
    void status_float(const char * key, float val);
    void progress_bar(const char * label, float pct, int width = 30);
} // namespace tui

// ---------------------------------------------------------------------------
// Engine config persistence
// ---------------------------------------------------------------------------
bool load_engine_config(EngineConfig & cfg, const char * path);
bool save_engine_config(const EngineConfig & cfg, const char * path);

// ---------------------------------------------------------------------------
// Chat commands
// ---------------------------------------------------------------------------
// Returns true if command was handled (user should not generate)
bool handle_chat_command(const char * input, EngineConfig & ec, SamplingParams & sp,
                         EmotionalState & mood, ModelState & state,
                         PersonalityConfig & pc, int & max_tokens);
