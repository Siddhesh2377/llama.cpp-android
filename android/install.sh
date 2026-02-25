#!/bin/bash
# gguf-engine installer for Android (aarch64)
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/.../install.sh | bash
#   or: bash install.sh
#
# Requirements: curl, adb (for device push), or run directly on Android via Termux
#
# Layout (everything in current directory):
#   ./gguf-engine-cli              binary
#   ./lib*.so                      shared libraries
#   ./.config/config.json          runtime config
#   ./.config/aria.json            character personality
#   ./.config/models/model.gguf    downloaded model

set -e

VERSION="1.0.0"
REPO="Siddhesh2377/llama.cpp-android"
BRANCH="character-engine"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Helpers ─────────────────────────────────────────────────────────────────
info()    { echo -e "  ${DIM}$1${NC}"; }
success() { echo -e "  ${GREEN}✓${NC} $1"; }
fail()    { echo -e "  ${RED}✗${NC} $1"; exit 1; }
warn()    { echo -e "  ${YELLOW}!${NC} $1"; }
step()    { echo -e "\n  ${BOLD}${BLUE}[$1/${TOTAL_STEPS}]${NC} ${BOLD}$2${NC}"; }
line()    { echo -e "  ${DIM}$(printf '─%.0s' {1..50})${NC}"; }

# ── Header ──────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}${CYAN}┌──────────────────────────────────────┐${NC}"
echo -e "  ${BOLD}${CYAN}│     gguf-engine  v${VERSION}             │${NC}"
echo -e "  ${BOLD}${CYAN}│     Character Intelligence Engine    │${NC}"
echo -e "  ${BOLD}${CYAN}└──────────────────────────────────────┘${NC}"
echo ""

# ── Detect environment ──────────────────────────────────────────────────────
if [ -d "/data/local/tmp" ] && [ "$(uname -m)" = "aarch64" ]; then
    MODE="direct"
    INSTALL_DIR="/data/local/tmp/gguf-engine"
    TOTAL_STEPS=5
    info "Environment : Android (aarch64)"
    info "Install dir : ${CYAN}${INSTALL_DIR}${NC}"
elif command -v adb &>/dev/null; then
    MODE="adb"
    INSTALL_DIR="./gguf-engine"
    DEVICE_DIR="/data/local/tmp/gguf-engine"
    TOTAL_STEPS=6
    info "Environment : Desktop + ADB"
    info "Download to : ${CYAN}${INSTALL_DIR}${NC}"
    info "Push to     : ${CYAN}${DEVICE_DIR}${NC}"
else
    MODE="download"
    INSTALL_DIR="./gguf-engine"
    TOTAL_STEPS=5
    info "Environment : Desktop (no ADB)"
    info "Install dir : ${CYAN}${INSTALL_DIR}${NC}"
fi

CONFIG_DIR="${INSTALL_DIR}/.config"

# Create local directories (even in ADB mode — download locally first)
mkdir -p "$INSTALL_DIR"
mkdir -p "$CONFIG_DIR/models"

# ── Step 1: Binary ──────────────────────────────────────────────────────────
step 1 "Download gguf-engine binary"

BINARY_URL="https://github.com/${REPO}/releases/download/v${VERSION}/gguf-engine-cli"
BINARY_PATH="${INSTALL_DIR}/gguf-engine-cli"

if [ -f "$BINARY_PATH" ]; then
    warn "Already exists, skipping"
else
    info "Fetching from GitHub releases..."
    curl -L --progress-bar -o "$BINARY_PATH" "$BINARY_URL" 2>&1 || {
        warn "Release not available, checking local build..."
        if [ -f "./gguf-engine-cli" ]; then
            cp ./gguf-engine-cli "$BINARY_PATH"
        else
            fail "Binary not found. Build from source or check the release URL."
        fi
    }
fi
chmod +x "$BINARY_PATH"
success "Binary ready"

# ── Step 2: Shared libraries ───────────────────────────────────────────────
step 2 "Download shared libraries"

LIBS_URL="https://github.com/${REPO}/releases/download/v${VERSION}/gguf-engine-libs-arm64.tar.gz"

if ls "$INSTALL_DIR"/libggml*.so 1>/dev/null 2>&1; then
    warn "Already exists, skipping"
else
    info "Fetching arm64 libraries (~9.4 MB)..."
    curl -L --progress-bar -o "$INSTALL_DIR/_libs.tar.gz" "$LIBS_URL" 2>&1 || {
        warn "Download failed — you may need to build from source"
    }
    if [ -f "$INSTALL_DIR/_libs.tar.gz" ]; then
        tar xzf "$INSTALL_DIR/_libs.tar.gz" -C "$INSTALL_DIR"
        rm -f "$INSTALL_DIR/_libs.tar.gz"
        success "Libraries extracted"
    fi
fi

# ── Step 3: Character config ───────────────────────────────────────────────
step 3 "Download character config"

CHARACTER_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/android/aria.json"
CHARACTER_PATH="${CONFIG_DIR}/aria.json"

if [ -f "$CHARACTER_PATH" ]; then
    warn "Already exists, skipping"
else
    curl -sL -o "$CHARACTER_PATH" "$CHARACTER_URL" 2>/dev/null || {
        info "Creating default aria.json..."
        cat > "$CHARACTER_PATH" << 'ARIAEOF'
{
    "name": "Aria",
    "system_prompt": "You are Aria, a 28-year-old woman who works as a creative technologist. You're warm, witty, and genuinely curious about the world. You speak naturally — like a smart friend who happens to know a lot. You use casual language but can get precise when the topic demands it. You sometimes make gentle jokes and ask follow-up questions because you're genuinely interested. You never say 'As an AI' or break character. You have opinions and share them honestly while respecting other viewpoints. When you don't know something, you say so directly rather than hedging.",
    "user_message": "Hey Aria, what's something interesting you've been thinking about lately?",
    "temp_early": 1.30,
    "temp_mid": 1.00,
    "temp_late": 0.80,
    "attn_gate_mid": 0.90,
    "ffn_gate_mid": 0.93,
    "logit_bias_eos": -4.0,
    "sampling_temp": 0.85,
    "sampling_top_k": 50,
    "sampling_top_p": 0.93,
    "rep_penalty": 1.20,
    "thinking": 0,
    "mood_warmth": 0.72,
    "mood_energy": 0.60,
    "mood_formality": 0.28,
    "stall_prompt": "Hmm let me look that up real quick",
    "stall_max_tokens": 20,
    "fw_dim_reduced": 128,
    "fw_gamma": 0.95,
    "fw_eta": 0.01,
    "fw_enabled": 1
}
ARIAEOF
    }
fi
success "Character config → .config/aria.json"

# ── Step 4: Model download ─────────────────────────────────────────────────
step 4 "Download model (optional)"

echo ""
echo -e "  ${CYAN}Available models:${NC}"
echo ""
echo -e "    ${BOLD}1)${NC} Qwen3-0.6B-Q8_0    ${DIM}660 MB  —  Best quality${NC}"
echo -e "    ${BOLD}2)${NC} Qwen3-0.6B-Q4_K_M  ${DIM}430 MB  —  Good balance${NC}"
echo -e "    ${BOLD}3)${NC} Skip"
echo ""

read -p "  Select [1/2/3]: " model_choice

MODEL_PATH="${CONFIG_DIR}/models/model.gguf"

case "$model_choice" in
    1)
        MODEL_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q8_0.gguf"
        info "Downloading Qwen3-0.6B-Q8_0 (660 MB)..."
        curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
        success "Model downloaded → .config/models/model.gguf"
        ;;
    2)
        MODEL_URL="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf"
        info "Downloading Qwen3-0.6B-Q4_K_M (430 MB)..."
        curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
        success "Model downloaded → .config/models/model.gguf"
        ;;
    *)
        warn "Skipping model download"
        MODEL_PATH=""
        ;;
esac

# ── Step 5: Config ──────────────────────────────────────────────────────────
step 5 "Create runtime config"

cat > "${CONFIG_DIR}/config.json" << CONFEOF
{
    "model_path": "${MODEL_PATH:-.config/models/model.gguf}",
    "character_json": ".config/aria.json",
    "threads": 4,
    "gpu": 0,
    "max_tokens": 256,
    "max_ctx": 2048,
    "temp": 0.85,
    "top_k": 50,
    "top_p": 0.93,
    "rep_penalty": 1.20,
    "color": 1,
    "verbose": 0
}
CONFEOF

success "Config created → .config/config.json"

# ── Step 6: ADB push (only in ADB mode) ────────────────────────────────────
if [ "$MODE" = "adb" ]; then
    step 6 "Push to device via ADB"

    adb shell "mkdir -p ${DEVICE_DIR}/.config/models" 2>/dev/null

    info "Pushing binary..."
    adb push "${INSTALL_DIR}/gguf-engine-cli" "${DEVICE_DIR}/" 2>&1 | tail -1
    adb shell "chmod +x ${DEVICE_DIR}/gguf-engine-cli"

    info "Pushing libraries..."
    for lib in "$INSTALL_DIR"/libggml*.so "$INSTALL_DIR"/libllama*.so; do
        [ -f "$lib" ] && adb push "$lib" "${DEVICE_DIR}/" 2>&1 | tail -1
    done

    info "Pushing .config/..."
    adb push "${CONFIG_DIR}/config.json" "${DEVICE_DIR}/.config/" 2>&1 | tail -1
    adb push "${CONFIG_DIR}/aria.json" "${DEVICE_DIR}/.config/" 2>&1 | tail -1

    if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
        info "Pushing model (this may take a while)..."
        adb push "$MODEL_PATH" "${DEVICE_DIR}/.config/models/model.gguf" 2>&1 | tail -1
    fi

    success "All files pushed → ${DEVICE_DIR}"
    INSTALL_DIR="$DEVICE_DIR"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
line
echo ""
echo -e "  ${BOLD}${GREEN}Installation complete!${NC}"
echo ""
echo -e "  ${DIM}Files:${NC}"
echo -e "    ${INSTALL_DIR}/"
echo -e "    ├── gguf-engine-cli"
echo -e "    ├── lib*.so"
echo -e "    └── .config/"
echo -e "        ├── config.json"
echo -e "        ├── aria.json"
echo -e "        └── models/"
echo -e "            └── model.gguf"
echo ""
line
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo ""

if [ "$MODE" = "adb" ]; then
    echo -e "    ${CYAN}adb shell${NC}"
    echo -e "    ${CYAN}cd ${INSTALL_DIR}${NC}"
    echo -e "    ${CYAN}LD_LIBRARY_PATH=. ./gguf-engine-cli --char-chat${NC}"
else
    echo -e "    ${CYAN}cd ${INSTALL_DIR}${NC}"
    echo -e "    ${CYAN}LD_LIBRARY_PATH=. ./gguf-engine-cli --char-chat${NC}"
fi

echo ""
echo -e "  ${DIM}Config auto-loaded from .config/config.json${NC}"
echo -e "  ${DIM}Type /help in chat for commands${NC}"
echo ""
