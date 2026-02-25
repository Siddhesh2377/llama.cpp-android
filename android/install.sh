#!/bin/bash
# gguf-engine installer for Android (aarch64)
# Downloads the standalone executable, character config, and optionally a model.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/Siddhesh2377/llama.cpp-android/character-engine-v1/android/install.sh | bash
#   or: bash install.sh
#
# Requirements: curl, adb (for device push), or run directly on Android via Termux
#
# Layout:
#   ./gguf-engine-cli          (binary)
#   ./lib*.so                  (shared libraries)
#   ./.config/config.json      (runtime config)
#   ./.config/aria.json        (character personality)
#   ./.config/models/model.gguf (downloaded model)

set -e

VERSION="1.0.0"
REPO="Siddhesh2377/llama.cpp-android"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

header() {
    echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}  ║   gguf-engine v${VERSION} installer  ║${NC}"
    echo -e "${BOLD}${CYAN}  ╚══════════════════════════════════════╝${NC}\n"
}

info() { echo -e "${DIM}  $1${NC}"; }
success() { echo -e "${GREEN}  ✓ $1${NC}"; }
error() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }
warn() { echo -e "${YELLOW}  ! $1${NC}"; }

header

# Detect environment
if [ -d "/data/local/tmp" ] && [ "$(uname -m)" = "aarch64" ]; then
    MODE="direct"
    INSTALL_DIR="/data/local/tmp"
    info "Detected: Running on Android (aarch64)"
elif command -v adb &>/dev/null; then
    MODE="adb"
    info "Detected: ADB available, will push to device"
else
    MODE="download"
    INSTALL_DIR="./gguf-engine"
    info "Detected: Desktop mode, downloading to ./gguf-engine"
fi

CONFIG_DIR="${INSTALL_DIR}/.config"
echo ""

# Create directories
if [ "$MODE" = "download" ] || [ "$MODE" = "direct" ]; then
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$CONFIG_DIR/models"
fi

# ===========================================================================
# Step 1: Download binary
# ===========================================================================
echo -e "  ${BOLD}Step 1: Download gguf-engine binary${NC}"
BINARY_URL="https://github.com/${REPO}/releases/download/v${VERSION}/gguf-engine-cli"
BINARY_PATH="${INSTALL_DIR}/gguf-engine-cli"

if [ "$MODE" = "adb" ]; then
    BINARY_PATH="/tmp/gguf-engine-cli"
fi

if [ -f "$BINARY_PATH" ]; then
    warn "Binary already exists, skipping download"
else
    info "Downloading from GitHub..."
    curl -L --progress-bar -o "$BINARY_PATH" "$BINARY_URL" 2>&1 || {
        warn "GitHub release not available, trying local build..."
        if [ -f "./gguf-engine-cli" ]; then
            cp ./gguf-engine-cli "$BINARY_PATH"
        else
            error "Binary not found. Build from source or check the release URL."
        fi
    }
fi
chmod +x "$BINARY_PATH"
success "Binary ready"

# ===========================================================================
# Step 2: Download shared libraries
# ===========================================================================
echo -e "\n  ${BOLD}Step 2: Download shared libraries${NC}"
LIBS_URL="https://github.com/${REPO}/releases/download/v${VERSION}/gguf-engine-libs-arm64.tar.gz"
LIBS_DIR="$INSTALL_DIR"
if [ "$MODE" = "adb" ]; then
    LIBS_DIR="/tmp"
fi

if ls "$LIBS_DIR"/libggml*.so 1>/dev/null 2>&1; then
    warn "Shared libraries already exist, skipping"
else
    info "Downloading shared libraries (~9.4 MB)..."
    curl -L --progress-bar -o "$LIBS_DIR/gguf-engine-libs-arm64.tar.gz" "$LIBS_URL" 2>&1 || {
        warn "Library download failed. You may need to build from source."
    }
    if [ -f "$LIBS_DIR/gguf-engine-libs-arm64.tar.gz" ]; then
        tar xzf "$LIBS_DIR/gguf-engine-libs-arm64.tar.gz" -C "$LIBS_DIR"
        rm -f "$LIBS_DIR/gguf-engine-libs-arm64.tar.gz"
        success "Shared libraries extracted"
    fi
fi

# ===========================================================================
# Step 3: Download character config → .config/
# ===========================================================================
echo -e "\n  ${BOLD}Step 3: Download character config${NC}"
CHARACTER_URL="https://raw.githubusercontent.com/${REPO}/character-engine-v1/android/aria.json"

if [ "$MODE" = "adb" ]; then
    CHARACTER_PATH="/tmp/aria.json"
else
    CHARACTER_PATH="${CONFIG_DIR}/aria.json"
fi

if [ -f "$CHARACTER_PATH" ]; then
    warn "aria.json already exists, skipping"
else
    curl -sL -o "$CHARACTER_PATH" "$CHARACTER_URL" 2>/dev/null || {
        # Create default aria.json
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
success "Character config ready → .config/aria.json"

# ===========================================================================
# Step 4: Download model → .config/models/
# ===========================================================================
echo -e "\n  ${BOLD}Step 4: Download model (optional)${NC}"
echo ""
echo -e "  ${CYAN}Available models:${NC}"
echo "    1) Qwen3-0.6B-Q8_0    (660 MB) — Best quality for 0.6B"
echo "    2) Qwen3-0.6B-Q4_K_M  (430 MB) — Good balance"
echo "    3) Skip model download"
echo ""

read -p "  Select [1/2/3]: " model_choice

if [ "$MODE" = "adb" ]; then
    MODEL_PATH="/tmp/model.gguf"
else
    MODEL_PATH="${CONFIG_DIR}/models/model.gguf"
fi

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

# ===========================================================================
# Step 5: Push to device via ADB
# ===========================================================================
if [ "$MODE" = "adb" ]; then
    echo -e "\n  ${BOLD}Step 5: Push to device${NC}"
    INSTALL_DIR="/data/local/tmp"
    CONFIG_DIR="${INSTALL_DIR}/.config"

    # Create .config dirs on device
    adb shell "mkdir -p ${CONFIG_DIR}/models" 2>/dev/null

    # Push binary + libs to install dir
    adb push /tmp/gguf-engine-cli ${INSTALL_DIR}/ 2>&1 | tail -1
    for lib in /tmp/libggml*.so; do
        [ -f "$lib" ] && adb push "$lib" ${INSTALL_DIR}/ 2>&1 | tail -1
    done
    adb shell "chmod +x ${INSTALL_DIR}/gguf-engine-cli"

    # Push config to .config/
    adb push /tmp/aria.json ${CONFIG_DIR}/ 2>&1 | tail -1

    # Push model to .config/models/
    if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
        info "Pushing model to device (this may take a while)..."
        adb push "$MODEL_PATH" ${CONFIG_DIR}/models/model.gguf 2>&1 | tail -1
        MODEL_PATH="${CONFIG_DIR}/models/model.gguf"
    else
        MODEL_PATH=".config/models/model.gguf"
    fi
    success "Files pushed to device"
elif [ "$MODE" = "direct" ]; then
    chmod +x "$BINARY_PATH"
    MODEL_PATH="${MODEL_PATH:-.config/models/model.gguf}"
fi

# ===========================================================================
# Step 6: Create .config/config.json
# ===========================================================================
echo -e "\n  ${BOLD}Step 6: Create config${NC}"

CONFIG_JSON=$(cat << CONFEOF
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
)

if [ "$MODE" = "adb" ]; then
    echo "$CONFIG_JSON" | adb shell "cat > ${CONFIG_DIR}/config.json"
else
    echo "$CONFIG_JSON" > "${CONFIG_DIR}/config.json"
fi
success "Config created → .config/config.json"

# ===========================================================================
# Done
# ===========================================================================
echo ""
echo -e "  ${BOLD}${GREEN}Installation complete!${NC}"
echo ""
echo -e "  ${DIM}Layout:${NC}"
echo -e "  ${DIM}  ./gguf-engine-cli          binary${NC}"
echo -e "  ${DIM}  ./lib*.so                  shared libraries${NC}"
echo -e "  ${DIM}  ./.config/config.json      settings${NC}"
echo -e "  ${DIM}  ./.config/aria.json        character${NC}"
echo -e "  ${DIM}  ./.config/models/          models${NC}"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"

if [ "$MODE" = "adb" ]; then
    echo -e "    ${CYAN}adb shell${NC}"
    echo -e "    ${CYAN}cd ${INSTALL_DIR} && LD_LIBRARY_PATH=. ./gguf-engine-cli --char-chat${NC}"
elif [ "$MODE" = "direct" ]; then
    echo -e "    ${CYAN}cd ${INSTALL_DIR}${NC}"
    echo -e "    ${CYAN}LD_LIBRARY_PATH=. ./gguf-engine-cli --char-chat${NC}"
else
    echo -e "    ${CYAN}cd ${INSTALL_DIR}${NC}"
    echo -e "    ${CYAN}# Push to device, then:${NC}"
    echo -e "    ${CYAN}LD_LIBRARY_PATH=. ./gguf-engine-cli --char-chat${NC}"
fi

echo ""
echo -e "  ${DIM}All config in .config/ — model path + character auto-loaded from config.json${NC}"
echo -e "  ${DIM}Type /help in chat for available commands${NC}"
echo ""
