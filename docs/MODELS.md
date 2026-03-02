# Supported Models

This fork preserves **all compute graphs** from upstream llama.cpp. Any model in GGUF format works.

---

## Architectures

100+ model architectures supported, including:

| Family | Models |
|--------|--------|
| **LLaMA** | LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.2, Code Llama |
| **Qwen** | Qwen, Qwen 1.5, Qwen 2, Qwen 2.5, Qwen 3, QwQ |
| **Mistral** | Mistral 7B, Mixtral 8x7B, Mistral Small/Medium |
| **Phi** | Phi-2, Phi-3, Phi-3.5, Phi-4 |
| **Gemma** | Gemma, Gemma 2, Gemma 3 |
| **DeepSeek** | DeepSeek, DeepSeek-V2, DeepSeek-V3 |
| **Command** | Command-R, Command-R+ |
| **StarCoder** | StarCoder, StarCoder2 |
| **GPT** | GPT-2, GPT-J, GPT-NeoX |
| **Falcon** | Falcon 7B/40B/180B |
| **RWKV** | RWKV v5, RWKV v6 |
| **Mamba** | Mamba, Mamba2 |
| **LFM** | LFM2-350M, LFM2-1.2B |
| **Others** | InternLM, Yi, Baichuan, ChatGLM, BLOOM, MPT, OLMo, Jais, ... |

### Vision Language Models (VLM)

VLM support requires a text model GGUF + a vision projector (mmproj) GGUF.

| Architecture | Models | Features |
|-------------|--------|----------|
| **LLaVA** | LLaVA-1.5, LLaVA-1.6, BakLLaVA | Standard CLIP encoder |
| **SigLIP** | Gemma3-Vision | SigLIP encoder |
| **Qwen2-VL** | Qwen2-VL-2B/7B/72B | M-RoPE positional encoding |
| **Qwen3-VL** | Qwen3-VL | M-RoPE positional encoding |
| **Pixtral** | Pixtral, Mistral-Vision | Pixel-level attention |
| **MiniCPM-V** | MiniCPM-V, MiniCPM-V 2.6 | Efficient vision |
| **InternVL** | InternVL2, InternVL2.5 | High-res vision |
| **CogVLM** | CogVLM, CogVLM2 | Visual grounding |
| **SmolVLM** | SmolVLM-500M, SmolVLM-2.2B | Lightweight, mobile-friendly |
| **GLM4V** | GLM-4V | Multi-image support |
| **Llama4** | Llama-4-Scout/Maverick | Meta's VLM |
| **MobileNetV5** | Gemma3n-Vision | Mobile-optimized |
| **Kimi-VL** | Kimi-VL, Kimi-K2.5-VL | Long-context vision |
| **Whisper** | Whisper (audio) | Audio encoder |
| **Conformer** | Conformer (audio) | Audio encoder |

---

## Quantization Formats

All GGUF quantization types are supported:

### Standard Quantization

| Type | Bits/Weight | Description |
|------|-------------|-------------|
| `Q4_0` | 4.5 | Basic 4-bit, fast |
| `Q4_1` | 5.0 | 4-bit with non-zero offset |
| `Q5_0` | 5.5 | 5-bit quantization |
| `Q5_1` | 6.0 | 5-bit with non-zero offset |
| `Q8_0` | 8.5 | 8-bit, best quality/size balance |

### K-Quantization (Recommended)

| Type | Bits/Weight | Description |
|------|-------------|-------------|
| `Q2_K` | 3.35 | Smallest, some quality loss |
| `Q3_K_S` | 3.50 | Small |
| `Q3_K_M` | 3.91 | Medium |
| `Q3_K_L` | 4.27 | Large |
| `Q4_K_S` | 4.58 | Small, good balance |
| `Q4_K_M` | 4.85 | **Best general-purpose choice** |
| `Q5_K_S` | 5.54 | High quality |
| `Q5_K_M` | 5.69 | Higher quality |
| `Q6_K` | 6.56 | Near-FP16 quality |

### IQ (Importance-Weighted) Quantization

| Type | Bits/Weight | Description |
|------|-------------|-------------|
| `IQ1_S` | 1.56 | Extreme compression |
| `IQ1_M` | 1.75 | |
| `IQ2_XXS` | 2.06 | Very small |
| `IQ2_XS` | 2.31 | |
| `IQ2_S` | 2.50 | |
| `IQ2_M` | 2.70 | |
| `IQ3_XXS` | 3.06 | |
| `IQ3_XS` | 3.30 | Good quality for size |
| `IQ4_NL` | 4.50 | Non-linear 4-bit |
| `IQ4_XS` | 4.25 | |

### Full Precision

| Type | Bits/Weight | Description |
|------|-------------|-------------|
| `F16` | 16.0 | Half precision |
| `BF16` | 16.0 | BFloat16 |
| `F32` | 32.0 | Full precision (not recommended for mobile) |

---

## Choosing a Model for Mobile

### Size Guidelines

| Device RAM | Max Model Size | Recommended |
|------------|---------------|-------------|
| 4 GB | ~1-2 GB GGUF | 0.5-1B params @ Q4_K_M |
| 6 GB | ~2-3 GB GGUF | 1-3B params @ Q4_K_M |
| 8 GB | ~4-5 GB GGUF | 3-7B params @ Q4_K_M |
| 12 GB | ~6-8 GB GGUF | 7B params @ Q6_K or Q8_0 |

### Recommended Configurations

| Use Case | Model | Quant | Size | Speed |
|----------|-------|-------|------|-------|
| Fast chat | LFM2-350M | Q8_0 | ~350 MB | 29-30 t/s |
| Chat | Qwen3-0.6B | Q8_0 | ~630 MB | 17-19 t/s |
| Vision | SmolVLM-500M + mmproj | Q8_0 | ~500 MB | 28 t/s |
| General | Qwen 2.5-1.5B | Q4_K_M | ~1.0 GB | 6-10 t/s |
| Quality | Gemma3-4B | Q4_K_M | ~2.5 GB | 3-5 t/s |
| Code | Qwen 2.5-Coder-1.5B | Q4_K_M | ~1.0 GB | 6-10 t/s |

Speed estimates are for Cortex-X3 class devices.

---

## GGUF Format

Models must be in GGUF format. Common sources:

- [Hugging Face](https://huggingface.co/models?library=gguf) — search for `gguf` in filters
- [TheBloke](https://huggingface.co/TheBloke) — pre-quantized GGUF models
- [bartowski](https://huggingface.co/bartowski) — quantized models

Models in other formats (PyTorch, SafeTensors, ONNX) must be converted to GGUF first using `llama.cpp`'s convert scripts (not included in this fork).

---

## Loading Models

### From file path
```c
ggml_engine_load_model(engine, "/data/local/tmp/model.gguf");
```

### From file descriptor (Android SAF)
```c
// Get fd from Android content resolver
int fd = open_from_content_uri(uri);
ggml_engine_load_model_from_fd(engine, fd);
```

The file descriptor path supports Android's Storage Access Framework, allowing users to select models from any storage provider without requiring direct file path access.

### Loading VLM Models

VLM models require two GGUF files: the text model and the vision projector (mmproj).

```c
// 1. Load text model first
ggml_engine_load_model(engine, "/data/local/tmp/smolvlm-500m.gguf");

// 2. Load vision projector
ggml_engine_vlm_params vlm_params = ggml_engine_vlm_default_params();
ggml_engine_vlm_t * vlm = ggml_engine_vlm_load(
    engine, "/data/local/tmp/mmproj-smolvlm-500m.gguf", vlm_params);

// 3. From file descriptor (Android SAF)
ggml_engine_vlm_t * vlm = ggml_engine_vlm_load_from_fd(engine, fd, vlm_params);
```

Look for `mmproj-*.gguf` files on Hugging Face alongside the text model GGUF.
