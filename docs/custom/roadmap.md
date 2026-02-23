# Roadmap

## Completed

### Core Intervention Surfaces
- [x] **A — Activation Capture**: Per-layer hidden state readout for probing and control vector extraction
- [x] **C — Attention Score Bias**: Log-space bias injection on KQ scores pre-softmax
- [x] **D — Head Rescaling**: Per-head scalar multiplier on attention output
- [x] **E — Attention Temperature**: Per-head softmax sharpness (sharp early layers, flat late layers)
- [x] **G — LayerNorm Affine Shift**: Post-normalization additive offset (cheapest personality mod)
- [x] **GR — Gated Residual**: Per-layer attn/FFN gate scalars [0, 2]

### Advanced Systems
- [x] **P4 — Hypernetwork FFN LoRA**: Rank-4 runtime-trainable FFN adaptation for middle layers
- [x] **P5 — Dynamic Sparse Masks**: Per-layer FFN neuron masking with random initialization
- [x] **P6 — KAN-lite Activation Overlay**: 8-knot piecewise-linear spline over FFN activations
- [x] **P7 — Forward Learning (SPSA)**: Gradient-free on-device parameter tuning

### Infrastructure
- [x] **Early Exit**: Skip layers after N blocks for speculative decoding
- [x] **State Save/Load**: Binary persistence of all learnable intervention parameters
- [x] **Per-layer flash attention check**: Only disable flash on layers that need it
- [x] **Graph reuse**: `can_reuse()` skips graph rebuild when params unchanged

---

## In Progress

### Heterogeneous NPU/GPU/CPU Scheduling
> Target: Snapdragon 7s Gen 3+ (Hexagon NPU, Adreno GPU, Kryo CPU)

The goal is to split transformer layers across all three processors on the SoC for maximum throughput:

**Architecture**:
```
NPU (Hexagon HMX)     GPU (Adreno 810)       CPU (Kryo)
  INT4/INT8 GEMM        FP16 compute           FP32 fallback
  +-----------+          +----------+           +----------+
  | Layers    |          | Layers   |           | Layers   |
  | 0..N_npu  |  ------> | N_npu..  |  -------> | N_gpu..  |
  |           |          | N_gpu    |           | N_total  |
  +-----------+          +----------+           +----------+
       |                      |                      |
       +----------------------+----------------------+
                              |
                     Unified LPDDR5 Memory
                        (25.6 GB/s)
```

**Key challenges**:
1. **Layer partitioning**: Profile each layer's compute vs memory ratio. NPU excels at INT4/INT8 GEMM (HMX accelerator), GPU handles FP16 attention, CPU does everything else.
2. **Zero-copy handoff**: All three processors share LPDDR5 via ION/dma-buf. Hidden states stay in shared memory; no copy between processors.
3. **Pipeline scheduling**: Overlap NPU layer N+1 with GPU layer N. Triple-buffer hidden states.
4. **Quantization heterogeneity**: NPU runs Q4_0/Q8_0, GPU runs FP16 attention, CPU runs FP32 intervention surfaces. Mixed precision within a single forward pass.
5. **QNN SDK integration**: Qualcomm Neural Network SDK provides the NPU runtime. Need to:
   - Compile GGML ops to QNN graph format
   - Map ggml tensors to QNN tensors (zero-copy via shared memory)
   - Handle the QNN execution context lifecycle

**Implementation plan**:
1. Abstract backend scheduler in ggml (layer-level granularity, not op-level)
2. Profile per-layer latency on each backend (build cost model)
3. Solve partition assignment: minimize total forward pass latency
4. Implement zero-copy tensor sharing between CPU↔NPU↔GPU
5. Pipeline execution with double/triple buffering

**Expected gains**: 1.5-2.5x token/s improvement on Snapdragon 7s Gen 3 vs CPU-only.

**Research refs**:
- [gguf-forward-pass-dynamic-emotion-mobile.md](../../ai_gguf/plan/gguf-forward-pass-dynamic-emotion-mobile.md)
- [snapdragon-7s-gen3-hetero-arch.html](../../ai_gguf/plan/snapdragon-7s-gen3-hetero-arch.html)

---

## Planned

### Gated Residual with Learned Gates
Currently gates are set externally. Future: learn gate values via SPSA (P7) to automatically discover which layers matter for a given persona.

### Sigmoid Attention (Research)
Replace softmax with sigmoid in attention. Papers show promising results but fundamentally changes attention distribution. Needs careful evaluation — currently marked "DO NOT IMPLEMENT" until more evidence.

### Cross-Layer Weight Sharing with Emotion Offset
Share FFN weights across groups of layers (e.g., layers 4-8 share one FFN) with small per-layer emotion offsets. Reduces memory footprint significantly for mobile.

### Mamba/RWKV Hybrid Layers
Replace some transformer layers with linear-time recurrent layers (Mamba, RWKV). Could dramatically reduce KV cache memory for long conversations. Requires architecture-level changes.

### Continuous Batching for Multi-Character
Serve multiple persona contexts simultaneously with shared KV cache prefix and per-persona intervention states. Useful for group chat scenarios.
