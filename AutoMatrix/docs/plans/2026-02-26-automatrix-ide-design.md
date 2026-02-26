# AutoMatrix v1.0 — Production-Ready Visual ML IDE Design

**Date**: 2026-02-26
**Status**: Approved
**Author**: Claude + User collaborative brainstorm

---

## 1. Vision

AutoMatrix is an omnipotent AI-ML visual IDE for GGUF-based VLM inference on Android. Single-user power tool. Three workflows:

1. **Load → Edit → Deploy → Test**: Load GGUF, visually edit compute graph, deploy modified config to device, run inference, iterate.
2. **Design → Build → Test**: Design new architectures from scratch using visual blocks (Unreal Blueprint style), compile to config, build binary, deploy and test.
3. **Inspect → Optimize → Compare**: Load model, inspect tensors/layers, try quant strategies visually, run A/B tests on device, compare tok/s and quality.

**UI Paradigm**: Unreal Blueprint — structured node graph with typed pins, execution flow, data flow.
**Layout**: IntelliJ IDEA — one screen, dockable tool windows, document tabs, blueprint canvas center.
**Icons**: Tabler Icons (`@tabler/icons-svelte`)

---

## 2. Architecture: Blueprint Node Engine

### 2.1 Node Model

```typescript
interface Node {
  id: string;
  type: string;           // "op", "input", "output", "group", "comment"
  label: string;
  category: string;       // "attn", "ffn", "norm", "embed", "head", "math", "custom"
  position: { x: number; y: number };
  size: { w: number; h: number };  // auto-sized from pins
  collapsed: boolean;
  enabled: boolean;
  color: string;          // accent color by category
  metadata: Record<string, unknown>;
  pins: Pin[];
}

interface Pin {
  id: string;
  name: string;
  direction: "in" | "out";
  dataType: PinType;
  shape?: number[];       // tensor dimensions
  dtype?: string;         // "f16", "q8_0", "q5_0"
  connected: boolean;
  value?: unknown;        // for scalars/config: default value
}

interface Edge {
  id: string;
  from: { nodeId: string; pinId: string };
  to: { nodeId: string; pinId: string };
  valid: boolean;         // type-checked
}
```

### 2.2 Pin Types (12 types)

| Pin Type | Color | Description |
|----------|-------|-------------|
| **tensor** | `#e8963a` orange | N-D tensor with shape + dtype (weights, activations) |
| **image** | `#4fc3f7` cyan | Image data: pixels, patches, visual features |
| **tokens** | `#81c784` green | Token sequences: IDs, masks, position encodings |
| **kv_cache** | `#ffb74d` amber | KV cache state with layer/head/seq metadata |
| **scalar** | `#64b5f6` blue | Single values: temperature, top_k, learning rate |
| **math** | `#ba68c8` purple | Math operations: softmax, silu, gelu, rmsnorm, matmul |
| **config** | `#a1887f` brown | Structured config: arch params, quant settings |
| **flow** | `#ffffff` white | Execution flow: order, branching, loops (Unreal-style) |
| **memory** | `#ef5350` red | Memory regions: UMA pool, cache lines, allocations |
| **device** | `#ffd54f` yellow | Hardware: cores, threads, NEON/I8MM units, GPU CUs |
| **stream** | `#4db6ac` teal | Streaming data: SSE output, inference tokens, logs |
| **binary** | `#78909c` steel | Raw binary data: GGUF chunks, AMXP payloads, buffers |

Note: Mobile devices use UMA (Unified Memory Architecture) — CPU and GPU share same physical memory. No separate VRAM.

### 2.3 Interactions

| Action | Input |
|--------|-------|
| Pan | Middle-click drag or Space+left-drag |
| Zoom | Scroll wheel (toward cursor) |
| Select | Left-click node. Shift+click multi-select. Drag-box area select. |
| Move | Left-drag selected node(s) |
| Connect | Left-drag from pin → snap to compatible pin (type-checked) |
| Disconnect | Right-click edge → delete |
| Context Menu | Right-click canvas → add node. Right-click node → duplicate/delete/collapse/group. |
| Delete | Del key |
| Duplicate | Ctrl+D |
| Undo/Redo | Ctrl+Z / Ctrl+Y |
| Group | Ctrl+G |
| Fit View | F |
| Search | Ctrl+F |

### 2.4 Engine Features

- Undo/redo stack (100 levels)
- Copy/paste nodes (with edge reconnection)
- Snap-to-grid (toggleable)
- Auto-layout (hierarchical left-to-right)
- Validation: red pins when types don't match, red edges when shapes incompatible
- Minimap in bottom-right corner
- Node search/filter overlay

### 2.5 Node Theme Colors

| Category | Dark BG | Dark Accent | Light BG | Light Accent |
|----------|---------|-------------|----------|--------------|
| embed | `#1a3a4d` | `#6ec6ff` | `#e3f2fd` | `#1976d2` |
| attn | `#3d2a14` | `#e8963a` | `#fff3e0` | `#e65100` |
| ffn | `#3a1f3d` | `#ce93d8` | `#f3e5f5` | `#7b1fa2` |
| norm | `#1f3a25` | `#a5d6a7` | `#e8f5e9` | `#2e7d32` |
| head | `#3d1a1a` | `#ef5350` | `#ffebee` | `#c62828` |
| math | `#2a1f3d` | `#ba68c8` | `#ede7f6` | `#6a1b9a` |
| device | `#3d3a1a` | `#ffd54f` | `#fffde7` | `#f57f17` |
| memory | `#3d1a1a` | `#ef5350` | `#ffebee` | `#b71c1c` |

Each node: rounded rect with category bg, 4px left accent bar, theme-aware text color.

---

## 3. Layout: IntelliJ-Style Unified Workspace

```
┌──────────────────────────────────────────────────────────────────┐
│ A│AUTOMATRIX  [▶ Run] [⏹ Stop] [📤 Deploy] [🔄 Rebuild]  ⚙ 🌓 │  ← Toolbar
├──┬───────────────────────────────────────────────────────────┬───┤
│  │ model.gguf ✕ │ llama-arch.amxp ✕ │ surgery-1 ✕ │  +    │   │  ← Doc tabs
│L │──────────────────────────────────────────────────────────│ R │
│E │                                                          │ I │
│F │              BLUEPRINT CANVAS                            │ G │
│T │              (Node Graph / Document)                     │ H │
│  │                                                          │ T │
│T │              [nodes + wires + pins]                      │   │
│O │                                                          │ T │
│O │                                                          │ O │
│L │                                                          │ O │
│S │                                                          │ L │
│  │                                                          │ S │
├──┼──────────────────────────────────────────────────────────┼───┤
│  │ > Console  │ > Devices │ > Benchmarks │ > Terminal       │   │  ← Bottom
├──┴──────────────────────────────────────────────────────────┴───┤
│ ◉ A059  │ ◆ SmolVLM-500M q5_0 268MB │ CPU 4t NEON+I8MM │ v0.3│  ← Status
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 Tool Windows

**Left sidebar** (click icon to open/close):

| Icon | Tool Window | Content |
|------|-------------|---------|
| `IconFolder` | Model Inspector | Model tree: tensors, layers, config, metadata |
| `IconComponents` | Node Palette | Draggable nodes by category |
| `IconPuzzle` | Plugins | AMXP browser, double-click → opens as doc tab |
| `IconDeviceMobile` | Devices | ADB device list + properties |
| `IconTestPipe` | Test Cases | Test case list, run/compare |

**Right sidebar**:

| Icon | Tool Window | Content |
|------|-------------|---------|
| `IconAdjustments` | Properties | Selected node inspector, editable pins |
| `IconChartBar` | Benchmarks | Charts, history, A/B compare |
| `IconScissors` | Surgery Tools | Prune/merge/requant/export tools |

**Bottom panels** (click to expand):

| Icon | Tool Window | Content |
|------|-------------|---------|
| `IconTerminal2` | Console | Log output |
| `IconDeviceMobile` | Device Output | ADB shell output, push logs |
| `IconChartLine` | Benchmarks | Live inference output, timing charts |
| `IconCode` | Terminal | Raw command I/O |

### 3.2 Document Tabs

Each "document" opens as a tab with its own canvas:

| Document Type | Source | Canvas Content |
|--------------|--------|----------------|
| Model Graph | Load GGUF | Compute graph (ops + tensor flow) |
| Architecture Plugin | Open .amxp | Arch template (transformer blocks, configs) |
| Backend Plugin | Open .amxp | Hardware pipeline (cores, dispatch, memory) |
| Quant Plugin | Open .amxp | Bit allocation + benchmarks |
| Sampling Plugin | Open .amxp | Logit processing pipeline |
| Surgery Session | New surgery | Real model + surgery tools |
| Benchmark Report | After test run | Comparison charts + node annotations |
| Test Case | New/open test | Config form + results + charts |

### 3.3 Toolbar

| Button | Icon | Action |
|--------|------|--------|
| Run | `IconPlayerPlay` | Deploy current config → run inference on device |
| Stop | `IconPlayerStop` | Kill running inference |
| Deploy | `IconUpload` | Push binary + model files to device |
| Rebuild | `IconRefresh` | Generate C++ → cross-compile → push |
| Settings | `IconSettings` | Open settings panel (theme, zoom, paths) |
| Theme | `IconSun`/`IconMoon` | Toggle dark/light |

---

## 4. Node Types Per Context

### 4.1 Model Graph (loaded from GGUF)

| Node | Category | Pins In | Pins Out |
|------|----------|---------|----------|
| Token Embed | embed | tokens→ | →tensor[embd, seq] |
| Pos Embed | embed | tokens→ | →tensor[embd, seq] |
| Attention | attn | tensor→, kv_cache→ | →tensor, →kv_cache |
| FFN | ffn | tensor→ | →tensor |
| RMS Norm | norm | tensor→, scalar(eps)→ | →tensor |
| Layer Norm | norm | tensor→, scalar(eps)→ | →tensor |
| LM Head | head | tensor→ | →tokens(logits) |
| Residual Add | math | tensor→, tensor→ | →tensor |
| MatMul | math | tensor→, tensor→ | →tensor |
| SiLU | math | tensor→ | →tensor |
| GELU | math | tensor→ | →tensor |
| Softmax | math | tensor→ | →tensor |
| RoPE | math | tensor→, config→ | →tensor |

### 4.2 Architecture Plugin

| Node | Description |
|------|-------------|
| Transformer Block | Group: Attn + FFN + Norms. Expandable/collapsible. |
| Vision Encoder | SigLIP/CLIP block with patch embed + layers. |
| Projector | Pixel shuffle + FC for vision→LLM projection |
| Attention Config | heads, kv_heads, head_dim, GQA/MHA/MQA |
| FFN Config | intermediate_size, activation, gate_up fused |
| Norm Config | rmsnorm/layernorm, eps |
| Fusion | QKV fused, gate+up fused markers |

### 4.3 Backend Plugin

| Node | Description |
|------|-------------|
| CPU Core | Physical core: freq, ISA (NEON, I8MM, dotprod) |
| GPU CU | Compute unit: shader cores, clock, BW |
| Memory Pool | UMA pool: size, bandwidth, latency (shared CPU+GPU) |
| Dispatch | Route ops to CPU/GPU based on rules |
| Thread Pool | OMP config: count, affinity, places |
| Cache Line | L1/L2/L3 cache modeling |

### 4.4 Quant Plugin

| Node | Description |
|------|-------------|
| Bit Allocator | Visual bar showing bits per weight, slider 2-8 |
| Block Config | Block size (32/64/128/256), super-block grouping |
| Dequant Kernel | vec_dot impl: I8MM, NEON, scalar |
| Quality Meter | Color gradient red→green based on quality estimate |
| Benchmark | Connected to device: tok/s, BW, memory |
| Comparison | Side-by-side two configs with diff highlighting |

### 4.5 Sampling Plugin

| Node | Description |
|------|-------------|
| Logit Input | Raw logits from LM head |
| Temperature | Scalar divisor, slider 0.0-2.0 |
| Top-K Filter | Keep top K, slider 1-100 |
| Top-P Filter | Nucleus sampling, slider 0.0-1.0 |
| Min-P Filter | Min probability threshold, slider 0.0-0.5 |
| Repetition Penalty | Penalty factor, slider 1.0-2.0 |
| Sampler | Selection: argmax, multinomial, mirostat |
| Token Output | Selected token ID |

### 4.6 Surgery (Model Graph + Surgery Tools)

| Tool Node | Description |
|-----------|-------------|
| Layer Delete | Remove layer, reconnect skip connections |
| Layer Duplicate | Clone layer with weights |
| Head Prune | Visual head grid, toggle heads on/off |
| Weight Requant | Requantize specific tensor to target type |
| Tensor Inspector | Histogram, min/max/mean/std, outlier count |
| LoRA Patch | Apply rank-decomposed delta to weights |
| Merge Point | Two model inputs → frankenmerge |
| Vocab Prune | Remove unused tokens from vocab |
| Export | Save modified model as new GGUF |

---

## 5. Simulation Pipeline

### 5.1 Quick Mode (config flags → vlm-test)

For changes that don't require recompilation:

1. Canvas analyzes graph → generates run config (JSON)
2. Config includes: enabled layers, per-layer quant, threads, GPU, sampling, paths
3. Backend translates config → vlm-test CLI flags
4. ADB push config to device
5. ADB shell: `./vlm-test --config /data/local/tmp/run-config.json`
6. SSE stream stdout → Console panel
7. Parse timing stats → Benchmarks panel → update node annotations

**Requires**: New `--config` flag in vlm-test.cpp (reads JSON config file).

### 5.2 Full Mode (codegen + compile)

For structural changes (new layer types, custom ops):

1. Canvas → generates graph.cpp code
2. Backend invokes CMake cross-compile for ARM
3. ADB push new binary + libraries
4. Execute on device
5. Stream results back

### 5.3 Result Visualization

After inference:
- **Node badges**: time (ms), memory (MB), throughput (GFLOPS) per node
- **Color coding**: green=fast, yellow=moderate, red=bottleneck
- **Benchmarks panel**: timeline chart, memory waterfall (UMA), tok/s history
- **A/B comparison**: select two runs, diff metrics and output

---

## 6. Test Case System

### 6.1 Test Case Model

```typescript
interface TestCase {
  id: string;
  name: string;
  created: number;

  config: {
    model: string;        // GGUF path on device
    mmproj?: string;
    image?: string;
    prompt: string;
    threads: number;
    quant: string;
    maxTokens: number;
    gpu: boolean;
    sampling: string;     // plugin ID
    customFlags: string[];
  };

  expected: {
    minTokS?: number;
    maxPrefillMs?: number;
    maxVisionMs?: number;
    maxMemoryMB?: number;
    outputContains?: string[];
    outputNotContains?: string[];
  };

  runs: TestRun[];
}

interface TestRun {
  timestamp: number;
  status: "pass" | "fail" | "error";
  metrics: {
    visionMs: number;
    prefillMs: number;
    decodeMs: number;
    tokS: number;
    memoryMB: number;
  };
  output: string;
  failReasons: string[];
}
```

### 6.2 Test Case UI

**Tool Window (left sidebar)**:
- List of saved test cases with pass/fail badges
- Create / Run Single / Run All / Run Selected
- Filter: passing, failing, model, quant

**Test Case Editor (document tab)**:
- Config section: form fields (model picker, quant dropdown, thread slider)
- Expectations section: threshold sliders for metrics + text matchers
- Results section: run history table with status badges
- Chart: tok/s over time (regression detection)
- Diff view: compare any two runs

**Test Suite Runner**:
- Sequential execution on device
- Progress bar (N/M)
- Summary: X pass, Y fail, Z error
- Export as JSON/CSV

### 6.3 Storage

Test cases saved as AMXP binary (type=4 `AMXP_TEST`) in `plugins/tests/`.

### 6.4 Backend

- `POST /api/tests/run` — run single test case
- `POST /api/tests/suite` — run all, SSE stream progress
- `GET /api/tests` — list all test cases
- `POST /api/tests/save` — save test case

---

## 7. Bug Fixes

### Black Nodes
Nodes use category-specific background colors (Section 2.5) instead of `var(--bg-panel)`.

### TopBar in Editor Mode
Eliminated by new design — no separate "editor mode". Toolbar replaces TopBar.

### JSON Control Characters
Already fixed: `json_escape_str` handles all control chars.

---

## 8. Backend Additions Required

### New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/surgery/prune | Remove layers/heads from model |
| POST | /api/surgery/requant | Requantize specific tensors |
| POST | /api/surgery/merge | Merge two GGUF files |
| POST | /api/surgery/export | Write modified GGUF |
| GET | /api/model/tensor/:name | Get tensor values for inspector |
| POST | /api/tests/run | Run single test case |
| POST | /api/tests/suite | Run all tests (SSE) |
| GET | /api/tests | List test cases |
| POST | /api/tests/save | Save test case |
| GET | /api/inference/stream | SSE streaming inference output |

### vlm-test Additions

- `--config PATH` flag: read JSON config file instead of individual flags
- JSON config format matching TestCase.config schema

---

## 9. Implementation Priority

### Phase 1: Core Engine (Must Have)
1. NodeCanvas component (SVG, pan/zoom/drag/connect/pins)
2. IntelliJ layout (tool windows, document tabs, toolbar)
3. Bug fixes (node colors, theme)
4. Model Graph document type
5. Quick-mode simulation (config → vlm-test → results)

### Phase 2: Plugin Editors
6. Architecture plugin node editor
7. Quant plugin node editor
8. Sampling plugin node editor
9. Backend plugin node editor

### Phase 3: Surgery + Testing
10. Surgery document type with tools
11. Test case system
12. Benchmark comparison charts

### Phase 4: Full Pipeline
13. Full-mode codegen + compile
14. Tensor value inspector
15. Model export (modified GGUF)
16. A/B benchmark comparison
