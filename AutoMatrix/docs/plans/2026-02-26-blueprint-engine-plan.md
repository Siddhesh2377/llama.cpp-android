# AutoMatrix v1.0 Blueprint Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform AutoMatrix from a tab-based inspector into an IntelliJ-style unified visual IDE with a Blueprint node-graph engine at its core.

**Architecture:** Single-screen workspace with dockable tool windows (left/right/bottom), document tabs for multi-canvas editing, and a reusable SVG-based Blueprint node engine with 12 typed pins. Everything — model graphs, plugin editors, surgery, benchmarks — renders on the same canvas component.

**Tech Stack:** SvelteKit 5 (runes), TypeScript, SVG for node rendering, Tabler Icons, cpp-httplib backend, AMXP binary plugins, ADB bridge.

---

## Phase 1: Core Engine + Layout Overhaul

### Task 1: Blueprint Node Engine — Data Model + Store

**Files:**
- Create: `web/src/lib/engine/types.ts`
- Create: `web/src/lib/engine/store.ts`

**Step 1: Create the node engine type system**

Create `web/src/lib/engine/types.ts` with all types for the Blueprint engine:

```typescript
// Pin data types — the 12 typed pins
export type PinType =
  | 'tensor' | 'image' | 'tokens' | 'kv_cache'
  | 'scalar' | 'math' | 'config' | 'flow'
  | 'memory' | 'device' | 'stream' | 'binary';

export const PIN_COLORS: Record<PinType, string> = {
  tensor:   '#e8963a',
  image:    '#4fc3f7',
  tokens:   '#81c784',
  kv_cache: '#ffb74d',
  scalar:   '#64b5f6',
  math:     '#ba68c8',
  config:   '#a1887f',
  flow:     '#ffffff',
  memory:   '#ef5350',
  device:   '#ffd54f',
  stream:   '#4db6ac',
  binary:   '#78909c',
};

// Node categories and their theme-aware colors
export type NodeCategory =
  | 'embed' | 'attn' | 'ffn' | 'norm' | 'head'
  | 'math' | 'device' | 'memory' | 'custom';

export interface CategoryColors {
  darkBg: string;
  darkAccent: string;
  lightBg: string;
  lightAccent: string;
}

export const CATEGORY_COLORS: Record<NodeCategory, CategoryColors> = {
  embed:  { darkBg: '#1a3a4d', darkAccent: '#6ec6ff', lightBg: '#e3f2fd', lightAccent: '#1976d2' },
  attn:   { darkBg: '#3d2a14', darkAccent: '#e8963a', lightBg: '#fff3e0', lightAccent: '#e65100' },
  ffn:    { darkBg: '#3a1f3d', darkAccent: '#ce93d8', lightBg: '#f3e5f5', lightAccent: '#7b1fa2' },
  norm:   { darkBg: '#1f3a25', darkAccent: '#a5d6a7', lightBg: '#e8f5e9', lightAccent: '#2e7d32' },
  head:   { darkBg: '#3d1a1a', darkAccent: '#ef5350', lightBg: '#ffebee', lightAccent: '#c62828' },
  math:   { darkBg: '#2a1f3d', darkAccent: '#ba68c8', lightBg: '#ede7f6', lightAccent: '#6a1b9a' },
  device: { darkBg: '#3d3a1a', darkAccent: '#ffd54f', lightBg: '#fffde7', lightAccent: '#f57f17' },
  memory: { darkBg: '#3d1a1a', darkAccent: '#ef5350', lightBg: '#ffebee', lightAccent: '#b71c1c' },
  custom: { darkBg: '#2a2a2a', darkAccent: '#8a7e74', lightBg: '#f5f5f5', lightAccent: '#616161' },
};

export interface BpPin {
  id: string;
  name: string;
  direction: 'in' | 'out';
  dataType: PinType;
  shape?: number[];
  dtype?: string;
  connected: boolean;
  value?: unknown;
}

export interface BpNode {
  id: string;
  type: string;          // "op", "input", "output", "group", "comment"
  label: string;
  category: NodeCategory;
  position: { x: number; y: number };
  size: { w: number; h: number };
  collapsed: boolean;
  enabled: boolean;
  metadata: Record<string, unknown>;
  pins: BpPin[];
}

export interface BpEdge {
  id: string;
  from: { nodeId: string; pinId: string };
  to: { nodeId: string; pinId: string };
  valid: boolean;
}

export interface BpDocument {
  id: string;
  name: string;
  type: 'model' | 'arch' | 'backend' | 'quant' | 'sampling' | 'surgery' | 'benchmark' | 'test';
  nodes: BpNode[];
  edges: BpEdge[];
  viewport: { x: number; y: number; zoom: number };
  dirty: boolean;
}
```

**Step 2: Create the document store**

Create `web/src/lib/engine/store.ts`:

```typescript
import { writable, derived, get } from 'svelte/store';
import type { BpDocument, BpNode, BpEdge } from './types';

// All open documents
export const documents = writable<BpDocument[]>([]);

// Active document ID
export const activeDocId = writable<string | null>(null);

// Derived: active document
export const activeDoc = derived(
  [documents, activeDocId],
  ([$docs, $id]) => $docs.find(d => d.id === $id) || null
);

// Undo/redo stacks per document
const undoStacks = new Map<string, BpDocument[]>();
const redoStacks = new Map<string, BpDocument[]>();
const MAX_UNDO = 100;

export function openDocument(doc: BpDocument) {
  documents.update(docs => {
    const existing = docs.findIndex(d => d.id === doc.id);
    if (existing >= 0) {
      docs[existing] = doc;
    } else {
      docs.push(doc);
    }
    return docs;
  });
  activeDocId.set(doc.id);
}

export function closeDocument(docId: string) {
  documents.update(docs => docs.filter(d => d.id !== docId));
  undoStacks.delete(docId);
  redoStacks.delete(docId);
  // Switch to another tab
  const remaining = get(documents);
  activeDocId.set(remaining.length > 0 ? remaining[remaining.length - 1].id : null);
}

export function updateDocument(docId: string, updater: (doc: BpDocument) => BpDocument) {
  documents.update(docs => {
    const idx = docs.findIndex(d => d.id === docId);
    if (idx < 0) return docs;

    // Push current state to undo stack
    const stack = undoStacks.get(docId) || [];
    stack.push(structuredClone(docs[idx]));
    if (stack.length > MAX_UNDO) stack.shift();
    undoStacks.set(docId, stack);
    redoStacks.set(docId, []);

    docs[idx] = updater(docs[idx]);
    docs[idx].dirty = true;
    return docs;
  });
}

export function undo(docId: string) {
  const stack = undoStacks.get(docId);
  if (!stack || stack.length === 0) return;
  const prev = stack.pop()!;
  documents.update(docs => {
    const idx = docs.findIndex(d => d.id === docId);
    if (idx < 0) return docs;
    const redo = redoStacks.get(docId) || [];
    redo.push(structuredClone(docs[idx]));
    redoStacks.set(docId, redo);
    docs[idx] = prev;
    return docs;
  });
}

export function redo(docId: string) {
  const stack = redoStacks.get(docId);
  if (!stack || stack.length === 0) return;
  const next = stack.pop()!;
  documents.update(docs => {
    const idx = docs.findIndex(d => d.id === docId);
    if (idx < 0) return docs;
    const undoStack = undoStacks.get(docId) || [];
    undoStack.push(structuredClone(docs[idx]));
    undoStacks.set(docId, undoStack);
    docs[idx] = next;
    return docs;
  });
}

// Selection state (not per-document, just current)
export const selectedNodeIds = writable<Set<string>>(new Set());
export const selectedEdgeId = writable<string | null>(null);

// Interaction state
export const canvasInteraction = writable<{
  mode: 'idle' | 'pan' | 'drag-node' | 'drag-wire' | 'box-select';
  origin?: { x: number; y: number };
  wireFrom?: { nodeId: string; pinId: string };
}>({ mode: 'idle' });
```

**Step 3: Commit**

```bash
git add web/src/lib/engine/
git commit -m "feat(engine): add Blueprint node engine types and document store"
```

---

### Task 2: Blueprint Node Engine — SVG Canvas Component

**Files:**
- Create: `web/src/lib/engine/NodeCanvas.svelte`
- Create: `web/src/lib/engine/NodeRenderer.svelte`
- Create: `web/src/lib/engine/EdgeRenderer.svelte`
- Create: `web/src/lib/engine/PinRenderer.svelte`

**Step 1: Create PinRenderer.svelte**

The pin circle on a node edge. Shows data type color, connection state.

```svelte
<script lang="ts">
  import { PIN_COLORS, type BpPin } from './types';

  let { pin, x, y, nodeId, onStartWire, onEndWire }: {
    pin: BpPin;
    x: number;
    y: number;
    nodeId: string;
    onStartWire: (nodeId: string, pinId: string) => void;
    onEndWire: (nodeId: string, pinId: string) => void;
  } = $props();

  const r = 5;
  const color = $derived(PIN_COLORS[pin.dataType]);
</script>

<g class="pin" role="button" tabindex="-1"
   onmousedown={(e) => { e.stopPropagation(); if (pin.direction === 'out') onStartWire(nodeId, pin.id); }}
   onmouseup={(e) => { e.stopPropagation(); if (pin.direction === 'in') onEndWire(nodeId, pin.id); }}
>
  <circle cx={x} cy={y} r={r + 3} fill="transparent" />
  <circle cx={x} cy={y} {r} fill={pin.connected ? color : 'transparent'} stroke={color} stroke-width="1.5" />
  <text x={pin.direction === 'in' ? x + 10 : x - 10} y={y + 3.5}
        text-anchor={pin.direction === 'in' ? 'start' : 'end'}
        fill="var(--text-secondary)" font-size="9" font-family="var(--font-mono)">
    {pin.name}
  </text>
  {#if pin.shape && pin.shape.length > 0}
    <text x={pin.direction === 'in' ? x + 10 : x - 10}
          y={y + 13}
          text-anchor={pin.direction === 'in' ? 'start' : 'end'}
          fill="var(--text-tertiary)" font-size="7" font-family="var(--font-mono)">
      [{pin.shape.join('×')}]
    </text>
  {/if}
</g>
```

**Step 2: Create NodeRenderer.svelte**

Single node: rounded rect with category accent bar, pins on left/right, label, collapse, enable state.

```svelte
<script lang="ts">
  import { CATEGORY_COLORS, type BpNode } from './types';
  import PinRenderer from './PinRenderer.svelte';
  import { theme } from '$lib/stores/settings';

  let { node, selected, onStartWire, onEndWire }: {
    node: BpNode;
    selected: boolean;
    onStartWire: (nodeId: string, pinId: string) => void;
    onEndWire: (nodeId: string, pinId: string) => void;
  } = $props();

  const colors = $derived(CATEGORY_COLORS[node.category] || CATEGORY_COLORS.custom);
  const bg = $derived($theme === 'dark' ? colors.darkBg : colors.lightBg);
  const accent = $derived($theme === 'dark' ? colors.darkAccent : colors.lightAccent);

  const inPins = $derived(node.pins.filter(p => p.direction === 'in'));
  const outPins = $derived(node.pins.filter(p => p.direction === 'out'));
  const pinSpacing = 22;
  const headerH = 26;
  const bodyH = $derived(Math.max(inPins.length, outPins.length) * pinSpacing + 8);
  const nodeW = $derived(node.size.w || 180);
  const nodeH = $derived(node.collapsed ? headerH : headerH + bodyH);
</script>

<g class="bp-node" opacity={node.enabled ? 1 : 0.4}
   transform="translate({node.position.x},{node.position.y})">
  <!-- Shadow -->
  <rect x="2" y="2" width={nodeW} height={nodeH} rx="4" fill="rgba(0,0,0,0.3)" />
  <!-- Body -->
  <rect width={nodeW} height={nodeH} rx="4" fill={bg}
        stroke={selected ? accent : 'var(--border)'} stroke-width={selected ? 2 : 1} />
  <!-- Accent bar -->
  <rect width="4" height={nodeH} rx="2" fill={accent} />
  <!-- Header -->
  <text x="12" y="17" fill={accent} font-size="11" font-weight="600" font-family="var(--font-mono)">
    {node.label}
  </text>
  <!-- Disabled X -->
  {#if !node.enabled}
    <line x1="4" y1="4" x2={nodeW - 4} y2={nodeH - 4} stroke="var(--error)" stroke-width="1" opacity="0.5" />
    <line x1={nodeW - 4} y1="4" x2="4" y2={nodeH - 4} stroke="var(--error)" stroke-width="1" opacity="0.5" />
  {/if}
  <!-- Pins (only if not collapsed) -->
  {#if !node.collapsed}
    {#each inPins as pin, i}
      <PinRenderer {pin} x={0} y={headerH + i * pinSpacing + 12} nodeId={node.id} {onStartWire} {onEndWire} />
    {/each}
    {#each outPins as pin, i}
      <PinRenderer {pin} x={nodeW} y={headerH + i * pinSpacing + 12} nodeId={node.id} {onStartWire} {onEndWire} />
    {/each}
  {/if}
</g>
```

**Step 3: Create EdgeRenderer.svelte**

Bezier curves between connected pins, colored by pin data type.

```svelte
<script lang="ts">
  import { PIN_COLORS, type PinType } from './types';

  let { x1, y1, x2, y2, dataType, valid }: {
    x1: number; y1: number; x2: number; y2: number;
    dataType: PinType; valid: boolean;
  } = $props();

  const color = $derived(PIN_COLORS[dataType]);
  const dx = $derived(Math.abs(x2 - x1) * 0.5);
  const path = $derived(`M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`);
</script>

<path d={path} fill="none" stroke={valid ? color : 'var(--error)'} stroke-width="2"
      stroke-dasharray={valid ? 'none' : '4 4'} opacity="0.7" />
```

**Step 4: Create NodeCanvas.svelte**

The main canvas component: SVG container with pan/zoom, renders nodes + edges, handles all interactions.

This is the biggest component. It handles:
- Wheel zoom (addEventListener with passive:false)
- Pan (middle-click or Space+left-drag)
- Node drag (left-click on node)
- Wire drag (left-drag from output pin to input pin)
- Box select (left-drag on empty canvas)
- Right-click context menu
- Keyboard shortcuts (Del, Ctrl+D, Ctrl+Z/Y, F)
- Minimap

Create `web/src/lib/engine/NodeCanvas.svelte`. This file is ~400 lines. Key structure:

```
<script>
  - Props: docId (which document to render)
  - Derived: nodes, edges from activeDoc
  - Pan/zoom state
  - Event handlers: onWheel, onMouseDown, onMouseMove, onMouseUp, onKeyDown, onContextMenu
  - Pin position calculator (maps pin IDs to absolute SVG coordinates)
  - Wire drag state (temporary edge while dragging)
  - Box select state
  - Context menu state
  - Auto-layout function (hierarchical)
  - Minimap rendering
</script>
<div class="canvas-container" bind:this={containerEl}>
  <svg viewBox="..." >
    <g transform="translate(panX, panY) scale(zoom)">
      <!-- Edges -->
      {#each edges as edge}
        <EdgeRenderer ... />
      {/each}
      <!-- Temp wire (during drag) -->
      {#if wireState}
        <EdgeRenderer ... />
      {/if}
      <!-- Box select rect -->
      {#if boxSelect}
        <rect ... />
      {/if}
      <!-- Nodes -->
      {#each nodes as node}
        <NodeRenderer ... />
      {/each}
    </g>
  </svg>
  <!-- Context menu (HTML overlay) -->
  {#if contextMenu}
    <div class="context-menu" ...>
      <button>Add Node...</button>
      <button>Duplicate</button>
      <button>Delete</button>
      ...
    </div>
  {/if}
  <!-- Minimap (bottom-right) -->
  <div class="minimap">
    <svg ...> mini node rects </svg>
  </div>
</div>
```

The full implementation follows the same interaction model as described in the design doc (Section 2.3).

**Step 5: Commit**

```bash
git add web/src/lib/engine/
git commit -m "feat(engine): add Blueprint SVG canvas with nodes, edges, pins"
```

---

### Task 3: IntelliJ Layout — Tool Windows + Document Tabs

**Files:**
- Create: `web/src/lib/layout/Workspace.svelte`
- Create: `web/src/lib/layout/ToolWindow.svelte`
- Create: `web/src/lib/layout/DocumentTabs.svelte`
- Create: `web/src/lib/layout/Toolbar.svelte`
- Create: `web/src/lib/layout/store.ts`
- Modify: `web/src/lib/components/AppShell.svelte` (replace entirely)
- Modify: `web/src/routes/+page.svelte`
- Delete (stop using): `web/src/lib/components/Sidebar.svelte`, `TabBar.svelte`, `TopBar.svelte`

**Step 1: Create layout store**

`web/src/lib/layout/store.ts` — manages which tool windows are open and their positions:

```typescript
import { writable } from 'svelte/store';

export type ToolWindowPosition = 'left' | 'right' | 'bottom';

export interface ToolWindowState {
  id: string;
  label: string;
  icon: string;      // Tabler icon name
  position: ToolWindowPosition;
  open: boolean;
  width?: number;     // for left/right
  height?: number;    // for bottom
}

export const toolWindows = writable<ToolWindowState[]>([
  // Left
  { id: 'model',    label: 'Model',    icon: 'IconFolder',        position: 'left',   open: true,  width: 240 },
  { id: 'palette',  label: 'Palette',  icon: 'IconComponents',    position: 'left',   open: false, width: 240 },
  { id: 'plugins',  label: 'Plugins',  icon: 'IconPuzzle',        position: 'left',   open: false, width: 240 },
  { id: 'devices',  label: 'Devices',  icon: 'IconDeviceMobile',  position: 'left',   open: false, width: 240 },
  { id: 'tests',    label: 'Tests',    icon: 'IconTestPipe',      position: 'left',   open: false, width: 240 },
  // Right
  { id: 'props',    label: 'Properties', icon: 'IconAdjustments',  position: 'right',  open: true,  width: 260 },
  { id: 'bench',    label: 'Benchmarks', icon: 'IconChartBar',     position: 'right',  open: false, width: 260 },
  { id: 'surgery',  label: 'Surgery',    icon: 'IconScissors',     position: 'right',  open: false, width: 260 },
  // Bottom
  { id: 'console',  label: 'Console',    icon: 'IconTerminal2',    position: 'bottom', open: true,  height: 160 },
  { id: 'devout',   label: 'Device',     icon: 'IconDeviceMobile', position: 'bottom', open: false, height: 160 },
  { id: 'terminal', label: 'Terminal',   icon: 'IconCode',         position: 'bottom', open: false, height: 160 },
]);

export function toggleToolWindow(id: string) {
  toolWindows.update(windows => {
    const w = windows.find(w => w.id === id);
    if (w) {
      // Close other windows in same position (accordion)
      windows.filter(o => o.position === w.position && o.id !== id).forEach(o => o.open = false);
      w.open = !w.open;
    }
    return windows;
  });
}
```

**Step 2: Create ToolWindow.svelte**

A collapsible, resizable panel that docks to left/right/bottom:

```svelte
<script lang="ts">
  import type { ToolWindowState } from './store';

  let { window, children }: { window: ToolWindowState; children: any } = $props();
</script>

{#if window.open}
  <div class="tool-window"
       style:width={window.position !== 'bottom' ? `${window.width}px` : undefined}
       style:height={window.position === 'bottom' ? `${window.height}px` : undefined}>
    <div class="tw-header">
      <span class="tw-label">{window.label}</span>
    </div>
    <div class="tw-body">
      {@render children()}
    </div>
  </div>
{/if}
```

**Step 3: Create DocumentTabs.svelte**

Tab strip for open documents with close buttons and + button:

```svelte
<script lang="ts">
  import { documents, activeDocId, closeDocument } from '$lib/engine/store';
</script>

<div class="doc-tabs">
  {#each $documents as doc}
    <button class="doc-tab" class:active={$activeDocId === doc.id}
            onclick={() => activeDocId.set(doc.id)}>
      <span class="doc-name">{doc.dirty ? '● ' : ''}{doc.name}</span>
      <button class="doc-close" onclick|stopPropagation={() => closeDocument(doc.id)}>×</button>
    </button>
  {/each}
  <button class="doc-tab add-tab" title="New document">+</button>
</div>
```

**Step 4: Create Toolbar.svelte**

Top bar with brand + action buttons:

```svelte
<script lang="ts">
  import { theme, showSettings } from '$lib/stores/settings';
  import { serverOnline } from '$lib/stores/device';
  // Tabler icons
  import IconPlayerPlay from '@tabler/icons-svelte/IconPlayerPlay.svelte';
  import IconPlayerStop from '@tabler/icons-svelte/IconPlayerStop.svelte';
  import IconUpload from '@tabler/icons-svelte/IconUpload.svelte';
  import IconRefresh from '@tabler/icons-svelte/IconRefresh.svelte';
  import IconSettings from '@tabler/icons-svelte/IconSettings.svelte';
  import IconSun from '@tabler/icons-svelte/IconSun.svelte';
  import IconMoon from '@tabler/icons-svelte/IconMoon.svelte';

  // Action callbacks passed from parent
  let { onRun, onStop, onDeploy, onRebuild }: {
    onRun: () => void; onStop: () => void;
    onDeploy: () => void; onRebuild: () => void;
  } = $props();
</script>

<header class="toolbar">
  <div class="toolbar-left">
    <span class="brand-mark">A</span>
    <span class="brand-name">AUTOMATRIX</span>
  </div>
  <div class="toolbar-actions">
    <button class="tb-btn run" onclick={onRun} title="Run inference (F5)">
      <IconPlayerPlay size={16} /> Run
    </button>
    <button class="tb-btn" onclick={onStop} title="Stop (Shift+F5)">
      <IconPlayerStop size={16} />
    </button>
    <div class="tb-sep"></div>
    <button class="tb-btn" onclick={onDeploy} title="Deploy to device">
      <IconUpload size={16} /> Deploy
    </button>
    <button class="tb-btn" onclick={onRebuild} title="Rebuild binary">
      <IconRefresh size={16} />
    </button>
  </div>
  <div class="toolbar-right">
    <div class="server-status" class:online={$serverOnline}>
      <span class="status-dot" class:online={$serverOnline}></span>
      <span>{$serverOnline ? 'CONNECTED' : 'OFFLINE'}</span>
    </div>
    <button class="tb-icon" onclick={() => theme.toggle()}>
      {#if $theme === 'dark'}<IconSun size={14} />{:else}<IconMoon size={14} />{/if}
    </button>
    <button class="tb-icon" onclick={() => showSettings.update(v => !v)}>
      <IconSettings size={14} />
    </button>
  </div>
</header>
```

**Step 5: Create Workspace.svelte — the main layout**

Replaces AppShell. Assembles: Toolbar + tool window buttons (left strip) + left panels + document tabs + canvas + right panels + bottom panels + status bar.

```
┌─ Toolbar ──────────────────────────────────────────────────────┐
├─┬─ Doc Tabs ───────────────────────────────────────────────┬───┤
│B│                                                          │ B │
│u│  [Left Tool Window]  │  NodeCanvas  │  [Right Tool Win]  │ u │
│t│                      │              │                    │ t │
│t│                      │              │                    │ t │
│o│                      │              │                    │ o │
│n│                      │              │                    │ n │
│s│                      │              │                    │ s │
├─┼──────────────────────┴──────────────┴────────────────────┼───┤
│ │  [Bottom Tool Window: Console / Device / Terminal]        │   │
├─┴──────────────────────────────────────────────────────────┴───┤
│ StatusBar                                                      │
└────────────────────────────────────────────────────────────────┘
```

Left/right button strips are vertical icon columns (like IntelliJ) that toggle tool windows.

**Step 6: Replace AppShell with Workspace**

Modify `+page.svelte` to use the new Workspace instead of AppShell.

**Step 7: Commit**

```bash
git add web/src/lib/layout/ web/src/routes/+page.svelte
git commit -m "feat(layout): IntelliJ-style workspace with tool windows and document tabs"
```

---

### Task 4: Wire Existing Panels into Tool Windows

**Files:**
- Modify: `web/src/lib/components/Outliner.svelte` → becomes Model tool window content
- Modify: `web/src/lib/components/Properties.svelte` → becomes Properties tool window content
- Modify: `web/src/lib/components/Console.svelte` → becomes Console tool window content
- Keep: `web/src/lib/components/StatusBar.svelte` (stays in Workspace)
- Keep: `web/src/lib/components/SettingsPanel.svelte` (overlay, stays)

The existing panel components are already well-built. They just need minor adjustments:
- Remove their own `panel` wrapper divs (ToolWindow provides the chrome)
- Export any needed state for cross-panel communication

**Step 1:** Strip panel chrome from Outliner, Properties, Console (the ToolWindow wraps them now)
**Step 2:** Wire them as children of ToolWindow in Workspace
**Step 3:** Verify existing stores (model, device, console) still work
**Step 4:** Commit

```bash
git commit -m "feat(layout): wire existing panels into tool windows"
```

---

### Task 5: Node Palette Tool Window

**Files:**
- Create: `web/src/lib/panels/NodePalette.svelte`
- Create: `web/src/lib/engine/node-templates.ts`

**Step 1: Create node templates**

`web/src/lib/engine/node-templates.ts` — factory functions for each node type. Grouped by category for the palette:

```typescript
import type { BpNode, BpPin, NodeCategory, PinType } from './types';

let nextId = 1;
function uid() { return `n${nextId++}`; }
function pin(name: string, dir: 'in' | 'out', type: PinType, shape?: number[]): BpPin {
  return { id: `${name}-${dir}`, name, direction: dir, dataType: type, shape, connected: false };
}

export interface NodeTemplate {
  label: string;
  category: NodeCategory;
  description: string;
  create: (x: number, y: number) => BpNode;
}

export const MODEL_NODES: NodeTemplate[] = [
  {
    label: 'Token Embed', category: 'embed', description: 'Embedding lookup table',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'Token Embed', category: 'embed',
      position: { x, y }, size: { w: 180, h: 0 }, collapsed: false, enabled: true, metadata: {},
      pins: [pin('tokens', 'in', 'tokens'), pin('embed', 'out', 'tensor')]
    })
  },
  {
    label: 'Attention', category: 'attn', description: 'Multi-head attention (QKV + softmax)',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'Attention', category: 'attn',
      position: { x, y }, size: { w: 200, h: 0 }, collapsed: false, enabled: true,
      metadata: { heads: 15, kvHeads: 5, headDim: 64 },
      pins: [
        pin('input', 'in', 'tensor'), pin('kv_cache', 'in', 'kv_cache'),
        pin('output', 'out', 'tensor'), pin('kv_out', 'out', 'kv_cache')
      ]
    })
  },
  {
    label: 'FFN', category: 'ffn', description: 'Feed-forward network (gate+up → activation → down)',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'FFN', category: 'ffn',
      position: { x, y }, size: { w: 180, h: 0 }, collapsed: false, enabled: true,
      metadata: { intermediate: 2560, activation: 'silu', fusedGateUp: true },
      pins: [pin('input', 'in', 'tensor'), pin('output', 'out', 'tensor')]
    })
  },
  {
    label: 'RMS Norm', category: 'norm', description: 'Root Mean Square normalization',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'RMS Norm', category: 'norm',
      position: { x, y }, size: { w: 160, h: 0 }, collapsed: false, enabled: true,
      metadata: { eps: 1e-5 },
      pins: [pin('input', 'in', 'tensor'), pin('eps', 'in', 'scalar'), pin('output', 'out', 'tensor')]
    })
  },
  {
    label: 'LM Head', category: 'head', description: 'Output projection to vocabulary logits',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'LM Head', category: 'head',
      position: { x, y }, size: { w: 180, h: 0 }, collapsed: false, enabled: true, metadata: {},
      pins: [pin('input', 'in', 'tensor'), pin('logits', 'out', 'tokens')]
    })
  },
  {
    label: 'MatMul', category: 'math', description: 'Matrix multiplication',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'MatMul', category: 'math',
      position: { x, y }, size: { w: 160, h: 0 }, collapsed: false, enabled: true, metadata: {},
      pins: [pin('A', 'in', 'tensor'), pin('B', 'in', 'tensor'), pin('C', 'out', 'tensor')]
    })
  },
  {
    label: 'SiLU', category: 'math', description: 'SiLU activation function',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'SiLU', category: 'math',
      position: { x, y }, size: { w: 140, h: 0 }, collapsed: false, enabled: true, metadata: {},
      pins: [pin('input', 'in', 'tensor'), pin('output', 'out', 'tensor')]
    })
  },
  {
    label: 'GELU', category: 'math', description: 'GELU activation function',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'GELU', category: 'math',
      position: { x, y }, size: { w: 140, h: 0 }, collapsed: false, enabled: true, metadata: {},
      pins: [pin('input', 'in', 'tensor'), pin('output', 'out', 'tensor')]
    })
  },
  {
    label: 'Softmax', category: 'math', description: 'Softmax normalization',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'Softmax', category: 'math',
      position: { x, y }, size: { w: 140, h: 0 }, collapsed: false, enabled: true, metadata: {},
      pins: [pin('input', 'in', 'tensor'), pin('output', 'out', 'tensor')]
    })
  },
  {
    label: 'RoPE', category: 'math', description: 'Rotary Position Embedding',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'RoPE', category: 'math',
      position: { x, y }, size: { w: 160, h: 0 }, collapsed: false, enabled: true, metadata: {},
      pins: [pin('input', 'in', 'tensor'), pin('config', 'in', 'config'), pin('output', 'out', 'tensor')]
    })
  },
  {
    label: 'Residual Add', category: 'math', description: 'Skip connection addition',
    create: (x, y) => ({
      id: uid(), type: 'op', label: 'Residual Add', category: 'math',
      position: { x, y }, size: { w: 160, h: 0 }, collapsed: false, enabled: true, metadata: {},
      pins: [pin('A', 'in', 'tensor'), pin('B', 'in', 'tensor'), pin('sum', 'out', 'tensor')]
    })
  },
];

// More templates for backend, quant, sampling, surgery in separate exports
```

**Step 2: Create NodePalette.svelte**

Categorized, searchable list of draggable node templates:

```svelte
<script lang="ts">
  import { MODEL_NODES, type NodeTemplate } from '$lib/engine/node-templates';

  let search = $state('');
  let expandedCat = $state<string | null>(null);

  const categories = $derived.by(() => {
    const cats = new Map<string, NodeTemplate[]>();
    for (const tmpl of MODEL_NODES) {
      if (search && !tmpl.label.toLowerCase().includes(search.toLowerCase())) continue;
      const cat = tmpl.category;
      if (!cats.has(cat)) cats.set(cat, []);
      cats.get(cat)!.push(tmpl);
    }
    return cats;
  });

  function onDragStart(e: DragEvent, tmpl: NodeTemplate) {
    e.dataTransfer?.setData('application/bp-node', JSON.stringify({ label: tmpl.label }));
  }
</script>

<div class="palette">
  <input type="text" placeholder="Search nodes..." bind:value={search} />
  {#each [...categories] as [cat, templates]}
    <div class="pal-category">
      <button class="pal-cat-header" onclick={() => expandedCat = expandedCat === cat ? null : cat}>
        {cat.toUpperCase()} ({templates.length})
      </button>
      {#if expandedCat === cat || search}
        {#each templates as tmpl}
          <div class="pal-item" draggable="true" ondragstart={(e) => onDragStart(e, tmpl)}>
            <span class="pal-dot" style:background={CATEGORY_COLORS[tmpl.category]?.darkAccent}></span>
            <span>{tmpl.label}</span>
          </div>
        {/each}
      {/if}
    </div>
  {/each}
</div>
```

**Step 3: Commit**

```bash
git add web/src/lib/engine/node-templates.ts web/src/lib/panels/NodePalette.svelte
git commit -m "feat(palette): add node palette with draggable templates"
```

---

### Task 6: Model Graph Document — Load GGUF → Blueprint Canvas

**Files:**
- Create: `web/src/lib/engine/model-to-graph.ts`
- Modify: `web/src/lib/api/client.ts` (add graph-to-blueprint converter)

**Step 1: Create model-to-graph converter**

Takes the model info + compute graph from the backend and converts it into a `BpDocument` with proper Blueprint nodes and edges.

`web/src/lib/engine/model-to-graph.ts`:

```typescript
import type { BpDocument, BpNode, BpEdge } from './types';
import type { ModelInfo } from '$lib/stores/model';
import type { ModelGraph, GraphNode, GraphEdge } from '$lib/api/client';

function categorizeNodeType(type: string): NodeCategory {
  if (type === 'embed' || type === 'pos_embed') return 'embed';
  if (type.includes('attn')) return 'attn';
  if (type.includes('ffn')) return 'ffn';
  if (type.includes('norm')) return 'norm';
  if (type === 'head' || type === 'output') return 'head';
  return 'math';
}

function pinsForType(type: string): BpPin[] {
  // Return appropriate typed pins based on node type
  // ... (implementation based on node-templates.ts patterns)
}

export function modelToBpDocument(modelInfo: ModelInfo, graph: ModelGraph): BpDocument {
  const nodes: BpNode[] = graph.nodes.map(gn => ({
    id: gn.id,
    type: 'op',
    label: gn.label,
    category: categorizeNodeType(gn.type),
    position: { x: gn.x, y: gn.y },
    size: { w: gn.width || 180, h: 0 },
    collapsed: false,
    enabled: true,
    metadata: {
      sublabel: gn.sublabel,
      layer: gn.layer,
      originalType: gn.type,
    },
    pins: pinsForType(gn.type),
  }));

  const edges: BpEdge[] = graph.edges.map((ge, i) => ({
    id: `e${i}`,
    from: { nodeId: ge.from, pinId: 'output-out' },
    to: { nodeId: ge.to, pinId: 'input-in' },
    valid: true,
  }));

  return {
    id: `model-${Date.now()}`,
    name: `${modelInfo.name || modelInfo.arch}.gguf`,
    type: 'model',
    nodes,
    edges,
    viewport: { x: 60, y: 20, zoom: 1 },
    dirty: false,
  };
}
```

**Step 2: Add "Open Model" action in Workspace**

When user loads a GGUF via the Model tool window, call `getModelGraph()` → `modelToBpDocument()` → `openDocument()`. The canvas automatically renders it.

**Step 3: Commit**

```bash
git commit -m "feat(model): convert GGUF model graph to Blueprint document"
```

---

### Task 7: Properties Panel — Blueprint-Aware Inspector

**Files:**
- Modify: `web/src/lib/components/Properties.svelte`

When a node is selected on the canvas, the Properties tool window shows:
- Node label (editable)
- Category badge
- Enable/disable toggle
- All pins with their types, shapes, dtypes
- Metadata fields (editable): quant type, activation, eps, heads, etc.
- Position (x, y)
- Actions: Duplicate, Delete

The Properties panel subscribes to `selectedNodeIds` from the engine store and reads the active document to get node data.

**Commit:**
```bash
git commit -m "feat(props): Blueprint-aware node inspector in Properties panel"
```

---

## Phase 2: Plugin Node Editors

### Task 8: Architecture Plugin Editor

**Files:**
- Create: `web/src/lib/engine/arch-templates.ts`
- Modify: `web/src/lib/panels/PluginsPanel.svelte` (double-click opens editor)

Architecture plugins open as BpDocuments with architecture-specific nodes:
- Transformer Block (group node, expandable)
- Attention Config (heads, kv_heads, head_dim)
- FFN Config (intermediate_size, activation, fused)
- Norm Config (type, eps)
- Vision Encoder, Projector

When saved, the canvas state serializes back to AMXP JSON body.

**Commit:**
```bash
git commit -m "feat(plugins): architecture plugin node editor"
```

---

### Task 9: Quant Plugin Editor

**Files:**
- Create: `web/src/lib/engine/quant-templates.ts`

Quant plugins get visual nodes:
- Bit Allocator (horizontal bar with slider)
- Block Config (block size dropdown)
- Quality Meter (color gradient visualization)
- Benchmark node (connects to device, shows tok/s)

**Commit:**
```bash
git commit -m "feat(plugins): quant plugin node editor"
```

---

### Task 10: Sampling Plugin Editor

**Files:**
- Create: `web/src/lib/engine/sampling-templates.ts`

Sampling pipeline as nodes wired in sequence:
Logit Input → Temperature → Top-K → Top-P → Min-P → RepPenalty → Sampler → Token Output

Each node has slider controls in the Properties panel.

**Commit:**
```bash
git commit -m "feat(plugins): sampling plugin node editor"
```

---

### Task 11: Backend Plugin Editor

**Files:**
- Create: `web/src/lib/engine/backend-templates.ts`

Hardware pipeline nodes:
- CPU Core (with ISA extensions)
- GPU CU (compute units)
- Memory Pool (UMA — single shared pool)
- Thread Pool (OMP config)
- Dispatch (routing rules)

**Commit:**
```bash
git commit -m "feat(plugins): backend plugin node editor"
```

---

## Phase 3: Simulation + Surgery

### Task 12: Quick-Mode Simulation

**Files:**
- Modify: `server/src/main.cpp` (add SSE streaming endpoint)
- Modify: `android/vlm-test.cpp` (add `--config` JSON flag)
- Create: `web/src/lib/engine/config-generator.ts`
- Modify: `web/src/lib/api/client.ts` (add SSE client)

**Step 1: Add `--config` to vlm-test**

New flag reads a JSON file with all inference params instead of individual CLI flags:

```json
{
  "model": "/sdcard/Download/SmolVLM-500M-Instruct-q8_0.gguf",
  "mmproj": "/sdcard/Download/SmolVLM-500M-Instruct-q8_0.mmproj",
  "image": "/sdcard/Download/VLM-TEST-IMAGE.jpg",
  "prompt": "Describe this image",
  "threads": 4,
  "quant": "q5_0",
  "max_tokens": 128,
  "gpu": false,
  "disabled_layers": [],
  "per_layer_quant": {}
}
```

**Step 2: Create config generator**

`web/src/lib/engine/config-generator.ts` — reads active BpDocument and generates the JSON config from canvas state (which nodes are enabled, per-node quant, etc.)

**Step 3: Add SSE streaming**

Backend streams inference output line-by-line as SSE events. Frontend reads with `EventSource` and updates Console + Benchmarks panels in real-time.

**Step 4: Add result annotations**

After inference, parse timing stats and add annotation badges to nodes on the canvas (ms per op, memory used, bottleneck highlighting).

**Commit:**
```bash
git commit -m "feat(sim): quick-mode simulation with SSE streaming and result annotations"
```

---

### Task 13: Surgery Document Type

**Files:**
- Create: `web/src/lib/engine/surgery-templates.ts`
- Create: `web/src/lib/panels/SurgeryTools.svelte`
- Modify: `server/src/main.cpp` (add surgery endpoints)

**Step 1: Surgery tool nodes**

Add surgery-specific nodes to the palette:
- Layer Delete, Layer Duplicate, Head Prune
- Weight Requant, Tensor Inspector, LoRA Patch
- Merge Point, Vocab Prune, Export

**Step 2: Surgery tools panel (right sidebar)**

Quick-access buttons for common operations:
- "Prune Layer" — click then click node to remove
- "Requant" — select nodes, pick target quant
- "Export GGUF" — save modified model

**Step 3: Backend surgery endpoints**

- `POST /api/surgery/requant` — requantize tensors
- `GET /api/model/tensor/:name` — tensor value inspection
- `POST /api/surgery/export` — write modified GGUF

**Commit:**
```bash
git commit -m "feat(surgery): surgical tools for layer/head/weight manipulation"
```

---

## Phase 4: Test Cases + Polish

### Task 14: Test Case System

**Files:**
- Create: `web/src/lib/engine/test-types.ts`
- Create: `web/src/lib/panels/TestPanel.svelte`
- Create: `web/src/lib/views/TestEditor.svelte`
- Modify: `server/src/main.cpp` (add test endpoints)
- Modify: `server/src/amxp_format.h` (add AMXP_TEST type)

**Step 1: Add AMXP_TEST type (type=4)**

**Step 2: Create TestCase types matching design doc**

**Step 3: Test Panel (left tool window)** — lists test cases with pass/fail badges

**Step 4: Test Editor (opens as document tab)** — config form + expectations + results chart

**Step 5: Backend endpoints** — run/save/list test cases

**Commit:**
```bash
git commit -m "feat(tests): test case system with run/compare/history"
```

---

### Task 15: Benchmark Panel

**Files:**
- Create: `web/src/lib/panels/BenchmarkPanel.svelte`

Charts showing:
- tok/s history over runs (line chart)
- Memory usage per run (bar chart)
- A/B comparison table
- Per-node timing breakdown

Use inline SVG charts (no heavy charting library needed).

**Commit:**
```bash
git commit -m "feat(bench): benchmark panel with charts and A/B comparison"
```

---

### Task 16: CSS Theme Variables for Node Colors

**Files:**
- Modify: `web/src/app.css`

Add CSS variables for all node category colors so they respond to theme changes:

```css
:root {
  --node-embed-bg: #1a3a4d;
  --node-embed-accent: #6ec6ff;
  --node-attn-bg: #3d2a14;
  --node-attn-accent: #e8963a;
  /* ... etc for all 9 categories */
}

:root[data-theme="light"] {
  --node-embed-bg: #e3f2fd;
  --node-embed-accent: #1976d2;
  --node-attn-bg: #fff3e0;
  --node-attn-accent: #e65100;
  /* ... etc */
}
```

**Commit:**
```bash
git commit -m "fix(theme): add CSS variables for node colors, fix black nodes bug"
```

---

### Task 17: Keyboard Shortcuts

**Files:**
- Create: `web/src/lib/engine/shortcuts.ts`
- Modify: `web/src/lib/layout/Workspace.svelte`

Global keyboard handler:
- `Del` → delete selected nodes
- `Ctrl+D` → duplicate selected
- `Ctrl+Z` → undo
- `Ctrl+Y` → redo
- `Ctrl+G` → group selected
- `F` → fit view (zoom to fit all nodes)
- `Ctrl+F` → search nodes
- `F5` → run inference
- `Shift+F5` → stop inference

**Commit:**
```bash
git commit -m "feat(shortcuts): keyboard shortcuts for all canvas operations"
```

---

## Implementation Priority Order

1. **Task 1** — Engine types + store (foundation for everything)
2. **Task 2** — SVG canvas component (the core rendering)
3. **Task 16** — CSS theme fix (fix black nodes immediately)
4. **Task 3** — IntelliJ layout (replaces broken nav)
5. **Task 4** — Wire existing panels
6. **Task 5** — Node palette
7. **Task 6** — Model graph loading
8. **Task 7** — Properties inspector
9. **Task 17** — Keyboard shortcuts
10. **Task 8-11** — Plugin editors (can be parallelized)
11. **Task 12** — Simulation pipeline
12. **Task 13** — Surgery
13. **Task 14** — Test cases
14. **Task 15** — Benchmarks

---

## Verification Criteria

After all tasks complete:

1. `cd AutoMatrix/web && npm run dev` → app starts, shows IntelliJ-style workspace
2. Load GGUF → model opens as Blueprint canvas with colored nodes and typed pins
3. Nodes are visible in both dark and light themes (no black nodes)
4. Pan (middle-click), zoom (wheel), drag nodes, connect pins
5. Right-click → context menu with add/delete/duplicate
6. Ctrl+Z/Y → undo/redo works
7. Double-click architecture plugin → opens as editable node graph in new tab
8. Hit Run → inference executes on device, results stream to console
9. Node annotations show timing after inference
10. Surgery: delete a layer → re-run → compare benchmarks
11. Test case: create, set expectations, run, see pass/fail
