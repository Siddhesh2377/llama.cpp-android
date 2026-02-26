// Blueprint Engine — Node Templates
// Factory functions for all ML node types used in the palette and canvas.

import type { BpNode, BpPin, NodeCategory, PinType } from './types';

// ---------------------------------------------------------------------------
// ID generation
// ---------------------------------------------------------------------------

let nextId = 1;
export function resetNodeIds() { nextId = 1; }
function uid() { return `n${nextId++}`; }

function pin(name: string, dir: 'in' | 'out', type: PinType, shape?: number[]): BpPin {
	return { id: `${name}-${dir}`, name, direction: dir, dataType: type, shape, connected: false };
}

// ---------------------------------------------------------------------------
// Template interface
// ---------------------------------------------------------------------------

export interface NodeTemplate {
	label: string;
	category: NodeCategory;
	description: string;
	create: (x: number, y: number) => BpNode;
}

// ---------------------------------------------------------------------------
// Helper: base node factory
// ---------------------------------------------------------------------------

function makeTemplate(
	label: string,
	category: NodeCategory,
	description: string,
	pins: BpPin[],
	metadata: Record<string, unknown> = {},
): NodeTemplate {
	return {
		label,
		category,
		description,
		create: (x: number, y: number): BpNode => ({
			id: uid(),
			type: 'op',
			label,
			category,
			position: { x, y },
			size: { w: 180, h: 0 }, // h computed by renderer
			collapsed: false,
			enabled: true,
			metadata: { ...metadata },
			pins: pins.map(p => ({ ...p })),
		}),
	};
}

// ===========================================================================
// MODEL NODES — transformer LLM graph
// ===========================================================================

export const MODEL_NODES: NodeTemplate[] = [
	makeTemplate('Token Embed', 'embed', 'Lookup token embeddings from vocab table', [
		pin('tokens', 'in', 'tokens'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('Pos Embed', 'embed', 'Add positional encoding to embeddings', [
		pin('tokens', 'in', 'tokens'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('Attention', 'attn', 'Multi-head grouped-query attention block', [
		pin('tensor', 'in', 'tensor'),
		pin('kv_cache', 'in', 'kv_cache'),
		pin('tensor', 'out', 'tensor'),
		pin('kv_cache', 'out', 'kv_cache'),
	], { heads: 15, kvHeads: 5, headDim: 64 }),
	makeTemplate('FFN', 'ffn', 'Feed-forward network with gate projection', [
		pin('tensor', 'in', 'tensor'),
		pin('tensor', 'out', 'tensor'),
	], { intermediate: 2560, activation: 'silu', fusedGateUp: true }),
	makeTemplate('RMS Norm', 'norm', 'Root mean square layer normalization', [
		pin('tensor', 'in', 'tensor'),
		pin('eps', 'in', 'scalar'),
		pin('tensor', 'out', 'tensor'),
	], { eps: 1e-5 }),
	makeTemplate('Layer Norm', 'norm', 'Standard layer normalization', [
		pin('tensor', 'in', 'tensor'),
		pin('eps', 'in', 'scalar'),
		pin('tensor', 'out', 'tensor'),
	], { eps: 1e-5 }),
	makeTemplate('LM Head', 'head', 'Linear projection to vocabulary logits', [
		pin('tensor', 'in', 'tensor'),
		pin('logits', 'out', 'tokens'),
	]),
	makeTemplate('Residual Add', 'math', 'Element-wise addition for skip connections', [
		pin('A', 'in', 'tensor'),
		pin('B', 'in', 'tensor'),
		pin('sum', 'out', 'tensor'),
	]),
	makeTemplate('MatMul', 'math', 'Matrix multiplication (A @ B)', [
		pin('A', 'in', 'tensor'),
		pin('B', 'in', 'tensor'),
		pin('C', 'out', 'tensor'),
	]),
	makeTemplate('SiLU', 'math', 'Sigmoid linear unit activation', [
		pin('tensor', 'in', 'tensor'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('GELU', 'math', 'Gaussian error linear unit activation', [
		pin('tensor', 'in', 'tensor'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('Softmax', 'math', 'Softmax normalization along last axis', [
		pin('tensor', 'in', 'tensor'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('RoPE', 'math', 'Rotary positional embedding transform', [
		pin('tensor', 'in', 'tensor'),
		pin('config', 'in', 'config'),
		pin('tensor', 'out', 'tensor'),
	]),
];

// ===========================================================================
// VISION NODES — SigLIP / vision encoder pipeline
// ===========================================================================

export const VISION_NODES: NodeTemplate[] = [
	makeTemplate('Patch Embed', 'embed', 'Extract and embed image patches', [
		pin('image', 'in', 'image'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('Vision Attention', 'attn', 'Multi-head self-attention for vision', [
		pin('tensor', 'in', 'tensor'),
		pin('tensor', 'out', 'tensor'),
	], { heads: 12, headDim: 64 }),
	makeTemplate('Vision FFN', 'ffn', 'Vision encoder feed-forward block', [
		pin('tensor', 'in', 'tensor'),
		pin('tensor', 'out', 'tensor'),
	], { intermediate: 3072, activation: 'gelu' }),
	makeTemplate('Vision Norm', 'norm', 'Vision encoder layer normalization', [
		pin('tensor', 'in', 'tensor'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('Pixel Shuffle', 'math', 'Spatial-to-depth rearrangement', [
		pin('tensor', 'in', 'tensor'),
		pin('scale_factor', 'in', 'config'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('Projector FC', 'math', 'Fully connected vision-to-LLM projection', [
		pin('tensor', 'in', 'tensor'),
		pin('tensor', 'out', 'tensor'),
	]),
];

// ===========================================================================
// SAMPLING NODES — token sampling pipeline
// ===========================================================================

export const SAMPLING_NODES: NodeTemplate[] = [
	makeTemplate('Logit Input', 'head', 'Receive raw logits from LM head', [
		pin('tokens', 'in', 'tokens'),
		pin('tensor', 'out', 'tensor'),
	]),
	makeTemplate('Temperature', 'math', 'Scale logits by temperature', [
		pin('tensor', 'in', 'tensor'),
		pin('temp', 'in', 'scalar'),
		pin('tensor', 'out', 'tensor'),
	], { temperature: 0.8 }),
	makeTemplate('Top-K Filter', 'math', 'Keep only top-K highest logits', [
		pin('tensor', 'in', 'tensor'),
		pin('k', 'in', 'scalar'),
		pin('tensor', 'out', 'tensor'),
	], { k: 40 }),
	makeTemplate('Top-P Filter', 'math', 'Nucleus sampling — cumulative probability cutoff', [
		pin('tensor', 'in', 'tensor'),
		pin('p', 'in', 'scalar'),
		pin('tensor', 'out', 'tensor'),
	], { p: 0.9 }),
	makeTemplate('Min-P Filter', 'math', 'Minimum probability threshold filter', [
		pin('tensor', 'in', 'tensor'),
		pin('p', 'in', 'scalar'),
		pin('tensor', 'out', 'tensor'),
	], { p: 0.05 }),
	makeTemplate('Sampler', 'head', 'Draw token from filtered distribution', [
		pin('tensor', 'in', 'tensor'),
		pin('tokens', 'out', 'tokens'),
	], { method: 'multinomial' }),
];

// ===========================================================================
// HARDWARE NODES — backend / device graph
// ===========================================================================

export const HARDWARE_NODES: NodeTemplate[] = [
	makeTemplate('CPU Core', 'device', 'ARM CPU compute unit with ISA extensions', [
		pin('config', 'in', 'config'),
		pin('device', 'out', 'device'),
	], { freq: '2.5GHz', isa: ['neon', 'i8mm', 'dotprod'] }),
	makeTemplate('GPU CU', 'device', 'GPU shader compute unit (Adreno / Mali)', [
		pin('config', 'in', 'config'),
		pin('device', 'out', 'device'),
	], { shaderCores: 512, clock: '900MHz' }),
	makeTemplate('Memory Pool', 'memory', 'Unified or dedicated memory region', [
		pin('device', 'in', 'device'),
		pin('memory', 'out', 'memory'),
	], { type: 'UMA', sizeMB: 6144, bwGBs: 14 }),
	makeTemplate('Thread Pool', 'device', 'OpenMP thread pool with core affinity', [
		pin('count', 'in', 'scalar'),
		pin('affinity', 'in', 'config'),
		pin('device', 'out', 'device'),
	], { threads: 4, places: '{7},{4},{5},{6}' }),
	makeTemplate('Dispatch', 'device', 'Route ops to CPU or GPU based on rules', [
		pin('cpu', 'in', 'device'),
		pin('gpu', 'in', 'device'),
		pin('rules', 'in', 'config'),
		pin('flow', 'out', 'flow'),
	]),
];

// ===========================================================================
// Combined export
// ===========================================================================

export const ALL_NODES: NodeTemplate[] = [
	...MODEL_NODES,
	...VISION_NODES,
	...SAMPLING_NODES,
	...HARDWARE_NODES,
];
