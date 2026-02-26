// Blueprint Engine — Core Types
// All TypeScript types and constants for the node graph system.

// ---------------------------------------------------------------------------
// Pin types — typed data flowing between nodes
// ---------------------------------------------------------------------------

export type PinType =
	| 'tensor'
	| 'image'
	| 'tokens'
	| 'kv_cache'
	| 'scalar'
	| 'math'
	| 'config'
	| 'flow'
	| 'memory'
	| 'device'
	| 'stream'
	| 'binary';

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

// ---------------------------------------------------------------------------
// Node categories
// ---------------------------------------------------------------------------

export type NodeCategory =
	| 'embed'
	| 'attn'
	| 'ffn'
	| 'norm'
	| 'head'
	| 'math'
	| 'device'
	| 'memory'
	| 'custom';

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

// ---------------------------------------------------------------------------
// Core interfaces
// ---------------------------------------------------------------------------

export interface BpPin {
	id: string;
	name: string;
	direction: 'in' | 'out';
	dataType: PinType;
	shape?: number[];       // tensor dimensions e.g. [960, 64]
	dtype?: string;         // "f16", "q8_0", "q5_0"
	connected: boolean;
	value?: unknown;        // for scalars/config
}

export interface BpNode {
	id: string;
	type: string;           // "op", "input", "output", "group", "comment"
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
