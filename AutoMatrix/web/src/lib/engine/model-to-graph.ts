// Model-to-Graph Converter
// Converts ModelInfo + ModelGraph (from backend API) into a BpDocument for the Blueprint engine.

import type { BpDocument, BpNode, BpEdge, BpPin, NodeCategory, PinType } from './types';

// ---------------------------------------------------------------------------
// Map backend graph_builder node types to Blueprint categories
// ---------------------------------------------------------------------------

function categorize(type: string): NodeCategory {
	if (type === 'embed' || type === 'pos_embed') return 'embed';
	if (type.includes('attn')) return 'attn';
	if (type.includes('ffn')) return 'ffn';
	if (type.includes('norm')) return 'norm';
	if (type === 'head' || type === 'output') return 'head';
	if (type === 'proj' || type === 'projector') return 'math';
	return 'custom';
}

// ---------------------------------------------------------------------------
// Generate typed pins based on node category
// ---------------------------------------------------------------------------

function pinsFor(category: NodeCategory): BpPin[] {
	switch (category) {
		case 'embed':
			return [
				{ id: 'tokens-in', name: 'tokens', direction: 'in', dataType: 'tokens', connected: false },
				{ id: 'embed-out', name: 'embed', direction: 'out', dataType: 'tensor', connected: false },
			];
		case 'attn':
			return [
				{ id: 'input-in', name: 'input', direction: 'in', dataType: 'tensor', connected: false },
				{ id: 'kv-in', name: 'kv_cache', direction: 'in', dataType: 'kv_cache', connected: false },
				{ id: 'output-out', name: 'output', direction: 'out', dataType: 'tensor', connected: false },
				{ id: 'kv-out', name: 'kv_out', direction: 'out', dataType: 'kv_cache', connected: false },
			];
		case 'ffn':
			return [
				{ id: 'input-in', name: 'input', direction: 'in', dataType: 'tensor', connected: false },
				{ id: 'output-out', name: 'output', direction: 'out', dataType: 'tensor', connected: false },
			];
		case 'norm':
			return [
				{ id: 'input-in', name: 'input', direction: 'in', dataType: 'tensor', connected: false },
				{ id: 'output-out', name: 'output', direction: 'out', dataType: 'tensor', connected: false },
			];
		case 'head':
			return [
				{ id: 'input-in', name: 'input', direction: 'in', dataType: 'tensor', connected: false },
				{ id: 'logits-out', name: 'logits', direction: 'out', dataType: 'tokens', connected: false },
			];
		default:
			return [
				{ id: 'input-in', name: 'input', direction: 'in', dataType: 'tensor', connected: false },
				{ id: 'output-out', name: 'output', direction: 'out', dataType: 'tensor', connected: false },
			];
	}
}

// ---------------------------------------------------------------------------
// Main converter
// ---------------------------------------------------------------------------

export function modelToBpDocument(
	modelInfo: { name?: string; arch: string; quantType?: string; embeddingSize?: number; headCount?: number; headCountKV?: number },
	graph: { nodes: Array<{ id: string; label: string; type: string; sublabel: string; layer: number; x: number; y: number; width: number; height: number }>; edges: Array<{ from: string; to: string }>; layerCount: number; arch: string }
): BpDocument {
	const nodes: BpNode[] = graph.nodes.map(gn => {
		const category = categorize(gn.type);
		const pins = pinsFor(category);
		return {
			id: gn.id,
			type: 'op',
			label: gn.label,
			category,
			position: { x: gn.x || 60, y: gn.y || 0 },
			size: { w: gn.width || 180, h: 0 },
			collapsed: false,
			enabled: true,
			metadata: {
				sublabel: gn.sublabel,
				layer: gn.layer,
				backendType: gn.type,
				quant: modelInfo.quantType || 'q8_0',
			},
			pins,
		};
	});

	// Build edges: connect nodes using first output pin -> first input pin
	const edges: BpEdge[] = graph.edges.map((ge, i) => {
		const fromNode = nodes.find(n => n.id === ge.from);
		const toNode = nodes.find(n => n.id === ge.to);
		const fromPin = fromNode?.pins.find(p => p.direction === 'out')?.id || 'output-out';
		const toPin = toNode?.pins.find(p => p.direction === 'in')?.id || 'input-in';

		// Mark pins as connected
		if (fromNode) {
			const fp = fromNode.pins.find(p => p.id === fromPin);
			if (fp) fp.connected = true;
		}
		if (toNode) {
			const tp = toNode.pins.find(p => p.id === toPin);
			if (tp) tp.connected = true;
		}

		return {
			id: `e${i}`,
			from: { nodeId: ge.from, pinId: fromPin },
			to: { nodeId: ge.to, pinId: toPin },
			valid: true,
		};
	});

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
