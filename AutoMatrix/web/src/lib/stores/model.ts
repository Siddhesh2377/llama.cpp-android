import { writable, derived } from 'svelte/store';

export interface TensorInfo {
	name: string;
	shape: number[];
	type: string;
	size: number;
}

export interface ModelLayer {
	index: number;
	name: string;
	tensors: TensorInfo[];
}

export interface ModelInfo {
	filename: string;
	filepath: string;
	arch: string;
	name: string;
	quantType: string;
	params: string;
	contextLength: number;
	embeddingSize: number;
	layerCount: number;
	headCount: number;
	headCountKV: number;
	vocabSize: number;
	fileSize: number;
	tensorCount: number;
	metadata: Record<string, string | number | boolean>;
	layers: ModelLayer[];
	tensors: TensorInfo[];
}

export const model = writable<ModelInfo | null>(null);
export const modelLoading = writable(false);
export const modelError = writable<string | null>(null);

export const hasModel = derived(model, $m => $m !== null);

export function formatBytes(bytes: number): string {
	if (bytes < 1024) return bytes + ' B';
	if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
	if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
	return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}
