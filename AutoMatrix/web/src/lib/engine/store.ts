// Blueprint Engine — Document Management Store
// Manages open documents, selection, undo/redo, and canvas interaction state.

import { writable, derived, get } from 'svelte/store';
import type { BpDocument } from './types';

// ---------------------------------------------------------------------------
// Document stores
// ---------------------------------------------------------------------------

export const documents = writable<BpDocument[]>([]);
export const activeDocId = writable<string | null>(null);

export const activeDoc = derived(
	[documents, activeDocId],
	([$documents, $activeDocId]) =>
		$activeDocId ? $documents.find(d => d.id === $activeDocId) ?? null : null
);

// ---------------------------------------------------------------------------
// Undo / Redo stacks — per document, max 100 entries
// ---------------------------------------------------------------------------

const MAX_HISTORY = 100;
const undoStacks = new Map<string, BpDocument[]>();
const redoStacks = new Map<string, BpDocument[]>();

function pushUndo(docId: string, snapshot: BpDocument): void {
	let stack = undoStacks.get(docId);
	if (!stack) {
		stack = [];
		undoStacks.set(docId, stack);
	}
	stack.push(snapshot);
	if (stack.length > MAX_HISTORY) {
		stack.shift();
	}
	// Any new change clears the redo stack
	redoStacks.set(docId, []);
}

// ---------------------------------------------------------------------------
// Document operations
// ---------------------------------------------------------------------------

export function openDocument(doc: BpDocument): void {
	documents.update(docs => {
		const idx = docs.findIndex(d => d.id === doc.id);
		if (idx >= 0) {
			docs[idx] = doc;
			return [...docs];
		}
		return [...docs, doc];
	});
	activeDocId.set(doc.id);
}

export function closeDocument(docId: string): void {
	undoStacks.delete(docId);
	redoStacks.delete(docId);

	documents.update(docs => {
		const filtered = docs.filter(d => d.id !== docId);
		return filtered;
	});

	// Switch to the last remaining document, or null
	const remaining = get(documents);
	if (get(activeDocId) === docId) {
		activeDocId.set(remaining.length > 0 ? remaining[remaining.length - 1].id : null);
	}
}

export function updateDocument(
	docId: string,
	updater: (doc: BpDocument) => BpDocument
): void {
	documents.update(docs => {
		const idx = docs.findIndex(d => d.id === docId);
		if (idx < 0) return docs;

		// Snapshot current state for undo
		pushUndo(docId, structuredClone(docs[idx]));

		const updated = updater(structuredClone(docs[idx]));
		updated.dirty = true;
		docs[idx] = updated;
		return [...docs];
	});
}

export function undo(docId: string): void {
	const stack = undoStacks.get(docId);
	if (!stack || stack.length === 0) return;

	documents.update(docs => {
		const idx = docs.findIndex(d => d.id === docId);
		if (idx < 0) return docs;

		// Push current state to redo
		let redo = redoStacks.get(docId);
		if (!redo) {
			redo = [];
			redoStacks.set(docId, redo);
		}
		redo.push(structuredClone(docs[idx]));
		if (redo.length > MAX_HISTORY) {
			redo.shift();
		}

		// Pop undo
		docs[idx] = stack.pop()!;
		return [...docs];
	});
}

export function redo(docId: string): void {
	const stack = redoStacks.get(docId);
	if (!stack || stack.length === 0) return;

	documents.update(docs => {
		const idx = docs.findIndex(d => d.id === docId);
		if (idx < 0) return docs;

		// Push current state to undo
		let undoStack = undoStacks.get(docId);
		if (!undoStack) {
			undoStack = [];
			undoStacks.set(docId, undoStack);
		}
		undoStack.push(structuredClone(docs[idx]));
		if (undoStack.length > MAX_HISTORY) {
			undoStack.shift();
		}

		// Pop redo
		docs[idx] = stack.pop()!;
		return [...docs];
	});
}

// ---------------------------------------------------------------------------
// Selection state
// ---------------------------------------------------------------------------

export const selectedNodeIds = writable<Set<string>>(new Set());
export const selectedEdgeId = writable<string | null>(null);

// ---------------------------------------------------------------------------
// Canvas interaction state
// ---------------------------------------------------------------------------

export interface CanvasInteraction {
	mode: 'idle' | 'pan' | 'drag-node' | 'drag-wire' | 'box-select';
	origin: { x: number; y: number } | null;
	wireFrom: { nodeId: string; pinId: string } | null;
}

export const canvasInteraction = writable<CanvasInteraction>({
	mode: 'idle',
	origin: null,
	wireFrom: null,
});
