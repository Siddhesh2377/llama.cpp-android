// Keyboard shortcut handler for canvas operations.
// Call setupShortcuts() in onMount and invoke the returned cleanup in onDestroy.

import { get } from 'svelte/store';
import { activeDocId, activeDoc, selectedNodeIds, selectedEdgeId, updateDocument, undo, redo } from './store';

export interface ShortcutCallbacks {
	onRun?: () => void;
	onStop?: () => void;
	onFitView?: () => void;
	onSearch?: () => void;
}

export function setupShortcuts(callbacks: ShortcutCallbacks = {}): () => void {
	function handler(e: KeyboardEvent) {
		// Don't intercept if user is typing in an input/textarea/select
		const tag = (e.target as HTMLElement)?.tagName;
		if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

		const docId = get(activeDocId);
		const ctrl = e.ctrlKey || e.metaKey;

		// Ctrl+Z — Undo
		if (ctrl && e.key === 'z' && !e.shiftKey) {
			e.preventDefault();
			if (docId) undo(docId);
			return;
		}

		// Ctrl+Y or Ctrl+Shift+Z — Redo
		if ((ctrl && e.key === 'y') || (ctrl && e.key === 'z' && e.shiftKey)) {
			e.preventDefault();
			if (docId) redo(docId);
			return;
		}

		// Delete or Backspace — Delete selected nodes
		if (e.key === 'Delete' || e.key === 'Backspace') {
			e.preventDefault();
			if (!docId) return;
			const selected = get(selectedNodeIds);
			if (selected.size === 0) return;
			updateDocument(docId, doc => {
				doc.nodes = doc.nodes.filter(n => !selected.has(n.id));
				doc.edges = doc.edges.filter(edge => !selected.has(edge.from.nodeId) && !selected.has(edge.to.nodeId));
				return doc;
			});
			selectedNodeIds.set(new Set());
			return;
		}

		// Ctrl+D — Duplicate selected nodes
		if (ctrl && e.key === 'd') {
			e.preventDefault();
			if (!docId) return;
			const selected = get(selectedNodeIds);
			if (selected.size === 0) return;
			const doc = get(activeDoc);
			if (!doc) return;

			const newIds = new Set<string>();
			updateDocument(docId, doc => {
				const toDuplicate = doc.nodes.filter(n => selected.has(n.id));
				for (const node of toDuplicate) {
					const newId = `${node.id}-copy-${Date.now()}`;
					newIds.add(newId);
					doc.nodes.push({
						...structuredClone(node),
						id: newId,
						position: { x: node.position.x + 30, y: node.position.y + 30 },
					});
				}
				return doc;
			});
			selectedNodeIds.set(newIds);
			return;
		}

		// Ctrl+A — Select all nodes
		if (ctrl && e.key === 'a') {
			e.preventDefault();
			const doc = get(activeDoc);
			if (!doc) return;
			selectedNodeIds.set(new Set(doc.nodes.map(n => n.id)));
			return;
		}

		// F — Fit view (call callback)
		if (e.key === 'f' && !ctrl) {
			e.preventDefault();
			callbacks.onFitView?.();
			return;
		}

		// Ctrl+F — Search
		if (ctrl && e.key === 'f') {
			e.preventDefault();
			callbacks.onSearch?.();
			return;
		}

		// F5 — Run
		if (e.key === 'F5' && !e.shiftKey) {
			e.preventDefault();
			callbacks.onRun?.();
			return;
		}

		// Shift+F5 — Stop
		if (e.key === 'F5' && e.shiftKey) {
			e.preventDefault();
			callbacks.onStop?.();
			return;
		}

		// Escape — Deselect all
		if (e.key === 'Escape') {
			selectedNodeIds.set(new Set());
			selectedEdgeId.set(null);
			return;
		}

		// E — Toggle enable/disable selected nodes
		if (e.key === 'e' && !ctrl) {
			if (!docId) return;
			const selected = get(selectedNodeIds);
			if (selected.size === 0) return;
			updateDocument(docId, doc => {
				for (const node of doc.nodes) {
					if (selected.has(node.id)) {
						node.enabled = !node.enabled;
					}
				}
				return doc;
			});
			return;
		}

		// C — Toggle collapse selected nodes
		if (e.key === 'c' && !ctrl) {
			if (!docId) return;
			const selected = get(selectedNodeIds);
			if (selected.size === 0) return;
			updateDocument(docId, doc => {
				for (const node of doc.nodes) {
					if (selected.has(node.id)) {
						node.collapsed = !node.collapsed;
					}
				}
				return doc;
			});
			return;
		}
	}

	document.addEventListener('keydown', handler);
	return () => document.removeEventListener('keydown', handler);
}
