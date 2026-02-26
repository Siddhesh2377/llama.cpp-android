<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { get } from 'svelte/store';
	import { theme } from '$lib/stores/settings';
	import {
		activeDoc,
		activeDocId,
		selectedNodeIds,
		selectedEdgeId,
		canvasInteraction,
		updateDocument,
		type CanvasInteraction,
	} from './store';
	import {
		CATEGORY_COLORS,
		type BpNode,
		type BpEdge,
		type BpDocument,
		type PinType,
		type NodeCategory,
	} from './types';
	import NodeRenderer from './NodeRenderer.svelte';
	import EdgeRenderer from './EdgeRenderer.svelte';

	// -----------------------------------------------------------------------
	// State
	// -----------------------------------------------------------------------

	let zoom = $state(1);
	let panX = $state(0);
	let panY = $state(0);
	let containerEl: HTMLDivElement | undefined = $state(undefined);

	let contextMenu: { x: number; y: number; nodeId?: string } | null = $state(null);
	let wireEndPos: { x: number; y: number } | null = $state(null);
	let boxSelectRect = $state<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

	// Drag tracking (not reactive, just cached values)
	let dragStartMouse = { x: 0, y: 0 };
	let dragStartPan = { x: 0, y: 0 };
	let dragNodeStart: { x: number; y: number } | null = null;
	let dragNodeId: string | null = null;
	let wireFromNodeId: string | null = null;
	let wireFromPinId: string | null = null;

	// Subscriptions — store raw values, not store objects
	let currentDoc: BpDocument | null = $state<BpDocument | null>(null);
	let currentTheme: string = $state('dark');
	let currentSelectedIds: Set<string> = $state(new Set());
	let currentInteraction: CanvasInteraction = $state<CanvasInteraction>({
		mode: 'idle',
		origin: null,
		wireFrom: null,
	});

	const isDark = $derived(currentTheme === 'dark');
	const nodes = $derived<BpNode[]>(currentDoc?.nodes ?? []);
	const edges = $derived<BpEdge[]>(currentDoc?.edges ?? []);

	// -----------------------------------------------------------------------
	// Store subscriptions
	// -----------------------------------------------------------------------

	const unsubDoc = activeDoc.subscribe((d: BpDocument | null) => (currentDoc = d));
	const unsubTheme = theme.subscribe((t: string) => (currentTheme = t));
	const unsubSelected = selectedNodeIds.subscribe((s: Set<string>) => (currentSelectedIds = s));
	const unsubInteraction = canvasInteraction.subscribe((i: CanvasInteraction) => (currentInteraction = i));

	onDestroy(() => {
		unsubDoc();
		unsubTheme();
		unsubSelected();
		unsubInteraction();
	});

	// -----------------------------------------------------------------------
	// Pin position helper
	// -----------------------------------------------------------------------

	const HEADER_H = 26;
	const PIN_SPACING = 22;

	function getNodeW(node: BpNode): number {
		return node.size.w || 180;
	}

	function getNodeH(node: BpNode): number {
		const inPins = node.pins.filter((p) => p.direction === 'in').length;
		const outPins = node.pins.filter((p) => p.direction === 'out').length;
		const pinCount = Math.max(inPins, outPins);
		return node.collapsed ? HEADER_H : HEADER_H + pinCount * PIN_SPACING + 8;
	}

	function getPinPosition(
		node: BpNode,
		pinId: string
	): { x: number; y: number } | null {
		const pin = node.pins.find((p) => p.id === pinId);
		if (!pin) return null;

		const isInput = pin.direction === 'in';
		const sameDirPins = node.pins.filter((p) => p.direction === pin.direction);
		const idx = sameDirPins.indexOf(pin);

		const x = isInput ? node.position.x : node.position.x + getNodeW(node);
		const y = node.position.y + HEADER_H + idx * PIN_SPACING + 12;
		return { x, y };
	}

	// Find which pin type an edge carries (from the source pin)
	function getEdgePinType(edge: BpEdge): PinType {
		const fromNode = nodes.find((n: BpNode) => n.id === edge.from.nodeId);
		if (!fromNode) return 'flow';
		const pin = fromNode.pins.find((p) => p.id === edge.from.pinId);
		return pin?.dataType ?? 'flow';
	}

	// -----------------------------------------------------------------------
	// Wheel zoom (needs non-passive listener)
	// -----------------------------------------------------------------------

	let wheelHandler: ((e: WheelEvent) => void) | null = null;

	onMount(() => {
		if (!containerEl) return;

		// Sync viewport from document
		if (currentDoc) {
			panX = currentDoc.viewport.x;
			panY = currentDoc.viewport.y;
			zoom = currentDoc.viewport.zoom;
		}

		wheelHandler = (e: WheelEvent) => {
			e.preventDefault();
			const rect = containerEl!.getBoundingClientRect();
			const mx = e.clientX - rect.left;
			const my = e.clientY - rect.top;

			const oldZoom = zoom;
			const factor = e.deltaY > 0 ? 0.9 : 1.1;
			const newZoom = Math.max(0.1, Math.min(5, oldZoom * factor));

			// Zoom toward cursor
			panX = mx - ((mx - panX) * newZoom) / oldZoom;
			panY = my - ((my - panY) * newZoom) / oldZoom;
			zoom = newZoom;
		};

		containerEl.addEventListener('wheel', wheelHandler, { passive: false });
	});

	onDestroy(() => {
		if (containerEl && wheelHandler) {
			containerEl.removeEventListener('wheel', wheelHandler);
		}
	});

	// -----------------------------------------------------------------------
	// Mouse to canvas coordinate conversion
	// -----------------------------------------------------------------------

	function clientToCanvas(clientX: number, clientY: number): { x: number; y: number } {
		const rect = containerEl?.getBoundingClientRect();
		if (!rect) return { x: 0, y: 0 };
		return {
			x: (clientX - rect.left - panX) / zoom,
			y: (clientY - rect.top - panY) / zoom,
		};
	}

	// -----------------------------------------------------------------------
	// Hit testing
	// -----------------------------------------------------------------------

	function hitTestNode(canvasPos: { x: number; y: number }): BpNode | null {
		for (let i = nodes.length - 1; i >= 0; i--) {
			const n = nodes[i];
			const nw = getNodeW(n);
			const nh = getNodeH(n);

			if (
				canvasPos.x >= n.position.x &&
				canvasPos.x <= n.position.x + nw &&
				canvasPos.y >= n.position.y &&
				canvasPos.y <= n.position.y + nh
			) {
				return n;
			}
		}
		return null;
	}

	// -----------------------------------------------------------------------
	// Interaction handlers
	// -----------------------------------------------------------------------

	function handleCanvasMouseDown(e: MouseEvent) {
		if (e.button === 2) return; // right-click handled by context menu
		if (e.button !== 0) return;

		contextMenu = null;

		const canvasPos = clientToCanvas(e.clientX, e.clientY);
		const clickedNode = hitTestNode(canvasPos);

		if (clickedNode) {
			// Node click: select and start drag
			if (e.shiftKey) {
				selectedNodeIds.update((s: Set<string>) => {
					const next = new Set(s);
					if (next.has(clickedNode.id)) {
						next.delete(clickedNode.id);
					} else {
						next.add(clickedNode.id);
					}
					return next;
				});
			} else if (!currentSelectedIds.has(clickedNode.id)) {
				selectedNodeIds.set(new Set([clickedNode.id]));
			}
			selectedEdgeId.set(null);

			// Start node drag
			dragNodeId = clickedNode.id;
			dragNodeStart = { ...clickedNode.position };
			dragStartMouse = { x: e.clientX, y: e.clientY };

			canvasInteraction.set({
				mode: 'drag-node',
				origin: { x: e.clientX, y: e.clientY },
				wireFrom: null,
			});
		} else {
			// Empty canvas click: deselect and start pan
			selectedNodeIds.set(new Set());
			selectedEdgeId.set(null);

			dragStartMouse = { x: e.clientX, y: e.clientY };
			dragStartPan = { x: panX, y: panY };

			canvasInteraction.set({
				mode: 'pan',
				origin: { x: e.clientX, y: e.clientY },
				wireFrom: null,
			});
		}
	}

	function handleCanvasMouseMove(e: MouseEvent) {
		const mode = currentInteraction.mode;

		if (mode === 'pan') {
			const dx = e.clientX - dragStartMouse.x;
			const dy = e.clientY - dragStartMouse.y;
			panX = dragStartPan.x + dx;
			panY = dragStartPan.y + dy;
		} else if (mode === 'drag-node' && dragNodeId && dragNodeStart) {
			const dx = (e.clientX - dragStartMouse.x) / zoom;
			const dy = (e.clientY - dragStartMouse.y) / zoom;

			const docId = get(activeDocId);
			if (docId) {
				// Direct mutation for smooth dragging
				const doc = get(activeDoc);
				if (doc) {
					const node = doc.nodes.find((n: BpNode) => n.id === dragNodeId);
					if (node) {
						node.position.x = dragNodeStart.x + dx;
						node.position.y = dragNodeStart.y + dy;
						// Force re-render
						currentDoc = { ...doc };
					}
				}
			}
		} else if (mode === 'drag-wire') {
			const canvasPos = clientToCanvas(e.clientX, e.clientY);
			wireEndPos = canvasPos;
		}
	}

	function handleCanvasMouseUp(_e: MouseEvent) {
		const mode = currentInteraction.mode;

		if (mode === 'drag-node' && dragNodeId && dragNodeStart) {
			// Commit the final position via updateDocument
			const finalDx = (_e.clientX - dragStartMouse.x) / zoom;
			const finalDy = (_e.clientY - dragStartMouse.y) / zoom;
			const nid = dragNodeId;
			const startPos = { ...dragNodeStart };

			const docId = get(activeDocId);
			if (docId) {
				updateDocument(docId, (doc: BpDocument) => {
					const node = doc.nodes.find((n: BpNode) => n.id === nid);
					if (node) {
						node.position.x = startPos.x + finalDx;
						node.position.y = startPos.y + finalDy;
					}
					return doc;
				});
			}
		} else if (mode === 'drag-wire') {
			// Wire end without hitting a pin: cancel
			wireEndPos = null;
			wireFromNodeId = null;
			wireFromPinId = null;
		}

		dragNodeId = null;
		dragNodeStart = null;

		canvasInteraction.set({
			mode: 'idle',
			origin: null,
			wireFrom: null,
		});
	}

	function handleContextMenu(e: MouseEvent) {
		e.preventDefault();
		const canvasPos = clientToCanvas(e.clientX, e.clientY);
		const hitNode = hitTestNode(canvasPos);

		contextMenu = { x: e.clientX, y: e.clientY, nodeId: hitNode?.id };

		if (hitNode && !currentSelectedIds.has(hitNode.id)) {
			selectedNodeIds.set(new Set([hitNode.id]));
		}
	}

	function closeContextMenu() {
		contextMenu = null;
	}

	// -----------------------------------------------------------------------
	// Wire start / end callbacks (passed to NodeRenderer -> PinRenderer)
	// -----------------------------------------------------------------------

	function handleStartWire(nodeId: string, pinId: string) {
		wireFromNodeId = nodeId;
		wireFromPinId = pinId;

		const node = nodes.find((n: BpNode) => n.id === nodeId);
		if (node) {
			const pos = getPinPosition(node, pinId);
			if (pos) {
				wireEndPos = { ...pos };
			}
		}

		canvasInteraction.set({
			mode: 'drag-wire',
			origin: null,
			wireFrom: { nodeId, pinId },
		});
	}

	function handleEndWire(nodeId: string, pinId: string) {
		if (currentInteraction.mode !== 'drag-wire') return;
		if (!wireFromNodeId || !wireFromPinId) return;

		// Don't connect to same node
		if (wireFromNodeId === nodeId) {
			cancelWire();
			return;
		}

		// Type checking
		const fromNode = nodes.find((n: BpNode) => n.id === wireFromNodeId);
		const toNode = nodes.find((n: BpNode) => n.id === nodeId);
		if (!fromNode || !toNode) {
			cancelWire();
			return;
		}

		const fromPin = fromNode.pins.find((p) => p.id === wireFromPinId);
		const toPin = toNode.pins.find((p) => p.id === pinId);
		if (!fromPin || !toPin) {
			cancelWire();
			return;
		}

		// Validate: output -> input
		if (fromPin.direction !== 'out' || toPin.direction !== 'in') {
			cancelWire();
			return;
		}

		// Type compatibility: must match, or one is 'flow'
		const compatible =
			fromPin.dataType === toPin.dataType ||
			fromPin.dataType === 'flow' ||
			toPin.dataType === 'flow';

		const docId = get(activeDocId);
		if (docId) {
			const edgeId = `e_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
			const fromNId = wireFromNodeId;
			const fromPId = wireFromPinId;
			updateDocument(docId, (doc: BpDocument) => {
				// Remove existing edge to the same input pin
				doc.edges = doc.edges.filter(
					(e: BpEdge) => !(e.to.nodeId === nodeId && e.to.pinId === pinId)
				);

				doc.edges.push({
					id: edgeId,
					from: { nodeId: fromNId, pinId: fromPId },
					to: { nodeId, pinId },
					valid: compatible,
				});

				// Mark pins as connected
				const fn = doc.nodes.find((n: BpNode) => n.id === fromNId);
				const tn = doc.nodes.find((n: BpNode) => n.id === nodeId);
				if (fn) {
					const fp = fn.pins.find((p) => p.id === fromPId);
					if (fp) fp.connected = true;
				}
				if (tn) {
					const tp = tn.pins.find((p) => p.id === pinId);
					if (tp) tp.connected = true;
				}

				return doc;
			});
		}

		cancelWire();
	}

	function cancelWire() {
		wireEndPos = null;
		wireFromNodeId = null;
		wireFromPinId = null;
		canvasInteraction.set({ mode: 'idle', origin: null, wireFrom: null });
	}

	// -----------------------------------------------------------------------
	// Context menu actions
	// -----------------------------------------------------------------------

	function ctxDuplicate() {
		if (!contextMenu?.nodeId) return;
		const docId = get(activeDocId);
		if (!docId) return;

		const nid = contextMenu.nodeId;
		updateDocument(docId, (doc: BpDocument) => {
			const orig = doc.nodes.find((n: BpNode) => n.id === nid);
			if (!orig) return doc;

			const newNode: BpNode = structuredClone(orig);
			newNode.id = `n_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
			newNode.position = {
				x: orig.position.x + 30,
				y: orig.position.y + 30,
			};
			// Reset pin connections
			newNode.pins.forEach((p) => {
				p.id = `${newNode.id}_${p.name}`;
				p.connected = false;
			});
			doc.nodes.push(newNode);
			return doc;
		});
		closeContextMenu();
	}

	function ctxDelete() {
		if (!contextMenu?.nodeId) return;
		const docId = get(activeDocId);
		if (!docId) return;

		const nid = contextMenu.nodeId;
		updateDocument(docId, (doc: BpDocument) => {
			doc.edges = doc.edges.filter(
				(e: BpEdge) => e.from.nodeId !== nid && e.to.nodeId !== nid
			);
			doc.nodes = doc.nodes.filter((n: BpNode) => n.id !== nid);
			return doc;
		});
		selectedNodeIds.update((s: Set<string>) => {
			const next = new Set(s);
			next.delete(nid);
			return next;
		});
		closeContextMenu();
	}

	function ctxToggleEnable() {
		if (!contextMenu?.nodeId) return;
		const docId = get(activeDocId);
		if (!docId) return;

		const nid = contextMenu.nodeId;
		updateDocument(docId, (doc: BpDocument) => {
			const node = doc.nodes.find((n: BpNode) => n.id === nid);
			if (node) node.enabled = !node.enabled;
			return doc;
		});
		closeContextMenu();
	}

	function ctxFitView() {
		if (nodes.length === 0) {
			panX = 0;
			panY = 0;
			zoom = 1;
			closeContextMenu();
			return;
		}

		const rect = containerEl?.getBoundingClientRect();
		if (!rect) {
			closeContextMenu();
			return;
		}

		let minX = Infinity,
			minY = Infinity,
			maxX = -Infinity,
			maxY = -Infinity;
		for (const n of nodes) {
			const nw = getNodeW(n);
			const nh = getNodeH(n);
			minX = Math.min(minX, n.position.x);
			minY = Math.min(minY, n.position.y);
			maxX = Math.max(maxX, n.position.x + nw);
			maxY = Math.max(maxY, n.position.y + nh);
		}

		const contentW = maxX - minX + 80;
		const contentH = maxY - minY + 80;
		const scaleX = rect.width / contentW;
		const scaleY = rect.height / contentH;
		zoom = Math.min(scaleX, scaleY, 2);
		panX = (rect.width - contentW * zoom) / 2 - minX * zoom + 40 * zoom;
		panY = (rect.height - contentH * zoom) / 2 - minY * zoom + 40 * zoom;
		closeContextMenu();
	}

	function ctxAutoLayout() {
		const docId = get(activeDocId);
		if (!docId || nodes.length === 0) {
			closeContextMenu();
			return;
		}

		updateDocument(docId, (doc: BpDocument) => {
			const cols = Math.ceil(Math.sqrt(doc.nodes.length));
			doc.nodes.forEach((n: BpNode, i: number) => {
				const col = i % cols;
				const row = Math.floor(i / cols);
				n.position.x = col * 240 + 40;
				n.position.y = row * 180 + 40;
			});
			return doc;
		});
		closeContextMenu();
	}

	// -----------------------------------------------------------------------
	// Grid dots pattern
	// -----------------------------------------------------------------------

	const gridSize = 20;

	// -----------------------------------------------------------------------
	// Minimap
	// -----------------------------------------------------------------------

	const MINIMAP_W = 150;
	const MINIMAP_H = 100;

	interface MinimapBounds {
		minX: number;
		minY: number;
		maxX: number;
		maxY: number;
		scale: number;
	}

	function getMinimapBounds(): MinimapBounds {
		if (nodes.length === 0) {
			return { minX: 0, minY: 0, maxX: 1000, maxY: 600, scale: 0.15 };
		}

		let minX = Infinity,
			minY = Infinity,
			maxX = -Infinity,
			maxY = -Infinity;
		for (const n of nodes) {
			const nw = getNodeW(n);
			const nh = getNodeH(n);
			minX = Math.min(minX, n.position.x);
			minY = Math.min(minY, n.position.y);
			maxX = Math.max(maxX, n.position.x + nw);
			maxY = Math.max(maxY, n.position.y + nh);
		}

		// Add padding
		const pad = 50;
		minX -= pad;
		minY -= pad;
		maxX += pad;
		maxY += pad;

		const contentW = maxX - minX || 1;
		const contentH = maxY - minY || 1;
		const scaleX = MINIMAP_W / contentW;
		const scaleY = MINIMAP_H / contentH;
		const scale = Math.min(scaleX, scaleY);

		return { minX, minY, maxX, maxY, scale };
	}

	function handleMinimapClick(e: MouseEvent) {
		const target = e.currentTarget as HTMLElement;
		const rect = target.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;
		const bounds = getMinimapBounds();

		// Convert minimap click to canvas coordinates
		const canvasX = mx / bounds.scale + bounds.minX;
		const canvasY = my / bounds.scale + bounds.minY;

		// Center viewport on that point
		const containerRect = containerEl?.getBoundingClientRect();
		if (!containerRect) return;

		panX = containerRect.width / 2 - canvasX * zoom;
		panY = containerRect.height / 2 - canvasY * zoom;
	}

	function getMinimapViewport(): { x: number; y: number; w: number; h: number } {
		const bounds = getMinimapBounds();
		const containerRect = containerEl?.getBoundingClientRect();
		if (!containerRect) return { x: 0, y: 0, w: MINIMAP_W, h: MINIMAP_H };

		const viewLeft = -panX / zoom;
		const viewTop = -panY / zoom;
		const viewW = containerRect.width / zoom;
		const viewH = containerRect.height / zoom;

		return {
			x: (viewLeft - bounds.minX) * bounds.scale,
			y: (viewTop - bounds.minY) * bounds.scale,
			w: viewW * bounds.scale,
			h: viewH * bounds.scale,
		};
	}

	// -----------------------------------------------------------------------
	// Minimap node data (computed to avoid @const in template)
	// -----------------------------------------------------------------------

	interface MinimapNode {
		id: string;
		x: number;
		y: number;
		w: number;
		h: number;
		accent: string;
	}

	const minimapNodes: MinimapNode[] = $derived.by(() => {
		const bounds = getMinimapBounds();
		return nodes.map((node: BpNode) => {
			const catColors = CATEGORY_COLORS[node.category] ?? CATEGORY_COLORS.custom;
			const accent = isDark ? catColors.darkAccent : catColors.lightAccent;
			const nw = getNodeW(node);
			const nh = getNodeH(node);
			return {
				id: node.id,
				x: (node.position.x - bounds.minX) * bounds.scale,
				y: (node.position.y - bounds.minY) * bounds.scale,
				w: nw * bounds.scale,
				h: nh * bounds.scale,
				accent,
			};
		});
	});

	const minimapViewport = $derived(getMinimapViewport());

	// -----------------------------------------------------------------------
	// Temp wire rendering
	// -----------------------------------------------------------------------

	function getWireStartPos(): { x: number; y: number } | null {
		if (!wireFromNodeId || !wireFromPinId) return null;
		const node = nodes.find((n: BpNode) => n.id === wireFromNodeId);
		if (!node) return null;
		return getPinPosition(node, wireFromPinId);
	}

	function getWireDataType(): PinType {
		if (!wireFromNodeId || !wireFromPinId) return 'flow';
		const node = nodes.find((n: BpNode) => n.id === wireFromNodeId);
		if (!node) return 'flow';
		const pin = node.pins.find((p) => p.id === wireFromPinId);
		return pin?.dataType ?? 'flow';
	}

	// -----------------------------------------------------------------------
	// Edge data (resolved positions for rendering)
	// -----------------------------------------------------------------------

	interface ResolvedEdge {
		id: string;
		x1: number;
		y1: number;
		x2: number;
		y2: number;
		dataType: PinType;
		valid: boolean;
	}

	const resolvedEdges: ResolvedEdge[] = $derived.by(() => {
		const result: ResolvedEdge[] = [];
		for (const edge of edges) {
			const fromNode = nodes.find((n: BpNode) => n.id === edge.from.nodeId);
			const toNode = nodes.find((n: BpNode) => n.id === edge.to.nodeId);
			if (!fromNode || !toNode) continue;
			const fromPos = getPinPosition(fromNode, edge.from.pinId);
			const toPos = getPinPosition(toNode, edge.to.pinId);
			if (!fromPos || !toPos) continue;
			result.push({
				id: edge.id,
				x1: fromPos.x,
				y1: fromPos.y,
				x2: toPos.x,
				y2: toPos.y,
				dataType: getEdgePinType(edge),
				valid: edge.valid,
			});
		}
		return result;
	});
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	class="canvas-container"
	bind:this={containerEl}
	onmousedown={handleCanvasMouseDown}
	onmousemove={handleCanvasMouseMove}
	onmouseup={handleCanvasMouseUp}
	oncontextmenu={handleContextMenu}
	onclick={() => { if (contextMenu) closeContextMenu(); }}
	role="application"
	aria-label="Blueprint canvas"
>
	<svg width="100%" height="100%">
		<!-- Grid dots pattern -->
		<defs>
			<pattern
				id="grid-dots"
				x={panX % (gridSize * zoom)}
				y={panY % (gridSize * zoom)}
				width={gridSize * zoom}
				height={gridSize * zoom}
				patternUnits="userSpaceOnUse"
			>
				<circle
					cx={1}
					cy={1}
					r={0.8}
					fill="var(--text-tertiary)"
					opacity={0.4}
				/>
			</pattern>
		</defs>
		<rect width="100%" height="100%" fill="url(#grid-dots)" />

		<g transform="translate({panX},{panY}) scale({zoom})">
			<!-- Edges -->
			{#each resolvedEdges as edge (edge.id)}
				<EdgeRenderer
					x1={edge.x1}
					y1={edge.y1}
					x2={edge.x2}
					y2={edge.y2}
					dataType={edge.dataType}
					valid={edge.valid}
				/>
			{/each}

			<!-- Temp wire during drag -->
			{#if currentInteraction.mode === 'drag-wire' && wireEndPos}
				{@const startPos = getWireStartPos()}
				{#if startPos}
					<EdgeRenderer
						x1={startPos.x}
						y1={startPos.y}
						x2={wireEndPos.x}
						y2={wireEndPos.y}
						dataType={getWireDataType()}
						valid={true}
					/>
				{/if}
			{/if}

			<!-- Box select rect (placeholder) -->
			{#if boxSelectRect}
				<rect
					x={Math.min(boxSelectRect.x1, boxSelectRect.x2)}
					y={Math.min(boxSelectRect.y1, boxSelectRect.y2)}
					width={Math.abs(boxSelectRect.x2 - boxSelectRect.x1)}
					height={Math.abs(boxSelectRect.y2 - boxSelectRect.y1)}
					fill="var(--accent-dim)"
					stroke="var(--accent)"
					stroke-width={1}
					stroke-dasharray="4 2"
				/>
			{/if}

			<!-- Nodes -->
			{#each nodes as node (node.id)}
				<NodeRenderer
					{node}
					selected={currentSelectedIds.has(node.id)}
					{isDark}
					onstartWire={handleStartWire}
					onendWire={handleEndWire}
				/>
			{/each}
		</g>
	</svg>

	<!-- Context menu (HTML overlay) -->
	{#if contextMenu}
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="context-menu"
			style="left: {contextMenu.x}px; top: {contextMenu.y}px;"
			onmousedown={(e: MouseEvent) => e.stopPropagation()}
		>
			<button onclick={() => { closeContextMenu(); }}>Add Node...</button>
			{#if contextMenu.nodeId}
				<button onclick={ctxDuplicate}>Duplicate</button>
				<button onclick={ctxDelete}>Delete</button>
				<button onclick={ctxToggleEnable}>Toggle Enable</button>
				<div class="context-separator"></div>
			{/if}
			<button onclick={ctxFitView}>Fit View</button>
			<button onclick={ctxAutoLayout}>Auto Layout</button>
		</div>
	{/if}

	<!-- Minimap -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="minimap" onmousedown={handleMinimapClick}>
		<svg width={MINIMAP_W} height={MINIMAP_H}>
			{#each minimapNodes as mn (mn.id)}
				<rect
					x={mn.x}
					y={mn.y}
					width={mn.w}
					height={mn.h}
					fill={mn.accent}
					opacity={0.7}
					rx={1}
				/>
			{/each}
			<!-- Viewport rect -->
			<rect
				x={minimapViewport.x}
				y={minimapViewport.y}
				width={minimapViewport.w}
				height={minimapViewport.h}
				fill="none"
				stroke="var(--accent)"
				stroke-width={1}
				opacity={0.6}
			/>
		</svg>
	</div>
</div>

<style>
	.canvas-container {
		flex: 1;
		position: relative;
		overflow: hidden;
		background: var(--bg-base);
		cursor: default;
		user-select: none;
	}

	.context-menu {
		position: fixed;
		z-index: 100;
		background: var(--bg-raised);
		border: 1px solid var(--border);
		border-radius: 4px;
		min-width: 160px;
		padding: 4px 0;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
	}

	.context-menu button {
		display: block;
		width: 100%;
		text-align: left;
		padding: 4px 12px;
		font-size: 11px;
		font-family: var(--font-mono);
		color: var(--text-primary);
		background: none;
		border: none;
		cursor: pointer;
	}

	.context-menu button:hover {
		background: var(--accent-dim);
		color: var(--accent);
	}

	.context-separator {
		height: 1px;
		background: var(--border);
		margin: 4px 0;
	}

	.minimap {
		position: absolute;
		bottom: 8px;
		right: 8px;
		width: 150px;
		height: 100px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		border-radius: 4px;
		opacity: 0.8;
		overflow: hidden;
		cursor: pointer;
	}
</style>
