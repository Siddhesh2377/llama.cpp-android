<script lang="ts">
	import { model } from '$lib/stores/model';
	import { getModelGraph, type GraphNode, type GraphEdge, type ModelGraph } from '$lib/api/client';
	import { consoleStore } from '$lib/stores/console';
	import { onMount } from 'svelte';

	let graph: ModelGraph | null = null;
	let loading = false;
	let selectedNode: GraphNode | null = null;

	// Pan & zoom state
	let viewX = 0;
	let viewY = 0;
	let scale = 1;
	let isPanning = false;
	let panStartX = 0;
	let panStartY = 0;
	let panStartViewX = 0;
	let panStartViewY = 0;

	let svgEl: SVGSVGElement;

	// Node type -> color mapping
	const nodeColors: Record<string, string> = {
		embed: 'var(--info)',
		attn: 'var(--accent)',
		attn_proj: '#d4843a',
		norm: 'var(--text-tertiary)',
		ffn: 'var(--success)',
		ffn_gate: 'var(--success)',
		ffn_down: '#3a8855',
		head: '#8a5ec4',
		proj: '#d4843a',
		ellipsis: 'var(--text-tertiary)',
		other: 'var(--border)'
	};

	function getNodeColor(type: string): string {
		return nodeColors[type] || nodeColors.other;
	}

	async function loadGraph() {
		if (!$model) return;
		loading = true;
		consoleStore.info('Building compute graph...', 'graph');
		graph = await getModelGraph();
		if (graph) {
			consoleStore.success(`Graph: ${graph.nodes.length} nodes, ${graph.edges.length} edges`, 'graph');
			// Center the view
			if (graph.nodes.length > 0) {
				const maxY = Math.max(...graph.nodes.map(n => n.y + n.height));
				const maxX = Math.max(...graph.nodes.map(n => n.x + n.width));
				viewX = -(maxX / 2 - 300);
				viewY = 0;
				// Fit vertically
				if (svgEl) {
					const rect = svgEl.getBoundingClientRect();
					if (maxY > rect.height) {
						scale = Math.max(0.3, rect.height / (maxY + 40));
					}
				}
			}
		}
		loading = false;
	}

	// Reload graph when model changes
	$: if ($model) {
		loadGraph();
	} else {
		graph = null;
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const delta = e.deltaY > 0 ? 0.9 : 1.1;
		const newScale = Math.max(0.15, Math.min(3, scale * delta));

		// Zoom toward cursor
		if (svgEl) {
			const rect = svgEl.getBoundingClientRect();
			const cx = e.clientX - rect.left;
			const cy = e.clientY - rect.top;
			viewX = cx - (cx - viewX) * (newScale / scale);
			viewY = cy - (cy - viewY) * (newScale / scale);
		}

		scale = newScale;
	}

	function handleMouseDown(e: MouseEvent) {
		if (e.button !== 0) return;
		// Check if we clicked on a node
		const target = e.target as SVGElement;
		if (target.closest('.graph-node')) return;

		isPanning = true;
		panStartX = e.clientX;
		panStartY = e.clientY;
		panStartViewX = viewX;
		panStartViewY = viewY;
	}

	function handleMouseMove(e: MouseEvent) {
		if (!isPanning) return;
		viewX = panStartViewX + (e.clientX - panStartX);
		viewY = panStartViewY + (e.clientY - panStartY);
	}

	function handleMouseUp() {
		isPanning = false;
	}

	function selectNode(node: GraphNode) {
		selectedNode = selectedNode?.id === node.id ? null : node;
	}

	function getEdgePath(edge: GraphEdge): string {
		if (!graph) return '';
		const from = graph.nodes.find(n => n.id === edge.from);
		const to = graph.nodes.find(n => n.id === edge.to);
		if (!from || !to) return '';

		const x1 = from.x + from.width / 2;
		const y1 = from.y + from.height;
		const x2 = to.x + to.width / 2;
		const y2 = to.y;
		const midY = (y1 + y2) / 2;

		return `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;
	}

	function resetView() {
		viewX = 0;
		viewY = 0;
		scale = 1;
		if (graph && graph.nodes.length > 0) {
			const maxX = Math.max(...graph.nodes.map(n => n.x + n.width));
			viewX = -(maxX / 2 - 300);
		}
	}
</script>

<div class="graph-view">
	{#if !$model}
		<div class="empty-state">
			<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><circle cx="12" cy="12" r="3"/><path d="M12 3v6M12 15v6M3 12h6M15 12h6"/></svg>
			<span class="empty-title">Graph Editor</span>
			<span class="empty-hint">Load a model in the Architecture tab to view its compute graph</span>
		</div>
	{:else if loading}
		<div class="empty-state">
			<span class="empty-title glow-pulse">Building graph...</span>
		</div>
	{:else if graph}
		<!-- Toolbar -->
		<div class="graph-toolbar">
			<span class="toolbar-label">{graph.arch} | {graph.nodes.length} nodes | {graph.layerCount} layers</span>
			<div class="toolbar-spacer"></div>
			<button class="toolbar-btn" on:click={() => { scale = Math.min(3, scale * 1.2); }} title="Zoom in">+</button>
			<button class="toolbar-btn" on:click={() => { scale = Math.max(0.15, scale * 0.8); }} title="Zoom out">-</button>
			<span class="toolbar-zoom">{Math.round(scale * 100)}%</span>
			<button class="toolbar-btn" on:click={resetView} title="Reset view">
				<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9"/><polyline points="3 3 3 9 9 9"/></svg>
			</button>
		</div>

		<!-- Canvas -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<svg
			class="graph-canvas"
			bind:this={svgEl}
			on:wheel={handleWheel}
			on:mousedown={handleMouseDown}
			on:mousemove={handleMouseMove}
			on:mouseup={handleMouseUp}
			on:mouseleave={handleMouseUp}
			style="cursor: {isPanning ? 'grabbing' : 'grab'}"
		>
			<!-- Grid pattern -->
			<defs>
				<pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"
					patternTransform="translate({viewX % 20} {viewY % 20}) scale({scale})">
					<circle cx="10" cy="10" r="0.5" fill="var(--border-subtle)" />
				</pattern>
			</defs>
			<rect width="100%" height="100%" fill="url(#grid)" />

			<g transform="translate({viewX}, {viewY}) scale({scale})">
				<!-- Edges -->
				{#each graph.edges as edge}
					{@const isSelected = selectedNode && (edge.from === selectedNode.id || edge.to === selectedNode.id)}
					<path
						d={getEdgePath(edge)}
						fill="none"
						stroke={isSelected ? 'var(--accent)' : 'var(--border)'}
						stroke-width={isSelected ? 2 : 1}
						opacity={selectedNode && !isSelected ? 0.2 : 0.8}
					/>
				{/each}

				<!-- Nodes -->
				{#each graph.nodes as node}
					{@const color = getNodeColor(node.type)}
					{@const isSelected = selectedNode?.id === node.id}
					{@const isConnected = selectedNode && graph.edges.some(e =>
						(e.from === selectedNode.id && e.to === node.id) ||
						(e.to === selectedNode.id && e.from === node.id)
					)}
					<!-- svelte-ignore a11y_click_events_have_key_events -->
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<g
						class="graph-node"
						on:click={() => selectNode(node)}
						opacity={selectedNode && !isSelected && !isConnected ? 0.35 : 1}
						style="cursor: pointer"
					>
						<!-- Node background -->
						<rect
							x={node.x}
							y={node.y}
							width={node.width}
							height={node.height}
							fill="var(--bg-raised)"
							stroke={isSelected ? color : 'var(--border)'}
							stroke-width={isSelected ? 2 : 1}
						/>
						<!-- Left accent bar -->
						<rect
							x={node.x}
							y={node.y}
							width="3"
							height={node.height}
							fill={color}
						/>
						<!-- Label -->
						<text
							x={node.x + 12}
							y={node.y + (node.sublabel ? 16 : node.height / 2 + 4)}
							fill="var(--text-primary)"
							font-size="11"
							font-family="var(--font-mono)"
							font-weight="500"
						>{node.label}</text>
						<!-- Sublabel -->
						{#if node.sublabel}
							<text
								x={node.x + 12}
								y={node.y + 30}
								fill="var(--text-tertiary)"
								font-size="9"
								font-family="var(--font-mono)"
							>{node.sublabel}</text>
						{/if}
					</g>
				{/each}
			</g>
		</svg>

		<!-- Selected node detail -->
		{#if selectedNode}
			<div class="node-detail">
				<div class="detail-header">
					<div class="detail-color" style="background: {getNodeColor(selectedNode.type)}"></div>
					<span class="detail-label">{selectedNode.label}</span>
					<button class="detail-close" on:click={() => selectedNode = null}>
						<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
					</button>
				</div>
				<div class="detail-body">
					<div class="detail-row"><span class="dim">Type</span><span>{selectedNode.type}</span></div>
					<div class="detail-row"><span class="dim">Size</span><span>{selectedNode.sublabel || '-'}</span></div>
					{#if selectedNode.layer >= 0}
						<div class="detail-row"><span class="dim">Layer</span><span>{selectedNode.layer}</span></div>
					{/if}
				</div>
			</div>
		{/if}
	{/if}
</div>

<style>
	.graph-view {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		position: relative;
	}

	.empty-state {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--spacing-lg);
		color: var(--text-tertiary);
		opacity: 0.5;
	}

	.empty-title { font-size: var(--font-size-lg); font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
	.empty-hint { font-size: var(--font-size-sm); }

	/* Toolbar */
	.graph-toolbar {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-sm) var(--spacing-lg);
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		font-size: var(--font-size-xs);
		user-select: none;
		z-index: 10;
	}

	.toolbar-label {
		color: var(--text-secondary);
		font-weight: 500;
		letter-spacing: 0.04em;
	}

	.toolbar-spacer { flex: 1; }

	.toolbar-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 24px;
		height: 24px;
		border: 1px solid var(--border);
		color: var(--text-secondary);
		font-size: var(--font-size-md);
		font-weight: 600;
		transition: all var(--transition-fast);
	}

	.toolbar-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.toolbar-zoom {
		color: var(--text-tertiary);
		min-width: 36px;
		text-align: center;
	}

	/* Canvas */
	.graph-canvas {
		flex: 1;
		width: 100%;
		background: var(--bg-base);
		display: block;
	}

	.graph-canvas :global(.graph-node:hover rect:first-child) {
		stroke: var(--accent);
		stroke-width: 1.5;
	}

	/* Node detail overlay */
	.node-detail {
		position: absolute;
		bottom: var(--spacing-lg);
		right: var(--spacing-lg);
		width: 220px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		z-index: 20;
		animation: fade-in-up 150ms ease;
	}

	.detail-header {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-sm) var(--spacing-md);
		border-bottom: 1px solid var(--border);
		background: var(--bg-raised);
	}

	.detail-color {
		width: 8px;
		height: 8px;
		flex-shrink: 0;
	}

	.detail-label {
		font-size: var(--font-size-xs);
		font-weight: 600;
		flex: 1;
	}

	.detail-close {
		color: var(--text-tertiary);
		display: flex;
		align-items: center;
	}

	.detail-close:hover { color: var(--text-secondary); }

	.detail-body {
		padding: var(--spacing-md);
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.detail-row {
		display: flex;
		justify-content: space-between;
		font-size: var(--font-size-xs);
	}
</style>
