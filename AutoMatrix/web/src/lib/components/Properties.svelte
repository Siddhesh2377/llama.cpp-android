<script lang="ts">
	import { model, formatBytes } from '$lib/stores/model';
	import { theme } from '$lib/stores/settings';
	import { selectedNodeIds, activeDoc, activeDocId, updateDocument } from '$lib/engine/store';
	import { CATEGORY_COLORS, PIN_COLORS, type BpNode, type BpDocument, type NodeCategory, type PinType } from '$lib/engine/types';
	import { get } from 'svelte/store';

	// -----------------------------------------------------------------------
	// Model metadata groups (Mode 1 — no node selected)
	// -----------------------------------------------------------------------

	interface PropGroup {
		label: string;
		props: { key: string; value: string; tag?: string }[];
	}

	$: groups = $model ? buildGroups($model) : [];

	function buildGroups(m: NonNullable<typeof $model>): PropGroup[] {
		return [
			{
				label: 'Format',
				props: [
					{ key: 'Type', value: 'GGUF' },
					{ key: 'Version', value: '3' },
					{ key: 'Endian', value: 'Little' },
					{ key: 'File Size', value: formatBytes(m.fileSize) },
				]
			},
			{
				label: 'Model',
				props: [
					{ key: 'Architecture', value: m.arch, tag: 'arch' },
					{ key: 'Parameters', value: m.params },
					{ key: 'Quantization', value: m.quantType, tag: 'quant' },
					{ key: 'Tensor Count', value: String(m.tensorCount) },
				]
			},
			{
				label: 'Config',
				props: [
					{ key: 'Embedding', value: String(m.embeddingSize) },
					{ key: 'Layers', value: String(m.layerCount) },
					{ key: 'Heads', value: `${m.headCount} / ${m.headCountKV} KV` },
					{ key: 'Context', value: String(m.contextLength) },
					{ key: 'Vocab', value: String(m.vocabSize) },
				]
			},
			{
				label: 'Workspace',
				props: [
					{ key: 'Mode', value: 'Inspect', tag: 'mode' },
					{ key: 'Backend', value: 'CPU (NEON)' },
					{ key: 'Threads', value: '4' },
				]
			}
		];
	}

	// -----------------------------------------------------------------------
	// Node inspector (Mode 2 — node selected)
	// -----------------------------------------------------------------------

	const QUANT_OPTIONS = ['q8_0', 'q5_0', 'q5_1', 'q4_0', 'q4_1', 'f16'];

	$: doc = $activeDoc;
	$: selected = $selectedNodeIds;
	$: selectedNode = (doc && selected.size === 1)
		? doc.nodes.find((n: BpNode) => selected.has(n.id)) ?? null
		: null;

	$: metaEntries = selectedNode
		? Object.entries(selectedNode.metadata)
		: [];

	$: categoryColor = selectedNode
		? getCategoryColor(selectedNode.category)
		: null;

	function getCategoryColor(cat: NodeCategory): { bg: string; accent: string } {
		const colors = CATEGORY_COLORS[cat];
		if (!colors) return { bg: '#2a2a2a', accent: '#8a7e74' };
		return $theme === 'dark'
			? { bg: colors.darkBg, accent: colors.darkAccent }
			: { bg: colors.lightBg, accent: colors.lightAccent };
	}

	function getPinColor(dataType: PinType): string {
		return PIN_COLORS[dataType] ?? '#8a7e74';
	}

	function updateNode(field: string, value: unknown) {
		if (!selectedNode || !doc) return;
		const nodeId = selectedNode.id;
		const docId = doc.id;
		updateDocument(docId, (d: BpDocument) => {
			const node = d.nodes.find((n: BpNode) => n.id === nodeId);
			if (!node) return d;
			switch (field) {
				case 'label': node.label = value as string; break;
				case 'enabled': node.enabled = value as boolean; break;
				case 'collapsed': node.collapsed = value as boolean; break;
				case 'x': node.position.x = value as number; break;
				case 'y': node.position.y = value as number; break;
				default:
					if (field.startsWith('meta.')) {
						node.metadata[field.slice(5)] = value;
					}
					break;
			}
			return d;
		});
	}

	function duplicateNode() {
		if (!selectedNode || !doc) return;
		const docId = doc.id;
		const src = selectedNode;
		updateDocument(docId, (d: BpDocument) => {
			const newId = `${src.id}_copy_${Date.now().toString(36)}`;
			const clone: BpNode = structuredClone(src);
			clone.id = newId;
			clone.label = `${src.label} (copy)`;
			clone.position = { x: src.position.x + 40, y: src.position.y + 40 };
			clone.pins = clone.pins.map(p => ({
				...p,
				id: `${newId}_${p.name}`,
				connected: false,
			}));
			d.nodes.push(clone);
			return d;
		});
	}

	function deleteNode() {
		if (!selectedNode || !doc) return;
		const docId = doc.id;
		const nodeId = selectedNode.id;
		updateDocument(docId, (d: BpDocument) => {
			// Remove edges connected to this node
			d.edges = d.edges.filter(e =>
				e.from.nodeId !== nodeId && e.to.nodeId !== nodeId
			);
			// Remove the node
			d.nodes = d.nodes.filter((n: BpNode) => n.id !== nodeId);
			return d;
		});
		// Clear selection
		selectedNodeIds.set(new Set());
	}
</script>

<div class="panel">
	<div class="panel-header">
		<span class="label">Properties</span>
		{#if selectedNode}
			<span class="badge accent">Node</span>
		{/if}
	</div>
	<div class="panel-body">
		{#if selectedNode && categoryColor}
			<!-- ============================================================ -->
			<!-- Mode 2: Node Inspector                                       -->
			<!-- ============================================================ -->
			<div class="inspector stagger-reveal">

				<!-- Header -->
				<div class="node-header" style="border-left: 3px solid {categoryColor.accent};">
					<div class="node-header-top">
						<span class="cat-dot" style="background: {categoryColor.accent};"></span>
						<span class="cat-name" style="color: {categoryColor.accent};">{selectedNode.category}</span>
					</div>
					<span class="node-id">{selectedNode.id}</span>
				</div>

				<!-- General -->
				<div class="prop-section">
					<div class="prop-section-label">General</div>
					<div class="prop-section-body">
						<div class="prop-row">
							<span class="prop-key">Label</span>
							<input
								type="text"
								class="prop-input"
								value={selectedNode.label}
								on:change={(e) => updateNode('label', e.currentTarget.value)}
							/>
						</div>
						<div class="prop-row">
							<span class="prop-key">Type</span>
							<span class="prop-value mono dim">{selectedNode.type}</span>
						</div>
						<div class="prop-row">
							<span class="prop-key">Enabled</span>
							<label class="toggle-label">
								<input
									type="checkbox"
									class="toggle-check"
									checked={selectedNode.enabled}
									on:change={(e) => updateNode('enabled', e.currentTarget.checked)}
								/>
								<span class="toggle-track">
									<span class="toggle-thumb"></span>
								</span>
							</label>
						</div>
						<div class="prop-row">
							<span class="prop-key">Collapsed</span>
							<label class="toggle-label">
								<input
									type="checkbox"
									class="toggle-check"
									checked={selectedNode.collapsed}
									on:change={(e) => updateNode('collapsed', e.currentTarget.checked)}
								/>
								<span class="toggle-track">
									<span class="toggle-thumb"></span>
								</span>
							</label>
						</div>
					</div>
				</div>

				<!-- Pins -->
				{#if selectedNode.pins.length > 0}
					<div class="prop-section">
						<div class="prop-section-label">Pins <span class="dim">({selectedNode.pins.length})</span></div>
						<div class="prop-section-body">
							{#each selectedNode.pins as pin}
								<div class="pin-row">
									<span class="pin-dir" title={pin.direction === 'in' ? 'Input' : 'Output'}>
										{#if pin.direction === 'in'}
											<svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 5h6M6 3l2 2-2 2" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>
										{:else}
											<svg width="10" height="10" viewBox="0 0 10 10"><path d="M8 5H2M4 3L2 5l2 2" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>
										{/if}
									</span>
									<span class="pin-name">{pin.name}</span>
									<span class="pin-type-badge" style="color: {getPinColor(pin.dataType)}; border-color: {getPinColor(pin.dataType)}40; background: {getPinColor(pin.dataType)}15;">
										{pin.dataType}
									</span>
									{#if pin.shape && pin.shape.length > 0}
										<span class="pin-shape">[{pin.shape.join(', ')}]</span>
									{/if}
									<span class="pin-connected" class:filled={pin.connected} style="border-color: {getPinColor(pin.dataType)};" title={pin.connected ? 'Connected' : 'Unconnected'}>
										{#if pin.connected}
											<span class="pin-connected-fill" style="background: {getPinColor(pin.dataType)};"></span>
										{/if}
									</span>
								</div>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Metadata -->
				{#if metaEntries.length > 0}
					<div class="prop-section">
						<div class="prop-section-label">Metadata</div>
						<div class="prop-section-body">
							{#each metaEntries as [key, val]}
								<div class="prop-row">
									<span class="prop-key">{key}</span>
									{#if key === 'quant'}
										<select
											class="prop-select"
											value={String(val)}
											on:change={(e) => updateNode(`meta.${key}`, e.currentTarget.value)}
										>
											{#each QUANT_OPTIONS as opt}
												<option value={opt}>{opt}</option>
											{/each}
										</select>
									{:else if typeof val === 'boolean'}
										<label class="toggle-label">
											<input
												type="checkbox"
												class="toggle-check"
												checked={val}
												on:change={(e) => updateNode(`meta.${key}`, e.currentTarget.checked)}
											/>
											<span class="toggle-track">
												<span class="toggle-thumb"></span>
											</span>
										</label>
									{:else if typeof val === 'number'}
										<input
											type="number"
											class="prop-input num"
											value={val}
											on:change={(e) => updateNode(`meta.${key}`, parseFloat(e.currentTarget.value) || 0)}
										/>
									{:else}
										<input
											type="text"
											class="prop-input"
											value={String(val ?? '')}
											on:change={(e) => updateNode(`meta.${key}`, e.currentTarget.value)}
										/>
									{/if}
								</div>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Position -->
				<div class="prop-section">
					<div class="prop-section-label">Position</div>
					<div class="prop-section-body">
						<div class="prop-row">
							<span class="prop-key">X</span>
							<input
								type="number"
								class="prop-input num"
								value={selectedNode.position.x}
								on:change={(e) => updateNode('x', parseFloat(e.currentTarget.value) || 0)}
							/>
						</div>
						<div class="prop-row">
							<span class="prop-key">Y</span>
							<input
								type="number"
								class="prop-input num"
								value={selectedNode.position.y}
								on:change={(e) => updateNode('y', parseFloat(e.currentTarget.value) || 0)}
							/>
						</div>
						<div class="prop-row">
							<span class="prop-key">Size</span>
							<span class="prop-value mono dim">{selectedNode.size.w} x {selectedNode.size.h}</span>
						</div>
					</div>
				</div>

				<!-- Actions -->
				<div class="prop-section">
					<div class="prop-section-label">Actions</div>
					<div class="prop-section-body actions">
						<button class="action-btn" on:click={duplicateNode}>
							<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="0"/><path d="M5 15H4V4h11v1"/></svg>
							Duplicate
						</button>
						<button class="action-btn danger" on:click={deleteNode}>
							<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M19 6v14H5V6"/><path d="M10 11v6M14 11v6"/></svg>
							Delete
						</button>
					</div>
				</div>
			</div>

		{:else if $model}
			<!-- ============================================================ -->
			<!-- Mode 1: Model metadata (original)                            -->
			<!-- ============================================================ -->
			<div class="prop-groups stagger-reveal">
				{#each groups as group}
					<div class="prop-group">
						<div class="prop-group-label">{group.label}</div>
						{#each group.props as prop}
							<div class="prop-row">
								<span class="prop-key">{prop.key}</span>
								<span class="prop-value">
									{#if prop.tag}
										<span class="badge accent">{prop.value}</span>
									{:else}
										{prop.value}
									{/if}
								</span>
							</div>
						{/each}
					</div>
				{/each}
			</div>

		{:else}
			<div class="empty-state">
				<span class="empty-text dim">No model selected</span>
			</div>
		{/if}
	</div>
</div>

<style>
	/* ------------------------------------------------------------------ */
	/* Mode 1 — Model metadata (preserved from original)                  */
	/* ------------------------------------------------------------------ */

	.prop-groups {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	.prop-group {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.prop-group-label {
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
		padding-bottom: var(--spacing-sm);
		border-bottom: 1px solid var(--border-subtle);
		margin-bottom: var(--spacing-sm);
	}

	.empty-state {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
	}

	.empty-text {
		font-size: var(--font-size-sm);
	}

	/* ------------------------------------------------------------------ */
	/* Shared                                                              */
	/* ------------------------------------------------------------------ */

	.prop-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 2px 0;
		gap: var(--spacing-md);
	}

	.prop-key {
		font-size: var(--font-size-sm);
		color: var(--text-secondary);
		flex-shrink: 0;
	}

	.prop-value {
		font-size: var(--font-size-sm);
		color: var(--text-primary);
		text-align: right;
		font-weight: 500;
	}

	/* ------------------------------------------------------------------ */
	/* Mode 2 — Node Inspector                                            */
	/* ------------------------------------------------------------------ */

	.inspector {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	/* Node header */
	.node-header {
		padding: var(--spacing-md);
		background: var(--bg-raised);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
	}

	.node-header-top {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.cat-dot {
		width: 8px;
		height: 8px;
		flex-shrink: 0;
	}

	.cat-name {
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
	}

	.node-id {
		font-family: var(--font-mono);
		font-size: var(--font-size-xs);
		color: var(--text-tertiary);
		letter-spacing: 0.02em;
	}

	/* Sections */
	.prop-section {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.prop-section-label {
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
		padding-bottom: var(--spacing-sm);
		border-bottom: 1px solid var(--border-subtle);
		margin-bottom: var(--spacing-sm);
	}

	.prop-section-body {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	/* Inputs */
	.prop-input {
		width: 100%;
		max-width: 140px;
		padding: 2px 6px;
		font-size: var(--font-size-xs);
		font-family: var(--font-mono);
		background: var(--bg-base);
		color: var(--text-primary);
		border: 1px solid var(--border);
		text-align: right;
	}

	.prop-input:focus {
		border-color: var(--accent);
		outline: none;
	}

	.prop-input.num {
		max-width: 80px;
	}

	.prop-select {
		max-width: 140px;
		padding: 2px 6px;
		font-size: var(--font-size-xs);
		font-family: var(--font-mono);
		background: var(--bg-base);
		color: var(--text-primary);
		border: 1px solid var(--border);
		text-align: right;
		cursor: pointer;
	}

	.prop-select:focus {
		border-color: var(--accent);
		outline: none;
	}

	/* Toggle switch */
	.toggle-label {
		position: relative;
		display: inline-flex;
		align-items: center;
		cursor: pointer;
	}

	.toggle-check {
		position: absolute;
		opacity: 0;
		width: 0;
		height: 0;
	}

	.toggle-track {
		width: 28px;
		height: 14px;
		background: var(--border);
		position: relative;
		transition: background var(--transition-fast);
	}

	.toggle-check:checked + .toggle-track {
		background: var(--accent);
	}

	.toggle-thumb {
		position: absolute;
		top: 2px;
		left: 2px;
		width: 10px;
		height: 10px;
		background: var(--text-primary);
		transition: transform var(--transition-fast);
	}

	.toggle-check:checked + .toggle-track .toggle-thumb {
		transform: translateX(14px);
	}

	/* Pin rows */
	.pin-row {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 3px 0;
		font-size: var(--font-size-xs);
	}

	.pin-dir {
		flex-shrink: 0;
		color: var(--text-tertiary);
		display: flex;
		align-items: center;
	}

	.pin-name {
		color: var(--text-primary);
		flex-shrink: 0;
	}

	.pin-type-badge {
		display: inline-flex;
		align-items: center;
		padding: 0 4px;
		font-size: 9px;
		font-weight: 500;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		border: 1px solid;
	}

	.pin-shape {
		font-family: var(--font-mono);
		font-size: 9px;
		color: var(--text-tertiary);
		margin-left: auto;
	}

	.pin-connected {
		width: 8px;
		height: 8px;
		border: 1.5px solid;
		border-radius: 50%;
		flex-shrink: 0;
		position: relative;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.pin-connected:not(.filled) {
		margin-left: auto;
	}

	.pin-connected.filled .pin-shape ~ & {
		margin-left: 0;
	}

	.pin-connected-fill {
		width: 4px;
		height: 4px;
		border-radius: 50%;
	}

	/* Action buttons */
	.prop-section-body.actions {
		gap: var(--spacing-sm);
	}

	.action-btn {
		display: flex;
		align-items: center;
		gap: 6px;
		width: 100%;
		padding: var(--spacing-sm) var(--spacing-md);
		font-size: var(--font-size-xs);
		font-weight: 500;
		letter-spacing: 0.04em;
		color: var(--text-secondary);
		background: var(--bg-base);
		border: 1px solid var(--border);
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.action-btn:hover {
		color: var(--text-primary);
		border-color: var(--accent);
		background: var(--accent-dim);
	}

	.action-btn.danger {
		color: var(--error);
	}

	.action-btn.danger:hover {
		border-color: var(--error);
		background: var(--error-dim);
		color: var(--error);
	}
</style>
