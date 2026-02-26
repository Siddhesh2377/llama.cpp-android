<script lang="ts">
	import { model } from '$lib/stores/model';
	import { formatBytes } from '$lib/stores/model';

	interface TreeNode {
		label: string;
		value?: string;
		children?: TreeNode[];
		expanded?: boolean;
		icon?: 'folder' | 'file' | 'tensor' | 'param';
	}

	let expandedNodes: Set<string> = new Set(['model', 'config', 'layers']);

	function toggle(path: string) {
		if (expandedNodes.has(path)) {
			expandedNodes.delete(path);
		} else {
			expandedNodes.add(path);
		}
		expandedNodes = expandedNodes;
	}

	$: tree = $model ? buildTree($model) : null;

	function buildTree(m: NonNullable<typeof $model>): TreeNode[] {
		return [
			{
				label: 'Model',
				icon: 'folder',
				children: [
					{ label: 'Name', value: m.name || m.arch, icon: 'param' },
					{ label: 'File', value: m.filename, icon: 'file' },
					{ label: 'Architecture', value: m.arch, icon: 'param' },
					{ label: 'Parameters', value: m.params, icon: 'param' },
					{ label: 'Quantization', value: m.quantType, icon: 'param' },
					{ label: 'File Size', value: formatBytes(m.fileSize), icon: 'param' },
					{ label: 'Tensors', value: String(m.tensorCount), icon: 'param' },
				]
			},
			{
				label: 'Config',
				icon: 'folder',
				children: [
					{ label: 'Context Length', value: String(m.contextLength), icon: 'param' },
					{ label: 'Embedding Size', value: String(m.embeddingSize), icon: 'param' },
					{ label: 'Layers', value: String(m.layerCount), icon: 'param' },
					{ label: 'Heads', value: String(m.headCount), icon: 'param' },
					{ label: 'KV Heads', value: String(m.headCountKV), icon: 'param' },
					{ label: 'Vocab Size', value: String(m.vocabSize), icon: 'param' },
				]
			},
			{
				label: 'Layers',
				icon: 'folder',
				children: m.layers?.map(l => ({
					label: `Layer ${l.index}`,
					icon: 'folder' as const,
					children: l.tensors.map(t => ({
						label: t.name.split('.').pop() || t.name,
						value: `${t.type} ${t.shape.join('×')}`,
						icon: 'tensor' as const
					}))
				})) || []
			}
		];
	}
</script>

<div class="panel">
	<div class="panel-header">
		<span class="label">Outliner</span>
		{#if $model}
			<span class="badge accent">{$model.arch}</span>
		{/if}
	</div>
	<div class="panel-body">
		{#if tree}
			<div class="tree stagger-reveal">
				{#each tree as node, i}
					{@const path = node.label.toLowerCase()}
					<div class="tree-item">
						{#if node.children}
							<button class="tree-toggle" on:click={() => toggle(path)}>
								<span class="tree-arrow" class:expanded={expandedNodes.has(path)}>&#9656;</span>
								<span class="tree-icon folder"></span>
								<span class="tree-label">{node.label}</span>
								<span class="tree-count">{node.children.length}</span>
							</button>
							{#if expandedNodes.has(path)}
								<div class="tree-children">
									{#each node.children as child}
										{@const childPath = `${path}.${child.label.toLowerCase()}`}
										<div class="tree-item depth-1">
											{#if child.children}
												<button class="tree-toggle" on:click={() => toggle(childPath)}>
													<span class="tree-arrow" class:expanded={expandedNodes.has(childPath)}>&#9656;</span>
													<span class="tree-icon folder"></span>
													<span class="tree-label">{child.label}</span>
													<span class="tree-count">{child.children.length}</span>
												</button>
												{#if expandedNodes.has(childPath)}
													<div class="tree-children">
														{#each child.children as leaf}
															<div class="tree-item depth-2">
																<span class="tree-icon tensor"></span>
																<span class="tree-label">{leaf.label}</span>
																{#if leaf.value}
																	<span class="tree-value">{leaf.value}</span>
																{/if}
															</div>
														{/each}
													</div>
												{/if}
											{:else}
												<span class="tree-icon param"></span>
												<span class="tree-label">{child.label}</span>
												{#if child.value}
													<span class="tree-value">{child.value}</span>
												{/if}
											{/if}
										</div>
									{/each}
								</div>
							{/if}
						{/if}
					</div>
				{/each}
			</div>
		{:else}
			<div class="empty-state">
				<div class="empty-icon">
					<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
				</div>
				<span class="empty-text">No model loaded</span>
				<span class="empty-hint">Load a GGUF file to inspect</span>
			</div>
		{/if}
	</div>
</div>

<style>
	.tree {
		font-size: var(--font-size-sm);
	}

	.tree-item {
		display: flex;
		flex-direction: column;
	}

	.tree-item.depth-1 { padding-left: 14px; }
	.tree-item.depth-2 {
		padding-left: 28px;
		display: flex;
		flex-direction: row;
		align-items: center;
		gap: 6px;
		padding-top: 1px;
		padding-bottom: 1px;
	}

	.tree-toggle {
		display: flex;
		align-items: center;
		gap: 4px;
		width: 100%;
		text-align: left;
		padding: 2px 0;
		font-size: var(--font-size-sm);
	}

	.tree-toggle:hover {
		color: var(--accent);
	}

	.tree-arrow {
		font-size: 10px;
		color: var(--text-tertiary);
		transition: transform var(--transition-fast);
		width: 10px;
		display: inline-block;
	}

	.tree-arrow.expanded {
		transform: rotate(90deg);
	}

	.tree-icon {
		width: 8px;
		height: 8px;
		flex-shrink: 0;
	}

	.tree-icon.folder { background: var(--accent); opacity: 0.6; }
	.tree-icon.tensor { background: var(--info); opacity: 0.5; }
	.tree-icon.param { background: var(--text-tertiary); opacity: 0.5; }

	.tree-label {
		color: var(--text-primary);
		flex-shrink: 0;
	}

	.tree-value {
		color: var(--text-secondary);
		margin-left: auto;
		text-align: right;
		font-size: var(--font-size-xs);
	}

	.tree-count {
		color: var(--text-tertiary);
		font-size: var(--font-size-xs);
		margin-left: auto;
	}

	.tree-children {
		margin-top: 1px;
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		gap: var(--spacing-md);
		color: var(--text-tertiary);
	}

	.empty-icon {
		opacity: 0.3;
	}

	.empty-text {
		font-size: var(--font-size-sm);
		font-weight: 500;
	}

	.empty-hint {
		font-size: var(--font-size-xs);
		opacity: 0.6;
	}
</style>
