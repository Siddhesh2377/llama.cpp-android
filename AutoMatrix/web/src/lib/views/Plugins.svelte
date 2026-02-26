<script lang="ts">
	import { plugins } from '$lib/stores/plugins';
	import { archPlugins, backendPlugins, quantPlugins, samplingPlugins } from '$lib/stores/plugins';

	let activeCategory = $state<string>('all');
	let selectedPlugin = $state<Record<string, unknown> | null>(null);

	const categories = [
		{ id: 'all', label: 'All', icon: 'M4 6h16M4 12h16M4 18h16' },
		{ id: 'architecture', label: 'Architectures', icon: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5' },
		{ id: 'backend', label: 'Backends', icon: 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z' },
		{ id: 'quant', label: 'Quantization', icon: 'M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4' },
		{ id: 'sampling', label: 'Sampling', icon: 'M3 3v18h18M7 16l4-8 4 4 4-10' },
	];

	let filteredPlugins = $derived(
		activeCategory === 'all'
			? $plugins
			: $plugins.filter(p => p.type === activeCategory)
	);

	function typeColor(type: string): string {
		switch (type) {
			case 'architecture': return 'var(--accent)';
			case 'backend': return '#6ec6ff';
			case 'quant': return '#a5d6a7';
			case 'sampling': return '#ce93d8';
			default: return 'var(--text-secondary)';
		}
	}

	function selectPlugin(p: Record<string, unknown>) {
		selectedPlugin = selectedPlugin === p ? null : p;
	}
</script>

<div class="plugins-view">
	<div class="plugins-sidebar">
		<div class="sidebar-header">
			<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2">
				<path d="M12 2L2 7l10 5 10-5-10-5z"/>
				<path d="M2 17l10 5 10-5"/>
			</svg>
			<span>Plugin Registry</span>
		</div>

		<div class="category-list">
			{#each categories as cat}
				<button
					class="category-btn"
					class:active={activeCategory === cat.id}
					onclick={() => activeCategory = cat.id}
				>
					<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<path d={cat.icon}/>
					</svg>
					<span>{cat.label}</span>
					<span class="count">
						{cat.id === 'all' ? $plugins.length :
						 cat.id === 'architecture' ? $archPlugins.length :
						 cat.id === 'backend' ? $backendPlugins.length :
						 cat.id === 'quant' ? $quantPlugins.length :
						 $samplingPlugins.length}
					</span>
				</button>
			{/each}
		</div>

		<div class="format-info">
			<div class="format-badge">AMXP v1</div>
			<span class="format-desc">Binary header + JSON body</span>
		</div>
	</div>

	<div class="plugins-grid">
		{#if filteredPlugins.length === 0}
			<div class="empty-state">No plugins loaded</div>
		{:else}
			{#each filteredPlugins as plugin}
				<button
					class="plugin-card"
					class:selected={selectedPlugin === plugin}
					onclick={() => selectPlugin(plugin)}
				>
					<div class="card-header">
						<span class="type-dot" style="background: {typeColor(plugin.type)}"></span>
						<span class="plugin-name">{plugin.name}</span>
						{#if (plugin.body as Record<string, unknown>)?.recommended}
							<span class="rec-badge">REC</span>
						{/if}
					</div>
					<div class="card-type">{plugin.type}</div>
					<div class="card-desc">{(plugin.body as Record<string, unknown>)?.description || ''}</div>

					{#if plugin.type === 'quant'}
						{@const b = plugin.body as Record<string, unknown>}
						<div class="card-stats">
							<div class="stat">
								<span class="stat-label">Quality</span>
								<div class="stat-bar">
									<div class="stat-fill quality" style="width: {((b.quality_rating as number) || 0) * 100}%"></div>
								</div>
							</div>
							<div class="stat">
								<span class="stat-label">Speed</span>
								<div class="stat-bar">
									<div class="stat-fill speed" style="width: {((b.speed_rating as number) || 0) * 100}%"></div>
								</div>
							</div>
							<span class="stat-bits">{b.bits_per_weight}b</span>
						</div>
					{/if}

					{#if plugin.type === 'architecture'}
						{@const b = plugin.body as Record<string, unknown>}
						<div class="card-tags">
							<span class="tag">{(b.activation as string) || '?'}</span>
							<span class="tag">{(b.norm_type as string) || '?'}</span>
							{#if (b.attention as Record<string, unknown>)?.rope}
								<span class="tag">RoPE</span>
							{/if}
						</div>
					{/if}

					{#if plugin.type === 'backend'}
						{@const b = plugin.body as Record<string, unknown>}
						<div class="card-tags">
							{#each ((b.features as string[]) || []).slice(0, 4) as feat}
								<span class="tag">{feat}</span>
							{/each}
						</div>
					{/if}
				</button>
			{/each}
		{/if}
	</div>

	{#if selectedPlugin}
		<div class="plugin-detail">
			<div class="detail-header">
				<span class="type-dot" style="background: {typeColor(selectedPlugin.type as string)}"></span>
				<h3>{(selectedPlugin as Record<string, unknown>).name}</h3>
				<button class="close-btn" onclick={() => selectedPlugin = null}>
					<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
				</button>
			</div>
			<div class="detail-body">
				<pre>{JSON.stringify((selectedPlugin as Record<string, unknown>).body || selectedPlugin, null, 2)}</pre>
			</div>
		</div>
	{/if}
</div>

<style>
	.plugins-view {
		display: flex;
		height: 100%;
		overflow: hidden;
	}

	.plugins-sidebar {
		width: 200px;
		border-right: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		padding: 12px 0;
	}

	.sidebar-header {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 0 14px 12px;
		font-size: 12px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-primary);
		border-bottom: 1px solid var(--border);
		margin-bottom: 8px;
	}

	.category-list {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: 4px 8px;
	}

	.category-btn {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 8px;
		border: none;
		background: none;
		color: var(--text-secondary);
		font-size: 11px;
		cursor: pointer;
		border-radius: 4px;
		transition: all 0.15s;
	}

	.category-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
	.category-btn.active { background: var(--bg-active); color: var(--accent); }
	.category-btn .count {
		margin-left: auto;
		font-size: 10px;
		opacity: 0.5;
	}

	.format-info {
		padding: 10px 14px;
		border-top: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.format-badge {
		font-size: 10px;
		font-weight: 700;
		color: var(--accent);
		letter-spacing: 0.1em;
	}

	.format-desc {
		font-size: 9px;
		color: var(--text-tertiary);
	}

	.plugins-grid {
		flex: 1;
		padding: 16px;
		overflow-y: auto;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
		gap: 12px;
		align-content: start;
	}

	.empty-state {
		grid-column: 1 / -1;
		text-align: center;
		padding: 60px 20px;
		color: var(--text-tertiary);
		font-size: 13px;
	}

	.plugin-card {
		background: var(--bg-panel);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 14px;
		cursor: pointer;
		transition: all 0.15s;
		text-align: left;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.plugin-card:hover { border-color: var(--text-tertiary); }
	.plugin-card.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }

	.card-header {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.type-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.plugin-name {
		font-size: 13px;
		font-weight: 600;
		color: var(--text-primary);
	}

	.rec-badge {
		font-size: 8px;
		font-weight: 700;
		background: var(--accent);
		color: var(--bg-base);
		padding: 1px 5px;
		border-radius: 3px;
		letter-spacing: 0.05em;
		margin-left: auto;
	}

	.card-type {
		font-size: 10px;
		color: var(--text-tertiary);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.card-desc {
		font-size: 11px;
		color: var(--text-secondary);
		line-height: 1.4;
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}

	.card-stats {
		display: flex;
		flex-direction: column;
		gap: 4px;
		margin-top: 4px;
		position: relative;
	}

	.stat {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.stat-label {
		font-size: 9px;
		color: var(--text-tertiary);
		width: 38px;
		text-transform: uppercase;
	}

	.stat-bar {
		flex: 1;
		height: 4px;
		background: var(--bg-active);
		border-radius: 2px;
		overflow: hidden;
	}

	.stat-fill {
		height: 100%;
		border-radius: 2px;
		transition: width 0.3s;
	}

	.stat-fill.quality { background: #a5d6a7; }
	.stat-fill.speed { background: #6ec6ff; }

	.stat-bits {
		position: absolute;
		right: 0;
		top: -2px;
		font-size: 10px;
		font-weight: 700;
		color: var(--text-secondary);
	}

	.card-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 4px;
	}

	.tag {
		font-size: 9px;
		padding: 1px 6px;
		border-radius: 3px;
		background: var(--bg-active);
		color: var(--text-secondary);
	}

	.plugin-detail {
		width: 320px;
		border-left: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.detail-header {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 12px 14px;
		border-bottom: 1px solid var(--border);
	}

	.detail-header h3 {
		font-size: 13px;
		font-weight: 600;
		margin: 0;
		flex: 1;
	}

	.close-btn {
		background: none;
		border: none;
		color: var(--text-tertiary);
		cursor: pointer;
		padding: 4px;
		border-radius: 4px;
	}

	.close-btn:hover { background: var(--bg-hover); color: var(--text-primary); }

	.detail-body {
		flex: 1;
		overflow-y: auto;
		padding: 12px;
	}

	.detail-body pre {
		font-size: 10px;
		line-height: 1.5;
		color: var(--text-secondary);
		white-space: pre-wrap;
		word-break: break-word;
		margin: 0;
	}
</style>
