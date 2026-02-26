<script lang="ts">
	import { ALL_NODES, type NodeTemplate } from '$lib/engine/node-templates';
	import { CATEGORY_COLORS } from '$lib/engine/types';

	let search = $state('');
	let expandedCat = $state<string | null>(null);

	// Group templates by category, filter by search
	const categories = $derived.by(() => {
		const cats = new Map<string, NodeTemplate[]>();
		for (const tmpl of ALL_NODES) {
			if (search && !tmpl.label.toLowerCase().includes(search.toLowerCase())) continue;
			const cat = tmpl.category;
			if (!cats.has(cat)) cats.set(cat, []);
			cats.get(cat)!.push(tmpl);
		}
		return cats;
	});
</script>

<div class="palette">
	<div class="palette-search">
		<input type="text" placeholder="Search nodes..." bind:value={search} />
	</div>
	<div class="palette-list">
		{#each [...categories] as [cat, templates]}
			<div class="pal-category">
				<button class="pal-cat-header" onclick={() => expandedCat = expandedCat === cat ? null : cat}>
					<span class="pal-cat-dot" style:background={CATEGORY_COLORS[cat as keyof typeof CATEGORY_COLORS]?.darkAccent || '#888'}></span>
					<span>{cat.toUpperCase()}</span>
					<span class="pal-cat-count">({templates.length})</span>
				</button>
				{#if expandedCat === cat || search.length > 0}
					{#each templates as tmpl}
						<div class="pal-item" role="listitem" draggable="true"
							 ondragstart={(e) => e.dataTransfer?.setData('application/bp-node', tmpl.label)}>
							<span class="pal-dot" style:background={CATEGORY_COLORS[tmpl.category]?.darkAccent}></span>
							<div class="pal-info">
								<span class="pal-name">{tmpl.label}</span>
								<span class="pal-desc">{tmpl.description}</span>
							</div>
						</div>
					{/each}
				{/if}
			</div>
		{/each}
	</div>
</div>

<style>
	.palette {
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
	}

	.palette-search {
		padding: var(--spacing-md);
		border-bottom: 1px solid var(--border);
	}

	.palette-search input {
		width: 100%;
		padding: var(--spacing-sm) var(--spacing-md);
		background: var(--bg-raised);
		border: 1px solid var(--border);
		color: var(--text-primary);
		font-family: var(--font-mono);
		font-size: var(--font-size-sm);
		outline: none;
		transition: border-color var(--transition-fast);
	}

	.palette-search input::placeholder {
		color: var(--text-tertiary);
	}

	.palette-search input:focus {
		border-color: var(--accent);
	}

	.palette-list {
		flex: 1;
		overflow-y: auto;
		padding: var(--spacing-sm) 0;
	}

	.pal-category {
		display: flex;
		flex-direction: column;
	}

	.pal-cat-header {
		display: flex;
		flex-direction: row;
		align-items: center;
		gap: 6px;
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		color: var(--text-secondary);
		padding: 6px 8px;
		width: 100%;
		text-align: left;
		cursor: pointer;
		transition: color var(--transition-fast);
	}

	.pal-cat-header:hover {
		color: var(--text-primary);
	}

	.pal-cat-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.pal-cat-count {
		margin-left: auto;
		color: var(--text-tertiary);
		font-size: 9px;
	}

	.pal-item {
		display: flex;
		flex-direction: row;
		align-items: center;
		gap: 8px;
		padding: 4px 8px 4px 16px;
		cursor: grab;
		transition: background var(--transition-fast);
	}

	.pal-item:hover {
		background: var(--bg-raised);
	}

	.pal-item:active {
		cursor: grabbing;
	}

	.pal-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.pal-info {
		display: flex;
		flex-direction: column;
		gap: 1px;
		min-width: 0;
	}

	.pal-name {
		font-size: 11px;
		color: var(--text-primary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.pal-desc {
		font-size: 9px;
		color: var(--text-tertiary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
</style>
