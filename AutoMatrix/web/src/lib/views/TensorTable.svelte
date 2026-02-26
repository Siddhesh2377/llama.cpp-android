<script lang="ts">
	import { model, formatBytes, type TensorInfo } from '$lib/stores/model';

	let searchQuery = '';
	let sortBy: 'name' | 'type' | 'size' = 'name';
	let sortAsc = true;
	let typeFilter = '';

	// Collect unique types
	$: allTypes = $model?.tensors
		? [...new Set($model.tensors.map(t => t.type))].sort()
		: [];

	// Filter + sort
	$: filteredTensors = (() => {
		if (!$model?.tensors) return [];
		let list = [...$model.tensors];

		// Text search
		if (searchQuery) {
			const q = searchQuery.toLowerCase();
			list = list.filter(t => t.name.toLowerCase().includes(q));
		}

		// Type filter
		if (typeFilter) {
			list = list.filter(t => t.type === typeFilter);
		}

		// Sort
		list.sort((a, b) => {
			let cmp = 0;
			if (sortBy === 'name') cmp = a.name.localeCompare(b.name);
			else if (sortBy === 'type') cmp = a.type.localeCompare(b.type);
			else if (sortBy === 'size') cmp = a.size - b.size;
			return sortAsc ? cmp : -cmp;
		});

		return list;
	})();

	// Memory breakdown by type
	$: memoryByType = (() => {
		if (!$model?.tensors) return [];
		const map = new Map<string, number>();
		for (const t of $model.tensors) {
			map.set(t.type, (map.get(t.type) || 0) + t.size);
		}
		return [...map.entries()]
			.sort((a, b) => b[1] - a[1])
			.map(([type, size]) => ({ type, size, pct: (size / $model!.fileSize * 100) }));
	})();

	function toggleSort(col: typeof sortBy) {
		if (sortBy === col) {
			sortAsc = !sortAsc;
		} else {
			sortBy = col;
			sortAsc = true;
		}
	}

	function shapeStr(shape: number[]): string {
		if (!shape || shape.length === 0) return '-';
		return shape.join(' x ');
	}
</script>

<div class="tensor-table-view">
	{#if !$model}
		<div class="empty-state">
			<span class="dim">Load a model to view tensors</span>
		</div>
	{:else}
		<!-- Memory breakdown -->
		<div class="memory-bar">
			{#each memoryByType as seg}
				<div
					class="memory-segment"
					style="flex: {seg.size}"
					title="{seg.type}: {formatBytes(seg.size)} ({seg.pct.toFixed(1)}%)"
				>
					<span class="seg-label">{seg.type}</span>
				</div>
			{/each}
		</div>

		<!-- Filters -->
		<div class="table-toolbar">
			<input
				type="text"
				placeholder="Search tensors..."
				bind:value={searchQuery}
				class="search-input"
			/>
			<select bind:value={typeFilter} class="type-filter">
				<option value="">All types</option>
				{#each allTypes as t}
					<option value={t}>{t}</option>
				{/each}
			</select>
			<span class="tensor-count">{filteredTensors.length} / {$model.tensors?.length || 0}</span>
		</div>

		<!-- Table -->
		<div class="table-wrapper">
			<table>
				<thead>
					<tr>
						<th class="col-name sortable" on:click={() => toggleSort('name')}>
							Name {sortBy === 'name' ? (sortAsc ? '▲' : '▼') : ''}
						</th>
						<th class="col-type sortable" on:click={() => toggleSort('type')}>
							Type {sortBy === 'type' ? (sortAsc ? '▲' : '▼') : ''}
						</th>
						<th class="col-shape">Shape</th>
						<th class="col-size sortable" on:click={() => toggleSort('size')}>
							Size {sortBy === 'size' ? (sortAsc ? '▲' : '▼') : ''}
						</th>
					</tr>
				</thead>
				<tbody>
					{#each filteredTensors as tensor (tensor.name)}
						<tr>
							<td class="col-name">
								<span class="tensor-name">{tensor.name}</span>
							</td>
							<td class="col-type">
								<span class="badge accent">{tensor.type}</span>
							</td>
							<td class="col-shape dim">{shapeStr(tensor.shape)}</td>
							<td class="col-size">{formatBytes(tensor.size)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<style>
	.tensor-table-view {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.empty-state {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	/* Memory bar */
	.memory-bar {
		display: flex;
		height: 20px;
		min-height: 20px;
		border-bottom: 1px solid var(--border);
		overflow: hidden;
	}

	.memory-segment {
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 2px;
		background: var(--accent-dim);
		border-right: 1px solid var(--bg-base);
		overflow: hidden;
		transition: background var(--transition-fast);
	}

	.memory-segment:hover {
		background: var(--accent);
	}

	.memory-segment:hover .seg-label {
		color: var(--bg-base);
	}

	.seg-label {
		font-size: 8px;
		font-weight: 600;
		letter-spacing: 0.04em;
		color: var(--accent);
		white-space: nowrap;
		overflow: hidden;
	}

	/* Toolbar */
	.table-toolbar {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-sm) var(--spacing-md);
		border-bottom: 1px solid var(--border);
		background: var(--bg-surface);
	}

	.search-input {
		flex: 1;
		padding: var(--spacing-sm) var(--spacing-md);
		font-size: var(--font-size-xs);
	}

	.type-filter {
		padding: var(--spacing-sm) var(--spacing-md);
		font-size: var(--font-size-xs);
		min-width: 100px;
	}

	.tensor-count {
		font-size: var(--font-size-xs);
		color: var(--text-tertiary);
		white-space: nowrap;
	}

	/* Table */
	.table-wrapper {
		flex: 1;
		overflow: auto;
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--font-size-xs);
	}

	thead {
		position: sticky;
		top: 0;
		z-index: 5;
	}

	th {
		padding: var(--spacing-sm) var(--spacing-md);
		text-align: left;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--text-secondary);
		background: var(--bg-raised);
		border-bottom: 1px solid var(--border);
		white-space: nowrap;
		user-select: none;
	}

	th.sortable {
		cursor: pointer;
	}

	th.sortable:hover {
		color: var(--accent);
	}

	td {
		padding: 3px var(--spacing-md);
		border-bottom: 1px solid var(--border-subtle);
		white-space: nowrap;
	}

	tr:hover td {
		background: var(--accent-glow);
	}

	.col-name { min-width: 250px; }
	.col-type { width: 80px; }
	.col-shape { width: 150px; }
	.col-size { width: 80px; text-align: right; }

	.tensor-name {
		color: var(--text-primary);
		font-weight: 400;
	}
</style>
