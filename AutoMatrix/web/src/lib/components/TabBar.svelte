<script lang="ts">
	import { activeTab, type Tab } from '$lib/stores/settings';

	const tabs: { id: Tab; label: string; icon: string }[] = [
		{ id: 'architecture', label: 'ARCHITECTURE', icon: 'M3 12l3-3 3 3 3-3 3 3 3-3 3 3' },
		{ id: 'graph', label: 'GRAPH', icon: 'M12 3v18M3 12h18M5.5 5.5l13 13M18.5 5.5l-13 13' },
		{ id: 'training', label: 'PLUGINS', icon: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5' },
		{ id: 'surgery', label: 'SURGERY', icon: 'M6 3v12M18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM18 9a9 9 0 0 1-9 9' },
		{ id: 'simulation', label: 'SIMULATION', icon: 'M5 3l14 9-14 9V3z' }
	];
</script>

<div class="tabbar">
	{#each tabs as tab}
		<button
			class="tab"
			class:active={$activeTab === tab.id}
			on:click={() => activeTab.set(tab.id)}
		>
			<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<path d={tab.icon} />
			</svg>
			<span>{tab.label}</span>
		</button>
	{/each}
</div>

<style>
	.tabbar {
		height: var(--tabbar-h);
		display: flex;
		align-items: stretch;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		user-select: none;
		overflow-x: auto;
	}

	.tab {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 0 var(--spacing-xl);
		font-size: var(--font-size-xs);
		font-weight: 500;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
		border-bottom: 2px solid transparent;
		transition: all var(--transition-fast);
		white-space: nowrap;
	}

	.tab:hover {
		color: var(--text-secondary);
		background: var(--accent-glow);
	}

	.tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
		background: var(--accent-glow);
	}

	.tab svg {
		opacity: 0.7;
	}

	.tab.active svg {
		opacity: 1;
	}
</style>
