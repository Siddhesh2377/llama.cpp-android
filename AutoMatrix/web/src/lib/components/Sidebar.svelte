<script lang="ts">
	import { activeNav, type NavItem } from '$lib/stores/settings';

	const navItems: { id: NavItem; label: string; icon: string }[] = [
		{ id: 'models', label: 'Models', icon: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5' },
		{ id: 'editor', label: 'Editor', icon: 'M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z' },
		{ id: 'settings', label: 'Settings', icon: 'M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z' }
	];

	function select(id: NavItem) {
		activeNav.set(id);
	}
</script>

<nav class="sidebar">
	{#each navItems as item}
		<button
			class="nav-btn"
			class:active={$activeNav === item.id}
			on:click={() => select(item.id)}
			title={item.label}
		>
			<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
				<path d={item.icon} />
			</svg>
			{#if $activeNav === item.id}
				<div class="active-indicator"></div>
			{/if}
		</button>
	{/each}
</nav>

<style>
	.sidebar {
		width: var(--sidebar-w);
		display: flex;
		flex-direction: column;
		align-items: center;
		padding: var(--spacing-md) 0;
		gap: var(--spacing-xs);
		background: var(--bg-surface);
		border-right: 1px solid var(--border);
	}

	.nav-btn {
		position: relative;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 34px;
		height: 34px;
		color: var(--text-tertiary);
		transition: color var(--transition-fast);
	}

	.nav-btn:hover {
		color: var(--text-secondary);
	}

	.nav-btn.active {
		color: var(--accent);
	}

	.active-indicator {
		position: absolute;
		left: -7px;
		top: 50%;
		transform: translateY(-50%);
		width: 2px;
		height: 16px;
		background: var(--accent);
	}
</style>
