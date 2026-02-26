<script lang="ts">
	import { theme, showSettings, type Theme } from '$lib/stores/settings';
	import { serverOnline } from '$lib/stores/device';

	function toggleTheme() {
		theme.toggle();
	}
</script>

<header class="topbar">
	<div class="topbar-left">
		<div class="brand">
			<span class="brand-mark">A</span>
			<span class="brand-name">AUTOMATRIX</span>
			<span class="brand-sep">|</span>
			<span class="brand-version">V0.3.0</span>
		</div>
	</div>

	<div class="topbar-center">
		<!-- future: search / command palette -->
	</div>

	<div class="topbar-right">
		<div class="server-status" class:online={$serverOnline}>
			<span class="status-dot" class:online={$serverOnline} class:offline={!$serverOnline}></span>
			<span class="status-label">{$serverOnline ? 'CONNECTED' : 'OFFLINE'}</span>
		</div>

		<button class="topbar-btn" on:click={toggleTheme} title="Toggle theme">
			{#if $theme === 'dark'}
				<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
			{:else}
				<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
			{/if}
		</button>

		<button class="topbar-btn" on:click={() => showSettings.update(v => !v)} title="Settings">
			<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
		</button>
	</div>
</header>

<style>
	.topbar {
		height: var(--topbar-h);
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 var(--spacing-md);
		background: var(--bg-raised);
		border-bottom: 1px solid var(--border);
		user-select: none;
		gap: var(--spacing-lg);
	}

	.topbar-left, .topbar-right {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
	}

	.topbar-center {
		flex: 1;
	}

	.brand {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
		font-size: var(--font-size-sm);
		font-weight: 600;
		letter-spacing: 0.12em;
	}

	.brand-mark {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 20px;
		height: 20px;
		background: var(--accent);
		color: var(--bg-base);
		font-size: var(--font-size-sm);
		font-weight: 700;
	}

	.brand-name {
		color: var(--text-primary);
	}

	.brand-sep {
		color: var(--text-tertiary);
		margin: 0 2px;
	}

	.brand-version {
		color: var(--text-tertiary);
		font-weight: 400;
		font-size: var(--font-size-xs);
	}

	.server-status {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: var(--font-size-xs);
		color: var(--text-tertiary);
		letter-spacing: 0.06em;
	}

	.server-status.online {
		color: var(--success);
	}

	.status-label {
		font-weight: 500;
	}

	.topbar-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 28px;
		height: 28px;
		color: var(--text-secondary);
		transition: color var(--transition-fast);
	}

	.topbar-btn:hover {
		color: var(--accent);
	}
</style>
