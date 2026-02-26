<script lang="ts">
	import { theme, showSettings } from '$lib/stores/settings';
	import { serverOnline } from '$lib/stores/device';

	let {
		onRun,
		onStop,
		onDeploy,
		onRebuild,
	}: {
		onRun: () => void;
		onStop: () => void;
		onDeploy: () => void;
		onRebuild: () => void;
	} = $props();
</script>

<header class="toolbar">
	<div class="toolbar-left">
		<div class="brand">
			<span class="brand-mark">A</span>
			<span class="brand-name">AUTOMATRIX</span>
		</div>
	</div>

	<div class="toolbar-center">
		<button class="action-btn run-btn" onclick={onRun} title="Run inference">
			<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none">
				<polygon points="5,3 19,12 5,21" />
			</svg>
			<span>Run</span>
		</button>
		<button class="action-btn" onclick={onStop} title="Stop">
			<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none">
				<rect x="4" y="4" width="16" height="16" />
			</svg>
		</button>
		<span class="toolbar-sep"></span>
		<button class="action-btn" onclick={onDeploy} title="Deploy to device">
			<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
				<polyline points="17 8 12 3 7 8" />
				<line x1="12" y1="3" x2="12" y2="15" />
			</svg>
		</button>
		<button class="action-btn" onclick={onRebuild} title="Rebuild">
			<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<polyline points="23 4 23 10 17 10" />
				<polyline points="1 20 1 14 7 14" />
				<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
			</svg>
		</button>
	</div>

	<div class="toolbar-right">
		<div class="server-status" class:online={$serverOnline}>
			<span class="status-dot" class:online={$serverOnline} class:offline={!$serverOnline}></span>
			<span class="status-label">{$serverOnline ? 'CONNECTED' : 'OFFLINE'}</span>
		</div>

		<button class="toolbar-btn" onclick={() => theme.toggle()} title="Toggle theme">
			{#if $theme === 'dark'}
				<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
			{:else}
				<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
			{/if}
		</button>

		<button class="toolbar-btn" onclick={() => showSettings.update(v => !v)} title="Settings">
			<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
		</button>
	</div>
</header>

<style>
	.toolbar {
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

	.toolbar-left, .toolbar-right {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
	}

	.toolbar-center {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
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

	.action-btn {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 3px 8px;
		font-size: var(--font-size-xs);
		font-weight: 500;
		color: var(--text-secondary);
		border-radius: 3px;
		transition: background var(--transition-fast), color var(--transition-fast);
	}

	.action-btn:hover {
		background: var(--bg-overlay);
		color: var(--text-primary);
	}

	.run-btn {
		background: var(--success);
		color: #fff;
		padding: 4px 10px;
	}

	.run-btn:hover {
		background: var(--success);
		filter: brightness(1.15);
		color: #fff;
	}

	.toolbar-sep {
		width: 1px;
		height: 16px;
		background: var(--border);
		margin: 0 4px;
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

	.toolbar-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 28px;
		height: 28px;
		color: var(--text-secondary);
		transition: color var(--transition-fast);
	}

	.toolbar-btn:hover {
		color: var(--accent);
	}
</style>
