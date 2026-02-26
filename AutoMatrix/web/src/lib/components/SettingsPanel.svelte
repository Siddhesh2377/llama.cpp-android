<script lang="ts">
	import { theme, zoom, showSettings } from '$lib/stores/settings';

	let zoomValue: number;
	zoom.subscribe(v => zoomValue = v);
</script>

{#if $showSettings}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="settings-overlay" on:click={() => showSettings.set(false)}>
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="settings-panel" on:click|stopPropagation>
			<div class="panel-header">
				<span class="label">Settings</span>
				<button class="close-btn" on:click={() => showSettings.set(false)}>
					<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
				</button>
			</div>
			<div class="settings-body">
				<!-- Theme -->
				<div class="setting-group">
					<div class="setting-label">Theme</div>
					<div class="theme-toggle">
						<button
							class="theme-btn"
							class:active={$theme === 'dark'}
							on:click={() => theme.set('dark')}
						>
							<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
							Dark
						</button>
						<button
							class="theme-btn"
							class:active={$theme === 'light'}
							on:click={() => theme.set('light')}
						>
							<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
							Light
						</button>
					</div>
				</div>

				<!-- Zoom -->
				<div class="setting-group">
					<div class="setting-label">
						Zoom
						<span class="setting-value">{Math.round(zoomValue * 100)}%</span>
					</div>
					<div class="zoom-control">
						<span class="zoom-label">50%</span>
						<input
							type="range"
							min="0.5"
							max="2"
							step="0.05"
							value={zoomValue}
							on:input={(e) => zoom.set(parseFloat(e.currentTarget.value))}
						/>
						<span class="zoom-label">200%</span>
					</div>
					<button class="reset-btn" on:click={() => zoom.set(1)}>Reset to 100%</button>
				</div>

				<!-- About -->
				<div class="setting-group">
					<div class="setting-label">About</div>
					<div class="about-info">
						<div class="about-row">
							<span class="dim">Version</span>
							<span>0.3.0</span>
						</div>
						<div class="about-row">
							<span class="dim">Engine</span>
							<span>gguf-engine</span>
						</div>
						<div class="about-row">
							<span class="dim">Backend</span>
							<span>GGML (CPU NEON + OpenCL)</span>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.settings-overlay {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.5);
		z-index: 1000;
		display: flex;
		justify-content: flex-end;
	}

	.settings-panel {
		width: 320px;
		height: 100%;
		background: var(--bg-surface);
		border-left: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		animation: slide-in 150ms ease;
	}

	@keyframes slide-in {
		from { transform: translateX(100%); }
		to { transform: translateX(0); }
	}

	.settings-body {
		flex: 1;
		overflow-y: auto;
		padding: var(--spacing-xl);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-2xl);
	}

	.setting-group {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-md);
	}

	.setting-label {
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-secondary);
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.setting-value {
		color: var(--accent);
		font-weight: 500;
	}

	.theme-toggle {
		display: flex;
		gap: var(--spacing-sm);
	}

	.theme-btn {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 6px;
		padding: var(--spacing-md) var(--spacing-lg);
		border: 1px solid var(--border);
		font-size: var(--font-size-sm);
		color: var(--text-secondary);
		transition: all var(--transition-fast);
	}

	.theme-btn:hover {
		border-color: var(--text-tertiary);
		color: var(--text-primary);
	}

	.theme-btn.active {
		border-color: var(--accent);
		color: var(--accent);
		background: var(--accent-dim);
	}

	.zoom-control {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
	}

	.zoom-control input[type="range"] {
		flex: 1;
	}

	.zoom-label {
		font-size: var(--font-size-xs);
		color: var(--text-tertiary);
		width: 32px;
		text-align: center;
	}

	.reset-btn {
		font-size: var(--font-size-xs);
		color: var(--text-secondary);
		padding: var(--spacing-sm) var(--spacing-md);
		border: 1px solid var(--border);
		text-align: center;
	}

	.reset-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.close-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 20px;
		height: 20px;
		color: var(--text-tertiary);
		margin-left: auto;
	}

	.close-btn:hover { color: var(--text-secondary); }

	.about-info {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.about-row {
		display: flex;
		justify-content: space-between;
		font-size: var(--font-size-sm);
	}
</style>
