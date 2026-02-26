<script lang="ts">
	import { consoleStore, type LogEntry } from '$lib/stores/console';
	import { showConsole } from '$lib/stores/settings';
	import { onMount, afterUpdate } from 'svelte';

	let scrollContainer: HTMLDivElement;
	let autoScroll = true;

	afterUpdate(() => {
		if (autoScroll && scrollContainer) {
			scrollContainer.scrollTop = scrollContainer.scrollHeight;
		}
	});

	function handleScroll() {
		if (!scrollContainer) return;
		const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
		autoScroll = scrollHeight - scrollTop - clientHeight < 30;
	}

	function formatTime(date: Date): string {
		return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
	}

	const levelColors: Record<string, string> = {
		info: 'var(--text-secondary)',
		warn: 'var(--warning)',
		error: 'var(--error)',
		success: 'var(--success)',
		debug: 'var(--text-tertiary)'
	};

	const levelLabels: Record<string, string> = {
		info: 'INF',
		warn: 'WRN',
		error: 'ERR',
		success: 'OK ',
		debug: 'DBG'
	};
</script>

{#if $showConsole}
	<div class="console">
		<div class="panel-header">
			<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
			<span class="label">Console</span>
			<span class="dim" style="font-size: var(--font-size-xs)">{$consoleStore.length}</span>
			<div style="flex:1"></div>
			<button class="console-action" on:click={() => consoleStore.clear()} title="Clear">
				<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
			</button>
			<button class="console-action" on:click={() => showConsole.set(false)} title="Close">
				<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
			</button>
		</div>
		<div class="console-body" bind:this={scrollContainer} on:scroll={handleScroll}>
			{#each $consoleStore as entry (entry.id)}
				<div class="log-line">
					<span class="log-time">{formatTime(entry.timestamp)}</span>
					<span class="log-level" style="color: {levelColors[entry.level]}">{levelLabels[entry.level]}</span>
					{#if entry.source}
						<span class="log-source">[{entry.source}]</span>
					{/if}
					<span class="log-msg" style="color: {entry.level === 'error' ? 'var(--error)' : entry.level === 'success' ? 'var(--success)' : 'var(--text-primary)'}">{entry.message}</span>
				</div>
			{/each}
			{#if $consoleStore.length === 0}
				<div class="log-line dim">Ready.</div>
			{/if}
		</div>
	</div>
{:else}
	<button class="console-collapsed" on:click={() => showConsole.set(true)}>
		<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
		<span>CONSOLE</span>
		{#if $consoleStore.length > 0}
			<span class="count">{$consoleStore.length}</span>
		{/if}
	</button>
{/if}

<style>
	.console {
		height: var(--console-h);
		min-height: var(--console-h);
		display: flex;
		flex-direction: column;
		background: var(--bg-surface);
		border-top: 1px solid var(--border);
	}

	.console-body {
		flex: 1;
		overflow-y: auto;
		padding: var(--spacing-sm) var(--spacing-md);
		font-size: var(--font-size-xs);
		line-height: 1.6;
	}

	.log-line {
		display: flex;
		gap: 8px;
		white-space: nowrap;
	}

	.log-time {
		color: var(--text-tertiary);
		flex-shrink: 0;
	}

	.log-level {
		font-weight: 600;
		flex-shrink: 0;
		width: 24px;
	}

	.log-source {
		color: var(--text-tertiary);
		flex-shrink: 0;
	}

	.log-msg {
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.console-action {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 18px;
		height: 18px;
		color: var(--text-tertiary);
	}

	.console-action:hover {
		color: var(--text-secondary);
	}

	.console-collapsed {
		display: flex;
		align-items: center;
		gap: 6px;
		height: 22px;
		padding: 0 var(--spacing-md);
		border-top: 1px solid var(--border);
		background: var(--bg-surface);
		font-size: var(--font-size-xs);
		color: var(--text-tertiary);
		letter-spacing: 0.08em;
		font-weight: 500;
		width: 100%;
		text-align: left;
	}

	.console-collapsed:hover {
		color: var(--text-secondary);
		background: var(--bg-raised);
	}

	.count {
		background: var(--accent-dim);
		color: var(--accent);
		padding: 0 4px;
		font-size: 9px;
		font-weight: 600;
	}
</style>
