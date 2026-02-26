<script lang="ts">
	import { toggleToolWindow, type ToolWindowDef } from './store';

	let {
		windows,
		position,
	}: {
		windows: ToolWindowDef[];
		position: 'left' | 'right';
	} = $props();

	// Inline SVG paths for tool window icons (Tabler-style, 24x24 viewBox)
	const iconPaths: Record<string, string> = {
		'folder':        'M5 4h4l3 3h7a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z',
		'components':    'M3 12l3 3 3-3-3-3zM15 12l3 3 3-3-3-3zM9 6l3 3 3-3-3-3zM9 18l3 3 3-3-3-3z',
		'puzzle':        'M4 7h3a1 1 0 0 0 1-1V4a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v2a1 1 0 0 0 1 1h3a1 1 0 0 1 1 1v3a1 1 0 0 0 1 1h2a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1h-2a1 1 0 0 0-1 1v3a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1v-3a1 1 0 0 0-1-1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z',
		'device-mobile': 'M7 3h10a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM11 18h2',
		'test-pipe':     'M7 3v4M17 3v4M3 7h18M5 7v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7M9 14l2 2 4-4',
		'adjustments':   'M6 4v4M6 12v8M12 4v12M12 20v0M18 4v8M18 16v4M4 8h4M10 16h4M16 12h4',
		'chart-bar':     'M3 20h18M5 20V10M9 20V4M13 20V14M17 20V8',
		'scissors':      'M6 7a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM20 4L8.12 15.88M14.47 14.48L20 20M8.12 8.12L12 12',
		'terminal-2':    'M5 7l5 5-5 5M13 17h6',
		'code':          'M7 8l-4 4 4 4M17 8l4 4-4 4M14 4l-4 16',
	};
</script>

<div class="tw-bar" class:tw-bar-left={position === 'left'} class:tw-bar-right={position === 'right'}>
	{#each windows as win (win.id)}
		<button
			class="tw-btn"
			class:active={win.open}
			class:indicator-left={position === 'left' && win.open}
			class:indicator-right={position === 'right' && win.open}
			title={win.label}
			onclick={() => toggleToolWindow(win.id)}
		>
			<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<path d={iconPaths[win.icon] || iconPaths['folder']} />
			</svg>
		</button>
	{/each}
</div>

<style>
	.tw-bar {
		display: flex;
		flex-direction: column;
		width: 28px;
		min-width: 28px;
		background: var(--bg-surface);
		padding: 4px 0;
		gap: 2px;
		align-items: center;
	}

	.tw-bar-left {
		border-right: 1px solid var(--border);
	}

	.tw-bar-right {
		border-left: 1px solid var(--border);
	}

	.tw-btn {
		width: 24px;
		height: 24px;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--text-tertiary);
		cursor: pointer;
		border: none;
		background: none;
		padding: 0;
		position: relative;
		transition: color var(--transition-fast);
	}

	.tw-btn:hover {
		color: var(--text-secondary);
	}

	.tw-btn.active {
		color: var(--accent);
	}

	.tw-btn.indicator-left::after {
		content: '';
		position: absolute;
		left: -2px;
		top: 4px;
		bottom: 4px;
		width: 2px;
		background: var(--accent);
	}

	.tw-btn.indicator-right::after {
		content: '';
		position: absolute;
		right: -2px;
		top: 4px;
		bottom: 4px;
		width: 2px;
		background: var(--accent);
	}
</style>
