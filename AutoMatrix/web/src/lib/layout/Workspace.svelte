<script lang="ts">
	import Toolbar from './Toolbar.svelte';
	import DocumentTabs from './DocumentTabs.svelte';
	import ToolWindowBar from './ToolWindowBar.svelte';
	import {
		leftWindow,
		rightWindow,
		bottomWindow,
		leftWindows,
		rightWindows,
		bottomWindows,
		toggleToolWindow,
		type ToolWindowDef,
	} from './store';

	import Outliner from '$lib/components/Outliner.svelte';
	import Properties from '$lib/components/Properties.svelte';
	import Console from '$lib/components/Console.svelte';
	import NodePalette from '$lib/panels/NodePalette.svelte';
	import Plugins from '$lib/views/Plugins.svelte';
	import StatusBar from '$lib/components/StatusBar.svelte';
	import SettingsPanel from '$lib/components/SettingsPanel.svelte';
	import NodeCanvas from '$lib/engine/NodeCanvas.svelte';

	import { documents, activeDocId, openDocument } from '$lib/engine/store';
	import { modelToBpDocument } from '$lib/engine/model-to-graph';
	import { devices } from '$lib/stores/device';
	import { startPolling, stopPolling, getPlugins, getModelInfo, getModelGraph } from '$lib/api/client';
	import { consoleStore } from '$lib/stores/console';
	import { model } from '$lib/stores/model';
	import { plugins } from '$lib/stores/plugins';
	import { theme, zoom, showConsole } from '$lib/stores/settings';
	import { onMount, onDestroy } from 'svelte';
	import { browser } from '$app/environment';

	let unsubModel: (() => void) | null = null;

	async function openModelGraph() {
		const info = await getModelInfo();
		const graph = await getModelGraph();
		if (info && graph) {
			const doc = modelToBpDocument(info, graph);
			openDocument(doc);
			consoleStore.info(`Opened model graph: ${doc.name}`, 'editor');
		}
	}

	onMount(() => {
		if (browser) {
			const savedTheme = localStorage.getItem('amx-theme');
			if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);

			const savedZoom = localStorage.getItem('amx-zoom');
			if (savedZoom) document.documentElement.style.setProperty('--zoom', savedZoom);
		}

		consoleStore.info('AutoMatrix v0.3.0 initialized', 'system');
		consoleStore.info('Connecting to backend...', 'system');
		startPolling();

		getPlugins().then(p => {
			if (p.length > 0) {
				plugins.set(p);
				consoleStore.info(`Loaded ${p.length} plugins (AMXP)`, 'plugins');
			}
		});

		// Sync showConsole with bottom window state
		showConsole.set(true);

		// Auto-open model graph when a model is loaded
		unsubModel = model.subscribe(m => {
			if (m) {
				setTimeout(() => openModelGraph(), 100);
			}
		});
	});

	onDestroy(() => {
		stopPolling();
		if (unsubModel) unsubModel();
	});

	// Action stubs
	function handleRun() {
		consoleStore.info('Run triggered', 'toolbar');
	}

	function handleStop() {
		consoleStore.info('Stop triggered', 'toolbar');
	}

	function handleDeploy() {
		consoleStore.info('Deploy triggered', 'toolbar');
	}

	function handleRebuild() {
		consoleStore.info('Rebuild triggered', 'toolbar');
	}
</script>

<div class="workspace">
	<Toolbar onRun={handleRun} onStop={handleStop} onDeploy={handleDeploy} onRebuild={handleRebuild} />

	<div class="workspace-body">
		<!-- Left tool window bar -->
		<ToolWindowBar windows={$leftWindows} position="left" />

		<!-- Left panel -->
		{#if $leftWindow}
			<div class="left-panel" style="width: {$leftWindow.size}px">
				<div class="panel-header">
					<span>{$leftWindow.label}</span>
				</div>
				<div class="panel-content">
					{#if $leftWindow.id === 'model'}
						<Outliner />
					{:else if $leftWindow.id === 'palette'}
						<NodePalette />
					{:else if $leftWindow.id === 'plugins'}
						<Plugins />
					{:else if $leftWindow.id === 'devices'}
						<div class="device-list">
							{#each $devices as dev (dev.serial)}
								<div class="device-item">
									<span class="device-status" class:online={dev.status === 'online'} class:offline={dev.status !== 'online'}></span>
									<div class="device-info">
										<div class="device-model">{dev.model}</div>
										<div class="device-meta">{dev.serial} &middot; {dev.chipset}</div>
									</div>
								</div>
							{:else}
								<div style="padding:12px;color:var(--text-tertiary);font-size:11px">No devices connected</div>
							{/each}
						</div>
					{:else}
						<div style="padding:12px;color:var(--text-tertiary);font-size:11px">{$leftWindow.label} — coming soon</div>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Center area: document tabs + canvas -->
		<div class="center-area">
			<DocumentTabs />

			<div class="canvas-area">
				{#if $documents.length > 0}
					<NodeCanvas />
				{:else}
					<div class="empty-state">
						<div class="empty-state-inner">
							<span class="empty-mark">A</span>
							<span>Load a model or create a new document</span>
						</div>
					</div>
				{/if}
			</div>

			<!-- Bottom panel -->
			{#if $bottomWindow}
				<div class="bottom-panel" style="height: {$bottomWindow.size}px">
					<div class="bottom-tabs">
						{#each $bottomWindows as bw (bw.id)}
							<button
								class="bottom-tab"
								class:active={bw.open}
								onclick={() => toggleToolWindow(bw.id)}
							>
								{bw.label}
							</button>
						{/each}
					</div>
					<div class="bottom-content">
						{#if $bottomWindow.id === 'console'}
							<Console />
						{:else}
							<div style="padding:12px;color:var(--text-tertiary);font-size:11px">{$bottomWindow.label} — coming soon</div>
						{/if}
					</div>
				</div>
			{:else}
				<!-- Collapsed bottom bar showing all bottom window labels -->
				<div class="bottom-tabs-collapsed">
					{#each $bottomWindows as bw (bw.id)}
						<button class="bottom-tab" onclick={() => toggleToolWindow(bw.id)}>
							{bw.label}
						</button>
					{/each}
				</div>
			{/if}
		</div>

		<!-- Right panel -->
		{#if $rightWindow}
			<div class="right-panel" style="width: {$rightWindow.size}px">
				<div class="panel-header">
					<span>{$rightWindow.label}</span>
				</div>
				<div class="panel-content">
					{#if $rightWindow.id === 'props'}
						<Properties />
					{:else}
						<div style="padding:12px;color:var(--text-tertiary);font-size:11px">{$rightWindow.label} — coming soon</div>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Right tool window bar -->
		<ToolWindowBar windows={$rightWindows} position="right" />
	</div>

	<StatusBar />
	<SettingsPanel />
</div>

<style>
	.workspace {
		display: flex;
		flex-direction: column;
		height: 100vh;
		width: 100vw;
		overflow: hidden;
	}

	.workspace-body {
		flex: 1;
		display: flex;
		overflow: hidden;
	}

	.center-area {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
		overflow: hidden;
	}

	.canvas-area {
		flex: 1;
		display: flex;
		overflow: hidden;
		position: relative;
	}

	.left-panel, .right-panel {
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: var(--bg-surface);
	}

	.left-panel {
		border-right: 1px solid var(--border);
	}

	.right-panel {
		border-left: 1px solid var(--border);
	}

	.panel-header {
		height: 28px;
		min-height: 28px;
		display: flex;
		align-items: center;
		padding: 0 8px;
		background: var(--bg-raised);
		border-bottom: 1px solid var(--border);
		font-size: 10px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--text-secondary);
		user-select: none;
	}

	.panel-content {
		flex: 1;
		overflow: auto;
	}

	.panel-content > :global(*) {
		flex: 1;
		display: flex;
		flex-direction: column;
		height: 100%;
	}

	.bottom-panel {
		border-top: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.bottom-tabs, .bottom-tabs-collapsed {
		display: flex;
		height: 24px;
		min-height: 24px;
		background: var(--bg-raised);
		border-bottom: 1px solid var(--border);
		user-select: none;
	}

	.bottom-tabs-collapsed {
		border-top: 1px solid var(--border);
		border-bottom: none;
	}

	.bottom-tab {
		padding: 0 10px;
		font-size: 10px;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--text-tertiary);
		cursor: pointer;
		background: none;
		border: none;
		border-bottom: 2px solid transparent;
		transition: color var(--transition-fast);
	}

	.bottom-tab:hover {
		color: var(--text-secondary);
	}

	.bottom-tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}

	.bottom-content {
		flex: 1;
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}

	/* Override Console's own showConsole guard — in Workspace, we always show it when bottom panel is open */
	.bottom-content > :global(*) {
		flex: 1;
		display: flex;
		flex-direction: column;
	}

	.empty-state {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--text-tertiary);
		font-size: 13px;
		background: var(--bg-base);
	}

	.empty-state-inner {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 12px;
	}

	.empty-mark {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 40px;
		height: 40px;
		background: var(--accent-dim);
		color: var(--accent);
		font-size: 20px;
		font-weight: 700;
		opacity: 0.6;
	}

	.placeholder-content {
		display: flex;
		align-items: center;
		justify-content: center;
		flex: 1;
		color: var(--text-tertiary);
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.device-list {
		display: flex;
		flex-direction: column;
		padding: 4px 0;
	}

	.device-item {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 10px;
	}

	.device-item:hover {
		background: var(--bg-hover);
	}

	.device-status {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.device-status.online {
		background: var(--green, #4caf50);
	}

	.device-status.offline {
		background: var(--text-tertiary);
	}

	.device-info {
		min-width: 0;
	}

	.device-model {
		font-size: 11px;
		font-weight: 500;
		color: var(--text-primary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.device-meta {
		font-size: 10px;
		color: var(--text-tertiary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
</style>
