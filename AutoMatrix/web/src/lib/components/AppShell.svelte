<script lang="ts">
	import TopBar from './TopBar.svelte';
	import Sidebar from './Sidebar.svelte';
	import TabBar from './TabBar.svelte';
	import Outliner from './Outliner.svelte';
	import Properties from './Properties.svelte';
	import Console from './Console.svelte';
	import StatusBar from './StatusBar.svelte';
	import SettingsPanel from './SettingsPanel.svelte';
	import Architecture from '$lib/views/Architecture.svelte';
	import Graph from '$lib/views/Graph.svelte';
	import Simulation from '$lib/views/Simulation.svelte';
	import Plugins from '$lib/views/Plugins.svelte';
	import Editor from '$lib/views/Editor.svelte';
	import Placeholder from '$lib/views/Placeholder.svelte';
	import { activeTab, activeNav } from '$lib/stores/settings';
	import { startPolling, stopPolling, getPlugins } from '$lib/api/client';
	import { consoleStore } from '$lib/stores/console';
	import { plugins } from '$lib/stores/plugins';
	import { theme, zoom } from '$lib/stores/settings';
	import { onMount, onDestroy } from 'svelte';
	import { browser } from '$app/environment';

	onMount(() => {
		// Apply persisted theme & zoom
		if (browser) {
			const savedTheme = localStorage.getItem('amx-theme');
			if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);

			const savedZoom = localStorage.getItem('amx-zoom');
			if (savedZoom) document.documentElement.style.setProperty('--zoom', savedZoom);
		}

		consoleStore.info('AutoMatrix v0.3.0 initialized', 'system');
		consoleStore.info('Connecting to backend...', 'system');
		startPolling();

		// Load plugins
		getPlugins().then(p => {
			if (p.length > 0) {
				plugins.set(p);
				consoleStore.info(`Loaded ${p.length} plugins (AMXP)`, 'plugins');
			}
		});
	});

	onDestroy(() => {
		stopPolling();
	});
</script>

<div class="app-shell">
	<TopBar />

	<div class="app-body">
		<Sidebar />

		<div class="main-area">
			<TabBar />

			<div class="workspace">
				{#if $activeNav === 'editor'}
					<!-- Full-width editor mode -->
					<div class="viewport-area" style="flex:1">
						<Editor />
					</div>
				{:else}
					<!-- Left panel: Outliner -->
					<div class="panel-left">
						<Outliner />
					</div>

					<!-- Center: Viewport -->
					<div class="viewport-area">
						{#if $activeTab === 'architecture'}
							<Architecture />
						{:else if $activeTab === 'graph'}
							<Graph />
						{:else if $activeTab === 'training'}
							<Plugins />
						{:else if $activeTab === 'surgery'}
							<Placeholder title="Surgery" icon="M6 3v12M18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6z" />
						{:else if $activeTab === 'simulation'}
							<Simulation />
						{/if}
					</div>

					<!-- Right panel: Properties -->
					<div class="panel-right">
						<Properties />
					</div>
				{/if}
			</div>

			<Console />
		</div>
	</div>

	<StatusBar />
	<SettingsPanel />
</div>

<style>
	.app-shell {
		display: flex;
		flex-direction: column;
		height: 100vh;
		width: 100vw;
		overflow: hidden;
		position: relative;
	}

	.app-body {
		flex: 1;
		display: flex;
		overflow: hidden;
	}

	.main-area {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.workspace {
		flex: 1;
		display: flex;
		overflow: hidden;
	}

	.panel-left {
		width: 260px;
		min-width: 200px;
		display: flex;
		flex-direction: column;
		border-right: 1px solid var(--border);
	}

	.panel-left > :global(*) {
		flex: 1;
		display: flex;
		flex-direction: column;
	}

	.viewport-area {
		flex: 1;
		display: flex;
		flex-direction: column;
		background: var(--bg-base);
		min-width: 0;
		overflow: hidden;
	}

	.panel-right {
		width: 240px;
		min-width: 180px;
		display: flex;
		flex-direction: column;
		border-left: 1px solid var(--border);
	}

	.panel-right > :global(*) {
		flex: 1;
		display: flex;
		flex-direction: column;
	}
</style>
