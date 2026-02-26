<script lang="ts">
	import { primaryDevice } from '$lib/stores/device';
	import { serverOnline, systemInfo } from '$lib/stores/device';
	import { model, formatBytes } from '$lib/stores/model';
	import { activeTab } from '$lib/stores/settings';
</script>

<footer class="statusbar">
	<div class="statusbar-left">
		<!-- Device info -->
		{#if $primaryDevice}
			<div class="status-item">
				<span class="status-dot online glow-pulse"></span>
				<span>{$primaryDevice.model || $primaryDevice.serial}</span>
			</div>
			{#if $primaryDevice.chipset}
				<div class="status-item dim">
					<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2"/></svg>
					<span>{$primaryDevice.chipset}</span>
				</div>
			{/if}
			{#if $primaryDevice.ram}
				<div class="status-item dim">
					<span>{$primaryDevice.ram}</span>
				</div>
			{/if}
		{:else}
			<div class="status-item">
				<span class="status-dot offline"></span>
				<span class="dim">No device</span>
			</div>
		{/if}

		<div class="sep"></div>

		<!-- Model info -->
		{#if $model}
			<div class="status-item accent">
				<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/></svg>
				<span>{$model.name || $model.arch}</span>
			</div>
			<div class="status-item dim">{$model.quantType}</div>
			<div class="status-item dim">{formatBytes($model.fileSize)}</div>
		{/if}
	</div>

	<div class="statusbar-right">
		{#if $systemInfo}
			<div class="status-item dim">
				<span>{$systemInfo.cpuModel}</span>
			</div>
		{/if}

		<div class="status-item dim">
			<span>{$activeTab.toUpperCase()}</span>
		</div>

		<div class="status-item" style="color: var(--text-tertiary)">
			AutoMatrix v0.3.0
		</div>
	</div>
</footer>

<style>
	.statusbar {
		height: var(--statusbar-h);
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 var(--spacing-md);
		background: var(--bg-raised);
		border-top: 1px solid var(--border);
		font-size: var(--font-size-xs);
		user-select: none;
		gap: var(--spacing-lg);
	}

	.statusbar-left, .statusbar-right {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		overflow: hidden;
	}

	.statusbar-left {
		flex: 1;
		min-width: 0;
	}

	.status-item {
		display: flex;
		align-items: center;
		gap: 4px;
		white-space: nowrap;
		flex-shrink: 0;
	}

	.sep {
		width: 1px;
		height: 12px;
		background: var(--border);
		flex-shrink: 0;
	}
</style>
