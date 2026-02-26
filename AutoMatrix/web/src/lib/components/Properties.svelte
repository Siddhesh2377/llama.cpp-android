<script lang="ts">
	import { model, formatBytes } from '$lib/stores/model';

	interface PropGroup {
		label: string;
		props: { key: string; value: string; tag?: string }[];
	}

	$: groups = $model ? buildGroups($model) : [];

	function buildGroups(m: NonNullable<typeof $model>): PropGroup[] {
		return [
			{
				label: 'Format',
				props: [
					{ key: 'Type', value: 'GGUF' },
					{ key: 'Version', value: '3' },
					{ key: 'Endian', value: 'Little' },
					{ key: 'File Size', value: formatBytes(m.fileSize) },
				]
			},
			{
				label: 'Model',
				props: [
					{ key: 'Architecture', value: m.arch, tag: 'arch' },
					{ key: 'Parameters', value: m.params },
					{ key: 'Quantization', value: m.quantType, tag: 'quant' },
					{ key: 'Tensor Count', value: String(m.tensorCount) },
				]
			},
			{
				label: 'Config',
				props: [
					{ key: 'Embedding', value: String(m.embeddingSize) },
					{ key: 'Layers', value: String(m.layerCount) },
					{ key: 'Heads', value: `${m.headCount} / ${m.headCountKV} KV` },
					{ key: 'Context', value: String(m.contextLength) },
					{ key: 'Vocab', value: String(m.vocabSize) },
				]
			},
			{
				label: 'Workspace',
				props: [
					{ key: 'Mode', value: 'Inspect', tag: 'mode' },
					{ key: 'Backend', value: 'CPU (NEON)' },
					{ key: 'Threads', value: '4' },
				]
			}
		];
	}
</script>

<div class="panel">
	<div class="panel-header">
		<span class="label">Properties</span>
	</div>
	<div class="panel-body">
		{#if $model}
			<div class="prop-groups stagger-reveal">
				{#each groups as group}
					<div class="prop-group">
						<div class="prop-group-label">{group.label}</div>
						{#each group.props as prop}
							<div class="prop-row">
								<span class="prop-key">{prop.key}</span>
								<span class="prop-value">
									{#if prop.tag}
										<span class="badge accent">{prop.value}</span>
									{:else}
										{prop.value}
									{/if}
								</span>
							</div>
						{/each}
					</div>
				{/each}
			</div>
		{:else}
			<div class="empty-state">
				<span class="empty-text dim">No model selected</span>
			</div>
		{/if}
	</div>
</div>

<style>
	.prop-groups {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	.prop-group {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.prop-group-label {
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
		padding-bottom: var(--spacing-sm);
		border-bottom: 1px solid var(--border-subtle);
		margin-bottom: var(--spacing-sm);
	}

	.prop-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 2px 0;
		gap: var(--spacing-md);
	}

	.prop-key {
		font-size: var(--font-size-sm);
		color: var(--text-secondary);
		flex-shrink: 0;
	}

	.prop-value {
		font-size: var(--font-size-sm);
		color: var(--text-primary);
		text-align: right;
		font-weight: 500;
	}

	.empty-state {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
	}

	.empty-text {
		font-size: var(--font-size-sm);
	}
</style>
