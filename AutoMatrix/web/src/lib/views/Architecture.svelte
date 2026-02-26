<script lang="ts">
	import { model, modelLoading, formatBytes } from '$lib/stores/model';
	import { loadModel, scanModels, type ScannedFile } from '$lib/api/client';
	import { serverOnline } from '$lib/stores/device';
	import TensorTable from './TensorTable.svelte';

	let filepath = '';
	let scanDir = '';
	let scannedFiles: ScannedFile[] = [];
	let scanning = false;

	type ArchView = 'diagram' | 'tensors';
	let archView: ArchView = 'diagram';

	async function handleLoad() {
		if (!filepath.trim()) return;
		await loadModel(filepath.trim());
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') handleLoad();
	}

	async function handleScan() {
		if (!scanDir.trim()) return;
		scanning = true;
		scannedFiles = await scanModels(scanDir.trim());
		scanning = false;
	}

	function handleScanKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') handleScan();
	}

	async function selectFile(file: ScannedFile) {
		filepath = file.path;
		await loadModel(file.path);
	}

	function unloadModel() {
		model.set(null);
		scannedFiles = [];
	}

	// Block diagram data from model
	$: blocks = $model ? buildBlocks($model) : [];

	interface Block {
		id: string;
		label: string;
		sublabel?: string;
		type: 'embed' | 'transformer' | 'head' | 'norm';
	}

	function buildBlocks(m: NonNullable<typeof $model>): Block[] {
		const b: Block[] = [];
		b.push({
			id: 'embed',
			label: 'Token Embedding',
			sublabel: `${m.vocabSize} x ${m.embeddingSize}`,
			type: 'embed'
		});
		b.push({
			id: 'blocks',
			label: `Transformer Blocks x${m.layerCount}`,
			sublabel: `${m.headCount}h / ${m.headCountKV}kv / ${m.embeddingSize}d`,
			type: 'transformer'
		});
		b.push({
			id: 'norm',
			label: 'RMS Norm',
			sublabel: `${m.embeddingSize}`,
			type: 'norm'
		});
		b.push({
			id: 'head',
			label: 'LM Head',
			sublabel: `${m.embeddingSize} -> ${m.vocabSize}`,
			type: 'head'
		});
		return b;
	}
</script>

<div class="architecture-view">
	{#if $model}
		<!-- Model loaded -->
		<div class="arch-header">
			<div class="model-title">
				<span class="model-name">{$model.name || $model.arch}</span>
				<span class="badge accent">{$model.quantType}</span>
				<span class="badge info">{$model.params}</span>
			</div>
			<div class="view-toggle">
				<button class="toggle-btn" class:active={archView === 'diagram'} on:click={() => archView = 'diagram'}>
					<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
					BLOCKS
				</button>
				<button class="toggle-btn" class:active={archView === 'tensors'} on:click={() => archView = 'tensors'}>
					<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M3 12h18M3 18h18"/></svg>
					TENSORS
				</button>
			</div>
			<button class="unload-btn" on:click={unloadModel} title="Unload model">
				<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
				UNLOAD
			</button>
		</div>

		{#if archView === 'tensors'}
			<TensorTable />
		{:else}
		<div class="viewport">
			<div class="block-diagram stagger-reveal">
				{#each blocks as block, i}
					<div class="block block-{block.type}">
						<div class="block-label">{block.label}</div>
						{#if block.sublabel}
							<div class="block-sublabel">{block.sublabel}</div>
						{/if}
					</div>
					{#if i < blocks.length - 1}
						<div class="block-connector">
							<div class="connector-line"></div>
							<div class="connector-arrow">&#9660;</div>
						</div>
					{/if}
				{/each}
			</div>

			<div class="model-stats">
				<div class="stat">
					<span class="stat-label">Tensors</span>
					<span class="stat-value">{$model.tensorCount}</span>
				</div>
				<div class="stat">
					<span class="stat-label">Size</span>
					<span class="stat-value">{formatBytes($model.fileSize)}</span>
				</div>
				<div class="stat">
					<span class="stat-label">Context</span>
					<span class="stat-value">{$model.contextLength.toLocaleString()}</span>
				</div>
				<div class="stat">
					<span class="stat-label">Embedding</span>
					<span class="stat-value">{$model.embeddingSize}</span>
				</div>
			</div>

			<!-- Metadata dump -->
			{#if $model.metadata && Object.keys($model.metadata).length > 0}
				<div class="metadata-section">
					<div class="section-header">Raw Metadata</div>
					<div class="metadata-grid">
						{#each Object.entries($model.metadata).slice(0, 30) as [key, value]}
							<div class="meta-row">
								<span class="meta-key">{key}</span>
								<span class="meta-value">{String(value)}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
		{/if}
	{:else}
		<!-- No model: show load prompt + file browser -->
		<div class="load-view">
			<div class="load-hero">
				<div class="hero-icon">
					<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1" stroke-linecap="round">
						<path d="M12 2L2 7l10 5 10-5-10-5z"/>
						<path d="M2 17l10 5 10-5"/>
						<path d="M2 12l10 5 10-5"/>
					</svg>
				</div>
				<h2 class="hero-title">Load a Model</h2>
				<p class="hero-desc">Enter a path directly or scan a directory for GGUF files</p>
			</div>

			<!-- Direct path input -->
			<div class="load-form">
				<div class="form-label">Direct Path</div>
				<div class="input-row">
					<input
						type="text"
						placeholder="/path/to/model.gguf"
						bind:value={filepath}
						on:keydown={handleKeydown}
						class="filepath-input"
					/>
					<button
						class="load-btn"
						on:click={handleLoad}
						disabled={$modelLoading || !filepath.trim()}
					>
						{#if $modelLoading}
							<span class="loading-dots">...</span>
						{:else}
							LOAD
						{/if}
					</button>
				</div>
			</div>

			<!-- Directory scanner -->
			<div class="load-form">
				<div class="form-label">Scan Directory</div>
				<div class="input-row">
					<input
						type="text"
						placeholder="/home/user/models or ~/Downloads"
						bind:value={scanDir}
						on:keydown={handleScanKeydown}
						class="filepath-input"
					/>
					<button
						class="scan-btn"
						on:click={handleScan}
						disabled={scanning || !scanDir.trim() || !$serverOnline}
					>
						{#if scanning}
							<span class="loading-dots">...</span>
						{:else}
							SCAN
						{/if}
					</button>
				</div>
			</div>

			<!-- Scan results -->
			{#if scannedFiles.length > 0}
				<div class="file-list">
					<div class="file-list-header">
						<span>{scannedFiles.length} model{scannedFiles.length !== 1 ? 's' : ''} found</span>
					</div>
					<div class="file-list-body">
						{#each scannedFiles as file}
							<button class="file-row" on:click={() => selectFile(file)}>
								<div class="file-icon">
									<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
								</div>
								<div class="file-info">
									<span class="file-name">{file.name}</span>
									<span class="file-path">{file.path}</span>
								</div>
								<span class="file-size">{formatBytes(file.size)}</span>
							</button>
						{/each}
					</div>
				</div>
			{/if}

			{#if !$serverOnline}
				<div class="offline-notice">
					<span class="status-dot offline"></span>
					<span>Backend server offline. Start amx-server to load models.</span>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.architecture-view {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	/* Load view */
	.load-view {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		padding: var(--spacing-2xl);
		gap: var(--spacing-xl);
		overflow-y: auto;
	}

	.load-hero {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--spacing-lg);
		padding-top: var(--spacing-2xl);
	}

	.hero-icon {
		opacity: 0.5;
		animation: fade-in-up 400ms ease;
	}

	.hero-title {
		font-size: var(--font-size-xl);
		font-weight: 600;
		color: var(--text-primary);
		letter-spacing: 0.04em;
	}

	.hero-desc {
		font-size: var(--font-size-sm);
		color: var(--text-secondary);
	}

	.load-form {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
		width: 100%;
		max-width: 520px;
	}

	.form-label {
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
	}

	.input-row {
		display: flex;
		gap: var(--spacing-sm);
	}

	.filepath-input {
		flex: 1;
		padding: var(--spacing-md) var(--spacing-lg);
		font-size: var(--font-size-sm);
	}

	.load-btn, .scan-btn {
		padding: var(--spacing-md) var(--spacing-xl);
		font-size: var(--font-size-xs);
		font-weight: 700;
		letter-spacing: 0.1em;
		transition: all var(--transition-fast);
		min-width: 72px;
	}

	.load-btn {
		background: var(--accent);
		color: var(--bg-base);
	}

	.scan-btn {
		background: var(--bg-overlay);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}

	.load-btn:hover:not(:disabled) { background: var(--accent-hover); }
	.scan-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
	.load-btn:disabled, .scan-btn:disabled { opacity: 0.4; cursor: not-allowed; }

	.loading-dots {
		animation: glow-pulse 1s ease-in-out infinite;
	}

	/* File list */
	.file-list {
		width: 100%;
		max-width: 520px;
		border: 1px solid var(--border);
		background: var(--bg-surface);
		max-height: 300px;
		display: flex;
		flex-direction: column;
	}

	.file-list-header {
		padding: var(--spacing-sm) var(--spacing-md);
		font-size: var(--font-size-xs);
		color: var(--text-tertiary);
		border-bottom: 1px solid var(--border);
		background: var(--bg-raised);
		font-weight: 500;
		letter-spacing: 0.04em;
	}

	.file-list-body {
		overflow-y: auto;
		flex: 1;
	}

	.file-row {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-md) var(--spacing-lg);
		width: 100%;
		text-align: left;
		border-bottom: 1px solid var(--border-subtle);
		transition: background var(--transition-fast);
	}

	.file-row:hover {
		background: var(--accent-glow);
	}

	.file-row:last-child {
		border-bottom: none;
	}

	.file-icon {
		color: var(--accent);
		flex-shrink: 0;
		opacity: 0.7;
	}

	.file-info {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 1px;
	}

	.file-name {
		font-size: var(--font-size-sm);
		font-weight: 500;
		color: var(--text-primary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.file-path {
		font-size: var(--font-size-xs);
		color: var(--text-tertiary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.file-size {
		font-size: var(--font-size-xs);
		color: var(--text-secondary);
		flex-shrink: 0;
		font-weight: 500;
	}

	.offline-notice {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-md) var(--spacing-lg);
		border: 1px solid var(--border);
		font-size: var(--font-size-xs);
		color: var(--text-secondary);
	}

	/* Architecture header */
	.arch-header {
		display: flex;
		align-items: center;
		gap: var(--spacing-lg);
		padding: var(--spacing-sm) var(--spacing-lg);
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		min-height: 36px;
	}

	.view-toggle {
		display: flex;
		gap: 1px;
		margin-left: auto;
	}

	.toggle-btn {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: var(--spacing-sm) var(--spacing-md);
		font-size: var(--font-size-xs);
		font-weight: 500;
		letter-spacing: 0.06em;
		color: var(--text-tertiary);
		border: 1px solid var(--border);
		transition: all var(--transition-fast);
	}

	.toggle-btn:first-child { border-right: none; }
	.toggle-btn:hover { color: var(--text-secondary); }
	.toggle-btn.active {
		color: var(--accent);
		border-color: var(--accent);
		background: var(--accent-dim);
	}

	/* Viewport (model loaded) */
	.viewport {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		padding: var(--spacing-2xl);
		gap: var(--spacing-2xl);
		overflow-y: auto;
	}

	.model-title {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
	}

	.model-name {
		font-size: var(--font-size-sm);
		font-weight: 600;
		color: var(--text-primary);
	}

	.unload-btn {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: var(--spacing-sm) var(--spacing-md);
		font-size: var(--font-size-xs);
		color: var(--text-tertiary);
		border: 1px solid var(--border);
		letter-spacing: 0.06em;
		transition: all var(--transition-fast);
	}

	.unload-btn:hover {
		border-color: var(--error);
		color: var(--error);
	}

	/* Block diagram */
	.block-diagram {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0;
	}

	.block {
		width: 280px;
		padding: var(--spacing-lg) var(--spacing-xl);
		border: 1px solid var(--border);
		text-align: center;
		background: var(--bg-raised);
		position: relative;
	}

	.block-embed { border-color: var(--info); border-left: 3px solid var(--info); }
	.block-transformer { border-color: var(--accent); border-left: 3px solid var(--accent); background: var(--accent-glow); }
	.block-norm { border-color: var(--text-tertiary); border-left: 3px solid var(--text-tertiary); }
	.block-head { border-color: var(--success); border-left: 3px solid var(--success); }

	.block-label { font-size: var(--font-size-sm); font-weight: 600; color: var(--text-primary); }
	.block-sublabel { font-size: var(--font-size-xs); color: var(--text-secondary); margin-top: 2px; }

	.block-connector { display: flex; flex-direction: column; align-items: center; height: 24px; }
	.connector-line { width: 1px; flex: 1; background: var(--border); }
	.connector-arrow { font-size: 8px; color: var(--text-tertiary); line-height: 1; }

	/* Stats bar */
	.model-stats { display: flex; gap: var(--spacing-2xl); flex-wrap: wrap; justify-content: center; }
	.stat { display: flex; flex-direction: column; align-items: center; gap: 2px; }
	.stat-label { font-size: var(--font-size-xs); color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.1em; }
	.stat-value { font-size: var(--font-size-md); font-weight: 600; color: var(--text-primary); }

	/* Metadata section */
	.metadata-section {
		width: 100%;
		max-width: 600px;
	}

	.section-header {
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
		padding-bottom: var(--spacing-sm);
		border-bottom: 1px solid var(--border);
		margin-bottom: var(--spacing-md);
	}

	.metadata-grid {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}

	.meta-row {
		display: flex;
		gap: var(--spacing-lg);
		padding: 2px 0;
		font-size: var(--font-size-xs);
	}

	.meta-key {
		color: var(--text-secondary);
		min-width: 240px;
		flex-shrink: 0;
	}

	.meta-value {
		color: var(--text-primary);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
</style>
