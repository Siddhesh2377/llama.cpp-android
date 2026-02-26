<script lang="ts">
	import { model } from '$lib/stores/model';
	import { primaryDevice, devices } from '$lib/stores/device';
	import { serverOnline } from '$lib/stores/device';
	import { consoleStore } from '$lib/stores/console';
	import { deployPush, runInference, type InferenceConfig } from '$lib/api/client';

	// Config
	let modelPath = '/sdcard/Download/SmolVLM-500M-Instruct-q8_0.gguf';
	let mmprojPath = '/sdcard/Download/SmolVLM-500M-Instruct-q8_0.mmproj';
	let imagePath = '/sdcard/Download/VLM-TEST-IMAGE.jpg';
	let prompt = 'Describe this image in detail.';
	let threads = '4';
	let quant = 'q5';
	let maxTokens = '128';

	// State
	let running = false;
	let pushing = false;
	let output = '';
	let inferenceTime = 0;

	async function handleDeploy() {
		pushing = true;
		const serial = $primaryDevice?.serial;
		await deployPush(serial);
		pushing = false;
	}

	async function handleRun() {
		running = true;
		output = '';
		inferenceTime = 0;
		const start = performance.now();

		const config: InferenceConfig = {
			serial: $primaryDevice?.serial,
			modelPath,
			mmprojPath: mmprojPath || undefined,
			imagePath: imagePath || undefined,
			prompt,
			threads,
			quant,
			maxTokens,
		};

		const result = await runInference(config);
		inferenceTime = Math.round(performance.now() - start);

		if (result) {
			output = result.output;
		}
		running = false;
	}

	// Parse timing from output
	$: parsedStats = parseStats(output);

	function parseStats(text: string): { visionMs?: number; prefillMs?: number; decodeMs?: number; tokPerSec?: number } {
		const stats: any = {};
		// Look for common patterns in vlm-test output
		const visionMatch = text.match(/vision.*?(\d+)\s*ms/i);
		if (visionMatch) stats.visionMs = parseInt(visionMatch[1]);

		const prefillMatch = text.match(/prefill.*?(\d+)\s*ms/i);
		if (prefillMatch) stats.prefillMs = parseInt(prefillMatch[1]);

		const tokMatch = text.match(/([\d.]+)\s*tok\/s/i);
		if (tokMatch) stats.tokPerSec = parseFloat(tokMatch[1]);

		const decodeMatch = text.match(/([\d.]+)\s*ms\/tok/i);
		if (decodeMatch) stats.decodeMs = parseFloat(decodeMatch[1]);

		return stats;
	}
</script>

<div class="simulation-view">
	{#if !$serverOnline}
		<div class="empty-state">
			<span class="status-dot offline"></span>
			<span class="empty-title">Server Offline</span>
			<span class="empty-hint">Start amx-server to run inference</span>
		</div>
	{:else}
		<div class="sim-layout">
			<!-- Config Panel -->
			<div class="sim-config">
				<div class="config-section">
					<div class="section-label">Device</div>
					{#if $primaryDevice}
						<div class="device-card">
							<span class="status-dot online glow-pulse"></span>
							<div class="device-info">
								<span class="device-name">{$primaryDevice.model || $primaryDevice.serial}</span>
								<span class="device-detail dim">{$primaryDevice.chipset || $primaryDevice.abiList}</span>
							</div>
						</div>
					{:else}
						<div class="device-card offline">
							<span class="status-dot offline"></span>
							<span class="dim">No device connected</span>
						</div>
					{/if}
				</div>

				<div class="config-section">
					<div class="section-label">Model (on device)</div>
					<input type="text" bind:value={modelPath} placeholder="Device model path" />
				</div>

				<div class="config-section">
					<div class="section-label">Vision Projector</div>
					<input type="text" bind:value={mmprojPath} placeholder="mmproj path (optional)" />
				</div>

				<div class="config-section">
					<div class="section-label">Image</div>
					<input type="text" bind:value={imagePath} placeholder="Image path (optional)" />
				</div>

				<div class="config-section">
					<div class="section-label">Prompt</div>
					<textarea bind:value={prompt} rows="2" placeholder="Enter prompt..."></textarea>
				</div>

				<div class="config-row">
					<div class="config-section half">
						<div class="section-label">Threads</div>
						<select bind:value={threads}>
							<option value="1">1</option>
							<option value="2">2</option>
							<option value="4">4</option>
							<option value="8">8</option>
						</select>
					</div>
					<div class="config-section half">
						<div class="section-label">Quantization</div>
						<select bind:value={quant}>
							<option value="">Q8_0 (default)</option>
							<option value="q5">Q5_0 (recommended)</option>
							<option value="q4">Q4_0</option>
						</select>
					</div>
				</div>

				<div class="config-section">
					<div class="section-label">Max Tokens</div>
					<input type="text" bind:value={maxTokens} placeholder="128" />
				</div>

				<div class="action-buttons">
					<button
						class="deploy-btn"
						on:click={handleDeploy}
						disabled={pushing || !$primaryDevice}
					>
						{pushing ? 'PUSHING...' : 'DEPLOY'}
					</button>
					<button
						class="run-btn"
						on:click={handleRun}
						disabled={running || !$primaryDevice}
					>
						{#if running}
							<span class="glow-pulse">RUNNING...</span>
						{:else}
							RUN INFERENCE
						{/if}
					</button>
				</div>
			</div>

			<!-- Output Panel -->
			<div class="sim-output">
				<div class="output-header">
					<span class="label">Output</span>
					{#if inferenceTime > 0}
						<span class="badge accent">{(inferenceTime / 1000).toFixed(1)}s total</span>
					{/if}
					{#if parsedStats.tokPerSec}
						<span class="badge success">{parsedStats.tokPerSec} tok/s</span>
					{/if}
					{#if parsedStats.visionMs}
						<span class="badge info">vision {parsedStats.visionMs}ms</span>
					{/if}
				</div>
				<div class="output-body">
					{#if running}
						<div class="running-indicator">
							<div class="spinner"></div>
							<span>Running inference on device...</span>
						</div>
					{:else if output}
						<pre class="output-text">{output}</pre>
					{:else}
						<div class="output-placeholder">
							<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><path d="M5 3l14 9-14 9V3z"/></svg>
							<span>Configure and run inference to see output</span>
						</div>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	.simulation-view {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.empty-state {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--spacing-md);
		color: var(--text-tertiary);
	}

	.empty-title { font-size: var(--font-size-lg); font-weight: 600; }
	.empty-hint { font-size: var(--font-size-sm); }

	.sim-layout {
		flex: 1;
		display: flex;
		overflow: hidden;
	}

	/* Config */
	.sim-config {
		width: 320px;
		min-width: 280px;
		padding: var(--spacing-lg);
		overflow-y: auto;
		border-right: 1px solid var(--border);
		background: var(--bg-surface);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-lg);
	}

	.config-section {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-sm);
	}

	.config-row {
		display: flex;
		gap: var(--spacing-md);
	}

	.config-section.half { flex: 1; }

	.section-label {
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
	}

	.sim-config input, .sim-config select, .sim-config textarea {
		width: 100%;
		font-family: var(--font-mono);
		font-size: var(--font-size-xs);
		padding: var(--spacing-sm) var(--spacing-md);
		background: var(--bg-base);
		color: var(--text-primary);
		border: 1px solid var(--border);
		resize: vertical;
	}

	.sim-config input:focus, .sim-config select:focus, .sim-config textarea:focus {
		border-color: var(--accent);
		outline: none;
	}

	.device-card {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-md);
		border: 1px solid var(--border);
		background: var(--bg-raised);
	}

	.device-card.offline { opacity: 0.5; }

	.device-info {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}

	.device-name { font-size: var(--font-size-sm); font-weight: 500; }
	.device-detail { font-size: var(--font-size-xs); }

	.action-buttons {
		display: flex;
		gap: var(--spacing-sm);
		margin-top: var(--spacing-md);
	}

	.deploy-btn, .run-btn {
		flex: 1;
		padding: var(--spacing-md);
		font-size: var(--font-size-xs);
		font-weight: 700;
		letter-spacing: 0.1em;
		transition: all var(--transition-fast);
	}

	.deploy-btn {
		background: var(--bg-overlay);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}

	.deploy-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }

	.run-btn {
		background: var(--accent);
		color: var(--bg-base);
	}

	.run-btn:hover:not(:disabled) { background: var(--accent-hover); }
	.deploy-btn:disabled, .run-btn:disabled { opacity: 0.4; cursor: not-allowed; }

	/* Output */
	.sim-output {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.output-header {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-sm) var(--spacing-lg);
		border-bottom: 1px solid var(--border);
		background: var(--bg-raised);
		font-size: var(--font-size-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--text-secondary);
	}

	.output-body {
		flex: 1;
		overflow-y: auto;
		padding: var(--spacing-lg);
	}

	.output-text {
		font-family: var(--font-mono);
		font-size: var(--font-size-xs);
		color: var(--text-primary);
		white-space: pre-wrap;
		word-break: break-word;
		line-height: 1.6;
		margin: 0;
	}

	.output-placeholder {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		gap: var(--spacing-lg);
		color: var(--text-tertiary);
		opacity: 0.4;
		font-size: var(--font-size-sm);
	}

	.running-indicator {
		display: flex;
		align-items: center;
		gap: var(--spacing-lg);
		color: var(--accent);
		font-size: var(--font-size-sm);
	}

	.spinner {
		width: 16px;
		height: 16px;
		border: 2px solid var(--border);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}
</style>
