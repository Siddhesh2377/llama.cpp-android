<script lang="ts">
	import { documents, activeDocId, closeDocument } from '$lib/engine/store';

	const typeIcons: Record<string, string> = {
		model: '\u25C6', arch: '\u2B21', backend: '\u2699', quant: '\u25A3',
		sampling: '\u25C7', surgery: '\u2702', benchmark: '\uD83D\uDCCA', test: '\u2713'
	};
</script>

<div class="doc-tabs">
	{#each $documents as doc (doc.id)}
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="doc-tab" class:active={$activeDocId === doc.id}
				role="tab" tabindex="0"
				onclick={() => activeDocId.set(doc.id)}
				onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') activeDocId.set(doc.id); }}>
			<span class="doc-icon">{typeIcons[doc.type] || '\u25C7'}</span>
			<span class="doc-name">{doc.dirty ? '\u25CF ' : ''}{doc.name}</span>
			<button class="doc-close" onclick={(e) => { e.stopPropagation(); closeDocument(doc.id); }}>{'\u00D7'}</button>
		</div>
	{/each}
	<button class="doc-tab add-tab" title="New document">+</button>
</div>

<style>
	.doc-tabs {
		display: flex;
		flex-direction: row;
		height: 28px;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		overflow-x: auto;
		gap: 0;
	}

	.doc-tab {
		display: flex;
		flex-direction: row;
		align-items: center;
		padding: 0 12px;
		gap: 6px;
		font-size: 11px;
		border-right: 1px solid var(--border-subtle);
		color: var(--text-secondary);
		cursor: pointer;
		white-space: nowrap;
		background: none;
		border-top: none;
		border-bottom: 2px solid transparent;
		border-left: none;
		transition: background var(--transition-fast);
	}

	.doc-tab:hover {
		background: var(--bg-raised);
	}

	.doc-tab.active {
		background: var(--bg-base);
		color: var(--text-primary);
		border-bottom: 2px solid var(--accent);
	}

	.doc-close {
		font-size: 12px;
		color: var(--text-tertiary);
		width: 16px;
		height: 16px;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		border: none;
		background: none;
		cursor: pointer;
		border-radius: 2px;
		transition: color var(--transition-fast);
	}

	.doc-close:hover {
		color: var(--error);
	}

	.doc-icon {
		font-size: 10px;
	}

	.add-tab {
		color: var(--text-tertiary);
		border-right: none;
	}

	.add-tab:hover {
		color: var(--accent);
	}
</style>
