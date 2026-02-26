<script lang="ts">
	export let title: string = 'Coming Soon';
	export let icon: string = 'M12 2L2 7l10 5 10-5-10-5z';
</script>

<div class="placeholder">
	<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round">
		<path d={icon} />
	</svg>
	<span class="placeholder-title">{title}</span>
	<span class="placeholder-hint">This workspace is under construction</span>
</div>

<style>
	.placeholder {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--spacing-lg);
		color: var(--text-tertiary);
		opacity: 0.5;
	}

	.placeholder-title {
		font-size: var(--font-size-lg);
		font-weight: 600;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.placeholder-hint {
		font-size: var(--font-size-sm);
	}
</style>
