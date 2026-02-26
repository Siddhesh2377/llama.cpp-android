<script lang="ts">
	import { PIN_COLORS, type BpPin } from './types';

	let {
		pin,
		x,
		y,
		nodeId,
		onstartWire,
		onendWire,
	}: {
		pin: BpPin;
		x: number;
		y: number;
		nodeId: string;
		onstartWire: (nodeId: string, pinId: string) => void;
		onendWire: (nodeId: string, pinId: string) => void;
	} = $props();

	const color = $derived(PIN_COLORS[pin.dataType] ?? '#888');
	const isInput = $derived(pin.direction === 'in');

	// Text anchor and offset based on pin direction
	const textAnchor = $derived(isInput ? 'start' : 'end');
	const textDx = $derived(isInput ? 10 : -10);

	// Shape label like [960x64]
	const shapeLabel = $derived(
		pin.shape && pin.shape.length > 0 ? `[${pin.shape.join('\u00d7')}]` : null
	);

	function handleMouseDown(e: MouseEvent) {
		if (pin.direction === 'out') {
			e.stopPropagation();
			e.preventDefault();
			onstartWire(nodeId, pin.id);
		}
	}

	function handleMouseUp(e: MouseEvent) {
		if (pin.direction === 'in') {
			e.stopPropagation();
			onendWire(nodeId, pin.id);
		}
	}
</script>

<g class="pin" role="none">
	<!-- Visible pin circle -->
	<circle
		cx={x}
		cy={y}
		r={5}
		fill={pin.connected ? color : 'transparent'}
		stroke={color}
		stroke-width={1.5}
	/>

	<!-- Pin name -->
	<text
		x={x + textDx}
		y={y}
		text-anchor={textAnchor}
		dominant-baseline="central"
		fill="var(--text-secondary)"
		font-size="10"
		font-family="var(--font-mono)"
	>
		{pin.name}
	</text>

	<!-- Shape label below name -->
	{#if shapeLabel}
		<text
			x={x + textDx}
			y={y + 11}
			text-anchor={textAnchor}
			dominant-baseline="central"
			fill="var(--text-tertiary)"
			font-size="8"
			font-family="var(--font-mono)"
		>
			{shapeLabel}
		</text>
	{/if}

	<!-- Invisible hit area for easier clicking -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<circle
		cx={x}
		cy={y}
		r={8}
		fill="transparent"
		stroke="none"
		style="cursor: crosshair;"
		onmousedown={handleMouseDown}
		onmouseup={handleMouseUp}
	/>
</g>
