<script lang="ts">
	import { PIN_COLORS, type PinType } from './types';

	let {
		x1,
		y1,
		x2,
		y2,
		dataType,
		valid,
	}: {
		x1: number;
		y1: number;
		x2: number;
		y2: number;
		dataType: PinType;
		valid: boolean;
	} = $props();

	const color = $derived(valid ? (PIN_COLORS[dataType] ?? '#888') : 'var(--error)');
	const dx = $derived(Math.abs(x2 - x1) * 0.5);
	const pathD = $derived(
		`M ${x1},${y1} C ${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`
	);
</script>

<path
	d={pathD}
	fill="none"
	stroke={color}
	stroke-width={2}
	stroke-dasharray={valid ? 'none' : '4 4'}
	opacity={0.7}
/>
