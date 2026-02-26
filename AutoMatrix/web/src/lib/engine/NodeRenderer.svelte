<script lang="ts">
	import { CATEGORY_COLORS, type BpNode } from './types';
	import PinRenderer from './PinRenderer.svelte';

	let {
		node,
		selected,
		isDark,
		onstartWire,
		onendWire,
	}: {
		node: BpNode;
		selected: boolean;
		isDark: boolean;
		onstartWire: (nodeId: string, pinId: string) => void;
		onendWire: (nodeId: string, pinId: string) => void;
	} = $props();

	const HEADER_H = 26;
	const PIN_SPACING = 22;

	const catColors = $derived(CATEGORY_COLORS[node.category] ?? CATEGORY_COLORS.custom);
	const bgColor = $derived(isDark ? catColors.darkBg : catColors.lightBg);
	const accentColor = $derived(isDark ? catColors.darkAccent : catColors.lightAccent);

	const nodeW = $derived(node.size.w || 180);

	const inputPins = $derived(node.pins.filter(p => p.direction === 'in'));
	const outputPins = $derived(node.pins.filter(p => p.direction === 'out'));

	const pinCount = $derived(Math.max(inputPins.length, outputPins.length));
	const nodeH = $derived(
		node.collapsed ? HEADER_H : HEADER_H + pinCount * PIN_SPACING + 8
	);
</script>

<g transform="translate({node.position.x}, {node.position.y})" class="node-group">
	<!-- Drop shadow -->
	<rect
		x={2}
		y={2}
		width={nodeW}
		height={nodeH}
		rx={4}
		fill="rgba(0,0,0,0.3)"
	/>

	<!-- Body -->
	<rect
		x={0}
		y={0}
		width={nodeW}
		height={nodeH}
		rx={4}
		fill={bgColor}
		stroke={selected ? accentColor : 'var(--border)'}
		stroke-width={selected ? 2 : 1}
	/>

	<!-- Left accent bar -->
	<rect
		x={0}
		y={0}
		width={4}
		height={nodeH}
		rx={2}
		fill={accentColor}
	/>

	<!-- Header label -->
	<text
		x={12}
		y={17}
		fill={accentColor}
		font-size="11"
		font-weight="600"
		font-family="var(--font-mono)"
	>
		{node.label}
	</text>

	<!-- Disabled indicator (X) -->
	{#if !node.enabled}
		<line x1={4} y1={4} x2={nodeW - 4} y2={nodeH - 4} stroke="#ef5350" stroke-width={2} opacity={0.5} />
		<line x1={nodeW - 4} y1={4} x2={4} y2={nodeH - 4} stroke="#ef5350" stroke-width={2} opacity={0.5} />
	{/if}

	<!-- Pins (only when not collapsed) -->
	{#if !node.collapsed}
		{#each inputPins as pin, i}
			<PinRenderer
				{pin}
				x={0}
				y={HEADER_H + i * PIN_SPACING + 12}
				nodeId={node.id}
				{onstartWire}
				{onendWire}
			/>
		{/each}
		{#each outputPins as pin, i}
			<PinRenderer
				{pin}
				x={nodeW}
				y={HEADER_H + i * PIN_SPACING + 12}
				nodeId={node.id}
				{onstartWire}
				{onendWire}
			/>
		{/each}
	{/if}
</g>
