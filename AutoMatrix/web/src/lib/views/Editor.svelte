<script lang="ts">
	import { model, formatBytes } from '$lib/stores/model';
	import { plugins } from '$lib/stores/plugins';
	import { quantPlugins, archPlugins, backendPlugins, samplingPlugins } from '$lib/stores/plugins';
	import { consoleStore } from '$lib/stores/console';
	import { savePlugin, getPlugins } from '$lib/api/client';
	import { onMount } from 'svelte';

	// --- Types ---
	interface EditorNode {
		id: string; label: string; type: string; sublabel: string;
		x: number; y: number; w: number; h: number; layer: number;
		quant: string; enabled: boolean; editable: boolean;
	}

	// --- State ---
	let editMode = $state<'graph' | 'layers' | 'config' | 'plugins'>('graph');
	let zoom = $state(1);
	let panX = $state(60);
	let panY = $state(20);
	let dirty = $state(false);
	let canvasEl: HTMLDivElement | undefined = $state();

	// Interaction state
	let dragging = $state<'pan' | 'node' | null>(null);
	let dragNodeId = $state<string | null>(null);
	let dragOrigin = $state({ x: 0, y: 0, nx: 0, ny: 0 });
	let selectedId = $state<string | null>(null);
	let contextMenu = $state<{ x: number; y: number; nodeId: string | null } | null>(null);

	// Graph data (mutable for editing)
	let nodes = $state<EditorNode[]>([]);
	let layerQuants = $state<Record<number, string>>({});
	let globalQuant = $state('q8_0');
	let editableConfig = $state<Record<string, string>>({});

	// Editing plugins directly
	let editingPlugin = $state<{ type: string; id: string; json: string } | null>(null);

	// Derived
	let selectedNode = $derived(nodes.find(n => n.id === selectedId) || null);
	let arch = $derived($model?.arch || 'unknown');

	let quantOptions = $derived(
		$quantPlugins.map(p => {
			const b = p.body as Record<string, unknown>;
			return { id: b.id as string, name: b.name as string, bits: b.bits_per_weight as number || 0,
				quality: b.quality_rating as number || 0, speed: b.speed_rating as number || 0 };
		})
	);

	// --- Initialize graph from model ---
	$effect(() => {
		if ($model && nodes.length === 0) buildGraph();
	});

	$effect(() => {
		if ($model && Object.keys(editableConfig).length === 0) {
			editableConfig = {
				context_length: String($model.contextLength || 0),
				embedding_size: String($model.embeddingSize || 0),
				head_count: String($model.headCount || 0),
				head_count_kv: String($model.headCountKV || 0),
				vocab_size: String($model.vocabSize || 0),
			};
		}
	});

	function buildGraph() {
		if (!$model) return;
		const out: EditorNode[] = [];
		const cx = 300, nw = 220, nh = 48, gap = 10;
		let y = 30;

		out.push({ id: 'embed', label: 'Token Embedding', type: 'embed',
			sublabel: `${$model.embeddingSize}d`, x: cx - nw/2, y, w: nw, h: nh,
			layer: -1, quant: 'f16', enabled: true, editable: false });
		y += nh + gap;

		const nLayers = $model.layerCount || 0;
		for (let li = 0; li < nLayers; li++) {
			const q = layerQuants[li] || globalQuant;
			out.push({ id: `L${li}_attn`, label: `Block ${li} · Attention`, type: 'attn',
				sublabel: `GQA · RoPE`, x: cx - nw/2, y, w: nw, h: nh,
				layer: li, quant: q, enabled: true, editable: true });
			y += nh + 6;

			out.push({ id: `L${li}_ffn`, label: `Block ${li} · FFN`, type: 'ffn',
				sublabel: `SiLU Gate+Up+Down`, x: cx - nw/2, y, w: nw, h: nh,
				layer: li, quant: q, enabled: true, editable: true });
			y += nh + gap;
		}

		out.push({ id: 'out_norm', label: 'Output Norm', type: 'norm',
			sublabel: 'RMSNorm', x: cx - nw/2, y, w: nw, h: nh,
			layer: -1, quant: 'f32', enabled: true, editable: false });
		y += nh + gap;

		out.push({ id: 'head', label: 'LM Head', type: 'head',
			sublabel: `→ ${$model.vocabSize} vocab`, x: cx - nw/2, y, w: nw, h: nh,
			layer: -1, quant: globalQuant, enabled: true, editable: true });

		nodes = out;
	}

	// --- Wheel zoom (non-passive) ---
	onMount(() => {
		if (!canvasEl) return;
		const handler = (e: WheelEvent) => {
			e.preventDefault();
			const rect = canvasEl!.getBoundingClientRect();
			const mx = e.clientX - rect.left;
			const my = e.clientY - rect.top;
			const factor = e.deltaY < 0 ? 1.12 : 0.89;
			const nz = Math.max(0.15, Math.min(4, zoom * factor));
			// Zoom toward cursor
			panX = mx - (mx - panX) * (nz / zoom);
			panY = my - (my - panY) * (nz / zoom);
			zoom = nz;
		};
		canvasEl.addEventListener('wheel', handler, { passive: false });
		return () => canvasEl?.removeEventListener('wheel', handler);
	});

	// --- Mouse interactions ---
	function onCanvasDown(e: MouseEvent) {
		contextMenu = null;
		if (e.button === 2) return; // right-click handled by context menu

		// Check if clicking on a node
		const svgPt = screenToGraph(e.clientX, e.clientY);
		const hit = findNodeAt(svgPt.x, svgPt.y);

		if (hit) {
			selectedId = hit.id;
			dragging = 'node';
			dragNodeId = hit.id;
			dragOrigin = { x: e.clientX, y: e.clientY, nx: hit.x, ny: hit.y };
		} else {
			selectedId = null;
			dragging = 'pan';
			dragOrigin = { x: e.clientX - panX, y: e.clientY - panY, nx: 0, ny: 0 };
		}
	}

	function onCanvasMove(e: MouseEvent) {
		if (dragging === 'pan') {
			panX = e.clientX - dragOrigin.x;
			panY = e.clientY - dragOrigin.y;
		} else if (dragging === 'node' && dragNodeId) {
			const dx = (e.clientX - dragOrigin.x) / zoom;
			const dy = (e.clientY - dragOrigin.y) / zoom;
			const node = nodes.find(n => n.id === dragNodeId);
			if (node) {
				node.x = dragOrigin.nx + dx;
				node.y = dragOrigin.ny + dy;
				dirty = true;
			}
		}
	}

	function onCanvasUp() { dragging = null; dragNodeId = null; }

	function onContextMenu(e: MouseEvent) {
		e.preventDefault();
		const svgPt = screenToGraph(e.clientX, e.clientY);
		const hit = findNodeAt(svgPt.x, svgPt.y);
		contextMenu = { x: e.clientX, y: e.clientY, nodeId: hit?.id || null };
		if (hit) selectedId = hit.id;
	}

	function screenToGraph(sx: number, sy: number) {
		const rect = canvasEl?.getBoundingClientRect() || { left: 0, top: 0 };
		return { x: (sx - rect.left - panX) / zoom, y: (sy - rect.top - panY) / zoom };
	}

	function findNodeAt(gx: number, gy: number): EditorNode | null {
		for (let i = nodes.length - 1; i >= 0; i--) {
			const n = nodes[i];
			if (gx >= n.x && gx <= n.x + n.w && gy >= n.y && gy <= n.y + n.h) return n;
		}
		return null;
	}

	// --- CRUD operations ---
	function addNode(type: string, afterId?: string) {
		const afterNode = afterId ? nodes.find(n => n.id === afterId) : nodes[nodes.length - 1];
		const y = afterNode ? afterNode.y + afterNode.h + 20 : 100;
		const x = afterNode ? afterNode.x : 190;
		const id = `custom_${Date.now()}`;
		const newNode: EditorNode = {
			id, label: `New ${type}`, type, sublabel: 'Custom',
			x, y, w: 220, h: 48, layer: -1,
			quant: globalQuant, enabled: true, editable: true
		};
		if (afterId) {
			const idx = nodes.findIndex(n => n.id === afterId);
			nodes.splice(idx + 1, 0, newNode);
			// Shift nodes below
			for (let i = idx + 2; i < nodes.length; i++) {
				nodes[i].y += 58;
			}
		} else {
			nodes.push(newNode);
		}
		nodes = [...nodes]; // trigger reactivity
		selectedId = id;
		dirty = true;
		consoleStore.info(`Added ${type} node`, 'editor');
	}

	function deleteNode(id: string) {
		const idx = nodes.findIndex(n => n.id === id);
		if (idx < 0) return;
		nodes.splice(idx, 1);
		nodes = [...nodes];
		if (selectedId === id) selectedId = null;
		dirty = true;
		consoleStore.info(`Deleted node`, 'editor');
	}

	function duplicateNode(id: string) {
		const src = nodes.find(n => n.id === id);
		if (!src) return;
		addNode(src.type, id);
		const dup = nodes.find(n => n.id === selectedId);
		if (dup) {
			dup.label = src.label + ' (copy)';
			dup.sublabel = src.sublabel;
			dup.quant = src.quant;
		}
		dirty = true;
	}

	function toggleNode(id: string) {
		const node = nodes.find(n => n.id === id);
		if (node) { node.enabled = !node.enabled; nodes = [...nodes]; dirty = true; }
	}

	function setNodeQuant(id: string, quant: string) {
		const node = nodes.find(n => n.id === id);
		if (node) {
			node.quant = quant;
			if (node.layer >= 0) layerQuants[node.layer] = quant;
			nodes = [...nodes];
			dirty = true;
		}
	}

	function renameNode(id: string, label: string) {
		const node = nodes.find(n => n.id === id);
		if (node) { node.label = label; nodes = [...nodes]; dirty = true; }
	}

	function setGlobalQuant(q: string) {
		globalQuant = q;
		nodes.forEach(n => { if (n.editable && n.layer >= 0) n.quant = q; });
		layerQuants = {};
		nodes = [...nodes];
		dirty = true;
	}

	function resetLayout() {
		const cx = 300, nw = 220, gap = 10;
		let y = 30;
		for (const n of nodes) {
			n.x = cx - nw / 2;
			n.y = y;
			y += n.h + gap;
		}
		nodes = [...nodes];
		panX = 60; panY = 20; zoom = 1;
	}

	async function saveConfig() {
		const pluginBody = {
			id: arch + '_custom',
			name: ($model?.name || arch) + ' (Custom)',
			version: '1.0', base_arch: arch,
			config_overrides: editableConfig,
			quant_map: { global: globalQuant, per_layer: Object.keys(layerQuants).length > 0 ? layerQuants : undefined },
			graph: nodes.map(n => ({ id: n.id, type: n.type, label: n.label, quant: n.quant, enabled: n.enabled, layer: n.layer }))
		};
		const ok = await savePlugin('arch', pluginBody.id, pluginBody);
		if (ok) { dirty = false; plugins.set(await getPlugins()); }
	}

	function editPluginJson(type: string, id: string, body: Record<string, unknown>) {
		editingPlugin = { type, id, json: JSON.stringify(body, null, 2) };
		editMode = 'plugins';
	}

	async function savePluginJson() {
		if (!editingPlugin) return;
		try {
			const body = JSON.parse(editingPlugin.json);
			const ok = await savePlugin(editingPlugin.type, editingPlugin.id, body);
			if (ok) { plugins.set(await getPlugins()); consoleStore.success(`Saved ${editingPlugin.id}`, 'editor'); }
		} catch (e) { consoleStore.error(`Invalid JSON: ${e}`, 'editor'); }
	}

	// --- Colors ---
	function nc(type: string): string {
		const c: Record<string, string> = {
			embed: '#6ec6ff', attn: '#e8963a', ffn: '#ce93d8', norm: '#a5d6a7',
			head: '#ef5350', proj: '#ffb74d', ellipsis: '#666', custom: '#ffd54f'
		};
		return c[type] || '#888';
	}
</script>

<div class="editor-view">
	<!-- Toolbar -->
	<div class="toolbar">
		<div class="tool-group">
			{#each [['graph','Graph','M12 3v18M3 12h18'],['layers','Layers','M2 7l10 5 10-5M2 17l10 5 10-5M2 12l10 5 10-5'],['config','Config','M12 20h9M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z'],['plugins','Plugins','M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4']] as [id, label, icon]}
				<button class="tool-btn" class:active={editMode === id} onclick={() => editMode = id as typeof editMode}>
					<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d={icon}/></svg>
					{label}
				</button>
			{/each}
		</div>

		{#if editMode === 'graph'}
			<div class="tool-sep"></div>
			<button class="tool-btn" onclick={() => addNode('attn')}>+ Attn</button>
			<button class="tool-btn" onclick={() => addNode('ffn')}>+ FFN</button>
			<button class="tool-btn" onclick={() => addNode('norm')}>+ Norm</button>
			<div class="tool-sep"></div>
			<span class="quant-label">Quant:</span>
			<select class="quant-select" onchange={(e) => setGlobalQuant((e.target as HTMLSelectElement).value)}>
				{#each quantOptions as q}
					<option value={q.id} selected={q.id === globalQuant}>{q.name} ({q.bits}b)</option>
				{/each}
			</select>
			<button class="tool-btn" onclick={resetLayout} title="Reset layout">Reset</button>
		{/if}

		{#if dirty}
			<button class="save-btn" onclick={saveConfig}>Save</button>
		{/if}
	</div>

	{#if !$model}
		<div class="empty-state">
			<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-tertiary)" stroke-width="1">
				<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>
			</svg>
			<p>Load a model in Architecture tab to start editing</p>
		</div>

	{:else if editMode === 'graph'}
		<div class="graph-area">
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div class="graph-canvas" bind:this={canvasEl}
				onmousedown={onCanvasDown} onmousemove={onCanvasMove}
				onmouseup={onCanvasUp} onmouseleave={onCanvasUp}
				oncontextmenu={onContextMenu}
			>
				<svg width="100%" height="100%" style="cursor: {dragging === 'pan' ? 'grabbing' : dragging === 'node' ? 'move' : 'default'}">
					<defs>
						<pattern id="eg" width="40" height="40" patternUnits="userSpaceOnUse"
							patternTransform="translate({panX},{panY}) scale({zoom})">
							<path d="M40 0L0 0 0 40" fill="none" stroke="var(--border)" stroke-width="0.5" opacity="0.3"/>
						</pattern>
					</defs>
					<rect width="100%" height="100%" fill="url(#eg)"/>

					<g transform="translate({panX},{panY}) scale({zoom})">
						<!-- Edges -->
						{#each nodes as node, i}
							{#if i > 0}
								{@const prev = nodes[i - 1]}
								<path d="M{prev.x + prev.w/2},{prev.y + prev.h} C{prev.x + prev.w/2},{prev.y + prev.h + 15} {node.x + node.w/2},{node.y - 15} {node.x + node.w/2},{node.y}"
									fill="none" stroke={node.enabled ? 'var(--text-tertiary)' : '#f44'} stroke-width="1.5" opacity="0.4"
									stroke-dasharray={node.enabled ? '' : '4 3'}
								/>
							{/if}
						{/each}

						<!-- Nodes -->
						{#each nodes as node (node.id)}
							{@const sel = selectedId === node.id}
							<g style="cursor: pointer" opacity={node.enabled ? 1 : 0.4}>
								<!-- Shadow -->
								{#if sel}
									<rect x={node.x - 2} y={node.y - 2} width={node.w + 4} height={node.h + 4}
										rx="8" fill="none" stroke="var(--accent)" stroke-width="2" stroke-dasharray="4 2"/>
								{/if}

								<rect x={node.x} y={node.y} width={node.w} height={node.h}
									rx="6" fill="var(--bg-panel)"
									stroke={nc(node.type)} stroke-width={sel ? 1.5 : 0.8}
								/>
								<!-- Left accent -->
								<rect x={node.x} y={node.y + 4} width="3" height={node.h - 8} rx="1.5" fill={nc(node.type)}/>

								<text x={node.x + 12} y={node.y + 19} fill="var(--text-primary)"
									font-size="11" font-weight="600" font-family="var(--font-mono)">{node.label}</text>
								<text x={node.x + 12} y={node.y + 34} fill="var(--text-tertiary)"
									font-size="9" font-family="var(--font-mono)">{node.sublabel}</text>

								<!-- Quant badge -->
								<rect x={node.x + node.w - 50} y={node.y + 8} width="42" height="14"
									rx="3" fill={nc(node.type)} opacity="0.15"/>
								<text x={node.x + node.w - 29} y={node.y + 19} fill={nc(node.type)}
									font-size="9" font-weight="700" text-anchor="middle" font-family="var(--font-mono)">
									{node.quant.toUpperCase()}
								</text>

								<!-- Disabled X -->
								{#if !node.enabled}
									<line x1={node.x + 4} y1={node.y + 4} x2={node.x + node.w - 4} y2={node.y + node.h - 4}
										stroke="#f44" stroke-width="1.5" opacity="0.6"/>
								{/if}
							</g>
						{/each}
					</g>
				</svg>

				<!-- Context menu -->
				{#if contextMenu}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div class="ctx-menu" style="left:{contextMenu.x}px;top:{contextMenu.y}px"
						onmouseleave={() => contextMenu = null}>
						{#if contextMenu.nodeId}
							<button onclick={() => { duplicateNode(contextMenu!.nodeId!); contextMenu = null; }}>Duplicate</button>
							<button onclick={() => { toggleNode(contextMenu!.nodeId!); contextMenu = null; }}>
								{nodes.find(n => n.id === contextMenu?.nodeId)?.enabled ? 'Disable' : 'Enable'}
							</button>
							<button onclick={() => { addNode('attn', contextMenu!.nodeId!); contextMenu = null; }}>Insert After</button>
							<hr/>
							<button class="danger" onclick={() => { deleteNode(contextMenu!.nodeId!); contextMenu = null; }}>Delete</button>
						{:else}
							<button onclick={() => { addNode('attn'); contextMenu = null; }}>Add Attention</button>
							<button onclick={() => { addNode('ffn'); contextMenu = null; }}>Add FFN</button>
							<button onclick={() => { addNode('norm'); contextMenu = null; }}>Add Norm</button>
							<hr/>
							<button onclick={() => { resetLayout(); contextMenu = null; }}>Reset Layout</button>
						{/if}
					</div>
				{/if}
			</div>

			<!-- Inspector panel -->
			{#if selectedNode}
				<div class="inspector">
					<div class="insp-hdr">
						<span class="insp-dot" style="background:{nc(selectedNode.type)}"></span>
						<span class="insp-title">{selectedNode.label}</span>
						<button class="insp-close" onclick={() => selectedId = null}>×</button>
					</div>
					<div class="insp-row">
						<label>Label</label>
						<input type="text" class="insp-input" value={selectedNode.label}
							oninput={(e) => renameNode(selectedNode!.id, (e.target as HTMLInputElement).value)}/>
					</div>
					<div class="insp-row">
						<label>Type</label>
						<span class="insp-val">{selectedNode.type}</span>
					</div>
					<div class="insp-row">
						<label>Layer</label>
						<span class="insp-val">{selectedNode.layer >= 0 ? selectedNode.layer : '—'}</span>
					</div>
					<div class="insp-row">
						<label>Quant</label>
						<select class="insp-select" onchange={(e) => setNodeQuant(selectedNode!.id, (e.target as HTMLSelectElement).value)}>
							{#each quantOptions as q}
								<option value={q.id} selected={q.id === selectedNode.quant}>{q.name} ({q.bits}b)</option>
							{/each}
						</select>
					</div>
					<div class="insp-row">
						<label>Enabled</label>
						<button class="toggle-btn" class:on={selectedNode.enabled}
							onclick={() => toggleNode(selectedNode!.id)}>
							{selectedNode.enabled ? 'ON' : 'OFF'}
						</button>
					</div>
					<div class="insp-row">
						<label>Position</label>
						<span class="insp-val">{Math.round(selectedNode.x)}, {Math.round(selectedNode.y)}</span>
					</div>
					<div class="insp-actions">
						<button onclick={() => duplicateNode(selectedNode!.id)}>Duplicate</button>
						<button class="danger" onclick={() => deleteNode(selectedNode!.id)}>Delete</button>
					</div>
				</div>
			{/if}
		</div>

	{:else if editMode === 'layers'}
		<div class="layer-list">
			<div class="layer-header">
				<span>#</span><span>Layer</span><span>Quant</span><span>Tensors</span><span>Size</span><span>Status</span>
			</div>
			{#each ($model?.layers || []) as layer}
				<div class="layer-row" class:custom-quant={layerQuants[layer.index] !== undefined}>
					<span class="mono">{layer.index}</span>
					<span class="bold">{layer.name}</span>
					<select class="lr-quant" onchange={(e) => { layerQuants[layer.index] = (e.target as HTMLSelectElement).value; dirty = true; }}>
						{#each quantOptions as q}
							<option value={q.id} selected={q.id === (layerQuants[layer.index] || globalQuant)}>{q.name}</option>
						{/each}
					</select>
					<span class="mono">{layer.tensors?.length || 0}</span>
					<span class="mono dim">{formatBytes(layer.tensors?.reduce((s: number, t: {size:number}) => s + t.size, 0) || 0)}</span>
					<span class="status-dot on"></span>
				</div>
			{/each}
		</div>

	{:else if editMode === 'config'}
		<div class="config-editor">
			<div class="cg"><h4>Architecture</h4>
				{#each [['Arch', arch],['Name', $model?.name || '—'],['File', $model?.filename || '—']] as [k,v]}
					<div class="cr"><label>{k}</label><span class="cv ro">{v}</span></div>
				{/each}
			</div>
			<div class="cg"><h4>Dimensions</h4>
				{#each Object.entries(editableConfig) as [key, value]}
					<div class="cr"><label>{key.replace(/_/g, ' ')}</label>
						<input type="number" class="ci" value={value}
							oninput={(e) => { editableConfig[key] = (e.target as HTMLInputElement).value; dirty = true; }}/>
					</div>
				{/each}
			</div>
			<div class="cg"><h4>Quantization</h4>
				<div class="cr"><label>Global</label>
					<select class="ci" onchange={(e) => setGlobalQuant((e.target as HTMLSelectElement).value)}>
						{#each quantOptions as q}
							<option value={q.id} selected={q.id === globalQuant}>{q.name} ({q.bits}b)</option>
						{/each}
					</select>
				</div>
				{#if Object.keys(layerQuants).length > 0}
					<div class="cr"><label>Overrides</label><span class="cv">{Object.keys(layerQuants).length} layers</span></div>
				{/if}
			</div>
			<div class="cg"><h4>Memory</h4>
				<div class="cr"><label>Weights</label><span class="cv">{formatBytes(($model?.tensors || []).reduce((s, t) => s + (t.size || 0), 0))}</span></div>
				<div class="cr"><label>Tensors</label><span class="cv">{$model?.tensorCount}</span></div>
				<div class="cr"><label>File</label><span class="cv">{formatBytes($model?.fileSize || 0)}</span></div>
			</div>
		</div>

	{:else if editMode === 'plugins'}
		<div class="plugins-editor">
			{#if editingPlugin}
				<div class="pe-header">
					<span>Editing: {editingPlugin.id} ({editingPlugin.type})</span>
					<button class="save-btn" onclick={savePluginJson}>Save</button>
					<button class="tool-btn" onclick={() => editingPlugin = null}>Close</button>
				</div>
				<textarea class="pe-textarea" bind:value={editingPlugin.json}></textarea>
			{:else}
				<div class="pe-grid">
					{#each $plugins as p}
						<button class="pe-card" onclick={() => editPluginJson(p.type, (p.body as Record<string,unknown>).id as string, p.body as Record<string,unknown>)}>
							<span class="pe-dot" style="background:{nc(p.type === 'architecture' ? 'attn' : p.type === 'quant' ? 'ffn' : 'norm')}"></span>
							<span class="pe-name">{p.name}</span>
							<span class="pe-type">{p.type}</span>
						</button>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.editor-view { display: flex; flex-direction: column; height: 100%; overflow: hidden; position: relative; }
	.toolbar { display: flex; align-items: center; gap: 6px; padding: 5px 10px; border-bottom: 1px solid var(--border); background: var(--bg-surface); flex-shrink: 0; flex-wrap: wrap; }
	.tool-group { display: flex; gap: 3px; }
	.tool-btn { display: flex; align-items: center; gap: 4px; padding: 3px 8px; font-size: 10px; border: 1px solid var(--border); border-radius: 4px; background: none; color: var(--text-secondary); cursor: pointer; }
	.tool-btn:hover { background: var(--bg-hover); }
	.tool-btn.active { background: var(--accent-glow); border-color: var(--accent); color: var(--accent); }
	.tool-sep { width: 1px; height: 20px; background: var(--border); margin: 0 4px; }
	.quant-label { font-size: 10px; color: var(--text-tertiary); }
	.quant-select { font-size: 10px; padding: 2px 4px; background: var(--bg-panel); border: 1px solid var(--border); border-radius: 3px; color: var(--text-primary); }
	.save-btn { display: flex; align-items: center; gap: 4px; padding: 3px 10px; font-size: 10px; font-weight: 700; background: var(--accent); color: var(--bg-base); border: none; border-radius: 4px; cursor: pointer; margin-left: auto; }
	.empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: var(--text-tertiary); font-size: 13px; }

	/* Graph */
	.graph-area { flex: 1; display: flex; position: relative; overflow: hidden; }
	.graph-canvas { flex: 1; overflow: hidden; background: var(--bg-base); position: relative; }

	/* Context menu */
	.ctx-menu { position: fixed; background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; padding: 4px; min-width: 140px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); z-index: 100; }
	.ctx-menu button { display: block; width: 100%; text-align: left; padding: 5px 10px; font-size: 11px; border: none; background: none; color: var(--text-secondary); cursor: pointer; border-radius: 3px; }
	.ctx-menu button:hover { background: var(--bg-hover); color: var(--text-primary); }
	.ctx-menu button.danger { color: #ef5350; }
	.ctx-menu button.danger:hover { background: rgba(239,83,80,0.1); }
	.ctx-menu hr { border: none; border-top: 1px solid var(--border); margin: 3px 0; }

	/* Inspector */
	.inspector { width: 260px; border-left: 1px solid var(--border); background: var(--bg-surface); padding: 12px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
	.insp-hdr { display: flex; align-items: center; gap: 6px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
	.insp-dot { width: 8px; height: 8px; border-radius: 50%; }
	.insp-title { font-size: 12px; font-weight: 600; flex: 1; }
	.insp-close { background: none; border: none; color: var(--text-tertiary); font-size: 16px; cursor: pointer; padding: 0 4px; }
	.insp-row { display: flex; justify-content: space-between; align-items: center; }
	.insp-row label { font-size: 10px; color: var(--text-tertiary); text-transform: uppercase; }
	.insp-val { font-size: 11px; color: var(--text-primary); font-family: var(--font-mono); }
	.insp-input { font-size: 11px; padding: 2px 6px; width: 140px; background: var(--bg-active); border: 1px solid var(--border); border-radius: 3px; color: var(--text-primary); font-family: var(--font-mono); }
	.insp-input:focus { border-color: var(--accent); outline: none; }
	.insp-select { font-size: 10px; padding: 2px 4px; background: var(--bg-active); border: 1px solid var(--border); border-radius: 3px; color: var(--text-primary); max-width: 140px; }
	.toggle-btn { padding: 2px 10px; font-size: 10px; font-weight: 700; border: 1px solid var(--border); border-radius: 3px; background: var(--bg-active); color: var(--text-tertiary); cursor: pointer; }
	.toggle-btn.on { background: var(--accent-glow); border-color: var(--accent); color: var(--accent); }
	.insp-actions { display: flex; gap: 6px; margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }
	.insp-actions button { flex: 1; padding: 4px; font-size: 10px; border: 1px solid var(--border); border-radius: 3px; background: none; color: var(--text-secondary); cursor: pointer; }
	.insp-actions button:hover { background: var(--bg-hover); }
	.insp-actions button.danger { color: #ef5350; border-color: #ef5350; }

	/* Layers */
	.layer-list { flex: 1; overflow-y: auto; }
	.layer-header, .layer-row { display: grid; grid-template-columns: 36px 1fr 100px 56px 72px 32px; align-items: center; padding: 5px 12px; font-size: 11px; gap: 6px; }
	.layer-header { position: sticky; top: 0; background: var(--bg-surface); border-bottom: 1px solid var(--border); font-size: 9px; font-weight: 600; text-transform: uppercase; color: var(--text-tertiary); z-index: 1; }
	.layer-row { border-bottom: 1px solid var(--border); }
	.layer-row:hover { background: var(--bg-hover); }
	.layer-row.custom-quant { background: var(--accent-glow); }
	.mono { font-family: var(--font-mono); color: var(--text-secondary); }
	.bold { font-weight: 500; color: var(--text-primary); }
	.dim { color: var(--text-tertiary); }
	.lr-quant { font-size: 10px; padding: 2px 4px; background: var(--bg-active); border: 1px solid var(--border); border-radius: 3px; color: var(--text-primary); }
	.status-dot { width: 6px; height: 6px; border-radius: 50%; justify-self: center; }
	.status-dot.on { background: #66bb6a; }

	/* Config */
	.config-editor { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 16px; }
	.cg h4 { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--accent); margin: 0 0 8px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
	.cr { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; }
	.cr label { font-size: 11px; color: var(--text-secondary); text-transform: capitalize; }
	.cv { font-size: 11px; font-family: var(--font-mono); color: var(--text-primary); }
	.cv.ro { color: var(--text-tertiary); }
	.ci { font-size: 11px; padding: 3px 6px; width: 130px; text-align: right; background: var(--bg-active); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-family: var(--font-mono); }
	.ci:focus { border-color: var(--accent); outline: none; }

	/* Plugins editor */
	.plugins-editor { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
	.pe-header { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 11px; color: var(--text-primary); }
	.pe-textarea { flex: 1; font-family: var(--font-mono); font-size: 11px; line-height: 1.5; padding: 12px; background: var(--bg-base); color: var(--text-primary); border: none; resize: none; outline: none; }
	.pe-grid { flex: 1; overflow-y: auto; padding: 12px; display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; align-content: start; }
	.pe-card { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: var(--bg-panel); border: 1px solid var(--border); border-radius: 5px; cursor: pointer; text-align: left; }
	.pe-card:hover { border-color: var(--text-tertiary); }
	.pe-dot { width: 6px; height: 6px; border-radius: 50%; }
	.pe-name { font-size: 11px; font-weight: 600; color: var(--text-primary); flex: 1; }
	.pe-type { font-size: 9px; color: var(--text-tertiary); text-transform: uppercase; }
</style>
