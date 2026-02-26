import { writable, derived } from 'svelte/store';

export type ToolWindowPosition = 'left' | 'right' | 'bottom';

export interface ToolWindowDef {
	id: string;
	label: string;
	icon: string;
	position: ToolWindowPosition;
	open: boolean;
	size: number;
}

const defaultWindows: ToolWindowDef[] = [
	// Left
	{ id: 'model',    label: 'Model',      icon: 'folder',         position: 'left',   open: true,  size: 240 },
	{ id: 'palette',  label: 'Palette',    icon: 'components',     position: 'left',   open: false, size: 240 },
	{ id: 'plugins',  label: 'Plugins',    icon: 'puzzle',         position: 'left',   open: false, size: 260 },
	{ id: 'devices',  label: 'Devices',    icon: 'device-mobile',  position: 'left',   open: false, size: 240 },
	{ id: 'tests',    label: 'Tests',      icon: 'test-pipe',      position: 'left',   open: false, size: 240 },
	// Right
	{ id: 'props',    label: 'Properties', icon: 'adjustments',    position: 'right',  open: true,  size: 260 },
	{ id: 'bench',    label: 'Benchmarks', icon: 'chart-bar',      position: 'right',  open: false, size: 260 },
	{ id: 'surgery',  label: 'Surgery',    icon: 'scissors',       position: 'right',  open: false, size: 260 },
	// Bottom
	{ id: 'console',  label: 'Console',    icon: 'terminal-2',     position: 'bottom', open: true,  size: 160 },
	{ id: 'devout',   label: 'Device',     icon: 'device-mobile',  position: 'bottom', open: false, size: 160 },
	{ id: 'terminal', label: 'Terminal',   icon: 'code',           position: 'bottom', open: false, size: 160 },
];

export const toolWindows = writable<ToolWindowDef[]>(structuredClone(defaultWindows));

export function toggleToolWindow(id: string) {
	toolWindows.update(windows => {
		const target = windows.find(w => w.id === id);
		if (!target) return windows;
		// Accordion: close others in same position
		for (const w of windows) {
			if (w.position === target.position && w.id !== id) w.open = false;
		}
		target.open = !target.open;
		return [...windows];
	});
}

// Derived: open tool windows by position
export const leftWindow = derived(toolWindows, $w => $w.find(w => w.position === 'left' && w.open) ?? null);
export const rightWindow = derived(toolWindows, $w => $w.find(w => w.position === 'right' && w.open) ?? null);
export const bottomWindow = derived(toolWindows, $w => $w.find(w => w.position === 'bottom' && w.open) ?? null);
export const leftWindows = derived(toolWindows, $w => $w.filter(w => w.position === 'left'));
export const rightWindows = derived(toolWindows, $w => $w.filter(w => w.position === 'right'));
export const bottomWindows = derived(toolWindows, $w => $w.filter(w => w.position === 'bottom'));
