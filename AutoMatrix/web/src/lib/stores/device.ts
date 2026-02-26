import { writable, derived } from 'svelte/store';

export interface DeviceInfo {
	serial: string;
	model: string;
	chipset: string;
	ram: string;
	status: 'online' | 'offline' | 'unauthorized';
	abiList: string;
}

export interface SystemInfo {
	hostname: string;
	platform: string;
	cpuModel: string;
	cpuCores: number;
	memoryTotal: number;
	memoryFree: number;
}

export const devices = writable<DeviceInfo[]>([]);
export const systemInfo = writable<SystemInfo | null>(null);
export const serverOnline = writable(false);

export const primaryDevice = derived(devices, $d =>
	$d.find(d => d.status === 'online') || $d[0] || null
);
