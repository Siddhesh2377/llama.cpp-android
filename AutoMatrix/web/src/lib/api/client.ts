import { consoleStore } from '$lib/stores/console';
import { model, modelLoading, modelError, type ModelInfo } from '$lib/stores/model';
import { devices, systemInfo, serverOnline, type DeviceInfo, type SystemInfo } from '$lib/stores/device';

const BASE = '/api';

async function request<T>(method: string, path: string, body?: unknown): Promise<T | null> {
	try {
		const res = await fetch(`${BASE}${path}`, {
			method,
			headers: body ? { 'Content-Type': 'application/json' } : {},
			body: body ? JSON.stringify(body) : undefined
		});
		if (!res.ok) {
			const text = await res.text();
			throw new Error(`${res.status}: ${text}`);
		}
		return await res.json() as T;
	} catch (e) {
		const msg = e instanceof Error ? e.message : String(e);
		consoleStore.error(`API ${method} ${path}: ${msg}`, 'api');
		return null;
	}
}

export async function checkHealth(): Promise<boolean> {
	const res = await request<{ status: string }>('GET', '/health');
	const online = res?.status === 'ok';
	serverOnline.set(online);
	return online;
}

export async function loadModel(filepath: string): Promise<ModelInfo | null> {
	modelLoading.set(true);
	modelError.set(null);
	consoleStore.info(`Loading model: ${filepath}`, 'model');

	const info = await request<ModelInfo>('POST', '/model/load', { filepath });
	if (info) {
		model.set(info);
		consoleStore.success(`Model loaded: ${info.name || info.arch} (${info.quantType})`, 'model');
	} else {
		modelError.set('Failed to load model');
	}
	modelLoading.set(false);
	return info;
}

export async function getModelInfo(): Promise<ModelInfo | null> {
	const info = await request<ModelInfo>('GET', '/model/info');
	if (info) model.set(info);
	return info;
}

export async function getDevices(): Promise<DeviceInfo[]> {
	const res = await request<{ devices: DeviceInfo[] }>('GET', '/devices');
	const devs = res?.devices || [];
	devices.set(devs);
	return devs;
}

export async function getSystemInfo(): Promise<SystemInfo | null> {
	const info = await request<SystemInfo>('GET', '/system');
	if (info) systemInfo.set(info);
	return info;
}

export interface ScannedFile {
	path: string;
	name: string;
	size: number;
}

export async function scanModels(directory: string): Promise<ScannedFile[]> {
	consoleStore.info(`Scanning for models in: ${directory}`, 'scan');
	const res = await request<{ directory: string; files: ScannedFile[] }>('POST', '/models/scan', { directory });
	const files = res?.files || [];
	if (files.length > 0) {
		consoleStore.success(`Found ${files.length} GGUF file(s)`, 'scan');
	} else {
		consoleStore.warn(`No GGUF files found in ${directory}`, 'scan');
	}
	return files;
}

export interface GraphNode {
	id: string;
	label: string;
	type: string;
	sublabel: string;
	layer: number;
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface GraphEdge {
	from: string;
	to: string;
}

export interface ModelGraph {
	nodes: GraphNode[];
	edges: GraphEdge[];
	layerCount: number;
	arch: string;
}

export async function getModelGraph(): Promise<ModelGraph | null> {
	const graph = await request<ModelGraph>('GET', '/model/graph');
	return graph;
}

export interface InferenceConfig {
	serial?: string;
	modelPath?: string;
	mmprojPath?: string;
	imagePath?: string;
	prompt?: string;
	threads?: string;
	quant?: string;
	maxTokens?: string;
}

export interface InferenceResult {
	success: boolean;
	output: string;
}

export async function deployPush(serial?: string, binaryDir?: string): Promise<InferenceResult | null> {
	consoleStore.info('Pushing binary to device...', 'deploy');
	const res = await request<InferenceResult>('POST', '/deploy/push', { serial, binaryDir });
	if (res?.success) {
		consoleStore.success('Binary pushed successfully', 'deploy');
	} else {
		consoleStore.error(`Push failed: ${res?.output || 'unknown error'}`, 'deploy');
	}
	return res;
}

export async function runInference(config: InferenceConfig): Promise<InferenceResult | null> {
	consoleStore.info('Starting inference on device...', 'inference');
	const res = await request<InferenceResult>('POST', '/inference/run', config);
	if (res?.success) {
		consoleStore.success('Inference complete', 'inference');
	} else {
		consoleStore.error(`Inference failed: ${res?.output || 'unknown error'}`, 'inference');
	}
	return res;
}

// --- Plugins ---

export interface AmxpPlugin {
	id: string;
	name: string;
	type: 'architecture' | 'backend' | 'quant' | 'sampling';
	version: number;
	filepath: string;
	body: Record<string, unknown>;
}

export interface PluginsResponse {
	pluginDir: string;
	plugins: AmxpPlugin[];
}

export async function getPlugins(): Promise<AmxpPlugin[]> {
	const res = await request<PluginsResponse>('GET', '/plugins');
	return res?.plugins || [];
}

export async function getPluginsByType(type: string): Promise<Record<string, unknown>[]> {
	const res = await request<Record<string, unknown>[]>('GET', `/plugins/${type}`);
	return res || [];
}

export async function savePlugin(type: string, id: string, body: Record<string, unknown>): Promise<boolean> {
	const res = await request<{ success: boolean }>('POST', '/plugins/save', { type, id, body });
	if (res?.success) {
		consoleStore.success(`Plugin saved: ${id}`, 'plugins');
	}
	return res?.success || false;
}

// Poll server health + devices every 5s
let pollTimer: ReturnType<typeof setInterval> | null = null;

export function startPolling() {
	if (pollTimer) return;
	checkHealth();
	getDevices();
	getSystemInfo();
	pollTimer = setInterval(() => {
		checkHealth();
		getDevices();
	}, 5000);
}

export function stopPolling() {
	if (pollTimer) {
		clearInterval(pollTimer);
		pollTimer = null;
	}
}
