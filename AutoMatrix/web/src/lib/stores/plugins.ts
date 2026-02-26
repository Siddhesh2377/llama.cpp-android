import { writable, derived } from 'svelte/store';
import type { AmxpPlugin } from '$lib/api/client';

export const plugins = writable<AmxpPlugin[]>([]);

export const archPlugins = derived(plugins, $p => $p.filter(p => p.type === 'architecture'));
export const backendPlugins = derived(plugins, $p => $p.filter(p => p.type === 'backend'));
export const quantPlugins = derived(plugins, $p => $p.filter(p => p.type === 'quant'));
export const samplingPlugins = derived(plugins, $p => $p.filter(p => p.type === 'sampling'));
