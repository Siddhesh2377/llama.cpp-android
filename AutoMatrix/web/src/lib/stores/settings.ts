import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

export type Theme = 'dark' | 'light';

function createThemeStore() {
	const stored = browser ? localStorage.getItem('amx-theme') as Theme : null;
	const { subscribe, set, update } = writable<Theme>(stored || 'dark');

	return {
		subscribe,
		set(value: Theme) {
			if (browser) {
				localStorage.setItem('amx-theme', value);
				document.documentElement.setAttribute('data-theme', value);
			}
			set(value);
		},
		toggle() {
			update(t => {
				const next = t === 'dark' ? 'light' : 'dark';
				if (browser) {
					localStorage.setItem('amx-theme', next);
					document.documentElement.setAttribute('data-theme', next);
				}
				return next;
			});
		}
	};
}

function createZoomStore() {
	const stored = browser ? parseFloat(localStorage.getItem('amx-zoom') || '1') : 1;
	const { subscribe, set } = writable(stored);

	return {
		subscribe,
		set(value: number) {
			const clamped = Math.max(0.5, Math.min(2.0, value));
			if (browser) {
				localStorage.setItem('amx-zoom', String(clamped));
				document.documentElement.style.setProperty('--zoom', String(clamped));
			}
			set(clamped);
		}
	};
}

export const theme = createThemeStore();
export const zoom = createZoomStore();

export type Tab = 'architecture' | 'graph' | 'training' | 'surgery' | 'simulation';
export const activeTab = writable<Tab>('architecture');

export type NavItem = 'models' | 'editor' | 'settings';
export const activeNav = writable<NavItem>('models');

export const showSettings = writable(false);
export const showConsole = writable(true);
