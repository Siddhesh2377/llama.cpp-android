import { writable, derived } from 'svelte/store';

export type LogLevel = 'info' | 'warn' | 'error' | 'success' | 'debug';

export interface LogEntry {
	id: number;
	timestamp: Date;
	level: LogLevel;
	message: string;
	source?: string;
}

let nextId = 0;

function createConsoleStore() {
	const { subscribe, update } = writable<LogEntry[]>([]);

	function log(level: LogLevel, message: string, source?: string) {
		update(entries => [
			...entries,
			{ id: nextId++, timestamp: new Date(), level, message, source }
		].slice(-500));
	}

	return {
		subscribe,
		info: (msg: string, src?: string) => log('info', msg, src),
		warn: (msg: string, src?: string) => log('warn', msg, src),
		error: (msg: string, src?: string) => log('error', msg, src),
		success: (msg: string, src?: string) => log('success', msg, src),
		debug: (msg: string, src?: string) => log('debug', msg, src),
		clear: () => update(() => [])
	};
}

export const consoleStore = createConsoleStore();
