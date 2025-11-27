/**
 * Frontend Logger Service
 * Captures console logs, errors, and unhandled exceptions and sends them to the backend.
 * Logs are stored in a file accessible to Claude for debugging.
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  data?: unknown;
  source?: string;
  stack?: string;
}

class FrontendLogger {
  private buffer: LogEntry[] = [];
  private flushInterval: number | null = null;
  private readonly maxBufferSize = 50;
  private readonly flushIntervalMs = 5000;
  private originalConsole: {
    log: typeof console.log;
    info: typeof console.info;
    warn: typeof console.warn;
    error: typeof console.error;
  };

  constructor() {
    this.originalConsole = {
      log: console.log.bind(console),
      info: console.info.bind(console),
      warn: console.warn.bind(console),
      error: console.error.bind(console),
    };
  }

  init() {
    this.interceptConsole();
    this.setupGlobalErrorHandlers();
    this.startFlushInterval();
    this.info('Frontend logger initialized', { userAgent: navigator.userAgent });
  }

  private interceptConsole() {
    console.log = (...args: unknown[]) => {
      this.originalConsole.log(...args);
      this.log('debug', this.formatArgs(args));
    };

    console.info = (...args: unknown[]) => {
      this.originalConsole.info(...args);
      this.log('info', this.formatArgs(args));
    };

    console.warn = (...args: unknown[]) => {
      this.originalConsole.warn(...args);
      this.log('warn', this.formatArgs(args));
    };

    console.error = (...args: unknown[]) => {
      this.originalConsole.error(...args);
      this.log('error', this.formatArgs(args), this.extractStack(args));
    };
  }

  private setupGlobalErrorHandlers() {
    window.addEventListener('error', (event) => {
      this.log('error', `Uncaught error: ${event.message}`, event.error?.stack, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      const reason = event.reason;
      const message = reason instanceof Error ? reason.message : String(reason);
      const stack = reason instanceof Error ? reason.stack : undefined;
      this.log('error', `Unhandled promise rejection: ${message}`, stack);
    });
  }

  private formatArgs(args: unknown[]): string {
    return args
      .map((arg) => {
        if (typeof arg === 'string') return arg;
        if (arg instanceof Error) return `${arg.name}: ${arg.message}`;
        try {
          return JSON.stringify(arg);
        } catch {
          return String(arg);
        }
      })
      .join(' ');
  }

  private extractStack(args: unknown[]): string | undefined {
    for (const arg of args) {
      if (arg instanceof Error && arg.stack) {
        return arg.stack;
      }
    }
    return undefined;
  }

  private log(level: LogLevel, message: string, stack?: string, data?: unknown) {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      source: 'frontend',
    };

    if (stack) entry.stack = stack;
    if (data) entry.data = data;

    this.buffer.push(entry);

    if (this.buffer.length >= this.maxBufferSize) {
      this.flush();
    }
  }

  debug(message: string, data?: unknown) {
    this.log('debug', message, undefined, data);
  }

  info(message: string, data?: unknown) {
    this.log('info', message, undefined, data);
  }

  warn(message: string, data?: unknown) {
    this.log('warn', message, undefined, data);
  }

  error(message: string, error?: Error | unknown) {
    const stack = error instanceof Error ? error.stack : undefined;
    const data = error instanceof Error ? undefined : error;
    this.log('error', message, stack, data);
  }

  logErrorBoundary(error: Error, errorInfo: { componentStack?: string }) {
    this.log('error', `React ErrorBoundary caught: ${error.message}`, error.stack, {
      componentStack: errorInfo.componentStack,
    });
    this.flush(); // Immediately flush error boundary errors
  }

  logApiError(url: string, status: number, message: string) {
    this.log('error', `API Error: ${status} ${url} - ${message}`, undefined, {
      url,
      status,
    });
  }

  private startFlushInterval() {
    this.flushInterval = window.setInterval(() => {
      this.flush();
    }, this.flushIntervalMs);
  }

  async flush() {
    if (this.buffer.length === 0) return;

    const entries = [...this.buffer];
    this.buffer = [];

    try {
      await fetch('/api/v1/logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entries }),
      });
    } catch {
      // If flush fails, restore entries to buffer (but limit size)
      this.buffer = [...entries.slice(-20), ...this.buffer].slice(-this.maxBufferSize);
    }
  }

  destroy() {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    // Restore original console methods
    console.log = this.originalConsole.log;
    console.info = this.originalConsole.info;
    console.warn = this.originalConsole.warn;
    console.error = this.originalConsole.error;
  }
}

export const logger = new FrontendLogger();
