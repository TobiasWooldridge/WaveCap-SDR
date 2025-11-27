import React, { createContext, useContext } from "react";
import { useHealthStream } from "../hooks/useHealthStream";
import type { ErrorEvent, ErrorStats, ErrorType } from "../types";

interface ErrorContextValue {
  isConnected: boolean;
  stats: Partial<Record<ErrorType, ErrorStats>>;
  recentErrors: ErrorEvent[];
  hasActiveErrors: boolean;
}

const ErrorContext = createContext<ErrorContextValue | null>(null);

interface ErrorProviderProps {
  children: React.ReactNode;
}

export function ErrorProvider({ children }: ErrorProviderProps) {
  const health = useHealthStream();

  return <ErrorContext.Provider value={health}>{children}</ErrorContext.Provider>;
}

export function useErrorContext(): ErrorContextValue {
  const ctx = useContext(ErrorContext);
  if (!ctx) {
    throw new Error("useErrorContext must be used within ErrorProvider");
  }
  return ctx;
}

// Optional hook that returns null if not within provider (for components that may be rendered outside)
export function useErrorContextOptional(): ErrorContextValue | null {
  return useContext(ErrorContext);
}
