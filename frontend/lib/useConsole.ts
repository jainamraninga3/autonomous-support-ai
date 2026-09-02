"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { tailLogs } from "./api";
import type { ConsoleEntry, LogLevel } from "./types";

/** Bounded, so a long session can't grow the DOM without limit. */
const MAX_ENTRIES = 2000;
const POLL_MS = 1000;

let counter = 0;
const nextId = () => `c${Date.now()}-${counter++}`;

function normaliseLevel(level: string | null): LogLevel {
  const upper = (level ?? "").toUpperCase();
  if (["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].includes(upper)) {
    return upper as LogLevel;
  }
  return "PLAIN";
}

/**
 * Streams the backend's real log file into the console panel, and lets the
 * UI push its own entries alongside it.
 *
 * Polling with a byte cursor rather than SSE: the cursor is stateless, it
 * survives a reload, and it can't leak a long-lived connection per tab.
 * The first poll intentionally starts at the END of the file (offset -1),
 * so opening the page shows what happens next instead of replaying up to
 * 5MB of history.
 */
export function useConsole() {
  const [entries, setEntries] = useState<ConsoleEntry[]>([]);
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null);
  const [paused, setPaused] = useState(false);

  const offsetRef = useRef<number>(-1);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const push = useCallback((entry: Omit<ConsoleEntry, "id" | "at">) => {
    setEntries((prev) => {
      const next = [...prev, { ...entry, id: nextId(), at: Date.now() }];
      return next.length > MAX_ENTRIES ? next.slice(next.length - MAX_ENTRIES) : next;
    });
  }, []);

  const pushClient = useCallback(
    (text: string, level: LogLevel = "INFO") =>
      push({ origin: "client", level, logger: "ui", text }),
    [push],
  );

  const clear = useCallback(() => setEntries([]), []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      if (cancelled) return;
      if (pausedRef.current) {
        timer = setTimeout(poll, POLL_MS);
        return;
      }

      try {
        const data = await tailLogs(offsetRef.current);
        if (cancelled) return;

        offsetRef.current = data.next_offset;
        setBackendReachable(true);

        if (data.truncated) {
          pushClient("Backend log rotated (5MB limit) — reading restarted from the top.", "WARNING");
        }

        if (data.lines.length) {
          setEntries((prev) => {
            const incoming: ConsoleEntry[] = data.lines.map((line) => ({
              id: nextId(),
              origin: "backend",
              level: normaliseLevel(line.level),
              at: Date.now(),
              logger: line.logger,
              text: line.message ?? line.raw,
            }));
            const next = [...prev, ...incoming];
            return next.length > MAX_ENTRIES ? next.slice(next.length - MAX_ENTRIES) : next;
          });
        }
      } catch {
        if (cancelled) return;
        // Don't spam the console with one entry per failed poll — the
        // status dot carries this, and a restart makes it expected.
        setBackendReachable(false);
      }

      timer = setTimeout(poll, POLL_MS);
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [pushClient]);

  return { entries, pushClient, clear, backendReachable, paused, setPaused };
}
