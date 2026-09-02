"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ConsoleEntry, LogLevel } from "@/lib/types";

const LEVEL_STYLES: Record<LogLevel, string> = {
  DEBUG: "text-slate-500",
  INFO: "text-sky-300",
  WARNING: "text-amber-300",
  ERROR: "text-rose-300",
  CRITICAL: "text-rose-200 font-semibold",
  PLAIN: "text-slate-400",
};

const LEVEL_TAG: Record<LogLevel, string> = {
  DEBUG: "DBG",
  INFO: "INF",
  WARNING: "WRN",
  ERROR: "ERR",
  CRITICAL: "CRT",
  PLAIN: "···",
};

type Filter = "all" | "app" | "problems";

/**
 * "app" hides the libraries. Without it the console is mostly SQLAlchemy
 * echo and httpx request lines, which drown the handful of lines that
 * actually explain what the pipeline decided.
 */
function matchesFilter(entry: ConsoleEntry, filter: Filter): boolean {
  if (filter === "all") return true;
  if (filter === "problems") return ["WARNING", "ERROR", "CRITICAL"].includes(entry.level);
  return entry.origin === "client" || (entry.logger?.startsWith("app.") ?? false);
}

export function ConsolePanel({
  entries,
  onClear,
  backendReachable,
  paused,
  onTogglePause,
}: {
  entries: ConsoleEntry[];
  onClear: () => void;
  backendReachable: boolean | null;
  paused: boolean;
  onTogglePause: () => void;
}) {
  const [filter, setFilter] = useState<Filter>("app");
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const visible = useMemo(
    () => entries.filter((entry) => matchesFilter(entry, filter)),
    [entries, filter],
  );

  useEffect(() => {
    if (!autoScroll) return;
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [visible.length, autoScroll]);

  return (
    <section className="panel flex min-h-0 flex-col overflow-hidden">
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-white/10 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-emerald-400 to-teal-500 text-sm">
            ⌘
          </span>
          <div>
            <h2 className="text-sm font-semibold text-white">Backend console</h2>
            <p className="text-[11px] text-slate-500">
              live from <code>backend/logs/app.log</code>
              {paused && <span className="ml-1 text-amber-400">· paused</span>}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <div className="mr-1 flex rounded-lg border border-white/10 bg-white/5 p-0.5">
            {(["app", "all", "problems"] as Filter[]).map((option) => (
              <button
                key={option}
                onClick={() => setFilter(option)}
                className={`rounded-md px-2 py-1 text-[11px] font-medium capitalize transition-colors ${
                  filter === option
                    ? "bg-white/10 text-white"
                    : "text-slate-500 hover:text-slate-300"
                }`}
                title={
                  option === "app"
                    ? "Only this app's own log lines — hides SQLAlchemy and httpx noise"
                    : option === "problems"
                      ? "Warnings and errors only"
                      : "Everything, including libraries"
                }
              >
                {option}
              </button>
            ))}
          </div>

          <button
            onClick={() => setAutoScroll((v) => !v)}
            className={
              autoScroll
                ? "inline-flex items-center gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-300 transition-colors hover:bg-sky-500/20"
                : "btn-ghost"
            }
            title="Keep scrolling to the newest line"
          >
            follow
          </button>
          <button onClick={onTogglePause} className="btn-ghost" title="Stop polling the log file">
            {paused ? "resume" : "pause"}
          </button>
          <button onClick={onClear} className="btn-ghost">
            clear
          </button>
        </div>
      </header>

      <div
        ref={scrollRef}
        onScroll={(event) => {
          // Scrolling up disables follow, so reading history isn't yanked
          // back to the bottom by the next log line.
          const element = event.currentTarget;
          const atBottom =
            element.scrollHeight - element.scrollTop - element.clientHeight < 40;
          if (!atBottom && autoScroll) setAutoScroll(false);
        }}
        className="min-h-0 flex-1 overflow-y-auto bg-slate-950/50 px-3 py-2 font-mono text-[11.5px] leading-relaxed"
      >
        {visible.length === 0 && (
          <div className="flex h-full items-center justify-center px-6 text-center">
            <p className="text-xs text-slate-600">
              {backendReachable === false
                ? "Backend unreachable — nothing to show. Start it, or check CORS_ALLOW_ORIGINS."
                : entries.length === 0
                  ? "Waiting for backend activity… send a message and the pipeline appears here."
                  : `No lines match the "${filter}" filter.`}
            </p>
          </div>
        )}

        {visible.map((entry) => (
          <div
            key={entry.id}
            className="group flex gap-2 rounded px-1 py-[1px] hover:bg-white/[0.03]"
          >
            <span className="shrink-0 select-none text-slate-600">
              {new Date(entry.at).toLocaleTimeString([], { hour12: false })}
            </span>
            <span className={`shrink-0 select-none ${LEVEL_STYLES[entry.level]}`}>
              {LEVEL_TAG[entry.level]}
            </span>
            {entry.origin === "client" ? (
              <span className="shrink-0 select-none text-violet-400" title="From this UI, not the server">
                ui
              </span>
            ) : (
              entry.logger && (
                <span
                  className="hidden shrink-0 select-none truncate text-slate-600 sm:inline sm:max-w-[11rem]"
                  title={entry.logger}
                >
                  {entry.logger.replace(/^app\./, "")}
                </span>
              )
            )}
            <span className={`whitespace-pre-wrap break-all ${LEVEL_STYLES[entry.level]}`}>
              {entry.text}
            </span>
          </div>
        ))}
      </div>

      <footer className="flex shrink-0 items-center justify-between border-t border-white/10 px-4 py-2 text-[11px] text-slate-600">
        <span>
          {visible.length} / {entries.length} lines
        </span>
        <span>
          {filter === "app"
            ? "showing app.* only"
            : filter === "problems"
              ? "warnings + errors"
              : "all sources"}
        </span>
      </footer>
    </section>
  );
}
