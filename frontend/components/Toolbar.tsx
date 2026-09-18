"use client";

import { useState } from "react";
import { API_BASE_URL } from "@/lib/api";
import type { HealthResponse, User } from "@/lib/types";

export interface PendingAction {
  key: string;
  label: string;
  /** Shown in the confirm dialog — say exactly what is destroyed. */
  warning: string;
  run: () => Promise<void>;
}

function StatusDot({ ok }: { ok: boolean | null }) {
  const color =
    ok === null ? "bg-slate-500" : ok ? "bg-emerald-400" : "bg-rose-500";
  return (
    <span className="relative flex h-2 w-2">
      {ok && (
        <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${color} opacity-60`} />
      )}
      <span className={`relative inline-flex h-2 w-2 rounded-full ${color}`} />
    </span>
  );
}

export function Toolbar({
  health,
  backendReachable,
  busy,
  user,
  onRefreshChat,
  onResetPostgres,
  onResetVectorStore,
  onResetAll,
  onRestart,
  onCheckHealth,
  onLogin,
  onSignup,
  onLogout,
}: {
  health: HealthResponse | null;
  backendReachable: boolean | null;
  busy: string | null;
  /** null = logged out. Drives both the auth area and who sees the
   *  destructive buttons. */
  user: User | null;
  onRefreshChat: () => void;
  onResetPostgres: () => Promise<void>;
  onResetVectorStore: () => Promise<void>;
  onResetAll: () => Promise<void>;
  onRestart: () => Promise<void>;
  onCheckHealth: () => void;
  onLogin: () => void;
  onSignup: () => void;
  onLogout: () => void;
}) {
  const [pending, setPending] = useState<PendingAction | null>(null);

  // Admin-only. This is presentation, NOT the security boundary — the
  // backend requires `role == "admin"` on every one of these routes
  // (`require_admin`), so hiding the buttons only spares a plain user a
  // guaranteed 403.
  const isAdmin = user?.role === "admin";

  // Every one of these is irreversible, so none of them fire on a single
  // click — they go through the confirm dialog below.
  const dangerous: PendingAction[] = [
    {
      key: "postgres",
      label: "Reset PostgreSQL",
      warning:
        "Deletes every document, its chunk text, the stored PDF bytes, and all chat history from PostgreSQL. " +
        "The PDFs are only stored there — if you don't have the source files elsewhere, they are gone. " +
        "Weaviate is left alone, so embeddings will be left pointing at documents that no longer exist.",
      run: onResetPostgres,
    },
    {
      key: "weaviate",
      label: "Reset Weaviate",
      warning:
        "Deletes and recreates the Weaviate DocumentChunk collection, discarding every embedding. " +
        "PostgreSQL is untouched, so your documents and chunk text survive — you can re-embed them " +
        "with POST /documents/{id}/ingest without re-uploading.",
      run: onResetVectorStore,
    },
    {
      key: "all",
      label: "Reset everything",
      warning:
        "Both of the above: all documents, PDFs, chunks, chat history, and every embedding. " +
        "This is a full wipe back to an empty system.",
      run: onResetAll,
    },
    {
      key: "restart",
      label: "Restart backend",
      warning:
        "Sends SIGTERM to the backend process. Under docker compose it restarts automatically " +
        "(restart: unless-stopped) and should be back in a few seconds. If you are running uvicorn " +
        "directly, NOTHING will restart it and the API will simply stop.",
      run: onRestart,
    },
  ];

  return (
    <>
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-sky-400 via-indigo-500 to-violet-600 text-lg shadow-lg shadow-indigo-950/50">
            ◆
          </span>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-white">
              Autonomous Support AI
            </h1>
            <div className="flex items-center gap-2 text-[11px] text-slate-500">
              <StatusDot ok={backendReachable} />
              <button
                onClick={onCheckHealth}
                className="transition-colors hover:text-slate-300"
                title={`Re-check ${API_BASE_URL}/health`}
              >
                {backendReachable === null
                  ? "checking…"
                  : backendReachable === false
                    ? "backend unreachable"
                    : health
                      ? `db ${health.database} · vectors ${health.vector_store}`
                      : "connected"}
              </button>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <button onClick={onRefreshChat} className="btn-ghost" disabled={!!busy}>
            ↺ New chat
          </button>

          {isAdmin && (
            <>
              <span className="mx-1 h-5 w-px bg-white/10" />
              {dangerous.map((action) => (
                <button
                  key={action.key}
                  onClick={() => setPending(action)}
                  disabled={!!busy}
                  className={action.key === "restart" ? "btn-warn" : "btn-danger"}
                >
                  {busy === action.key ? "working…" : action.label}
                </button>
              ))}
            </>
          )}

          <span className="mx-1 h-5 w-px bg-white/10" />
          {user ? (
            <>
              <span className="flex items-center gap-1.5 px-1 text-xs text-slate-300">
                {user.full_name}
                {isAdmin && (
                  <span className="badge border-indigo-400/30 bg-indigo-500/15 text-indigo-200">
                    admin
                  </span>
                )}
              </span>
              <button onClick={onLogout} className="btn-ghost" disabled={!!busy}>
                Log out
              </button>
            </>
          ) : (
            <>
              <button onClick={onLogin} className="btn-primary">
                Log in
              </button>
              <button onClick={onSignup} className="btn-ghost">
                Sign up
              </button>
            </>
          )}
        </div>
      </header>

      {pending && (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm"
          onClick={() => setPending(null)}
        >
          <div
            className="panel w-full max-w-md p-5"
            onClick={(event) => event.stopPropagation()}
          >
            <h3 className="text-base font-semibold text-white">{pending.label}?</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">{pending.warning}</p>
            <p className="mt-3 text-xs font-medium text-rose-300">This cannot be undone.</p>
            <div className="mt-5 flex justify-end gap-2">
              <button onClick={() => setPending(null)} className="btn-ghost">
                Cancel
              </button>
              <button
                onClick={async () => {
                  const action = pending;
                  setPending(null);
                  await action.run();
                }}
                className={pending.key === "restart" ? "btn-warn" : "btn-danger"}
              >
                Yes, {pending.label.toLowerCase()}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
