"use client";

import { useCallback, useEffect, useState } from "react";
import { AuthModal, type AuthMode } from "@/components/AuthModal";
import { ChatPanel } from "@/components/ChatPanel";
import { ConsolePanel } from "@/components/ConsolePanel";
import { Toolbar } from "@/components/Toolbar";
import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";
import { useConsole } from "@/lib/useConsole";
import type { ChatMessage, HealthResponse, User } from "@/lib/types";

let messageCounter = 0;
const nextMessageId = () => `m${Date.now()}-${messageCounter++}`;

export default function Page() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode | null>(null);

  const { entries, pushClient, clear, backendReachable, paused, setPaused } = useConsole();

  const checkHealth = useCallback(async () => {
    try {
      const result = await api.getHealth();
      setHealth(result);
      pushClient(`health: db=${result.database} vector_store=${result.vector_store}`);
    } catch (error) {
      setHealth(null);
      pushClient(
        error instanceof ApiError ? error.message : "Health check failed.",
        "ERROR",
      );
    }
  }, [pushClient]);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  // Restore a session from the token in localStorage. A 401 here is the
  // NORMAL logged-out state, not a failure — it is silent, and the stored
  // token is cleared so a stale one isn't sent on every later request.
  useEffect(() => {
    if (!api.getToken()) return;
    void (async () => {
      try {
        const me = await api.getMe();
        setUser(me);
        pushClient(`signed in as ${me.username} (${me.role})`);
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          api.setToken(null);
        } else {
          pushClient(
            `could not restore session: ${error instanceof ApiError ? error.message : String(error)}`,
            "WARNING",
          );
        }
      }
    })();
  }, [pushClient]);

  const onAuthSuccess = ({ token, user: account }: { token: string; user: User }) => {
    api.setToken(token);
    setUser(account);
    setAuthMode(null);
    pushClient(`signed in as ${account.username} (${account.role})`);
  };

  const onLogout = async () => {
    try {
      await api.logout();
    } catch {
      // The server rejecting the logout must never leave the UI stuck in a
      // logged-in state — clearing the token locally is what actually logs
      // the person out from their point of view.
    }
    api.setToken(null);
    setUser(null);
    // A new user must not inherit the previous one's conversation: the
    // backend would refuse to hand over that session anyway (ownership is
    // enforced in SQL), so keeping the id would silently start a new
    // conversation with the old messages still on screen.
    setMessages([]);
    setConversationId(null);
    pushClient("logged out");
  };

  const send = async (text: string) => {
    // Defense in depth on top of the backend's 401 — this only exists so
    // the person gets a sentence instead of a failed request.
    if (!user) {
      setMessages((prev) => [
        ...prev,
        { id: nextMessageId(), role: "user", content: text, at: Date.now() },
        {
          id: nextMessageId(),
          role: "assistant",
          content: "Please log in to ask a question — use the **Log in** button at the top right.",
          at: Date.now(),
        },
      ]);
      setAuthMode("login");
      return;
    }

    setMessages((prev) => [
      ...prev,
      { id: nextMessageId(), role: "user", content: text, at: Date.now() },
    ]);
    setSending(true);
    pushClient(`POST /api/v1/chat  ${JSON.stringify(text)}`);

    const started = performance.now();
    try {
      const response = await api.sendChat(text, conversationId);
      const elapsedMs = performance.now() - started;

      // The backend creates the session on the first message; reuse its id
      // from then on so the conversation threads server-side.
      if (response.conversation_id && response.conversation_id !== conversationId) {
        setConversationId(response.conversation_id);
      }

      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: "assistant",
          content: response.reply,
          at: Date.now(),
          meta: {
            answerSource: response.answer_source,
            citations: response.citations,
            rewrittenQuery: response.rewritten_query,
            englishQuery: response.english_query,
            retrievedChunkCount: response.retrieved_chunk_count,
            verified: response.verified,
            verificationReason: response.verification_reason,
            elapsedMs,
          },
        },
      ]);

      pushClient(
        `↳ ${(elapsedMs / 1000).toFixed(1)}s · source=${response.answer_source ?? "?"}` +
          ` · chunks=${response.retrieved_chunk_count}` +
          ` · citations=${response.citations.length}` +
          (response.verified === null ? "" : ` · verified=${response.verified}`),
      );
      if (response.verified === false && response.verification_reason) {
        pushClient(`verifier rejected the answer: ${response.verification_reason}`, "WARNING");
      }
    } catch (error) {
      const message = error instanceof ApiError ? error.message : String(error);
      // The 24h session expired, or it was revoked. Drop to the logged-out
      // state rather than leaving a name in the toolbar for a session the
      // server no longer honours.
      if (error instanceof ApiError && error.status === 401) {
        api.setToken(null);
        setUser(null);
        setAuthMode("login");
      }
      setMessages((prev) => [
        ...prev,
        {
          id: nextMessageId(),
          role: "assistant",
          content: message,
          at: Date.now(),
          error: message,
        },
      ]);
      pushClient(`request failed: ${message}`, "ERROR");
    } finally {
      setSending(false);
    }
  };

  /** Wraps an admin action with the busy flag and console reporting. */
  const runAction = async (
    key: string,
    label: string,
    call: () => Promise<{ message: string; details: string[] }>,
  ) => {
    setBusy(key);
    pushClient(`${label}…`, "WARNING");
    try {
      const result = await call();
      pushClient(result.message, "WARNING");
      result.details.forEach((detail) => pushClient(`  ${detail}`, "WARNING"));
      await checkHealth();
    } catch (error) {
      pushClient(
        `${label} failed: ${error instanceof ApiError ? error.message : String(error)}`,
        "ERROR",
      );
    } finally {
      setBusy(null);
    }
  };

  const refreshChat = () => {
    setMessages([]);
    setConversationId(null);
    pushClient("chat cleared — next message starts a new conversation");
  };

  return (
    <main className="flex h-screen flex-col">
      <Toolbar
        health={health}
        backendReachable={backendReachable}
        busy={busy}
        user={user}
        onLogin={() => setAuthMode("login")}
        onSignup={() => setAuthMode("signup")}
        onLogout={() => void onLogout()}
        onCheckHealth={() => void checkHealth()}
        onRefreshChat={refreshChat}
        onResetPostgres={() =>
          runAction("postgres", "Resetting PostgreSQL", api.resetPostgres).then(() => {
            // Every document is gone, so the current conversation's
            // citations now reference nothing. Start clean.
            refreshChat();
          })
        }
        onResetVectorStore={() =>
          runAction("weaviate", "Resetting Weaviate", api.resetVectorStore)
        }
        onResetAll={() =>
          runAction("all", "Resetting PostgreSQL + Weaviate", api.resetAll).then(refreshChat)
        }
        onRestart={async () => {
          setBusy("restart");
          pushClient("Requesting backend restart…", "WARNING");
          try {
            const result = await api.restartBackend();
            pushClient(result.message, result.will_restart ? "WARNING" : "ERROR");
            result.details.forEach((detail) => pushClient(`  ${detail}`, "WARNING"));
          } catch (error) {
            // The connection often drops as the process exits before the
            // response is fully read — that means it worked, so don't
            // report it as a failure.
            pushClient(
              `Restart requested; the connection dropped as expected (${
                error instanceof ApiError ? error.message : String(error)
              })`,
              "WARNING",
            );
          } finally {
            setBusy(null);
            // Give the supervisor time to bring it back before re-checking.
            setTimeout(() => void checkHealth(), 6000);
          }
        }}
      />

      {/* 60/40 split as specified — side by side on large screens, stacked
          on small ones where a 40%-height console would be unusable. */}
      <div className="grid min-h-0 flex-1 gap-3 p-3 pt-0 lg:grid-cols-[3fr_2fr]">
        <ChatPanel
          messages={messages}
          sending={sending}
          conversationId={conversationId}
          onSend={(text) => void send(text)}
        />
        <ConsolePanel
          entries={entries}
          onClear={clear}
          backendReachable={backendReachable}
          paused={paused}
          onTogglePause={() => setPaused((v) => !v)}
        />
      </div>

      {authMode && (
        <AuthModal
          mode={authMode}
          onModeChange={setAuthMode}
          onClose={() => setAuthMode(null)}
          onSuccess={onAuthSuccess}
        />
      )}
    </main>
  );
}
