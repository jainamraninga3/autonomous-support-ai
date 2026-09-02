"use client";

import { useCallback, useEffect, useState } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { ConsolePanel } from "@/components/ConsolePanel";
import { Toolbar } from "@/components/Toolbar";
import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";
import { useConsole } from "@/lib/useConsole";
import type { ChatMessage, HealthResponse } from "@/lib/types";

let messageCounter = 0;
const nextMessageId = () => `m${Date.now()}-${messageCounter++}`;

export default function Page() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

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

  const send = async (text: string) => {
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
    </main>
  );
}
