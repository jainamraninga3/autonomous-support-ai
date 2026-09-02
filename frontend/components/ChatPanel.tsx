"use client";

import { useEffect, useRef, useState } from "react";
import { MessageBubble } from "./MessageBubble";
import type { ChatMessage } from "@/lib/types";

const SUGGESTIONS = [
  "How many sick leaves do I get in a year?",
  "Compare sick leave and earned leave.",
  "मुझे कितने दिन की छुट्टी मिलती है?",
  "hi",
];

export function ChatPanel({
  messages,
  sending,
  conversationId,
  onSend,
}: {
  messages: ChatMessage[];
  sending: boolean;
  conversationId: string | null;
  onSend: (text: string) => void;
}) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, sending]);

  const submit = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || sending) return;
    onSend(trimmed);
    setDraft("");
    inputRef.current?.focus();
  };

  return (
    <section className="panel flex min-h-0 flex-col overflow-hidden">
      <header className="flex shrink-0 items-center justify-between border-b border-white/10 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-sky-400 to-indigo-500 text-sm">
            💬
          </span>
          <div>
            <h2 className="text-sm font-semibold text-white">Chat</h2>
            <p className="text-[11px] text-slate-500">
              {conversationId
                ? `conversation ${conversationId.slice(0, 8)}…`
                : "new conversation"}
            </p>
          </div>
        </div>
        <span className="text-[11px] text-slate-500">
          {messages.filter((m) => m.role === "user").length} sent
        </span>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center gap-5 text-center">
            <div>
              <p className="text-sm font-medium text-slate-300">
                Ask about your ingested documents
              </p>
              <p className="mx-auto mt-1 max-w-sm text-xs leading-relaxed text-slate-500">
                Answers come with citations. Ask in any language — Hindi,
                Gujarati, Marathi, Kannada, Bengali — and the reply comes back
                in that language, even though the documents are English.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => submit(suggestion)}
                  className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-400 transition-colors hover:border-sky-500/40 hover:bg-sky-500/10 hover:text-sky-200"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {sending && (
          <div className="flex animate-fade-up justify-start">
            <div className="flex items-center gap-2 rounded-2xl rounded-bl-md border border-white/10 bg-slate-800/70 px-4 py-3">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-slate-400"
                  style={{ animationDelay: `${i * 160}ms` }}
                />
              ))}
              <span className="ml-1 text-xs text-slate-500">
                classifying → retrieving → generating → verifying
              </span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <footer className="shrink-0 border-t border-white/10 p-3">
        <div className="flex items-end gap-2 rounded-xl border border-white/10 bg-slate-950/60 p-2 focus-within:border-sky-500/40">
          <textarea
            ref={inputRef}
            rows={1}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends; Shift+Enter is a newline.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit(draft);
              }
            }}
            placeholder="Ask a question…  (Enter to send, Shift+Enter for a new line)"
            className="max-h-32 min-h-[2.25rem] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none"
          />
          <button
            onClick={() => submit(draft)}
            disabled={!draft.trim() || sending}
            className="shrink-0 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 px-3.5 py-2 text-sm font-medium text-white shadow-lg shadow-indigo-950/40 transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
          >
            Send
          </button>
        </div>
      </footer>
    </section>
  );
}
