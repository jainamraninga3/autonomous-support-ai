"use client";

import { useState } from "react";
import { Markdown } from "./Markdown";
import type { AnswerSource, ChatMessage } from "@/lib/types";

/**
 * How each graph branch is labelled. These are the four real outcomes the
 * backend distinguishes, and showing them is the point — "no citations"
 * means something very different for a refusal than for a RAG answer.
 */
const SOURCE_STYLES: Record<AnswerSource, { label: string; className: string; hint: string }> = {
  rag: {
    label: "from your documents",
    className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    hint: "Grounded in the retrieved chunks and checked against them.",
  },
  general_fallback: {
    label: "general knowledge",
    className: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    hint: "Nothing relevant was found in your documents, so this is NOT sourced from them.",
  },
  small_talk: {
    label: "small talk",
    className: "border-sky-500/30 bg-sky-500/10 text-sky-300",
    hint: "A greeting — answered directly, no retrieval ran.",
  },
  off_topic_refusal: {
    label: "off topic",
    className: "border-slate-500/30 bg-slate-500/10 text-slate-300",
    hint: "Judged unrelated to your documents. Your message was never sent to the LLM.",
  },
};

function Badge({
  className,
  title,
  children,
}: {
  className: string;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <span className={`badge ${className}`} title={title}>
      {children}
    </span>
  );
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const [showDetail, setShowDetail] = useState(false);
  const isUser = message.role === "user";
  const meta = message.meta;

  if (isUser) {
    return (
      <div className="flex animate-fade-up justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-gradient-to-br from-sky-500 to-indigo-600 px-4 py-2.5 text-sm text-white shadow-lg shadow-indigo-950/40">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </div>
    );
  }

  const source = meta?.answerSource ? SOURCE_STYLES[meta.answerSource] : null;

  return (
    <div className="flex animate-fade-up justify-start">
      <div className="max-w-[92%] space-y-2">
        <div
          className={`rounded-2xl rounded-bl-md border px-4 py-3 text-sm shadow-lg shadow-black/20 ${
            message.error
              ? "border-rose-500/30 bg-rose-950/30 text-rose-200"
              : "border-white/10 bg-slate-800/70 text-slate-100"
          }`}
        >
          {/* An error is our own plain-text message, not model output —
              rendering it as markdown would mangle paths and URLs in it. */}
          {message.error ? (
            <p className="whitespace-pre-wrap break-words leading-relaxed">{message.content}</p>
          ) : (
            <Markdown content={message.content} />
          )}
        </div>

        {meta && (
          <div className="flex flex-wrap items-center gap-1.5 pl-1">
            {source && (
              <Badge className={source.className} title={source.hint}>
                {source.label}
              </Badge>
            )}

            {/* Only meaningful on the RAG path — null elsewhere because
                there is nothing to verify. */}
            {meta.verified === true && (
              <Badge
                className="border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                title={meta.verificationReason ?? undefined}
              >
                verified
              </Badge>
            )}
            {meta.verified === false && (
              <Badge
                className="border-rose-500/30 bg-rose-500/10 text-rose-300"
                title={meta.verificationReason ?? undefined}
              >
                unverified
              </Badge>
            )}

            <Badge
              className="border-white/10 bg-white/5 text-slate-400"
              title="Chunks retrieved from Weaviate before generation. Can be > 0 even when nothing relevant was found — retrieval finds candidates, generation decides if they answer the question."
            >
              {meta.retrievedChunkCount} chunks
            </Badge>

            <Badge className="border-white/10 bg-white/5 text-slate-400">
              {(meta.elapsedMs / 1000).toFixed(1)}s
            </Badge>

            {(meta.citations.length > 0 ||
              meta.rewrittenQuery ||
              meta.englishQuery ||
              meta.verificationReason) && (
              <button
                onClick={() => setShowDetail((v) => !v)}
                className="badge border-white/10 bg-white/5 text-slate-400 hover:bg-white/10 hover:text-slate-200"
              >
                {showDetail ? "hide details" : "details"}
              </button>
            )}
          </div>
        )}

        {meta && showDetail && (
          <div className="animate-fade-up space-y-3 rounded-xl border border-white/10 bg-slate-950/60 p-3 text-xs">
            {meta.rewrittenQuery && (
              <Detail label="Rewritten for retrieval" hint="Used only for search — the answer is generated from your original question.">
                <code className="text-sky-300">{meta.rewrittenQuery}</code>
              </Detail>
            )}

            {meta.englishQuery && (
              <Detail
                label="English retrieval query"
                hint="A second retrieval pass in English, merged with the original-language pass. The answer itself is never translated."
              >
                <code className="text-violet-300">{meta.englishQuery}</code>
              </Detail>
            )}

            {meta.verificationReason && (
              <Detail label="Verifier said">
                <span className="text-slate-300">{meta.verificationReason}</span>
              </Detail>
            )}

            {meta.citations.length > 0 && (
              <Detail label={`Sources (${meta.citations.length})`}>
                <ul className="mt-1 space-y-1">
                  {meta.citations.map((citation) => (
                    <li key={citation.chunk_id} className="flex items-baseline gap-2 text-slate-400">
                      <span className="shrink-0 rounded bg-white/5 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                        {citation.label}
                      </span>
                      <span className="truncate">
                        {citation.document_name}
                        <span className="text-slate-500">
                          {" "}
                          · p.{citation.start_page}
                          {citation.end_page !== citation.start_page && `–${citation.end_page}`}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              </Detail>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Detail({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500" title={hint}>
        {label}
      </p>
      <div className="break-words">{children}</div>
    </div>
  );
}
