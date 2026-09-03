/** Mirrors the backend's Pydantic schemas (backend/app/schemas/). */

export interface Citation {
  label: string;
  document_name: string;
  start_page: number;
  end_page: number;
  chunk_id: string;
}

/**
 * `POST /api/v1/chat` response.
 *
 * Everything after `citations` is diagnostic — the backend exposes its own
 * pipeline internals so a caller can tell WHY it got the answer it got,
 * rather than string-matching the reply text.
 */
export interface ChatResponse {
  reply: string;
  conversation_id: string | null;
  citations: Citation[];
  rewritten_query: string | null;
  english_query: string | null;
  retrieved_chunk_count: number;
  verified: boolean | null;
  verification_reason: string | null;
  answer_source: AnswerSource | null;
}

/** Which branch of the graph produced the reply. */
export type AnswerSource =
  /** Grounded in the documents and verified against them. */
  | "rag"
  /** Not a company question — politely declined, with no LLM call at all. */
  | "off_topic"
  /** A company question the documents didn't cover. */
  | "general_fallback"
  /** The documents produced an answer, but verification couldn't confirm it. */
  | "unverified_fallback"
  /** A greeting, a thank-you, or a question about the assistant itself. */
  | "small_talk";

export interface HealthResponse {
  status: string;
  database: string;
  vector_store: string;
  [key: string]: unknown;
}

export interface ResetResponse {
  message: string;
  details: string[];
}

export interface RestartResponse {
  message: string;
  will_restart: boolean;
  details: string[];
}

export interface BackendLogLine {
  raw: string;
  timestamp: string | null;
  level: string | null;
  logger: string | null;
  message: string | null;
}

export interface LogTailResponse {
  lines: BackendLogLine[];
  next_offset: number;
  file_size: number;
  truncated: boolean;
}

/** A chat message as rendered in the UI. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  at: number;
  /** Populated for assistant messages; drives the diagnostic badges. */
  meta?: {
    answerSource: AnswerSource | null;
    citations: Citation[];
    rewrittenQuery: string | null;
    englishQuery: string | null;
    retrievedChunkCount: number;
    verified: boolean | null;
    verificationReason: string | null;
    elapsedMs: number;
  };
  /** Set when the request itself failed, rather than the model refusing. */
  error?: string;
}

/**
 * One line in the console panel. Lines come from two places, kept
 * distinguishable on purpose: `backend` lines are the real server log,
 * `client` lines are what this UI did (requests, timings, failures).
 */
export interface ConsoleEntry {
  id: string;
  origin: "backend" | "client";
  level: LogLevel;
  at: number;
  logger: string | null;
  text: string;
}

export type LogLevel =
  | "DEBUG"
  | "INFO"
  | "WARNING"
  | "ERROR"
  | "CRITICAL"
  | "PLAIN";
