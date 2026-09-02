import type {
  ChatResponse,
  HealthResponse,
  LogTailResponse,
  ResetResponse,
  RestartResponse,
} from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/**
 * Thrown for any non-2xx response, carrying the status and the backend's
 * own error text so the UI can show something specific instead of
 * "something went wrong".
 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    // fetch() rejects for network failure AND for a CORS block, and the
    // browser deliberately doesn't say which. Both mean the same thing to
    // the user, so name both possibilities rather than guessing.
    throw new ApiError(
      `Could not reach the backend at ${API_BASE_URL}. It may be down, or ` +
        `CORS may be blocking this origin (see CORS_ALLOW_ORIGINS in backend/.env).`,
      0,
      cause,
    );
  }

  const text = await response.text();
  let body: unknown = text;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      /* keep the raw text — an HTML error page, most likely */
    }
  }

  if (!response.ok) {
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : typeof body === "string" && body
          ? body.slice(0, 300)
          : response.statusText;
    throw new ApiError(`${response.status} ${detail}`, response.status, body);
  }

  return body as T;
}

export function sendChat(message: string, conversationId: string | null) {
  return request<ChatResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
}

export function getHealth() {
  return request<HealthResponse>("/health");
}

export function tailLogs(offset: number) {
  return request<LogTailResponse>(`/api/v1/logs/tail?offset=${offset}`);
}

export function resetPostgres() {
  return request<ResetResponse>("/api/v1/admin/reset/postgres", { method: "POST" });
}

export function resetVectorStore() {
  return request<ResetResponse>("/api/v1/admin/reset/vector-store", { method: "POST" });
}

export function resetAll() {
  return request<ResetResponse>("/api/v1/admin/reset/all", { method: "POST" });
}

export function restartBackend() {
  return request<RestartResponse>("/api/v1/admin/restart", { method: "POST" });
}
