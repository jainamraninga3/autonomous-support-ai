import type {
  ChatResponse,
  HealthResponse,
  LoginResponse,
  LogTailResponse,
  ResetResponse,
  RestartResponse,
  User,
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

/**
 * Where the session token lives between page loads.
 *
 * `localStorage`, not a cookie: the backend authenticates with an
 * `Authorization: Bearer` header, which means nothing is sent automatically
 * by the browser and CSRF is structurally impossible here. The trade-off is
 * that XSS could read it — acceptable for an internal tool with no
 * third-party scripts, and the same trade-off every token-in-header SPA
 * makes.
 *
 * Every access is wrapped: `localStorage` throws outright in a few privacy
 * configurations, and this module is imported during SSR where `window`
 * does not exist at all.
 */
const TOKEN_KEY = "asai.token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  if (typeof window === "undefined") return;
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* private mode / storage disabled — the session just won't survive a reload */
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  const token = getToken();
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        // Attached to EVERY request rather than per-call, so a route that
        // becomes authenticated later needs no change here. The backend
        // ignores it on the routes that stay public (/health, /logs/tail).
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init?.headers ?? {}),
      },
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

// --- Auth ---
// `setToken` is called by the CALLER (page.tsx), not here: these functions
// stay pure request wrappers like every other one in this file, and the
// component that owns the user state owns the side effect too.

export function signup(username: string, password: string, fullName: string) {
  return request<LoginResponse>("/api/v1/auth/signup", {
    method: "POST",
    body: JSON.stringify({ username, password, full_name: fullName }),
  });
}

export function login(username: string, password: string) {
  return request<LoginResponse>("/api/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function logout() {
  return request<{ message: string }>("/api/v1/auth/logout", { method: "POST" });
}

/** Resolve the stored token to a user. A 401 here means "logged out",
 *  which is a normal state and not an error worth showing. */
export function getMe() {
  return request<User>("/api/v1/auth/me");
}
