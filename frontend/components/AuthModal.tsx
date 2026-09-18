"use client";

import { useEffect, useRef, useState } from "react";
import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";
import type { LoginResponse } from "@/lib/types";

export type AuthMode = "login" | "signup";

/**
 * One modal for both login and signup — they share every field except
 * "full name", and two near-identical components would drift apart.
 *
 * On success it hands the whole `LoginResponse` up; storing the token and
 * setting the user is `page.tsx`'s job, since that is where the rest of the
 * session state already lives.
 */
export function AuthModal({
  mode,
  onModeChange,
  onClose,
  onSuccess,
}: {
  mode: AuthMode;
  onModeChange: (mode: AuthMode) => void;
  onClose: () => void;
  onSuccess: (result: LoginResponse) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const firstFieldRef = useRef<HTMLInputElement>(null);

  const isSignup = mode === "signup";

  useEffect(() => {
    firstFieldRef.current?.focus();
  }, [mode]);

  // Escape closes, matching the reset-confirm dialog's click-outside.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const submit = async () => {
    if (busy) return;
    const user = username.trim();
    if (!user || !password || (isSignup && !fullName.trim())) {
      setError("Please fill in every field.");
      return;
    }
    // Mirrors the backend's own Field constraints, so the common mistakes
    // are caught without a round trip. The server still validates — this is
    // convenience, not the boundary.
    if (isSignup && user.length < 3) {
      setError("Username must be at least 3 characters.");
      return;
    }
    if (isSignup && password.length < 4) {
      setError("Password must be at least 4 characters.");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const result = isSignup
        ? await api.signup(user, password, fullName.trim())
        : await api.login(user, password);
      onSuccess(result);
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? // The backend sends `401 Invalid username or password.` — strip
            // the status code, which means nothing to the person reading it.
            caught.message.replace(/^\d{3}\s*/, "")
          : String(caught),
      );
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="panel w-full max-w-sm p-5"
        onClick={(event) => event.stopPropagation()}
      >
        <h3 className="text-base font-semibold text-white">
          {isSignup ? "Create an account" : "Log in"}
        </h3>
        <p className="mt-1 text-xs text-slate-500">
          {isSignup
            ? "New accounts are standard users. Admin access is granted separately."
            : "You need an account to ask questions."}
        </p>

        <div className="mt-4 space-y-2.5">
          {isSignup && (
            <Field
              label="Full name"
              value={fullName}
              onChange={setFullName}
              onEnter={submit}
              autoComplete="name"
              inputRef={firstFieldRef}
            />
          )}
          <Field
            label="Username"
            value={username}
            onChange={setUsername}
            onEnter={submit}
            autoComplete="username"
            inputRef={isSignup ? undefined : firstFieldRef}
          />
          <Field
            label="Password"
            value={password}
            onChange={setPassword}
            onEnter={submit}
            type="password"
            autoComplete={isSignup ? "new-password" : "current-password"}
          />
        </div>

        {error && (
          <p className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
            {error}
          </p>
        )}

        <div className="mt-5 flex items-center justify-between gap-2">
          <button
            onClick={() => {
              // Keep whatever was typed: someone who tried to log in and
              // then realised they need to sign up should not retype it.
              setError(null);
              onModeChange(isSignup ? "login" : "signup");
            }}
            className="text-xs text-slate-400 underline-offset-2 transition-colors hover:text-slate-200 hover:underline"
          >
            {isSignup ? "I already have an account" : "Create an account"}
          </button>
          <div className="flex gap-2">
            <button onClick={onClose} className="btn-ghost" disabled={busy}>
              Cancel
            </button>
            <button onClick={submit} className="btn-primary" disabled={busy}>
              {busy ? "working…" : isSignup ? "Sign up" : "Log in"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  onEnter,
  type = "text",
  autoComplete,
  inputRef,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  onEnter: () => void;
  type?: string;
  autoComplete?: string;
  inputRef?: React.RefObject<HTMLInputElement | null>;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </span>
      <input
        ref={inputRef}
        type={type}
        value={value}
        autoComplete={autoComplete}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            onEnter();
          }
        }}
        className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 outline-none transition-colors placeholder:text-slate-600 focus:border-indigo-400/50 focus:bg-white/10"
      />
    </label>
  );
}
