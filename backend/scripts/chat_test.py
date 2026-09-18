"""Quick manual tester for POST /api/v1/chat. No venv, no dependencies.

Talks to the running backend over HTTP like any other client — it does
NOT import backend code, so it tests what the app actually serves rather
than what the source says it should.

    python scripts/chat_test.py "what is leave policy"
    python scripts/chat_test.py "my name is Jainam" "what is my name"
    python scripts/chat_test.py -f scripts/chat_test_questions.txt
    python scripts/chat_test.py                      # interactive

Questions given together run in ONE conversation, in order, so follow-ups
and memory-style questions work ("my name is X" then "what is my name").
Pass --new-session to start each question fresh instead.

In a file, one question per line; a BLANK LINE starts a new conversation,
and `#` comments out a line. That makes each block a scenario.

Printed per answer: route, chunk count, whether the verifier passed,
seconds, and the word/bullet counts — the shape numbers are what the
brief-vs-detailed rules in `answer_generator.py` are tuned against.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8000/api/v1/chat"

# Filled in by `_login`. /api/v1/chat requires a logged-in caller as of
# 2026-09-18, so this script authenticates like any other client.
_AUTH_HEADERS = {}


def _login(chat_url, username, password, timeout):
    """Log in and remember the bearer token for the rest of the run.

    Derives the auth URL from --url so pointing this at a different host
    still works with one flag.
    """
    base = chat_url.split("/api/v1/")[0]
    request = urllib.request.Request(
        f"{base}/api/v1/auth/login",
        data=json.dumps({"username": username, "password": password}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        _AUTH_HEADERS["Authorization"] = f"Bearer {json.loads(response.read().decode('utf-8'))['token']}"


def _ask(url, message, conversation_id, user_id, timeout):
    # `user_id` is still sent, and the server now IGNORES it — identity
    # comes from the token. Kept so the flag doesn't break, and because a
    # request that still carries it is exactly what the override protects
    # against.
    payload = {"message": message, "user_id": user_id}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **_AUTH_HEADERS},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, time.monotonic() - started


def _shape(answer):
    lines = answer.split("\n")
    bullets = sum(1 for line in lines if line.strip().startswith(("-", "*")))
    nested = sum(1 for line in lines if re.match(r"^\s+[-*]", line))
    return len(answer.split()), bullets, nested


def _print_answer(question, body, seconds):
    answer = body.get("reply", "")
    words, bullets, nested = _shape(answer)
    print("\n" + "=" * 72)
    print(f"Q: {question}")
    print("-" * 72)
    print(answer or "(empty reply)")
    print("-" * 72)
    verified = body.get("verified")
    print(
        f"route={body.get('answer_source')}  chunks={body.get('retrieved_chunk_count')}  "
        f"verified={verified}  {seconds:.1f}s  |  {words} words, {bullets} bullets, {nested} nested"
    )
    print(f"conversation_id={body.get('conversation_id')}")
    if body.get("verification_reason") and verified is False:
        print(f"verifier: {body['verification_reason']}")
    citations = body.get("citations") or []
    if citations:
        names = sorted({c.get("document_name", "?") for c in citations})
        print(f"sources: {', '.join(names)}")


def _read_questions(path):
    """Return blocks of questions; a blank line separates conversations."""
    blocks, current = [], []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("#"):
                continue
            if not line:
                if current:
                    blocks.append(current)
                    current = []
                continue
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _run_block(url, questions, user_id, timeout, new_session, conversation_id=None):
    for question in questions:
        try:
            body, seconds = _ask(url, question, conversation_id, user_id, timeout)
        except urllib.error.HTTPError as exc:
            # The backend ANSWERED, with an error. Print its body: an
            # exhausted Groq quota arrives here as a 5xx, and "is the
            # backend up?" is the wrong thing to tell someone whose
            # backend is up and whose API keys are spent for the day.
            print(file=sys.stderr)
            print(f"HTTP {exc.code} from {url}", file=sys.stderr)
            print(exc.read().decode("utf-8", "replace")[:800], file=sys.stderr)
            return False
        except urllib.error.URLError as exc:
            # The usual cause is the stack being down, so say that rather
            # than printing a bare socket error.
            print(f"\nCould not reach {url}: {exc}", file=sys.stderr)
            print("Is the backend up? `docker compose up -d`", file=sys.stderr)
            return False
        _print_answer(question, body, seconds)
        if not new_session:
            conversation_id = body.get("conversation_id")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="*", help="questions to ask, in order")
    parser.add_argument("-f", "--file", help="file of questions; blank line = new conversation")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--user",
        default="chat-test",
        help="user_id sent with every message. IGNORED by the server since 2026-09-18 — "
        "identity comes from the logged-in account. Use --username to change who you are.",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("ASAI_USERNAME", "demo"),
        help="API account to log in as. Default: $ASAI_USERNAME, else 'demo'.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("ASAI_PASSWORD", "demo123"),
        help="Password for --username. Default: $ASAI_PASSWORD, else 'demo123'.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="start a fresh conversation for every question (no history)",
    )
    parser.add_argument(
        "--conversation-id",
        help=(
            "join an existing conversation. Pass another account's conversation_id "
            "together with a different --username to check isolation: you must NOT get "
            "their history back, and the reply should carry a different conversation_id"
        ),
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        _login(args.url, args.username, args.password, args.timeout)
    except urllib.error.HTTPError as exc:
        print(
            f"Login failed for '{args.username}' (HTTP {exc.code}). The seeded accounts "
            f"are demo/demo123 and admin/admin123.",
            file=sys.stderr,
        )
        return 1
    except urllib.error.URLError as exc:
        print(f"Could not reach the backend: {exc}", file=sys.stderr)
        return 1

    if args.file:
        blocks = _read_questions(args.file)
    elif args.questions:
        blocks = [args.questions]
    else:
        print("Interactive mode — one conversation, blank line to quit.\n")
        conversation_id = args.conversation_id
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not question:
                return 0
            try:
                body, seconds = _ask(args.url, question, conversation_id, args.user, args.timeout)
            except urllib.error.HTTPError as exc:
                print(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:800]}", file=sys.stderr)
                continue
            except urllib.error.URLError as exc:
                print(f"Could not reach {args.url}: {exc}", file=sys.stderr)
                return 1
            _print_answer(question, body, seconds)
            if not args.new_session:
                conversation_id = body.get("conversation_id")

    for index, questions in enumerate(blocks, start=1):
        if len(blocks) > 1:
            print(f"\n\n##### conversation {index} of {len(blocks)} #####")
        if not _run_block(
            args.url, questions, args.user, args.timeout, args.new_session, args.conversation_id
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
