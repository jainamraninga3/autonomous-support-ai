"""Upload AND ingest every PDF in a folder, in one command.

    python -m scripts.ingest_folder ../policy

Talks to the running API over HTTP (upload -> ingest, per file), so it
works the same whether the backend runs in Docker or locally, and needs
no database credentials of its own.

Why a script rather than a batch endpoint: a 30-file batch as a single
HTTP request takes minutes and reports nothing until it finishes, which
is worse than N quick calls you can watch. Here every file prints its
result as it completes, and Ctrl+C loses only the file in flight —
everything already ingested is committed.

Ends with a cross-store check (`--verify`, on by default): what
PostgreSQL holds, and whether each version is actually embedded in
Weaviate.

It also reports EXTRACTION DENSITY — chunks per page — and flags
anything thin. Extraction is text-layer only (no OCR), so a
screenshot-based manual extracts nothing but page headers and captions:
it ingests "successfully", reports `embedded: yes`, and then answers
nothing. A zero-chunk check alone misses these, because they do produce a
handful of chunks. Measured on a real policy folder, genuine text PDFs
land at 2-6 chunks/page while screenshot manuals sat at 0.2-1.0.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests

from scripts._console import fix_windows_console_encoding

fix_windows_console_encoding()

DEFAULT_API = "http://localhost:8000"
DEFAULT_WEAVIATE = "http://localhost:8080"

# Embedding a large manual can take a while, and the ingest call doesn't
# return until it's done. Long enough not to trip on a 100-page PDF,
# short enough to fail rather than hang forever.
UPLOAD_TIMEOUT = 300
INGEST_TIMEOUT = 900

# Chunks-per-page below which a document is almost certainly image-based.
# With CHUNK_SIZE_TOKENS=100 this is roughly 120 tokens of extracted text
# per page; a page of real prose is several hundred. Set well below the
# observed floor for genuine text PDFs (~2 chunks/page on a real policy
# folder) so it flags the hopeless cases, not the merely terse ones.
MIN_CHUNKS_PER_PAGE = 1.2


class Row:
    """One file's outcome, kept together for the summary table."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "failed"
        self.document_id: str | None = None
        self.version: int | None = None
        self.pages: int | None = None
        self.chunks: int | None = None
        self.embedded: int | None = None
        self.duplicate = False
        self.error: str | None = None
        self.seconds = 0.0


# Filled in by `_login`. `/api/v1/documents/*` became admin-only on
# 2026-09-18, so this script — which is an HTTP client like any other —
# now has to authenticate. Credentials come from ASAI_USERNAME/ASAI_PASSWORD
# or `--username`/`--password`, defaulting to the seeded demo admin.
_AUTH_HEADERS: dict[str, str] = {}


def _login(api: str, username: str, password: str) -> None:
    """Exchange credentials for a bearer token, once, for the whole run."""
    response = requests.post(
        f"{api}/api/v1/auth/login",
        json={"username": username, "password": password},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"Login failed for '{username}' (HTTP {response.status_code}). "
            f"Uploading needs an ADMIN account — the seeded one is admin/admin123, "
            f"or pass --username/--password."
        )
    _AUTH_HEADERS["Authorization"] = f"Bearer {response.json()['token']}"


def _post(url: str, **kwargs) -> dict:
    kwargs["headers"] = {**_AUTH_HEADERS, **kwargs.get("headers", {})}
    response = requests.post(url, **kwargs)
    if not response.ok:
        # FastAPI's error shape is {"detail": ...}; fall back to raw text
        # for anything else (a proxy error page, say).
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text[:300]
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return response.json()


def ingest_one(api: str, path: Path, force: bool) -> Row:
    row = Row(path.name)
    started = time.perf_counter()
    try:
        with path.open("rb") as handle:
            upload = _post(
                f"{api}/api/v1/documents/upload",
                files={"file": (path.name, handle, "application/pdf")},
                timeout=UPLOAD_TIMEOUT,
            )

        row.document_id = upload["document_id"]
        row.version = upload["version"]
        row.pages = upload["page_count"]
        row.chunks = upload["chunk_count"]
        row.duplicate = upload["is_duplicate"]

        ingest = _post(
            f"{api}/api/v1/documents/{row.document_id}/ingest",
            params={"force": str(force).lower()},
            timeout=INGEST_TIMEOUT,
        )
        row.embedded = ingest["embedded_chunk_count"]
        row.version = ingest["version"]

        if ingest["already_embedded"]:
            row.status = "already embedded"
        elif row.duplicate:
            row.status = "duplicate"
        else:
            row.status = "ingested"
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        row.error = f"{type(exc).__name__}: {exc}"
    row.seconds = time.perf_counter() - started
    return row


def verify(api: str, weaviate: str) -> None:
    print("\n" + "=" * 78)
    print("VERIFICATION")
    print("=" * 78)

    try:
        response = requests.get(f"{api}/api/v1/documents", headers=_AUTH_HEADERS, timeout=60)
        response.raise_for_status()
        documents = response.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not list documents: {exc}")
        return

    print(f"\nPostgreSQL — {len(documents)} document(s):\n")
    print(f"  {'document_id':38} {'name':44} {'ver':>3} {'pages':>5}  embedded")
    print(f"  {'-' * 38} {'-' * 44} {'-' * 3} {'-' * 5}  {'-' * 8}")
    not_embedded = []
    for document in documents:
        for version in document["versions"]:
            # `is_embedded` is checked live against Weaviate by the API, so
            # this single line confirms BOTH stores agree about this
            # version — the actual cross-store check.
            mark = "yes" if version["is_embedded"] else "NO"
            if not version["is_embedded"]:
                not_embedded.append(f"{document['name']} v{version['version']}")
            print(
                f"  {document['document_id']:38} {document['name'][:44]:44} "
                f"{version['version']:>3} {version['page_count']:>5}  {mark}"
            )

    try:
        query = (
            '{Aggregate{DocumentChunk(groupBy:["document_name"])'
            "{groupedBy{value} meta{count}}}}"
        )
        response = requests.post(
            f"{weaviate}/v1/graphql", json={"query": query}, timeout=30
        )
        response.raise_for_status()
        groups = response.json()["data"]["Aggregate"]["DocumentChunk"]
    except Exception as exc:  # noqa: BLE001
        print(f"\nWeaviate — could not read chunk counts ({exc})")
        print("  (skip: the vector store may not be reachable from this host)")
        groups = None

    if groups is not None:
        total = sum(group["meta"]["count"] for group in groups)
        counts = {g["groupedBy"]["value"]: g["meta"]["count"] for g in groups}
        pages_by_name = {
            document["name"]: sum(v["page_count"] for v in document["versions"])
            for document in documents
        }

        print(f"\nWeaviate — {total} embedded chunk(s) across {len(groups)} document(s):\n")
        print(f"  {'chunks':>6} {'pages':>5} {'per pg':>6}  document")
        print(f"  {'-' * 6} {'-' * 5} {'-' * 6}  {'-' * 52}")

        thin = []
        # Sorted by density, so the least useful documents sit at the top
        # where they'll actually be noticed.
        for name in sorted(counts, key=lambda n: counts[n] / max(pages_by_name.get(n, 1), 1)):
            chunks = counts[name]
            pages = pages_by_name.get(name, 0)
            density = chunks / pages if pages else 0.0
            mark = ""
            if pages and density < MIN_CHUNKS_PER_PAGE:
                mark = "  <-- almost no text"
                thin.append((name, pages, chunks, density))
            print(f"  {chunks:>6} {pages:>5} {density:>6.1f}  {name[:52]}{mark}")

        if thin:
            print(f"\n  ⚠ {len(thin)} document(s) extracted almost no text.")
            print("    Extraction is text-layer only (no OCR), so these are")
            print("    screenshot/scanned PDFs. They are stored and embedded, and")
            print("    report 'embedded: yes', but hold too little text to answer")
            print("    anything — a 20-page manual yielding 6 chunks is not")
            print("    searchable content:")
            for name, pages, chunks, density in thin:
                print(f"      {density:.1f} chunks/page  ({chunks} chunks / {pages} pages)  {name}")

    if not_embedded:
        print("\n  ⚠ NOT embedded in Weaviate — these will never be retrieved:")
        for item in not_embedded:
            print(f"      {item}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload and ingest every PDF in a folder via the running API."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="../policy",
        type=Path,
        help="Folder containing PDFs (default: ../policy, relative to backend/).",
    )
    parser.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default {DEFAULT_API}).")
    parser.add_argument(
        "--weaviate",
        default=DEFAULT_WEAVIATE,
        help=f"Weaviate base URL, used only by the verification step (default {DEFAULT_WEAVIATE}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-embed even when a version is already embedded. Use after changing "
            "CHUNK_SIZE_TOKENS together with a fresh upload — identical file content "
            "is otherwise deduplicated by hash and keeps its existing chunks."
        ),
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("ASAI_USERNAME", "admin"),
        help="API account to authenticate as. Must be an admin — the documents "
        "routes are admin-only. Default: $ASAI_USERNAME, else 'admin'.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("ASAI_PASSWORD", "admin123"),
        help="Password for --username. Default: $ASAI_PASSWORD, else 'admin123'.",
    )
    parser.add_argument("--recursive", action="store_true", help="Include PDFs in subfolders.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the verification report.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Print the verification report for what's already ingested, and exit.",
    )
    args = parser.parse_args()

    api = args.api.rstrip("/")

    try:
        health = requests.get(f"{api}/health", timeout=15).json()
        print(
            f"Backend: {api}  (database={health.get('database')}, "
            f"vector_store={health.get('vector_store')})"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot reach the backend at {api}: {exc}", file=sys.stderr)
        print("Start it first:  docker compose up -d", file=sys.stderr)
        return 1

    # Before anything that talks to a gated route — including --verify-only,
    # which lists documents.
    try:
        _login(api, args.username, args.password)
    except Exception as exc:  # noqa: BLE001
        print(f"{exc}", file=sys.stderr)
        return 1

    if args.verify_only:
        verify(api, args.weaviate.rstrip("/"))
        return 0

    folder = args.folder
    if not folder.is_dir():
        print(f"Not a folder: {folder.resolve()}", file=sys.stderr)
        return 1

    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdfs = sorted(p for p in folder.glob(pattern) if p.is_file())
    if not pdfs:
        print(f"No PDFs found in {folder.resolve()}", file=sys.stderr)
        return 1

    total_mb = sum(p.stat().st_size for p in pdfs) / 1_048_576
    print(f"Found {len(pdfs)} PDF(s) in {folder.resolve()} ({total_mb:.1f} MB)")
    print("The first file may take several minutes while the embedding model loads.\n")

    rows: list[Row] = []
    run_started = time.perf_counter()

    for index, path in enumerate(pdfs, start=1):
        # Printed BEFORE the work, and flushed, so a long embed doesn't
        # look like a hang.
        print(f"[{index:>2}/{len(pdfs)}] {path.name[:60]:60} ", end="", flush=True)
        row = ingest_one(api, path, args.force)
        rows.append(row)

        if row.error:
            print(f"FAILED  ({row.seconds:.1f}s)")
            print(f"         {row.error}")
        else:
            note = ""
            if row.chunks == 0:
                note = "  ⚠ 0 chunks — no text layer (scanned PDF?)"
            elif row.pages and row.chunks / row.pages < MIN_CHUNKS_PER_PAGE:
                note = f"  ⚠ thin: {row.chunks / row.pages:.1f} chunks/page — mostly images?"
            print(
                f"{row.status:16} {row.chunks:>4} chunks, {row.embedded:>4} embedded "
                f"({row.seconds:.1f}s){note}"
            )

    elapsed = time.perf_counter() - run_started
    ingested = sum(1 for r in rows if r.status in {"ingested", "duplicate"})
    already = sum(1 for r in rows if r.status == "already embedded")
    failed = sum(1 for r in rows if r.error)
    empty = [r for r in rows if not r.error and r.chunks == 0]
    embedded_total = sum(r.embedded or 0 for r in rows)

    print("\n" + "=" * 78)
    print(
        f"Done in {elapsed:.1f}s — {ingested} ingested, {already} already embedded, "
        f"{failed} failed, {embedded_total} chunks embedded"
    )

    if empty:
        print(f"\n  ⚠ {len(empty)} file(s) produced ZERO chunks. Extraction is text-layer")
        print("    only (no OCR), so these are scanned/image PDFs. They are stored but")
        print("    can never be retrieved or cited:")
        for row in empty:
            print(f"      {row.name}")

    if failed:
        print(f"\n  {failed} file(s) failed:")
        for row in rows:
            if row.error:
                print(f"      {row.name}: {row.error}")

    if not args.no_verify:
        verify(api, args.weaviate.rstrip("/"))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
