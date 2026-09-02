#!/bin/sh
# Container entrypoint: bring the schema up to date, then run the app.
#
# `set -e` matters here. Without it, a failed migration would be logged
# and then the app would start anyway against a stale schema — which is
# exactly the class of failure Alembic was introduced to prevent.
set -e

echo "[entrypoint] Applying database migrations (alembic upgrade head)..."
alembic upgrade head
echo "[entrypoint] Migrations up to date."

# `exec` so the app becomes PID 1 and receives SIGTERM directly from
# `docker stop` / `docker compose down`. Without it, this shell stays PID 1
# and swallows the signal, so shutdown waits for the 10s kill timeout
# instead of shutting down cleanly.
echo "[entrypoint] Starting: $*"
exec "$@"
