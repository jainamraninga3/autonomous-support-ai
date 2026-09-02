"""Deprecated — schema changes go through Alembic now.

This used to create tables directly from the ORM models via
`Base.metadata.create_all`. That is no longer safe to use alongside
Alembic: two independent sources of schema truth is how a database ends
up in a state neither of them expects.

It also had a failure mode worth remembering, since it is the reason
Alembic exists here: `create_all` only creates MISSING tables. When a
model's columns changed, it silently skipped that table and still
reported success, leaving the schema stale — and the failure surfaced
much later as `column ... does not exist` on an insert.

What to run instead, from `backend/`:

    alembic upgrade head          # apply all migrations (the container
                                  # does this automatically on startup)
    alembic revision --autogenerate -m "what changed"
    alembic downgrade -1          # undo the last migration
    alembic current               # which revision this database is on
    alembic check                 # do the models match the database?

For an existing database that already matches the models but has no
Alembic version table (i.e. it was built by the old `create_all` path):

    alembic stamp head            # record it as migrated, run nothing
"""

import sys

MESSAGE = __doc__


if __name__ == "__main__":
    print(MESSAGE, file=sys.stderr)
    raise SystemExit(1)
