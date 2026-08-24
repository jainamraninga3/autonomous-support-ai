.PHONY: install dev test run

install:
	cd backend && .venv/Scripts/pip install -r requirements/dev.txt

run:
	cd backend && .venv/Scripts/uvicorn app.main:app --reload

test:
	cd backend && .venv/Scripts/pytest
