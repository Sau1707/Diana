.PHONY: api web

API_HOST ?= 127.0.0.1
API_PORT ?= 8000
WEB_HOST ?= 127.0.0.1
WEB_PORT ?= 5173

api:
	uv run --project backend uvicorn backend.src.main:app --reload --host $(API_HOST) --port $(API_PORT)

web:
	npm run dev --prefix frontend -- --host $(WEB_HOST) --port $(WEB_PORT)
