PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
STREAMLIT ?= streamlit
COMPOSE ?= docker compose
PORT ?= 8501

.DEFAULT_GOAL := help

.PHONY: help setup ingest run test eval eval-golden docker-build compose-up compose-up-ollama compose-down clean-state

help:
	@printf "Available targets:\n"
	@printf "  make setup             Install Python dependencies\n"
	@printf "  make ingest            Rebuild the local Chroma index\n"
	@printf "  make run               Run the Streamlit app locally\n"
	@printf "  make test              Run unit tests\n"
	@printf "  make eval              Run representative query evaluation\n"
	@printf "  make eval-golden       Run golden query evaluation\n"
	@printf "  make docker-build      Build the app image\n"
	@printf "  make compose-up        Run app container against host Ollama\n"
	@printf "  make compose-up-ollama Run app plus optional Ollama container profile\n"
	@printf "  make compose-down      Stop Compose services\n"
	@printf "  make clean-state       Remove local chat history\n"

setup:
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) scripts/ingest.py

run:
	$(STREAMLIT) run app/streamlit_app.py

test:
	$(PYTHON) -m pytest

eval:
	$(PYTHON) scripts/run_eval.py

eval-golden:
	$(PYTHON) scripts/run_eval.py --golden eval/golden_queries.json

docker-build:
	docker build -t family-office-rag:local .

compose-up:
	STREAMLIT_PORT=$(PORT) $(COMPOSE) up --build app

compose-up-ollama:
	STREAMLIT_PORT=$(PORT) COMPOSE_OLLAMA_BASE_URL=http://ollama:11434 $(COMPOSE) --profile local-ollama up --build

compose-down:
	$(COMPOSE) down

clean-state:
	$(PYTHON) -c "from pathlib import Path; Path('state/chat_history.sqlite3').unlink(missing_ok=True)"
