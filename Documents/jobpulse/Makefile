.PHONY: install init train index test lint api app

install:
	pip install -r requirements-dev.txt && pip install -e .

init:       ## build the demo dataset (synthetic) and load it into SQLite
	jobpulse init

train:      ## train the salary model
	jobpulse train

index:      ## build the RAG index
	jobpulse index

test:
	pytest -q

lint:
	ruff check src tests app

api:
	uvicorn jobpulse.api.main:app --reload

app:
	streamlit run app/dashboard.py
