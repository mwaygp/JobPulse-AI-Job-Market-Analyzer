# 📈 JobPulse: AI Job Market Analyzer

Collect job postings, extract the skills they ask for, predict salaries, discover role
families, and chat with the data. JobPulse uses classic ML, NLP, and LLMs in one project, and
its output answers a practical question: **which languages and tools does the market actually
ask for?**

| Capability | Technique | Where |
|---|---|---|
| Skill demand tracker | Rule-based NLP (regex taxonomy) + optional spaCy `PhraseMatcher` | `nlp/skills.py` |
| Salary predictor | XGBoost (falls back to scikit-learn GBR) | `ml/salary.py` |
| Role clustering | Embeddings + KMeans + PCA | `ml/clustering.py` |
| Structured extraction | Claude API → JSON, compared against the rule-based extractor | `llm/extract.py` |
| "Ask the job market" | Retrieval-augmented generation (RAG) with Claude | `llm/rag.py` |
| Resume gap analyzer | Skill extraction + demand ranking | `analysis.py` |
| Storage / SQL | SQLite, normalized `job_skills` table, `sql/queries.sql` | `db.py` |
| Serving | FastAPI + Docker | `api/main.py` |
| Dashboard | Streamlit | `app/dashboard.py` |

## ⚠️ About the demo data

`jobpulse init` generates **synthetic** postings so everything runs offline in seconds. The
skill frequencies and salaries are illustrative assumptions, not market facts, and the
salary model scores very high on them (R² ≈ 0.97) only because the synthetic salaries come
from a formula. **To get real findings, load real data** (below) and re-run the same commands.

## Quick start

```bash
git clone <your-repo-url> && cd jobpulse
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .

jobpulse init          # build the demo dataset in SQLite
jobpulse report        # top skills (SQL) + highest-paying skills
jobpulse train         # salary model: MAE vs baseline + feature importance
jobpulse cluster -k 8  # discover role families
jobpulse index         # build the RAG index
jobpulse ask "What do remote NLP roles require most?"

streamlit run app/dashboard.py                  # dashboard
uvicorn jobpulse.api.main:app --reload          # API docs at http://localhost:8000/docs
```

Only the core scikit-learn stack is needed to run the pipeline. XGBoost, spaCy,
sentence-transformers and the Anthropic SDK are optional upgrades, and each feature falls back
gracefully when one is missing.

## Use real data

**Option A: a CSV (e.g. a Kaggle job-postings dataset)**

```bash
jobpulse ingest-csv path/to/postings.csv \
  --mapping '{"title":"Job Title","description":"Job Description","salary":"Salary"}'
```

Columns are auto-detected for common names; `--mapping` overrides them. Seniority and remote
status are inferred from the title and text.

**Option B: the Adzuna API** (free key at https://developer.adzuna.com)

```bash
cp .env.example .env     # add ADZUNA_APP_ID / ADZUNA_APP_KEY
jobpulse ingest-adzuna --query "machine learning engineer" --pages 10
```

Check each data source's terms of use (and `robots.txt` if you scrape) before storing or
redistributing postings. Don't commit raw data; `data/` is git-ignored.

Then re-run `jobpulse report`, `train`, `cluster`, and `index`.

## Turn on the LLM features

```bash
cp .env.example .env     # set ANTHROPIC_API_KEY
```

- `jobpulse ask ...` and the dashboard's **Ask the market** tab return written, cited
  answers instead of the offline summary.
- `POST /extract` returns structured JSON (required vs nice-to-have skills, seniority,
  remote policy) for any posting.
- `llm.extract.compare_extractors(df, n=50)` measures how closely the regex extractor and
  Claude agree. This makes a good write-up for your own results section.

## Your findings (fill these in after loading real data)

> Replace this section with real numbers; it's what recruiters remember.

- Python appears in **__%** of postings; SQL in **__%**; JavaScript in **__%**
- Highest-paying skills: **__**, **__**, **__**
- Role clusters found: **__**
- Salary model: MAE **$__** vs baseline **$__**
- Regex vs LLM extraction agreement (Jaccard): **__**

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status and posting count |
| `GET /skills/top?limit=20&by=seniority` | skill demand (by seniority, location, title, month) |
| `GET /skills/salary` | median salary by skill |
| `POST /skills/extract` | extract skills from any text |
| `POST /salary/predict` | predict salary for a role profile |
| `POST /resume/gap?role=Machine Learning` | missing skills ranked by demand |
| `GET /search?q=...` | semantic search over postings |
| `POST /ask` | RAG answer with sources |
| `POST /extract` | Claude structured extraction |

```bash
docker build -t jobpulse . && docker run -p 8000:8000 --env-file .env jobpulse
```

## Architecture

```
 CSV / Adzuna / synthetic ──► normalize ──► skill extraction ──► SQLite (jobs, job_skills)
                                                                      │
        ┌──────────────┬───────────────┬──────────────┬───────────────┴───┐
        ▼              ▼               ▼              ▼                   ▼
  skill demand    salary model    clustering     RAG index (embeddings)   SQL reports
  (pandas/SQL)    (XGBoost)       (KMeans)       + Claude answers
        └──────────────┴───────────────┴──────────────┴───────────────────┘
                              FastAPI  ◄──►  Streamlit dashboard
```

## Project layout

```
src/jobpulse/
  ingest/      sample_data.py · loaders.py (CSV, Adzuna) · pipeline.py
  nlp/         skills.py (taxonomy + extractors) · embeddings.py
  ml/          salary.py · clustering.py
  llm/         client.py · extract.py · rag.py
  api/         main.py (FastAPI)
  analysis.py  db.py  cli.py  config.py
app/dashboard.py     sql/queries.sql     tests/     .github/workflows/ci.yml
```

## Development

```bash
make test    # pytest
make lint    # ruff
```

Extend the skill taxonomy in `src/jobpulse/nlp/skills.py` (prefix an alias with `!` to match
it case-sensitively, which avoids false positives like "react quickly").

## Ideas to extend it

- Swap the in-memory index for ChromaDB or pgvector
- Fine-tune a small NER model for skill extraction and compare it to the rules and the LLM
- Add a scheduled GitHub Action that ingests new postings weekly and tracks skill trends
- Build a React/TypeScript frontend on top of the API
- Add MLflow experiment tracking for the salary model

## License

MIT, or your choice.
