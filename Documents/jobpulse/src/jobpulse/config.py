"""Central configuration. Paths, model names, and environment variables."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("JOBPULSE_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("JOBPULSE_MODEL_DIR", ROOT / "models"))
DB_PATH = DATA_DIR / "jobpulse.db"
SALARY_MODEL_PATH = MODEL_DIR / "salary_model.joblib"
INDEX_PATH = MODEL_DIR / "job_index.joblib"

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
RANDOM_STATE = 42

# Canonical schema every job posting is normalized into.
SCHEMA = [
    "id", "title", "company", "location", "remote",
    "seniority", "salary", "posted_date", "description",
]
