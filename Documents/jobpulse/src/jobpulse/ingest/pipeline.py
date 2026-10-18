"""Glue: raw postings -> skills extracted -> SQLite."""
from __future__ import annotations

import pandas as pd

from jobpulse.config import DB_PATH
from jobpulse.db import db_has_data, save_jobs
from jobpulse.ingest.sample_data import generate_jobs
from jobpulse.nlp.skills import extract_skills


def add_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``skills`` column ('|' joined canonical skills) from each description."""
    out = df.copy()
    out["skills"] = out["description"].map(lambda d: "|".join(extract_skills(d)))
    return out


def ingest(df: pd.DataFrame, path=DB_PATH) -> pd.DataFrame:
    processed = add_skills(df)
    save_jobs(processed, path)
    return processed


def build_sample_dataset(n: int = 1500, seed: int = 42, path=DB_PATH) -> pd.DataFrame:
    return ingest(generate_jobs(n=n, seed=seed), path)


def ensure_data(path=DB_PATH) -> None:
    """Create the synthetic demo dataset if the database is empty."""
    if not db_has_data(path):
        build_sample_dataset(path=path)
