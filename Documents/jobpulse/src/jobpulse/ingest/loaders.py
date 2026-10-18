"""Loaders for REAL job data: a CSV (e.g. a Kaggle dataset) or the Adzuna API."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import requests

from jobpulse.config import SCHEMA

_SENIORITY_RULES = [
    (r"\b(staff|principal|lead|head)\b", "Staff"),
    (r"\b(senior|sr\.?)\b", "Senior"),
    (r"\b(junior|jr\.?|entry|intern|graduate|associate)\b", "Junior"),
]

# Candidate column names for common public datasets, tried in order.
_GUESSES = {
    "title": ["title", "job_title", "Job Title", "position"],
    "company": ["company", "company_name", "Company", "employer"],
    "location": ["location", "Location", "job_location", "city"],
    "description": ["description", "job_description", "Job Description", "text"],
    "salary": ["salary", "salary_avg", "Salary", "avg_salary"],
    "posted_date": ["posted_date", "date_posted", "date", "created", "Date Posted"],
}


def infer_seniority(title: str) -> str:
    t = (title or "").lower()
    for pattern, label in _SENIORITY_RULES:
        if re.search(pattern, t):
            return label
    return "Mid-level"


def infer_remote(*texts: str) -> bool:
    return any(re.search(r"\bremote\b|work from home|\bwfh\b", t or "", re.I) for t in texts)


def _pick(df: pd.DataFrame, field: str, mapping: dict[str, str] | None):
    if mapping and field in mapping:
        return df[mapping[field]]
    for col in _GUESSES[field]:
        if col in df.columns:
            return df[col]
    return pd.Series([None] * len(df), index=df.index)


def normalize(df: pd.DataFrame, mapping: dict[str, str] | None = None, source: str = "csv") -> pd.DataFrame:
    """Coerce an arbitrary jobs DataFrame into the JobPulse schema."""
    out = pd.DataFrame({f: _pick(df, f, mapping) for f in _GUESSES})

    # Salary: use an explicit column, else average salary_min/salary_max if present.
    if out["salary"].isna().all() and {"salary_min", "salary_max"} <= set(df.columns):
        out["salary"] = (df["salary_min"] + df["salary_max"]) / 2
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")

    out["posted_date"] = pd.to_datetime(out["posted_date"], errors="coerce", utc=True).dt.date
    out["title"] = out["title"].fillna("").astype(str)
    out["description"] = out["description"].fillna("").astype(str)
    out["company"] = out["company"].fillna("Unknown").astype(str)
    out["location"] = out["location"].fillna("Unknown").astype(str)
    out["seniority"] = out["title"].map(infer_seniority)
    out["remote"] = [
        infer_remote(t, loc, d[:500]) for t, loc, d in zip(out["title"], out["location"], out["description"])
    ]
    out["id"] = [f"{source}-{i:06d}" for i in range(len(out))]
    out = out[out["description"].str.len() > 20].reset_index(drop=True)
    return out[SCHEMA]


def load_csv(path: str | Path, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    return normalize(pd.read_csv(path), mapping, source="csv")


def fetch_adzuna(
    what: str = "data scientist",
    country: str = "us",
    pages: int = 5,
    per_page: int = 50,
    app_id: str | None = None,
    app_key: str | None = None,
) -> pd.DataFrame:
    """Pull postings from the Adzuna API (free key: https://developer.adzuna.com).

    Check Adzuna's terms of use for storage/redistribution before publishing the data.
    """
    app_id = app_id or os.getenv("ADZUNA_APP_ID")
    app_key = app_key or os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise RuntimeError("Set ADZUNA_APP_ID and ADZUNA_APP_KEY (see .env.example)")

    rows = []
    for page in range(1, pages + 1):
        resp = requests.get(
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}",
            params={
                "app_id": app_id, "app_key": app_key, "what": what,
                "results_per_page": per_page, "content-type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        for r in results:
            rows.append({
                "title": r.get("title"),
                "company": (r.get("company") or {}).get("display_name"),
                "location": (r.get("location") or {}).get("display_name"),
                "description": r.get("description"),
                "salary_min": r.get("salary_min"),
                "salary_max": r.get("salary_max"),
                "posted_date": r.get("created"),
            })
    if not rows:
        raise RuntimeError("Adzuna returned no results")
    return normalize(pd.DataFrame(rows), source="adzuna")
