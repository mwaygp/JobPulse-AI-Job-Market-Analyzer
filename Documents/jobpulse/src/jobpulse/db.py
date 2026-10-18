"""SQLite storage. Two tables: ``jobs`` and a normalized ``job_skills`` (job_id, skill)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from jobpulse.config import DB_PATH


def _connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def save_jobs(df: pd.DataFrame, path: Path | str = DB_PATH) -> None:
    """Persist jobs (with a '|' joined ``skills`` column) plus the job_skills link table."""
    links = (
        df[["id", "skills"]]
        .assign(skill=df["skills"].fillna("").str.split("|"))
        .explode("skill")
        .query("skill.notna() and skill != ''")
        .rename(columns={"id": "job_id"})[["job_id", "skill"]]
    )
    to_store = df.assign(
        remote=df["remote"].astype(int), posted_date=df["posted_date"].astype(str)
    )
    with _connect(path) as con:
        to_store.to_sql("jobs", con, if_exists="replace", index=False)
        links.to_sql("job_skills", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_skill ON job_skills(skill)")


def load_jobs(path: Path | str = DB_PATH) -> pd.DataFrame:
    with _connect(path) as con:
        df = pd.read_sql("SELECT * FROM jobs", con)
    df["remote"] = df["remote"].astype(bool)
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    return df


def db_has_data(path: Path | str = DB_PATH) -> bool:
    if not Path(path).exists():
        return False
    try:
        with _connect(path) as con:
            return con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] > 0
    except sqlite3.Error:
        return False


def run_sql(sql: str, params: tuple = (), path: Path | str = DB_PATH) -> pd.DataFrame:
    with _connect(path) as con:
        return pd.read_sql(sql, con, params=params)


def top_skills_sql(limit: int = 15, path: Path | str = DB_PATH) -> pd.DataFrame:
    """Most requested skills, computed in plain SQL."""
    return run_sql(
        """
        SELECT skill,
               COUNT(*) AS postings,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM jobs), 1) AS pct
        FROM job_skills
        GROUP BY skill
        ORDER BY postings DESC
        LIMIT ?
        """,
        (limit,),
        path,
    )
