"""Pandas analytics over the jobs table: skill demand, pay by skill, resume gap."""
from __future__ import annotations

import pandas as pd

from jobpulse.nlp.skills import CATEGORY, extract_skills


def explode_skills(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (job, skill). Expects ``skills`` as a '|' joined string."""
    out = df.assign(skill=df["skills"].fillna("").str.split("|")).explode("skill")
    return out[out["skill"].notna() & (out["skill"] != "")]


def skill_demand(df: pd.DataFrame, by: str | None = None, top: int = 20) -> pd.DataFrame:
    """Share of postings mentioning each skill (optionally split by a column or 'month')."""
    ex = explode_skills(df)
    if by == "month":
        ex = ex.assign(month=pd.to_datetime(ex["posted_date"]).dt.to_period("M").astype(str))
        denom = df.assign(
            month=pd.to_datetime(df["posted_date"]).dt.to_period("M").astype(str)
        ).groupby("month").size()
        counts = ex.groupby(["month", "skill"]).size().rename("postings").reset_index()
        counts["pct"] = counts["postings"] / counts["month"].map(denom) * 100
        return counts

    if by:
        denom = df.groupby(by).size()
        counts = ex.groupby([by, "skill"]).size().rename("postings").reset_index()
        counts["pct"] = counts["postings"] / counts[by].map(denom) * 100
        return counts.sort_values([by, "postings"], ascending=[True, False]).groupby(by).head(top)

    counts = ex.groupby("skill").size().rename("postings").reset_index()
    counts["pct"] = counts["postings"] / len(df) * 100
    counts["category"] = counts["skill"].map(CATEGORY)
    return counts.sort_values("postings", ascending=False).head(top).reset_index(drop=True)


def salary_by_skill(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """Median salary of postings that mention each skill."""
    ex = explode_skills(df.dropna(subset=["salary"]))
    g = ex.groupby("skill")["salary"].agg(median_salary="median", postings="count").reset_index()
    return (
        g[g["postings"] >= min_count]
        .sort_values("median_salary", ascending=False)
        .reset_index(drop=True)
    )


def resume_gap(
    df: pd.DataFrame, resume_text: str, role: str | None = None, top: int = 15
) -> pd.DataFrame:
    """Skills demanded for a role that the resume does not mention, ranked by demand."""
    pool = df
    if role:
        pool = df[df["title"].str.contains(role, case=False, na=False)]
        if pool.empty:
            pool = df
    have = set(extract_skills(resume_text))
    demand = skill_demand(pool, top=200)
    demand["have"] = demand["skill"].isin(have)
    missing = demand[~demand["have"]].head(top).reset_index(drop=True)
    return missing[["skill", "category", "postings", "pct"]]
