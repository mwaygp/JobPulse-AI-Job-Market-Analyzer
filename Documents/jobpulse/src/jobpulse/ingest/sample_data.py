"""Synthetic job postings so the whole project runs offline.

IMPORTANT: this data is generated, not real. The skill frequencies and salaries below are
illustrative assumptions used to exercise the pipeline. For real market findings, load a
real dataset with ``jobpulse ingest-csv`` or ``jobpulse ingest-adzuna`` and re-run.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# role -> (base salary, [(skill text as written in postings, probability)])
ROLES: dict[str, tuple[int, list[tuple[str, float]]]] = {
    "Machine Learning Engineer": (140_000, [
        ("Python", 0.97), ("PyTorch", 0.7), ("TensorFlow", 0.35), ("scikit-learn", 0.6),
        ("SQL", 0.5), ("Docker", 0.6), ("AWS", 0.55), ("Kubernetes", 0.3), ("MLflow", 0.35),
        ("Git", 0.6), ("Hugging Face", 0.35), ("LLMs", 0.35), ("C++", 0.12)]),
    "NLP Engineer": (145_000, [
        ("Python", 0.97), ("PyTorch", 0.7), ("Hugging Face", 0.7), ("spaCy", 0.35),
        ("LLMs", 0.7), ("RAG", 0.4), ("LangChain", 0.3), ("SQL", 0.35), ("Docker", 0.45),
        ("AWS", 0.4), ("prompt engineering", 0.35), ("Git", 0.5)]),
    "Data Scientist": (125_000, [
        ("Python", 0.92), ("SQL", 0.9), ("pandas", 0.7), ("scikit-learn", 0.65),
        ("R programming", 0.2), ("Tableau", 0.25), ("Spark", 0.2), ("AWS", 0.3),
        ("Git", 0.5), ("XGBoost", 0.3), ("NumPy", 0.5)]),
    "Data Analyst": (92_000, [
        ("SQL", 0.95), ("Microsoft Excel", 0.6), ("Tableau", 0.5), ("Power BI", 0.45),
        ("Python", 0.45), ("pandas", 0.25), ("R programming", 0.1), ("Snowflake", 0.2)]),
    "Data Engineer": (130_000, [
        ("Python", 0.9), ("SQL", 0.95), ("Spark", 0.55), ("Airflow", 0.5), ("Kafka", 0.35),
        ("AWS", 0.6), ("Snowflake", 0.35), ("dbt", 0.3), ("Docker", 0.4), ("Scala", 0.15),
        ("BigQuery", 0.2), ("Terraform", 0.2)]),
    "Backend Engineer": (128_000, [
        ("Python", 0.35), ("Java", 0.4), ("Go", 0.25), ("Node.js", 0.2), ("C#", 0.12),
        ("PostgreSQL", 0.5), ("Docker", 0.6), ("Kubernetes", 0.4), ("AWS", 0.55),
        ("REST APIs", 0.7), ("Redis", 0.3), ("Kafka", 0.2), ("Git", 0.6), ("Spring Boot", 0.2),
        ("FastAPI", 0.15), ("Django", 0.12)]),
    "Frontend Engineer": (118_000, [
        ("JavaScript", 0.95), ("TypeScript", 0.75), ("React", 0.8), ("HTML5", 0.7),
        ("CSS3", 0.7), ("GraphQL", 0.25), ("Node.js", 0.3), ("Git", 0.55)]),
    "DevOps Engineer": (130_000, [
        ("Kubernetes", 0.7), ("Docker", 0.85), ("Terraform", 0.6), ("AWS", 0.7),
        ("Azure", 0.25), ("GCP", 0.2), ("Linux", 0.7), ("Python", 0.5), ("Bash", 0.4),
        ("CI/CD", 0.7), ("Git", 0.5)]),
}

# Illustrative pay premiums (USD) for in-demand skills.
PREMIUM = {
    "PyTorch": 8_000, "LLMs": 10_000, "RAG": 6_000, "Kubernetes": 5_000, "Spark": 5_000,
    "Go": 5_000, "Terraform": 4_000, "Kafka": 4_000, "Hugging Face": 4_000, "Scala": 4_000,
    "Microsoft Excel": -6_000, "HTML5": -4_000, "CSS3": -3_000,
}

SENIORITY = {"Junior": 0.78, "Mid-level": 1.0, "Senior": 1.25, "Staff": 1.5}
SENIORITY_P = [0.2, 0.35, 0.33, 0.12]

LOCATIONS = {
    "San Francisco, CA": 1.18, "New York, NY": 1.12, "Seattle, WA": 1.1, "Austin, TX": 1.0,
    "Phoenix, AZ": 0.93, "Chicago, IL": 1.0, "Denver, CO": 0.98, "Atlanta, GA": 0.95,
}

COMPANIES = [
    "Nimbus Labs", "Arcadia Systems", "Brightpath", "Kestrel Analytics", "Northwind Digital",
    "Helios Health", "Quanta Retail", "Lumen Finance", "Orbital Software", "Verdant Energy",
    "Pinecrest AI", "Atlas Logistics", "Cobalt Security", "Meridian Media", "Zephyr Cloud",
]


def generate_jobs(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    titles = list(ROLES)
    today = date.today()
    rows = []
    for i in range(n):
        title = titles[rng.integers(len(titles))]
        base, skill_probs = ROLES[title]
        seniority = rng.choice(list(SENIORITY), p=SENIORITY_P)
        location = rng.choice(list(LOCATIONS))
        remote = bool(rng.random() < 0.35)

        skills = [s for s, p in skill_probs if rng.random() < p]
        if not skills:
            skills = [skill_probs[0][0]]
        nice = skills[len(skills) * 2 // 3:]
        core = skills[: max(1, len(skills) * 2 // 3)]

        salary = base * SENIORITY[seniority] * LOCATIONS[location]
        salary *= 0.96 if remote else 1.0
        salary += sum(PREMIUM.get(s, 0) for s in skills)
        salary += rng.normal(0, 7_000)
        salary = round(max(salary, 45_000), -2)

        company = COMPANIES[rng.integers(len(COMPANIES))]
        yrs = {"Junior": "0-2", "Mid-level": "3-5", "Senior": "5+", "Staff": "8+"}[seniority]
        where = "Remote (US)" if remote else location
        description = (
            f"{company} is hiring a {seniority} {title} ({where}). You will design, build and "
            f"ship production systems with {yrs} years of experience. "
            f"Requirements: hands-on experience with {', '.join(core)}. "
            + (f"Nice to have: {', '.join(nice)}. " if nice else "")
            + "We value clear communication, ownership and a collaborative mindset."
        )

        rows.append({
            "id": f"sample-{i:05d}",
            "title": title,
            "company": company,
            "location": location,
            "remote": remote,
            "seniority": seniority,
            "salary": salary,
            "posted_date": today - timedelta(days=int(rng.integers(0, 365))),
            "description": description,
        })
    return pd.DataFrame(rows)
