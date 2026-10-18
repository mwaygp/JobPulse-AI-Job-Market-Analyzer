import tempfile
from pathlib import Path

from jobpulse.analysis import resume_gap, salary_by_skill, skill_demand
from jobpulse.db import load_jobs, save_jobs, top_skills_sql
from jobpulse.ingest.loaders import infer_remote, infer_seniority, normalize
from jobpulse.ingest.sample_data import generate_jobs


def test_generator_is_deterministic_and_schema_complete():
    a, b = generate_jobs(50, seed=3), generate_jobs(50, seed=3)
    assert a.drop(columns="posted_date").equals(b.drop(columns="posted_date"))
    assert {"id", "title", "salary", "description", "remote", "seniority"} <= set(a.columns)


def test_skills_extracted_from_descriptions(jobs):
    assert (jobs["skills"] != "").mean() > 0.95


def test_skill_demand_shape_and_order(jobs):
    d = skill_demand(jobs, top=10)
    assert len(d) == 10
    assert d["postings"].is_monotonic_decreasing
    assert d["pct"].between(0, 100).all()


def test_skill_demand_by_group(jobs):
    d = skill_demand(jobs, by="seniority", top=5)
    assert set(d["seniority"]) == set(jobs["seniority"])


def test_salary_by_skill(jobs):
    s = salary_by_skill(jobs, min_count=5)
    assert s["median_salary"].is_monotonic_decreasing


def test_resume_gap_excludes_known_skills(jobs):
    gap = resume_gap(jobs, "Experienced with Python and SQL", role="Machine Learning")
    assert not {"python", "sql"} & set(gap["skill"])
    assert len(gap) > 0


def test_db_roundtrip_and_sql(jobs):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.db"
        save_jobs(jobs, path)
        loaded = load_jobs(path)
        assert len(loaded) == len(jobs)
        top = top_skills_sql(5, path)
        assert len(top) == 5 and top["postings"].iloc[0] >= top["postings"].iloc[-1]


def test_loader_helpers():
    assert infer_seniority("Senior Data Scientist") == "Senior"
    assert infer_seniority("Junior Analyst") == "Junior"
    assert infer_seniority("Data Scientist") == "Mid-level"
    assert infer_remote("Fully remote role") is True
    assert infer_remote("On-site in Austin") is False


def test_normalize_arbitrary_csv_columns():
    import pandas as pd

    raw = pd.DataFrame({
        "Job Title": ["Senior ML Engineer"],
        "Job Description": ["Build models in Python and PyTorch. Remote friendly team."],
        "salary_min": [100000], "salary_max": [140000],
    })
    out = normalize(raw)
    assert out.loc[0, "seniority"] == "Senior"
    assert out.loc[0, "salary"] == 120000
    assert bool(out.loc[0, "remote"]) is True
