"""FastAPI service.  Run:  uvicorn jobpulse.api.main:app --reload   (docs at /docs)"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jobpulse import __version__
from jobpulse.analysis import resume_gap, salary_by_skill, skill_demand
from jobpulse.config import INDEX_PATH, SALARY_MODEL_PATH
from jobpulse.db import load_jobs
from jobpulse.ingest.pipeline import ensure_data
from jobpulse.llm.extract import extract_structured
from jobpulse.llm.rag import JobIndex, ask
from jobpulse.ml.salary import load_salary_model, predict_salary, train_salary_model
from jobpulse.nlp.skills import extract_skills

app = FastAPI(title="JobPulse API", version=__version__)


@lru_cache(maxsize=1)
def _jobs():
    ensure_data()
    return load_jobs()


@lru_cache(maxsize=1)
def _salary_bundle():
    if not SALARY_MODEL_PATH.exists():
        return train_salary_model(_jobs())
    return load_salary_model()


@lru_cache(maxsize=1)
def _index() -> JobIndex:
    if INDEX_PATH.exists():
        return JobIndex.load()
    idx = JobIndex().build(_jobs())
    idx.save()
    return idx


class SalaryRequest(BaseModel):
    title: str = "Machine Learning Engineer"
    seniority: str = "Mid-level"
    location: str = "Austin, TX"
    remote: bool = False
    skills: list[str] = Field(default_factory=lambda: ["python", "pytorch", "sql"])


class AskRequest(BaseModel):
    question: str
    k: int = Field(8, ge=1, le=25)


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "postings": len(_jobs())}


@app.get("/skills/top")
def top_skills(limit: int = 20, by: str | None = None):
    """Most requested skills, optionally split by seniority / location / month."""
    if by not in (None, "seniority", "location", "title", "month"):
        raise HTTPException(400, "by must be one of seniority, location, title, month")
    return skill_demand(_jobs(), by=by, top=limit).to_dict("records")


@app.get("/skills/salary")
def skills_salary(min_count: int = 10):
    """Median salary of postings mentioning each skill."""
    return salary_by_skill(_jobs(), min_count=min_count).to_dict("records")


@app.post("/skills/extract")
def skills_extract(req: TextRequest):
    """Rule-based skill extraction from any text (e.g. a resume)."""
    return {"skills": extract_skills(req.text)}


@app.post("/salary/predict")
def salary_predict(req: SalaryRequest):
    bundle = _salary_bundle()
    value = predict_salary(bundle, req.title, req.seniority, req.location, req.remote, req.skills)
    return {"predicted_salary": round(value, -2), "model_mae": bundle["metrics"]["mae"]}


@app.post("/resume/gap")
def resume_gap_endpoint(req: TextRequest, role: str | None = None):
    return resume_gap(_jobs(), req.text, role=role).to_dict("records")


@app.get("/search")
def search(q: str, k: int = 5):
    return _index().search(q, k=k)


@app.post("/ask")
def ask_market(req: AskRequest):
    return ask(req.question, _index(), k=req.k)


@app.post("/extract")
def llm_extract(req: TextRequest):
    """Structured extraction with Claude (requires ANTHROPIC_API_KEY)."""
    try:
        return extract_structured(req.text)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
