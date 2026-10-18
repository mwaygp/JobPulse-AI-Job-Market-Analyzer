import tempfile
from pathlib import Path

from jobpulse.llm.extract import parse_json_block
from jobpulse.llm.rag import JobIndex, ask
from jobpulse.ml.clustering import cluster_roles
from jobpulse.ml.salary import predict_salary, train_salary_model


def test_salary_model_beats_baseline(jobs):
    with tempfile.TemporaryDirectory() as tmp:
        bundle = train_salary_model(jobs, Path(tmp) / "m.joblib")
    m = bundle["metrics"]
    assert m["mae"] < m["baseline_mae"] * 0.6
    assert m["r2"] > 0.6
    assert len(bundle["importance"]) > 0


def test_salary_prediction_is_sensible(jobs):
    with tempfile.TemporaryDirectory() as tmp:
        bundle = train_salary_model(jobs, Path(tmp) / "m.joblib")
    senior = predict_salary(bundle, "Machine Learning Engineer", "Senior", "San Francisco, CA", False, ["python", "pytorch"])
    junior = predict_salary(bundle, "Machine Learning Engineer", "Junior", "San Francisco, CA", False, ["python", "pytorch"])
    assert senior > junior


def test_clustering_outputs(jobs):
    out, summary, sil = cluster_roles(jobs, k=5, backend="tfidf")
    assert len(summary) == 5
    assert {"cluster", "x", "y"} <= set(out.columns)
    assert -1 <= sil <= 1


def test_rag_retrieves_relevant_role(jobs):
    idx = JobIndex(backend="tfidf").build(jobs)
    hits = idx.search("frontend React TypeScript CSS web developer", k=5)
    assert sum("Frontend" in h["title"] for h in hits) >= 4


def test_rag_save_load_and_offline_answer(jobs, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    idx = JobIndex(backend="tfidf").build(jobs)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "idx.joblib"
        idx.save(p)
        loaded = JobIndex.load(p)
    out = ask("What do NLP engineers need?", loaded, k=5)
    assert out["mode"] == "offline" and len(out["sources"]) == 5


def test_parse_json_block_handles_fences_and_prose():
    assert parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_block('Sure! Here you go: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}


def test_parse_json_block_rejects_garbage():
    import pytest

    with pytest.raises(ValueError):
        parse_json_block("no json here")
