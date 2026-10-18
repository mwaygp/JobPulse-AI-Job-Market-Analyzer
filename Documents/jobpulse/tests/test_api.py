import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import importlib

    import jobpulse.config as cfg

    tmp = tmp_path_factory.mktemp("jp")
    # Point every path at a temp dir *before* importing the app modules.
    cfg.DATA_DIR = tmp
    cfg.DB_PATH = tmp / "t.db"
    cfg.SALARY_MODEL_PATH = tmp / "salary.joblib"
    cfg.INDEX_PATH = tmp / "index.joblib"
    for mod in ("jobpulse.db", "jobpulse.ingest.pipeline", "jobpulse.ml.salary",
                "jobpulse.llm.rag", "jobpulse.api.main"):
        importlib.import_module(mod)
        importlib.reload(importlib.import_module(mod))

    from fastapi.testclient import TestClient

    from jobpulse.api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["postings"] > 0


def test_top_skills_and_validation(client):
    assert len(client.get("/skills/top?limit=5").json()) == 5
    assert client.get("/skills/top?by=nonsense").status_code == 400


def test_salary_predict(client):
    r = client.post("/salary/predict", json={"skills": ["python", "pytorch"]})
    assert r.status_code == 200 and r.json()["predicted_salary"] > 30_000


def test_extract_without_key_is_503(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.post("/extract", json={"text": "Python dev"}).status_code == 503
