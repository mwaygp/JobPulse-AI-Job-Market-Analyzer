"""Salary prediction. Gradient-boosted trees on skills + title + seniority + location.

Uses XGBoost if installed, else scikit-learn's GradientBoostingRegressor.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from jobpulse.config import RANDOM_STATE, SALARY_MODEL_PATH

FEATURES = ["title", "seniority", "location", "remote", "skills"]


def _split_skills(s: str) -> list[str]:
    return [t for t in str(s).split("|") if t]


def _make_regressor():
    try:
        from xgboost import XGBRegressor

        return "xgboost", XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1,
        )
    except Exception:  # not installed, or missing native libs (e.g. libomp on macOS)
        from sklearn.ensemble import GradientBoostingRegressor

        return "sklearn-gbr", GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
            random_state=RANDOM_STATE,
        )


def _build_pipeline() -> tuple[str, Pipeline]:
    name, reg = _make_regressor()
    pre = ColumnTransformer([
        ("skills", CountVectorizer(
            tokenizer=_split_skills, token_pattern=None, lowercase=False, binary=True), "skills"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["title", "seniority", "location"]),
        ("num", "passthrough", ["remote"]),
    ])
    return name, Pipeline([("pre", pre), ("model", reg)])


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()
    x["remote"] = x["remote"].astype(int)
    x["skills"] = x["skills"].fillna("")
    return x


def feature_importance(pipe: Pipeline, top: int = 20) -> pd.DataFrame:
    names = pipe.named_steps["pre"].get_feature_names_out()
    names = [n.split("__", 1)[1] if "__" in n else n for n in names]
    imp = pipe.named_steps["model"].feature_importances_
    out = pd.DataFrame({"feature": names, "importance": imp})
    return out.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


def train_salary_model(df: pd.DataFrame, path: Path | str = SALARY_MODEL_PATH) -> dict:
    """Train, evaluate on a hold-out split, save, and return a bundle with metrics."""
    data = df.dropna(subset=["salary"])
    if len(data) < 50:
        raise ValueError(f"Need at least 50 postings with salary to train (got {len(data)})")

    x_train, x_test, y_train, y_test = train_test_split(
        _prepare(data), data["salary"].astype(float), test_size=0.2, random_state=RANDOM_STATE
    )
    backend, pipe = _build_pipeline()
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)

    baseline_mae = float(mean_absolute_error(y_test, np.full(len(y_test), y_train.median())))
    metrics = {
        "backend": backend,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "baseline_mae": baseline_mae,
        "r2": float(r2_score(y_test, pred)),
    }
    bundle = {"pipeline": pipe, "metrics": metrics, "importance": feature_importance(pipe)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_salary_model(path: Path | str = SALARY_MODEL_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `jobpulse train` first.")
    return joblib.load(path)


def predict_salary(
    bundle: dict, title: str, seniority: str, location: str, remote: bool, skills: list[str]
) -> float:
    row = pd.DataFrame([{
        "title": title, "seniority": seniority, "location": location,
        "remote": int(remote), "skills": "|".join(skills),
    }])
    return float(bundle["pipeline"].predict(row[FEATURES])[0])
