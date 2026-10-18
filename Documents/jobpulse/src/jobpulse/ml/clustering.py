"""Cluster job descriptions to discover which role 'families' actually exist."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from jobpulse.analysis import explode_skills
from jobpulse.config import RANDOM_STATE
from jobpulse.nlp.embeddings import Embedder


def cluster_roles(
    df: pd.DataFrame, k: int = 6, backend: str = "auto"
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return (jobs with cluster + 2D coords, per-cluster summary, silhouette score)."""
    emb = Embedder(backend=backend).fit_transform(df["description"].tolist())
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(emb)
    sil = float(silhouette_score(emb, labels)) if k > 1 else 0.0

    xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(emb)
    out = df.assign(cluster=labels, x=xy[:, 0], y=xy[:, 1])

    rows = []
    for c, grp in out.groupby("cluster"):
        titles = grp["title"].value_counts()
        skills = explode_skills(grp)["skill"].value_counts()
        rows.append({
            "cluster": c,
            "size": len(grp),
            "top_titles": ", ".join(f"{t} ({n})" for t, n in titles.head(3).items()),
            "top_skills": ", ".join(skills.head(6).index),
            "median_salary": grp["salary"].median(),
        })
    return out, pd.DataFrame(rows), sil
