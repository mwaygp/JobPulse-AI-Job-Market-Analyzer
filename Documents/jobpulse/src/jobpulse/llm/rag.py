"""'Ask the job market': retrieval-augmented Q&A over job postings.

Retrieval is a simple in-memory cosine search (fine up to ~100k postings). To scale, swap
``JobIndex`` for ChromaDB or pgvector; the interface (build / search) stays the same.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from jobpulse.config import INDEX_PATH
from jobpulse.llm.client import complete, get_client
from jobpulse.nlp.embeddings import Embedder

QA_SYSTEM = (
    "You are a job-market analyst. Answer the question using ONLY the job postings provided. "
    "Cite postings by their [id]. If the postings don't contain the answer, say so. Be concise."
)


class JobIndex:
    def __init__(self, backend: str = "auto") -> None:
        self.embedder = Embedder(backend=backend)
        self.matrix: np.ndarray | None = None
        self.records: list[dict] = []

    @staticmethod
    def _doc(row) -> str:
        skills = ", ".join(str(row["skills"]).split("|")) if row["skills"] else ""
        return f"{row['title']} at {row['company']} ({row['location']}). Skills: {skills}. {row['description']}"

    def build(self, df: pd.DataFrame) -> "JobIndex":
        docs = [self._doc(r) for _, r in df.iterrows()]
        self.matrix = self.embedder.fit_transform(docs)
        self.records = [
            {
                "id": r["id"], "title": r["title"], "company": r["company"],
                "location": r["location"], "salary": None if pd.isna(r["salary"]) else float(r["salary"]),
                "skills": r["skills"], "description": r["description"],
            }
            for _, r in df.iterrows()
        ]
        return self

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.matrix is None:
            raise RuntimeError("Index not built")
        q = self.embedder.transform([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [{**self.records[i], "score": float(scores[i])} for i in top]

    def save(self, path: Path | str = INDEX_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str = INDEX_PATH) -> "JobIndex":
        if not Path(path).exists():
            raise FileNotFoundError(f"No index at {path}. Run `jobpulse index` first.")
        return joblib.load(path)


def ask(question: str, index: JobIndex, k: int = 8, client=None) -> dict:
    """Answer a question about the market. Uses Claude when available, else an offline summary."""
    hits = index.search(question, k=k)
    client = client or get_client()

    if client is not None:
        context = "\n\n".join(
            f"[{h['id']}] {h['title']} at {h['company']} ({h['location']}). "
            f"Skills: {h['skills'].replace('|', ', ')}. {h['description'][:600]}"
            for h in hits
        )
        answer = complete(client, QA_SYSTEM, f"Postings:\n{context}\n\nQuestion: {question}", 700)
        mode = "llm"
    else:
        counts = Counter(s for h in hits for s in h["skills"].split("|") if s)
        top = ", ".join(f"{s} ({n}/{len(hits)})" for s, n in counts.most_common(8))
        answer = (
            f"(offline mode: set ANTHROPIC_API_KEY for a written answer)\n"
            f"Across the {len(hits)} most relevant postings, the most common skills are: {top}."
        )
        mode = "offline"
    return {"answer": answer, "mode": mode, "sources": [
        {k_: h[k_] for k_ in ("id", "title", "company", "location", "score")} for h in hits
    ]}
