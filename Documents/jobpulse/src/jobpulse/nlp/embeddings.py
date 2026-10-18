"""Text embeddings with a graceful fallback.

* ``sentence-transformers`` (if installed and the model can be loaded) gives semantic vectors.
* Otherwise TF-IDF + truncated SVD (LSA) is used, so the whole project runs offline with
  nothing but scikit-learn.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, backend: str = "auto", model_name: str = "all-MiniLM-L6-v2", dims: int = 128):
        if backend not in {"auto", "st", "tfidf"}:
            raise ValueError("backend must be 'auto', 'st' or 'tfidf'")
        self.backend = backend
        self.model_name = model_name
        self.dims = dims
        self._st = None
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self.active_backend: str | None = None

    def _try_load_st(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.model_name)
            return True
        except Exception as exc:  # missing package, no network, etc.
            if self.backend == "st":
                raise
            log.info("sentence-transformers unavailable (%s); using TF-IDF embeddings", exc)
            return False

    def fit(self, texts: list[str]) -> "Embedder":
        texts = [str(t) for t in texts]
        if self.backend in {"auto", "st"} and self._try_load_st():
            self.active_backend = "st"
            return self
        self._tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_features=20000
        )
        x = self._tfidf.fit_transform(texts)
        n_comp = max(2, min(self.dims, x.shape[1] - 1, x.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(x)
        self.active_backend = "tfidf"
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        if self.active_backend is None:
            raise RuntimeError("Call fit() first")
        if self.active_backend == "st":
            if self._st is None:  # restored from pickle
                self._try_load_st()
            vecs = self._st.encode(texts, show_progress_bar=False)
            return normalize(np.asarray(vecs))
        return normalize(self._svd.transform(self._tfidf.transform(texts)))

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_st"] = None  # don't pickle the torch model; reload lazily
        return state
