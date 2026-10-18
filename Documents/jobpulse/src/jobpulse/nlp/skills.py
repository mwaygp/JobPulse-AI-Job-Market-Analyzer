"""Skill taxonomy and rule-based skill extraction.

Aliases are matched case-insensitively with boundaries that understand tokens like
``c++``, ``c#`` and ``node.js``. An alias prefixed with ``!`` is matched case-SENSITIVELY,
which avoids false positives for words that are also normal English ("React quickly",
"Spark joy", "Spring").
"""
from __future__ import annotations

import re
from functools import lru_cache

# canonical name -> (category, [aliases])
TAXONOMY: dict[str, tuple[str, list[str]]] = {
    # Languages
    "python": ("language", ["python", "python3", "pyspark"]),
    "java": ("language", ["java"]),
    "javascript": ("language", ["javascript", "ecmascript"]),
    "typescript": ("language", ["typescript"]),
    "c++": ("language", ["c++"]),
    "c#": ("language", ["c#"]),
    "go": ("language", ["golang"]),
    "rust": ("language", ["!Rust"]),
    "sql": ("language", ["sql", "t-sql", "pl/sql"]),
    "r": ("language", ["r programming", "rstudio", "r language"]),
    "scala": ("language", ["scala"]),
    "kotlin": ("language", ["kotlin"]),
    "swift": ("language", ["!Swift"]),
    "php": ("language", ["php"]),
    "ruby": ("language", ["ruby"]),
    "bash": ("language", ["bash", "shell scripting"]),
    # ML / AI
    "scikit-learn": ("ml", ["scikit-learn", "sklearn", "scikit learn"]),
    "pytorch": ("ml", ["pytorch"]),
    "tensorflow": ("ml", ["tensorflow"]),
    "keras": ("ml", ["keras"]),
    "xgboost": ("ml", ["xgboost", "lightgbm"]),
    "pandas": ("ml", ["pandas"]),
    "numpy": ("ml", ["numpy"]),
    "spark": ("data", ["apache spark", "!Spark", "pyspark"]),
    "hugging face": ("nlp", ["hugging face", "huggingface"]),
    "spacy": ("nlp", ["spacy"]),
    "nltk": ("nlp", ["nltk"]),
    "langchain": ("llm", ["langchain", "langgraph"]),
    "llamaindex": ("llm", ["llamaindex", "llama index"]),
    "llm": ("llm", ["llm", "llms", "large language model", "large language models", "genai"]),
    "rag": ("llm", ["!RAG", "retrieval-augmented generation", "retrieval augmented generation"]),
    "prompt engineering": ("llm", ["prompt engineering", "prompt design"]),
    "mlflow": ("mlops", ["mlflow", "weights & biases", "wandb"]),
    # Data
    "postgresql": ("database", ["postgresql", "postgres"]),
    "mysql": ("database", ["mysql"]),
    "mongodb": ("database", ["mongodb"]),
    "redis": ("database", ["redis"]),
    "snowflake": ("database", ["snowflake"]),
    "bigquery": ("database", ["bigquery"]),
    "airflow": ("data", ["airflow"]),
    "kafka": ("data", ["kafka"]),
    "dbt": ("data", ["dbt"]),
    "tableau": ("bi", ["tableau"]),
    "power bi": ("bi", ["power bi", "powerbi"]),
    "excel": ("bi", ["microsoft excel", "ms excel", "advanced excel"]),
    # Cloud / DevOps
    "aws": ("cloud", ["aws", "amazon web services"]),
    "azure": ("cloud", ["azure"]),
    "gcp": ("cloud", ["gcp", "google cloud"]),
    "docker": ("devops", ["docker"]),
    "kubernetes": ("devops", ["kubernetes", "k8s"]),
    "terraform": ("devops", ["terraform"]),
    "git": ("devops", ["git"]),
    "linux": ("devops", ["linux"]),
    "ci/cd": ("devops", ["ci/cd", "cicd", "github actions", "jenkins"]),
    # Web
    "react": ("web", ["!React", "react.js", "reactjs"]),
    "node.js": ("web", ["node.js", "nodejs"]),
    "fastapi": ("web", ["fastapi"]),
    "django": ("web", ["django"]),
    "flask": ("web", ["flask"]),
    "spring boot": ("web", ["spring boot", "!Spring"]),
    "graphql": ("web", ["graphql"]),
    "rest api": ("web", ["rest api", "rest apis", "restful"]),
    "html": ("web", ["html", "html5"]),
    "css": ("web", ["css", "css3"]),
}

CATEGORY = {skill: cat for skill, (cat, _) in TAXONOMY.items()}

# Characters that, when adjacent to an alias, mean it is part of a longer token.
_LEFT = r"(?<![A-Za-z0-9+#.])"
_RIGHT = r"(?![A-Za-z0-9+#])"


def _compile(aliases: list[str]) -> tuple[re.Pattern | None, re.Pattern | None]:
    ci = [re.escape(a) for a in aliases if not a.startswith("!")]
    cs = [re.escape(a[1:]) for a in aliases if a.startswith("!")]
    ci_pat = re.compile(_LEFT + "(?:" + "|".join(ci) + ")" + _RIGHT, re.I) if ci else None
    cs_pat = re.compile(_LEFT + "(?:" + "|".join(cs) + ")" + _RIGHT) if cs else None
    return ci_pat, cs_pat


@lru_cache(maxsize=1)
def _patterns() -> dict[str, tuple[re.Pattern | None, re.Pattern | None]]:
    return {skill: _compile(aliases) for skill, (_, aliases) in TAXONOMY.items()}


@lru_cache(maxsize=1)
def _alias_lookup() -> dict[str, str]:
    lookup = {}
    for skill, (_, aliases) in TAXONOMY.items():
        lookup[skill.lower()] = skill
        for a in aliases:
            lookup[a.lstrip("!").lower()] = skill
    return lookup


def normalize_skill(name: str) -> str | None:
    """Map any alias ("Postgres", "K8s") to its canonical skill, or None if unknown."""
    return _alias_lookup().get(name.strip().lower())


def extract_skills(text: str) -> list[str]:
    """Return the sorted list of canonical skills mentioned in ``text``."""
    if not text:
        return []
    found = []
    for skill, (ci, cs) in _patterns().items():
        if (ci and ci.search(text)) or (cs and cs.search(text)):
            found.append(skill)
    return sorted(found)


class SpacySkillExtractor:
    """Same taxonomy, matched with spaCy's PhraseMatcher on a blank English pipeline.

    Needs only ``pip install spacy`` (no model download). Useful as a drop-in you can
    extend with a trained NER model later. Case-sensitive ("!") aliases are matched on
    exact text; everything else on lowercase.
    """

    def __init__(self) -> None:
        import spacy
        from spacy.matcher import PhraseMatcher

        self.nlp = spacy.blank("en")
        self._lower = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._exact = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        for skill, (_, aliases) in TAXONOMY.items():
            ci = [self.nlp.make_doc(a) for a in aliases if not a.startswith("!")]
            cs = [self.nlp.make_doc(a[1:]) for a in aliases if a.startswith("!")]
            if ci:
                self._lower.add(skill, ci)
            if cs:
                self._exact.add(skill, cs)

    def __call__(self, text: str) -> list[str]:
        doc = self.nlp(text or "")
        hits = {self.nlp.vocab.strings[m] for m, _, _ in self._lower(doc)}
        hits |= {self.nlp.vocab.strings[m] for m, _, _ in self._exact(doc)}
        return sorted(hits)
