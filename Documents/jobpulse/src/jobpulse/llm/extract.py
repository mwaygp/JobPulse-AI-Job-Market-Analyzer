"""LLM-based structured extraction from messy job descriptions, plus a comparison
against the rule-based extractor (a good write-up for your README)."""
from __future__ import annotations

import json
import re

import pandas as pd

from jobpulse.llm.client import complete, get_client
from jobpulse.nlp.skills import extract_skills, normalize_skill

SYSTEM = """You extract structured data from job postings.
Return ONLY a JSON object, no prose and no markdown fences, with exactly these keys:
{"required_skills": [str], "nice_to_have_skills": [str],
 "seniority": "Junior" | "Mid-level" | "Senior" | "Staff" | null,
 "remote_policy": "remote" | "hybrid" | "onsite" | null,
 "years_experience": number | null}
Skills are technologies, languages, frameworks and tools only (no soft skills)."""


def parse_json_block(text: str) -> dict:
    """Parse a JSON object out of an LLM reply, tolerating code fences and stray prose."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def _canon(skills) -> list[str]:
    out = []
    for s in skills or []:
        c = normalize_skill(str(s))
        out.append(c if c else str(s).strip().lower())
    return sorted(set(out))


def extract_structured(description: str, client=None) -> dict:
    client = client or get_client()
    if client is None:
        raise RuntimeError("No Claude client: set ANTHROPIC_API_KEY and `pip install anthropic`")
    data = parse_json_block(complete(client, SYSTEM, description[:6000]))
    data["required_skills"] = _canon(data.get("required_skills"))
    data["nice_to_have_skills"] = _canon(data.get("nice_to_have_skills"))
    return data


def compare_extractors(df: pd.DataFrame, n: int = 20, client=None, seed: int = 0) -> dict:
    """Jaccard overlap between rule-based and LLM skill sets on a random sample."""
    client = client or get_client()
    if client is None:
        raise RuntimeError("No Claude client available")
    sample = df.sample(min(n, len(df)), random_state=seed)
    scores = []
    for desc in sample["description"]:
        rules = set(extract_skills(desc))
        llm = extract_structured(desc, client)
        llm_set = set(llm["required_skills"]) | set(llm["nice_to_have_skills"])
        union = rules | llm_set
        scores.append(len(rules & llm_set) / len(union) if union else 1.0)
    return {"n": len(scores), "mean_jaccard": sum(scores) / len(scores)}
