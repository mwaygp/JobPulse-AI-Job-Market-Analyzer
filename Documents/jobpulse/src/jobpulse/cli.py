"""Command line interface:  jobpulse <command>  (or python -m jobpulse.cli <command>)"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from jobpulse.analysis import salary_by_skill
from jobpulse.config import DB_PATH, INDEX_PATH, SALARY_MODEL_PATH
from jobpulse.db import load_jobs, top_skills_sql
from jobpulse.ingest import loaders
from jobpulse.ingest.pipeline import build_sample_dataset, ingest

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 60)


def cmd_init(a):
    df = build_sample_dataset(n=a.n)
    print(f"Loaded {len(df)} SYNTHETIC postings into {DB_PATH}")


def cmd_ingest_csv(a):
    mapping = json.loads(a.mapping) if a.mapping else None
    df = ingest(loaders.load_csv(a.path, mapping))
    print(f"Ingested {len(df)} postings from {a.path}")


def cmd_ingest_adzuna(a):
    df = ingest(loaders.fetch_adzuna(what=a.query, country=a.country, pages=a.pages))
    print(f"Ingested {len(df)} postings from Adzuna")


def cmd_report(a):
    print("\nMost requested skills (SQL):\n")
    print(top_skills_sql(a.top).to_string(index=False))
    print("\nHighest-paying skills (median salary of postings mentioning them):\n")
    print(salary_by_skill(load_jobs(), min_count=a.min_count).head(a.top).to_string(index=False))


def cmd_train(a):
    from jobpulse.ml.salary import train_salary_model

    b = train_salary_model(load_jobs())
    m = b["metrics"]
    print(f"Model: {m['backend']}  | train={m['n_train']} test={m['n_test']}")
    print(f"MAE ${m['mae']:,.0f}  (baseline ${m['baseline_mae']:,.0f})  R2 {m['r2']:.3f}")
    print("\nTop features:\n", b["importance"].head(15).to_string(index=False))
    print(f"\nSaved to {SALARY_MODEL_PATH}")


def cmd_cluster(a):
    from jobpulse.ml.clustering import cluster_roles

    _, summary, sil = cluster_roles(load_jobs(), k=a.k)
    print(f"Silhouette: {sil:.3f}\n")
    print(summary.to_string(index=False))


def cmd_index(a):
    from jobpulse.llm.rag import JobIndex

    idx = JobIndex().build(load_jobs())
    idx.save()
    print(f"Indexed {len(idx.records)} postings ({idx.embedder.active_backend}) -> {INDEX_PATH}")


def cmd_ask(a):
    from jobpulse.llm.rag import JobIndex, ask

    out = ask(a.question, JobIndex.load())
    print(out["answer"], "\n\nSources:")
    for s in out["sources"]:
        print(f"  [{s['id']}] {s['title']} @ {s['company']} ({s['score']:.2f})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="jobpulse", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="build the synthetic demo dataset")
    s.add_argument("-n", type=int, default=1500)
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("ingest-csv", help="load a real CSV dataset")
    s.add_argument("path")
    s.add_argument("--mapping", help='JSON, e.g. \'{"title":"Job Title","description":"Job Description"}\'')
    s.set_defaults(fn=cmd_ingest_csv)

    s = sub.add_parser("ingest-adzuna", help="pull postings from the Adzuna API")
    s.add_argument("--query", default="data scientist")
    s.add_argument("--country", default="us")
    s.add_argument("--pages", type=int, default=5)
    s.set_defaults(fn=cmd_ingest_adzuna)

    s = sub.add_parser("report", help="print top skills and pay-by-skill")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--min-count", type=int, default=10)
    s.set_defaults(fn=cmd_report)

    sub.add_parser("train", help="train the salary model").set_defaults(fn=cmd_train)

    s = sub.add_parser("cluster", help="cluster roles")
    s.add_argument("-k", type=int, default=6)
    s.set_defaults(fn=cmd_cluster)

    sub.add_parser("index", help="build the RAG index").set_defaults(fn=cmd_index)

    s = sub.add_parser("ask", help="ask the job market a question")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
