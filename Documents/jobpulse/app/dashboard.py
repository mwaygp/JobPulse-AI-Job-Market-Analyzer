"""Streamlit dashboard.  Run:  streamlit run app/dashboard.py"""
import streamlit as st

from jobpulse.analysis import resume_gap, salary_by_skill, skill_demand
from jobpulse.config import SALARY_MODEL_PATH
from jobpulse.db import load_jobs
from jobpulse.ingest.pipeline import ensure_data
from jobpulse.llm.rag import JobIndex, ask
from jobpulse.ml.clustering import cluster_roles
from jobpulse.ml.salary import load_salary_model, predict_salary, train_salary_model

st.set_page_config(page_title="JobPulse", page_icon="📈", layout="wide")


@st.cache_data
def get_jobs():
    ensure_data()
    return load_jobs()


@st.cache_resource
def get_salary_bundle():
    return load_salary_model() if SALARY_MODEL_PATH.exists() else train_salary_model(get_jobs())


@st.cache_resource
def get_index():
    return JobIndex().build(get_jobs())


@st.cache_data
def get_clusters(k: int):
    return cluster_roles(get_jobs(), k=k)


jobs = get_jobs()
st.title("📈 JobPulse: AI Job Market Analyzer")
st.caption(
    f"{len(jobs):,} postings loaded. If you have not ingested real data, this is a SYNTHETIC "
    "demo dataset; see the README to load real postings."
)

tab_skills, tab_pay, tab_clusters, tab_ask, tab_resume = st.tabs(
    ["Skill demand", "Salary", "Role clusters", "Ask the market", "Resume gap"]
)

with tab_skills:
    c1, c2 = st.columns([1, 3])
    seniority = c1.multiselect("Seniority", sorted(jobs["seniority"].unique()))
    title = c1.multiselect("Role", sorted(jobs["title"].unique()))
    only_remote = c1.checkbox("Remote only")
    view = jobs
    if seniority:
        view = view[view["seniority"].isin(seniority)]
    if title:
        view = view[view["title"].isin(title)]
    if only_remote:
        view = view[view["remote"]]
    if view.empty:
        c2.warning("No postings match these filters.")
    else:
        demand = skill_demand(view, top=20)
        c2.subheader(f"Most requested skills ({len(view):,} postings)")
        c2.bar_chart(demand.set_index("skill")["pct"], y_label="% of postings")
        c2.dataframe(demand, use_container_width=True, hide_index=True)

with tab_pay:
    left, right = st.columns(2)
    left.subheader("Median salary by skill")
    pay = salary_by_skill(jobs, min_count=15)
    left.bar_chart(pay.head(20).set_index("skill")["median_salary"])

    right.subheader("Salary predictor")
    bundle = get_salary_bundle()
    m = bundle["metrics"]
    right.caption(f"{m['backend']} | hold-out MAE ${m['mae']:,.0f} (baseline ${m['baseline_mae']:,.0f})")
    p_title = right.selectbox("Role", sorted(jobs["title"].unique()))
    p_sen = right.selectbox("Seniority", ["Junior", "Mid-level", "Senior", "Staff"], index=1)
    p_loc = right.selectbox("Location", sorted(jobs["location"].unique()))
    p_remote = right.checkbox("Remote", key="pred_remote")
    all_skills = sorted(skill_demand(jobs, top=200)["skill"])
    p_skills = right.multiselect("Skills", all_skills, default=[s for s in ("python", "sql") if s in all_skills])
    if right.button("Predict"):
        value = predict_salary(bundle, p_title, p_sen, p_loc, p_remote, p_skills)
        right.metric("Predicted salary", f"${value:,.0f}")

with tab_clusters:
    k = st.slider("Number of clusters", 2, 12, 6)
    clustered, summary, sil = get_clusters(k)
    st.caption(f"Silhouette score: {sil:.3f} (higher = better separated)")
    plot = clustered.assign(cluster=clustered["cluster"].astype(str))
    st.scatter_chart(plot, x="x", y="y", color="cluster")
    st.dataframe(summary, use_container_width=True, hide_index=True)

with tab_ask:
    q = st.text_input("Ask a question", "What skills do remote NLP roles require most?")
    if q:
        out = ask(q, get_index())
        st.write(out["answer"])
        st.caption(f"Mode: {out['mode']}")
        st.dataframe(out["sources"], use_container_width=True)

with tab_resume:
    role = st.text_input("Target role keyword", "Machine Learning")
    resume = st.text_area("Paste your resume text", height=200)
    if resume:
        gap = resume_gap(jobs, resume, role=role)
        st.subheader("Skills you are missing, ranked by demand")
        st.dataframe(gap, use_container_width=True, hide_index=True)
