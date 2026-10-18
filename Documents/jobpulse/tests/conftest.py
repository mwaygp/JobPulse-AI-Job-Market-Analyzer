import pytest

from jobpulse.ingest.pipeline import add_skills
from jobpulse.ingest.sample_data import generate_jobs


@pytest.fixture(scope="session")
def jobs():
    return add_skills(generate_jobs(n=600, seed=1))
