from jobpulse.nlp.skills import extract_skills, normalize_skill


def test_java_does_not_match_javascript():
    assert extract_skills("We use JavaScript daily") == ["javascript"]
    assert "java" not in extract_skills("We use JavaScript daily")


def test_special_character_tokens():
    got = extract_skills("Experience with C++, C#, Node.js and CI/CD required.")
    assert {"c++", "c#", "node.js", "ci/cd"} <= set(got)


def test_case_insensitive_and_aliases():
    got = extract_skills("PYTHON, Postgres, k8s and sklearn")
    assert {"python", "postgresql", "kubernetes", "scikit-learn"} <= set(got)


def test_case_sensitive_ambiguous_words():
    assert "react" not in extract_skills("You must react quickly to incidents")
    assert "react" in extract_skills("Build UIs with React and TypeScript")
    assert "spark" not in extract_skills("Our team will spark ideas")
    assert "spark" in extract_skills("Pipelines in Spark")


def test_git_not_matched_inside_github():
    assert "git" not in extract_skills("Find us on GitHub")
    assert "git" in extract_skills("Comfortable with Git workflows")


def test_empty_and_none():
    assert extract_skills("") == []
    assert extract_skills(None) == []


def test_normalize_skill():
    assert normalize_skill("Postgres") == "postgresql"
    assert normalize_skill("K8s") == "kubernetes"
    assert normalize_skill("totally-unknown-thing") is None


def test_spacy_extractor_matches_regex_on_simple_text():
    import pytest

    pytest.importorskip("spacy")
    from jobpulse.nlp.skills import SpacySkillExtractor

    text = "Python, SQL and Docker on AWS"
    assert SpacySkillExtractor()(text) == extract_skills(text)
