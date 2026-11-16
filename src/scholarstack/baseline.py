"""TF-IDF baselines — "generic keyword advice" with no retrieval of similar people
and no generation.

* TfidfRetriever: sparse retrieval of resume chunks (retrieval-level baseline).
* keyword_advice: the user-facing baseline — scores resume/JD keyword overlap and
  lists the highest-weighted JD terms that the resume is missing.

The fitted vocabulary + IDF weights are exported so the browser can run exactly
the same baseline live (see web/js/tfidf.js).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .config import CFG, log, path

# Words common in job ads that carry no skill information.
JOB_STOP = {"experience", "work", "working", "job", "position", "role", "team", "company", "including",
            "ability", "able", "skills", "required", "preferred", "requirements", "responsibilities",
            "years", "year", "candidate", "candidates", "employer", "opportunity", "employment", "apply",
            "status", "benefits", "equal", "applicants", "qualified", "race", "color", "religion", "sex",
            "gender", "national", "origin", "disability", "veteran", "age", "orientation", "identity",
            "company", "name", "city", "state", "current", "new", "use", "using", "strong", "excellent",
            "duties", "time", "full", "part", "pay", "hour", "hours", "salary", "per", "day", "days",
            "week", "weeks", "month", "months", "date", "must", "will", "may", "also", "well", "within",
            "etc", "e.g", "eg", "ie", "us", "our", "we", "you", "your", "plus", "related", "relevant"}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | JOB_STOP)


def make_vectorizer(max_features: int | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=3, max_df=0.4,
                           sublinear_tf=True, max_features=max_features, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#]+\b")


class TfidfRetriever:
    def __init__(self, chunks: pd.DataFrame):
        self.chunks = chunks.reset_index(drop=True)
        self.vec = make_vectorizer()
        self.matrix = self.vec.fit_transform(self.chunks["text"])

    def search(self, query: str, k: int) -> pd.DataFrame:
        q = self.vec.transform([query])
        scores = linear_kernel(q, self.matrix).ravel()
        top = np.argsort(-scores)[:k]
        hits = self.chunks.iloc[top].copy()
        hits["score"] = scores[top]
        hits["rank"] = np.arange(1, len(hits) + 1)
        return hits


def fit_advice_model(sample_postings: int = 20_000, max_features: int = 15_000) -> TfidfVectorizer:
    """IDF fitted on resumes + a sample of postings: the reference corpus for keyword weighting."""
    resumes = pd.read_parquet(path("processed") / "resumes.parquet", columns=["text"])
    postings = pd.read_parquet(path("processed") / "postings", columns=["description"])
    corpus = pd.concat([resumes["text"],
                        postings["description"].sample(min(sample_postings, len(postings)), random_state=CFG["seed"])])
    vec = make_vectorizer(max_features=max_features)
    vec.fit(corpus)
    log.info("TF-IDF advice model: %d terms from %d documents", len(vec.vocabulary_), len(corpus))
    return vec


def keyword_advice(vec: TfidfVectorizer, jd: str, resume: str | None = None, top_n: int = 25) -> dict:
    terms = vec.get_feature_names_out()
    jd_v = vec.transform([jd])
    order = np.argsort(-jd_v.toarray().ravel())
    keywords = [terms[i] for i in order[:top_n] if jd_v[0, i] > 0]
    out = {"keywords": keywords}
    if resume:
        r_v = vec.transform([resume])
        out["match_score"] = float(linear_kernel(jd_v, r_v)[0, 0])
        present = set(np.asarray(terms)[r_v.nonzero()[1]])
        out["missing_keywords"] = [t for t in keywords if t not in present]
        out["matched_keywords"] = [t for t in keywords if t in present]
    return out


def export_for_browser(vec: TfidfVectorizer, out_file) -> None:
    payload = {
        "vocab": {t: round(float(vec.idf_[i]), 4) for t, i in vec.vocabulary_.items()},
        "stop_words": STOP_WORDS,
        "token_pattern": r"\b[a-zA-Z][a-zA-Z+#]+\b",
        "ngram_range": [1, 2],
        "sublinear_tf": True,
    }
    out_file.write_text(json.dumps(payload, separators=(",", ":")))
    log.info("exported TF-IDF model (%d terms, %.0f KB) -> %s", len(payload["vocab"]),
             out_file.stat().st_size / 1024, out_file.name)
