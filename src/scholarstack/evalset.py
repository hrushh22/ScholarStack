"""Held-out evaluation set of job descriptions, stratified by resume category.

A posting is eligible when its title maps to exactly one category AND its
description mentions that category's keywords at least twice (guards against
false title matches such as "Lot Driver - Airport" -> AVIATION). The set is
sampled once with a fixed seed and never used for tuning prompts or parameters.
"""
from __future__ import annotations

import re

import pandas as pd

from .categories import CATEGORY_RULES
from .config import CFG, log, path


def build_eval_set() -> pd.DataFrame:
    p = pd.read_parquet(path("processed") / "postings",
                        columns=["job_id", "title", "company_name", "description", "category", "n_words",
                                 "formatted_experience_level", "location"])
    p = p.dropna(subset=["category"])
    p = p[(p["n_words"] >= 150) & (p["n_words"] <= 1200)]
    p["kw_hits"] = [len(re.findall(CATEGORY_RULES[c], d, flags=re.I)) for c, d in zip(p["category"], p["description"])]
    p = p[p["kw_hits"] >= 2]
    n = CFG["evaluation"]["retrieval_jds_per_category"]
    sample = (p.sample(frac=1.0, random_state=CFG["seed"]).groupby("category").head(n)
               .sort_values(["category", "job_id"]).reset_index(drop=True))
    # A fixed subset (first per category) is used for the costlier LLM generation experiments.
    sample["gen_subset"] = sample.groupby("category").cumcount() < CFG["evaluation"]["generation_jds_per_category"]
    sample["jd"] = sample["title"] + "\n\n" + sample["description"]
    sample.to_parquet(path("processed") / "eval_jds.parquet", index=False)
    log.info("evaluation set: %d job descriptions across %d categories (%d in generation subset)",
             len(sample), sample["category"].nunique(), sample["gen_subset"].sum())
    return sample


def load_eval_set() -> pd.DataFrame:
    return pd.read_parquet(path("processed") / "eval_jds.parquet")
