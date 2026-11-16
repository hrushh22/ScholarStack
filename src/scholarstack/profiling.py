"""Data profiling & validation: schema, null rates, duplicates, length outliers."""
from __future__ import annotations

import json

import dask.dataframe as dd
import pandas as pd

from .config import log, path
from .storage import open_raw
from .textutil import word_count


def _frame_profile(df: pd.DataFrame, text_cols: list[str], key: str | None = None) -> dict:
    prof = {
        "rows": int(len(df)),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "null_rate": {c: round(float(df[c].isna().mean()), 4) for c in df.columns},
    }
    if key:
        prof["duplicate_keys"] = int(df[key].duplicated().sum())
    for c in text_cols:
        wc = df[c].map(word_count)
        q1, q3 = wc.quantile([0.25, 0.75])
        iqr = q3 - q1
        prof[f"{c}_words"] = {
            "mean": round(float(wc.mean()), 1), "median": float(wc.median()),
            "min": int(wc.min()), "max": int(wc.max()),
            "outliers_low": int((wc < max(q1 - 1.5 * iqr, 0)).sum()),
            "outliers_high": int((wc > q3 + 1.5 * iqr).sum()),
        }
        prof[f"{c}_duplicate_texts"] = int(df[c].fillna("").str.strip().duplicated().sum())
    return prof


def profile_all() -> dict:
    report: dict = {}

    with open_raw("resumes/Resume/Resume.csv") as fh:
        resumes = pd.read_csv(fh)
    report["resumes"] = _frame_profile(resumes, ["Resume_str"], key="ID")
    report["resumes"]["category_counts"] = resumes["Category"].value_counts().to_dict()

    postings = dd.read_parquet(path("interim") / "postings_raw")
    pdf = postings.compute()
    report["postings"] = _frame_profile(pdf, ["description"], key="job_id")
    report["postings"]["work_type_counts"] = pdf["formatted_work_type"].value_counts().to_dict()
    report["postings"]["experience_level_counts"] = pdf["formatted_experience_level"].value_counts(dropna=False).rename(
        index=lambda x: x if isinstance(x, str) else "Not stated").to_dict()

    onet = {}
    for name in ("Occupation Data", "Essential Skills", "Transferable Skills", "Knowledge",
                 "Software Skills", "Task Statements", "Job Titles", "Abilities"):
        with open_raw(f"onet/{name}.txt") as fh:
            t = pd.read_csv(fh, sep="\t", dtype=str)
        onet[name] = {"rows": int(len(t)), "columns": list(t.columns),
                      "occupations": int(t["O*NET-SOC Code"].nunique())}
    report["onet"] = onet

    out = path("results") / "data_profile.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("profile written to %s", out)
    return report
