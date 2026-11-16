"""Profiling, cleaning and integration of the three sources.

Resumes  -> section-aware parse of the LiveCareer HTML (pandas; 2.4k rows)
Postings -> CSV staged to Parquet, then cleaned/deduplicated/joined with Dask
O*NET    -> one profile per occupation (skills, knowledge, tasks, technologies)
"""
from __future__ import annotations

import json
import re

import dask.dataframe as dd
import pandas as pd
from lxml import html as LH

from .categories import match_categories, unique_category
from .config import CFG, log, path
from .storage import open_raw
from .textutil import is_mostly_english, normalize, text_hash, word_count

# --------------------------------------------------------------------------- resumes
# LiveCareer section codes -> canonical resume sections
SECTION_MAP = {
    "NAME": "title",
    "SUMM": "summary", "PRFL": "summary", "EXFU": "summary", "RTTL": "summary",
    "HILT": "skills", "SKLL": "skills", "TSKL": "skills", "COMS": "skills", "LANG": "skills",
    "ACCM": "accomplishments", "AWAR": "accomplishments",
    "EXPR": "experience", "MILI": "experience", "WRKH": "experience", "EXCA": "experience",
    "EDUC": "education",
    "CERT": "certifications",
    "AFIL": "affiliations",
    "PUBL": "publications", "PRES": "publications",
}
BLOCK_TAGS = {"p", "li", "div", "br", "tr", "td", "h1", "h2", "h3", "ul", "table"}


def _block_text(el) -> str:
    for node in el.iter():
        if node.tag in BLOCK_TAGS:
            node.tail = "\n" + (node.tail or "")
    return normalize(el.text_content())


def parse_resume_html(markup: str) -> list[dict]:
    """Split a LiveCareer resume into sections -> paragraphs (one job = one paragraph)."""
    doc = LH.fromstring(markup)
    sections = []
    for sec in doc.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " section ")]'):
        m = re.match(r"SECTION_([A-Z]{4})", sec.get("id", ""))
        code = m.group(1) if m else "OTHR"
        title_el = sec.xpath('.//div[contains(@class, "sectiontitle")]')
        heading = normalize(title_el[0].text_content()) if title_el else ""
        paras = [t for t in (_block_text(p) for p in sec.xpath('.//div[contains(@class, "paragraph")]')) if t]
        if paras:
            sections.append({"code": code, "section": SECTION_MAP.get(code, "additional"),
                             "heading": heading, "paragraphs": paras})
    return sections


def clean_resumes() -> pd.DataFrame:
    with open_raw("resumes/Resume/Resume.csv") as fh:
        raw = pd.read_csv(fh)
    rows, dropped = [], {"short": 0, "duplicate": 0, "non_english": 0}
    seen = set()
    for rec in raw.itertuples(index=False):
        sections = parse_resume_html(rec.Resume_html)
        title_sec = next((s for s in sections if s["section"] == "title"), None)
        title = title_sec["paragraphs"][0].split("\n")[0] if title_sec else ""
        body = "\n\n".join(
            (s["heading"] + "\n" if s["heading"] else "") + "\n".join(s["paragraphs"])
            for s in sections if s["section"] != "title"
        )
        n = word_count(body)
        h = text_hash(body)
        if n < CFG["cleaning"]["min_resume_words"]:
            dropped["short"] += 1
            continue
        if h in seen:
            dropped["duplicate"] += 1
            continue
        if not is_mostly_english(body):
            dropped["non_english"] += 1
            continue
        seen.add(h)
        rows.append({
            "resume_id": str(rec.ID), "category": rec.Category, "title": title.title(),
            "text": body, "n_words": n, "n_sections": len(sections),
            "sections": json.dumps(sections),
        })
    df = pd.DataFrame(rows)
    out = path("processed") / "resumes.parquet"
    df.to_parquet(out, index=False)
    log.info("resumes: %d kept of %d (dropped %s) -> %s", len(df), len(raw), dropped, out.name)
    return df


# --------------------------------------------------------------------------- postings
POSTING_COLS = [
    "job_id", "company_name", "title", "description", "skills_desc", "location",
    "formatted_work_type", "formatted_experience_level", "remote_allowed", "normalized_salary",
    "pay_period", "min_salary", "med_salary", "max_salary", "views", "applies", "listed_time",
]


def stage_postings(rows_per_part: int = 10_000) -> int:
    """CSV -> Parquet parts. The CSV has quoted multi-line descriptions, which
    Dask's byte-range CSV reader cannot split safely, so it is staged once with
    pandas' streaming reader and every later step runs on Parquet with Dask."""
    out = path("interim") / "postings_raw"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.parquet"):
        old.unlink()
    total = 0
    with open_raw("linkedin/postings.csv") as fh:
        for i, chunk in enumerate(pd.read_csv(fh, chunksize=rows_per_part, usecols=POSTING_COLS,
                                              dtype={"job_id": "int64", "zip_code": "string"})):
            for c in ("company_name", "title", "description", "skills_desc", "location",
                      "formatted_work_type", "formatted_experience_level", "pay_period"):
                chunk[c] = chunk[c].astype("string")
            chunk.to_parquet(out / f"part-{i:03d}.parquet", index=False)
            total += len(chunk)
    log.info("staged %d postings into %d Parquet parts", total, i + 1)
    return total


def _clean_partition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("").map(normalize)
    df["description"] = df["description"].fillna("").map(normalize)
    df["skills_desc"] = df["skills_desc"].fillna("").map(normalize)
    df["n_words"] = df["description"].map(word_count).astype("int32")
    df["english"] = df["description"].map(is_mostly_english)
    df["dedupe_key"] = [text_hash(t, c or "", d[:2000]) for t, c, d in
                        zip(df["title"], df["company_name"].fillna(""), df["description"])]
    cats = df["title"].map(match_categories)
    df["matched_categories"] = cats.map(lambda c: "|".join(c))
    df["category"] = df["title"].map(unique_category).astype("string")
    df["remote_allowed"] = df["remote_allowed"].fillna(0).astype(bool)
    return df


def clean_postings() -> dd.DataFrame:
    raw = path("raw") / "linkedin"
    ddf = dd.read_parquet(path("interim") / "postings_raw")
    n_raw = len(ddf)

    meta = _clean_partition(ddf._meta)
    ddf = ddf.map_partitions(_clean_partition, meta=meta)
    min_words = CFG["cleaning"]["min_posting_words"]
    ddf = ddf[(ddf["n_words"] >= min_words) & ddf["english"] & (ddf["title"].str.len() > 0)]
    ddf = ddf.drop_duplicates(subset="dedupe_key")

    # Companion tables (small) -> one row per job, joined onto the Dask frame.
    skills = pd.read_csv(raw / "jobs" / "job_skills.csv").merge(
        pd.read_csv(raw / "mappings" / "skills.csv"), on="skill_abr", how="left")
    skills = skills.groupby("job_id")["skill_name"].agg(lambda s: "|".join(sorted(set(s.dropna())))).rename("linkedin_functions")
    inds = pd.read_csv(raw / "jobs" / "job_industries.csv").merge(
        pd.read_csv(raw / "mappings" / "industries.csv"), on="industry_id", how="left")
    inds = inds.groupby("job_id")["industry_name"].agg(lambda s: "|".join(sorted(set(s.dropna())))).rename("industries")
    side = pd.concat([skills, inds], axis=1).reset_index()
    side["linkedin_functions"] = side["linkedin_functions"].astype("string")
    side["industries"] = side["industries"].astype("string")
    ddf = ddf.merge(side, on="job_id", how="left")
    ddf = ddf.drop(columns=["english", "dedupe_key"])

    out = path("processed") / "postings"
    ddf.to_parquet(out, write_index=False, overwrite=True)
    n_clean = len(dd.read_parquet(out))
    log.info("postings: %d -> %d after cleaning/dedup (Dask, %d partitions) -> %s",
             n_raw, n_clean, ddf.npartitions, out.name)
    return dd.read_parquet(out)


# --------------------------------------------------------------------------- O*NET
def _onet(name: str) -> pd.DataFrame:
    with open_raw(f"onet/{name}.txt") as fh:
        return pd.read_csv(fh, sep="\t", dtype=str)


def _rated(name: str, min_importance: float) -> pd.DataFrame:
    df = _onet(name)
    df = df[(df["Scale ID"] == "IM") & (df["Recommend Suppress"] != "Y")]
    df["importance"] = df["Data Value"].astype(float)
    return df[df["importance"] >= min_importance][["O*NET-SOC Code", "Element Name", "importance"]]


def build_onet_profiles() -> pd.DataFrame:
    cfg = CFG["onet"]
    occ = _onet("Occupation Data").rename(columns={"O*NET-SOC Code": "code", "Title": "title", "Description": "description"})
    zones = _onet("Job Zones").rename(columns={"O*NET-SOC Code": "code", "Job Zone": "job_zone"})[["code", "job_zone"]]
    occ = occ.merge(zones, on="code", how="left")

    def ranked(df: pd.DataFrame, label: str) -> pd.Series:
        df = df.sort_values("importance", ascending=False)
        return df.groupby("O*NET-SOC Code").apply(
            lambda g: [{"name": n, "importance": round(v, 2)} for n, v in zip(g["Element Name"], g["importance"])],
            include_groups=False).rename(label)

    ess = ranked(_rated("Essential Skills", cfg["min_importance"]), "essential_skills")
    trf = ranked(_rated("Transferable Skills", cfg["min_importance"]), "transferable_skills")
    knw = ranked(_rated("Knowledge", cfg["min_knowledge_importance"]), "knowledge")

    tasks = _onet("Task Statements")
    tasks["core"] = tasks["Task Type"].eq("Core")
    tasks = tasks.sort_values(["O*NET-SOC Code", "core"], ascending=[True, False])
    tasks = tasks.groupby("O*NET-SOC Code")["Task"].agg(lambda s: list(s)[: cfg["max_tasks"]]).rename("tasks")

    sw = _onet("Software Skills")
    sw["rank"] = (sw["Hot Technology"].eq("Y")).astype(int) + (sw["In Demand"].eq("Y")).astype(int)
    sw = sw.sort_values(["O*NET-SOC Code", "rank"], ascending=[True, False])
    tech = sw.groupby("O*NET-SOC Code").apply(
        lambda g: [{"name": n, "category": c, "hot": h == "Y"} for n, c, h in
                   zip(g["Workplace Example"], g["Element Name"], g["Hot Technology"])][: cfg["max_technologies"]],
        include_groups=False).rename("technologies")

    titles = _onet("Job Titles").groupby("O*NET-SOC Code")["Job Title"].agg(list).rename("alt_titles")

    prof = occ.set_index("code").join([ess, trf, knw, tasks, tech, titles]).reset_index()
    for col in ("essential_skills", "transferable_skills", "knowledge", "tasks", "technologies", "alt_titles"):
        prof[col] = prof[col].map(lambda v: v if isinstance(v, list) else [])

    def profile_text(r) -> str:
        skills = ", ".join(s["name"] for s in (r.essential_skills + r.transferable_skills)[:12])
        tech_ = ", ".join(t["name"] for t in r.technologies[:12])
        know = ", ".join(k["name"] for k in r.knowledge[:6])
        return (f"{r.title}. {r.description} Skills: {skills}. Technologies: {tech_}. "
                f"Knowledge: {know}. Tasks: {' '.join(r.tasks[:4])}")

    prof["profile_text"] = prof.apply(profile_text, axis=1)
    # Occupations with no rated data (e.g. "All Other" catch-alls) are kept for title
    # matching but flagged so generation does not rely on empty profiles.
    prof["has_profile"] = prof["essential_skills"].map(len) > 0
    out = path("processed") / "onet_occupations.parquet"
    store = prof.copy()
    for col in ("essential_skills", "transferable_skills", "knowledge", "tasks", "technologies", "alt_titles"):
        store[col] = store[col].map(json.dumps)
    store.to_parquet(out, index=False)
    log.info("O*NET: %d occupations (%d with rated skills) -> %s", len(prof), prof["has_profile"].sum(), out.name)
    return prof


def load_onet_profiles() -> pd.DataFrame:
    df = pd.read_parquet(path("processed") / "onet_occupations.parquet")
    for col in ("essential_skills", "transferable_skills", "knowledge", "tasks", "technologies", "alt_titles"):
        df[col] = df[col].map(json.loads)
    return df
