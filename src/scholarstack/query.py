"""Query construction: how a job description is turned into a retrieval query.

Sentence-transformers truncate input (256 tokens here) and job ads often open
with company boilerplate, so embedding the raw ad wastes most of the budget.
Following the architecture (JD -> skill extraction -> embed query), the query
is built from the title plus the extracted skills, optionally followed by the
start of the description. The browser builds exactly the same string.
"""
from __future__ import annotations

from .skills import SkillExtractor


def split_title(jd: str) -> tuple[str, str]:
    lines = [l.strip() for l in jd.strip().split("\n") if l.strip()]
    if not lines:
        return "", ""
    if len(lines[0].split()) <= 12:
        return lines[0], "\n".join(lines[1:])
    return "", jd


def jd_skills(jd: str, ex: SkillExtractor, include_soft: bool = False, top: int = 25) -> list[str]:
    ids = ex.extract(jd)
    if not include_soft:
        ids = [i for i in ids if ex.by_id[i]["kind"] != "soft"]
    return ids[:top]


def build_query(jd: str, ex: SkillExtractor, variant: str = "hybrid", title: str | None = None) -> str:
    t, body = split_title(jd)
    title = title or t
    skills = ", ".join(ex.names(jd_skills(jd, ex)))
    if variant == "raw":
        return jd
    if variant == "skills":
        return f"{title}. Skills: {skills}"
    if variant == "hybrid":
        lead = " ".join(body.split()[:80])
        return f"{title}. Skills: {skills}. {lead}"
    raise ValueError(variant)


QUERY_VARIANTS = ("raw", "skills", "hybrid")
