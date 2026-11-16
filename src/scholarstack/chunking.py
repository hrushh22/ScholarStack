"""Two chunking strategies compared in the retrieval experiments.

section : one chunk per resume section; experience is split per job (one HTML
          paragraph = one position), long chunks split at line boundaries.
fixed   : sliding word windows over the whole resume, ignoring structure.

Every chunk keeps its source metadata (resume id, category, section), and the
text that gets embedded carries a short contextual header ("<title> | <section>")
so a bullet like "Reconciled accounts" still knows it came from an accountant.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .config import CFG, log, path
from .textutil import word_count

SECTION_CODE = {"summary": "SUM", "skills": "SKL", "experience": "EXP", "education": "EDU",
                "accomplishments": "ACC", "certifications": "CRT", "affiliations": "AFL",
                "publications": "PUB", "additional": "ADD"}
_PLACEHOLDER = re.compile(r"\s*\b(Company Name|City\s*,\s*State)\b\s*[-,]?")


def _clean_line(line: str) -> str:
    return re.sub(r"\s{2,}", " ", _PLACEHOLDER.sub(" ", line)).strip(" -,")


def _clean_block(text: str) -> str:
    lines = [_clean_line(l) for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


def _split_long(text: str, max_words: int) -> list[str]:
    if word_count(text) <= max_words:
        return [text]
    parts, cur, n = [], [], 0
    for line in text.split("\n"):
        w = word_count(line)
        if cur and n + w > max_words:
            parts.append("\n".join(cur))
            cur, n = [], 0
        if w > max_words:  # a single very long line -> hard word split
            words = line.split()
            for i in range(0, len(words), max_words):
                parts.append(" ".join(words[i:i + max_words]))
            continue
        cur.append(line)
        n += w
    if cur:
        parts.append("\n".join(cur))
    return parts


def section_chunks(resumes: pd.DataFrame) -> pd.DataFrame:
    max_words = CFG["chunking"]["section"]["max_words"]
    rows = []
    for r in resumes.itertuples(index=False):
        counter: dict[str, int] = {}
        for sec in json.loads(r.sections):
            if sec["section"] == "title":
                continue
            units = sec["paragraphs"] if sec["section"] == "experience" else ["\n".join(sec["paragraphs"])]
            for unit in units:
                for piece in _split_long(_clean_block(unit), max_words):
                    if word_count(piece) < 4:
                        continue
                    code = SECTION_CODE.get(sec["section"], "ADD")
                    counter[code] = counter.get(code, 0) + 1
                    rows.append({"chunk_id": f"{r.resume_id}-{code}{counter[code]}", "resume_id": r.resume_id,
                                 "category": r.category, "resume_title": r.title, "section": sec["section"],
                                 "text": piece})
    df = pd.DataFrame(rows)
    df["embed_text"] = df["resume_title"] + " | " + df["section"] + ": " + df["text"].str.replace("\n", " ")
    df["n_words"] = df["text"].map(word_count)
    return df


def fixed_chunks(resumes: pd.DataFrame) -> pd.DataFrame:
    win, overlap = CFG["chunking"]["fixed"]["window_words"], CFG["chunking"]["fixed"]["overlap_words"]
    step = win - overlap
    rows = []
    for r in resumes.itertuples(index=False):
        words = _clean_block(r.text).split()
        for i, start in enumerate(range(0, max(len(words) - overlap, 1), step), start=1):
            piece = " ".join(words[start:start + win])
            rows.append({"chunk_id": f"{r.resume_id}-W{i}", "resume_id": r.resume_id, "category": r.category,
                         "resume_title": r.title, "section": "window", "text": piece})
    df = pd.DataFrame(rows)
    df["embed_text"] = df["resume_title"] + " | " + df["text"]
    df["n_words"] = df["text"].map(word_count)
    return df


def build_chunks() -> dict[str, pd.DataFrame]:
    resumes = pd.read_parquet(path("processed") / "resumes.parquet")
    out = {}
    for name, fn in (("section", section_chunks), ("fixed", fixed_chunks)):
        df = fn(resumes)
        df.to_parquet(path("processed") / f"chunks_{name}.parquet", index=False)
        log.info("%s chunks: %d (mean %.0f words) from %d resumes", name, len(df), df["n_words"].mean(),
                 df["resume_id"].nunique())
        out[name] = df
    return out


def load_chunks(strategy: str) -> pd.DataFrame:
    return pd.read_parquet(path("processed") / f"chunks_{strategy}.parquet")
