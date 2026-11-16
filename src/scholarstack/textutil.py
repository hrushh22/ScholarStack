"""Small, dependency-free text helpers shared across the pipeline."""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata

_WS = re.compile(r"[ \t 　]+")
_NL = re.compile(r"\s*\n\s*")
_PLACEHOLDERS = re.compile(r"\b(Company Name|City\s*,\s*State|City|State)\b")


def normalize(text: str | None) -> str:
    """Unicode-normalise, unescape HTML entities and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", html.unescape(text))
    t = t.replace("－", "-").replace("•", "-").replace("\r", "\n")
    t = _WS.sub(" ", t)
    t = _NL.sub("\n", t)
    return t.strip()


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def word_count(text: str | None) -> int:
    return len(text.split()) if isinstance(text, str) else 0


def text_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(one_line(p or "").lower().encode())
        h.update(b"\x1f")
    return h.hexdigest()


def strip_placeholders(text: str) -> str:
    """LiveCareer resumes anonymise employers as 'Company Name' / 'City, State'."""
    return one_line(_PLACEHOLDERS.sub("", text).replace(" - ,", "").replace(" , ", " "))


def is_mostly_english(text: str, threshold: float = 0.85) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(c.isascii() for c in letters) / len(letters) >= threshold
