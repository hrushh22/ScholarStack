"""The 24 resume categories and title rules that map a job posting onto them.

The Kaggle resume categories were built from keywords in the resume's target
title (e.g. ADVOCATE = patient/customer advocates, not lawyers), so postings
are labelled with matching title keywords. A posting is used for evaluation
only when exactly one category matches, which keeps the proxy labels clean.
"""
from __future__ import annotations

import re

CATEGORY_RULES: dict[str, str] = {
    "ACCOUNTANT": r"\baccountant|\baccounting\b|\bbookkeep|\bcpa\b|accounts (?:payable|receivable)|\bauditor\b",
    "ADVOCATE": r"\badvocat",
    "AGRICULTURE": r"agricultur|agronom|\bfarm\b|\bfarm (?:hand|manager|worker)|\bcrop\b|horticult|extension agent|\branch\b|greenhouse|livestock|\bgrower\b",
    "APPAREL": r"apparel|fashion|garment|textile|clothing|footwear",
    "ARTS": r"\barts?\b(?! director)|\bartist\b|museum|gallery|curator|\btheat(?:er|re)\b|\bmusic\b",
    "AUTOMOBILE": r"automo|automotive|\bauto\b|vehicle|dealership|collision|car sales",
    "AVIATION": r"aviation|aircraft|\bpilot\b|airline|avionic|airport|airframe",
    "BANKING": r"\bbank|\bteller\b|loan officer|mortgage|\blending\b",
    "BPO": r"\bbpo\b|call cent(?:er|re)|contact cent(?:er|re)|customer service representative|customer care|customer support",
    "BUSINESS-DEVELOPMENT": r"business development|\bbdr\b|\bpartnerships?\b",
    "CHEF": r"\bchef\b|\bcook\b|culinary|kitchen",
    "CONSTRUCTION": r"construction|carpenter|superintendent|foreman|estimator",
    "CONSULTANT": r"consultant|consulting",
    "DESIGNER": r"designer|graphic design|\bux\b|interior design",
    "DIGITAL-MEDIA": r"digital (?:media|marketing|content|producer)|social media|media buyer|content creator|video producer",
    "ENGINEERING": r"engineer",
    "FINANCE": r"\bfinanc|\bfp&a\b|treasury|investment",
    "FITNESS": r"fitness|personal trainer|\byoga\b|pilates|\bgym\b",
    "HEALTHCARE": r"health ?care|\bnurse\b|\brn\b|\blpn\b|\bcna\b|\bmedical\b|clinical|patient care|hospital|physician|pharmac",
    "HR": r"\bhr\b|human resources|recruit|talent acquisition|people operations|\bpayroll\b",
    "INFORMATION-TECHNOLOGY": r"information technology|\bit\b|help desk|service desk|desktop support|systems administrator|network administrator",
    "PUBLIC-RELATIONS": r"public relations|\bpr\b|communications (?:manager|specialist|director|coordinator)|media relations|publicist",
    "SALES": r"\bsales\b|account executive",
    "TEACHER": r"teacher|teaching|\beducator\b|\btutor\b|paraprofessional|preschool",
}

CATEGORIES = sorted(CATEGORY_RULES)
_COMPILED = {c: re.compile(p, re.I) for c, p in CATEGORY_RULES.items()}


def match_categories(title: str | None) -> list[str]:
    if not title:
        return []
    return [c for c, rx in _COMPILED.items() if rx.search(title)]


def unique_category(title: str | None) -> str | None:
    hits = match_categories(title)
    return hits[0] if len(hits) == 1 else None


def pretty(category: str) -> str:
    special = {"HR": "HR", "BPO": "BPO", "INFORMATION-TECHNOLOGY": "Information Technology"}
    return special.get(category, category.replace("-", " ").title())
