"""Rule-based skill extraction aligned to O*NET.

The lexicon has two parts:
  * technology skills - O*NET "Software Skills" examples (hot / in-demand or used by
    >= 3 occupations), with aliases derived automatically ("Microsoft Excel" -> "excel",
    "Structured query language SQL" -> "sql");
  * domain & soft skills - config/skills_curated.yaml, each group aligned to an
    O*NET knowledge/skill element.

Matching uses an Aho-Corasick automaton with word-boundary checks, so a full pass
over ~100k job descriptions takes seconds. Ambiguous product names ("React",
"Go", "Chef") only match when capitalised.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache

import ahocorasick
import pandas as pd
import yaml

from .config import ROOT, log, path
from .storage import open_raw

VENDORS = [
    "Microsoft", "Oracle", "Adobe", "Apache", "Atlassian", "Google", "IBM", "SAP", "Autodesk", "Intuit",
    "Amazon", "Amazon Web Services", "Apple", "Cisco", "The MathWorks", "ESRI", "Dassault Systemes", "Bentley",
    "Trimble", "Red Hat", "Mozilla", "Salesforce", "Henry Schein", "Kronos", "Splunk", "Alteryx", "Teradata",
    "Facebook", "Meta", "Hewlett Packard", "HP", "VMware", "Dell", "SAS", "Tableau", "Siemens", "Infor",
    "Sage", "Epic", "Cerner", "Bloomberg", "Moody's", "Thomson Reuters", "Wolters Kluwer", "Corel", "Avid",
    "Ellucian", "Blackbaud", "Esri", "PTC", "Ansys", "National Instruments", "Rockwell Automation", "Trello",
]
# Aliases never matched (ordinary words, or clashes such as the "Chef" config tool vs. chefs).
BLOCKED = {"word", "access", "project", "teams", "edge", "box", "office", "dash", "visual", "server",
           "windows", "basic", "data", "studio", "go", "c", "r", "d", "j", "x", "e", "suite", "forms",
           "maps", "sheets", "docs", "drive", "calendar", "photos", "keynote", "pages", "numbers",
           "notes", "mail", "messages", "chrome", "presentation", "publisher", "money", "time",
           "chef", "analytics", "ads", "classroom", "workspace", "meet", "cloud", "one", "point",
           "ax", "bi", "cs", "hms", "hr", "icon", "imagine", "max", "pro", "painter", "sis", "san", "director"}
# Technology names that are also common English words: matched only when capitalised,
# even if the corpus statistics below would otherwise drop them.
FORCE_CASE_SENSITIVE = {"react", "swift", "ruby", "spark", "excel", "puppet", "unity", "hive", "kafka",
                        "flask", "zoom", "slack", "outlook", "safari", "sas", "jenkins", "git", "rust"}
KEEP_SHORT = {"c++", "c#", ".net"}
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#.]*[A-Za-z0-9+#]|[A-Za-z]")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", "-", text.lower()).strip("-")


def _onet_aliases(example: str) -> tuple[str, set[str], bool]:
    """Display name, aliases, and whether the example is a generic category ("Tax software")."""
    generic = bool(re.search(r"\ssoftware$", example.strip(), flags=re.I)) and len(example.split()) == 2
    name = re.sub(r"\s+software$", "", example.strip(), flags=re.I)
    aliases = {name}
    tokens = name.split()
    display = name
    if len(tokens) > 1 and re.fullmatch(r"[A-Z][A-Z0-9+#/]{1,6}", tokens[-1]):
        acronym, rest = tokens[-1], tokens[:-1]
        aliases.add(acronym)
        # "Structured query language SQL" -> also "structured query language"
        if len(rest) >= 2 and rest[0][0].upper() == acronym[0]:
            aliases.add(" ".join(rest))
        if len(tokens) >= 3:
            display = acronym
    for v in sorted(VENDORS, key=len, reverse=True):
        if name.startswith(v + " ") and len(name) > len(v) + 2:
            aliases.add(name[len(v) + 1:])          # "Autodesk AutoCAD" -> "AutoCAD"
            break
    return display, aliases, generic


def _case_profile(texts) -> Counter:
    counts: Counter = Counter()
    for t in texts:
        counts.update(_TOKEN.findall(t))
    return counts


def _case_mode(alias: str, corpus: Counter) -> str:
    """'ci' (any case), 'cs' (capitalised only) or 'skip', from how the word is used in real postings."""
    if " " in alias or len(alias) > 24:
        return "ci"
    low = alias.lower()
    total = sum(corpus[v] for v in {low, low.capitalize(), low.upper(), alias})
    if total < 20:
        return "ci"
    lower_share = corpus[low] / total
    if low in FORCE_CASE_SENSITIVE:
        return "cs"
    if lower_share > 0.5:
        return "skip"
    return "cs" if lower_share > 0.1 else "ci"


def build_lexicon(sample_size: int = 30_000) -> list[dict]:
    entries: dict[str, dict] = {}
    alias_owner: dict[str, str] = {}

    postings = pd.read_parquet(path("processed") / "postings", columns=["description"])
    corpus = _case_profile(postings["description"].sample(min(sample_size, len(postings)), random_state=0))

    def add(skill_id: str, name: str, group: str, kind: str, onet_element: str, aliases: set[str],
            hot: bool = False, check_case: bool = False):
        e = entries.setdefault(skill_id, {"id": skill_id, "name": name, "group": group, "kind": kind,
                                          "onet_element": onet_element, "hot": hot, "aliases": [], "cs": []})
        for a in sorted(aliases):
            a = a.strip()
            low = a.lower()
            if not a or low in BLOCKED or (len(low) < 2 and low not in KEEP_SHORT):
                continue
            if low in alias_owner and alias_owner[low] != skill_id:
                continue  # first owner wins (curated entries are added first)
            mode = _case_mode(a, corpus) if check_case else "ci"
            if mode == "skip":
                continue
            alias_owner[low] = skill_id
            if mode == "cs":
                cs = a if a[0].isupper() else a.capitalize()
                if cs not in e["cs"]:
                    e["cs"].append(cs)
            elif low not in e["aliases"]:
                e["aliases"].append(low)

    curated = yaml.safe_load((ROOT / "config" / "skills_curated.yaml").read_text())
    alignment = curated.pop("_onet_alignment")
    for group, skills in curated.items():
        kind = "soft" if group == "Soft Skills" else "domain"
        for name, aliases in skills.items():
            # The canonical name is only an alias when it is a multi-word phrase
            # (so "performance" or "scheduling" never match on their own).
            extra = {name} if " " in name else set()
            add(_slug(name), name, group, kind, alignment.get(group, ""), set(aliases) | extra)

    with open_raw("onet/Software Skills.txt") as fh:
        sw = pd.read_csv(fh, sep="\t", dtype=str)
    occ_count = sw.groupby("Workplace Example")["O*NET-SOC Code"].nunique()
    flags = sw.groupby("Workplace Example").agg(
        hot=("Hot Technology", lambda s: (s == "Y").any()),
        demand=("In Demand", lambda s: (s == "Y").any()),
        element=("Element Name", lambda s: s.mode().iat[0]))
    flags["occupations"] = occ_count
    keep = flags[(flags.hot | flags.demand | (flags.occupations >= 3))]
    for example, row in keep.sort_values("occupations", ascending=False).iterrows():
        display, aliases, generic = _onet_aliases(example)
        if generic and _case_mode(display, corpus) != "ci":
            continue  # e.g. "Tax software", "Inventory software": a category, not a skill name
        add(_slug(display), display, row.element, "technology", row.element, aliases,
            hot=bool(row.hot), check_case=True)

    lexicon = [e for e in entries.values() if e["aliases"] or e["cs"]]
    log.info("skill lexicon: %d skills (%d technology, %d domain, %d soft)", len(lexicon),
             sum(e["kind"] == "technology" for e in lexicon), sum(e["kind"] == "domain" for e in lexicon),
             sum(e["kind"] == "soft" for e in lexicon))
    return lexicon


def save_lexicon(lexicon: list[dict]) -> None:
    (path("processed") / "skills_lexicon.json").write_text(json.dumps(lexicon, indent=1))


@lru_cache(maxsize=1)
def load_lexicon() -> tuple[dict, ...]:
    return tuple(json.loads((path("processed") / "skills_lexicon.json").read_text()))


class SkillExtractor:
    """Fast whole-phrase matcher over the lexicon."""

    def __init__(self, lexicon=None):
        lexicon = lexicon or load_lexicon()
        self.by_id = {e["id"]: e for e in lexicon}
        self._ci = ahocorasick.Automaton()
        self._cs = ahocorasick.Automaton()
        for e in lexicon:
            for a in e["aliases"]:
                self._ci.add_word(a, (e["id"], len(a)))
            for a in e["cs"]:
                self._cs.add_word(a, (e["id"], len(a)))
        self._ci.make_automaton()
        if len(self._cs):
            self._cs.make_automaton()

    @staticmethod
    def _bounded(text: str, end: int, length: int) -> bool:
        start = end - length + 1
        before = text[start - 1] if start > 0 else " "
        after = text[end + 1] if end + 1 < len(text) else " "
        return not before.isalnum() and not (after.isalnum() or after in "+#")

    def matches(self, text: str) -> list[str]:
        """Skill ids in text order, leftmost-longest: "financial planning and analysis"
        is one FP&A match, not also a "financial planning" match."""
        if not text:
            return []
        low = text.lower()
        hits = [(end - n + 1, n, sid) for end, (sid, n) in self._ci.iter(low) if self._bounded(low, end, n)]
        if len(self._cs):
            hits += [(end - n + 1, n, sid) for end, (sid, n) in self._cs.iter(text) if self._bounded(text, end, n)]
        hits.sort(key=lambda h: (h[0], -h[1]))
        chosen, last_end = [], -1
        for start, n, sid in hits:
            if start >= last_end:
                chosen.append(sid)
                last_end = start + n
        return chosen

    def counts(self, text: str) -> Counter:
        return Counter(self.matches(text))

    def extract(self, text: str) -> list[str]:
        """Skill ids in order of frequency."""
        return [s for s, _ in self.counts(text).most_common()]

    def names(self, ids) -> list[str]:
        return [self.by_id[i]["name"] for i in ids if i in self.by_id]
