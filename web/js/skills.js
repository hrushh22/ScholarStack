// Rule-based skill extraction — a port of src/scholarstack/skills.py (SkillExtractor).
// Same lexicon (web/data/skills_lexicon.json), same whole-phrase + boundary rules,
// same case-sensitive handling for ambiguous product names.

const ALNUM = /[\p{L}\p{N}]/u;
const isAlnum = (ch) => ALNUM.test(ch);

function bounded(text, start, len) {
  const before = start > 0 ? text[start - 1] : " ";
  const after = start + len < text.length ? text[start + len] : " ";
  return !isAlnum(before) && !(isAlnum(after) || after === "+" || after === "#");
}

export class SkillExtractor {
  constructor(lexicon) {
    this.byId = new Map(lexicon.map((e) => [e.id, e]));
    this.ci = [];
    this.cs = [];
    for (const e of lexicon) {
      for (const a of e.aliases) this.ci.push([a, e.id]);
      for (const a of e.cs) this.cs.push([a, e.id]);
    }
  }

  #scan(haystack, aliases, hits) {
    for (const [alias, id] of aliases) {
      let i = haystack.indexOf(alias);
      while (i !== -1) {
        if (bounded(haystack, i, alias.length)) hits.push([i, alias.length, id]);
        i = haystack.indexOf(alias, i + 1);
      }
    }
  }

  /** Skill ids in text order, leftmost-longest (mirrors Python's SkillExtractor.matches). */
  matches(text) {
    if (!text) return [];
    const hits = [];
    this.#scan(text.toLowerCase(), this.ci, hits);
    this.#scan(text, this.cs, hits);
    hits.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
    const chosen = [];
    let lastEnd = -1;
    for (const [start, len, id] of hits) {
      if (start >= lastEnd) {
        chosen.push(id);
        lastEnd = start + len;
      }
    }
    return chosen;
  }

  /** Map of skill id -> count, in first-seen order (mirrors Python's Counter). */
  counts(text) {
    const counts = new Map();
    for (const id of this.matches(text)) counts.set(id, (counts.get(id) ?? 0) + 1);
    return counts;
  }

  /** Skill ids ordered by frequency (stable for ties). */
  extract(text) {
    return [...this.counts(text).entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id);
  }

  /** Hard (non-soft) skills of a job description, as Python's query.jd_skills. */
  jdSkills(text, top = 25) {
    return this.extract(text).filter((id) => this.byId.get(id).kind !== "soft").slice(0, top);
  }

  names(ids) {
    return ids.filter((id) => this.byId.has(id)).map((id) => this.byId.get(id).name);
  }
}

/** Title = first line when it is short, as Python's query.split_title. */
export function splitTitle(jd) {
  const lines = jd.trim().split("\n").map((l) => l.trim()).filter(Boolean);
  if (!lines.length) return ["", ""];
  if (lines[0].split(/\s+/).length <= 12) return [lines[0], lines.slice(1).join("\n")];
  return ["", jd];
}

/** Retrieval query construction, identical to Python's query.build_query. */
export function buildQuery(jd, ex, variant, titleOverride) {
  const [t, body] = splitTitle(jd);
  const title = titleOverride || t;
  const skills = ex.names(ex.jdSkills(jd)).join(", ");
  if (variant === "raw") return jd;
  if (variant === "skills") return `${title}. Skills: ${skills}`;
  const lead = body.split(/\s+/).filter(Boolean).slice(0, 80).join(" ");
  return `${title}. Skills: ${skills}. ${lead}`;
}
