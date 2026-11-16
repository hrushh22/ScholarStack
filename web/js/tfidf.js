// TF-IDF keyword-advice baseline - a port of src/scholarstack/baseline.py.
// Reproduces scikit-learn's TfidfVectorizer: lowercase, token pattern, stop words
// removed before forming 1-2-grams, sublinear tf (1 + log tf), fitted idf, L2 norm.

export class TfidfModel {
  constructor({ vocab, stop_words, token_pattern }) {
    this.idf = new Map(Object.entries(vocab));
    this.stop = new Set(stop_words);
    this.pattern = new RegExp(token_pattern, "g");
  }

  vector(text) {
    const tokens = (text.toLowerCase().match(this.pattern) ?? []).filter((t) => !this.stop.has(t));
    const tf = new Map();
    const add = (term) => {
      if (this.idf.has(term)) tf.set(term, (tf.get(term) ?? 0) + 1);
    };
    tokens.forEach((t, i) => {
      add(t);
      if (i + 1 < tokens.length) add(`${t} ${tokens[i + 1]}`);
    });
    const vec = new Map();
    let norm = 0;
    for (const [term, n] of tf) {
      const w = (1 + Math.log(n)) * this.idf.get(term);
      vec.set(term, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [term, w] of vec) vec.set(term, w / norm);
    return vec;
  }

  static cosine(a, b) {
    let s = 0;
    for (const [term, w] of a) if (b.has(term)) s += w * b.get(term);
    return s;
  }

  /** Generic keyword advice: top JD terms, and (with a resume) the match score and missing terms. */
  advice(jd, resume, topN = 25) {
    const jv = this.vector(jd);
    const keywords = [...jv.entries()].sort((x, y) => y[1] - x[1]).slice(0, topN).map(([t]) => t);
    const out = { keywords };
    if (resume && resume.trim()) {
      const rv = this.vector(resume);
      out.match_score = TfidfModel.cosine(jv, rv);
      const present = new Set(rv.keys());
      out.missing_keywords = keywords.filter((t) => !present.has(t));
      out.matched_keywords = keywords.filter((t) => present.has(t));
    }
    return out;
  }
}
