// Evaluation dashboard: native SVG charts (hover tooltips + table view) from data/results.json.
import { esc, pct, prettyCategory } from "./render.js";

const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "#e87ba4"];
const SYSTEM = { baseline: "TF-IDF keywords", llm_only: "LLM only", rag: "RAG (ours)", rag_no_onet: "RAG without O*NET",
  rag_small: "RAG, 8B model", rag_k3: "RAG, k = 3", rag_k15: "RAG, k = 15" };
const tip = () => document.getElementById("tooltip");

function showTip(evt, html) {
  const t = tip();
  t.innerHTML = html;
  t.classList.remove("hidden");
  const x = Math.min(evt.clientX + 14, innerWidth - t.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, innerHeight - t.offsetHeight - 8);
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
const hideTip = () => tip().classList.add("hidden");

function legend(series) {
  if (series.length < 2) return "";
  return `<div class="chart-legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>`;
}

function table(headers, rows) {
  return `<details class="small" style="margin-top:6px"><summary class="muted">Show table</summary>
    <table class="data"><thead><tr>${headers.map((h, i) => `<th class="${i ? "num" : ""}">${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></details>`;
}

/** Line chart: series [{name, color, points:[{x,y}]}] on shared x values. */
function lineChart(el, { series, xs, yMax = 1, yFmt = (v) => pct(v), xLabel = "", title = "" }) {
  const W = 560, H = 250, m = { l: 44, r: 14, t: 12, b: 36 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const sx = (x) => m.l + ((x - xMin) / (xMax - xMin || 1)) * iw;
  const sy = (y) => m.t + ih - (y / yMax) * ih;
  const yt = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  const grid = yt.map((v) => `<g class="grid"><line x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/></g>
    <text x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${yFmt(v)}</text>`).join("");
  const xt = xs.map((x) => `<text x="${sx(x)}" y="${H - m.b + 16}" text-anchor="middle">${x}</text>`).join("");
  const paths = series.map((s) => {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    const dots = s.points.map((p) => `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`).join("");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>${dots}`;
  }).join("");
  el.innerHTML = `${title ? `<h3>${esc(title)}</h3>` : ""}${legend(series)}<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">
    ${grid}<line x1="${m.l}" x2="${W - m.r}" y1="${sy(0)}" y2="${sy(0)}" stroke="var(--line-strong)"/>${xt}
    <text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">${esc(xLabel)}</text>
    <line class="xhair" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" stroke="var(--line-strong)" visibility="hidden"/>
    ${paths}<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/></svg></div>
    ${table(["x", ...series.map((s) => s.name)], xs.map((x) => [x, ...series.map((s) => yFmt(s.points.find((p) => p.x === x)?.y ?? NaN))]))}`;
  const svg = el.querySelector("svg");
  const hair = svg.querySelector(".xhair");
  svg.querySelector("rect").addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const x = xs.reduce((a, b) => (Math.abs(sx(b) - px) < Math.abs(sx(a) - px) ? b : a));
    hair.setAttribute("x1", sx(x));
    hair.setAttribute("x2", sx(x));
    hair.setAttribute("visibility", "visible");
    const rows = series.map((s) => [s, s.points.find((p) => p.x === x)?.y]).sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0));
    showTip(e, `<div class="t-title">${esc(xLabel)} = ${x}</div>${rows.map(([s, y]) =>
      `<div class="t-row"><span><span class="dot" style="background:${s.color}"></span> ${esc(s.name)}</span><b>${yFmt(y)}</b></div>`).join("")}`);
  });
  svg.querySelector("rect").addEventListener("mouseleave", () => { hair.setAttribute("visibility", "hidden"); hideTip(); });
}

/** Horizontal bars: rows [{label, values:[{name, value, color}]}]. */
function barChart(el, { rows, max = 1, fmt = (v) => pct(v, 1), title = "", note = "" }) {
  const series = rows[0]?.values.map((v) => ({ name: v.name, color: v.color })) ?? [];
  const per = series.length, barH = per > 1 ? 12 : 16, gap = 3, groupGap = 14;
  const labelW = 170, W = 560, m = { r: 52, t: 4 };
  const groupH = per * barH + (per - 1) * gap;
  const H = m.t + rows.length * (groupH + groupGap);
  const sx = (v) => labelW + (Math.max(0, v) / max) * (W - labelW - m.r);
  let y = m.t;
  const parts = rows.map((row) => {
    const top = y;
    y += groupH + groupGap;
    const bars = row.values.map((v, i) => {
      const by = top + i * (barH + gap);
      const w = Math.max(0, sx(v.value ?? 0) - labelW);
      return `<g class="bar" data-tip="${esc(`<div class="t-title">${esc(row.label)}</div><div class="t-row"><span><span class="dot" style="background:${v.color}"></span> ${esc(v.name)}</span><b>${esc(fmt(v.value))}</b></div>`)}">
        <rect x="${labelW}" y="${by - 1}" width="${W - labelW}" height="${barH + 2}" fill="transparent"/>
        <path d="M${labelW},${by}h${Math.max(0, w - 4)}a4,4 0 0 1 4,4v${barH - 8}a4,4 0 0 1 -4,4h${-Math.max(0, w - 4)}z" fill="${v.color}"/>
        <text x="${labelW + w + 6}" y="${by + barH / 2 + 4}">${esc(fmt(v.value))}</text></g>`;
    }).join("");
    return `<text x="${labelW - 10}" y="${top + groupH / 2 + 4}" text-anchor="end">${esc(row.label)}</text>${bars}`;
  }).join("");
  el.innerHTML = `${title ? `<h3>${esc(title)}</h3>` : ""}${note ? `<p class="small muted">${note}</p>` : ""}${legend(series)}
    <div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">
    <line x1="${labelW}" x2="${labelW}" y1="0" y2="${H - groupGap + 4}" stroke="var(--line-strong)"/>${parts}</svg></div>
    ${table(["", ...series.map((s) => s.name)], rows.map((r) => [r.label, ...r.values.map((v) => fmt(v.value))]))}`;
  el.querySelectorAll(".bar").forEach((g) => {
    g.addEventListener("mousemove", (e) => showTip(e, g.dataset.tip));
    g.addEventListener("mouseleave", hideTip);
  });
}

function kpi(label, value, note) {
  return `<div class="tile kpi"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="note">${esc(note)}</div></div>`;
}

const block = (id, cls = "card") => `<div class="${cls}" id="${id}"></div>`;

export async function renderDashboard(root, res, api) {
  if (!res) {
    root.innerHTML = '<div class="empty">Evaluation results are not available yet. Run the pipeline (see README).</div>';
    return;
  }
  const ret = res.retrieval ?? [];
  const at = (exp, chunking, model, k = 8, query) => ret.find((r) => r.experiment === exp && r.chunking === chunking &&
    r.model === model && r.k === k && (!query || r.query === query));
  const dense = at("model_chunking", "section", "all-MiniLM-L6-v2");
  const tf = at("model_chunking", "section", "tf-idf");
  const gen = Object.fromEntries((res.generation ?? []).map((g) => [g.system, g]));
  const eda = res.eda ?? {};

  root.innerHTML = `
    <div class="kpis">
      ${kpi("Resumes indexed", (eda.resumes_clean ?? 2480).toLocaleString(), "27,207 section chunks, 24 categories")}
      ${kpi("Job postings processed", (eda.postings_clean ?? 0).toLocaleString(), `of ${(eda.postings_raw ?? 0).toLocaleString()} raw (Dask)`)}
      ${kpi("Retrieval precision@8", dense ? pct(dense.precision_category) : "–", `vs ${tf ? pct(tf.precision_category) : "–"} for TF-IDF`)}
      ${gen.rag ? kpi("Grounded statements", pct(gen.rag.grounded), `vs ${pct(gen.llm_only?.grounded)} without retrieval`)
        : kpi("O*NET occupations", "1,016", "978 linked to LinkedIn demand")}
    </div>

    <div class="section-title"><h2>Retrieval: finding the right evidence</h2>
      <p>240 held-out job descriptions (10 per category). Relevance proxies: the retrieved excerpt comes from a resume in the
      job's category, or shares at least one hard skill with it.</p></div>
    <div class="grid-2">${block("c-ksweep")}${block("c-ksweep-skill")}</div>
    <div class="grid-2" style="margin-top:16px">${block("c-query")}${block("c-models")}</div>
    <div class="grid-2" style="margin-top:16px">${block("c-bycat")}${block("c-onet")}</div>

    <div class="section-title"><h2>Generation: grounded vs. generic</h2>
      <p>One held-out job description per category. Grounded = the statement has cosine ≥ ${res.config?.groundedness_threshold ?? 0.55}
      with a real resume excerpt retrieved for that job.</p></div>
    <div id="gen-area"></div>

    <div class="section-title"><h2>Scale &amp; speed</h2><p>Vector search and embedding throughput (Apple M-series laptop, CPU search).</p></div>
    <div class="grid-2">${block("c-faiss")}${block("c-embed")}</div>

    <div class="section-title"><h2>Live usage</h2><p>Runs and ratings collected by this website.</p></div>
    <div class="card" id="c-usage"><p class="small muted">Loading…</p></div>

    <div class="section-title"><h2>Exploratory data analysis</h2><p>Figures generated by <code>pipeline/09_eda.py</code> and <code>pipeline/14_report_figures.py</code>.</p></div>
    <div class="fig-gallery" id="gallery"></div>`;

  // --- retrieval charts
  const sweep = [["section", "all-MiniLM-L6-v2", "MiniLM, section (served)"], ["section", "gte-small", "GTE-small, section"],
    ["section", "bge-small-en-v1_5", "BGE-small, section"], ["fixed", "all-MiniLM-L6-v2", "MiniLM, fixed windows"],
    ["section", "tf-idf", "TF-IDF baseline"]];
  const xs = [...new Set(ret.filter((r) => r.experiment === "model_chunking").map((r) => r.k))].sort((a, b) => a - b);
  for (const [id, metric, title] of [["c-ksweep", "precision_category", "Category precision@k"],
    ["c-ksweep-skill", "precision_skill", "Skill-overlap precision@k"]]) {
    lineChart(document.getElementById(id), {
      title, xs, xLabel: "k",
      series: sweep.map(([ch, mo, name], i) => ({ name, color: COLORS[i],
        points: xs.map((x) => ({ x, y: at("model_chunking", ch, mo, x)?.[metric] })) })),
    });
  }
  const q = ["raw", "hybrid", "skills"].map((v) => ret.find((r) => r.experiment === "query" && r.query === v && r.k === 8));
  const qLabel = { raw: "Raw job description", hybrid: "Title + skills + lead text", skills: "Title + extracted skills" };
  barChart(document.getElementById("c-query"), {
    title: "Query construction (P@8)", note: "Embedding the extracted skills avoids truncating long ads that start with boilerplate.",
    rows: q.filter(Boolean).map((r) => ({ label: qLabel[r.query], values: [
      { name: "Category", value: r.precision_category, color: COLORS[0] },
      { name: "Skill overlap", value: r.precision_skill, color: COLORS[1] }] })),
  });
  const models = [["all-MiniLM-L6-v2", "MiniLM-L6"], ["gte-small", "GTE-small"], ["bge-small-en-v1_5", "BGE-small"], ["tf-idf", "TF-IDF"]];
  barChart(document.getElementById("c-models"), {
    title: "Embedding model × chunking (category P@8)",
    rows: models.map(([mo, label]) => ({ label, values: [
      { name: "Section chunks", value: at("model_chunking", "section", mo)?.precision_category, color: COLORS[0] },
      { name: "Fixed windows", value: at("model_chunking", "fixed", mo)?.precision_category, color: COLORS[1] }] })),
  });
  const bycat = [...(res.retrieval_by_category ?? [])].sort((a, b) => b["p@8_category"] - a["p@8_category"]);
  barChart(document.getElementById("c-bycat"), {
    title: "Category P@8 by job category", fmt: (v) => pct(v),
    rows: bycat.map((r) => ({ label: prettyCategory(r.category), values: [{ name: "P@8", value: r["p@8_category"], color: COLORS[0] }] })),
  });
  const o = res.onet_linking;
  document.getElementById("c-onet").innerHTML = `<h3>Linking jobs to O*NET</h3>
    <p class="small muted">Each posting title was matched to O*NET's 54k alternate titles with FAISS; a job description is
    matched live to occupation profiles. Agreement between the two on the evaluation set:</p>
    ${o ? `<div class="tiles" style="grid-template-columns:repeat(3,1fr)">
      <div class="tile"><div class="label">Top-1 exact</div><div class="value">${pct(o.top1_exact)}</div></div>
      <div class="tile"><div class="label">Top-3 exact</div><div class="value">${pct(o.top3_exact)}</div></div>
      <div class="tile"><div class="label">Same major group</div><div class="value">${pct(o.top1_same_major_group)}</div></div></div>
      <p class="small">O*NET distinguishes 1,016 occupations, so near-misses (e.g. “Accountants and Auditors” vs “Budget Analysts”)
      are common; the profile still supplies relevant skills and demand.</p>` : "<p class='muted'>Not available.</p>"}
    <p class="small"><b>${(eda.postings_mapped_to_category_pct ?? 0)}%</b> of postings map to one of the 24 resume categories by title;
    87% of postings are linked to an O*NET occupation.</p>`;

  // --- generation
  const ga = document.getElementById("gen-area");
  if (!res.generation?.length) {
    ga.innerHTML = `<div class="card empty">Generation results appear here after running <code>pipeline/10_eval_generation.py</code>.</div>`;
  } else {
    ga.innerHTML = `<div class="grid-2">${block("c-gen")}${block("c-judge")}</div><div class="grid-2" style="margin-top:16px">${block("c-gen2")}${block("c-tau")}</div>`;
    const order = ["baseline", "llm_only", "rag_small", "rag_no_onet", "rag_k3", "rag", "rag_k15"].filter((s) => gen[s]);
    barChart(document.getElementById("c-gen"), {
      title: "Grounded statements", fmt: (v) => (v == null ? "n/a" : pct(v)),
      rows: order.filter((s) => s !== "baseline").map((s) => ({ label: SYSTEM[s], values: [{ name: "Grounded", value: gen[s].grounded, color: s === "rag" ? COLORS[0] : "#86b6ef" }] })),
    });
    barChart(document.getElementById("c-gen2"), {
      title: "Skill coverage",
      rows: order.map((s) => ({ label: SYSTEM[s], values: [
        { name: "JD skills", value: gen[s].jd_skill_coverage, color: COLORS[0] },
        { name: "Market (O*NET-linked) skills", value: gen[s].market_skill_coverage, color: COLORS[1] }] })),
    });
    const curve = res.groundedness_curve ?? [];
    const taus = Object.keys(curve[0] ?? {}).filter((k) => k.startsWith("grounded@")).map((k) => Number(k.split("@")[1]));
    lineChart(document.getElementById("c-tau"), {
      title: "Groundedness vs. support threshold", xs: taus, xLabel: "cosine threshold",
      series: ["rag", "rag_no_onet", "rag_small", "llm_only"].map((s, i) => [s, i]).filter(([s]) => curve.find((c) => c.system === s))
        .map(([s, i]) => ({ name: SYSTEM[s], color: COLORS[i], points: taus.map((t) => ({ x: t, y: curve.find((c) => c.system === s)[`grounded@${t}`] })) })),
    });
    const judge = (res.judge ?? []).filter((j) => j.criterion === "preferred");
    const pairs = [...new Set(judge.map((j) => j.pair))];
    document.getElementById("c-judge").innerHTML = `<h3>LLM-judge preference</h3><p class="small muted">Pairwise comparison by a separate
      LLM call acting as a career advisor; A/B order randomised to cancel position bias.</p>` + (pairs.length ? pairs.map((p) => {
      const rows = judge.filter((j) => j.pair === p);
      const [a, b] = p.split(" vs ");
      const share = (w) => rows.filter((r) => r.winner === w).length / rows.length;
      return `<p style="margin:14px 0 4px"><b>${esc(SYSTEM[a])}</b> vs <b>${esc(SYSTEM[b])}</b> <span class="muted small">(${rows.length} jobs)</span></p>
        <div class="bars"><div class="bar-row"><span>${esc(SYSTEM[a])} preferred</span><span class="track"><span class="fill" style="display:block;width:${share(a) * 100}%"></span></span><span class="v">${pct(share(a))}</span></div>
        <div class="bar-row"><span>Tie</span><span class="track"><span class="fill" style="display:block;background:var(--line-strong);width:${share("tie") * 100}%"></span></span><span class="v">${pct(share("tie"))}</span></div>
        <div class="bar-row"><span>${esc(SYSTEM[b])} preferred</span><span class="track"><span class="fill" style="display:block;background:var(--s2);width:${share(b) * 100}%"></span></span><span class="v">${pct(share(b))}</span></div></div>`;
    }).join("") : "<p class='muted'>No judge results.</p>");
  }

  // --- scale
  const fa = res.faiss ?? [];
  document.getElementById("c-faiss").innerHTML = `<h3>FAISS index benchmark</h3>
    <table class="data"><thead><tr><th>Index</th><th>Corpus</th><th class="num">Vectors</th><th class="num">Build (s)</th><th class="num">Latency (ms)</th><th class="num">Recall@10</th></tr></thead>
    <tbody>${fa.map((r) => `<tr><td>${esc(r.index)}</td><td>${esc(r.corpus)}</td><td class="num">${r.vectors.toLocaleString()}</td>
    <td class="num">${r.build_s.toFixed(2)}</td><td class="num">${r.latency_ms.toFixed(2)}</td><td class="num">${pct(r["recall@10"], 1)}</td></tr>`).join("")}</tbody></table>
    <p class="small muted" style="margin-top:8px">The live site serves the same vectors from Postgres/pgvector (HNSW); the loader checks that its top-8 matches FAISS exact search.</p>`;
  const et = res.embedding_timing ?? [];
  document.getElementById("c-embed").innerHTML = `<h3>Batched embedding throughput</h3>
    <table class="data"><thead><tr><th>Corpus</th><th>Model</th><th class="num">Texts</th><th class="num">Seconds</th><th class="num">Texts/s</th></tr></thead>
    <tbody>${et.map((r) => `<tr><td>${esc(r.what)}</td><td>${esc(r.model)}</td><td class="num">${r.n.toLocaleString()}</td><td class="num">${r.seconds.toFixed(1)}</td><td class="num">${Math.round(r.per_sec)}</td></tr>`).join("")}
    <tr><td>Job postings (all)</td><td>all-MiniLM-L6-v2</td><td class="num">106,291</td><td class="num">577</td><td class="num">184</td></tr></tbody></table>`;

  // --- gallery
  const figs = [["eda_resume_categories", "Resumes per category: imbalance of 5.5x between the largest and smallest class."],
    ["eda_lengths", "Resume and job-description length distributions, used to choose chunk sizes."],
    ["eda_top_skills", "Skills in resumes vs. job postings for six categories."],
    ["eda_embedding_space", "Resume embeddings cluster by occupation (10-NN purity 0.65)."],
    ["eda_postings_overview", "Industries, work types and experience levels of the postings."],
    ["eda_postings_missingness", "Missing values in the raw postings."],
    ["eda_category_coverage", "Postings mapped to each resume category vs. resumes available."],
    ["results_faiss_benchmark", "FAISS: exact vs. approximate indexes."]];
  document.getElementById("gallery").innerHTML = figs.map(([f, cap]) =>
    `<figure><img src="figures/${f}.png" alt="${esc(cap)}" loading="lazy"><figcaption>${esc(cap)}</figcaption></figure>`).join("");

  // --- live usage
  const u = document.getElementById("c-usage");
  if (!api.isConfigured()) {
    u.innerHTML = "<p class='small muted'>Backend not configured.</p>";
  } else {
    try {
      const s = await api.usageStats();
      const avg = s.avg_rating ?? {};
      u.innerHTML = `<div class="tiles" style="margin:0">
        <div class="tile"><div class="label">Reference resumes generated</div><div class="value">${s.runs}</div></div>
        <div class="tile"><div class="label">Ratings submitted</div><div class="value">${s.ratings}</div></div>
        <div class="tile"><div class="label">Avg rating, RAG</div><div class="value">${avg.rag ?? "–"}</div><div class="note">out of 5</div></div>
        <div class="tile"><div class="label">Avg rating, baselines</div><div class="value">${avg.baseline ?? "–"} / ${avg.llm_only ?? "–"}</div><div class="note">TF-IDF / LLM only</div></div></div>`;
    } catch (e) {
      u.innerHTML = `<p class="small muted">Usage stats unavailable: ${esc(e.message)}</p>`;
    }
  }
}
