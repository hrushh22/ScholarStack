// Results page: SVG charts (hover tooltips, a table under each chart) built from data/results.json.
import { esc, pct, prettyCategory } from "./render.js?v=202611161640";

const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
const SYSTEM = { baseline: "Keyword baseline", llm_only: "LLM only (120B)", rag: "ScholarStack (120B)", rag_small: "ScholarStack (20B)",
  rag_small_no_onet: "20B, no O*NET", rag_small_k3: "20B, 3 excerpts", rag_small_k15: "20B, 15 excerpts" };
const tip = () => document.getElementById("tooltip");

function showTip(evt, html) {
  const t = tip();
  t.innerHTML = html;
  t.classList.remove("hidden");
  t.style.left = `${Math.min(evt.clientX + 14, innerWidth - t.offsetWidth - 8)}px`;
  t.style.top = `${Math.min(evt.clientY + 14, innerHeight - t.offsetHeight - 8)}px`;
}
const hideTip = () => tip().classList.add("hidden");

function legend(series) {
  if (series.length < 2) return "";
  return `<div class="chart-legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>`;
}

function table(headers, rows) {
  return `<details class="small" style="margin-top:6px"><summary class="muted">Show the numbers</summary>
    <table class="data"><thead><tr>${headers.map((h, i) => `<th class="${i ? "num" : ""}">${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></details>`;
}

/** Line chart: series [{name, color, points:[{x,y}]}] over shared x values. */
function lineChart(el, { series, xs, yMax = 1, yFmt = (v) => pct(v), xLabel = "", title = "", note = "" }) {
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
    return `<path class="line" pathLength="1" d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>${dots}`;
  }).join("");
  el.innerHTML = `${title ? `<h3>${esc(title)}</h3>` : ""}${note ? `<p class="small muted">${note}</p>` : ""}${legend(series)}
    <div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">
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
  let n = 0;
  const parts = rows.map((row) => {
    const top = y;
    y += groupH + groupGap;
    const bars = row.values.map((v, i) => {
      const by = top + i * (barH + gap);
      const w = Math.max(0, sx(v.value ?? 0) - labelW);
      return `<g class="bar" data-tip="${esc(`<div class="t-title">${esc(row.label)}</div><div class="t-row"><span><span class="dot" style="background:${v.color}"></span> ${esc(v.name)}</span><b>${esc(fmt(v.value))}</b></div>`)}">
        <rect x="${labelW}" y="${by - 1}" width="${W - labelW}" height="${barH + 2}" fill="transparent"/>
        <path style="--i:${n++}" d="M${labelW},${by}h${Math.max(0, w - 4)}a4,4 0 0 1 4,4v${barH - 8}a4,4 0 0 1 -4,4h${-Math.max(0, w - 4)}z" fill="${v.color}"/>
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

const kpi = (label, value, note) =>
  `<div class="tile kpi lit reveal"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="note">${esc(note)}</div></div>`;
const block = (id) => `<div class="card lift reveal" id="${id}"></div>`;
const section = (title, text) => `<div class="section-title reveal"><h2>${title}</h2><p>${text}</p></div>`;

export async function renderDashboard(root, res, api) {
  if (!res) {
    root.innerHTML = '<div class="empty">No results yet. Run the pipeline first (the README has the steps).</div>';
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
      ${kpi("Resumes indexed", (eda.resumes_clean ?? 2480).toLocaleString(), "27,207 section chunks in 24 categories")}
      ${kpi("Job postings processed", (eda.postings_clean ?? 0).toLocaleString(), `out of ${(eda.postings_raw ?? 0).toLocaleString()} raw, cleaned with Dask`)}
      ${kpi("Right category in top 8", dense ? pct(dense.precision_category) : "-", `vs ${tf ? pct(tf.precision_category) : "-"} with TF-IDF`)}
      ${gen.rag ? kpi("Lines backed by real resumes", pct(gen.rag.grounded), `vs ${pct(gen.llm_only?.grounded)} for the LLM alone`)
        : kpi("O*NET occupations", "1,016", "978 linked to LinkedIn demand")}
    </div>

    ${section("Retrieval: finding the right evidence", "240 held-out postings, 10 per category. We count a retrieved excerpt as relevant if it comes from a resume in the posting's category, or if it shares at least one hard skill with the posting.")}
    <div class="grid-2">${block("c-ksweep")}${block("c-ksweep-skill")}</div>
    <div class="grid-2" style="margin-top:16px">${block("c-query")}${block("c-models")}</div>
    <div class="grid-2" style="margin-top:16px">${block("c-bycat")}${block("c-onet")}</div>

    ${section("Generation: grounded or generic?", "One held-out posting per category. Every line each system writes is checked against excerpts from real resumes retrieved for that posting. All systems are checked against the same excerpts.")}
    <div id="gen-area"></div>

    ${section("Scale and speed", "Vector search and embedding speed on a MacBook (Apple silicon, search on CPU).")}
    <div class="grid-2">${block("c-faiss")}${block("c-embed")}</div>

    ${section("Live usage", "Runs and ratings from people using this site.")}
    <div class="card reveal" id="c-usage"><p class="small muted">Loading...</p></div>

    ${section("Exploring the data", "Figures from <code>pipeline/09_eda.py</code> and <code>pipeline/14_report_figures.py</code>.")}
    <div class="fig-gallery" id="gallery"></div>`;

  // --- retrieval
  const sweep = [["section", "all-MiniLM-L6-v2", "MiniLM, section chunks (live)"], ["section", "gte-small", "GTE-small, section"],
    ["section", "bge-small-en-v1_5", "BGE-small, section"], ["fixed", "all-MiniLM-L6-v2", "MiniLM, fixed windows"],
    ["section", "tf-idf", "TF-IDF"]];
  const xs = [...new Set(ret.filter((r) => r.experiment === "model_chunking").map((r) => r.k))].sort((a, b) => a - b);
  for (const [id, metric, title] of [["c-ksweep", "precision_category", "Right category, by k"],
    ["c-ksweep-skill", "precision_skill", "Shares a hard skill, by k"]]) {
    lineChart(document.getElementById(id), {
      title, xs, xLabel: "k (excerpts retrieved)",
      series: sweep.map(([ch, mo, name], i) => ({ name, color: COLORS[i],
        points: xs.map((x) => ({ x, y: at("model_chunking", ch, mo, x)?.[metric] })) })),
    });
  }
  const q = ["raw", "hybrid", "skills"].map((v) => ret.find((r) => r.experiment === "query" && r.query === v && r.k === 8));
  const qLabel = { raw: "Whole job ad", hybrid: "Title, skills, first lines", skills: "Title + extracted skills" };
  barChart(document.getElementById("c-query"), {
    title: "What we embed matters",
    note: "Long ads often open with company boilerplate, and the model only reads the first 256 tokens. Embedding the title plus the skills we extracted works best.",
    rows: q.filter(Boolean).map((r) => ({ label: qLabel[r.query], values: [
      { name: "Right category", value: r.precision_category, color: COLORS[0] },
      { name: "Shares a skill", value: r.precision_skill, color: COLORS[1] }] })),
  });
  const models = [["all-MiniLM-L6-v2", "MiniLM-L6"], ["gte-small", "GTE-small"], ["bge-small-en-v1_5", "BGE-small"], ["tf-idf", "TF-IDF"]];
  barChart(document.getElementById("c-models"), {
    title: "Embedding model and chunking",
    note: "Right category in the top 8. MiniLM is about as good as GTE-small and five times faster, so it's the one running on this site.",
    rows: models.map(([mo, label]) => ({ label, values: [
      { name: "Section chunks", value: at("model_chunking", "section", mo)?.precision_category, color: COLORS[0] },
      { name: "Fixed windows", value: at("model_chunking", "fixed", mo)?.precision_category, color: COLORS[1] }] })),
  });
  const bycat = [...(res.retrieval_by_category ?? [])].sort((a, b) => b["p@8_category"] - a["p@8_category"]);
  barChart(document.getElementById("c-bycat"), {
    title: "By job category", fmt: (v) => pct(v),
    note: "Specific jobs (chef, trainer) are easy. Broad keyword categories like BPO and Automobile are hard because their resumes cover many unrelated jobs.",
    rows: bycat.map((r) => ({ label: prettyCategory(r.category), values: [{ name: "Right category in top 8", value: r["p@8_category"], color: COLORS[0] }] })),
  });
  const o = res.onet_linking;
  document.getElementById("c-onet").innerHTML = `<h3>Matching jobs to O*NET</h3>
    <p class="small muted">We matched every posting title to O*NET's 54k alternate job titles with FAISS. On the live site,
    the posting is matched to whole occupation profiles instead. Here's how often the two agree:</p>
    ${o ? `<div class="tiles" style="grid-template-columns:repeat(3,1fr)">
      <div class="tile lit"><div class="label">Same occupation</div><div class="value">${pct(o.top1_exact)}</div></div>
      <div class="tile lit"><div class="label">In our top 3</div><div class="value">${pct(o.top3_exact)}</div></div>
      <div class="tile lit"><div class="label">Same job family</div><div class="value">${pct(o.top1_same_major_group)}</div></div></div>
      <p class="small">O*NET splits work into 1,016 occupations, so near misses happen a lot (Accountants and Auditors vs Budget
      Analysts, say). Even then the profile still brings in the right kind of skills.</p>` : "<p class='muted'>Not available.</p>"}
    <p class="small"><b>${(eda.postings_mapped_to_category_pct ?? 0)}%</b> of postings fall into one of the 24 resume categories by title,
    and 87% are linked to an O*NET occupation.</p>`;

  // --- generation
  const ga = document.getElementById("gen-area");
  if (!res.generation?.length) {
    ga.innerHTML = `<div class="card empty reveal">Generation results show up here after running <code>pipeline/10_eval_generation.py</code>.</div>`;
  } else {
    const hasJudge = (res.judge ?? []).some((j) => j.criterion === "preferred");
    ga.innerHTML = hasJudge
      ? `<div class="grid-2">${block("c-gen")}${block("c-judge")}</div><div class="grid-2" style="margin-top:16px">${block("c-gen2")}${block("c-calib")}</div>`
      : `<div class="grid-2">${block("c-gen")}${block("c-gen2")}</div><div class="grid-2" style="margin-top:16px">${block("c-calib")}</div>`;
    const order = ["llm_only", "rag_small", "rag", "rag_small_no_onet", "rag_small_k3", "rag_small_k15"].filter((s) => gen[s]);
    const c0 = res.calibration ?? {};
    const showChance = (c0.negative_control_n ?? 0) >= 100;
    const checked = order.filter((s) => gen[s]?.verified_statements).map((s) =>
      `${SYSTEM[s]}: ${gen[s].verified_statements} lines from ${gen[s].verified_jobs} jobs`).join("; ");
    barChart(document.getElementById("c-gen"), {
      title: "Lines backed by real resumes",
      note: `A separate model (a different family than the writer) checks each line against the closest lines from real
        resumes retrieved for the job.${showChance ? " The gray bar is the chance level: how often the same lines pass against an unrelated job's resumes." : ""}
        Checked so far: ${esc(checked)}.`,
      rows: order.filter((s) => gen[s]?.grounded != null).map((s) => ({ label: SYSTEM[s], values: [
        { name: "Backed by this job's resumes", value: gen[s].grounded, color: COLORS[0] },
        ...(showChance ? [{ name: "Chance level", value: gen[s].chance_level ?? null, color: "var(--line-strong)" }] : [])] })),
      fmt: (v) => pct(v),
    });
    barChart(document.getElementById("c-gen2"), {
      title: "Skill coverage",
      rows: ["baseline", ...order].filter((s) => gen[s]).map((s) => ({ label: SYSTEM[s], values: [
        { name: "Skills from the posting", value: gen[s].jd_skill_coverage, color: COLORS[0] },
        { name: "Skills employers ask for most", value: gen[s].market_skill_coverage, color: COLORS[1] }] })),
      fmt: (v) => pct(v),
    });
    const c = res.calibration ?? {};
    document.getElementById("c-calib").innerHTML = `<h3>Can we trust the grounding check?</h3>
      <dl class="kv">
        <dt>Checker</dt><dd>${esc(c.verifier ?? "-")}, a different model family than the writer</dd>
        <dt>Chance level</dt><dd>We also check lines against resumes retrieved for an unrelated job. So far ${pct(c.negative_control_supported)}
          of ${c.negative_control_n ?? 0} lines passed that way, which tells us the checker isn't just saying yes to everything.</dd>
        <dt>Citations</dt><dd>${pct(gen.rag?.citation_rate)} of ScholarStack's lines cite at least one excerpt. Citations to anything we didn't retrieve get removed.</dd>
        <dt>Live check</dt><dd>The quick check on the Generate page (embedding similarity of at least ${c.embedding_tau ?? "-"} with one of the
          retrieved lines) agrees with the checker ${pct(c.embedding_agreement_with_verifier)} of the time. It's a fast hint, not the real measure:
          it's fair to ScholarStack but goes easy on the plain LLM.</dd>
      </dl>
      <p class="small muted" style="margin-top:8px">A sample of the checker's calls is in <code>reports/results/groundedness_audit.csv</code> so we can review them by hand.</p>`;
    const judge = (res.judge ?? []).filter((j) => j.criterion === "preferred");
    const pairs = [...new Set(judge.map((j) => j.pair))];
    if (hasJudge) document.getElementById("c-judge").innerHTML = `<h3>Which one would a career advisor pick?</h3><p class="small muted">A separate model plays
      career advisor and compares two outputs for the same job on how specific, actionable, on-target and realistic they are. We shuffle
      which one it sees first.</p>` + (pairs.length ? pairs.map((p) => {
      const rows = judge.filter((j) => j.pair === p);
      const [a, b] = p.split(" vs ");
      const share = (w) => rows.filter((r) => r.winner === w).length / rows.length;
      return `<p style="margin:14px 0 4px"><b>${esc(SYSTEM[a])}</b> vs <b>${esc(SYSTEM[b])}</b> <span class="muted small">(${rows.length} jobs)</span></p>
        <div class="bars"><div class="bar-row"><span>Picked ${esc(SYSTEM[a])}</span><span class="track"><span class="fill" style="width:${share(a) * 100}%"></span></span><span class="v">${pct(share(a))}</span></div>
        <div class="bar-row"><span>Tie</span><span class="track"><span class="fill" style="background:var(--line-strong);width:${share("tie") * 100}%"></span></span><span class="v">${pct(share("tie"))}</span></div>
        <div class="bar-row"><span>Picked ${esc(SYSTEM[b])}</span><span class="track"><span class="fill" style="background:var(--s2);width:${share(b) * 100}%"></span></span><span class="v">${pct(share(b))}</span></div></div>`;
    }).join("") : "<p class='muted'>No judge results.</p>");
  }

  // --- scale
  const fa = res.faiss ?? [];
  document.getElementById("c-faiss").innerHTML = `<h3>FAISS index benchmark</h3>
    <table class="data"><thead><tr><th>Index</th><th>Data</th><th class="num">Vectors</th><th class="num">Build (s)</th><th class="num">Query (ms)</th><th class="num">Recall@10</th></tr></thead>
    <tbody>${fa.map((r) => `<tr><td>${esc(r.index)}</td><td>${esc(r.corpus)}</td><td class="num">${r.vectors.toLocaleString()}</td>
    <td class="num">${r.build_s.toFixed(2)}</td><td class="num">${r.latency_ms.toFixed(2)}</td><td class="num">${pct(r["recall@10"], 1)}</td></tr>`).join("")}</tbody></table>
    <p class="small muted" style="margin-top:8px">The site serves the same vectors from Postgres with pgvector (HNSW). When we load the
    database we check that its top 8 matches FAISS exact search.</p>`;
  const et = res.embedding_timing ?? [];
  document.getElementById("c-embed").innerHTML = `<h3>Embedding in batches</h3>
    <table class="data"><thead><tr><th>Data</th><th>Model</th><th class="num">Texts</th><th class="num">Seconds</th><th class="num">Per second</th></tr></thead>
    <tbody>${et.map((r) => `<tr><td>${esc(r.what)}</td><td>${esc(r.model)}</td><td class="num">${r.n.toLocaleString()}</td><td class="num">${r.seconds.toFixed(1)}</td><td class="num">${Math.round(r.per_sec)}</td></tr>`).join("")}
    <tr><td>All job postings</td><td>all-MiniLM-L6-v2</td><td class="num">106,291</td><td class="num">577</td><td class="num">184</td></tr></tbody></table>`;

  // --- gallery
  const figs = [["eda_resume_categories", "Resumes per category. The biggest class is 5.5 times the smallest."],
    ["eda_lengths", "How long resumes and job ads are. We used this to pick chunk sizes."],
    ["eda_top_skills", "Skills in resumes compared with what postings ask for, in six categories."],
    ["eda_embedding_space", "Resume embeddings group by occupation (10-NN purity 0.65)."],
    ["eda_postings_overview", "Industries, work types and experience levels in the postings."],
    ["eda_postings_missingness", "Missing values in the raw postings. Salary is missing 71% of the time."],
    ["eda_category_coverage", "Postings in each resume category next to the resumes we have."],
    ["results_faiss_benchmark", "FAISS: exact search vs approximate indexes."]];
  document.getElementById("gallery").innerHTML = figs.map(([f, cap]) =>
    `<figure class="reveal"><img src="figures/${f}.png" alt="${esc(cap)}" loading="lazy"><figcaption>${esc(cap)}</figcaption></figure>`).join("");

  // --- live usage
  const u = document.getElementById("c-usage");
  if (!api.isConfigured()) {
    u.innerHTML = "<p class='small muted'>The backend isn't set up on this copy of the site.</p>";
  } else {
    try {
      const s = await api.usageStats();
      const avg = s.avg_rating ?? {};
      u.innerHTML = `<div class="tiles" style="margin:0">
        <div class="tile lit"><div class="label">Reference resumes generated</div><div class="value">${s.runs}</div></div>
        <div class="tile lit"><div class="label">Ratings</div><div class="value">${s.ratings}</div></div>
        <div class="tile lit"><div class="label">Average rating, ScholarStack</div><div class="value">${avg.rag ?? "-"}</div><div class="note">out of 5</div></div>
        <div class="tile lit"><div class="label">Average rating, baselines</div><div class="value">${avg.baseline ?? "-"} / ${avg.llm_only ?? "-"}</div><div class="note">keywords / LLM alone</div></div></div>`;
    } catch (e) {
      u.innerHTML = `<p class="small muted">Couldn't load usage numbers: ${esc(e.message)}</p>`;
    }
  }
}
