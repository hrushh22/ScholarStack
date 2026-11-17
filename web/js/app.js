// ScholarStack web app: the Generate page, routing, theme and the small animations.
import { CONFIG } from "./config.js?v=202611161640";
import * as api from "./api.js?v=202611161640";
import { renderDashboard } from "./dashboard.js?v=202611161640";
import { embed, loadEmbedder } from "./embed.js?v=202611161640";
import { renderMethod } from "./method.js?v=202611161640";
import { coverage, groundedness, outputText, statements } from "./metrics.js?v=202611161640";
import * as R from "./render.js?v=202611161640";
import { buildQuery, SkillExtractor } from "./skills.js?v=202611161640";
import { TfidfModel } from "./tfidf.js?v=202611161640";

const $ = (s) => document.querySelector(s);
const state = { ex: null, tfidf: null, samples: null, results: null, run: null };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------------ helpers
async function loadJSON(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`Could not load ${p}`);
  return r.json();
}
const debounce = (fn, ms) => {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
};
const words = (s) => (s.trim() ? s.trim().split(/\s+/).length : 0);
const secs = (ms) => `${(ms / 1000).toFixed(1)}s`;
function storage(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch { /* private mode */ }
  return null;
}

/** Animate a number from 0 to `to` inside `el`. */
function countUp(el, to, { suffix = "", decimals = 0, duration = 1100 } = {}) {
  if (reduceMotion || !Number.isFinite(to)) {
    el.textContent = Number.isFinite(to) ? `${to.toLocaleString(undefined, { maximumFractionDigits: decimals })}${suffix}` : "-";
    return;
  }
  const start = performance.now();
  const step = (now) => {
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = `${(to * eased).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/** Fade elements in as they scroll into view. */
const revealer = "IntersectionObserver" in window
  ? new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("in");
        revealer.unobserve(e.target);
      }
    }
  }, { threshold: 0.12 })
  : null;
export function observeReveals(root = document) {
  root.querySelectorAll(".reveal:not(.in)").forEach((el) => (revealer ? revealer.observe(el) : el.classList.add("in")));
}

// ------------------------------------------------------------------ theme, routing, hero
function initTheme() {
  const saved = storage("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    storage("theme", document.documentElement.dataset.theme);
  });
}

function showView(name) {
  const view = ["generate", "evaluation", "method"].includes(name) ? name : "generate";
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", t.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  observeReveals();
}

function showPanel(name) {
  document.querySelectorAll(".subtab").forEach((t) => t.setAttribute("aria-selected", t.dataset.panel === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("hidden", p.id !== `panel-${name}`));
}

function startTypewriter(roles) {
  const el = $("#typed");
  if (!el || !roles.length) return;
  if (reduceMotion) {
    el.textContent = roles[0];
    return;
  }
  let i = 0;
  let text = roles[0];
  let deleting = true;
  const tick = () => {
    if (deleting) {
      text = text.slice(0, -1);
      if (!text) {
        deleting = false;
        i += 1;
      }
    } else {
      const next = roles[i % roles.length];
      text = next.slice(0, text.length + 1);
      if (text === next) {
        deleting = true;
        el.textContent = text;
        return setTimeout(tick, 1900);
      }
    }
    el.textContent = text || " ";
    return setTimeout(tick, deleting ? 35 : 70);
  };
  setTimeout(tick, 2200);
}

function fillTicker(names) {
  const pick = (n, seed) => {
    const out = [];
    for (let j = 0; j < n; j++) out.push(names[(seed * 31 + j * 17) % names.length]);
    return out;
  };
  for (const [id, seed] of [["#ticker-a", 3], ["#ticker-b", 11]]) {
    const items = pick(28, seed).map((n) => `<span>${R.esc(n)}</span>`).join("");
    $(id).innerHTML = items + items; // doubled so the loop is seamless
  }
}

function heroCounters() {
  document.querySelectorAll(".hero-stats b[data-count]").forEach((b, i) => {
    setTimeout(() => countUp(b, Number(b.dataset.count), { duration: 1400 }), 150 * i);
  });
}

// ------------------------------------------------------------------ pipeline UI
const STEPS = ["skills", "embed", "retrieve", "onet", "generate", "verify"];
function setStep(name, status, note) {
  const el = document.querySelector(`.step[data-step="${name}"]`);
  el.classList.remove("running", "done", "failed");
  if (status) el.classList.add(status);
  if (note) el.querySelector("span").textContent = note;
  const done = STEPS.filter((s) => document.querySelector(`.step[data-step="${s}"]`).classList.contains("done")).length;
  $("#rail").style.width = `${(92 * done) / STEPS.length}%`;
}
function resetSteps() {
  const defaults = { skills: "rule-based", embed: "MiniLM", retrieve: "pgvector", onet: "occupation", generate: "LLM", verify: "grounding" };
  for (const [k, v] of Object.entries(defaults)) setStep(k, null, v);
  document.querySelectorAll(".tile").forEach((t) => t.classList.remove("lit"));
}
function setTile(id, value, meter) {
  const el = $(`#t-${id}`);
  el.closest(".tile").classList.add("lit");
  if (typeof value === "number") countUp(el, Math.round(100 * value), { suffix: "%", duration: 900 });
  else el.textContent = value;
  if (meter !== undefined && $(`#m-${id}`)) $(`#m-${id}`).style.width = `${Math.round(100 * (meter ?? 0))}%`;
}
function clearTiles() {
  for (const id of ["grounded", "coverage", "evidence", "time"]) $(`#t-${id}`).textContent = "-";
  for (const id of ["grounded", "coverage"]) $(`#m-${id}`).style.width = "0";
}
const status = (msg) => { $("#status").textContent = msg; };

// ------------------------------------------------------------------ inputs
function fullJd() {
  const title = $("#job-title").value.trim();
  const desc = $("#jd").value.trim();
  return { title, jd: title ? `${title}\n\n${desc}` : desc };
}

function updateSkills() {
  if (!state.ex) return;
  const { jd } = fullJd();
  const ids = state.ex.jdSkills(jd);
  $("#jd-skills").innerHTML = ids.map((id, i) =>
    `<span class="chip pop" style="--i:${i}">${R.esc(state.ex.byId.get(id).name)}</span>`).join("")
    || '<span class="small muted">Paste a description and the skills show up here.</span>';
  $("#skill-count").textContent = ids.length ? `(${ids.length})` : "";
  $("#jd-count").textContent = `${words($("#jd").value)} words`;
}

function populate() {
  const sel = $("#sample-jd");
  for (const [i, s] of state.samples.jds.entries()) {
    const at = s.company ? ` at ${s.company}` : "";
    sel.insertAdjacentHTML("beforeend", `<option value="${i}">${R.esc(s.label)}${R.esc(at)}</option>`);
  }
  sel.addEventListener("change", () => {
    const s = state.samples.jds[sel.value];
    if (!s) return;
    $("#job-title").value = s.title;
    $("#jd").value = s.description;
    updateSkills();
  });
  const rs = $("#sample-resume");
  for (const r of state.samples.resumes) rs.insertAdjacentHTML("beforeend", `<option value="${R.esc(r.id)}">${R.esc(r.label)}</option>`);
  rs.addEventListener("change", () => {
    const r = state.samples.resumes.find((x) => x.id === rs.value);
    $("#resume").value = r ? r.text : "";
  });
  const cat = $("#category");
  for (const [k, v] of Object.entries(state.results?.categories ?? {})) cat.insertAdjacentHTML("beforeend", `<option value="${R.esc(k)}">${R.esc(v)}</option>`);
  $("#k").addEventListener("input", () => { $("#k-val").textContent = $("#k").value; });
}

function warmEmbedder() {
  const bar = $("#model-progress");
  const files = new Map();
  loadEmbedder((p) => {
    if (p.status === "progress" && p.total) {
      files.set(p.file, [p.loaded, p.total]);
      const [l, t] = [...files.values()].reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
      bar.classList.remove("hidden");
      bar.firstElementChild.style.width = `${(100 * l) / t}%`;
      status(`Downloading the embedding model (${Math.round(t / 1e6)} MB, only the first time)... ${Math.round((100 * l) / t)}%`);
    }
  }).then(() => {
    bar.classList.add("hidden");
    status("Ready.");
  }).catch((e) => status(`Couldn't load the embedding model: ${e.message}`));
}

// ------------------------------------------------------------------ generation
async function runGenerate() {
  const { title, jd } = fullJd();
  const resume = $("#resume").value.trim();
  if (words(jd) < 30) {
    status("That looks too short. Paste the whole job description (30+ words).");
    return;
  }
  const k = Number($("#k").value);
  const category = $("#category").value || null;
  const compareLlm = $("#compare-llm").checked;
  const ex = state.ex;
  const btn = $("#go");
  btn.disabled = true;
  btn.textContent = "Working on it...";
  resetSteps();
  clearTiles();
  const t0 = performance.now();
  const run = (state.run = { title, jd, resume, k, category });

  try {
    // 1. skills, query and the keyword baseline all run in the browser
    setStep("skills", "running");
    run.jdIds = ex.jdSkills(jd);
    run.query = buildQuery(jd, ex, CONFIG.queryVariant ?? "skills", title);
    run.baseline = state.tfidf.advice(jd, resume);
    run.resumeIds = resume ? ex.extract(resume) : [];
    setStep("skills", "done", `${run.jdIds.length} skills`);

    // 2. embed the query with the same model that built the index
    setStep("embed", "running");
    status("Embedding the job...");
    run.qvec = (await embed([run.query]))[0];
    setStep("embed", "done", "384-d vector");

    if (!api.isConfigured()) {
      renderOffline(run);
      setStep("retrieve", "failed", "not set up");
      return;
    }

    // 3-4. evidence and O*NET come back fast, so show them while the model writes
    setStep("retrieve", "running");
    setStep("onet", "running");
    status("Finding resumes from similar jobs...");
    const [chunks, occs] = await Promise.all([api.matchChunks(run.qvec, k, category), api.matchOccupations(run.qvec, 1)]);
    run.chunks = chunks;
    run.occupation = occs[0] ?? null;
    setStep("retrieve", "done", `${chunks.length} excerpts`);
    setStep("onet", "done", run.occupation ? run.occupation.code : "no match");
    $("#panel-evidence").innerHTML = R.renderEvidence(chunks, run.occupation);
    run.marketIds = (run.occupation?.market?.top_skills ?? []).slice(0, 10).map((s) => s.id).filter((id) => ex.byId.has(id));
    $("#panel-gap").innerHTML = R.renderGap({ out: null, resumeText: resume, jdIds: run.jdIds, marketIds: run.marketIds,
      resumeIds: run.resumeIds, tfidf: run.baseline, ex });

    // 5. write the reference resume (and the plain-LLM version if asked)
    setStep("generate", "running");
    status("Writing the reference resume from the evidence...");
    const payload = { jd, title, resume, embedding: run.qvec, k, category, jd_skills: ex.names(run.jdIds) };
    const ragP = api.generate({ ...payload, mode: "rag" });
    const llmP = compareLlm ? api.generate({ ...payload, mode: "llm_only" }).catch((e) => ({ error: e.message })) : null;
    const rag = await ragP;
    run.rag = rag;
    run.chunks = rag.retrieved?.length ? rag.retrieved : run.chunks;
    setStep("generate", "done", rag.model?.split("/").pop() ?? "done");
    $("#panel-resume").innerHTML = R.renderResume(rag.output, run.chunks, null);
    showPanel("resume");
    setTile("time", secs(performance.now() - t0));
    $("#t-model").textContent = `${rag.model ?? ""}`;

    // 6. check every line against the evidence, right here in the browser
    setStep("verify", "running");
    status("Checking each line against the evidence...");
    const tau = CONFIG.groundednessThreshold ?? 0.55;
    run.ground = await groundedness(rag.output, run.chunks.map((c) => c.content), tau);
    run.outIds = ex.extract(outputText(rag.output));
    run.cov = coverage(run.jdIds, run.outIds);
    $("#panel-resume").innerHTML = R.renderResume(rag.output, run.chunks, run.ground.statements);
    const cited = new Set(statements(rag.output).flatMap((s) => s.sources));
    setTile("grounded", run.ground.share, run.ground.share);
    setTile("coverage", run.cov?.share ?? 0, run.cov?.share ?? 0);
    setTile("evidence", `${cited.size}/${run.chunks.length}`);
    $("#t-evidence-note").textContent = rag.invalid_citations_removed
      ? `${rag.invalid_citations_removed} bad citation(s) removed` : "out of those retrieved";
    setStep("verify", "done", `${R.pct(run.ground.share)} grounded`);
    $("#panel-gap").innerHTML = R.renderGap({ out: rag.output, resumeText: resume, jdIds: run.jdIds,
      marketIds: run.marketIds, resumeIds: run.resumeIds, tfidf: run.baseline, ex });

    if (llmP) {
      run.llm = await llmP;
      if (run.llm && !run.llm.error) {
        run.llmGround = await groundedness(run.llm.output, run.chunks.map((c) => c.content), tau);
        run.llmCov = coverage(run.jdIds, ex.extract(outputText(run.llm.output)));
      }
    }
    renderCompare(run);
    setTile("time", secs(performance.now() - t0));
    status(`Done in ${secs(performance.now() - t0)}. Click any citation number to see the resume it came from.`);
  } catch (err) {
    console.error(err);
    document.querySelectorAll(".step.running").forEach((el) => el.classList.replace("running", "failed"));
    status("");
    $("#panel-resume").innerHTML = `<div class="error-box"><b>Something went wrong.</b> ${R.esc(err.message)}<br>
      <span class="small">If it mentions a rate limit, give it a minute and try again. We're on a free tier.</span></div>`;
    showPanel("resume");
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate reference resume";
  }
}

function renderOffline(run) {
  status("The backend isn't set up on this copy of the site, so only the in-browser steps ran.");
  $("#panel-resume").innerHTML = `<div class="notice">The generation backend (Supabase + LLM) isn't set up here, so you're only
    seeing the steps that run in your browser. The README explains how to deploy it.</div>
    <h3>Skills we found</h3><div class="chips">${run.jdIds.map((id, i) => `<span class="chip pop" style="--i:${i}">${R.esc(state.ex.byId.get(id).name)}</span>`).join("")}</div>
    <h3 style="margin-top:16px">Search query</h3><p><code>${R.esc(run.query)}</code></p>`;
  renderCompare(run);
}

function renderCompare(run) {
  const ex = state.ex;
  const baseCov = coverage(run.jdIds, ex.extract(run.baseline.keywords.join(" ; ")));
  const cols = [R.renderCompareColumn({
    key: "baseline", title: "Keyword baseline (TF-IDF)", color: "var(--muted)", runId: run.rag?.run_id,
    metrics: [["Grounded lines", "n/a"], ["Job skills covered", R.pct(baseCov?.share)], ["Uses real resumes", "no"]],
    body: R.renderBaseline(run.baseline),
  })];
  if (run.llm && !run.llm.error) {
    cols.push(R.renderCompareColumn({
      key: "llm_only", title: "LLM only, no retrieval", color: "var(--s2)", runId: run.llm.run_id ?? run.rag?.run_id,
      metrics: [["Grounded lines", R.pct(run.llmGround?.share)], ["Job skills covered", R.pct(run.llmCov?.share)], ["Citations", "none"]],
      body: R.renderPlainResume(run.llm.output),
    }));
  } else if (run.llm?.error) {
    cols.push(`<div class="card"><h3>LLM only</h3><div class="error-box">${R.esc(run.llm.error)}</div></div>`);
  }
  if (run.rag) {
    cols.push(R.renderCompareColumn({
      key: "rag", title: "ScholarStack (RAG)", color: "var(--s1)", runId: run.rag.run_id,
      metrics: [["Grounded lines", R.pct(run.ground?.share)], ["Job skills covered", R.pct(run.cov?.share)],
        ["Excerpts cited", `${new Set(statements(run.rag.output).flatMap((s) => s.sources)).size}`]],
      body: R.renderPlainResume(run.rag.output),
    }));
  }
  const hint = run.llm ? "" : `<p class="small muted">Tick "Also run the plain LLM" under Options to add a third column.</p>`;
  $("#panel-compare").innerHTML = `${hint}<div class="grid-${cols.length === 3 ? 3 : 2}">${cols.join("")}</div>`;
}

// ------------------------------------------------------------------ citations popover + ratings
function initInteractions() {
  const pop = $("#popover");
  document.addEventListener("click", async (e) => {
    const cite = e.target.closest(".cite");
    if (cite) {
      const c = state.run?.chunks?.find((x) => x.chunk_id === cite.dataset.chunk);
      if (!c) return;
      pop.innerHTML = `<div class="meta">${R.esc(c.chunk_id)} · ${R.esc(R.prettyCategory(c.category))} · ${R.esc(c.resume_title)} ·
        ${R.esc(c.section)} · similarity ${c.similarity.toFixed(2)}</div><div class="body">${R.esc(c.content)}</div>`;
      pop.classList.remove("hidden");
      const r = cite.getBoundingClientRect();
      const top = Math.min(r.bottom + 6, innerHeight - pop.offsetHeight - 10);
      pop.style.top = `${Math.max(10, top)}px`;
      pop.style.left = `${Math.max(10, Math.min(r.left - 20, innerWidth - pop.offsetWidth - 10))}px`;
      e.stopPropagation();
      return;
    }
    if (!e.target.closest(".popover")) pop.classList.add("hidden");

    const star = e.target.closest(".stars button");
    if (star) {
      const box = star.parentElement;
      const v = Number(star.dataset.v);
      box.querySelectorAll("button").forEach((b) => b.classList.toggle("on", Number(b.dataset.v) <= v));
      try {
        await api.sendFeedback({ run_id: box.dataset.run || null, system: box.dataset.system, rating: v });
        box.insertAdjacentHTML("afterend", '<div class="small muted">Thanks, saved.</div>');
      } catch (err) {
        box.insertAdjacentHTML("afterend", `<div class="small muted">Couldn't save that: ${R.esc(err.message)}</div>`);
      }
    }
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") pop.classList.add("hidden"); });
}

// ------------------------------------------------------------------ boot
async function init() {
  initTheme();
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => { location.hash = t.dataset.view; }));
  addEventListener("hashchange", () => showView(location.hash.slice(1)));
  showView(location.hash.slice(1));
  document.querySelectorAll(".subtab").forEach((t) => t.addEventListener("click", () => showPanel(t.dataset.panel)));
  initInteractions();
  heroCounters();
  $("#go").addEventListener("click", runGenerate);
  $("#jd").addEventListener("input", debounce(updateSkills, 250));
  $("#job-title").addEventListener("input", debounce(updateSkills, 250));

  if (!api.isConfigured()) {
    const n = $("#backend-notice");
    n.innerHTML = "The generation backend isn't set up on this copy of the site (no Supabase settings in <code>js/config.js</code>). Skill extraction, embeddings and the keyword baseline still run in your browser.";
    n.classList.remove("hidden");
  }
  status("Loading...");
  const [lexicon, tfidf, samples, results] = await Promise.all([
    loadJSON("data/skills_lexicon.json"), loadJSON("data/tfidf.json"), loadJSON("data/samples.json"),
    loadJSON("data/results.json").catch(() => null),
  ]);
  state.ex = new SkillExtractor(lexicon);
  state.tfidf = new TfidfModel(tfidf);
  state.samples = samples;
  state.results = results;
  populate();
  updateSkills();
  fillTicker(lexicon.filter((e) => e.kind !== "soft").map((e) => e.name).filter((n) => n.length < 26));
  startTypewriter(samples.jds.map((j) => j.label));
  renderDashboard($("#dashboard"), results, api);
  renderMethod($("#method"), results);
  observeReveals();
  warmEmbedder();
}

init().catch((e) => status(`Failed to start: ${e.message}`));
