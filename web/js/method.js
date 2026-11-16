// "How it works" page — architecture, data, design decisions, limitations.
import { esc } from "./render.js";

const REPO = "https://github.com/hrushh22/ScholarStack/blob/main";

const OFFLINE = [
  ["01", "Acquire", "Scripted downloads: Kaggle resumes & LinkedIn postings, O*NET 31.0 text files (fsspec-readable).", "pipeline/01_acquire.py"],
  ["02", "Profile", "Schema, null rates, duplicates and length outliers; CSV staged to Parquet.", "pipeline/02_profile.py"],
  ["03", "Clean", "Resume HTML parsed into sections; 124k postings cleaned and de-duplicated with Dask.", "pipeline/03_clean.py"],
  ["04", "Skills", "1,457-skill lexicon from O*NET software skills + curated domain skills; Aho-Corasick matching.", "pipeline/04_skills.py"],
  ["05", "Chunk & embed", "Section vs. fixed-window chunks; three sentence-transformers, batched on GPU.", "pipeline/05_chunk_embed.py"],
  ["06", "Index", "FAISS Flat / HNSW / IVF over resume chunks and all 106k postings, benchmarked.", "pipeline/06_index.py"],
  ["07", "Link O*NET", "Posting titles matched to 54k O*NET titles; demand aggregated per occupation.", "pipeline/07_onet_link.py"],
  ["08–10", "Evaluate", "Retrieval precision@k, groundedness, skill coverage, LLM judge; TF-IDF and LLM-only baselines.", "pipeline/10_eval_generation.py"],
];
const ONLINE = [
  ["Browser", "Skill extraction", "Same lexicon and rules as Python (verified identical on 100 held-out JDs)."],
  ["Browser", "Query embedding", "all-MiniLM-L6-v2 via transformers.js; vectors identical to the offline index."],
  ["Supabase", "Retrieval", "pgvector HNSW over 27,207 resume chunks + O*NET profiles with LinkedIn demand."],
  ["Edge Function", "Generation", "Groq-hosted gpt-oss-120b (fallback gpt-oss-20b), JSON output; every statement must cite retrieved chunk ids."],
  ["Browser", "Verification", "Each statement re-embedded and compared with the evidence; invalid citations are dropped."],
];

export function renderMethod(root, res) {
  const cfg = res?.config ?? {};
  root.innerHTML = `
  <div class="card"><h2>Architecture</h2>
    <p>The heavy lifting runs offline in Python (big-data processing, indexing, experiments). The live site is static
    (GitHub Pages): the browser embeds the query and verifies the output, Supabase serves the vectors, and a small Edge Function
    calls the LLM so no API key ever reaches the browser.</p>
    <h3 style="margin-top:16px">Offline pipeline <span class="badge">Python · Dask · FAISS</span></h3>
    <div class="flow">${OFFLINE.map(([n, t, d, f]) => `<div class="node"><span class="n">${n}</span><b>${esc(t)}</b>
      <span class="small">${esc(d)}</span><br><a class="small" href="${REPO}/${f}" target="_blank" rel="noopener">${esc(f)}</a></div>`).join("")}</div>
    <h3 style="margin-top:18px">Live request <span class="badge">GitHub Pages · Supabase · Groq</span></h3>
    <div class="flow">${ONLINE.map(([w, t, d], i) => `<div class="node"><span class="n">${i + 1} · ${esc(w)}</span><b>${esc(t)}</b>
      <span class="small">${esc(d)}</span></div>`).join("")}</div>
  </div>

  <div class="grid-2" style="margin-top:16px">
    <div class="card"><h2>Data</h2>
      <table class="data"><thead><tr><th>Source</th><th>Size</th><th>Role</th></tr></thead><tbody>
        <tr><td><a href="https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset" target="_blank" rel="noopener">Resume dataset</a> (Kaggle, LiveCareer)</td><td>2,484 → 2,480</td><td>Evidence corpus: "people in similar roles"</td></tr>
        <tr><td><a href="https://www.kaggle.com/datasets/arshkon/linkedin-job-postings" target="_blank" rel="noopener">LinkedIn job postings</a> 2023–24 (Kaggle)</td><td>123,849 → 106,291</td><td>Job descriptions, skill demand, evaluation set</td></tr>
        <tr><td><a href="https://www.onetcenter.org/database.html" target="_blank" rel="noopener">O*NET 31.0</a> (U.S. DOL/ETA, CC BY 4.0)</td><td>1,016 occupations</td><td>Standard skills, knowledge, tasks, technologies</td></tr>
      </tbody></table>
      <p class="small muted" style="margin-top:8px">Raw resume text is never committed to the repository; it lives only in the
      database and is shown as short evidence excerpts.</p>
    </div>
    <div class="card"><h2>Served configuration</h2>
      <dl class="kv">
        <dt>Embedding model</dt><dd>${esc(cfg.embedding_model ?? "all-MiniLM-L6-v2")} (384-d)</dd>
        <dt>Chunking</dt><dd>${esc(cfg.chunking ?? "section")} (one chunk per section / per job)</dd>
        <dt>Query</dt><dd>job title + extracted skills (best in the query experiment)</dd>
        <dt>Evidence per request</dt><dd>k = ${esc(cfg.k ?? 8)} excerpts + top O*NET occupation</dd>
        <dt>LLM</dt><dd>${esc(cfg.llm_model ?? "")} (fallback ${esc(cfg.llm_fallback_model ?? "")}), temperature 0.2, JSON mode</dd>
        <dt>Grounding threshold</dt><dd>cosine ≥ ${esc(cfg.groundedness_threshold ?? 0.55)}</dd>
      </dl>
      <p class="small" style="margin-top:10px">Prompts are shared by the Python evaluation and the live Edge Function:
      <a href="${REPO}/supabase/functions/_shared/prompts.json" target="_blank" rel="noopener">prompts.json</a>.</p>
    </div>
  </div>

  <div class="grid-2" style="margin-top:16px">
    <div class="card"><h2>Limitations</h2><ul>
      <li>Resumes are example documents from one site, not verified employees; categories are keyword-based and broad (BPO, Automobile).</li>
      <li>No gold relevance labels exist; category agreement and skill overlap are proxies.</li>
      <li>Groundedness uses embedding similarity, which can accept paraphrases that add small unsupported details.</li>
      <li>LinkedIn postings are a 2023–24 snapshot; O*NET describes occupations, not individual jobs.</li>
      <li>The LLM judge may share biases with the generator; live human ratings are collected to complement it.</li>
    </ul></div>
    <div class="card"><h2>Responsible use</h2><ul>
      <li>A reference resume is something to learn from. Only claim experience you actually have.</li>
      <li>Your pasted resume is sent only for that request and is not stored; runs are logged without it.</li>
      <li>Source resumes may reflect historical bias in who held which jobs; treat patterns as descriptive, not prescriptive.</li>
      <li>Built only on free tiers: GitHub Pages, Supabase, Groq, Hugging Face models.</li>
    </ul></div>
  </div>

  <div class="card" style="margin-top:16px"><h2>Team</h2>
    <p>BDA 696 · San Diego State University. Sonika Srinivas (lead), Hrushik Nimesh Mehta, Dhrumil Vinay Panchal,
    Parjanya Hebbale Krishna.</p></div>`;
}
