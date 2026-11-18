// "How it works" page.
import { esc } from "./render.js?v=202611171703";

const REPO = "https://github.com/hrushh22/ScholarStack/blob/main";

const OFFLINE = [
  ["01", "Download", "Scripts pull the Kaggle resumes and LinkedIn postings and the O*NET 31.0 files. No manual downloads.", "pipeline/01_acquire.py"],
  ["02", "Profile", "Check columns, missing values, duplicates and odd lengths. Stage the 517 MB CSV as Parquet.", "pipeline/02_profile.py"],
  ["03", "Clean", "Split resume HTML into sections. Clean and de-duplicate 124k postings with Dask.", "pipeline/03_clean.py"],
  ["04", "Skills", "A 1,457-skill dictionary built from O*NET software plus our own list, matched with Aho-Corasick.", "pipeline/04_skills.py"],
  ["05", "Chunk and embed", "Section chunks vs fixed windows, embedded with three sentence-transformer models on the GPU.", "pipeline/05_chunk_embed.py"],
  ["06", "Index", "FAISS Flat, HNSW and IVF indexes over the resume chunks and all 106k postings.", "pipeline/06_index.py"],
  ["07", "Link to O*NET", "Match posting titles to 54k O*NET titles, then count what each occupation asks for.", "pipeline/07_onet_link.py"],
  ["08-10", "Evaluate", "Retrieval precision, grounding checks, skill coverage and a judge model, against two baselines.", "pipeline/10_eval_generation.py"],
];
const ONLINE = [
  ["Your browser", "Find skills", "Same dictionary and rules as the Python code. We checked they match on 100 postings."],
  ["Your browser", "Embed the job", "all-MiniLM-L6-v2 running in the page with transformers.js. Same vectors as the offline index."],
  ["Supabase", "Find evidence", "pgvector search over 27,207 resume chunks, plus the closest O*NET occupation."],
  ["Edge Function", "Write", "gpt-oss-120b on Groq (falls back to gpt-oss-20b). It has to cite the excerpts it uses."],
  ["Your browser", "Check", "Each line is compared with the evidence. Citations to anything we didn't retrieve are dropped."],
];

export function renderMethod(root, res) {
  const cfg = res?.config ?? {};
  root.innerHTML = `
  <div class="card reveal"><h2>The pipeline</h2>
    <p>Everything heavy runs offline in Python: data processing, indexing and the experiments. The site you're on is just
    static files. The browser embeds the posting and checks the output, Supabase stores the vectors, and a small Edge
    Function calls the LLM so the API key never reaches your browser.</p>
    <h3 style="margin-top:16px">Offline, in Python <span class="badge">Dask · FAISS · LangChain</span></h3>
    <div class="flow">${OFFLINE.map(([n, t, d, f]) => `<div class="node"><span class="n">${n}</span><b>${esc(t)}</b>
      <span class="small">${esc(d)}</span><br><a class="small" href="${REPO}/${f}" target="_blank" rel="noopener">${esc(f)}</a></div>`).join("")}</div>
    <h3 style="margin-top:18px">When you hit Generate <span class="badge">GitHub Pages · Supabase · Groq</span></h3>
    <div class="flow">${ONLINE.map(([w, t, d], i) => `<div class="node"><span class="n">${i + 1} · ${esc(w)}</span><b>${esc(t)}</b>
      <span class="small">${esc(d)}</span></div>`).join("")}</div>
  </div>

  <div class="grid-2" style="margin-top:16px">
    <div class="card lift reveal"><h2>Data</h2>
      <table class="data"><thead><tr><th>Source</th><th>Size</th><th>What we use it for</th></tr></thead><tbody>
        <tr><td><a href="https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset" target="_blank" rel="noopener">Resume dataset</a> (Kaggle, LiveCareer)</td><td>2,484 raw, 2,480 kept</td><td>The resumes we cite as evidence</td></tr>
        <tr><td><a href="https://www.kaggle.com/datasets/arshkon/linkedin-job-postings" target="_blank" rel="noopener">LinkedIn job postings</a> 2023-24 (Kaggle)</td><td>123,849 raw, 106,291 kept</td><td>Job ads, skill demand, the test set</td></tr>
        <tr><td><a href="https://www.onetcenter.org/database.html" target="_blank" rel="noopener">O*NET 31.0</a> (U.S. DOL/ETA, CC BY 4.0)</td><td>1,016 occupations</td><td>Standard skills, knowledge, tasks and tools</td></tr>
      </tbody></table>
      <p class="small muted" style="margin-top:8px">We never put raw resume text in the GitHub repo. It only lives in the database
      and shows up here as short excerpts.</p>
    </div>
    <div class="card lift reveal"><h2>What's running right now</h2>
      <dl class="kv">
        <dt>Embedding model</dt><dd>${esc(cfg.embedding_model ?? "all-MiniLM-L6-v2")} (384 dimensions)</dd>
        <dt>Chunks</dt><dd>one per resume section, one per job in the experience section</dd>
        <dt>Search query</dt><dd>job title plus the skills we extracted (it won our query test)</dd>
        <dt>Evidence</dt><dd>${esc(cfg.k ?? 8)} resume excerpts plus the closest O*NET occupation</dd>
        <dt>LLM</dt><dd>${esc(cfg.llm_model ?? "")}, falling back to ${esc(cfg.llm_fallback_model ?? "")}, temperature 0.2, JSON output</dd>
        <dt>Grounding check</dt><dd>embedding similarity of at least ${esc(cfg.groundedness_threshold ?? 0.55)}</dd>
      </dl>
      <p class="small" style="margin-top:10px">The Python evaluation and this site use the exact same prompts:
      <a href="${REPO}/supabase/functions/_shared/prompts.json" target="_blank" rel="noopener">prompts.json</a>.</p>
    </div>
  </div>

  <div class="grid-2" style="margin-top:16px">
    <div class="card lift reveal"><h2>Limitations</h2><ul>
      <li>The resumes are examples from one website, not verified employees. Some categories (BPO, Automobile) are broad keyword buckets.</li>
      <li>There are no hand-made relevance labels, so we use category match and skill overlap as stand-ins.</li>
      <li>The grounding check is another LLM. It can be fooled by a paraphrase that sneaks in a small extra detail.</li>
      <li>The postings are a 2023-24 snapshot, and O*NET describes occupations rather than specific jobs.</li>
      <li>The judge model may share blind spots with the writer, which is why the site also collects real ratings.</li>
    </ul></div>
    <div class="card lift reveal"><h2>Using it responsibly</h2><ul>
      <li>A reference resume is for learning. Only put experience on your resume that you actually have.</li>
      <li>A resume you paste is used for that one request and isn't stored. We log runs without it.</li>
      <li>The source resumes can reflect old biases about who held which jobs. Read the patterns as a description, not a rule.</li>
      <li>Everything runs on free tiers: GitHub Pages, Supabase, Groq and open models from Hugging Face.</li>
    </ul></div>
  </div>

  <div class="card reveal" style="margin-top:16px"><h2>Team</h2>
    <p>BDA 696, San Diego State University. Hrushik Mehta, Dhrumil Panchal, Sonika Srinivas,
    Parjanya Hebbale Krishna.</p></div>`;
}
