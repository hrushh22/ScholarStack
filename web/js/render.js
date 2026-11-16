// Rendering of results. Every dynamic string goes through esc() — LLM output and
// resume text are untrusted.

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const pct = (x, digits = 0) => (x == null || Number.isNaN(x) ? "–" : `${(100 * x).toFixed(digits)}%`);

const SECTION_LABEL = { experience: "Experience", skills: "Skills", summary: "Summary", education: "Education",
  accomplishments: "Accomplishments", certifications: "Certifications", affiliations: "Affiliations",
  publications: "Publications", additional: "Additional" };

export function prettyCategory(c) {
  const special = { HR: "HR", BPO: "BPO", "INFORMATION-TECHNOLOGY": "Information Technology" };
  return special[c] ?? String(c).split("-").map((w) => w[0] + w.slice(1).toLowerCase()).join(" ");
}

function citeChips(sources, chunkIndex) {
  return (sources ?? []).filter((id) => chunkIndex.has(id)).map((id) =>
    `<button class="cite" type="button" data-chunk="${esc(id)}" title="Show evidence ${esc(id)}">${chunkIndex.get(id)}</button>`
  ).join("");
}

function stmtClass(v) {
  if (!v) return "pending";
  return v.supported ? "ok" : "weak";
}

function stmtTitle(v) {
  if (!v) return "";
  return `Best match with retrieved evidence: cosine ${v.support.toFixed(2)}${v.supported ? " (supported)" : " (weak support)"}`;
}

/** Reference resume as a document. verify = statements with support (same order as metrics.statements). */
export function renderResume(out, chunks, verify, { mode = "rag" } = {}) {
  const chunkIndex = new Map(chunks.map((c, i) => [c.chunk_id, i + 1]));
  let n = 0;
  const li = (s) => {
    const v = verify?.[n++];
    return `<li class="stmt ${stmtClass(v)}" title="${esc(stmtTitle(v))}">${esc(s.text)}${citeChips(s.sources, chunkIndex)}</li>`;
  };
  const parts = [];
  parts.push(`<div class="r-role">${esc(out.target_role ?? "")}</div>`);
  parts.push(`<h2 class="r-headline">${esc(out.headline ?? "Reference resume")}</h2>`);
  if (out.summary?.length) parts.push(`<h4>Summary</h4><ul>${out.summary.map(li).join("")}</ul>`);
  const exp = (out.experience ?? []).map((r) => `<div class="r-job"><div class="r-job-title">${esc(r.title)}</div>
      <div class="r-job-ctx">${esc(r.context ?? "")}</div><ul>${(r.bullets ?? []).map(li).join("")}</ul></div>`).join("");
  const skills = (out.skills ?? []).map((g) => `<div class="skill-group"><div class="g">${esc(g.group)}</div>
      <div class="chips">${(g.items ?? []).map((i) => `<span class="chip">${esc(i.name)}${i.source ? `<span class="tag">${esc(i.source)}</span>` : ""}</span>`).join("")}</div></div>`).join("");
  if (skills) parts.push(`<h4>Skills</h4><div class="skill-groups">${skills}</div>`);
  if (exp) parts.push(`<h4>Experience</h4>${exp}`);
  if (out.education?.length) parts.push(`<h4>Education</h4><ul>${out.education.map(li).join("")}</ul>`);
  if (out.certifications?.length) parts.push(`<h4>Certifications</h4><ul>${out.certifications.map(li).join("")}</ul>`);
  const legend = mode === "rag"
    ? `<div class="legend-inline"><span class="l-ok">supported by retrieved evidence</span><span class="l-weak">weakly supported (below threshold)</span><span>numbers = cited resume excerpts (click to view)</span></div>`
    : "";
  return `${legend}<article class="resume">${parts.join("")}</article>
    <div class="disclaimer">This is a <b>reference</b> resume assembled from resumes of people in similar roles. Use it to see
    what employers and peers emphasise — never copy experience you do not have.</div>`;
}

export function renderEvidence(chunks, occupation) {
  const occ = occupation ? renderOccupation(occupation) : "";
  const items = chunks.map((c, i) => `<div class="evidence-item" id="ev-${esc(c.chunk_id)}">
      <div class="ev-head"><span><span class="ev-id">[${i + 1}] ${esc(c.chunk_id)}</span> · ${esc(prettyCategory(c.category))} ·
        ${esc(c.resume_title || "Untitled")} · ${esc(SECTION_LABEL[c.section] ?? c.section)}</span>
        <span class="simbar" title="Cosine similarity to the query"><span class="track"><span class="fill" style="width:${Math.max(0, Math.min(1, c.similarity)) * 100}%"></span></span>${c.similarity.toFixed(2)}</span></div>
      <div class="ev-text">${esc(c.content)}</div></div>`).join("");
  return `${occ}<h3 style="margin-top:18px">Retrieved resume excerpts</h3>
    <p class="small muted">From ${new Set(chunks.map((c) => c.resume_id)).size} different resumes in the 2,480-resume corpus,
    ranked by cosine similarity between the job's skill profile and each excerpt.</p>${items}`;
}

export function renderOccupation(o) {
  const m = o.market;
  const skills = [...(o.essential_skills ?? []), ...(o.transferable_skills ?? [])]
    .sort((a, b) => b.importance - a.importance).slice(0, 8);
  const demand = m?.top_skills?.length
    ? `<div class="bars">${m.top_skills.slice(0, 8).map((s) => `<div class="bar-row"><span>${esc(s.name)}</span>
        <span class="track"><span class="fill" style="width:${(s.share * 100).toFixed(1)}%;display:block"></span></span>
        <span class="v">${pct(s.share)}</span></div>`).join("")}</div>`
    : `<p class="small muted">No linked LinkedIn postings for this occupation.</p>`;
  const salary = m?.median_salary ? `$${Math.round(m.median_salary).toLocaleString()}` : "not enough data";
  return `<div class="grid-2">
    <div><h3>${esc(o.title)} <span class="badge">O*NET ${esc(o.code)}</span></h3>
      <p class="small">${esc(o.description)}</p>
      <dl class="kv"><dt>Match</dt><dd>cosine ${o.similarity?.toFixed(2) ?? "–"}</dd>
        <dt>Job zone</dt><dd>${esc(o.job_zone ?? "n/a")} of 5</dd>
        <dt>Key skills</dt><dd>${skills.map((s) => esc(s.name)).join(", ")}</dd>
        <dt>Technologies</dt><dd>${(o.technologies ?? []).slice(0, 10).map((t) => esc(t.name)).join(", ")}</dd></dl></div>
    <div><h3>Demand in ${m?.n_postings?.toLocaleString() ?? 0} LinkedIn postings</h3>
      <p class="small muted">Share of postings for this occupation that mention each skill. Median salary: ${salary};
      remote-eligible: ${pct(m?.remote_share)}.</p>${demand}</div></div>`;
}

export function renderGap({ out, resumeText, jdIds, marketIds, resumeIds, tfidf, ex }) {
  if (!resumeText?.trim()) {
    return `<div class="empty">Paste your resume (or pick a fictional sample) and generate again to see what is missing.</div>`;
  }
  const have = new Set(resumeIds);
  const chip = (id) => `<span class="chip ${have.has(id) ? "hit" : "miss"}">${have.has(id) ? "✓" : "✕"} ${esc(ex.byId.get(id)?.name ?? id)}</span>`;
  const share = (ids) => (ids.length ? ids.filter((id) => have.has(id)).length / ids.length : null);
  const g = out?.gap_analysis;
  const llm = g ? `
    <h3 style="margin-top:20px">Advisor's read</h3><p>${esc(g.fit_summary ?? "")}</p>
    <div class="grid-2">
      <div><h4>Strengths to keep</h4><ul>${(g.strengths ?? []).map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>
      <div><h4>Missing skills</h4><ul>${(g.missing_skills ?? []).map((s) => `<li><b>${esc(s.skill)}</b> — ${esc(s.why)}</li>`).join("")}</ul></div>
    </div>
    ${(g.weak_sections ?? []).length ? `<h4>Sections to strengthen</h4><table class="data"><thead><tr><th>Section</th><th>Issue</th><th>Suggestion</th></tr></thead><tbody>
      ${g.weak_sections.map((w) => `<tr><td>${esc(w.section)}</td><td>${esc(w.issue)}</td><td>${esc(w.suggestion)}</td></tr>`).join("")}</tbody></table>` : ""}
    ${(g.bullet_rewrites ?? []).length ? `<h4 style="margin-top:16px">Bullet rewrites</h4><table class="data"><thead><tr><th>Before</th><th>After</th></tr></thead><tbody>
      ${g.bullet_rewrites.map((b) => `<tr><td>${esc(b.before)}</td><td>${esc(b.after)}</td></tr>`).join("")}</tbody></table>` : ""}`
    : `<p class="small muted" style="margin-top:16px">The written gap analysis appears after generation.</p>`;
  return `<div class="tiles">
      <div class="tile"><div class="label">Job skills on your resume</div><div class="value">${pct(share(jdIds))}</div><div class="note">${jdIds.filter((i) => have.has(i)).length} of ${jdIds.length} detected skills</div></div>
      <div class="tile"><div class="label">In-demand skills (LinkedIn)</div><div class="value">${pct(share(marketIds))}</div><div class="note">top skills for the matched occupation</div></div>
      <div class="tile"><div class="label">Keyword match (TF-IDF)</div><div class="value">${tfidf?.match_score != null ? tfidf.match_score.toFixed(2) : "–"}</div><div class="note">cosine, 0–1 (the baseline score)</div></div>
      <div class="tile"><div class="label">Skills found on resume</div><div class="value">${resumeIds.length}</div><div class="note">rule-based extraction</div></div>
    </div>
    <h4>Skills from the job description</h4><div class="chips">${jdIds.map(chip).join("") || '<span class="muted small">none detected</span>'}</div>
    ${marketIds.length ? `<h4 style="margin-top:14px">Most-requested skills for this occupation</h4><div class="chips">${marketIds.map(chip).join("")}</div>` : ""}
    ${llm}`;
}

export function renderBaseline(advice) {
  return `<p class="small muted">Generic keyword advice: the highest-weighted terms of the job description (TF-IDF
    fitted on 22k resumes and postings). No retrieval, no generation.</p>
    <div class="chips">${advice.keywords.map((k) => `<span class="chip ${advice.missing_keywords ? (advice.missing_keywords.includes(k) ? "miss" : "hit") : ""}">${esc(k)}</span>`).join("")}</div>
    ${advice.match_score != null ? `<p class="small" style="margin-top:10px">Resume–job keyword match: <b>${advice.match_score.toFixed(2)}</b></p>` : ""}`;
}

export function renderCompareColumn({ key, title, color, metrics, body, runId }) {
  const rows = metrics.map(([label, value]) => `<li><span>${esc(label)}</span><b>${esc(value)}</b></li>`).join("");
  const stars = runId !== undefined
    ? `<div class="small muted" style="margin-top:10px">How useful is this? (saved anonymously)</div>
       <div class="stars" data-system="${esc(key)}" data-run="${esc(runId ?? "")}">${[1, 2, 3, 4, 5].map((n) => `<button type="button" data-v="${n}" aria-label="${n} stars">★</button>`).join("")}</div>`
    : "";
  return `<div class="compare-col card"><h3><span class="dot" style="background:${color}"></span>${esc(title)}</h3>
    <ul class="metric-list">${rows}</ul>${stars}<div class="preview">${body}</div></div>`;
}

export function renderPlainResume(out) {
  const lines = [out.headline ?? ""];
  for (const s of out.summary ?? []) lines.push(s.text);
  for (const g of out.skills ?? []) lines.push(`${g.group}: ${(g.items ?? []).map((i) => i.name).join(", ")}`);
  for (const r of out.experience ?? []) {
    lines.push(`\n${r.title} — ${r.context ?? ""}`);
    for (const b of r.bullets ?? []) lines.push(`• ${b.text}`);
  }
  return esc(lines.join("\n"));
}
