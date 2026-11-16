// HTML for the results panels. Everything dynamic goes through esc(): model output
// and pasted resumes are untrusted.

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const pct = (x, digits = 0) => (x == null || Number.isNaN(x) ? "-" : `${(100 * x).toFixed(digits)}%`);

const SECTION_LABEL = { experience: "Experience", skills: "Skills", summary: "Summary", education: "Education",
  accomplishments: "Accomplishments", certifications: "Certifications", affiliations: "Affiliations",
  publications: "Publications", additional: "Additional" };

export function prettyCategory(c) {
  const special = { HR: "HR", BPO: "BPO", "INFORMATION-TECHNOLOGY": "Information Technology" };
  return special[c] ?? String(c).split("-").map((w) => w[0] + w.slice(1).toLowerCase()).join(" ");
}

function citeChips(sources, chunkIndex) {
  return (sources ?? []).filter((id) => chunkIndex.has(id)).map((id) =>
    `<button class="cite" type="button" data-chunk="${esc(id)}" title="Show the source (${esc(id)})">${chunkIndex.get(id)}</button>`
  ).join("");
}

const stmtClass = (v) => (!v ? "pending" : v.supported ? "ok" : "weak");
const stmtTitle = (v) => (!v ? "" : `Closest evidence match: ${v.support.toFixed(2)}${v.supported ? " (backed by the evidence)" : " (weak match)"}`);

/** The reference resume. `verify` holds per-line support, in the same order as metrics.statements(). */
export function renderResume(out, chunks, verify, { mode = "rag" } = {}) {
  const chunkIndex = new Map(chunks.map((c, i) => [c.chunk_id, i + 1]));
  let n = 0;
  let order = 0;
  const li = (s) => {
    const v = verify?.[n++];
    return `<li class="stmt ${stmtClass(v)}" style="--i:${order++}" title="${esc(stmtTitle(v))}">${esc(s.text)}${citeChips(s.sources, chunkIndex)}</li>`;
  };
  const parts = [];
  parts.push(`<div class="r-role">${esc(out.target_role ?? "")}</div>`);
  parts.push(`<h2 class="r-headline">${esc(out.headline ?? "Reference resume")}</h2>`);
  if (out.summary?.length) parts.push(`<h4>Summary</h4><ul>${out.summary.map(li).join("")}</ul>`);
  const skills = (out.skills ?? []).map((g) => `<div class="skill-group" style="--i:${order++}"><div class="g">${esc(g.group)}</div>
      <div class="chips">${(g.items ?? []).map((i) => `<span class="chip">${esc(i.name)}${i.source ? `<span class="tag">${esc(i.source)}</span>` : ""}</span>`).join("")}</div></div>`).join("");
  const exp = (out.experience ?? []).map((r) => `<div class="r-job"><div class="r-job-title" style="--i:${order++}">${esc(r.title)}</div>
      <div class="r-job-ctx">${esc(r.context ?? "")}</div><ul>${(r.bullets ?? []).map(li).join("")}</ul></div>`).join("");
  if (skills) parts.push(`<h4>Skills</h4><div class="skill-groups">${skills}</div>`);
  if (exp) parts.push(`<h4>Experience</h4>${exp}`);
  if (out.education?.length) parts.push(`<h4>Education</h4><ul>${out.education.map(li).join("")}</ul>`);
  if (out.certifications?.length) parts.push(`<h4>Certifications</h4><ul>${out.certifications.map(li).join("")}</ul>`);
  const legend = mode === "rag"
    ? `<div class="legend-inline"><span class="l-ok">backed by the evidence</span><span class="l-weak">weak match</span><span>numbers are citations, click one to see the source</span></div>`
    : "";
  return `${legend}<article class="resume">${parts.join("")}</article>
    <div class="disclaimer">This is a <b>reference</b>, put together from resumes of people in similar roles. Use it to see
    what they emphasize and how they phrase it. Don't copy experience you don't have.</div>`;
}

export function renderEvidence(chunks, occupation) {
  const occ = occupation ? renderOccupation(occupation) : "";
  const items = chunks.map((c, i) => `<div class="evidence-item" style="animation:fadeUp .4s both;animation-delay:${i * 50}ms" id="ev-${esc(c.chunk_id)}">
      <div class="ev-head"><span><span class="ev-id">[${i + 1}] ${esc(c.chunk_id)}</span> · ${esc(prettyCategory(c.category))} ·
        ${esc(c.resume_title || "Untitled")} · ${esc(SECTION_LABEL[c.section] ?? c.section)}</span>
        <span class="simbar" title="How closely this excerpt matches the job"><span class="track"><span class="fill" style="width:${Math.max(0, Math.min(1, c.similarity)) * 100}%"></span></span>${c.similarity.toFixed(2)}</span></div>
      <div class="ev-text">${esc(c.content)}</div></div>`).join("");
  return `${occ}<h3 style="margin-top:20px">Resume excerpts we used</h3>
    <p class="small muted">From ${new Set(chunks.map((c) => c.resume_id)).size} different resumes in our 2,480-resume set,
    ranked by how closely they match the skills in the posting.</p>${items}`;
}

export function renderOccupation(o) {
  const m = o.market;
  const skills = [...(o.essential_skills ?? []), ...(o.transferable_skills ?? [])]
    .sort((a, b) => b.importance - a.importance).slice(0, 8);
  const demand = m?.top_skills?.length
    ? `<div class="bars">${m.top_skills.slice(0, 8).map((s) => `<div class="bar-row"><span>${esc(s.name)}</span>
        <span class="track"><span class="fill" style="width:${(s.share * 100).toFixed(1)}%"></span></span>
        <span class="v">${pct(s.share)}</span></div>`).join("")}</div>`
    : `<p class="small muted">We didn't find LinkedIn postings linked to this occupation.</p>`;
  const salary = m?.median_salary ? `$${Math.round(m.median_salary).toLocaleString()}` : "not enough data";
  return `<div class="grid-2">
    <div><h3>${esc(o.title)} <span class="badge">O*NET ${esc(o.code)}</span></h3>
      <p class="small">${esc(o.description)}</p>
      <dl class="kv"><dt>Match</dt><dd>${o.similarity?.toFixed(2) ?? "-"}</dd>
        <dt>Job zone</dt><dd>${esc(o.job_zone ?? "n/a")} of 5</dd>
        <dt>Key skills</dt><dd>${skills.map((s) => esc(s.name)).join(", ")}</dd>
        <dt>Tools</dt><dd>${(o.technologies ?? []).slice(0, 10).map((t) => esc(t.name)).join(", ")}</dd></dl></div>
    <div><h3>What ${m?.n_postings?.toLocaleString() ?? 0} LinkedIn postings ask for</h3>
      <p class="small muted">Share of postings for this occupation that mention each skill. Median salary ${salary},
      remote-friendly ${pct(m?.remote_share)}.</p>${demand}</div></div>`;
}

export function renderGap({ out, resumeText, jdIds, marketIds, resumeIds, tfidf, ex }) {
  if (!resumeText?.trim()) {
    return `<div class="empty"><div class="doc"></div>Paste your resume (or pick one of the made-up samples) and generate
      again to see what's missing.</div>`;
  }
  const have = new Set(resumeIds);
  let i = 0;
  const chip = (id) => `<span class="chip pop ${have.has(id) ? "hit" : "miss"}" style="--i:${i++}">${have.has(id) ? "✓" : "✕"} ${esc(ex.byId.get(id)?.name ?? id)}</span>`;
  const share = (ids) => (ids.length ? ids.filter((id) => have.has(id)).length / ids.length : null);
  const g = out?.gap_analysis;
  const notes = g ? `
    <h3 style="margin-top:22px">Advisor notes</h3><p>${esc(g.fit_summary ?? "")}</p>
    <div class="grid-2">
      <div><h4>Keep these</h4><ul>${(g.strengths ?? []).map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>
      <div><h4>Missing</h4><ul>${(g.missing_skills ?? []).map((s) => `<li><b>${esc(s.skill)}</b>: ${esc(s.why)}</li>`).join("")}</ul></div>
    </div>
    ${(g.weak_sections ?? []).length ? `<h4>Sections to work on</h4><table class="data"><thead><tr><th>Section</th><th>Issue</th><th>Suggestion</th></tr></thead><tbody>
      ${g.weak_sections.map((w) => `<tr><td>${esc(w.section)}</td><td>${esc(w.issue)}</td><td>${esc(w.suggestion)}</td></tr>`).join("")}</tbody></table>` : ""}
    ${(g.bullet_rewrites ?? []).length ? `<h4 style="margin-top:16px">Bullet rewrites</h4><table class="data"><thead><tr><th>Yours</th><th>Stronger version</th></tr></thead><tbody>
      ${g.bullet_rewrites.map((b) => `<tr><td>${esc(b.before)}</td><td>${esc(b.after)}</td></tr>`).join("")}</tbody></table>` : ""}`
    : out
      ? `<div class="notice" style="margin-top:16px">The model didn't send back written notes this time. The skill comparison
         above is complete. Hit <b>Generate</b> again if you want the notes too.</div>`
      : `<p class="small muted" style="margin-top:16px">The written notes show up once the resume is generated.</p>`;
  return `<div class="tiles">
      <div class="tile lit"><div class="label">Job skills on your resume</div><div class="value">${pct(share(jdIds))}</div><div class="note">${jdIds.filter((x) => have.has(x)).length} of ${jdIds.length} skills in the posting</div></div>
      <div class="tile lit"><div class="label">In-demand skills you have</div><div class="value">${pct(share(marketIds))}</div><div class="note">top skills for this occupation on LinkedIn</div></div>
      <div class="tile lit"><div class="label">Keyword match</div><div class="value">${tfidf?.match_score != null ? tfidf.match_score.toFixed(2) : "-"}</div><div class="note">TF-IDF cosine, 0 to 1</div></div>
      <div class="tile lit"><div class="label">Skills on your resume</div><div class="value">${resumeIds.length}</div><div class="note">that our extractor recognized</div></div>
    </div>
    <h4>Skills from the posting</h4><div class="chips">${jdIds.map(chip).join("") || '<span class="muted small">none found</span>'}</div>
    ${marketIds.length ? `<h4 style="margin-top:14px">What employers ask for most in this occupation</h4><div class="chips">${marketIds.map(chip).join("")}</div>` : ""}
    ${notes}`;
}

export function renderBaseline(advice) {
  return `<p class="small muted">Plain keyword advice: the highest-weighted words in the posting according to TF-IDF
    (fit on 22k resumes and postings). No retrieval and no generation.</p>
    <div class="chips">${advice.keywords.map((k, i) => `<span class="chip pop ${advice.missing_keywords ? (advice.missing_keywords.includes(k) ? "miss" : "hit") : ""}" style="--i:${i}">${esc(k)}</span>`).join("")}</div>
    ${advice.match_score != null ? `<p class="small" style="margin-top:10px">Keyword match with your resume: <b>${advice.match_score.toFixed(2)}</b></p>` : ""}`;
}

export function renderCompareColumn({ key, title, color, metrics, body, runId }) {
  const rows = metrics.map(([label, value]) => `<li><span>${esc(label)}</span><b>${esc(value)}</b></li>`).join("");
  const stars = runId !== undefined
    ? `<div class="small muted" style="margin-top:10px">How useful is this? (anonymous)</div>
       <div class="stars" data-system="${esc(key)}" data-run="${esc(runId ?? "")}">${[1, 2, 3, 4, 5].map((n) => `<button type="button" data-v="${n}" aria-label="${n} stars">★</button>`).join("")}</div>`
    : "";
  return `<div class="compare-col card lift"><h3><span class="dot" style="background:${color}"></span>${esc(title)}</h3>
    <ul class="metric-list">${rows}</ul>${stars}<div class="preview">${body}</div></div>`;
}

export function renderPlainResume(out) {
  const lines = [out.headline ?? ""];
  for (const s of out.summary ?? []) lines.push(s.text);
  for (const g of out.skills ?? []) lines.push(`${g.group}: ${(g.items ?? []).map((i) => i.name).join(", ")}`);
  for (const r of out.experience ?? []) {
    lines.push(`\n${r.title}${r.context ? `, ${r.context}` : ""}`);
    for (const b of r.bullets ?? []) lines.push(`• ${b.text}`);
  }
  return esc(lines.join("\n"));
}
