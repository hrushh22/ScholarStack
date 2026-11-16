// Live versions of the evaluation metrics in src/scholarstack/metrics.py.
import { dot, embed } from "./embed.js";

/** All factual statements of a generated resume, with their citations. */
export function statements(out) {
  const list = [];
  for (const s of out?.summary ?? []) list.push({ part: "summary", text: s.text ?? "", sources: s.sources ?? [] });
  for (const r of out?.experience ?? []) {
    for (const b of r.bullets ?? []) list.push({ part: "experience", text: b.text ?? "", sources: b.sources ?? [] });
  }
  for (const key of ["education", "certifications"]) {
    for (const s of out?.[key] ?? []) list.push({ part: key, text: s.text ?? "", sources: s.sources ?? [] });
  }
  return list.filter((s) => s.text.trim());
}

/** Everything the output asserts, as one text (for skill coverage). */
export function outputText(out) {
  const parts = [out?.headline ?? "", ...statements(out).map((s) => s.text)];
  for (const g of out?.skills ?? []) for (const i of g.items ?? []) parts.push(i.name ?? "");
  for (const r of out?.experience ?? []) parts.push(r.title ?? "");
  return parts.filter(Boolean).join("\n");
}

/**
 * Groundedness: each statement's best cosine similarity against the retrieved
 * evidence. Supported when >= tau (same threshold as the offline evaluation).
 */
export async function groundedness(out, evidenceTexts, tau) {
  const sts = statements(out);
  if (!sts.length || !evidenceTexts.length) return { statements: sts, share: 0, support: [] };
  const vecs = await embed([...sts.map((s) => s.text), ...evidenceTexts]);
  const sv = vecs.slice(0, sts.length);
  const ev = vecs.slice(sts.length);
  const support = sv.map((v) => Math.max(...ev.map((e) => dot(v, e))));
  const best = sv.map((v) => {
    const scores = ev.map((e) => dot(v, e));
    return scores.indexOf(Math.max(...scores));
  });
  sts.forEach((s, i) => {
    s.support = support[i];
    s.bestEvidence = best[i];
    s.supported = support[i] >= tau;
  });
  return { statements: sts, share: support.filter((x) => x >= tau).length / sts.length, support };
}

export function coverage(targetIds, foundIds) {
  if (!targetIds.length) return null;
  const found = new Set(foundIds);
  const hit = targetIds.filter((id) => found.has(id));
  return { share: hit.length / targetIds.length, hit, miss: targetIds.filter((id) => !found.has(id)) };
}
