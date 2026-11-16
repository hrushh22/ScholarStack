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

/** Evidence at two granularities — each chunk and each line in it (as metrics.evidence_units in Python). */
export function evidenceUnits(chunkTexts, minWords = 4) {
  const units = [];
  const owner = [];
  chunkTexts.forEach((text, i) => {
    units.push(text.split(/\s+/).filter(Boolean).join(" "));
    owner.push(i);
    const lines = text.split("\n").map((l) => l.replace(/^[\s\-•\t]+|[\s\-•\t]+$/g, ""));
    if (lines.length > 1) {
      for (const line of lines) {
        if (line.split(/\s+/).filter(Boolean).length >= minWords) {
          units.push(line);
          owner.push(i);
        }
      }
    }
  });
  return { units, owner };
}

/**
 * Fast grounding check: each statement's best cosine similarity with the evidence
 * units. The threshold tau was calibrated offline to agree with an LLM verifier.
 */
export async function groundedness(out, evidenceTexts, tau) {
  const sts = statements(out);
  if (!sts.length || !evidenceTexts.length) return { statements: sts, share: 0, support: [] };
  const { units, owner } = evidenceUnits(evidenceTexts);
  const vecs = await embed([...sts.map((s) => s.text), ...units]);
  const sv = vecs.slice(0, sts.length);
  const ev = vecs.slice(sts.length);
  const support = [];
  sv.forEach((v, i) => {
    const scores = ev.map((e) => dot(v, e));
    const best = scores.indexOf(Math.max(...scores));
    support.push(scores[best]);
    sts[i].support = scores[best];
    sts[i].bestEvidence = owner[best];
    sts[i].bestLine = units[best];
    sts[i].supported = scores[best] >= tau;
  });
  return { statements: sts, share: support.filter((x) => x >= tau).length / sts.length, support };
}

export function coverage(targetIds, foundIds) {
  if (!targetIds.length) return null;
  const found = new Set(foundIds);
  const hit = targetIds.filter((id) => found.has(id));
  return { share: hit.length / targetIds.length, hit, miss: targetIds.filter((id) => !found.has(id)) };
}
