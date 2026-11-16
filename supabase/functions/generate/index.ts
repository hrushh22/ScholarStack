// ScholarStack live generation endpoint (Supabase Edge Function, Deno).
//
// POST { mode: "rag" | "llm_only", jd, title?, resume?, embedding: number[384],
//        jd_skills?: string[], k?, category? }
//   rag      -> retrieve resume chunks + O*NET occupation (pgvector), prompt the LLM
//               with that evidence, validate citations, log the run
//   llm_only -> same prompt without retrieved evidence (the "no retrieval" comparison)
//
// The query embedding is computed in the browser with the same sentence-transformer
// used to build the index (all-MiniLM-L6-v2), so no model runs here.
// Prompt text and formatting mirror src/scholarstack/generate.py exactly.

import { createClient } from "jsr:@supabase/supabase-js@2";
import prompts from "../_shared/prompts.json" with { type: "json" };

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

const LLM_BASE_URL = Deno.env.get("LLM_BASE_URL") ?? "https://api.groq.com/openai/v1";
const LLM_MODEL = Deno.env.get("LLM_MODEL") ?? "llama-3.3-70b-versatile";
const LLM_FALLBACK_MODEL = Deno.env.get("LLM_FALLBACK_MODEL") ?? "llama-3.1-8b-instant";
const LLM_KEY = Deno.env.get("GROQ_API_KEY") ?? "";
const MAX_INPUT_CHARS = 15000;

function serviceKey(): string {
  const legacy = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (legacy) return legacy;
  try {
    const keys = JSON.parse(Deno.env.get("SUPABASE_SECRET_KEYS") ?? "{}");
    return keys.default ?? Object.values(keys)[0] ?? "";
  } catch {
    return "";
  }
}

const db = createClient(Deno.env.get("SUPABASE_URL")!, serviceKey(), { auth: { persistSession: false } });

// ---------------------------------------------------------------- formatting
const fill = (tpl: string, v: Record<string, string>) => tpl.replace(/\{\{(\w+)\}\}/g, (_, k) => v[k] ?? "");

function truncateWords(text: string, n: number): string {
  const words = (text ?? "").split(/\s+/).filter(Boolean);
  return words.slice(0, n).join(" ") + (words.length > n ? " ..." : "");
}

function pretty(category: string): string {
  const special: Record<string, string> = { HR: "HR", BPO: "BPO", "INFORMATION-TECHNOLOGY": "Information Technology" };
  return special[category] ??
    category.split("-").map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(" ");
}

type Chunk = { chunk_id: string; resume_id: string; category: string; resume_title: string; section: string;
  content: string; skills: string[]; similarity: number };
type Named = { name: string; importance?: number; share?: number };

function formatEvidence(chunks: Chunk[]): string {
  return chunks.map((c) =>
    `[${c.chunk_id}] (${pretty(c.category)} | ${c.resume_title} | ${c.section}) ${truncateWords(c.content, 110)}`
  ).join("\n");
}

// deno-lint-ignore no-explicit-any
function formatOnet(o: any): string {
  const skills = [...(o.essential_skills ?? []), ...(o.transferable_skills ?? [])]
    .sort((a: Named, b: Named) => (b.importance ?? 0) - (a.importance ?? 0)).slice(0, 8);
  const lines = [
    `${o.title} (O*NET-SOC ${o.code}, job zone ${o.job_zone ?? "n/a"})`,
    o.description,
    "Key skills: " + skills.map((s: Named) => s.name).join(", "),
    "Knowledge: " + (o.knowledge ?? []).slice(0, 5).map((k: Named) => k.name).join(", "),
    "Technologies: " + (o.technologies ?? []).slice(0, 12).map((t: Named) => t.name).join(", "),
    "Typical tasks: " + (o.tasks ?? []).slice(0, 4).join(" | "),
  ];
  const m = o.market;
  if (m?.top_skills?.length) {
    const top = m.top_skills.slice(0, 10).map((s: Named) => `${s.name} (${Math.round(100 * (s.share ?? 0))}%)`)
      .join(", ");
    lines.push(`Demand across ${m.n_postings} LinkedIn postings: ${top}`);
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------- LLM
async function chat(messages: { role: string; content: string }[]) {
  let lastError = "";
  for (const model of [LLM_MODEL, LLM_FALLBACK_MODEL]) {
    for (let attempt = 0; attempt < 2; attempt++) {
      const res = await fetch(`${LLM_BASE_URL}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${LLM_KEY}` },
        body: JSON.stringify({ model, messages, temperature: 0.2, max_tokens: 2200,
          response_format: { type: "json_object" } }),
      });
      if (res.ok) {
        const data = await res.json();
        return { data, model };
      }
      lastError = `${res.status} ${(await res.text()).slice(0, 300)}`;
      if (res.status === 429 && attempt === 0) {
        const wait = Math.min(Number(res.headers.get("retry-after") ?? "2"), 8);
        await new Promise((r) => setTimeout(r, wait * 1000));
        continue;
      }
      break; // try the fallback model
    }
  }
  throw new Error(`LLM request failed: ${lastError}`);
}

function parseJson(text: string) {
  const cleaned = text.trim().replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
  return JSON.parse(cleaned);
}

// deno-lint-ignore no-explicit-any
function validateCitations(out: any, ids: Set<string>): number {
  let dropped = 0;
  const lists = [out.summary ?? [], out.education ?? [], out.certifications ?? [],
    // deno-lint-ignore no-explicit-any
    ...(out.experience ?? []).map((r: any) => r.bullets ?? [])];
  for (const list of lists) {
    for (const s of list) {
      const src: string[] = Array.isArray(s.sources) ? s.sources : [];
      const keep = src.filter((x) => ids.has(x));
      dropped += src.length - keep.length;
      s.sources = keep;
    }
  }
  return dropped;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { ...CORS, "Content-Type": "application/json" } });

// ---------------------------------------------------------------- handler
Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  if (req.method === "GET") return json({ ok: true, model: LLM_MODEL, fallback: LLM_FALLBACK_MODEL });
  if (req.method !== "POST") return json({ error: "POST only" }, 405);

  const started = Date.now();
  try {
    const body = await req.json();
    const mode: "rag" | "llm_only" = body.mode === "llm_only" ? "llm_only" : "rag";
    const jd = String(body.jd ?? "").slice(0, MAX_INPUT_CHARS);
    const resume = body.resume ? String(body.resume).slice(0, MAX_INPUT_CHARS) : "";
    const title = String(body.title ?? "").slice(0, 200);
    const embedding: number[] = body.embedding;
    const k = Math.min(Math.max(Number(body.k ?? 8), 1), 20);
    const category = body.category || null;
    const jdSkills: string[] = Array.isArray(body.jd_skills) ? body.jd_skills.slice(0, 40).map(String) : [];

    if (jd.split(/\s+/).length < 20) return json({ error: "Please paste a full job description (20+ words)." }, 400);
    if (!Array.isArray(embedding) || embedding.length !== 384) return json({ error: "embedding must have 384 numbers" }, 400);

    const common = {
      title: title || "(not given)",
      jd: truncateWords(jd, 900),
      jd_skills: jdSkills.join(", ") || "(none detected)",
      applicant: resume ? fill(prompts.applicant_block, { resume: truncateWords(resume, 900) }) : "",
      schema: prompts.schema,
    };

    let chunks: Chunk[] = [];
    // deno-lint-ignore no-explicit-any
    let occupation: any = null;
    let messages;
    if (mode === "rag") {
      const [c, o] = await Promise.all([
        db.rpc("match_resume_chunks", { query_embedding: embedding, match_count: k, filter_category: category }),
        db.rpc("match_occupations", { query_embedding: embedding, match_count: 1 }),
      ]);
      if (c.error) throw new Error(`retrieval failed: ${c.error.message}`);
      if (o.error) throw new Error(`O*NET match failed: ${o.error.message}`);
      chunks = c.data as Chunk[];
      occupation = o.data?.[0] ?? null;
      messages = [
        { role: "system", content: prompts.system_rag },
        { role: "user", content: fill(prompts.user_rag, { ...common,
          onet: occupation ? formatOnet(occupation) : "(no match)", evidence: formatEvidence(chunks) }) },
      ];
    } else {
      messages = [
        { role: "system", content: prompts.system_llm_only },
        { role: "user", content: fill(prompts.user_llm_only, common) },
      ];
    }

    const { data, model } = await chat(messages);
    const output = parseJson(data.choices?.[0]?.message?.content ?? "{}");
    const dropped = mode === "rag" ? validateCitations(output, new Set(chunks.map((c) => c.chunk_id))) : 0;
    const latency = Date.now() - started;

    const { data: run } = await db.from("runs").insert({
      mode, job_title: title, jd_chars: jd.length, has_resume: !!resume, category, k, model,
      retrieved_ids: chunks.map((c) => c.chunk_id), onet_code: occupation?.code ?? null,
      // the gap analysis can quote the applicant's resume, which is never stored
      output: { ...output, gap_analysis: output.gap_analysis ? "omitted" : null },
      latency_ms: latency, prompt_tokens: data.usage?.prompt_tokens, output_tokens: data.usage?.completion_tokens,
    }).select("id").single();

    return json({
      run_id: run?.id ?? null, mode, model, latency_ms: latency, usage: data.usage ?? null,
      invalid_citations_removed: dropped, retrieved: chunks, occupation, output,
    });
  } catch (err) {
    console.error(err);
    return json({ error: err instanceof Error ? err.message : String(err) }, 500);
  }
});
