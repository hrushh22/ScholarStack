// Supabase calls: retrieval RPCs (PostgREST), the generation Edge Function, feedback.
import { CONFIG } from "./config.js";

function headers() {
  const h = { "Content-Type": "application/json", apikey: CONFIG.supabaseKey };
  // Legacy anon keys are JWTs and also go in Authorization; new publishable keys do not.
  if (CONFIG.supabaseKey.startsWith("eyJ")) h.Authorization = `Bearer ${CONFIG.supabaseKey}`;
  return h;
}

async function call(url, body, extra = {}) {
  const res = await fetch(url, { method: "POST", headers: { ...headers(), ...extra }, body: JSON.stringify(body) });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data?.error ?? data?.message ?? `${res.status} ${res.statusText}`);
  return data;
}

export const isConfigured = () => Boolean(CONFIG.supabaseUrl && CONFIG.supabaseKey);

export function matchChunks(embedding, k, category) {
  return call(`${CONFIG.supabaseUrl}/rest/v1/rpc/match_resume_chunks`,
    { query_embedding: embedding, match_count: k, filter_category: category || null });
}

export function matchOccupations(embedding, n = 3) {
  return call(`${CONFIG.supabaseUrl}/rest/v1/rpc/match_occupations`, { query_embedding: embedding, match_count: n });
}

export function generate(payload) {
  return call(`${CONFIG.supabaseUrl}/functions/v1/generate`, payload);
}

export function sendFeedback(row) {
  return call(`${CONFIG.supabaseUrl}/rest/v1/feedback`, row, { Prefer: "return=minimal" });
}

export function usageStats() {
  return call(`${CONFIG.supabaseUrl}/rest/v1/rpc/usage_stats`, {});
}
