// Sentence embeddings in the browser with transformers.js - the same
// all-MiniLM-L6-v2 model (mean pooling + L2 norm) used to build the index.
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { CONFIG } from "./config.js?v=202611161640";

env.allowLocalModels = false;
let extractor = null;

export function loadEmbedder(onProgress) {
  extractor ??= pipeline("feature-extraction", CONFIG.embeddingModel, {
    dtype: CONFIG.embeddingDtype ?? "fp32",
    progress_callback: onProgress,
  });
  return extractor;
}

/** Embed texts -> array of unit-length Float32 vectors. */
export async function embed(texts) {
  const model = await loadEmbedder();
  const out = await model(texts, { pooling: "mean", normalize: true });
  return out.tolist();
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
