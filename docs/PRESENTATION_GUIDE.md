# Presenting ScholarStack

Site: https://hrushh22.github.io/ScholarStack/

## The day before and 10 minutes before

- Open the site once on the laptop you'll present from. The first visit downloads a 90 MB embedding model; after
  that it's cached and the demo starts instantly. The status line under the button says "Ready." when it's loaded.
- Do one test run with the Staff Accountant sample so you know it's all working.
- Don't run `pipeline/10_eval_generation.py` on presentation day. It uses the same free Groq quota as the live demo.
- Pick light or dark mode with the Theme button (dark looks great on a projector).
- Keep a few screenshots of a finished run as a backup in case the venue Wi-Fi is bad.

## A 10 to 12 minute flow

| Time | Part | What to show |
|---|---|---|
| 1 min | The problem | Resume advice is usually generic keyword lists. We wanted advice backed by what people in the job actually wrote. |
| 1.5 min | Data | Results tab, top tiles: 2,480 resumes in 24 categories, 123,849 LinkedIn postings cleaned to 106,291 with Dask, O*NET 31.0 with 1,016 occupations. Scroll to the EDA figures (category imbalance, skills in resumes vs postings). |
| 2 min | How it works | How it works tab: the offline pipeline (download, clean, skills, chunk and embed, FAISS, O*NET linking, evaluation) and the live path (browser, Supabase, Groq). |
| 4 min | Live demo | See the script below. |
| 2.5 min | Results | Results tab: retrieval charts, "What we embed matters", the grounding chart, skill coverage, FAISS benchmark. |
| 1 min | Limits and next steps | How it works tab, Limitations card. |

## Demo script

1. Generate tab. In "Pick a real posting to try" choose **Staff Accountant**. Point out the skills that pop up under
   the description: that's step 1, rule-based extraction with our 1,457-skill dictionary.
2. In "Your resume" pick **Accounting graduate**. Open Options and tick **Also run the plain LLM**.
3. Click **Generate reference resume** and narrate the pipeline steps as they light up: skills, embedding in the
   browser, pgvector search, O*NET match, the LLM writing, then the check.
4. Reference resume tab: click one of the little numbered citations. The popup shows the real resume excerpt that line
   came from. Mention the green dots (backed by evidence) and that bad citations get removed automatically.
5. Evidence tab: the O*NET occupation it matched, what LinkedIn postings for that occupation ask for, and the 8 resume
   excerpts.
6. Gap analysis tab: which of the posting's skills the student has, what's missing, the advisor notes and the bullet
   rewrites.
7. Compare tab: keyword baseline vs plain LLM vs ScholarStack, side by side. Click a star rating to show that we
   collect human ratings too.

If a run fails with a rate limit message, wait about a minute and click Generate again. The site automatically falls
back to the smaller model if the big one is busy.

## Numbers worth saying out loud

- Retrieval: 62% of the top 8 excerpts come from the right job category, vs 32% for TF-IDF.
- Embedding "title + extracted skills" beat embedding the whole job ad (61.6% vs 56.5%), because ads often open with
  company boilerplate and the model only reads 256 tokens.
- MiniLM matched GTE-small and embeds a query six times faster (0.8 ms vs 5.0 ms), which is why it runs in the browser.
- FAISS on 106k postings: 1.96 ms per query exact, 0.25 ms with HNSW at 99.1% recall.
- Grounding: 90% of ScholarStack's lines are backed by real resumes vs 52% for the same LLM without retrieval. On the
  12 postings where both were checked it's 88% vs 45%.
- Skill coverage: ScholarStack covers 81% of the posting's skills vs 68% for the plain LLM and 30% for TF-IDF.
- The live database search returns the same top 8 as FAISS exact search.

## Questions you'll probably get

**How do you know it isn't just making things up?**
Every line has to cite the resume excerpts it used, and citations to anything we didn't retrieve are removed. Then a
second model from a different family checks each line against the actual resume text. 90% passed for ScholarStack
vs 52% for the LLM alone.

**Why not just ask ChatGPT?**
That's our "LLM only" baseline. It writes fluent but generic lines that real resumes mostly don't back up, and it
can't point to a source.

**Why does TF-IDF score higher on the skill-overlap chart?**
That measure counts exact keyword matches, which is exactly what TF-IDF does, so it's favored by design. On the
category measure, which needs actual meaning, dense retrieval nearly doubles TF-IDF.

**Where's the big data part?**
123,849 postings (517 MB) cleaned and de-duplicated on a Dask cluster, all 106k embedded in batches on the GPU, and
indexed with FAISS. We benchmarked exact vs HNSW vs IVF indexes.

**Why these models?**
All free. MiniLM for embeddings (fast and small enough for the browser), gpt-oss-120b on Groq for writing, Qwen as an
independent checker and judge, so the model grading the output isn't the one that wrote it.

**What about privacy?**
A resume you paste is only used for that request and never stored. Raw resume text from the dataset isn't in the
GitHub repo. API keys stay on the server.

**What doesn't work well?**
Broad categories like BPO and Automobile, because those resumes cover many unrelated jobs. The resumes are examples
from one website, not verified hires. Our relevance labels are stand-ins, and the checker is itself an LLM.

**What would you do next?**
More and better resume sources, a fine-tuned retriever, a user study with real students, and using the same pipeline
for course recommendations (our second project idea).

## If something breaks during the demo

- "Rate limit": wait a minute, try again.
- The page is slow the very first time: that's the model download. It's instant after that.
- No internet: the skill extraction and keyword baseline still run in the browser. Switch to your backup screenshots
  for the rest.
