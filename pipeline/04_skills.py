"""Step 4 - Skill extraction: build the O*NET-aligned lexicon and extract skills
from every resume and every job posting (postings in parallel with Dask).

Output: data/processed/{skills_lexicon.json, resume_skills.parquet, posting_skills/},
        reports/results/top_skills_by_category.csv
"""
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster

from scholarstack.config import log, path
from scholarstack.skills import SkillExtractor, build_lexicon, save_lexicon


def _extract_partition(df: pd.DataFrame) -> pd.DataFrame:
    ex = SkillExtractor()  # built once per partition; cheap
    text = df["title"].fillna("") + "\n" + df["description"].fillna("") + "\n" + df["skills_desc"].fillna("")
    return pd.DataFrame({"job_id": df["job_id"].values,
                         "category": df["category"].values,
                         "skills": pd.array(["|".join(ex.extract(t)) for t in text], dtype="string")})


if __name__ == "__main__":
    lexicon = build_lexicon()
    save_lexicon(lexicon)
    ex = SkillExtractor(lexicon)

    resumes = pd.read_parquet(path("processed") / "resumes.parquet")
    resumes["skills"] = ["|".join(ex.extract(t)) for t in resumes["text"]]
    resumes[["resume_id", "category", "skills"]].to_parquet(path("processed") / "resume_skills.parquet", index=False)
    log.info("resumes: mean %.1f skills per resume", resumes["skills"].str.count(r"\|").add(1).mean())

    with LocalCluster(n_workers=4, threads_per_worker=1, memory_limit="3GB") as cluster, Client(cluster):
        postings = dd.read_parquet(path("processed") / "postings")
        meta = pd.DataFrame({"job_id": pd.Series(dtype="int64"), "category": pd.Series(dtype="string"),
                             "skills": pd.Series(dtype="string")})
        out = path("processed") / "posting_skills"
        postings.map_partitions(_extract_partition, meta=meta).to_parquet(out, write_index=False, overwrite=True)
    ps = pd.read_parquet(path("processed") / "posting_skills")
    log.info("postings: %d processed, %.1f%% with >=1 skill", len(ps), 100 * (ps["skills"] != "").mean())

    # Top skills per category, for resumes and postings side by side (used in EDA).
    rows = []
    for source, df in (("resumes", resumes), ("postings", ps.dropna(subset=["category"]))):
        exploded = df.assign(skill=df["skills"].str.split("|")).explode("skill")
        exploded = exploded[exploded["skill"].fillna("") != ""]
        n_docs = df.groupby("category").size()
        top = exploded.groupby(["category", "skill"]).size().rename("docs").reset_index()
        top["share"] = top["docs"] / top["category"].map(n_docs)
        top = top.sort_values(["category", "docs"], ascending=[True, False]).groupby("category").head(25)
        top["source"] = source
        top["name"] = top["skill"].map(lambda s: ex.by_id[s]["name"])
        rows.append(top)
    pd.concat(rows).to_csv(path("results") / "top_skills_by_category.csv", index=False)
    log.info("top skills per category written")
