"""Step 9 — Exploratory data analysis: figures and summary tables for the report.

Output: reports/figures/eda_*.png, reports/results/eda_summary.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from scholarstack.categories import CATEGORIES, pretty
from scholarstack.chunking import load_chunks
from scholarstack.config import CFG, log, path, seed_everything
from scholarstack.embed import load_embeddings
from scholarstack.viz import BLUE_RAMP, INK_2, MUTED, OTHER, SERIES, hbar, plt, savefig

if __name__ == "__main__":
    seed_everything()
    prof = json.loads((path("results") / "data_profile.json").read_text())
    resumes = pd.read_parquet(path("processed") / "resumes.parquet")
    postings = pd.read_parquet(path("processed") / "postings")
    raw_postings = pd.read_parquet(path("interim") / "postings_raw")
    summary = {}

    # 1. Resume category distribution (class imbalance).
    counts = resumes["category"].value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 7))
    hbar(ax, [pretty(c) for c in counts.index], counts.values)
    ax.set_title("Resumes per job category")
    ax.set_xlabel("Resumes")
    savefig(fig, "eda_resume_categories", f"{len(resumes):,} resumes after cleaning. Largest/smallest class ratio "
            f"{counts.max() / counts.min():.1f}x (BPO and Automobile are under-represented).")
    summary["resume_category_imbalance"] = round(float(counts.max() / counts.min()), 2)

    # 2. Length distributions (inform chunk size).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, values, title, color in (
        (axes[0], resumes["n_words"], "Resume length", SERIES[0]),
        (axes[1], postings["n_words"].clip(upper=2000), "Job description length", SERIES[0]),
    ):
        ax.hist(values, bins=50, color=color, edgecolor="#fcfcfb", linewidth=1)
        med = float(np.median(values))
        ax.axvline(med, color=INK_2, linewidth=1)
        ax.text(med, ax.get_ylim()[1] * 0.95, f"  median {med:,.0f} words", color=INK_2, fontsize=8.5, va="top")
        ax.set_title(title)
        ax.set_xlabel("Words")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Documents")
    savefig(fig, "eda_lengths", "Job descriptions are clipped at 2,000 words for display.")
    summary["median_resume_words"] = int(resumes["n_words"].median())
    summary["median_jd_words"] = int(postings["n_words"].median())

    # 3. Missingness in raw postings.
    nulls = raw_postings.isna().mean().sort_values(ascending=False)
    nulls = nulls[nulls > 0].head(14)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    hbar(ax, nulls.index.tolist(), (100 * nulls.values).round(1), fmt="{:.0f}%")
    ax.set_title("Missing values in the raw LinkedIn postings")
    ax.set_xlabel("% of postings missing the field")
    savefig(fig, "eda_postings_missingness", f"{len(raw_postings):,} raw postings. Salary fields are missing for most "
            "postings and are not used by the core pipeline.")
    summary["salary_missing_pct"] = round(100 * float(raw_postings["normalized_salary"].isna().mean()), 1)

    # 4. Postings: industries, work types, experience levels, titles.
    inds = postings["industries"].dropna().str.split("|").explode().value_counts().head(12)
    work = postings["formatted_work_type"].value_counts()
    exp = postings["formatted_experience_level"].fillna("Not stated").value_counts()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    hbar(axes[0], inds.index.tolist(), inds.values)
    axes[0].set_title("Top industries")
    hbar(axes[1], work.index.tolist(), work.values)
    axes[1].set_title("Work type")
    hbar(axes[2], exp.index.tolist(), exp.values)
    axes[2].set_title("Experience level")
    for ax in axes:
        ax.set_xlabel("Postings")
    fig.tight_layout(w_pad=3)
    savefig(fig, "eda_postings_overview", f"{len(postings):,} cleaned, de-duplicated postings (2023-2024).")

    titles = postings["title"].str.strip().value_counts().head(15)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    hbar(ax, titles.index.tolist(), titles.values)
    ax.set_title("Most frequent posting titles")
    ax.set_xlabel("Postings")
    savefig(fig, "eda_top_titles")

    # 5. Coverage: postings mapped to each resume category vs resumes available.
    pc = postings["category"].value_counts().reindex(CATEGORIES).fillna(0)
    rc = resumes["category"].value_counts().reindex(CATEGORIES).fillna(0)
    order = pc.sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    hbar(axes[0], [pretty(c) for c in order], pc[order].values)
    axes[0].set_title("Postings whose title maps to the category")
    hbar(axes[1], [pretty(c) for c in order], rc[order].values, color=SERIES[0])
    axes[1].set_title("Resumes in the category")
    axes[1].tick_params(axis="y", labelleft=False)
    for ax in axes:
        ax.set_xlabel("Documents")
    fig.tight_layout(w_pad=2)
    unmapped = postings["category"].isna().mean()
    savefig(fig, "eda_category_coverage", f"{100 * (1 - unmapped):.0f}% of postings map to exactly one of the 24 "
            "resume categories by title; demand is concentrated in Healthcare, Engineering and Sales.")
    summary["postings_mapped_to_category_pct"] = round(100 * float(1 - unmapped), 1)

    # 6. Top skills: resumes vs postings, six example categories (small multiples).
    top = pd.read_csv(path("results") / "top_skills_by_category.csv")
    show = ["ACCOUNTANT", "INFORMATION-TECHNOLOGY", "HEALTHCARE", "CHEF", "TEACHER", "DESIGNER"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, cat in zip(axes.ravel(), show):
        t = top[top.category == cat]
        pst = t[t.source == "postings"].set_index("name")["share"]
        res = t[t.source == "resumes"].set_index("name")["share"]
        skills = pst.head(8).index.tolist()
        y = np.arange(len(skills))
        ax.hlines(y, [min(res.get(s, 0), pst[s]) for s in skills], [max(res.get(s, 0), pst[s]) for s in skills],
                  color="#e1e0d9", linewidth=2)
        ax.scatter([pst[s] for s in skills], y, color=SERIES[1], s=46, zorder=3, edgecolor="#fcfcfb",
                   linewidth=2, label="Job postings")
        ax.scatter([res.get(s, 0) for s in skills], y, color=SERIES[0], s=46, zorder=3, edgecolor="#fcfcfb",
                   linewidth=2, label="Resumes")
        ax.set_yticks(y, skills)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", visible=False)
        ax.set_title(pretty(cat))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(h_pad=2.5, w_pad=2, rect=(0, 0, 1, 0.96))
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1, 1))
    savefig(fig, "eda_top_skills", "Share of documents mentioning each skill (rule-based extraction). Gaps between "
            "the dots show where resumes under- or over-state what employers ask for.")

    # 7. Embedding space: resumes (mean of their chunk vectors), t-SNE, one panel per category.
    chunks = load_chunks("section")
    vecs = load_embeddings("chunks_section")
    chunks["row"] = np.arange(len(chunks))
    groups = chunks.groupby("resume_id")["row"].apply(list)
    rvecs = np.stack([vecs[r].mean(axis=0) for r in groups])
    rvecs /= np.linalg.norm(rvecs, axis=1, keepdims=True)
    rcat = chunks.groupby("resume_id")["category"].first().reindex(groups.index).values
    xy = TSNE(n_components=2, perplexity=35, init="pca", random_state=CFG["seed"]).fit_transform(rvecs)
    sil = silhouette_score(rvecs, rcat, metric="cosine")
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(rvecs)
    _, idx = nn.kneighbors(rvecs)
    purity = float(np.mean([(rcat[i[1:]] == rcat[i[0]]).mean() for i in idx]))
    fig, axes = plt.subplots(4, 6, figsize=(16, 11), sharex=True, sharey=True)
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        mask = rcat == cat
        ax.scatter(xy[~mask, 0], xy[~mask, 1], s=3, color=OTHER, alpha=0.35, linewidth=0)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=7, color=SERIES[0], linewidth=0)
        ax.set_title(pretty(cat), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Resume embeddings (all-MiniLM-L6-v2, t-SNE): each panel highlights one category",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color="#0b0b0b")
    fig.tight_layout()
    savefig(fig, "eda_embedding_space", f"10-NN category purity {purity:.2f} (chance ~0.04); cosine silhouette "
            f"{sil:.3f}. Occupation-specific categories (Chef, Teacher, Aviation) form tight clusters; broad ones "
            "(Consultant, BPO) scatter.")
    summary["embedding_knn_purity"] = round(purity, 3)
    summary["embedding_silhouette"] = round(float(sil), 4)
    summary["knn_purity_by_category"] = {c: round(float(np.mean([(rcat[i[1:]] == c).mean() for i in idx if rcat[i[0]] == c])), 3)
                                         for c in CATEGORIES}

    # 8. Outliers & data-quality numbers for the report.
    summary["resumes_raw"] = prof["resumes"]["rows"]
    summary["resumes_clean"] = len(resumes)
    summary["postings_raw"] = prof["postings"]["rows"]
    summary["postings_clean"] = len(postings)
    summary["postings_removed"] = prof["postings"]["rows"] - len(postings)
    summary["resume_outliers_long"] = prof["resumes"]["Resume_str_words"]["outliers_high"]
    summary["jd_outliers_long"] = prof["postings"]["description_words"]["outliers_high"]
    summary["onet_occupations"] = prof["onet"]["Occupation Data"]["rows"]
    (path("results") / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("EDA figures written to %s", path("figures"))
    print(json.dumps({k: v for k, v in summary.items() if k != "knn_purity_by_category"}, indent=2))
