"""Step 14 — Results figures for the report and slides (from reports/results/*.csv).

Run after the evaluation steps; figures for missing results are skipped.
"""

import numpy as np
import pandas as pd

from scholarstack.categories import pretty
from scholarstack.config import log, path
from scholarstack.viz import INK_2, MUTED, SERIES, SURFACE, hbar, plt, savefig

R = path("results")
SYSTEM_LABEL = {"baseline": "TF-IDF keywords", "llm_only": "LLM only", "rag": "RAG (ours)",
                "rag_no_onet": "RAG without O*NET", "rag_small": "RAG, 8B model", "rag_k3": "RAG, k = 3",
                "rag_k15": "RAG, k = 15"}


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)


def retrieval_figures():
    f = R / "retrieval_eval.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    mc = df[df.experiment == "model_chunking"]
    series = [("section", "all-MiniLM-L6-v2", "MiniLM, section chunks (served)"),
              ("section", "gte-small", "GTE-small, section"),
              ("section", "bge-small-en-v1_5", "BGE-small, section"),
              ("fixed", "all-MiniLM-L6-v2", "MiniLM, fixed windows"),
              ("section", "tf-idf", "TF-IDF baseline")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, metric, title in ((axes[0], "precision_category", "Category relevance"),
                              (axes[1], "precision_skill", "Skill-overlap relevance")):
        for i, (ch, model, label) in enumerate(series):
            s = mc[(mc.chunking == ch) & (mc.model == model)].sort_values("k")
            ax.plot(s.k, s[metric], color=SERIES[i], marker="o", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=label)
        ax.set_title(title)
        ax.set_xlabel("k (retrieved chunks)")
        ax.set_xticks(sorted(mc.k.unique()))
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylabel("Precision@k")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06), fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "results_retrieval_k_sweep", "240 held-out job descriptions, 10 per category. Query = title + "
            "extracted skills. Skill-overlap favours lexical matching by construction.")

    q = df[(df.experiment == "query") & (df.k == 8)].set_index("query")
    labels = {"raw": "Raw job description", "skills": "Title + extracted skills", "hybrid": "Title + skills + lead text"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.6), sharey=True)
    for ax, metric, title in ((axes[0], "precision_category", "Category P@8"),
                              (axes[1], "precision_skill", "Skill-overlap P@8")):
        order = ["raw", "hybrid", "skills"]
        hbar(ax, [labels[o] for o in order], [q.loc[o, metric] for o in order], fmt="{:.1%}")
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    savefig(fig, "results_query_construction", "MiniLM, section chunks. Embedding the extracted skills avoids the "
            "256-token truncation of long ads that open with company boilerplate.")

    k8 = mc[mc.k == 8]
    models = ["all-MiniLM-L6-v2", "gte-small", "bge-small-en-v1_5", "tf-idf"]
    names = {"all-MiniLM-L6-v2": "MiniLM-L6", "gte-small": "GTE-small", "bge-small-en-v1_5": "BGE-small",
             "tf-idf": "TF-IDF"}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(models))
    for j, (ch, lab) in enumerate((("section", "Section chunks"), ("fixed", "Fixed windows"))):
        vals = [k8[(k8.model == m) & (k8.chunking == ch)]["precision_category"].iloc[0] for m in models]
        ax.barh(y + (j - 0.5) * 0.36, vals, height=0.34, color=SERIES[j], label=lab)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi + (j - 0.5) * 0.36, f"{v:.0%}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [names[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Category P@8 by embedding model and chunking")
    ax.legend(loc="lower right")
    savefig(fig, "results_model_chunking")

    f = R / "retrieval_by_category.csv"
    if f.exists():
        bc = pd.read_csv(f).sort_values("p@8_category", ascending=False)
        fig, ax = plt.subplots(figsize=(7.5, 7))
        hbar(ax, [pretty(c) for c in bc.category], bc["p@8_category"].values, fmt="{:.0%}")
        mean = bc["p@8_category"].mean()
        ax.axvline(mean, color=INK_2, linewidth=1)
        ax.text(mean, -0.9, f"mean {mean:.0%}", ha="center", fontsize=8.5, color=INK_2)
        ax.set_xlim(0, 1.1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title("Category P@8 per job category (served configuration)")
        savefig(fig, "results_retrieval_by_category", "Broad or keyword-defined categories (BPO, Automobile, "
                "Agriculture, Advocate) are hardest: their resumes span many occupations.")


def faiss_figure():
    f = R / "faiss_benchmark.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    idx_order = ["Flat (exact)", "HNSW (M=32)", "IVF-Flat"]
    for ax, metric, title, fmt in ((axes[0], "latency_ms", "Query latency (ms, one query at a time)", "{:.2f}"),
                                   (axes[1], "recall@10", "Recall@10 vs exact search", "{:.1%}")):
        y = np.arange(len(idx_order))
        for j, corpus in enumerate(["resume chunks", "job postings"]):
            sub = df[df.corpus == corpus].set_index("index").loc[idx_order]
            ax.barh(y + (j - 0.5) * 0.36, sub[metric], height=0.34, color=SERIES[j],
                    label=f"{corpus} ({int(sub['vectors'].iloc[0]):,} vectors)")
            for yi, v in zip(y, sub[metric]):
                ax.text(v * 1.02, yi + (j - 0.5) * 0.36, fmt.format(v), va="center", fontsize=8, color=INK_2)
        ax.set_yticks(y, idx_order)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[0].set_xlim(0, df["latency_ms"].max() * 1.25)
    axes[1].set_xlim(0, 1.15)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    savefig(fig, "results_faiss_benchmark", "500 queries, 384-d MiniLM vectors, CPU. HNSW gives ~8x lower latency "
            "than exact search on 106k postings at 99% recall.")


def generation_figures():
    f = R / "generation_summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f).set_index("system")
    order = [x for x in ["baseline", "llm_only", "rag_small", "rag_no_onet", "rag_k3", "rag", "rag_k15"] if x in s.index]
    metrics = [("grounded", "Grounded statements"), ("jd_skill_coverage", "JD skill coverage"),
               ("market_skill_coverage", "Market (O*NET-linked) skill coverage")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        vals = [s.loc[o, m] if m in s and not pd.isna(s.loc[o, m]) else 0 for o in order]
        colors = [SERIES[0] if o == "rag" else "#86b6ef" for o in order]
        y = np.arange(len(order))
        ax.barh(y, vals, color=colors, height=0.62)
        for yi, v, o in zip(y, vals, order):
            txt = "n/a" if (m == "grounded" and o == "baseline") else f"{v:.0%}"
            ax.text(v + 0.01, yi, txt, va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(y, [SYSTEM_LABEL[o] for o in order])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title)
    fig.tight_layout()
    n = int(s["n"].max())
    savefig(fig, "results_generation", f"{n} held-out job descriptions (one per category). Grounded = statement has "
            "cosine >= 0.55 with a real resume chunk retrieved for that job.")

    f = R / "groundedness_threshold_curve.csv"
    if f.exists():
        c = pd.read_csv(f).set_index("system")
        taus = [float(col.split("@")[1]) for col in c.columns]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for i, o in enumerate([x for x in ["rag", "rag_no_onet", "rag_small", "llm_only"] if x in c.index]):
            ax.plot(taus, c.loc[o].values, color=SERIES[i], marker="o", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=SYSTEM_LABEL[o])
        ax.axvline(0.55, color=MUTED, linewidth=1)
        ax.set_xlabel("Support threshold (cosine)")
        ax.set_ylabel("Share of statements supported")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, 1.05)
        ax.set_title("Groundedness is robust to the threshold choice")
        ax.legend(loc="lower left")
        savefig(fig, "results_groundedness_threshold")

    f = R / "judge_eval.csv"
    if f.exists():
        j = pd.read_csv(f)
        pref = j[j.criterion == "preferred"]
        if len(pref):
            fig, ax = plt.subplots(figsize=(8, 2.2))
            for yi, (pair, g) in enumerate(pref.groupby("pair")):
                a, b = pair.split(" vs ")
                shares = [(g.winner == a).mean(), (g.winner == "tie").mean(), (g.winner == b).mean()]
                left = 0
                for share, color, lab in zip(shares, [SERIES[0], "#c3c2b7", SERIES[1]],
                                             [SYSTEM_LABEL[a], "Tie", SYSTEM_LABEL[b]]):
                    if share > 0:
                        ax.barh(yi, share, left=left, color=color, height=0.5, edgecolor=SURFACE, linewidth=2)
                        if share >= 0.08:
                            ax.text(left + 0.01, yi, f"{share:.0%}", va="center", fontsize=8.5,
                                    color="#ffffff" if color != "#c3c2b7" else "#0b0b0b")
                    left += share
                ax.text(1.01, yi, f"{SYSTEM_LABEL[a]} vs {SYSTEM_LABEL[b]}", va="center", fontsize=9, color=INK_2)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.grid(False)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_title("LLM-judge preference (randomised A/B order)")
            from matplotlib.patches import Patch

            ax.legend(handles=[Patch(color=SERIES[0], label="RAG (ours) preferred"), Patch(color="#c3c2b7", label="Tie"),
                               Patch(color=SERIES[1], label="Other system preferred")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
            savefig(fig, "results_judge")


if __name__ == "__main__":
    retrieval_figures()
    faiss_figure()
    generation_figures()
    log.info("results figures written to %s", path("figures"))
    print(sorted(p.name for p in path("figures").glob("results_*.png")))
