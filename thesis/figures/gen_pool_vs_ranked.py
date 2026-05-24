"""Generates the pool-ceiling vs ranked-recall comparison figure for Kapitel 7."""

import matplotlib.pyplot as plt
import numpy as np

systems = ["RAG", "Embedding-1Hop", "Embedding-2Hop", "GraphRAG-1Hop", "GraphRAG-2Hop"]
pool_recall_ceiling = [0.62, 2.54, 3.38, 67.4, 68.2]
ranked_recall_at20_indegree = [0.6, 2.4, 3.1, 12.7, 12.5]

x = np.arange(len(systems))
width = 0.38

fig, ax = plt.subplots(figsize=(9.5, 4.5))
bars1 = ax.bar(x - width / 2, pool_recall_ceiling, width,
               label="Pool-Recall-Ceiling (post_cap)", color="#4c78a8")
bars2 = ax.bar(x + width / 2, ranked_recall_at20_indegree, width,
               label="Ranked Recall@20 (In-Degree)", color="#f28e2b")

for b in bars1:
    ax.annotate(f"{b.get_height():.1f}%",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)
for b in bars2:
    ax.annotate(f"{b.get_height():.1f}%",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Recall (%)")
ax.set_xticks(x)
ax.set_xticklabels(systems, rotation=15, ha="right")
ax.set_ylim(0, 80)
ax.legend(loc="upper left", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("/home/ags/github/kg-rag-legal/thesis/figures/pool-vs-ranked-recall.png",
            dpi=200, bbox_inches="tight")
print("written: thesis/figures/pool-vs-ranked-recall.png")
