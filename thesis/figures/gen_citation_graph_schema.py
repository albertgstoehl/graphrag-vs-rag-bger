"""Generates the citation graph schema figure (entity-type level) for Kapitel 2.2."""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

ENTSCHEID_COLOR = "#4c78a8"
LAW_COLOR = "#54a24b"
EDGE_COLOR = "#3a3a3a"

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.axis("off")

# Entscheid (circle, left)
ent_cx, ent_cy = 3.0, 3.0
ent_r = 0.95
ax.add_patch(Circle((ent_cx, ent_cy), ent_r, facecolor=ENTSCHEID_COLOR,
                    edgecolor="black", linewidth=1.4, zorder=3))
ax.text(ent_cx, ent_cy, "Entscheid", ha="center", va="center",
        fontsize=12, color="white", fontweight="bold", zorder=4)

# Gesetzesartikel (rectangle, right)
law_cx, law_cy = 7.5, 3.0
law_w, law_h = 2.2, 1.1
ax.add_patch(Rectangle((law_cx - law_w / 2, law_cy - law_h / 2), law_w, law_h,
                       facecolor=LAW_COLOR, edgecolor="black", linewidth=1.4,
                       zorder=3))
ax.text(law_cx, law_cy, "Gesetzesartikel", ha="center", va="center",
        fontsize=12, color="white", fontweight="bold", zorder=4)

# Self-loop on Entscheid (case_to_case)
loop_start = (ent_cx - 0.35, ent_cy + ent_r * 0.95)
loop_end = (ent_cx + 0.35, ent_cy + ent_r * 0.95)
loop = FancyArrowPatch(loop_start, loop_end,
                       connectionstyle="arc3,rad=-2.5",
                       arrowstyle="-|>", mutation_scale=20,
                       linewidth=1.8, color=EDGE_COLOR)
ax.add_patch(loop)
ax.text(ent_cx, ent_cy + ent_r + 1.45, "case_to_case",
        ha="center", va="center", fontsize=12, color=EDGE_COLOR,
        family="monospace")

# Arrow Entscheid -> Gesetzesartikel (case_to_law)
arrow = FancyArrowPatch((ent_cx + ent_r, ent_cy),
                        (law_cx - law_w / 2, law_cy),
                        arrowstyle="-|>", mutation_scale=22,
                        linewidth=1.8, color=EDGE_COLOR)
ax.add_patch(arrow)
ax.text((ent_cx + ent_r + law_cx - law_w / 2) / 2, ent_cy + 0.3,
        "case_to_law", ha="center", va="center", fontsize=12,
        color=EDGE_COLOR, family="monospace")

plt.tight_layout()
plt.savefig("/home/ags/github/kg-rag-legal/thesis/figures/citation-graph-schema.png",
            dpi=200, bbox_inches="tight")
print("written: thesis/figures/citation-graph-schema.png")
