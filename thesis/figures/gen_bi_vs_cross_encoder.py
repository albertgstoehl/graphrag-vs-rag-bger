"""Generates the Bi-Encoder vs Cross-Encoder architecture comparison figure for Kapitel 2.1.3."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INPUT_COLOR = "#e8e8e8"
INPUT_TEXT = "black"
ENCODER_COLOR = "#4c78a8"
ENCODER_TEXT = "white"
HEAD_COLOR = "#f28e2b"
HEAD_TEXT = "black"


def box(ax, x, y, w, h, text, color, text_color, fontsize=10, weight="bold"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.2, edgecolor=color, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=weight)


def label(ax, x, y, text, fontsize=12, color="black"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color)


def arrow(ax, x1, y1, x2, y2, color="#444"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4))


fig, (ax_bi, ax_cross) = plt.subplots(1, 2, figsize=(11, 5.2))

# ==== LEFT: Bi-Encoder ====
ax_bi.set_xlim(0, 10)
ax_bi.set_ylim(0, 10)
ax_bi.set_aspect("equal")
ax_bi.axis("off")
ax_bi.set_title("Bi-Encoder", fontsize=13, fontweight="bold", pad=8)

box(ax_bi, 0.4, 8.5, 3.6, 1.0, "Query", INPUT_COLOR, INPUT_TEXT, fontsize=11)
box(ax_bi, 6.0, 8.5, 3.6, 1.0, "Dokument", INPUT_COLOR, INPUT_TEXT, fontsize=11)

box(ax_bi, 0.4, 6.0, 3.6, 1.5, "Encoder", ENCODER_COLOR, ENCODER_TEXT, fontsize=11)
box(ax_bi, 6.0, 6.0, 3.6, 1.5, "Encoder", ENCODER_COLOR, ENCODER_TEXT, fontsize=11)

arrow(ax_bi, 2.2, 8.5, 2.2, 7.55)
arrow(ax_bi, 7.8, 8.5, 7.8, 7.55)

label(ax_bi, 2.2, 5.0, r"$q$", fontsize=15)
label(ax_bi, 7.8, 5.0, r"$d$", fontsize=15)

arrow(ax_bi, 2.2, 6.0, 2.2, 5.3)
arrow(ax_bi, 7.8, 6.0, 7.8, 5.3)

box(ax_bi, 3.5, 2.5, 3.0, 1.3, r"$q \cdot d$", HEAD_COLOR, HEAD_TEXT, fontsize=13)

arrow(ax_bi, 2.5, 4.7, 4.1, 3.85)
arrow(ax_bi, 7.5, 4.7, 5.9, 3.85)

label(ax_bi, 5.0, 1.3, "Relevanz-Score", fontsize=11)
arrow(ax_bi, 5.0, 2.5, 5.0, 1.7)

# ==== RIGHT: Cross-Encoder ====
ax_cross.set_xlim(0, 10)
ax_cross.set_ylim(0, 10)
ax_cross.set_aspect("equal")
ax_cross.axis("off")
ax_cross.set_title("Cross-Encoder", fontsize=13, fontweight="bold", pad=8)

box(ax_cross, 0.4, 8.5, 9.2, 1.0,
    "[CLS]  Query  [SEP]  Dokument  [SEP]",
    INPUT_COLOR, INPUT_TEXT, fontsize=10, weight="normal")

box(ax_cross, 0.4, 6.0, 9.2, 1.5,
    "Transformer (gemeinsamer Forward-Pass)",
    ENCODER_COLOR, ENCODER_TEXT, fontsize=11)

arrow(ax_cross, 5.0, 8.5, 5.0, 7.55)

label(ax_cross, 5.0, 5.0, r"$h_{[\mathrm{CLS}]}$", fontsize=14)
arrow(ax_cross, 5.0, 6.0, 5.0, 5.3)

box(ax_cross, 3.5, 2.5, 3.0, 1.3, "Linear-Head", HEAD_COLOR, HEAD_TEXT, fontsize=11)

arrow(ax_cross, 5.0, 4.7, 5.0, 3.85)

label(ax_cross, 5.0, 1.3, "Relevanz-Score", fontsize=11)
arrow(ax_cross, 5.0, 2.5, 5.0, 1.7)

plt.tight_layout()
plt.savefig("/home/ags/github/kg-rag-legal/thesis/figures/bi-vs-cross-encoder.png",
            dpi=200, bbox_inches="tight")
print("written: thesis/figures/bi-vs-cross-encoder.png")
