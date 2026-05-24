"""Two minimal schematics, one per expansion variant.

Query is rendered as a yellow star, candidates as black circles.
1-hop candidates are full-size black, 2-hop candidates are smaller and
mid-grey, so the two expansion stages are visually distinguishable.

- graph-expansion-schema.svg
  Query at top-left, dashed cosine-match link to the seed-candidate (root,
  best ANN-result), solid citation arrows from the root to its 1-hop
  neighbours, further solid arrows from two of the 1-hop neighbours to the
  2-hop layer.
- embedding-expansion-schema.svg
  Query at the centre with a dashed proximity circle marking the 1-hop
  kNN-Nachbarschaft. Each of the two 2-hop-expanded 1-hop-Nachbarn trägt
  ein eigenes kleines kNN-Kreis um sich herum, mit den 2-hop-Kandidaten
  innerhalb dieses Kreises. Damit ist sichtbar dass die 2-hop-Expansion
  pro 1-hop-Kandidat passiert.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

OUT_GRAPH = "/home/ags/github/kg-rag-legal/thesis/figures/graph-expansion-schema.svg"
OUT_EMB = "/home/ags/github/kg-rag-legal/thesis/figures/embedding-expansion-schema.svg"

QUERY_STYLE = dict(marker="*", s=620, c="#f5c542",
                   edgecolors="black", linewidths=1.0, zorder=4)
C1_SIZE = 240
C2_SIZE = 130
C1_COLOR = "black"
C2_COLOR = "#888888"

LABEL_STYLE = dict(fontsize=12, color="#1a1a1a", ha="left", va="center")


def _frame(ax, lim=3.4):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def graph_panel():
    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    # Query
    qx, qy = -2.0, 2.3
    ax.scatter([qx], [qy], **QUERY_STYLE)
    ax.text(qx + 0.25, qy, "Query", **LABEL_STYLE)

    # Root candidate
    sx, sy = 0.0, 0.0
    ax.scatter([sx], [sy], s=420, c=C1_COLOR, zorder=3)
    ax.text(sx + 0.25, sy, "Kandidat", **LABEL_STYLE)

    cosine_link = FancyArrowPatch(
        (qx, qy), (sx * 0.92 + qx * 0.08, sy * 0.92 + qy * 0.08),
        arrowstyle="-", linestyle=(0, (4, 3)), color="black", lw=1.2,
    )
    ax.add_patch(cosine_link)

    one_hop = [(1.5, 1.1), (1.8, -0.5), (-1.4, 1.1),
               (-1.65, -0.95), (0.25, 1.85)]
    for nx, ny in one_hop:
        ax.scatter([nx], [ny], s=C1_SIZE, c=C1_COLOR, zorder=3)
        arrow = FancyArrowPatch(
            (sx + nx * 0.12, sy + ny * 0.12),
            (sx + nx * 0.85, sy + ny * 0.85),
            arrowstyle="->", mutation_scale=18, color="black", lw=1.6,
        )
        ax.add_patch(arrow)
    ax.text(one_hop[0][0] + 0.2, one_hop[0][1] + 0.05, "1-Hop", **LABEL_STYLE)

    two_hop_groups = [
        ((1.5, 1.1), [(2.55, 1.55), (2.45, 0.55)]),
        ((-1.65, -0.95), [(-2.55, -1.7), (-2.5, -0.4)]),
    ]
    for (px, py), targets in two_hop_groups:
        for tx, ty in targets:
            ax.scatter([tx], [ty], s=C2_SIZE, c=C2_COLOR, zorder=3)
            dx, dy = tx - px, ty - py
            arrow = FancyArrowPatch(
                (px + dx * 0.18, py + dy * 0.18),
                (px + dx * 0.82, py + dy * 0.82),
                arrowstyle="->", mutation_scale=14, color=C2_COLOR, lw=1.2,
            )
            ax.add_patch(arrow)
    last_2hop = two_hop_groups[0][1][0]
    ax.text(last_2hop[0] + 0.18, last_2hop[1], "2-Hop", **LABEL_STYLE)

    _frame(ax)
    plt.tight_layout()
    plt.savefig(OUT_GRAPH, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {OUT_GRAPH}")


def embedding_panel():
    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    # Query at centre
    qx, qy = 0.0, 0.0

    # Big dashed proximity circle around query (1-hop kNN-Nachbarschaft)
    big_radius = 1.7
    outer = Circle((qx, qy), big_radius, fill=False, linestyle=(0, (5, 4)),
                   edgecolor="black", lw=1.2, zorder=1)
    ax.add_patch(outer)

    ax.scatter([qx], [qy], **QUERY_STYLE)
    ax.text(qx + 0.25, qy, "Query", **LABEL_STYLE)

    # 1-hop candidates inside the big circle
    one_hop = [(1.15, 0.75), (1.35, -0.6), (-0.85, 1.15),
               (-1.25, -0.7), (0.2, 1.4)]
    for nx, ny in one_hop:
        ax.scatter([nx], [ny], s=C1_SIZE, c=C1_COLOR, zorder=3)
    ax.text(one_hop[0][0] + 0.2, one_hop[0][1], "Kandidat", **LABEL_STYLE)
    # Region label "1-Hop" placed on the outer-circle boundary so it labels
    # the whole 1-Hop-Nachbarschaft, not a single dot.
    ax.text(0, big_radius + 0.18, "1-Hop-Nachbarschaft",
            fontsize=13, color="#1a1a1a", ha="center", va="bottom",
            weight="semibold")

    # 2-hop kNN: small circle around each of two 1-hop candidates,
    # with 2-hop candidates inside the small circle.
    small_radius = 0.85
    expansion_centers = [one_hop[0], one_hop[3]]
    two_hop_offsets = [
        [(0.55, 0.45), (0.65, -0.25), (-0.4, 0.55)],
        [(-0.6, -0.4), (-0.55, 0.4), (0.5, -0.45)],
    ]
    for (cx, cy), offsets in zip(expansion_centers, two_hop_offsets):
        small = Circle((cx, cy), small_radius, fill=False,
                       linestyle=(0, (3, 3)), edgecolor=C2_COLOR,
                       lw=1.0, zorder=1)
        ax.add_patch(small)
        for ox, oy in offsets:
            ax.scatter([cx + ox], [cy + oy], s=C2_SIZE, c=C2_COLOR, zorder=3)
    last_2hop = (expansion_centers[0][0] + two_hop_offsets[0][0][0],
                 expansion_centers[0][1] + two_hop_offsets[0][0][1])
    ax.text(last_2hop[0] + 0.15, last_2hop[1], "2-Hop", **LABEL_STYLE)

    _frame(ax)
    plt.tight_layout()
    plt.savefig(OUT_EMB, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {OUT_EMB}")


if __name__ == "__main__":
    graph_panel()
    embedding_panel()
