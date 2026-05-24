"""Publication-grade figure illustrating the Graph-Nearness metric.

Renders a small citation-graph snippet with one Ground-Truth node and two
retrieved nodes (one 1-hop neighbour, one disconnected). Each retrieval gets
its GN score and graph distance annotated. A small formula block at the top
documents the GN = 1/(1+d) rule with worked values.

Style matches gen_pipeline_overview.py (same palette, fonts, card helpers).

Usage:
    python gen_graph_nearness_example.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath
import numpy as np

# ----------------------------------------------------------------------- style

PLEX_SANS = "IBM Plex Sans"
PLEX_SERIF = "IBM Plex Serif"

plt.rcParams.update({
    "font.family": PLEX_SANS,
    "font.size": 8.5,
    "axes.linewidth": 0.4,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- palette (in lockstep with gen_pipeline_overview.py) --------------------
BG          = "#fafaf7"
INK         = "#1c1d20"
INK_SOFT    = "#5a5d65"
INK_FAINT   = "#9a9ba0"
HAIRLINE    = "#d8d4cb"
CARD_FACE   = "#ffffff"

# accents
GT_C        = "#22466e"   # GraphRAG-2Hop dark blue, used as Ground-Truth accent
GRAPH1_C    = "#4f7aa5"   # GraphRAG-1Hop blue, used for connected retrieval
RAG_C       = "#9a8e76"   # neutral taupe, used for disconnected retrieval

# muted "no path" cue
DIM_INK     = "#bcb9af"
DIM_STROKE  = "#e2ddd2"


# ------------------------------------------------------------------- helpers

def card(ax, x, y, w, h, *, face=CARD_FACE, edge=HAIRLINE, lw=0.6,
         radius=0.012, shadow=True, shadow_alpha=0.06, zorder=3):
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.0012, y - 0.0028), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0, facecolor="#000000", alpha=shadow_alpha,
            zorder=zorder - 1,
        )
        ax.add_patch(sh)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def stripe(ax, x, y, w, h, color, alpha=1.0, zorder=4, radius=0.002):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=color, alpha=alpha, zorder=zorder,
    ))


def txt(ax, x, y, s, *, size=8.5, color=INK, weight="regular",
        ha="center", va="center", family=PLEX_SANS, zorder=6,
        italic=False, alpha=1.0):
    style = "italic" if italic else "normal"
    return ax.text(x, y, s, fontsize=size, color=color, family=family,
                   weight=weight, ha=ha, va=va, zorder=zorder, alpha=alpha,
                   style=style)


def arrow_line(ax, p0, p1, *, color=INK_SOFT, lw=1.0, alpha=1.0, zorder=5,
               head_size=0.014, dashed=False):
    """Draw a straight directed arrow from p0 to p1 with a small arrowhead.

    The line stops short of p1 by `head_size * 0.6` so the head sits cleanly
    against the destination boundary.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    L = np.hypot(dx, dy)
    if L == 0:
        return
    ux, uy = dx / L, dy / L

    # shortened endpoint for the shaft
    shaft_end = (p1[0] - ux * head_size * 0.55, p1[1] - uy * head_size * 0.55)

    linestyle = (0, (3.5, 2.2)) if dashed else "-"
    ax.plot([p0[0], shaft_end[0]], [p0[1], shaft_end[1]],
            color=color, lw=lw, alpha=alpha, zorder=zorder,
            linestyle=linestyle, solid_capstyle="round")

    # arrowhead — small V at p1
    ang = np.arctan2(dy, dx)
    a1 = ang + np.pi - 0.42
    a2 = ang + np.pi + 0.42
    s = head_size
    xs = [p1[0] + s * np.cos(a1), p1[0], p1[0] + s * np.cos(a2)]
    ys = [p1[1] + s * np.sin(a1), p1[1], p1[1] + s * np.sin(a2)]
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder + 1,
            solid_capstyle="round", solid_joinstyle="round")


def node(ax, cx, cy, w, h, label, sub, *, face, edge, accent,
         badge_text=None, badge_color=None, lw=1.0, label_size=9.6,
         sub_size=7.4, badge_size=6.4):
    """A rounded-rect citation-graph node with title, BGE label, optional
    badge ("Ground Truth" / "retrievt").

    Layout is structured around the badge: badge sits centered horizontally
    at the very top of the card (its center on the top edge), then label
    and sub-label stack below the badge with enough room to breathe.
    """
    x = cx - w / 2
    y = cy - h / 2
    card(ax, x, y, w, h, face=face, edge=edge, lw=lw, radius=0.020,
         shadow=True, shadow_alpha=0.07)
    # left accent stripe
    stripe(ax, x + 0.008, y + 0.012, 0.0070, h - 0.024,
           color=accent, alpha=1.0, radius=0.0025)
    # main label — pushed slightly below center so badge has room
    txt(ax, cx + 0.005, cy + 0.002, label,
        size=label_size, family=PLEX_SERIF, weight="medium", color=INK)
    # sub label
    txt(ax, cx + 0.005, cy - 0.028, sub,
        size=sub_size, color=INK_SOFT)
    # badge pill straddling the top edge of the card
    if badge_text is not None:
        badge_w = 0.105
        badge_h = 0.026
        bx_center = cx + 0.005
        by_center = y + h - 0.002  # center sits right on the top edge
        ax.add_patch(FancyBboxPatch(
            (bx_center - badge_w / 2, by_center - badge_h / 2),
            badge_w, badge_h,
            boxstyle="round,pad=0,rounding_size=0.010",
            linewidth=0.7, edgecolor=badge_color, facecolor="#ffffff",
            zorder=7,
        ))
        txt(ax, bx_center, by_center,
            badge_text,
            size=badge_size, color=badge_color, weight="semibold",
            zorder=8)


# ------------------------------------------------------------------- drawing

def render(out_path: Path) -> None:
    fig_w, fig_h = 7.5, 3.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)

    # GT card on the left, three retrieved cards stacked on the right.
    # 1-hop: solid blue arrow directly to GT (d=1, GN=0.5).
    # 2-hop: solid blue arrows via a small intermediate node (d=2, GN=0.33).
    # No path: dashed grey line (d=∞, GN=0). Tiny GT / RETRIEVT badges
    # on the cards make the role of each node explicit.

    # ------------------------------------------------------------- geometry
    gt_cx, gt_cy = 0.180, 0.50
    gt_w, gt_h = 0.210, 0.270

    r_w, r_h = 0.200, 0.155
    r1_cx, r1_cy = 0.640, 0.825   # 1-hop
    r2_cx, r2_cy = 0.640, 0.500   # 2-hop
    r3_cx, r3_cy = 0.640, 0.175   # no path

    # intermediate stepping-stone node for the 2-hop chain
    int_cx, int_cy = 0.435, 0.500
    int_w, int_h = 0.140, 0.090

    # ---------- edge endpoints on card rectangles
    def edge_endpoints(src_cx, src_cy, src_w, src_h,
                       dst_cx, dst_cy, dst_w, dst_h):
        def trim(cx, cy, w, h, ox, oy):
            dx = ox - cx; dy = oy - cy
            if dx == 0 and dy == 0:
                return cx, cy
            tx = (w / 2) / abs(dx) if dx != 0 else np.inf
            ty = (h / 2) / abs(dy) if dy != 0 else np.inf
            t = min(tx, ty)
            return cx + t * dx, cy + t * dy
        return (trim(src_cx, src_cy, src_w, src_h, dst_cx, dst_cy),
                trim(dst_cx, dst_cy, dst_w, dst_h, src_cx, src_cy))

    def label_at_mid(p0, p1, label, color, off=0.030):
        dx = p1[0] - p0[0]; dy = p1[1] - p0[1]
        L = np.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        px, py = -uy, ux
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        txt(ax, mid[0] + px * off, mid[1] + py * off,
            label, size=9.0, color=color, weight="semibold")

    # ---------- 1-hop edge: r1 → gt (solid blue) -------------------------
    p_src, p_dst = edge_endpoints(r1_cx, r1_cy, r_w, r_h,
                                  gt_cx, gt_cy, gt_w, gt_h)
    arrow_line(ax, p_src, p_dst, color=GRAPH1_C, lw=1.8,
               head_size=0.022, zorder=4)
    label_at_mid(p_src, p_dst, "d = 1", GRAPH1_C, off=0.030)

    # ---------- 2-hop edges: r2 → intermediate → gt ----------------------
    # second leg first so first leg arrow lies on top
    p_int_to_gt_src, p_int_to_gt_dst = edge_endpoints(
        int_cx, int_cy, int_w, int_h, gt_cx, gt_cy, gt_w, gt_h)
    arrow_line(ax, p_int_to_gt_src, p_int_to_gt_dst,
               color=GRAPH1_C, lw=1.6, head_size=0.020, zorder=4)
    p_r2_to_int_src, p_r2_to_int_dst = edge_endpoints(
        r2_cx, r2_cy, r_w, r_h, int_cx, int_cy, int_w, int_h)
    arrow_line(ax, p_r2_to_int_src, p_r2_to_int_dst,
               color=GRAPH1_C, lw=1.6, head_size=0.020, zorder=4)
    # single combined "d = 2" label centered between the two segments
    chain_mid = ((p_r2_to_int_src[0] + p_int_to_gt_dst[0]) / 2,
                 (p_r2_to_int_src[1] + p_int_to_gt_dst[1]) / 2 - 0.060)
    txt(ax, chain_mid[0], chain_mid[1], "d = 2",
        size=9.0, color=GRAPH1_C, weight="semibold")

    # ---------- no-path indicator: r3 ··· gt (dashed grey) ---------------
    p3_src, p3_dst = edge_endpoints(r3_cx, r3_cy, r_w, r_h,
                                    gt_cx, gt_cy, gt_w, gt_h)
    ax.plot([p3_src[0], p3_dst[0]], [p3_src[1], p3_dst[1]],
            color=DIM_STROKE, lw=1.2, alpha=0.9, zorder=3,
            linestyle=(0, (2.4, 2.8)), solid_capstyle="round")
    label_at_mid(p3_src, p3_dst, "d = ∞", INK_FAINT, off=0.030)

    # ---------- cards drawn ON TOP of edges -------------------------------
    def draw_card(cx, cy, w, h, label, *, edge, accent, lw, label_size=11.0):
        x = cx - w / 2; y = cy - h / 2
        card(ax, x, y, w, h, face="#ffffff", edge=edge, lw=lw,
             radius=0.020, shadow=True, shadow_alpha=0.07)
        stripe(ax, x + 0.008, y + 0.010, 0.0070, h - 0.020,
               color=accent, alpha=1.0, radius=0.0025)
        txt(ax, cx + 0.005, cy, label,
            size=label_size, family=PLEX_SERIF, weight="medium", color=INK)

    def draw_badge(cx, card_top, label, color, *, w=0.085, h=0.022):
        # badge sits fully above the card edge, with its bottom resting on top
        by_center = card_top + h / 2 + 0.002
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, by_center - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=0.009",
            linewidth=0.7, edgecolor=color, facecolor="#ffffff",
            zorder=7,
        ))
        txt(ax, cx, by_center, label.upper(),
            size=6.2, color=color, weight="semibold", zorder=8)

    # GT card + badge
    draw_card(gt_cx, gt_cy, gt_w, gt_h, "BGE 139 III 391",
              edge=GT_C, accent=GT_C, lw=1.6, label_size=11.5)
    draw_badge(gt_cx + 0.005, gt_cy + gt_h / 2,
               "Ground Truth", GT_C, w=0.095)

    # Retrieved cards + badges
    draw_card(r1_cx, r1_cy, r_w, r_h, "BGE 139 III 500",
              edge=GRAPH1_C, accent=GRAPH1_C, lw=1.0)
    draw_badge(r1_cx + 0.005, r1_cy + r_h / 2, "Retrievt", GRAPH1_C, w=0.070)

    draw_card(r2_cx, r2_cy, r_w, r_h, "BGE 142 V 234",
              edge=GRAPH1_C, accent=GRAPH1_C, lw=1.0)
    draw_badge(r2_cx + 0.005, r2_cy + r_h / 2, "Retrievt", GRAPH1_C, w=0.070)

    draw_card(r3_cx, r3_cy, r_w, r_h, "BGE 140 II 123",
              edge=RAG_C, accent=RAG_C, lw=1.0)
    draw_badge(r3_cx + 0.005, r3_cy + r_h / 2, "Retrievt", RAG_C, w=0.070)

    # Intermediate stepping-stone node (no badge, lighter style)
    draw_card(int_cx, int_cy, int_w, int_h, "BGE 138 III 720",
              edge=INK_FAINT, accent=INK_FAINT, lw=0.7, label_size=8.6)

    # ---------- GN labels next to retrieved cards -------------------------
    gn_x = r1_cx + r_w / 2 + 0.030
    txt(ax, gn_x, r1_cy, "GN = 0.5",
        size=12.5, family=PLEX_SERIF, weight="semibold",
        color=GRAPH1_C, ha="left")
    txt(ax, gn_x, r2_cy, "GN = 0.33",
        size=12.5, family=PLEX_SERIF, weight="semibold",
        color=GRAPH1_C, ha="left")
    txt(ax, gn_x, r3_cy, "GN = 0",
        size=12.5, family=PLEX_SERIF, weight="semibold",
        color=RAG_C, ha="left")

    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches=None,
                pad_inches=0)
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out = out_dir / "graph-nearness-example.png"
    render(out)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
