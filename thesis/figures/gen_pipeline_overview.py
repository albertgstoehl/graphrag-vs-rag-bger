"""Publication-grade pipeline-overview figure for the thesis.

Strict left-to-right column layout. Five stages, vertically stacked cards
inside each stage column. Cross-stage connectors are clean orthogonal
lines that never cross.

A single ``HIGHLIGHTS`` dict drives the dim/emphasise behaviour, so a new
variant is one flag away.

Usage:
    python gen_pipeline_overview.py                         # overview + recommended
    python gen_pipeline_overview.py --highlight none
    python gen_pipeline_overview.py --highlight graph1hop-indegree
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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

# ---- palette ----------------------------------------------------------------
BG          = "#fafaf7"
INK         = "#1c1d20"
INK_SOFT    = "#5a5d65"
INK_FAINT   = "#9a9ba0"
HAIRLINE    = "#d8d4cb"
CARD_FACE   = "#ffffff"

# system accents: cool blues for graph-family, warm ochres for embed-family,
# neutral taupe for RAG baseline.
RAG_C       = "#9a8e76"
EMB1_C      = "#d6a566"
EMB2_C      = "#a87530"
GRAPH1_C    = "#4f7aa5"
GRAPH2_C    = "#22466e"

# downstream stages share a single steel-blue conduit color
CONDUIT     = "#3d5a78"

# dim palette
DIM_FACE    = "#f4f2ec"
DIM_STROKE  = "#e2ddd2"
DIM_INK     = "#bcb9af"


# --------------------------------------------------------------------- config

@dataclass
class Highlight:
    system: str | None = None     # 'rag' | 'emb1' | 'emb2' | 'graph1' | 'graph2'
    ranking: str | None = None    # 'cosine' | 'crossenc' | 'indeg'
    k: int | None = None          # 5, 10, 20
    dim_downstream: bool = False  # legacy: dim ranking + topk + metrics columns
    dim_upstream: bool = False    # legacy: dim intake (Q+E+seeds) + expansion columns
    hide_query_embed: bool = False  # if True: skip rendering Query and Embedding cards
    hide_pool: bool = False       # if True: skip the per-system "Recall · Precision" pool pills
    dim_pool: bool = False        # if True: pool pills drawn in dim style even when their card is bright
    bright_pool: bool = False     # if True: dim ALL stages but keep the pool pills at full brightness
    focus: str | None = None      # 'intake' (Q+E) | 'seeds' (ANN-Seeds card only) | 'expansion' (4 expansion cards only) | 'systems' (seeds + 4 expansion) | 'ranking' | 'topk' | 'metrics' — emphasise this stage, dim all others


HIGHLIGHTS: dict[str, Highlight] = {
    "none": Highlight(),
    "systems-only": Highlight(dim_downstream=True, hide_query_embed=True),
    "evaluation-focus": Highlight(dim_upstream=True, hide_query_embed=True),
    "query-focus":   Highlight(focus="intake"),
    "seeds-focus":   Highlight(focus="seeds"),
    "expansion-focus": Highlight(focus="expansion"),
    "systems-focus": Highlight(focus="systems", dim_pool=True),
    "pool-focus":    Highlight(bright_pool=True),
    "ranking-focus": Highlight(focus="ranking"),
    "cosine-focus":       Highlight(focus="ranking", ranking="cosine"),
    "indegree-focus":     Highlight(focus="ranking", ranking="indeg"),
    "crossencoder-focus": Highlight(focus="ranking", ranking="crossenc"),
    "topk-focus":    Highlight(focus="topk"),
    "metrics-focus": Highlight(focus="metrics"),
    "graph1hop-indegree": Highlight(
        system="graph1", ranking="indeg", k=20,
    ),
    "graph1hop-indegree-k5": Highlight(
        system="graph1", ranking="indeg", k=5,
    ),
    "graph1hop-indegree-k20": Highlight(
        system="graph1", ranking="indeg", k=20,
    ),
    "embed1hop-crossencoder": Highlight(
        system="emb1", ranking="crossenc", k=10,
    ),
}


# ------------------------------------------------------------------- helpers

def card(ax, x, y, w, h, *, face=CARD_FACE, edge=HAIRLINE, lw=0.6,
         radius=0.010, shadow=True, shadow_alpha=0.05, zorder=3):
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.0010, y - 0.0024), w, h,
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
        italic=False, alpha=1.0, letter_spacing=0.0):
    style = "italic" if italic else "normal"
    return ax.text(x, y, s, fontsize=size, color=color, family=family,
                   weight=weight, ha=ha, va=va, zorder=zorder, alpha=alpha,
                   style=style)


def hline(ax, x0, x1, y, color=HAIRLINE, lw=0.5, zorder=2, alpha=1.0):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=zorder,
            solid_capstyle="round", alpha=alpha)


def vline(ax, x, y0, y1, color=HAIRLINE, lw=0.5, zorder=2, alpha=1.0):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=zorder,
            solid_capstyle="round", alpha=alpha)


def bezier(ax, p0, p1, *, color=INK_SOFT, lw=0.8, alpha=1.0, zorder=4,
           curvature=0.35, head=True, head_size=0.011):
    """Smooth horizontal-ish S-curve from p0 to p1."""
    dx = p1[0] - p0[0]
    mid_offset = abs(dx) * curvature
    c1 = (p0[0] + mid_offset, p0[1])
    c2 = (p1[0] - mid_offset, p1[1])
    verts = [p0, c1, c2, p1]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    path = MplPath(verts, codes)
    ax.add_patch(mp.PathPatch(path, facecolor="none", edgecolor=color,
                              linewidth=lw, alpha=alpha, zorder=zorder,
                              capstyle="round"))
    if head:
        # tangent at end ~ horizontal
        ang = 0.0
        a1 = ang + np.pi - 0.44
        a2 = ang + np.pi + 0.44
        s = head_size
        xs = [p1[0] + s * np.cos(a1), p1[0], p1[0] + s * np.cos(a2)]
        ys = [p1[1] + s * np.sin(a1), p1[1], p1[1] + s * np.sin(a2)]
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder + 1,
                solid_capstyle="round", solid_joinstyle="round")


def stage_badge(ax, x, y, label, *, color=INK_FAINT, size=6.8):
    """Tiny uppercase letter-spaced stage badge above each column."""
    spaced = "  ".join(list(label.upper()))
    txt(ax, x, y, spaced, size=size, color=color, weight="semibold",
        family=PLEX_SANS)


def pool_badge(ax, x_right, y_card_bot, *, accent, dim=False, h_card=None):
    """Compact "Pool → R-Ceil · P-Ceil" indicator pinned to the bottom-right
    inside edge of a system card.  Signals that this system produces its own
    candidate pool on which two ceiling metrics (Recall-Ceiling, Precision-
    Ceiling) are computed before the ranking stage.

    Anchored to the card's bottom-right corner; placed strictly INSIDE the
    card so it never collides with the inter-stage connector trunk.
    """
    bw_pill = 0.092
    bh_pill = 0.022
    pad_x = 0.005
    pad_y = 0.005
    x0 = x_right - bw_pill - pad_x
    y0 = y_card_bot + pad_y
    face = DIM_FACE if dim else "#fbf8f1"
    edge = DIM_STROKE if dim else accent
    # use a soft tinted background so the pill reads as belonging to the
    # accent-colored card without overpowering it
    ax.add_patch(FancyBboxPatch(
        (x0, y0), bw_pill, bh_pill,
        boxstyle="round,pad=0,rounding_size=0.007",
        linewidth=0.5, edgecolor=edge, facecolor=face,
        alpha=0.40 if dim else 1.0, zorder=5,
    ))
    txt_color = DIM_INK if dim else accent
    # Compact pool-quality label.  Just "Recall · Precision" — short
    # enough to fit cleanly inside the pill without overflowing.  The
    # "Ceiling" semantic comes from the Expansion-subtitle context
    # ("je 1 Pool") and the thesis prose.
    txt(ax, x0 + bw_pill / 2, y0 + bh_pill / 2,
        "Recall  ·  Precision",
        size=5.0, color=txt_color, weight="semibold",
        ha="center", va="center", zorder=6)


# ------------------------------------------------------------------- drawing

@dataclass
class Layout:
    fig_w: float = 7.5
    fig_h: float = 4.6


L = Layout()


def render(highlight: Highlight, out_path: Path) -> None:
    fig = plt.figure(figsize=(L.fig_w, L.fig_h))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)

    hide_qe = highlight.hide_query_embed
    focus = highlight.focus

    # Per-stage dim flags. The `focus` flag dims everything except the named
    # stage; legacy `dim_downstream` / `dim_upstream` flags add their effects
    # on top.  'systems' covers seeds card + 4 expansion variants.
    dim_qe        = (focus is not None and focus != "intake")                     or highlight.dim_upstream
    dim_seeds     = (focus is not None and focus not in ("systems", "seeds"))     or highlight.dim_upstream
    dim_expansion = (focus is not None and focus not in ("systems", "expansion")) or highlight.dim_upstream
    dim_ranking   = (focus is not None and focus != "ranking") or highlight.dim_downstream
    dim_topk      = (focus is not None and focus != "topk")    or highlight.dim_downstream
    dim_metrics   = (focus is not None and focus != "metrics") or highlight.dim_downstream

    if highlight.bright_pool:
        dim_qe = dim_seeds = dim_expansion = True
        dim_ranking = dim_topk = dim_metrics = True


    # =====================================================================
    # COLUMN ANCHORS  (left edges + widths, in figure-relative coordinates)
    # =====================================================================
    #  A: Intake (Query / Embed / Seeds)
    #  B: Expansion (5 system cards)
    #  C: Ranking (3 cards)
    #  D: k (3 cards)
    #  E: Metrics (compact list)
    #
    # Connectors live in the gutters between columns.
    cols = [
        ("intake",     0.030, 0.170),
        ("expansion",  0.275, 0.195),
        ("ranking",    0.560, 0.160),
        ("k",          0.760, 0.075),
        ("metrics",    0.875, 0.105),
    ]
    COL = {name: (x0, w) for name, x0, w in cols}

    BAND_Y0 = 0.090
    BAND_Y1 = 0.880

    # =====================================================================
    # Stage badges along the top of each column
    # =====================================================================
    badge_y = 0.940
    badge_labels = {
        "intake":    "Eingabe",
        "expansion": "Expansion",
        "ranking":   "Ranking",
        "k":         "Top-k",
        "metrics":   "Metriken",
    }
    for name, label in badge_labels.items():
        x0, w = COL[name]
        stage_badge(ax, x0 + w / 2, badge_y, label)

    # subtle title/subtitle line under badges
    sub_y = 0.910
    intake_sub = "Baseline · Seed-Quelle" if hide_qe else "Sachverhalt → Vektor"
    expansion_sub = (
        "4 Expansionsvarianten"
        if (highlight.hide_pool or highlight.dim_pool)
        else "4 Varianten · je 1 Pool"
    )
    for name, sub in [
        ("intake", intake_sub),
        ("expansion", expansion_sub),
        ("ranking", "3 Strategien"),
        ("k", "{5, 10, 20}"),
        ("metrics", "IR + Topologie"),
    ]:
        x0, w = COL[name]
        txt(ax, x0 + w / 2, sub_y, sub, size=6.8, color=INK_FAINT,
            italic=True)

    # =====================================================================
    # COLUMN A — Intake stack: Query → Embed → Seeds
    # =====================================================================
    ax0, aw = COL["intake"]
    a_cx = ax0 + aw / 2

    intake_items = [
        ("Query",      "DE · FR · IT",   PLEX_SERIF, 11.5, False),
        ("Embedding",  "BGE-M3 · 1024d", PLEX_SANS,  10.5, False),
        # baseline card: also serves as the RAG system in the comparison
        ("ANN-Seeds",  "60 · cosine kNN", PLEX_SANS, 10.5, True),
    ]
    if hide_qe:
        intake_items = [it for it in intake_items if it[4]]
    n_a = len(intake_items)
    gap_a = 0.040
    if hide_qe:
        # Keep card sized like one row of the 3-card layout, vertically
        # centered on the band so the seeds → expansion trunk stays balanced.
        h_a = (BAND_Y1 - BAND_Y0 - 2 * gap_a) / 3
        top_y_a = (BAND_Y0 + BAND_Y1) / 2 + h_a / 2
    else:
        h_a = (BAND_Y1 - BAND_Y0 - (n_a - 1) * gap_a) / n_a
        top_y_a = BAND_Y1
    intake_centers = []
    for i, (name, sub, fam, size, is_baseline) in enumerate(intake_items):
        y = top_y_a - (i + 1) * h_a - i * gap_a
        if is_baseline:
            # baseline card: ANN-Seeds card *is* the RAG system in the
            # comparison.  Styled as a SYSTEM card (left stripe in RAG_C)
            # with a dual-identity title to make the equivalence explicit.
            seeds_face = DIM_FACE if dim_seeds else CARD_FACE
            seeds_edge = DIM_STROKE if dim_seeds else RAG_C
            seeds_stripe_color = DIM_STROKE if dim_seeds else RAG_C
            seeds_stripe_alpha = 0.30 if dim_seeds else 1.0
            card(ax, ax0, y, aw, h_a, radius=0.014,
                 face=seeds_face,
                 edge=seeds_edge, lw=0.9 if not dim_seeds else 0.5,
                 shadow=not dim_seeds, shadow_alpha=0.05)
            stripe(ax, ax0 + 0.006, y + 0.012, 0.0055, h_a - 0.024,
                   color=seeds_stripe_color, alpha=seeds_stripe_alpha)
            cx_card = ax0 + (aw + 0.014) / 2  # nudge right of accent stripe
            seeds_text_color = DIM_INK if dim_seeds else INK
            seeds_sub_color = "#a0a0a0" if dim_seeds else INK_SOFT
            seeds_accent_color = "#a0a0a0" if dim_seeds else RAG_C
            txt(ax, cx_card, y + h_a / 2 + 0.022,
                "Seeds  =  ANN",
                size=10.5, family=PLEX_SANS, weight="semibold",
                color=seeds_text_color, ha="center")
            txt(ax, cx_card, y + h_a / 2 - 0.005,
                "60 · Cosine-kNN",
                size=7.6, color=seeds_sub_color, ha="center")
            txt(ax, cx_card, y + h_a / 2 - 0.028,
                "BASIS  SYSTEM",
                size=6.4, weight="semibold", color=seeds_accent_color, ha="center")
            # Per-system pool indicator. The seeds card *is* the RAG pool,
            # so it gets the same R-Ceil / P-Ceil badge as the four
            # expansion variants.  Pinned inside the bottom-right corner.
            if not highlight.hide_pool:
                if highlight.bright_pool:
                    pill_dim_seeds = False
                elif highlight.dim_pool:
                    pill_dim_seeds = True
                else:
                    pill_dim_seeds = dim_seeds
                pool_badge(ax, ax0 + aw, y,
                           accent=RAG_C, dim=pill_dim_seeds, h_card=h_a)
        else:
            qe_face  = DIM_FACE   if dim_qe else CARD_FACE
            qe_edge  = DIM_STROKE if dim_qe else HAIRLINE
            qe_text  = DIM_INK    if dim_qe else INK
            qe_sub   = "#a0a0a0"  if dim_qe else INK_SOFT
            card(ax, ax0, y, aw, h_a, radius=0.014,
                 face=qe_face, edge=qe_edge,
                 lw=0.5 if dim_qe else 0.6,
                 shadow=not dim_qe, shadow_alpha=0.045)
            txt(ax, a_cx, y + h_a / 2 + 0.018, name,
                size=size, family=fam, weight="medium", color=qe_text)
            txt(ax, a_cx, y + h_a / 2 - 0.020, sub,
                size=7.5, color=qe_sub)
        intake_centers.append((a_cx, y + h_a / 2, y, y + h_a))

    # arrows between A cards — dim if either endpoint is dimmed
    for i in range(n_a - 1):
        src_is_seeds = intake_items[i][4]
        dst_is_seeds = intake_items[i + 1][4]
        src_dim = dim_seeds if src_is_seeds else dim_qe
        dst_dim = dim_seeds if dst_is_seeds else dim_qe
        arrow_dim = src_dim or dst_dim
        a_col = DIM_STROKE if arrow_dim else INK_FAINT
        a_lw  = 0.5 if arrow_dim else 0.9
        a_alpha = 0.35 if arrow_dim else 1.0
        top_y = intake_centers[i][2]
        bot_y = intake_centers[i + 1][3]
        ax.plot([a_cx, a_cx], [top_y - 0.004, bot_y + 0.004],
                color=a_col, lw=a_lw, alpha=a_alpha, zorder=5,
                solid_capstyle="round")
        # tiny arrowhead at bottom
        s = 0.008
        ax.plot([a_cx - s * 0.6, a_cx, a_cx + s * 0.6],
                [bot_y + 0.004 + s, bot_y + 0.004, bot_y + 0.004 + s],
                color=a_col, lw=a_lw, alpha=a_alpha, zorder=6,
                solid_capstyle="round", solid_joinstyle="round")

    seeds_right_x = ax0 + aw
    seeds_center_y = intake_centers[-1][1]

    # =====================================================================
    # COLUMN B — Expansion stack (5 system cards)
    # =====================================================================
    bx0, bw = COL["expansion"]
    b_cx = bx0 + bw / 2

    # 4 expansion variants only. RAG is the ANN-Seeds card itself (intake col).
    systems = [
        ("emb1",   "Embedding 1-Hop",  "≤ 400  Kandidaten",  EMB1_C),
        ("emb2",   "Embedding 2-Hop",  "≤ 800  Kandidaten",  EMB2_C),
        ("graph1", "GraphRAG 1-Hop",   "≤ 400  Kandidaten",  GRAPH1_C),
        ("graph2", "GraphRAG 2-Hop",   "≤ 800  Kandidaten",  GRAPH2_C),
    ]
    n_b = len(systems)
    gap_inner = 0.030
    h_b = (BAND_Y1 - BAND_Y0 - (n_b - 1) * gap_inner) / n_b
    sys_y_centers = {}
    for i, (key, name, cap, accent) in enumerate(systems):
        y = BAND_Y1 - (i + 1) * h_b - i * gap_inner
        is_hi = (highlight.system is None) or (highlight.system == key)
        if dim_expansion:
            is_hi = False
        face = CARD_FACE if is_hi else DIM_FACE
        edge = accent if is_hi else DIM_STROKE
        text_c = INK if is_hi else DIM_INK
        cap_c = accent if is_hi else "#a0a0a0"
        card(ax, bx0, y, bw, h_b, face=face, edge=edge,
             lw=0.9 if is_hi else 0.5,
             radius=0.010, shadow=is_hi, shadow_alpha=0.05)
        # left accent stripe
        stripe(ax, bx0 + 0.005, y + 0.008, 0.0050, h_b - 0.016,
               color=accent, alpha=1.0 if is_hi else 0.30)
        text_left = bx0 + 0.020
        txt(ax, text_left, y + h_b / 2 + 0.012, name,
            size=9.0, weight="semibold" if is_hi else "medium",
            color=text_c, ha="left")
        txt(ax, text_left, y + h_b / 2 - 0.014, cap,
            size=7.4, weight="regular",
            color=cap_c, ha="left")
        # Per-system pool indicator pinned inside the bottom-right corner.
        # One small "POOL · R-Ceil · P-Ceil" pill per expansion variant
        # to make the per-system pool concept visually explicit.
        if not highlight.hide_pool:
            if highlight.bright_pool:
                pill_dim_exp = False
            elif highlight.dim_pool:
                pill_dim_exp = True
            else:
                pill_dim_exp = (not is_hi)
            pool_badge(ax, bx0 + bw, y,
                       accent=accent, dim=pill_dim_exp, h_card=h_b)
        sys_y_centers[key] = (bx0, bx0 + bw, y, y + h_b, y + h_b / 2)

    # ---- Connectors A → B  (orthogonal trunk-and-branch routing) ----
    # short horizontal stub from seeds → trunk_x; vertical trunk spanning the
    # 5 system rows; horizontal branches into each system card.
    trunk_ab_x = (seeds_right_x + bx0) / 2
    sys_ys = [sys_y_centers[k][4] for k, *_ in systems]
    trunk_top = max(sys_ys)
    trunk_bot = min(sys_ys)
    ab_dim = dim_seeds or dim_expansion
    trunk_col = DIM_STROKE if ab_dim else INK_FAINT
    trunk_alpha = 0.30 if ab_dim else 0.55
    # background neutral trunk (always visible for context)
    vline(ax, trunk_ab_x, trunk_bot, trunk_top,
          color=trunk_col, lw=0.6, alpha=trunk_alpha, zorder=4)
    # stub seeds → trunk
    stub_col = DIM_STROKE if ab_dim else INK_FAINT
    stub_alpha = 0.35 if ab_dim else 0.85
    hline(ax, seeds_right_x + 0.003, trunk_ab_x, seeds_center_y,
          color=stub_col, lw=0.9 if not ab_dim else 0.5,
          alpha=stub_alpha, zorder=5)
    # arrowhead from stub at trunk junction would be redundant — skip.
    for i, (key, *_rest) in enumerate(systems):
        is_hi = (highlight.system is None) or (highlight.system == key)
        if dim_expansion:
            is_hi = False
        accent = systems[i][3]
        if ab_dim:
            col = DIM_STROKE; lw = 0.5; alpha = 0.30
        elif highlight.system is None:
            col = INK_FAINT
            lw = 0.7
            alpha = 0.55
        elif is_hi:
            col = accent
            lw = 1.6
            alpha = 1.0
        else:
            col = DIM_STROKE
            lw = 0.5
            alpha = 0.35
        y_target = sys_y_centers[key][4]
        # Branch trunk → card
        hline(ax, trunk_ab_x, bx0 - 0.003, y_target,
              color=col, lw=lw, alpha=alpha,
              zorder=5 if not is_hi else 7)
        # arrowhead at card edge
        s = 0.010
        ax.plot([bx0 - 0.003 - s * 0.7, bx0 - 0.003, bx0 - 0.003 - s * 0.7],
                [y_target + s * 0.40, y_target, y_target - s * 0.40],
                color=col, lw=lw, alpha=alpha,
                zorder=6 if not is_hi else 8,
                solid_capstyle="round", solid_joinstyle="round")
    # if a system is highlighted, repaint that trunk segment + its branch in colour
    if highlight.system is not None and not dim_expansion:
        hi_y = sys_y_centers[highlight.system][4]
        sys_color = {
            "rag": RAG_C, "emb1": EMB1_C, "emb2": EMB2_C,
            "graph1": GRAPH1_C, "graph2": GRAPH2_C,
        }[highlight.system]
        # bright trunk only between seeds_y and hi_y (whichever is below the other)
        y_a = min(seeds_center_y, hi_y)
        y_b = max(seeds_center_y, hi_y)
        vline(ax, trunk_ab_x, y_a, y_b,
              color=sys_color, lw=1.5, alpha=1.0, zorder=7)
        hline(ax, seeds_right_x + 0.003, trunk_ab_x, seeds_center_y,
              color=sys_color, lw=1.5, alpha=1.0, zorder=7)

    # =====================================================================
    # COLUMN C — Ranking (3 cards)
    # =====================================================================
    cx0, cw = COL["ranking"]

    rankings = [
        ("cosine",   "Cosine",         "Vektor-Ähnlichkeit"),
        ("indeg",    "In-Degree",      "Zitations-Autorität"),
        ("crossenc", "Cross-Encoder",  "bge-reranker-v2-m3"),
    ]
    n_c = len(rankings)
    gap_c = 0.045
    h_c = (BAND_Y1 - BAND_Y0 - (n_c - 1) * gap_c) / n_c
    rank_y_centers = {}
    for i, (key, name, sub) in enumerate(rankings):
        y = BAND_Y1 - (i + 1) * h_c - i * gap_c
        is_hi = (highlight.ranking is None) or (highlight.ranking == key)
        if dim_ranking:
            is_hi = False
        face = CARD_FACE if is_hi else DIM_FACE
        edge = CONDUIT if is_hi else DIM_STROKE
        text_c = INK if is_hi else DIM_INK
        card(ax, cx0, y, cw, h_c, face=face, edge=edge,
             lw=0.9 if is_hi else 0.5,
             radius=0.010, shadow=is_hi, shadow_alpha=0.05)
        stripe(ax, cx0 + 0.005, y + 0.008, 0.0050, h_c - 0.016,
               color=CONDUIT, alpha=1.0 if is_hi else 0.30)
        txt(ax, cx0 + 0.018, y + h_c / 2 + 0.012, name,
            size=9.0, weight="semibold" if is_hi else "medium",
            color=text_c, ha="left")
        txt(ax, cx0 + 0.018, y + h_c / 2 - 0.014, sub,
            size=7.4, color=INK_SOFT if is_hi else DIM_INK,
            ha="left", italic=True)
        rank_y_centers[key] = (cx0, cx0 + cw, y + h_c / 2)

    # ---- Connectors B → C (fan-in from 5 sys cards to ranking column) ----
    # Use a single horizontal bus aligned with vertical midpoint of B column,
    # then split into 3 ranking entries from there.
    bus_x = bx0 + bw + 0.040
    bus_y_mid = (BAND_Y0 + BAND_Y1) / 2
    # 5 lines from sys cards to bus, then 3 lines from bus to rankings.

    # Decide overall conduit emphasis
    sys_hi_key = highlight.system
    if sys_hi_key is not None:
        bus_accent = {
            "rag": RAG_C, "emb1": EMB1_C, "emb2": EMB2_C,
            "graph1": GRAPH1_C, "graph2": GRAPH2_C,
        }[sys_hi_key]
    else:
        bus_accent = CONDUIT

    # Lines sys → trunk_bc → bus.  Same orthogonal pattern as A→B.
    trunk_bc_x = (bx0 + bw + bus_x) / 2
    trunk_top_bc = trunk_top
    trunk_bot_bc = trunk_bot
    bc_dim = dim_expansion or dim_ranking
    trunk_bc_col = DIM_STROKE if bc_dim else INK_FAINT
    trunk_bc_alpha = 0.30 if bc_dim else 0.55
    # background neutral trunk
    vline(ax, trunk_bc_x, trunk_bot_bc, trunk_top_bc,
          color=trunk_bc_col, lw=0.6, alpha=trunk_bc_alpha, zorder=4)
    for key, _name, _cap, accent in systems:
        is_hi = (sys_hi_key is None) or (sys_hi_key == key)
        if dim_expansion:
            is_hi = False
        x_left, x_right, y_bot, y_top, y_mid = sys_y_centers[key]
        if bc_dim:
            col = DIM_STROKE; lw = 0.5; alpha = 0.30
        elif sys_hi_key is None:
            col = INK_FAINT; lw = 0.7; alpha = 0.55
        elif is_hi:
            col = accent;     lw = 1.6; alpha = 1.0
        else:
            col = DIM_STROKE; lw = 0.5; alpha = 0.35
        hline(ax, x_right + 0.003, trunk_bc_x, y_mid,
              color=col, lw=lw, alpha=alpha,
              zorder=5 if not is_hi else 7)

    # stub trunk → bus dot
    stub_bc_col = DIM_STROKE if bc_dim else INK_FAINT
    stub_bc_alpha = 0.35 if bc_dim else 0.85
    hline(ax, trunk_bc_x, bus_x, bus_y_mid,
          color=stub_bc_col, lw=0.9 if not bc_dim else 0.5,
          alpha=stub_bc_alpha, zorder=5)
    if sys_hi_key is not None and not bc_dim:
        # highlight trunk segment between sys row and bus row
        hi_y = sys_y_centers[sys_hi_key][4]
        y_a = min(hi_y, bus_y_mid)
        y_b = max(hi_y, bus_y_mid)
        vline(ax, trunk_bc_x, y_a, y_b,
              color=bus_accent, lw=1.5, alpha=1.0, zorder=7)
        hline(ax, trunk_bc_x, bus_x, bus_y_mid,
              color=bus_accent, lw=1.5, alpha=1.0, zorder=7)

    # Convergence dot (on top of trunks). Marks the visual fan-in point
    # from the 5 candidate pools (Seeds + 4 expansion variants) into the
    # 3 ranking strategies.  The pool-quality metrics themselves
    # (Recall-Ceiling, Precision-Ceiling) are annotated PER SYSTEM via
    # the small "R · P" badges drawn on each system card (see below) —
    # one pool per system, not one shared pool.
    dot_col = DIM_STROKE if bc_dim else bus_accent
    ax.add_patch(mp.Circle((bus_x, bus_y_mid), 0.007,
                           facecolor=dot_col, edgecolor="none",
                           zorder=9))

    # Lines bus → ranking entries
    for key, name, _sub in rankings:
        is_hi = (highlight.ranking is None) or (highlight.ranking == key)
        if dim_ranking:
            is_hi = False
        x_left, x_right, y_mid = rank_y_centers[key]
        if bc_dim:
            col = DIM_STROKE; lw = 0.5; alpha = 0.30
        elif highlight.ranking is None and sys_hi_key is None:
            col = INK_FAINT; lw = 0.7; alpha = 0.55
        elif is_hi:
            col = bus_accent; lw = 1.5; alpha = 1.0
        else:
            col = DIM_STROKE; lw = 0.5; alpha = 0.4
        bezier(ax, (bus_x, bus_y_mid), (x_left - 0.003, y_mid),
               color=col, lw=lw, alpha=alpha,
               curvature=0.50, head=True, head_size=0.010,
               zorder=5 if not is_hi else 7)

    # =====================================================================
    # COLUMN D — k (3 cards, big numbers)
    # =====================================================================
    dx0, dw = COL["k"]
    ks = [5, 10, 20]
    n_d = len(ks)
    gap_d = gap_c
    h_d = h_c
    k_y_centers = {}
    for i, kv in enumerate(ks):
        y = BAND_Y1 - (i + 1) * h_d - i * gap_d
        is_hi = (highlight.k is None) or (highlight.k == kv)
        if dim_topk:
            is_hi = False
        face = CARD_FACE if is_hi else DIM_FACE
        edge = CONDUIT if is_hi else DIM_STROKE
        text_c = INK if is_hi else DIM_INK
        card(ax, dx0, y, dw, h_d, face=face, edge=edge,
             lw=0.9 if is_hi else 0.5,
             radius=0.010, shadow=is_hi, shadow_alpha=0.05)
        txt(ax, dx0 + dw / 2, y + h_d / 2 + 0.002, f"{kv}",
            size=20 if is_hi else 16,
            weight="semibold" if is_hi else "medium",
            color=text_c, family=PLEX_SERIF)
        k_y_centers[kv] = (dx0, dx0 + dw, y + h_d / 2)

    # ---- Connectors C → D (3 rank → 3 k, parallel orthogonal lines) ----
    cd_dim = dim_ranking or dim_topk
    for r_key, r_name, _sub in rankings:
        rx_left, rx_right, ry = rank_y_centers[r_key]
        is_r_hi = (highlight.ranking is None) or (highlight.ranking == r_key)
        if highlight.ranking is None and highlight.system is None:
            col = INK_FAINT; lw = 0.7; alpha = 0.55
        elif is_r_hi:
            col = bus_accent; lw = 1.5; alpha = 1.0
        else:
            col = DIM_STROKE; lw = 0.5; alpha = 0.35
        for kv in ks:
            is_k_hi = (highlight.k is None) or (highlight.k == kv)
            kx_left, kx_right, ky = k_y_centers[kv]
            edge_hi = is_r_hi and is_k_hi
            if cd_dim:
                col_e = DIM_STROKE; lw_e = 0.45; alpha_e = 0.20
            elif highlight.ranking is None and highlight.k is None and highlight.system is None:
                col_e = INK_FAINT; lw_e = 0.55; alpha_e = 0.30
            elif edge_hi:
                col_e = bus_accent; lw_e = 1.5; alpha_e = 1.0
            else:
                col_e = DIM_STROKE; lw_e = 0.45; alpha_e = 0.25
            bezier(ax, (rx_right + 0.003, ry), (kx_left - 0.003, ky),
                   color=col_e, lw=lw_e, alpha=alpha_e,
                   curvature=0.55, head=False,
                   zorder=5 if not edge_hi else 7)

    # =====================================================================
    # COLUMN E — Metrics list (compact)
    # =====================================================================
    ex0, ew = COL["metrics"]
    metrics = [
        ("Precision@k", "P"),
        ("Recall@k",    "R"),
        ("MRR",         "M"),
        ("nDCG@k",      "N"),
        ("HitRate",     "H"),
        ("Graph-Near.", "G"),
    ]
    n_e = len(metrics)
    gap_e = 0.014
    h_e = (BAND_Y1 - BAND_Y0 - (n_e - 1) * gap_e) / n_e
    metrics_left_anchors = []
    metric_y_mids = []
    metric_face = DIM_FACE if dim_metrics else CARD_FACE
    metric_edge = DIM_STROKE if dim_metrics else HAIRLINE
    metric_ink = DIM_INK if dim_metrics else INK
    for i, (name, _glyph) in enumerate(metrics):
        y = BAND_Y1 - (i + 1) * h_e - i * gap_e
        card(ax, ex0, y, ew, h_e, face=metric_face, edge=metric_edge,
             lw=0.5, radius=0.008, shadow=not dim_metrics, shadow_alpha=0.04)
        txt(ax, ex0 + ew / 2, y + h_e / 2, name,
            size=8.4, color=metric_ink, weight="regular", ha="center")
        metrics_left_anchors.append((ex0, y + h_e / 2))
        metric_y_mids.append(y + h_e / 2)

    # ---- Connectors D → E (single fan from each k card to metrics list) ----
    de_dim = dim_topk or dim_metrics
    for kv in ks:
        is_k_hi = (highlight.k is None) or (highlight.k == kv)
        kx_left, kx_right, ky = k_y_centers[kv]
        if de_dim:
            col = DIM_STROKE; lw = 0.5; alpha = 0.25
        elif highlight.k is None and highlight.system is None:
            col = INK_FAINT; lw = 0.7; alpha = 0.50
        elif is_k_hi:
            col = bus_accent; lw = 1.5; alpha = 1.0
        else:
            col = DIM_STROKE; lw = 0.5; alpha = 0.35
        merge_x = (kx_right + ex0) / 2
        merge_y = (BAND_Y0 + BAND_Y1) / 2
        bezier(ax, (kx_right + 0.003, ky), (merge_x, merge_y),
               color=col, lw=lw, alpha=alpha,
               curvature=0.55, head=False,
               zorder=5 if not is_k_hi else 7)
    # vertical spine on right side from merge_x going through all metrics
    # plus tiny horizontal tick into each metric
    merge_x = (k_y_centers[5][1] + ex0) / 2
    if dim_metrics:
        spine_col = DIM_STROKE
        spine_lw = 0.5
        spine_alpha = 0.25
    elif highlight.k is not None or highlight.system is not None:
        spine_col = bus_accent
        spine_lw = 1.4
        spine_alpha = 1.0
    else:
        spine_col = INK_FAINT
        spine_lw = 0.7
        spine_alpha = 0.55
    spine_y0 = metric_y_mids[-1]
    spine_y1 = metric_y_mids[0]
    vline(ax, merge_x, spine_y0, spine_y1,
          color=spine_col, lw=spine_lw, alpha=spine_alpha, zorder=5)
    # ticks into each metric
    for (_x, my) in metrics_left_anchors:
        hline(ax, merge_x, _x - 0.003, my,
              color=spine_col, lw=spine_lw * 0.85, alpha=spine_alpha,
              zorder=5)
        # arrowhead at metric edge
        s = 0.008
        ax.plot([_x - 0.003 - s * 0.7, _x - 0.003, _x - 0.003 - s * 0.7],
                [my + s * 0.35, my, my - s * 0.35],
                color=spine_col, lw=spine_lw * 0.85, alpha=spine_alpha,
                zorder=6, solid_capstyle="round", solid_joinstyle="round")

    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches=None,
                pad_inches=0)
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)


# --------------------------------------------------------------------- entry

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlight", choices=list(HIGHLIGHTS), default=None,
                    help="render a single named variant; default renders the overview AND the recommended variant")
    ap.add_argument("--out-dir", default="thesis/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.highlight:
        names = [args.highlight]
    else:
        names = [
            "none", "systems-only", "graph1hop-indegree",
            "query-focus", "seeds-focus", "expansion-focus",
            "systems-focus", "pool-focus",
            "ranking-focus",
            "cosine-focus", "indegree-focus", "crossencoder-focus",
            "topk-focus", "metrics-focus",
        ]

    for name in names:
        h = HIGHLIGHTS[name]
        if name == "none":
            out = out_dir / "pipeline-overview.png"
        else:
            out = out_dir / f"pipeline-overview-{name}.png"
        render(h, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
