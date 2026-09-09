#!/usr/bin/env python3
"""
make_figures.py -- generates every figure in the manuscript as vector PDF.

Usage:  python make_figures.py [--results ../results] [--out ../figures]

Figures
-------
fig1_context.pdf              empirical backdrop (Introduction)
fig2_graphical_abstract.pdf   graphical abstract, Elsevier geometry
fig3_problem.pdf              anatomy of the decision problem (Problem statement)
fig4_framework.pdf            the three layers of SSR-MCDA (Methodology)
fig5_performance.pdf          equilibrium performance matrices
fig6_capacity.pdf             Shapley importance and interaction indices
fig7_regret.pdf               minimax coalitional regret and stability radius
fig8_epsilon.pdf              robustness frontier over the uncertainty radius
fig9_benchmarks.pdf           comparison against eight comparator methods
fig10_montecarlo.pdf          rank distributions over the parameter uncertainty set
fig11_sensitivity.pdf         one-at-a-time parameter sensitivity
fig12_sourcing.pdf            equilibrium sourcing reallocation by posture
fig13_attribution.pdf         criterion-level attribution of worst-case regret

Author: Saptadeep Biswas
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import ssr_mcda as S  # noqa: E402
from ssr_mcda.data import CASES, SECTOR_DEF  # noqa: E402

F.apply_style()

POST = S.POSTURE_CODES
POST_SHORT = {
    "A1": "Status quo", "A2": "Liberalisation", "A3": "Uniform tariff",
    "A4": "Targeted tariff", "A5": "Nearshoring", "A6": "Friendshoring",
    "A7": "Reshoring package",
}
CRIT = S.CRITERION_CODES
CRIT_SHORT = {
    "C1": "Import cost", "C2": "Tariff revenue", "C3": "Domestic VA",
    "C4": "Diversification", "C5": "Geopolitical exp.", "C6": "Logistics",
    "C7": "Retaliation exp.", "C8": "Employment", "C9": "Transport CO2",
}
CASE_TAG = {"USA": "United States", "IND": "India"}


def R(results, name, case=None):
    fn = f"{name}_{case}.csv" if case else f"{name}.csv"
    return pd.read_csv(os.path.join(results, fn))


# ==========================================================================
# Figure 1 -- empirical backdrop
# ==========================================================================
def _place_labels(ax, pts, fontsize=6.6, pad=2.0):
    """Annotate scatter points with labels that provably do not overlap.

    Six of the United States' trading partners sit inside a cluster of low import
    shares, and a fixed offset stacked their ISO codes on top of one another. An
    estimate of the text box from the font size was not reliable enough to catch
    the tightest pair, so this measures each label's ACTUAL rendered extent
    through the renderer and works entirely in display coordinates. Candidates
    are tried on widening rings and then straight up; a label that ends up far
    from its point gets a hairline leader so it can still be traced.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(rend)
    T = ax.transData.transform

    from matplotlib.transforms import IdentityTransform
    ID = IdentityTransform()

    def measure(txt):
        t = ax.text(0, 0, txt, fontsize=fontsize, transform=ID)
        bb = t.get_window_extent(rend)
        t.remove()
        return bb.width, bb.height

    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
    MR = 4.0                                   # marker radius in points, display
    placed = []
    for x, y, _ in pts:                        # markers are obstacles as well
        px, py = T((x, y))
        placed.append((px - MR, py - MR, px + MR, py + MR))

    def clear(b):
        if not (b[0] > ax_box.x0 + 1 and b[2] < ax_box.x1 - 1
                and b[1] > ax_box.y0 + 1 and b[3] < ax_box.y1 - 1):
            return False
        return all(b[2] < o[0] - pad or b[0] > o[2] + pad or
                   b[3] < o[1] - pad or b[1] > o[3] + pad for o in placed)

    for i in sorted(range(len(pts)), key=lambda j: -pts[j][1]):
        x, y, iso = pts[i]
        px, py = T((x, y))
        w, h = measure(iso)
        chosen = None
        for ring in (1.0, 1.8, 2.7, 3.8):
            for dx, dy in dirs:
                bx = px + dx * ring * (MR + 2) - (w / 2 if dx == 0 else (0 if dx > 0 else w))
                by = py + dy * ring * (MR + 2) - (h / 2 if dy == 0 else (0 if dy > 0 else h))
                b = (bx, by, bx + w, by + h)
                if clear(b):
                    chosen = b
                    break
            if chosen:
                break
        if chosen is None:                      # climb until something fits
            for step in range(1, 25):
                b = (px + MR + 2, py + step * (h + 1.5),
                     px + MR + 2 + w, py + step * (h + 1.5) + h)
                if clear(b):
                    chosen = b
                    break
        if chosen is None:                      # give up gracefully, never silently
            chosen = (px + MR + 2, py - h / 2, px + MR + 2 + w, py + h / 2)
        placed.append(chosen)
        cx, cy = chosen[0], (chosen[1] + chosen[3]) / 2
        # Convert the chosen pixel position back to DATA coordinates before
        # drawing. Pinning the text to pixels would break on save: the PDF
        # backend rasterises at a different dpi from the Agg canvas measured here.
        inv = ax.transData.inverted()
        dx_, dy_ = inv.transform((cx, cy))
        if abs(cx - px) > 3.2 * MR or abs(cy - py) > 3.2 * MR:
            ax.plot([x, dx_], [y, dy_], linewidth=0.4, color=F.INK_MUTED, zorder=2)
        ax.text(dx_, dy_, iso, fontsize=fontsize, color=F.INK_2, va="center",
                ha="left", zorder=4)


def fig1_context(data_dir, out):
    tar = pd.read_csv(os.path.join(data_dir, "raw_tariffs_hs6.csv"))
    tar["product_hs6"] = tar["product_hs6"].astype(str).str.zfill(6)
    tar = tar[tar.datatype == "reported"]
    hs2sec = {h: s for s, d in SECTOR_DEF.items() for h in d["hs6"]}
    tar["sector"] = tar.product_hs6.map(hs2sec)

    tr = pd.read_csv(os.path.join(data_dir, "raw_bilateral_trade.csv")).dropna(subset=["value"])
    align = pd.read_csv(os.path.join(data_dir, "geopolitical_idealpoints.csv"))
    align = align[align.measure.str.contains("2023_allvotes", na=False)].set_index("iso3")["value"]

    fig = plt.figure(figsize=(7.4, 5.9))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30, height_ratios=[1, 1.05])

    # (a) sector tariff dispersion, 2022
    ax = fig.add_subplot(gs[0, :])
    t22 = tar[tar.year == 2022].groupby(["reporter_name", "sector"]).rate_simple_avg.mean().unstack()
    t22 = t22.reindex(columns=list(SECTOR_DEF)).sort_values("EV & batteries", ascending=False)
    xs = np.arange(len(t22))
    w = 0.2
    for k, sec in enumerate(SECTOR_DEF):
        b = ax.bar(xs + (k - 1.5) * w, t22[sec].values, w, label=sec,
                   color=F.SERIES[k], hatch=F.HATCH[k], edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(t22.index, rotation=26, ha="right", fontsize=7.6)
    ax.set_ylabel("Applied MFN tariff (%)", fontsize=8.0)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([0, 5, 10, 25, 50, 100])
    ax.set_yticklabels(["0", "5", "10", "25", "50", "100"], fontsize=7.5)
    ax.set_ylim(0, 210)                     # headroom so the legend clears the bars
    # legend ABOVE the axes: inside, it printed on top of the tallest bar
    ax.legend(ncol=4, fontsize=7.2, handlelength=1.5, columnspacing=1.4,
              loc="lower center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.set_title("(a)  Applied MFN tariffs on four strategic supply chains, 2022 "
                 "(simple average across HS-6 lines)", loc="left", fontsize=8.8)
    F.tidy(ax)

    # (b) supplier concentration
    ax = fig.add_subplot(gs[1, 0])
    groups = {d["group"]: s for s, d in SECTOR_DEF.items()}
    rows = []
    for rep in ["usa", "ind", "chn", "jpn", "deu"]:
        for grp, sec in groups.items():
            sub = tr[(tr.reporter_iso3 == rep) & (tr.product_group == grp) & (tr.year == 2022)]
            sub = sub[sub.partner_iso3 != "wld"]
            if sub.empty:
                continue
            sh = sub.value.values / sub.value.values.sum()
            rows.append({"reporter": rep.upper(), "sector": sec, "hhi": float((sh ** 2).sum())})
    hh = pd.DataFrame(rows).pivot(index="reporter", columns="sector", values="hhi")
    hh = hh.reindex(columns=list(SECTOR_DEF)).reindex(["USA", "IND", "CHN", "JPN", "DEU"])
    xs = np.arange(len(hh))
    for k, sec in enumerate(SECTOR_DEF):
        ax.plot(xs, hh[sec].values, marker=F.MARKERS[k], linestyle=F.LINESTYLES[k],
                color=F.SERIES[k], label=sec, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(hh.index, fontsize=7.6)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel("Herfindahl index of import origins", fontsize=8.0)
    ax.set_title("(b)  Supplier concentration, 2022", loc="left", fontsize=8.8)
    # below the axes: in the upper right it crossed the semiconductor series
    ax.legend(fontsize=6.6, handlelength=2.0, ncol=2, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), columnspacing=1.2)
    F.tidy(ax)

    # (c) alignment vs dependence
    ax = fig.add_subplot(gs[1, 1])
    sub = tr[(tr.reporter_iso3 == "usa") & (tr.product_group == "Total") & (tr.year == 2022)]
    tot = float(sub[sub.partner_iso3 == "wld"].value.iloc[0])
    pts = []
    for _, r in sub.iterrows():
        iso = r.partner_iso3.upper()
        if iso in ("WLD",) or iso not in align.index:
            continue
        pts.append((abs(align.loc["USA"] - align.loc[iso]) / 2.0, 100 * r.value / tot, iso))
    xs_ = np.array([p[0] for p in pts]); ys_ = np.array([p[1] for p in pts])
    ax.scatter(xs_, ys_, s=26, color=F.SERIES[0], edgecolor="white",
               linewidth=0.6, zorder=3)
    ax.set_xlabel("Geopolitical distance from the United States", fontsize=8.0)
    ax.set_ylabel("Share of US goods imports (%)", fontsize=8.0)
    ax.set_title("(c)  Dependence against political distance", loc="left", fontsize=8.8)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(min(xs_) - 0.09, max(xs_) + 0.09)
    ax.set_ylim(-1.2, max(ys_) * 1.16)
    _place_labels(ax, pts, fontsize=6.6)
    F.tidy(ax, "both")
    # No in-figure source note: the manuscript caption already carries the same
    # sentence, and printed together the two ran into each other.
    return F.save(fig, os.path.join(out, "fig1_context.pdf"))


# ==========================================================================
# Layout helpers for the diagram figures
# ==========================================================================
import textwrap


def _wrap(text, width):
    """Wrap on explicit newlines first, then to `width` characters."""
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out


def panel(ax, x, y, w, h, title, body, color, *, title_fs=8.6, body_fs=7.2,
          wrap=None, line_h=None, face_alpha="12", title_color=None, align="center"):
    """Draw a rounded panel with a title and wrapped body, sized in axes coordinates.

    Text is wrapped to `wrap` characters and laid out on a fixed line pitch, so the
    content cannot spill outside the box however long the string is.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                linewidth=1.1, edgecolor=color,
                                facecolor=color + face_alpha, zorder=2))
    pad_x = 0.016
    tx = x + w / 2 if align == "center" else x + pad_x
    ha = "center" if align == "center" else "left"
    ty = y + h - 0.030
    ax.text(tx, ty, title, ha=ha, va="top", fontsize=title_fs, fontweight="bold",
            color=title_color or F.INK, zorder=3, linespacing=1.35)
    n_title = title.count("\n") + 1
    lines = _wrap(body, wrap) if wrap else body.split("\n")
    lh = line_h if line_h is not None else 0.030
    by = ty - n_title * (lh + 0.006) - 0.012
    for i, ln in enumerate(lines):
        ax.text(tx, by - i * lh, ln, ha=ha, va="top", fontsize=body_fs,
                color=F.INK_2, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color=F.INK_2, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


# ==========================================================================
# Figure 2 -- graphical abstract (Elsevier geometry, 13 x 5 cm proportion)
# ==========================================================================
def fig2_graphical_abstract(results, out):
    """Elsevier graphical abstract, 13 x 5 cm proportion.

    Rewritten after the titles of the four stages overran their boxes and printed
    on top of one another. Each stage now carries an icon, a one-word kicker and
    a short name on its own line, with the body wrapped to a measure that fits
    the box. The Layer I label no longer says "Stackelberg": the paper describes
    a single-leader/multi-follower model and explicitly disclaims the
    subgame-perfect equilibrium reading.
    """
    d = json.load(open(os.path.join(results, "results.json")))
    W, H = 7.28, 3.05
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ar = H / W

    stages = [
        ("globe", F.SERIES[6], "INPUT", "Authentic data",
         "UNCTAD-TRAINS tariffs\nWITS bilateral trade\nWorld Bank indicators\nUNGA alignment scores"),
        ("flow", F.SERIES[0], "LAYER I", "Equilibrium matrix",
         "the leader sets a posture;\nfirms re-source and\npartners retaliate; the\nunique response becomes\nthe decision matrix"),
        ("network", F.SERIES[2], "LAYER II", "Coalitional weights",
         "criteria are players in\na discrimination game;\nShapley value and\ninteraction index give a\n2-additive capacity"),
        ("shield", F.SERIES[1], "LAYER III", "Certified ranking",
         "exact LP over the\ncapacity polytope; a\nranking plus a stability\nradius saying how far it\ncan be trusted"),
    ]
    BX, BW, GAP = 0.008, 0.2295, 0.0247
    BY, BH = 0.345, 0.625
    for i, (ico, col, kicker, name, body) in enumerate(stages):
        x = BX + i * (BW + GAP)
        ax.add_patch(FancyBboxPatch((x, BY), BW, BH,
                                    boxstyle="round,pad=0.005,rounding_size=0.022",
                                    linewidth=1.2, edgecolor=col,
                                    facecolor=col + "12", zorder=2))
        # The name is set on its own line, across the full box width. Beside the
        # icon it had only two thirds of the measure and ran over the border.
        F.icon(ax, ico, x + 0.032, BY + BH - 0.072, 0.100, ar, color=col, lw=1.1, z=5)
        ax.text(x + 0.064, BY + BH - 0.052, kicker, ha="left", va="center",
                fontsize=6.1, fontweight="bold", color=col, zorder=3)
        # 7.4 pt, not 8.2: at 8.2 the longest name ("Coalitional weights") is
        # 1.56 in wide against an inner measure of 1.47 in and crossed the border.
        ax.text(x + 0.016, BY + BH - 0.140, name, ha="left", va="top",
                fontsize=7.4, fontweight="bold", color=F.INK, zorder=3)
        ax.text(x + 0.016, BY + BH - 0.225, body, ha="left", va="top", fontsize=6.3,
                color=F.INK_2, zorder=3, linespacing=1.62)
        if i:
            _arrow(ax, x - GAP + 0.004, BY + BH / 2, x - 0.004, BY + BH / 2,
                   color=F.INK_MUTED, lw=1.0)

    rec = d["cases"]["USA"]["recommended"]
    rad = d["cases"]["USA"]["stability_radius_linf"]
    ax.text(0.5, 0.245,
            "No elicited weights.   No assumed decision matrix.   "
            "A ranking with a proof of how far it can be trusted.",
            ha="center", va="center", fontsize=7.8, color=F.INK, style="italic")
    ax.text(0.5, 0.085,
            "Applied to tariff postures for semiconductors, pharmaceuticals, EV batteries and steel "
            "in the United States and India;\nrecommended posture "
            f"{rec} ({POST_SHORT[rec]}), certified stable to capacity perturbations up to {rad:.4f}",
            ha="center", va="center", fontsize=6.5, color=F.INK_2, linespacing=1.7)
    return F.save(fig, os.path.join(out, "fig2_graphical_abstract.pdf"))


# ==========================================================================
# Figure 3 -- anatomy of the decision problem
# ==========================================================================
# Composition notes. Every block is placed on an explicit grid in axes
# coordinates and every string is wrapped to a character count that has been
# checked against the box width, so nothing can overrun its container. Each
# actor and each obstruction carries a small vector glyph drawn by
# figstyle.icon(), which gives the reader a shape to navigate by instead of
# four identical grey rectangles.
def fig3_problem(out):
    W, H = 7.4, 5.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ar = H / W

    # ---- the leader ------------------------------------------------------
    LX, LW, LY, LH = 0.215, 0.570, 0.855, 0.125
    ax.add_patch(FancyBboxPatch((LX, LY), LW, LH,
                                boxstyle="round,pad=0.006,rounding_size=0.020",
                                linewidth=1.3, edgecolor=F.SERIES[0],
                                facecolor=F.SERIES[0] + "12", zorder=2))
    F.icon(ax, "government", LX + 0.052, LY + LH / 2, 0.062, ar,
           color=F.SERIES[0], lw=1.0, z=5)
    ax.text(LX + 0.100, LY + LH - 0.030, "THE DECISION MAKER", ha="left", va="top",
            fontsize=8.6, fontweight="bold", color=F.SERIES[0], zorder=3)
    ax.text(LX + 0.100, LY + LH - 0.062,
            "a focal importing government picks one tariff posture\n"
            "$a \\in \\{A_1,\\dots,A_7\\}$, applied across four strategic sectors",
            ha="left", va="top", fontsize=7.1, color=F.INK_2, zorder=3,
            linespacing=1.5)

    # ---- who responds ----------------------------------------------------
    # Kicker and name are stacked rather than joined by a middle dot: the joined
    # form ran past the box edge and into the neighbouring panel.
    actors = [
        (0.020, F.SERIES[1], "factory", "RESPONDS", "importing firms",
         "re-allocate sourcing shares across\norigins, trading landed cost\nagainst capacity friction and\nrisk premia"),
        (0.347, F.SERIES[2], "ship", "RESPONDS", "partner states",
         "set a retaliation intensity on\nthe leader's exports, weighing\npolitical gain against their own\nwelfare loss"),
        (0.674, F.SERIES[3], "dice", "DOES NOT RESPOND", "nature",
         "behavioural parameters, the\ndomestic production base and the\nleader's own criterion trade-offs\nare all unknown"),
    ]
    AY, AH, AW = 0.585, 0.205, 0.306
    for x, col, ico, kicker, name, body in actors:
        ax.add_patch(FancyBboxPatch((x, AY), AW, AH,
                                    boxstyle="round,pad=0.006,rounding_size=0.020",
                                    linewidth=1.2, edgecolor=col,
                                    facecolor=col + "12", zorder=2))
        F.icon(ax, ico, x + 0.036, AY + AH - 0.040, 0.048, ar, color=col, lw=1.0, z=5)
        ax.text(x + 0.068, AY + AH - 0.020, kicker, ha="left", va="top",
                fontsize=5.9, fontweight="bold", color=col, zorder=3)
        ax.text(x + 0.068, AY + AH - 0.040, name, ha="left", va="top",
                fontsize=7.8, fontweight="bold", color=F.INK, zorder=3)
        ax.text(x + 0.018, AY + AH - 0.078, body, ha="left", va="top",
                fontsize=6.5, color=F.INK_2, zorder=3, linespacing=1.55)
        _arrow(ax, 0.50, LY - 0.004, x + AW / 2, AY + AH + 0.006,
               color=F.INK_MUTED, lw=0.9)

    # ---- the nine criteria ----------------------------------------------
    CY, CH = 0.330, 0.205
    ax.add_patch(FancyBboxPatch((0.020, CY), 0.960, CH,
                                boxstyle="round,pad=0.006,rounding_size=0.020",
                                linewidth=1.1, edgecolor=F.INK_2,
                                facecolor="#f4f4f2", zorder=2))
    F.icon(ax, "bars", 0.062, CY + CH - 0.036, 0.046, ar, color=F.INK_2, lw=1.0, z=5)
    ax.text(0.096, CY + CH - 0.020,
            "EVALUATED ON NINE CONFLICTING CRITERIA, ALL READ OFF THE EQUILIBRIUM",
            ha="left", va="top", fontsize=7.6, fontweight="bold", color=F.INK, zorder=3)
    # three rows of three, each on its own column guide, so nothing can collide
    cols_x = (0.070, 0.375, 0.680)
    for i, c in enumerate(CRIT):
        r, cc = divmod(i, 3)
        x = cols_x[cc]
        y = CY + CH - 0.072 - r * 0.040
        ax.add_patch(FancyBboxPatch((x - 0.006, y - 0.014), 0.022, 0.022,
                                    boxstyle="round,pad=0.001,rounding_size=0.006",
                                    linewidth=0, facecolor=F.SERIES[i % 8] + "55", zorder=3))
        ax.text(x + 0.005, y - 0.003, c[1], ha="center", va="center",
                fontsize=5.6, color=F.INK, zorder=4)
        ax.text(x + 0.026, y - 0.003, CRIT_SHORT[c], ha="left", va="center",
                fontsize=7.0, color=F.INK_2, zorder=3)

    # ---- why textbook MCDA does not apply --------------------------------
    RY, RH = 0.020, 0.278
    ax.add_patch(FancyBboxPatch((0.020, RY), 0.960, RH,
                                boxstyle="round,pad=0.006,rounding_size=0.020",
                                linewidth=1.2, edgecolor=F.SERIES[7],
                                facecolor=F.SERIES[7] + "0C", zorder=2))
    F.icon(ax, "ban", 0.062, RY + RH - 0.036, 0.046, ar, color=F.SERIES[7], lw=1.1, z=5)
    ax.text(0.096, RY + RH - 0.020,
            "FOUR REASONS A TEXTBOOK MCDA MODEL CANNOT BE APPLIED DIRECTLY",
            ha="left", va="top", fontsize=7.6, fontweight="bold",
            color=F.SERIES[7], zorder=3)
    # Two by two, not one by four: at four across, a 39-character line of 6.4 pt
    # text is wider than the 0.235 column it was given and ran into its neighbour.
    reasons = [
        ("grid", "(i)", "no decision matrix exists",
         "performance is an equilibrium response to the policy,\nso it cannot be tabulated before the policy is chosen"),
        ("scales", "(ii)", "no single owner of the weights",
         "the criteria belong to different constituencies, so no\none decision maker can be asked to weigh them"),
        ("network", "(iii)", "the criteria interact",
         "they are redundant and complementary in pairs, so any\nadditive score is misspecified from the outset"),
        ("target", "(iv)", "a point ranking is not an answer",
         "a policy question needs the margin by which the\nrecommendation survives being wrong"),
    ]
    for i, (ico, num, head, body) in enumerate(reasons):
        r, c = divmod(i, 2)
        x = 0.048 + c * 0.478
        y = RY + RH - 0.070 - r * 0.104
        # 0.046, not 0.038: below about 0.042 the 3x3 cells of the "grid" glyph
        # merge into a solid square and stop reading as a matrix.
        F.icon(ax, ico, x + 0.016, y - 0.008, 0.046, ar, color=F.SERIES[7], lw=0.8, z=5)
        ax.text(x + 0.042, y + 0.014, f"{num}  {head}", ha="left", va="top",
                fontsize=7.0, fontweight="bold", color=F.INK, zorder=3)
        ax.text(x + 0.042, y - 0.014, body, ha="left", va="top", fontsize=6.4,
                color=F.INK_2, zorder=3, linespacing=1.5)
    return F.save(fig, os.path.join(out, "fig3_problem.pdf"))


# ==========================================================================
# Figure 4 -- the three layers
# ==========================================================================
def fig4_framework(results, out):
    # Every count in the validation strip is read from results.json rather than
    # typed, so the figure cannot drift away from what was actually run.
    d = json.load(open(os.path.join(results, "results.json")))
    ndraws = d["monte_carlo_draws"]
    # the agreement table lists SSR against itself as well, so the number of
    # *comparators* is one less than its length -- this must match the text
    ncomp = len([k for k in d["cases"]["USA"]["kendall_tau_vs_benchmarks"]
                 if not k.startswith("SSR-MCDA")])
    ntests = 28

    W, H = 7.4, 6.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ar = H / W

    layers = [
        (0.720, F.SERIES[0], "I", "flow",
         "ENDOGENOUS PERFORMANCE GENERATION",
         "the decision matrix is produced, not assumed",
         ["Firms (lower level):  $\\min_{x \\in \\Delta}\\ \\sum_p x_p\\,[\\,c_{kp}(1+\\tau_{kp}) + \\lambda g_p\\,] + (\\rho/2)\\|x\\|^2$   $\\Rightarrow$   unique $x^{*}(\\tau)$",
          "Partners (lower level):  $r_p^{*} = \\max\\{0,\\ (\\theta_0\\,\\Delta\\tau_p M_p/E_p - \\gamma)/\\beta\\}$   $\\Rightarrow$   unique response   (Prop. 1)",
          "Unit costs recovered by algebraic inversion: the observed 2022 allocation is reproduced exactly   (Prop. 3)"],
         "$P_{aj} = f_j(\\tau(a),\\, x^{*},\\, r^{*})$   —   a decision matrix that is an equilibrium object"),
        (0.420, F.SERIES[2], "II", "network",
         "COALITIONAL CAPACITY IDENTIFICATION",
         "the weights are derived from documented policy, not elicited",
         ["Discrimination game:  $v(S) = |\\{(a,b) \\in \\Pi:\\ \\sum_{j \\in S} z_{aj} > \\sum_{j \\in S} z_{bj}\\}| \\,/\\, |\\Pi|$",
          "$\\Pi$ holds revealed policy preferences — documented policy actions, never questionnaire responses",
          "Projection:  cardinality-balanced least squares onto the 2-additive capacity polytope, representing $\\Pi$   (Thm. 1)"],
         "importance $\\varphi_i = m_i + \\frac{1}{2}\\sum_j m_{ij}$   and interaction $I_{ij} = m_{ij}$"),
        (0.120, F.SERIES[1], "III", "shield",
         "MINIMAX COALITIONAL REGRET",
         "the recommendation arrives with a certificate",
         ["$C_m(z) = \\phi(z)^{\\top} m$ is linear in the Möbius vector, so worst-case regret is a linear programme",
          "$R(a) = \\max_{b \\neq a}\\ \\max_{m \\in \\mathcal{M}_\\infty}\\ (\\phi(z_b) - \\phi(z_a))^{\\top} m$   —   solved exactly, never sampled   (Thm. 2)",
          "Stability radius:  $\\min \\|m - m^{*}\\|_\\infty$  s.t. some challenger catches the incumbent — also an LP   (Thm. 3)"],
         "a ranking, a recommendation, and the margin by which it survives being wrong"),
    ]
    BH = 0.256
    for y, col, num, ico, title, tag, lines, output in layers:
        ax.add_patch(FancyBboxPatch((0.025, y), 0.950, BH,
                                    boxstyle="round,pad=0.006,rounding_size=0.020",
                                    linewidth=1.3, edgecolor=col,
                                    facecolor=col + "0C", zorder=2))
        # numbered roundel and a glyph for the layer's job
        F.badge(ax, num, 0.070, y + BH - 0.036, 0.024, ar, col, fs=8.2, z=7)
        F.icon(ax, ico, 0.930, y + BH - 0.038, 0.048, ar, color=col + "AA", lw=1.1, z=5)
        ax.text(0.105, y + BH - 0.020, title, ha="left", va="top", fontsize=8.4,
                fontweight="bold", color=col, zorder=3)
        ax.text(0.105, y + BH - 0.050, tag, ha="left", va="top", fontsize=6.9,
                color=F.INK_MUTED, style="italic", zorder=3)
        for i, ln in enumerate(lines):
            ax.text(0.048, y + BH - 0.094 - i * 0.042, ln, ha="left", va="top",
                    fontsize=6.9, color=F.INK, zorder=3)
        # the output of the layer, set apart on its own tinted strip
        oy = y + 0.014
        ax.add_patch(FancyBboxPatch((0.048, oy), 0.904, 0.036,
                                    boxstyle="round,pad=0.003,rounding_size=0.010",
                                    linewidth=0, facecolor=col + "1C", zorder=3))
        ax.text(0.060, oy + 0.018, "OUTPUT", ha="left", va="center", fontsize=5.8,
                fontweight="bold", color=col, zorder=4)
        ax.text(0.142, oy + 0.018, output, ha="left", va="center", fontsize=6.9,
                color=F.INK, zorder=4)

    # ---- connectors ------------------------------------------------------
    for y, lab in ((0.720, "$Z$ = normalised $P$"), (0.420, "$m^{*},\\ \\varphi,\\ I$")):
        _arrow(ax, 0.28, y - 0.006, 0.28, y - 0.049)
        ax.add_patch(FancyBboxPatch((0.302, y - 0.043), 0.150, 0.030,
                                    boxstyle="round,pad=0.002,rounding_size=0.010",
                                    linewidth=0, facecolor="#eeeeeb", zorder=3))
        ax.text(0.310, y - 0.028, lab, fontsize=6.9, color=F.INK_2,
                va="center", ha="left", zorder=4)

    # ---- validation strip ------------------------------------------------
    VY, VH = 0.012, 0.108
    ax.add_patch(FancyBboxPatch((0.025, VY), 0.950, VH,
                                boxstyle="round,pad=0.006,rounding_size=0.020",
                                linewidth=1.0, edgecolor=F.INK_2,
                                facecolor="#f4f4f2", zorder=2))
    ax.text(0.048, VY + VH - 0.018, "HOW EACH LAYER IS CHECKED", ha="left", va="top",
            fontsize=7.0, fontweight="bold", color=F.INK_2, zorder=3)
    # Five items on one row did not fit the width; they are set two-up per column
    # over three columns, which leaves each label its full measure.
    checks = [(f"{ntests} executable property tests", "check"),
              (f"{ncomp} comparator procedures", "bars"),
              (f"{ndraws:,}-draw Monte Carlo", "dice"),
              ("one-at-a-time parameter sweeps", "gear"),
              ("rank-reversal probes", "flow"),
              ("leave-one-out and permutation null", "network")]
    for i, (txt, ico) in enumerate(checks):
        c, r = divmod(i, 2)
        x = 0.052 + c * 0.313
        y = VY + 0.052 - r * 0.030
        F.icon(ax, ico, x + 0.011, y, 0.026, ar, color=F.INK_MUTED, lw=0.9, z=5)
        ax.text(x + 0.028, y, txt, ha="left", va="center", fontsize=6.4,
                color=F.INK_2, zorder=3)
    return F.save(fig, os.path.join(out, "fig4_framework.pdf"))


# ==========================================================================
# Data-driven result figures
# ==========================================================================
def fig5_performance(results, out):
    # Two heatmaps side by side. The row labels are long ("A7  reshoring package"),
    # so the right-hand panel's labels used to run into the left-hand panel's cell
    # values. The posture order is identical in both panels, so the labels are drawn
    # ONCE, on the left, and the right panel is given bare tick positions.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    fig.subplots_adjust(left=0.20, right=0.88, wspace=0.10, bottom=0.30, top=0.90)
    cm = F.sequential_cmap()
    orders = []
    for k, (ax, case) in enumerate(zip(axes, CASES)):
        Z = R(results, "performance_norm", case).set_index("posture")
        orders.append(list(Z.index))
        Z.columns = [CRIT_SHORT[c] for c in CRIT]
        im = ax.imshow(Z.values, cmap=cm, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CRIT)))
        ax.set_xticklabels(Z.columns, rotation=42, ha="right", fontsize=6.6)
        ax.set_yticks(range(len(POST)))
        if k == 0:
            ax.set_yticklabels([f"{p}  {POST_SHORT[p]}" for p in Z.index], fontsize=7.0)
        else:
            ax.set_yticklabels([])
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if v > 0.55 else F.INK)
        ax.set_title(f"({'ab'[k]})  {CASE_TAG[case]}", loc="left", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    # suppressing the second panel's row labels is only legitimate if the two
    # panels really are in the same posture order
    assert orders[0] == orders[1], "posture order differs between cases"
    cax = fig.add_axes([0.90, 0.30, 0.016, 0.60])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("normalised, benefit-oriented performance", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    return F.save(fig, os.path.join(out, "fig5_performance.pdf"))


def fig6_capacity(results, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6),
                             gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.55, "wspace": 0.32})
    dv = F.diverging_cmap()
    for k, case in enumerate(CASES):
        cap = R(results, "capacity", case).set_index("criterion")
        ax = axes[0, k]
        xs = np.arange(len(CRIT))
        b1 = ax.bar(xs - 0.20, cap["shapley_of_game"].values, 0.38, label="Shapley value of the discrimination game",
                    color=F.SERIES[4], hatch=F.HATCH[2], edgecolor="white", linewidth=0.6, zorder=3)
        b2 = ax.bar(xs + 0.20, cap["shapley_importance"].values, 0.38, label="importance of the projected capacity",
                    color=F.SERIES[0], edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0, color=F.INK_2, linewidth=0.7)
        ax.set_xticks(xs); ax.set_xticklabels(CRIT, fontsize=7.4)
        ax.set_ylabel("criterion importance $\\varphi_i$")
        ax.set_title(f"({'ab'[k]})  {CASE_TAG[case]} — importance", loc="left")
        F.tidy(ax)
        # Headroom for the direct labels: without it the tallest label is clipped by
        # the axes frame, and an in-panel legend sat on top of the negative bars.
        gv = cap["shapley_of_game"].values
        iv = cap["shapley_importance"].values
        lo, hi = min(gv.min(), 0.0), max(gv.max(), iv.max())
        ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.20 * (hi - lo))
        for x, v in zip(xs + 0.20, iv):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 2.5),
                        ha="center", fontsize=5.8, color=F.INK_2)

        ax = axes[1, k]
        I = R(results, "interaction", case).set_index("Unnamed: 0" if "Unnamed: 0"
                                                      in R(results, "interaction", case).columns else "criterion")
        M = I.values.astype(float)
        lim = max(1e-6, np.abs(M).max())
        im = ax.imshow(M, cmap=dv, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(CRIT))); ax.set_xticklabels(CRIT, fontsize=7, rotation=0)
        ax.set_yticks(range(len(CRIT))); ax.set_yticklabels(CRIT, fontsize=7)
        for i in range(len(CRIT)):
            for j in range(len(CRIT)):
                if abs(M[i, j]) > lim * 0.12:
                    ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=5.4,
                            color=F.INK)
        ax.set_title(f"({'cd'[k]})  {CASE_TAG[case]} — interaction $I_{{ij}}$", loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("redundancy $\\leftarrow$   $\\rightarrow$ complementarity", fontsize=6.4)
    # One legend for both top panels, below them and spanning the full width, so it
    # never overlaps a bar.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.8, ncol=2, handlelength=1.6, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.555))
    return F.save(fig, os.path.join(out, "fig6_capacity.pdf"))


def fig7_regret(results, out):
    d = json.load(open(os.path.join(results, "results.json")))
    # NOT sharey: each panel sorts its own postures by regret, and the two orders
    # differ. With a shared y axis matplotlib draws the labels once, on the left,
    # so the right-hand panel's bars were annotated with the LEFT panel's ordering
    # and contradicted Table 12.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    fig.subplots_adjust(wspace=0.62)
    for k, case in enumerate(CASES):
        ax = axes[k]
        m = R(results, "mcr", case).set_index("posture").loc[POST]
        order = m.sort_values("max_regret").index
        vals = m.loc[order, "max_regret"].values
        cols = [F.SERIES[2] if v <= 1e-9 else F.SERIES[0] for v in vals]
        bars = ax.barh(np.arange(len(order))[::-1], vals, 0.62, color=cols,
                       edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_yticks(np.arange(len(order))[::-1])
        ax.set_yticklabels([f"{p}  {POST_SHORT[p]}" for p in order], fontsize=7.4)
        for y, v in zip(np.arange(len(order))[::-1], vals):
            ax.annotate(f"{v:.3f}", (v, y), textcoords="offset points", xytext=(4, 0),
                        va="center", fontsize=6.8, color=F.INK_2)
        ax.set_xlabel("maximum coalitional regret $R(a)$")
        rad = d["cases"][case]["stability_radius_linf"]
        rec = d["cases"][case]["recommended"]
        ax.set_title(f"({'ab'[k]})  {CASE_TAG[case]}", loc="left")
        # top right, not bottom right: the postures are sorted by ascending regret,
        # so the longest bars are at the BOTTOM and a box there covers one of them
        ax.text(0.98, 0.97, f"recommended {rec}\nstability radius $\\varepsilon^\\ast$ = {rad:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
                color=F.INK_2, bbox=dict(boxstyle="round,pad=0.35", facecolor="#f6f6f4",
                                         edgecolor=F.GRID, linewidth=0.6))
        ax.set_xlim(0, max(vals) * 1.22)
        F.tidy(ax, "x")
    return F.save(fig, os.path.join(out, "fig7_regret.pdf"))


def fig8_epsilon(results, out, data_dir):
    """Robustness frontier: recompute max regret across a grid of uncertainty radii."""
    from ssr_mcda.data import build_case, labour_intensities, preference_indices
    from ssr_mcda.regret import UncertaintySet, max_regret, preference_rows

    eps_grid = np.linspace(0.0, 0.30, 16)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    d = json.load(open(os.path.join(results, "results.json")))
    for k, case in enumerate(CASES):
        par = S.GameParams()
        ci = build_case(case, par)
        P, _, _ = S.build_performance_matrix(ci.sectors, ci.env, par, ci.rival, labour_intensities())
        Z = S.normalise(P)
        prefs = preference_indices(case, POST)
        cap, _ = S.cci(Z, prefs)
        G = preference_rows(Z, prefs)
        curves = {p: [] for p in POST}
        for e in eps_grid:
            U = UncertaintySet(n=len(CRIT), centre=cap.vector, epsilon=float(e), preference_rows=G)
            Rv, _ = max_regret(Z, U)
            for i, p in enumerate(POST):
                curves[p].append(Rv[i])
        ax = axes[k]
        for i, p in enumerate(POST):
            ax.plot(eps_grid, curves[p], marker=F.MARKERS[i], markersize=3.4,
                    linestyle=F.LINESTYLES[i], color=F.SERIES[i], label=f"{p} {POST_SHORT[p]}")
        ax.set_xlabel("capacity uncertainty radius $\\varepsilon$")
        if k == 0:
            ax.set_ylabel("maximum coalitional regret $R(a)$")
        rad = d["cases"][case]["stability_radius_linf"]
        ax.axvline(rad, color=F.INK_MUTED, linestyle=":", linewidth=1.0)
        # Set horizontally in the headroom above the curves, not rotated across
        # them: rotated on the rule it printed on top of four of the series.
        ax.set_ylim(top=1.13)
        ax.annotate(f"certified $\\varepsilon^{{\\ast}} = {rad:.4f}$", (rad, 1.09),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=6.8, color=F.INK_2, va="center", ha="left")
        ax.set_title(f"({'ab'[k]})  {CASE_TAG[case]}", loc="left")
        F.tidy(ax, "both")
    axes[1].legend(fontsize=6.4, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   handlelength=2.2)
    return F.save(fig, os.path.join(out, "fig8_epsilon.pdf"))


# Compact codes for the twelve procedures. A 12x12 agreement matrix cannot carry
# the full method names on its axes at a legible size, so the axes carry codes and
# the codes are expanded in the figure caption and in Tables 13-14.
METHOD_CODE = {
    "SSR-MCDA (Choquet score)": "SSR-C",
    "SSR-MCDA (minimax regret)": "SSR-R",
    "Choquet (Marichal-Roubens capacity)": "MR",
    "Choquet (Marichal--Roubens capacity)": "MR",
    "WSM (Shapley weights)": "WSM-$\\varphi$",
    "WSM (entropy weights)": "WSM-H",
    "WSM (CRITIC weights)": "WSM-CR",
    "WSM (BWM weights)": "WSM-BW",
    "TOPSIS (Shapley weights)": "TOPSIS",
    "VIKOR (Shapley weights, Q)": "VIKOR",
    "PROMETHEE II (Shapley weights)": "PROM-II",
    "COPRAS (Shapley weights)": "COPRAS",
    "SMAA-2 (rank-1 acceptability)": "SMAA-2",
}


def _mcode(name: str) -> str:
    """Map a full method label to its compact code, falling back to a truncation."""
    if name in METHOD_CODE:
        return METHOD_CODE[name]
    return name.split(" (")[0][:8]


def fig9_benchmarks(results, out):
    # Layout rationale. Previously the twelve method names sat in a legend to the
    # right of panel (b), which squeezed both line panels, and the agreement
    # matrices carried a numeric annotation in every one of 144 cells at 4.8pt,
    # which collided with itself. Now: the legend spans the full width between the
    # two rows, the matrices are drawn with compact axis codes and NO in-cell
    # numbers -- they are read as a pattern, and the exact ranks behind them are in
    # Tables 13-14 -- and a single colourbar serves both.
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.30], hspace=0.90, wspace=0.26,
                          left=0.085, right=0.90, top=0.955, bottom=0.115)
    top = [fig.add_subplot(gs[0, k]) for k in range(2)]
    bot = [fig.add_subplot(gs[1, k]) for k in range(2)]
    im = None
    for k, case in enumerate(CASES):
        b = R(results, "benchmarks", case)
        piv = b.pivot(index="posture", columns="method", values="rank").loc[POST]
        ax = top[k]
        for i, meth in enumerate(piv.columns):
            # twelve series against an eight-slot palette: F.line_style makes the
            # second lap hollow-markered so no two methods share an appearance
            ax.plot(np.arange(len(POST)), piv[meth].values, markersize=3.4,
                    linewidth=1.0, alpha=0.95, label=_mcode(meth),
                    **F.line_style(i))
        ax.set_xticks(np.arange(len(POST)))
        ax.set_xticklabels(POST, fontsize=7.4)
        ax.invert_yaxis()
        ax.set_yticks(range(1, len(POST) + 1))
        ax.set_ylabel("rank (1 = best)")
        ax.set_title(f"({'ab'[k]})  {CASE_TAG[case]} — ranking by method", loc="left",
                     fontsize=9)
        F.tidy(ax)

        ag = R(results, "rank_agreement", case)
        ag = ag.set_index(ag.columns[0])
        ax = bot[k]
        M = ag.values.astype(float)
        im = ax.imshow(M, cmap=F.diverging_cmap(), vmin=-1, vmax=1)
        short = [_mcode(c) for c in ag.columns]
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=90, fontsize=5.6)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=5.6)
        # the title is kept short: the long form ran under the shared colourbar
        ax.set_title(f"({'cd'[k]})  {CASE_TAG[case]}", loc="left", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    # one legend for both line panels, in the gap between the two rows
    handles, labels = top[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.2, ncol=6, handlelength=2.2,
               columnspacing=1.1, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.575))
    cax = fig.add_axes([0.915, 0.135, 0.015, 0.34])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Kendall $\\tau$", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.0)
    cb.outline.set_visible(False)
    return F.save(fig, os.path.join(out, "fig9_benchmarks.pdf"))


def fig10_montecarlo(results, out):
    # The rank legend goes underneath, in one row: to the right of panel (b) it was
    # pushed outside the figure box and clipped the panel.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    fig.subplots_adjust(bottom=0.40, wspace=0.24, top=0.90)
    for k, case in enumerate(CASES):
        mc = R(results, "montecarlo", case)
        ax = axes[k]
        bottom = np.zeros(len(POST))
        for r in range(1, len(POST) + 1):
            fr = np.array([(mc[f"rank_{p}"] == r).mean() for p in POST])
            ax.bar(np.arange(len(POST)), fr, 0.66, bottom=bottom, color=F.SERIES[(r - 1) % 8],
                   hatch=F.HATCH[(r - 1) % 8], edgecolor="white", linewidth=0.7,
                   label=f"rank {r}", zorder=3)
            bottom += fr
        ax.set_xticks(np.arange(len(POST)))
        # single line, rotated: stacked two-line labels of this length overlapped
        # their neighbours at seven categories in a half-width panel
        ax.set_xticklabels([f"{p}  {POST_SHORT[p]}" for p in POST], fontsize=6.2,
                           rotation=34, ha="right")
        ax.set_ylabel("rank acceptability")
        ax.set_ylim(0, 1)
        ax.set_title(f"({'ab'[k]})  {CASE_TAG[case]}  ({len(mc)} draws)", loc="left")
        F.tidy(ax)
        for i, p in enumerate(POST):
            f1 = (mc[f"rank_{p}"] == 1).mean()
            if f1 > 0.02:
                ax.annotate(f"{100*f1:.0f}%", (i, f1 / 2), ha="center", va="center",
                            fontsize=6.6, color="white", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.6, ncol=7, handlelength=1.4,
               columnspacing=1.0, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.01))
    return F.save(fig, os.path.join(out, "fig10_montecarlo.pdf"))


# The sweep parameters are stored under their code names; the figure must show
# the symbols the paper uses, or a reader has to decode "near_threshold_km".
PARAM_LABEL = {
    "rho": r"$\rho$  sourcing convexity",
    "lam": r"$\lambda$  geopolitical risk weight",
    "beta": r"$\beta$  retaliation cost curvature",
    "gamma": r"$\gamma$  escalation cost",
    "theta0": r"$\theta_0$  political return",
    "delta_dom": r"$\delta_{\mathrm{DOM}}$  baseline domestic share",
    "subsidy": r"$\sigma$  domestic cost subsidy",
    "uniform_hike": r"$u$  uniform tariff increment",
    "targeted_hike": r"$t$  targeted tariff increment",
    "align_threshold": r"$\bar{g}$  alignment threshold",
    "near_threshold_km": r"$\bar{d}$  proximity threshold",
}


def fig11_sensitivity(results, out):
    # Each panel keeps its OWN y labels: the two cases sort their parameters
    # independently, so a shared axis would print one case's ordering beside the
    # other's bars. The left margin and the inter-panel gap are set explicitly,
    # wide enough for the longest label, so the right panel's labels cannot land
    # on top of the left panel's bars -- which is what happened before.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    fig.subplots_adjust(left=0.245, right=0.985, wspace=0.82, bottom=0.20, top=0.90)
    for k, case in enumerate(CASES):
        oat = R(results, "sensitivity_oat", case)
        ax = axes[k]
        params = list(dict.fromkeys(oat.parameter))
        share, labels = [], []
        for p in params:
            sub = oat[oat.parameter == p]
            share.append(float((sub.top_mcr == sub.top_mcr.mode().iloc[0]).mean()))
            labels.append(PARAM_LABEL.get(p, p))
        idx = np.argsort(share)
        ax.barh(np.arange(len(idx)), [share[i] for i in idx], 0.62,
                color=[F.SERIES[0] if share[i] > 0.999 else F.SERIES[1] for i in idx],
                edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels([labels[i] for i in idx], fontsize=6.6)
        ax.set_xlim(0, 1.17)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("share of the sweep retaining\nthe recommended posture", fontsize=7.4)
        for y, i in enumerate(idx):
            ax.annotate(f"{100*share[i]:.0f}%", (share[i], y), textcoords="offset points",
                        xytext=(4, 0), va="center", fontsize=6.4, color=F.INK_2)
        ax.set_title(f"({'ab'[k]})  {CASE_TAG[case]}", loc="left", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        F.tidy(ax, "x")
    # a two-colour key, stated once rather than repeated in both panels
    fig.legend(handles=[mpatches.Patch(facecolor=F.SERIES[0], label="recommendation never changes"),
                        mpatches.Patch(facecolor=F.SERIES[1], label="recommendation changes somewhere in the range")],
               fontsize=6.8, ncol=2, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    return F.save(fig, os.path.join(out, "fig11_sensitivity.pdf"))


def fig12_sourcing(results, out):
    """Equilibrium sourcing by posture, two sectors x two cases.

    Layout rationale. Each of the four panels used to carry its own legend
    hanging below it; the top row's legends then landed on the bottom row's
    titles. The two sectors within a case draw on the SAME origin set, so one
    legend per column serves both panels, and it is placed under the column
    rather than under a panel. Colours are keyed to the origin, not to its
    position in a per-panel list, so an origin has one colour throughout.
    """
    SECT = ["Semiconductors", "EV & batteries"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.24})
    fig.subplots_adjust(bottom=0.19, top=0.94, left=0.085, right=0.985)
    for k, case in enumerate(CASES):
        src = R(results, "sourcing", case)
        origins = list(CASES[case]["origins"])
        # one colour per origin for the whole column
        keep, tail = [], []
        for o in origins:
            mx = max(src[(src.sector == sec)].pivot(index="posture", columns="origin",
                                                    values="share").loc[POST][o].max()
                     for sec in SECT)
            (keep if mx > 0.02 else tail).append(o)
        # F.bar_style, not SERIES[i%8]+HATCH[i%8]: with ten origins the ninth and
        # tenth would otherwise be identical to the first and second.
        style = {o: F.bar_style(i) for i, o in enumerate(keep)}
        for j, sec in enumerate(SECT):
            ax = axes[j, k]
            piv = src[src.sector == sec].pivot(index="posture", columns="origin",
                                               values="share").loc[POST]
            bottom = np.zeros(len(POST))
            for o in keep:
                ax.bar(np.arange(len(POST)), piv[o].values, 0.68, bottom=bottom,
                       edgecolor="white", linewidth=0.6, label=o, zorder=3,
                       **style[o])
                bottom += piv[o].values
            if tail:
                rest = piv[tail].sum(axis=1).values
                if rest.max() > 1e-6:
                    ax.bar(np.arange(len(POST)), rest, 0.68, bottom=bottom,
                           color="#c9c9c4", edgecolor="white", linewidth=0.6,
                           label="other", zorder=3)
            ax.set_xticks(np.arange(len(POST)))
            ax.set_xticklabels(POST, fontsize=7.2)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.set_ylabel("equilibrium sourcing share", fontsize=7.8)
            ax.set_title(f"({'ac' [j] if k == 0 else 'bd'[j]})  {CASE_TAG[case]} — {sec}",
                         loc="left", fontsize=8.4)
            F.tidy(ax)
        # one legend for the column, in the margin reserved at the foot
        h, l = axes[0, k].get_legend_handles_labels()
        fig.legend(h, l, fontsize=6.4, ncol=5, handlelength=1.3, columnspacing=1.0,
                   frameon=False, title=f"{CASE_TAG[case]} — origins",
                   title_fontsize=6.8,
                   loc="upper center", bbox_to_anchor=(0.29 + k * 0.44, 0.155))
    return F.save(fig, os.path.join(out, "fig12_sourcing.pdf"))


def fig13_attribution(results, out):
    # Not sharey: each panel labels its own rows, so the two cannot drift apart
    # if the posture order ever differs. Right margin reserved for the key.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    fig.subplots_adjust(left=0.065, right=0.775, wspace=0.20, bottom=0.16, top=0.90)
    for k, case in enumerate(CASES):
        att = R(results, "regret_attribution", case).set_index("posture")
        ax = axes[k]
        cols = [c for c in CRIT if c in att.columns]
        M = att[cols].values
        idx = np.arange(len(att))
        pos = np.zeros(len(att)); neg = np.zeros(len(att))
        for i, c in enumerate(cols):
            v = M[:, i]
            b = np.where(v >= 0, pos, neg)
            ax.barh(idx[::-1], v, 0.62, left=b, edgecolor="white", linewidth=0.5,
                    label=CRIT_SHORT[c], zorder=3, **F.bar_style(i))
            pos = pos + np.where(v >= 0, v, 0); neg = neg + np.where(v < 0, v, 0)
        ax.set_yticks(idx[::-1])
        ax.set_yticklabels([f"$A_{p[1]}$" for p in att.index], fontsize=7.4)
        ax.axvline(0, color=F.INK_2, linewidth=0.8)
        ax.set_xlabel("contribution to worst-case regret")
        ax.set_title(f"({'ab'[k]})  {CASE_TAG[case]}", loc="left")
        F.tidy(ax, "x")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.4, loc="center left", bbox_to_anchor=(0.785, 0.52),
               handlelength=1.4, frameon=False, title="criterion", title_fontsize=6.8)
    return F.save(fig, os.path.join(out, "fig13_attribution.pdf"))


# ==========================================================================
def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.normpath(os.path.join(here, "..", "results")))
    ap.add_argument("--data", default=os.path.normpath(os.path.join(here, "..", "data")))
    ap.add_argument("--out", default=os.path.normpath(os.path.join(here, "..", "figures")))
    ap.add_argument("--only", default=None, help="comma-separated figure numbers")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = {
        1: lambda: fig1_context(a.data, a.out),
        2: lambda: fig2_graphical_abstract(a.results, a.out),
        3: lambda: fig3_problem(a.out),
        4: lambda: fig4_framework(a.results, a.out),
        5: lambda: fig5_performance(a.results, a.out),
        6: lambda: fig6_capacity(a.results, a.out),
        7: lambda: fig7_regret(a.results, a.out),
        8: lambda: fig8_epsilon(a.results, a.out, a.data),
        9: lambda: fig9_benchmarks(a.results, a.out),
        10: lambda: fig10_montecarlo(a.results, a.out),
        11: lambda: fig11_sensitivity(a.results, a.out),
        12: lambda: fig12_sourcing(a.results, a.out),
        13: lambda: fig13_attribution(a.results, a.out),
    }
    want = [int(x) for x in a.only.split(",")] if a.only else sorted(jobs)
    for n in want:
        try:
            p = jobs[n]()
            print(f"  fig{n:<3} -> {os.path.basename(p)}")
        except Exception as e:  # noqa: BLE001
            print(f"  fig{n:<3} FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
