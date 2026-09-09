"""
figstyle.py -- shared figure styling for the SSR-MCDA manuscript.

Design rules applied throughout:

* Vector PDF output with embedded Type-42 fonts, as required by Elsevier.
* A categorical palette validated for colour-vision deficiency (worst adjacent
  CVD Delta E 9.1, worst adjacent normal-vision Delta E 19.6, OKLab x100).
* Because three slots sit below 3:1 contrast on a light surface, every figure
  carries *secondary encoding* -- hatch texture, marker shape or direct labels --
  so that no figure depends on hue alone. This also makes every figure legible
  when printed in greyscale.
* One measure per axis; no dual-axis charts anywhere.
* Recessive grid and axes; thin marks; selective direct labels.

Author: Saptadeep Biswas
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette (light surface #fcfcfb)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
HATCH = ["", "///", "...", "\\\\\\", "xxx", "---", "+++", "ooo"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))]

SURFACE = "#ffffff"
INK = "#111111"
INK_2 = "#4a4a4a"
INK_MUTED = "#7a7a7a"
GRID = "#d8d8d8"

# Sequential ramp (single hue, light to dark) for magnitude
SEQ = ["#eaf2fb", "#c5daf3", "#9cc0e9", "#6fa2dc", "#3f81cb", "#2160a8", "#123c6b"]
# Diverging pair (two hues, neutral grey midpoint) for signed quantities
DIV_NEG = "#eb6834"
DIV_MID = "#f0f0ee"
DIV_POS = "#2a78d6"


def apply_style() -> None:
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.dpi": 200,
    })


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ssr_div", [DIV_NEG, DIV_MID, DIV_POS], N=256)


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ssr_seq", SEQ, N=256)


def line_style(i: int) -> dict:
    """Distinguishable line style for series ``i``, for more than eight series.

    The palette, the marker set and the dash set each hold eight entries, so a
    plain ``i % 8`` on all three makes series i and series i+8 pixel-identical.
    Every second lap through the palette therefore switches to hollow markers,
    which also survives greyscale printing.
    """
    j, lap = i % 8, i // 8
    col = SERIES[j]
    return dict(color=col, marker=MARKERS[j], linestyle=LINESTYLES[j],
                markerfacecolor=col if lap % 2 == 0 else "#ffffff",
                markeredgecolor=col, markeredgewidth=0.8)


def bar_style(i: int) -> dict:
    """Distinguishable fill for series ``i``, for more than eight stacked series.

    Colour and hatch both cycle with period eight, so ``SERIES[i % 8]`` paired
    with ``HATCH[i % 8]`` makes series i and i+8 identical -- in the sourcing
    figure that put two different origins in the same blue with no texture.
    Advancing the hatch by one on each lap keeps every pair unique and keeps the
    figure readable in greyscale.
    """
    return dict(color=SERIES[i % 8], hatch=HATCH[(i + i // 8) % 8])


def tidy(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def label_bars(ax, bars, values, fmt="{:.2f}", dy=0.01, fontsize=7.2, color=INK):
    """Direct value labels -- the relief that the palette's contrast WARN requires."""
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height() + dy),
                    ha="center", va="bottom", fontsize=fontsize, color=color)


def panel_tag(ax, text, x=-0.11, y=1.06):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, path, **kw):
    fig.savefig(path, **kw)
    plt.close(fig)
    return path


# ==========================================================================
# Vector icon set for the two schematic figures
# ==========================================================================
# Every glyph is drawn from matplotlib primitives -- no icon font, no external
# asset, nothing that can fail to embed in the PDF. Each takes an axes whose
# data range is the unit square, a centre, a height in y-units, and `ar`, the
# figure's height/width ratio, which converts a y-unit into the x-unit that
# renders the same length on paper. Without `ar` every circle would print as
# an ellipse, because these axes are not square.
# --------------------------------------------------------------------------
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


class _Pen:
    """Draws in a local frame: x,y in [-0.5, 0.5] about the icon centre."""

    def __init__(self, ax, cx, cy, h, ar, color, lw, z):
        self.ax, self.cx, self.cy, self.h = ax, cx, cy, h
        self.w = h * ar                      # same physical length along x
        self.c, self.lw, self.z = color, lw, z

    def X(self, u):
        return self.cx + u * self.w

    def Y(self, v):
        return self.cy + v * self.h

    def line(self, pts, **kw):
        xs = [self.X(u) for u, _ in pts]
        ys = [self.Y(v) for _, v in pts]
        self.ax.add_line(Line2D(xs, ys, color=kw.pop("color", self.c),
                                linewidth=kw.pop("lw", self.lw),
                                solid_capstyle="round", zorder=self.z, **kw))

    def rect(self, u, v, du, dv, fill=False, **kw):
        self.ax.add_patch(Rectangle((self.X(u), self.Y(v)), du * self.w, dv * self.h,
                                    linewidth=kw.pop("lw", self.lw),
                                    edgecolor=kw.pop("color", self.c),
                                    facecolor=(kw.pop("fc", self.c) if fill else "none"),
                                    zorder=self.z, **kw))

    def poly(self, pts, fill=False, **kw):
        self.ax.add_patch(Polygon([(self.X(u), self.Y(v)) for u, v in pts],
                                  closed=True, linewidth=kw.pop("lw", self.lw),
                                  edgecolor=kw.pop("color", self.c),
                                  facecolor=(kw.pop("fc", self.c) if fill else "none"),
                                  joinstyle="round", zorder=self.z, **kw))

    def dot(self, u, v, r, fill=True, **kw):
        # r is in y-units of the local frame; scaled to a true circle via self.w
        self.ax.add_patch(Ellipse_circle(self.X(u), self.Y(v), r * self.h,
                                         self.w / self.h,
                                         edgecolor=kw.pop("color", self.c),
                                         facecolor=(kw.pop("fc", self.c) if fill else "none"),
                                         lw=kw.pop("lw", self.lw), z=self.z, ax=self.ax))

    def arc(self, u, v, r, t0, t1, **kw):
        from matplotlib.patches import Arc
        self.ax.add_patch(Arc((self.X(u), self.Y(v)), 2 * r * self.w, 2 * r * self.h,
                              theta1=t0, theta2=t1,
                              edgecolor=kw.pop("color", self.c),
                              linewidth=kw.pop("lw", self.lw), zorder=self.z))


def Ellipse_circle(x, y, r, xy_ratio, *, edgecolor, facecolor, lw, z, ax):
    """A circle of radius r y-units that still looks round on a non-square axes."""
    from matplotlib.patches import Ellipse
    return Ellipse((x, y), 2 * r * xy_ratio, 2 * r, edgecolor=edgecolor,
                   facecolor=facecolor, linewidth=lw, zorder=z)


def icon(ax, name, cx, cy, h, ar, color=INK_2, lw=0.9, z=6):
    """Draw icon `name` centred at (cx, cy) with height `h`. `ar` = fig_h/fig_w."""
    p = _Pen(ax, cx, cy, h, ar, color, lw, z)
    if name == "government":            # pediment on four columns
        p.poly([(-0.50, 0.12), (0.0, 0.46), (0.50, 0.12)], fill=True, fc=color, lw=0)
        for u in (-0.34, -0.12, 0.10, 0.32):
            p.rect(u, -0.30, 0.12, 0.40, fill=True, fc=color, lw=0)
        p.rect(-0.50, -0.46, 1.00, 0.14, fill=True, fc=color, lw=0)
    elif name == "factory":             # saw-tooth roof, chimney, two windows
        p.rect(0.20, 0.10, 0.13, 0.38, fill=True, fc=color, lw=0)
        p.poly([(-0.50, -0.40), (-0.50, 0.06), (-0.20, -0.14), (-0.20, 0.06),
                (0.10, -0.14), (0.10, 0.22), (0.46, 0.22), (0.46, -0.40)],
               fill=True, fc=color, lw=0)
        p.rect(-0.40, -0.30, 0.12, 0.16, fill=True, fc="#ffffff", lw=0)
        p.rect(-0.06, -0.30, 0.12, 0.16, fill=True, fc="#ffffff", lw=0)
    elif name == "ship":                # hull plus three stacked containers
        p.poly([(-0.50, -0.24), (0.50, -0.24), (0.34, -0.46), (-0.34, -0.46)],
               fill=True, fc=color, lw=0)
        p.rect(-0.34, -0.16, 0.26, 0.20, fill=True, fc=color, lw=0)
        p.rect(-0.02, -0.16, 0.26, 0.20, fill=True, fc=color, lw=0)
        p.rect(-0.18, 0.10, 0.26, 0.20, fill=True, fc=color, lw=0)
    elif name == "dice":                # deep uncertainty: a die, five pips
        p.rect(-0.42, -0.42, 0.84, 0.84, fill=False, lw=lw * 1.2)
        for u, v in ((-0.22, 0.22), (0.22, 0.22), (0.0, 0.0), (-0.22, -0.22), (0.22, -0.22)):
            p.dot(u, v, 0.075)
    elif name == "scales":              # balance: post, beam, two hanging pans
        p.line([(0.0, -0.40), (0.0, 0.38)], lw=lw * 1.3)
        p.line([(-0.42, 0.38), (0.42, 0.38)], lw=lw * 1.3)
        p.line([(-0.20, -0.40), (0.20, -0.40)], lw=lw * 1.3)
        for u in (-0.42, 0.42):
            p.line([(u, 0.38), (u, 0.10)], lw=lw * 0.9)
            p.arc(u, 0.10, 0.17, 180, 360, lw=lw * 1.1)
    elif name == "grid":                # a decision matrix
        # A filled block scored by white rules, not nine outlined cells: at the
        # sizes used here a 0.8 pt stroke is as wide as the cell it bounds, and
        # the outlined version printed as one solid square.
        p.rect(-0.45, -0.45, 0.90, 0.90, fill=True, fc=color, lw=0)
        for t in (-0.15, 0.15):
            p.line([(t, -0.45), (t, 0.45)], color="#ffffff", lw=max(1.1, lw * 1.6))
            p.line([(-0.45, t), (0.45, t)], color="#ffffff", lw=max(1.1, lw * 1.6))
    elif name == "network":             # criteria interacting
        pts = [(-0.38, 0.30), (0.38, 0.30), (0.0, -0.02), (-0.38, -0.36), (0.38, -0.36)]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                if (a, b) in ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)):
                    p.line([pts[a], pts[b]], lw=lw * 0.7, alpha=0.55)
        for u, v in pts:
            p.dot(u, v, 0.10)
    elif name == "target":              # a recommendation with a margin
        p.dot(0.0, 0.0, 0.44, fill=False, lw=lw * 1.1)
        p.dot(0.0, 0.0, 0.26, fill=False, lw=lw * 1.1)
        p.dot(0.0, 0.0, 0.10)
    elif name == "shield":              # certified robustness
        p.poly([(-0.36, 0.42), (0.36, 0.42), (0.36, -0.02), (0.0, -0.46),
                (-0.36, -0.02)], fill=False, lw=lw * 1.2)
        p.line([(-0.16, 0.08), (-0.03, -0.10), (0.20, 0.24)], lw=lw * 1.5)
    elif name == "gear":                # the computational engine
        for t in range(8):
            a = np.deg2rad(t * 45)
            p.line([(0.29 * np.cos(a), 0.29 * np.sin(a)),
                    (0.44 * np.cos(a), 0.44 * np.sin(a))], lw=lw * 2.4)
        p.dot(0.0, 0.0, 0.31, fill=False, lw=lw * 1.3)
        p.dot(0.0, 0.0, 0.13, fill=False, lw=lw * 1.3)
    elif name == "bars":                # an evaluation / score
        for i, hh in enumerate((0.30, 0.55, 0.85)):
            p.rect(-0.42 + i * 0.30, -0.42, 0.22, hh, fill=True, fc=color, lw=0)
    elif name == "ban":                 # why this cannot be done directly
        p.dot(0.0, 0.0, 0.42, fill=False, lw=lw * 1.4)
        p.line([(-0.28, 0.28), (0.28, -0.28)], lw=lw * 1.4)
    elif name == "check":               # validated
        p.dot(0.0, 0.0, 0.44, fill=False, lw=lw * 1.2)
        p.line([(-0.22, 0.02), (-0.05, -0.18), (0.24, 0.22)], lw=lw * 1.6)
    elif name == "flow":                # a response that closes back on itself
        # The local frame is physically isotropic (a step of 1 along u and along v
        # render the same length), so the tangent below is a true tangent.
        t0, t1, R = 40.0, 340.0, 0.36
        p.arc(0.0, 0.0, R, t0, t1, lw=lw * 1.6)
        a = np.deg2rad(t1)
        ex, ey = R * np.cos(a), R * np.sin(a)          # arc end
        tu, tv = -np.sin(a), np.cos(a)                 # tangent, direction of travel
        nu, nv = np.cos(a), np.sin(a)                  # outward normal
        p.poly([(ex + 0.17 * tu, ey + 0.17 * tv),
                (ex - 0.06 * tu + 0.10 * nu, ey - 0.06 * tv + 0.10 * nv),
                (ex - 0.06 * tu - 0.10 * nu, ey - 0.06 * tv - 0.10 * nv)],
               fill=True, fc=color, lw=0)
    elif name == "globe":
        p.dot(0.0, 0.0, 0.44, fill=False, lw=lw * 1.2)
        p.line([(-0.44, 0.0), (0.44, 0.0)], lw=lw * 0.9)
        p.line([(-0.30, 0.26), (0.30, 0.26)], lw=lw * 0.9)
        p.line([(-0.30, -0.26), (0.30, -0.26)], lw=lw * 0.9)
        p.arc(0.0, 0.0, 0.20, 0, 360, lw=lw * 0.9)
        p.line([(0.0, -0.44), (0.0, 0.44)], lw=lw * 0.9)
    else:
        raise KeyError(f"unknown icon {name!r}")


def badge(ax, text, cx, cy, r, ar, color, *, fs=8.0, z=7):
    """A filled roundel carrying a short label -- used for layer numbers."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((cx, cy), 2 * r * ar, 2 * r, facecolor=color,
                         edgecolor="none", zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="#ffffff", zorder=z + 1)
