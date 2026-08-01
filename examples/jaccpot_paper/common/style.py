"""One matplotlib style for every Jaccpot I figure.

Imported by the ``fig_*.ipynb`` notebooks only -- never by a ``bench/`` script.
Keeping matplotlib out of the benchmark layer is what stops a benchmark growing
a plot.

Colour
------
The categorical palette is Okabe-Ito, which is designed for dichromat
readability, in an order chosen by search rather than by eye:
:mod:`palette_check` scores every permutation and this one maximises the worst
*adjacent* pair separation, ΔE 18.0 under protan/deutan simulation (OKLab ×100)
against a floor of 8, with the worst normal-vision pair at 18.7 against a floor
of 15. ``tests/unit/test_paper_figure_style.py`` re-runs those checks, so the
palette cannot be edited back into something unreadable without a test failing.

Two documented caveats, both discharged by how the figures are drawn:

* Under an *all-pairs* comparison (the right gate for a scatter, where any two
  series can end up adjacent) the green/purple pair sits at ΔE 7.6, inside the
  6-8 floor band. That band is legal only with secondary encoding, so every
  series here also carries a distinct marker and, where there is room, a direct
  label. Darkening the palette to fix this was tried and rejected: it drops the
  all-pairs worst to 4.3, a hard fail.
* ``orange`` and ``sky`` land at 2.25:1 and 2.31:1 against white, under the 3:1
  target. A contrast warning obliges visible labels, which these figures always
  have (a legend is present for every multi-series panel). They are ordered late
  so a two- or three-series figure never reaches them.

Ordered series (expansion order, θ) use :data:`SEQUENTIAL`, a single-hue ramp --
ordered data must not be drawn in categorical hues, which imply identity rather
than magnitude.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "CATEGORICAL",
    "ENTITY",
    "MARKERS",
    "ONE_COL",
    "SEQUENTIAL",
    "TWO_COL",
    "apply",
    "entity_color",
    "figure",
    "finish",
    "save",
    "sequential_colors",
]

# Validated adjacent order -- see the module docstring. Do not reorder without
# rerunning `python -m examples.jaccpot_paper.common.palette_check`.
CATEGORICAL: tuple[str, ...] = (
    "#009E73",  # green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#56B4E9",  # sky
    "#E69F00",  # orange
    "#CC79A7",  # purple
)

# Single-hue ramp for ordered series. Monotone OKLCH lightness, min ΔL 0.098,
# light end 2.35:1 against white.
SEQUENTIAL: tuple[str, ...] = (
    "#7BAFD4",
    "#3F90C4",
    "#1E6FB0",
    "#0C4C8A",
    "#052A5E",
)

# Colour follows the entity, not its rank: a figure that drops a series must not
# repaint the survivors, and "jaccpot" must be the same colour in every figure.
ENTITY: dict[str, str] = {
    "jaccpot": CATEGORICAL[1],  # blue -- the method under study
    "direct": CATEGORICAL[2],  # vermillion -- the O(N^2) oracle
    "jaxfmm": CATEGORICAL[0],  # green -- the literature comparison
    "gpu": CATEGORICAL[1],
    "cpu": CATEGORICAL[4],
    # MAC arms (figure 03)
    "geometric": CATEGORICAL[1],
    "mass": CATEGORICAL[2],
    "mass_16b": CATEGORICAL[5],
    # Bases
    "real": CATEGORICAL[1],
    "solidfmm": CATEGORICAL[2],
    "complex": CATEGORICAL[5],
    # Gradient paths (figures 12/13)
    "forward": CATEGORICAL[1],
    "forward_backward": CATEGORICAL[2],
    "positions": CATEGORICAL[1],
    "masses": CATEGORICAL[2],
}

# Secondary encoding, so identity never rests on colour alone.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")

# Text ink -- values and labels wear these, never a series colour.
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#d9d9d6"

# Journal column widths in inches (MNRAS: 240pt / 504pt).
ONE_COL = 3.32
TWO_COL = 6.97


def apply(*, dark: bool = False) -> None:
    """Install the paper rcParams. Call once at the top of a figure notebook."""

    import matplotlib as mpl

    ink = "#f2f2f0" if dark else INK
    muted = "#9a9a97" if dark else INK_MUTED
    grid = "#3a3a38" if dark else GRID
    surface = "#1a1a19" if dark else "#ffffff"

    mpl.rcParams.update(
        {
            # Serif to sit alongside the manuscript body text. `usetex` is
            # deliberately off: it would make figure generation depend on a
            # working LaTeX install, and STIX already matches Times closely.
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            # Recessive frame: data first, chrome second.
            "axes.edgecolor": muted,
            "axes.linewidth": 0.6,
            "axes.labelcolor": ink,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": grid,
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            "text.color": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "xtick.labelcolor": ink,
            "ytick.labelcolor": ink,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            # Thin marks, generous markers.
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "lines.markeredgewidth": 0.8,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.2,
            "legend.labelspacing": 0.3,
            "figure.facecolor": surface,
            "axes.facecolor": surface,
            "savefig.facecolor": surface,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Keep text as text in the PDF so the manuscript can be searched and
            # the figure fonts match the body.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL)),
        }
    )


def figure(
    *,
    width: float = ONE_COL,
    height: Optional[float] = None,
    ncols: int = 1,
    nrows: int = 1,
    **kwargs: Any,
):
    """Return ``(fig, axes)`` at a journal column width.

    ``height`` defaults to a 0.72 aspect per row, which keeps a single-column
    panel close to the golden ratio without squashing a log-log decade.
    """

    import matplotlib.pyplot as plt

    if height is None:
        height = width * 0.72 * nrows / max(ncols, 1)
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def sequential_colors(n: int) -> list[str]:
    """Return ``n`` colours from :data:`SEQUENTIAL`, evenly spaced light->dark."""

    if n <= 0:
        return []
    if n == 1:
        return [SEQUENTIAL[len(SEQUENTIAL) // 2]]
    if n <= len(SEQUENTIAL):
        step = (len(SEQUENTIAL) - 1) / (n - 1)
        return [SEQUENTIAL[round(i * step)] for i in range(n)]
    # More series than ramp steps: interpolate in sRGB. Ordered data with this
    # many levels usually wants a colourbar instead of a legend.
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("jaccpot_seq", SEQUENTIAL)
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def entity_color(name: str, fallback_index: int = 0) -> str:
    """Return the fixed colour for a named entity (see :data:`ENTITY`)."""

    return ENTITY.get(name, CATEGORICAL[fallback_index % len(CATEGORICAL)])


def finish(ax, *, legend: bool = True, legend_kwargs: Optional[dict] = None) -> None:
    """Apply the shared final pass to one axes: recessive grid, tidy legend.

    A legend is present whenever the axes carries two or more labelled series,
    and suppressed for one (the title names it), per the accessibility pass.
    """

    handles, labels = ax.get_legend_handles_labels()
    if legend and len(labels) >= 2:
        ax.legend(**(legend_kwargs or {}))
    ax.grid(True, which="major")
    # A log axis with a decade or more of range gets minor gridlines, but at
    # half strength so they never compete with the data.
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log":
            ax.grid(True, which="minor", linewidth=0.3, alpha=0.6)
            break


def annotate_config(ax, text: str, *, loc: str = "lower left") -> None:
    """Stamp the measured config onto the panel, in muted ink.

    The annotation is built from the artifact's own ``config`` block, so it
    cannot drift from the data beside it.
    """

    positions = {
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=6.5,
        color=INK_MUTED,
    )


def footer(fig, text: str, *, y: float = -0.02) -> None:
    """Put the provenance line under the whole figure, in muted ink.

    Preferred over :func:`annotate_config` for the run configuration: inside an
    axes a long ``k=v`` string reliably collides with a y-axis label or a legend,
    and a caption that overlaps the data is worse than no caption.
    """

    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="top",
        fontsize=6.5,
        color=INK_MUTED,
    )


def save(fig, path: str, *, also_png: bool = False) -> str:
    """Write the figure to a single PDF (the manuscript's input format)."""

    import pathlib

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if also_png:
        fig.savefig(out.with_suffix(".png"))
    print(f"wrote {out}")
    return str(out)
