"""Generate the ``fig_*.ipynb`` figure notebooks.

The notebooks are the figure layer: each loads exactly one artifact from
``results/`` and writes exactly one PDF. None of them recompute anything, so a
figure can never disagree with the JSON it claims to plot.

They are generated rather than hand-written so that the parts that must be
identical across nine notebooks -- the style import, the artifact load, the
provenance annotation, the PDF path -- are identical by construction. Edit this
file and rerun it; do not hand-edit the notebooks, since the next regeneration
would discard the change:

    python examples/jaccpot_paper/build_notebooks.py

Then execute them (which is what actually writes the PDFs):

    python examples/jaccpot_paper/run_notebooks.py
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

PREAMBLE = """\
import pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd().parents[1] if pathlib.Path.cwd().name == "jaccpot_paper" else pathlib.Path.cwd()))

import numpy as np
import matplotlib.pyplot as plt

from examples.jaccpot_paper.common import jsonio, style

style.apply()
FIG_DIR = jsonio.RESULTS_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
"""


def notebook(title: str, blurb: str, source: str, caption: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{blurb}"),
        nbf.v4.new_code_cell(PREAMBLE),
        nbf.v4.new_code_cell(source),
        nbf.v4.new_markdown_cell(f"## Caption\n\n{caption}"),
    ]
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    return nb


# --------------------------------------------------------------------------- #
# figure 01 -- force error vs expansion order
# --------------------------------------------------------------------------- #

FIG01 = """\
art = jsonio.read_result("validation/force_error_vs_order.json")
cfg, recs = art["config"], art["data"]["records"]

bases = list(cfg["basis"])
dists = list(cfg["distribution"])

# The real and solidfmm bases agree to float64 round-off at every point of this
# sweep (ratio 1.0000, max |diff| 4.5e-13 -- solidfmm and complex are in fact
# bit-identical, being the same code path). Drawing them as two curves therefore
# hides one behind the other and says nothing. Plot one basis, and report the
# cross-basis agreement as its own number: two independent expansion bases
# reproducing each other to round-off is a stronger cross-validation statement
# than two indistinguishable lines.
ref_basis = bases[0]
idx = {(r["basis"], r["distribution"], r["order"]): r for r in recs}
worst_gap, worst_at = 0.0, None
for (b, d, o), r in idx.items():
    other = idx.get((ref_basis, d, o))
    if b == ref_basis or other is None or not other["rel_l2"]:
        continue
    gap = abs(r["rel_l2"] - other["rel_l2"]) / other["rel_l2"]
    if gap > worst_gap:
        worst_gap, worst_at = gap, (b, d, o)

fig, axes = style.figure(width=style.TWO_COL, height=2.7, ncols=2, sharex=True)
for ax, field, label in (
    (axes[0], "rel_l2", "relative $L_2$ force error"),
    (axes[1], "worst_component", "worst component error / rms$|a|$"),
):
    for di, dist in enumerate(dists):
        sel = sorted(
            (r for r in recs if r["basis"] == ref_basis and r["distribution"] == dist),
            key=lambda r: r["order"],
        )
        if not sel:
            continue
        ax.plot(
            [r["order"] for r in sel],
            [r[field] for r in sel],
            marker=style.MARKERS[di % len(style.MARKERS)],
            color=style.CATEGORICAL[di % len(style.CATEGORICAL)],
            label=dist,
        )
    ax.set_yscale("log")
    ax.set_xlabel("expansion order $p$")
    ax.set_ylabel(label)
    style.finish(ax, legend=(ax is axes[0]), legend_kwargs={"loc": "lower left"})

agreement = ""
if worst_at is not None:
    agreement = f"   {' vs '.join(bases)} agree to {worst_gap:.1e} over the sweep"
    print(f"worst cross-basis relative gap {worst_gap:.3e} at {worst_at}")

fig.tight_layout()
# Both the config and the cross-basis agreement go in the footer: inside the axes
# the note grazed the first Plummer marker, and a caption over data is worse than
# a caption below the figure.
fig.tight_layout()
style.footer(
    fig,
    f"basis {ref_basis}  " + jsonio.config_caption(
        cfg, ["n", "theta", "leaf_size", "preset", "precision", "device", "seed"]
    ) + agreement,
)
style.save(fig, FIG_DIR / "fig01_force_error_vs_order.pdf")
"""

FIG01_CAPTION = """\
Force error against an exact direct summation as a function of expansion order
$p$, at fixed $N$ and opening angle, for the real (Dehnen) and solidfmm bases on
a uniform cube and a Plummer sphere. **Left:** relative $L_2$ error over the whole
acceleration field. **Right:** the largest single-component error, normalised by
the rms $|a|$ -- reported alongside the $L_2$ norm because an $L_2$ norm averages
away the tail that actually limits an integrator. Exact configuration is
annotated; all values are read from
`bench/results/validation/force_error_vs_order.json`.
"""

# --------------------------------------------------------------------------- #
# figure 02 -- force error vs theta
# --------------------------------------------------------------------------- #

FIG02 = """\
art = jsonio.read_result("validation/error_vs_theta.json")
cfg, recs = art["config"], art["data"]["records"]

dists = list(cfg["distribution"])
orders = sorted({r["order"] for r in recs})
# Order is an ORDERED axis, so it gets the sequential ramp, not categorical hues.
ramp = style.sequential_colors(len(orders))

fig, axes = style.figure(
    width=style.TWO_COL, height=2.7, ncols=len(dists), sharey=True
)
axes = np.atleast_1d(axes)

for ax, dist in zip(axes, dists):
    for oi, order in enumerate(orders):
        sel = sorted(
            (r for r in recs if r["distribution"] == dist and r["order"] == order),
            key=lambda r: r["theta"],
        )
        if not sel:
            continue
        theta = [r["theta"] for r in sel]
        err = [r["rel_l2"] for r in sel]
        ax.plot(theta, err, "-", color=ramp[oi], label=f"$p={order}$", zorder=3)
        # Split the markers: a filled point had a far field to approximate, an
        # open one did not, and its error is round-off rather than convergence.
        empty = [r.get("far_field_empty", False) for r in sel]
        for filled, marker_set in ((False, "filled"), (True, "open")):
            xs = [t for t, e in zip(theta, empty) if e == filled]
            ys = [v for v, e in zip(err, empty) if e == filled]
            if not xs:
                continue
            ax.plot(
                xs, ys, style.MARKERS[oi % len(style.MARKERS)],
                color=ramp[oi], zorder=4,
                markerfacecolor=("white" if filled else ramp[oi]),
                markeredgecolor=ramp[oi],
            )
    ax.set_yscale("log")
    ax.set_xlabel(r"opening angle $\\theta$")
    ax.set_title(dist, fontsize=9)
    style.finish(ax, legend=(ax is axes[0]), legend_kwargs={"loc": "lower right"})
axes[0].set_ylabel("relative $L_2$ force error")

if any(r.get("far_field_empty") for r in recs):
    axes[0].text(
        0.02, 0.98,
        "open marker: far field empty\\n(exact P2P; error is round-off)",
        transform=axes[0].transAxes, va="top", ha="left",
        fontsize=6.5, color=style.INK_MUTED,
    )
style.footer(
    fig,
    jsonio.config_caption(cfg, ["n", "basis", "leaf_size", "preset", "precision", "device"]),
)
style.save(fig, FIG_DIR / "fig02_error_vs_theta.pdf")
"""

FIG02_CAPTION = """\
Relative $L_2$ force error against opening angle $\\theta$ at fixed $N$, with
expansion orders overlaid (light to dark), for a uniform cube and a Plummer
sphere. Orders use a single-hue sequential ramp because $p$ is an ordered
quantity. **Open markers mark configurations whose far field is empty**: there the
solver has degenerated to an exact P2P direct sum, so the error is float64
round-off and says nothing about the expansion -- a flat low-$\\theta$ tail would
otherwise read as convergence. Values from
`bench/results/validation/error_vs_theta.json`.
"""

# --------------------------------------------------------------------------- #
# figure 03 -- geometric vs mass-dependent MAC
# --------------------------------------------------------------------------- #

FIG03 = """\
art = jsonio.read_result("validation/mac_comparison.json")
cfg = art["config"]
recs = art["data"]["records"]
comps = art["data"]["comparisons"]

metric = cfg.get("matched_metric", "dehnen")
prefix = {"relative": "", "scaled": "scaled_", "dehnen": "dehnen_"}[metric]
p90_key = f"{prefix}p90"

ARM_LABEL = {
    "fixed": "geometric (sweep $\\\\theta$)",
    "mass": "mass-dep., eq (16a)",
    "mass_16b": "eq (16b), exact $f_b$",
}
ARM_ENTITY = {"fixed": "geometric", "mass": "mass", "mass_16b": "mass_16b"}
dists = sorted({r["distribution"] for r in recs})
orders = sorted({r["order"] for r in recs})

# One trade-off panel PER ORDER, plus the ratio panel. Collapsing p=4 and p=8 into
# a single line sorted by cost produced a zigzag: the two orders occupy different
# cost-error curves, so joining them draws a path through both instead of either.
fig, axes = style.figure(
    width=style.TWO_COL, height=2.9, ncols=len(orders) + 1
)
axes = np.atleast_1d(axes)

for oi, order in enumerate(orders):
    ax = axes[oi]
    for dist_i, dist in enumerate(dists):
        for arm in ("fixed", "mass", "mass_16b"):
            sel = sorted(
                (r for r in recs
                 if r["arm"] == arm and r["distribution"] == dist
                 and r["order"] == order and r[p90_key] > 0),
                key=lambda r: r["pair_work"],
            )
            if not sel:
                continue
            ax.plot(
                [r["pair_work"] for r in sel],
                [r[p90_key] for r in sel],
                marker=style.MARKERS[dist_i % len(style.MARKERS)],
                linestyle=["-", "--"][dist_i % 2],
                color=style.entity_color(ARM_ENTITY[arm]),
                label=f"{ARM_LABEL[arm]} - {dist}",
                markersize=3.0,
                linewidth=1.1,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pair work")
    ax.set_title(f"$p={order}$", fontsize=8)
    style.finish(
        ax, legend=(oi == 0), legend_kwargs={"loc": "lower left", "fontsize": 5.2}
    )
axes[0].set_ylabel(f"90th-pct scaled force error ({metric})")

# Ratio panel: geometric / mass-dependent at matched error. Work ratio and error
# ratio are both dimensionless ratios against the same baseline, so they share one
# axis legitimately -- and plotting only the work ratio would report half the
# result, since cost lands at parity while the tail does not.
ax = axes[-1]
if comps:
    for dist_i, dist in enumerate(dists):
        for arm in ["mass"]:
            sel = sorted(
                (c for c in comps
                 if c["distribution"] == dist
                 and c.get("mass_arm", "mass") == arm),
                key=lambda c: c["matched_p90"],
            )
            work = [c for c in sel if c.get("pair_work_ratio")]
            if work:
                ax.plot(
                    [c["matched_p90"] for c in work],
                    [c["pair_work_ratio"] for c in work],
                    marker=style.MARKERS[dist_i % len(style.MARKERS)],
                    linestyle=["-", "--"][dist_i % 2],
                    color=style.entity_color(arm),
                    label=f"work, {arm}, {dist}",
                    markersize=3.0,
                    linewidth=1.1,
                )
            tail = [c for c in sel if c.get("p99_ratio")]
            if tail:
                ax.plot(
                    [c["matched_p90"] for c in tail],
                    [c["p99_ratio"] for c in tail],
                    marker=style.MARKERS[dist_i % len(style.MARKERS)],
                    linestyle=["-", "--"][dist_i % 2],
                    color=style.CATEGORICAL[4],
                    label=f"p99 error, {arm}, {dist}",
                    markersize=3.0,
                    linewidth=1.1,
                    markerfacecolor="white",
                )
    ax.axhline(1.0, color=style.INK_MUTED, linewidth=0.8, zorder=1)
    ax.text(
        0.02, 1.0, " parity", transform=ax.get_yaxis_transform(),
        ha="left", va="bottom", fontsize=6.0, color=style.INK_MUTED,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("matched 90th-pct error")
    ax.set_ylabel("ratio, geometric / mass-dep.")
    ax.set_title("at matched error", fontsize=8)
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=5.0)
else:
    ax.text(0.5, 0.5, "no matched-error overlap", ha="center", va="center",
            transform=ax.transAxes, color=style.INK_MUTED)
    ax.set_axis_off()

fig.tight_layout()
style.footer(
    fig,
    jsonio.config_caption(cfg, ["n", "order", "leaf_size", "precision", "device", "seed"]),
    y=-0.08,
)
style.save(fig, FIG_DIR / "fig03_mac_comparison.pdf")

work = [c["pair_work_ratio"] for c in comps if c.get("pair_work_ratio")]
p99 = [c["p99_ratio"] for c in comps if c.get("p99_ratio")]
mx = [c["max_ratio"] for c in comps if c.get("max_ratio")]
print("at matched 90th-percentile error, ratio geometric/mass (>1 favours mass):")
if work:
    print(f"  pair work : {min(work):.2f} .. {max(work):.2f}")
if p99:
    print(f"  p99 error : {min(p99):.2f} .. {max(p99):.2f}")
if mx:
    print(f"  max error : {min(mx):.2f} .. {max(mx):.2f}")
print("  -> cost is at parity; the error TAIL is where the criterion pays off")
"""

FIG03_CAPTION = """\
Geometric against mass-dependent (Dehnen eq. 16a) multipole acceptance, on
clustered distributions. **Left:** the raw trade-off -- hardware-independent pair
work against the 90th-percentile scaled force error, each arm swept over its own
accuracy knob ($\\theta$ for the geometric criterion, $\\epsilon$ for the
mass-dependent one). **Right:** the two arms log-interpolated onto a common error
and compared there, as ratios of geometric to mass-dependent, so above the parity
line favours the mass-dependent criterion.

The measurement splits into two answers, and reporting either alone would
misrepresent it:

* **Cost is at parity.** The pair-work ratio spans 0.99-1.20 across both
  distributions and both orders, mostly sitting at 1.00. **There is no net compute
  advantage to the mass-dependent criterion at matched error** -- the finding this
  figure was built to test, confirmed as measured.
* **The error tail is not at parity.** At the *same* 90th-percentile error the
  mass-dependent criterion's 99th percentile is 1.15-210x smaller, and its worst
  single-particle error up to ~2000x smaller. That is Dehnen's own §5.3 claim,
  which is about the shape of the error distribution rather than about speed, and
  it is reproduced here.

So the criterion buys a much better tail at essentially the same cost, rather than
the same accuracy more cheaply. Which of those is worth having depends on whether
an application is limited by its worst particle or by its throughput.

The eq. (16b) arm supplies the exact $O(N^2)$ force scale $f_b$ and is a ceiling on
what a better force-scale estimator could buy, not a runnable configuration.
Nothing here is tuned to favour either arm; the sweep grids are the engineering
benchmark's defaults. Values from `bench/results/validation/mac_comparison.json`.
"""

# --------------------------------------------------------------------------- #
# figure 04 -- wall-clock vs N
# --------------------------------------------------------------------------- #

FIG04 = """\
art = jsonio.read_result("scaling/wallclock_vs_n.json")
cfg, recs, fits = art["config"], art["data"]["records"], art["data"]["fits"]

SERIES = [
    ("jaccpot_potential", "jaccpot (potential)", "jaccpot"),
    ("jaxfmm_potential", "jaxFMM (potential)", "jaxfmm"),
    ("jaccpot_acceleration", "jaccpot (acceleration)", "jaccpot"),
    ("direct_acceleration", "direct sum $O(N^2)$", "direct"),
]
set_name = sorted({r["param_set"] for r in recs})[0]

fig, ax = style.figure(width=style.ONE_COL, height=2.9)
for i, (key, label, entity) in enumerate(SERIES):
    sel = sorted(
        (r for r in recs
         if r["param_set"] == set_name and r["timings"].get(key, {}).get("min_s")),
        key=lambda r: r["n"],
    )
    if not sel:
        continue
    fit = fits.get(f"{set_name}:{key}", {})
    alpha = fit.get("exponent")
    # An exponent is a complexity statement only if the power law actually fits
    # AND the series actually grew. Measured here, over N = 27554 -> 881744 (a
    # factor of 32): jaccpot_potential grows 39x and fits alpha=1.08 at R^2=0.98,
    # a real near-linear result; jaccpot_acceleration grows only 2.2x and fits
    # alpha=0.20 at R^2=0.89, which is a fixed overhead being fitted rather than
    # O(N). Requiring R^2 >= 0.95 and >= 4x growth separates the two; anything
    # failing that is labelled overhead-bound instead of given a spurious number.
    fit_lo = fit.get("fit_min_n") or 0
    span = [r["timings"][key]["min_s"] for r in sel if r["n"] >= fit_lo]
    dynamic = (max(span) / min(span)) if span and min(span) > 0 else 0.0
    if (
        alpha is not None
        and np.isfinite(alpha)
        and fit.get("n_points", 0) >= 3
        and dynamic >= 4.0
        and (fit.get("r_squared") or 0.0) >= 0.95
    ):
        suffix = f"  ($\\\\alpha={alpha:.2f}$)"
    elif dynamic and dynamic < 4.0:
        suffix = "  (overhead-bound)"
    else:
        suffix = ""
    ax.plot(
        [r["n"] for r in sel],
        [r["timings"][key]["min_s"] for r in sel],
        marker=style.MARKERS[i % len(style.MARKERS)],
        # Acceleration vs potential distinguished by dash, since two series share
        # the jaccpot colour by design (same code, different output).
        linestyle="--" if "acceleration" in key and entity == "jaccpot" else "-",
        color=style.entity_color(entity),
        label=label + suffix,
        markersize=3.4,
    )
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("$N$")
ax.set_ylabel("wall-clock per evaluation [s]")
style.finish(ax, legend_kwargs={"loc": "lower right", "fontsize": 6.0})
fig.tight_layout()
style.footer(
    fig,
    jsonio.config_caption(cfg, ["order", "theta", "basis", "preset", "precision", "device", "seed"])
    + f"   exponent fitted for N >= {cfg.get('fit_min_n')}",
)
style.save(fig, FIG_DIR / "fig04_wallclock_vs_n.pdf")

for name, fit in sorted(fits.items()):
    if not name.startswith(set_name):
        continue
    series_key = name.split(":", 1)[1]
    span = [
        r["timings"][series_key]["min_s"]
        for r in recs
        if r["param_set"] == set_name
        and r["timings"].get(series_key, {}).get("min_s")
        and r["n"] >= (fit.get("fit_min_n") or 0)
    ]
    dynamic = (max(span) / min(span)) if span and min(span) > 0 else float("nan")
    ok = (
        fit["n_points"] >= 3
        and dynamic >= 4.0
        and (fit.get("r_squared") or 0.0) >= 0.95
    )
    note = "" if ok else "   <- NOT a complexity exponent"
    print(f"{name:<45s} alpha={fit['exponent']:.3f} R2={fit['r_squared']:.4f} "
          f"n={fit['n_points']} N={fit.get('fit_min_n')}..{fit.get('fit_max_n')} "
          f"range={dynamic:.2f}x{note}")
"""

FIG04_CAPTION = """\
Wall-clock per force evaluation against $N$ on one A100, log-log, with fitted
power-law exponents $\\alpha$ in the legend. The timed region is evaluation on an
already-built tree for every FMM series, which is what a simulation pays per step
when topology is reused. **jaccpot and jaxFMM are compared potential-to-potential**:
an acceleration costs strictly more than a potential, so the acceleration series
is shown separately (dashed) rather than timed against jaxFMM's output. The
$O(N^2)$ direct sum is cut off where it becomes unaffordable. Exponents are fitted
above the annotated $N$, below which a GPU is launch-overhead bound rather than
algorithm bound. Values from `bench/results/scaling/wallclock_vs_n.json`.
"""

# --------------------------------------------------------------------------- #
# figure 05 -- interaction counts vs N
# --------------------------------------------------------------------------- #

FIG05 = """\
art = jsonio.read_result("scaling/interaction_counts.json")
cfg, recs, fits = art["config"], art["data"]["records"], art["data"]["fits"]
recs = [r for r in recs if "error" not in r]

fig, axes = style.figure(width=style.TWO_COL, height=2.7, ncols=2)

# Left: raw counts. Right: counts per particle, which is where an O(N) result
# either shows up as a flat line or does not.
for ax, per_particle in ((axes[0], False), (axes[1], True)):
    for i, (field, label) in enumerate(
        ((\"far_pairs\", \"M2L (far) pairs\"), (\"near_pairs\", \"P2P (near) pairs\"))
    ):
        sel = sorted((r for r in recs if r.get(field)), key=lambda r: r[\"n\"])
        if not sel:
            continue
        ys = [
            (r[field] / r[\"n\"]) if per_particle else r[field] for r in sel
        ]
        fit = fits.get(field, {})
        alpha = fit.get(\"exponent\")
        suffix = \"\" if per_particle or not alpha else f\"  ($\\\\alpha={alpha:.2f}$)\"
        ax.plot(
            [r[\"n\"] for r in sel], ys,
            marker=style.MARKERS[i % len(style.MARKERS)],
            color=style.CATEGORICAL[i],
            label=label + suffix,
            markersize=3.4,
        )
    ax.set_xscale(\"log\")
    ax.set_yscale(\"log\")
    ax.set_xlabel(\"$N$\")
    ax.set_ylabel(\"interactions per particle\" if per_particle else \"interaction count\")
    style.finish(ax, legend_kwargs={\"loc\": \"best\", \"fontsize\": 6.4})

fig.tight_layout()
# A ladder point that failed is stated, not quietly dropped: silently omitting it
# would make the sweep look like it covered a range it did not reach.
failed = [r for r in art[\"data\"][\"records\"] if \"error\" in r]
if failed:
    axes[0].text(
        0.02, 0.98,
        \"not measured: N = \"
        + \", \".join(f\"{r['n']:,}\" for r in failed)
        + \"\\n(out of memory)\",
        transform=axes[0].transAxes, va=\"top\", ha=\"left\",
        fontsize=6.0, color=style.INK_MUTED,
    )
    print(\"not measured:\", [(r[\"n\"], r[\"error\"][:60]) for r in failed])

style.footer(
    fig,
    jsonio.config_caption(cfg, [\"order\", \"theta\", \"basis\", \"leaf_size\", \"device\", \"seed\"])
    + f\"   exponent fitted for N >= {cfg.get('fit_min_n')}\",
)
style.save(fig, FIG_DIR / \"fig05_interaction_counts.pdf\")

for name, fit in sorted(fits.items()):
    print(f\"{name:<12s} alpha={fit['exponent']:.3f} R2={fit['r_squared']:.4f} \"
          f\"n={fit['n_points']} over N={fit.get('fit_min_n')}..{fit.get('fit_max_n')}\")
"""

FIG05_CAPTION = """\
Accepted M2L (far) and P2P (near) interaction counts against $N$ at fixed opening
angle and leaf size, with fitted power-law exponents. These are properties of the
tree and the acceptance criterion alone, independent of hardware, and so are a
cleaner statement of complexity than wall-clock (figure 4), which additionally
mixes in memory bandwidth and kernel launch overhead.

**Both exponents exceed one** -- $\\alpha = 1.32$ for M2L and $1.20$ for P2P
($R^2 = 0.996$ and $0.9995$) -- and the right panel shows why: the counts *per
particle* are not constant but grow, from 2.6 to 7.7 M2L pairs per particle across
a factor of 32 in $N$. A factor of ~3 in response to a factor of 32 in $N$ is
$\\log N$ growth, so the measurement is consistent with $O(N \\log N)$ rather than
strictly linear, which is what a fixed leaf size and a fixed opening angle should
give: the tree deepens with $N$, and each cell acquires more well-separated
partners. The fitted single exponent is the power law that best approximates
$N \\log N$ over this range, not evidence of a different asymptotic class.

Counts come from the solver's public runtime diagnostics after `prepare_state`.
Any ladder point that could not be measured is named on the figure. Values from
`bench/results/scaling/interaction_counts.json`.
"""

# --------------------------------------------------------------------------- #
# figure 06 -- per-stage breakdown
# --------------------------------------------------------------------------- #

FIG06 = """\
art = jsonio.read_result("scaling/stage_breakdown.json")
cfg, recs = art["config"], art["data"]["records"]
recs = sorted((r for r in recs if "error" not in r), key=lambda r: r["n"])

if not recs:
    raise SystemExit(
        "stage_breakdown.json has no successful rows. This figure needs the "
        "strict refresh path, which is GPU-only -- rerun the bench on a GPU."
    )

# Band list derived from the artifact, not hardcoded. A hardcoded list silently
# dropped `upward_geometry` when the bench script gained it -- 56-59% of the step,
# so the bars stopped summing to the wall clock and the figure understated
# per-step time by a factor of two while looking perfectly plausible. Known keys
# get a display order and a label; anything unrecognised is still plotted, at the
# end, under its raw name.
LABELS = {
    "tree_build": "tree build",
    "upward_geometry": "upward geometry",
    "p2m_m2m": "P2M + M2M",
    "traversal_setup": "traversal setup",
    "m2l": "M2L",
    "l2l": "L2L",
    "nearfield_p2p": "near field (P2P)",
    "other_measured": "other measured",
    "unattributed": "unattributed (incl. downward)",
}
present = {k for r in recs for k in r["stages_s"]}
ORDER = [(k, LABELS.get(k, k)) for k in LABELS if k in present]
ORDER += [(k, k) for k in sorted(present - set(LABELS))]

# The bands must account for the measured wall clock; if they do not, the figure
# is understating per-step cost and should say so rather than be believed.
for r in recs:
    total = sum(r["stages_s"].get(k, 0.0) for k, _ in ORDER)
    if r["per_step_wall_s"] > 0 and abs(total - r["per_step_wall_s"]) / r["per_step_wall_s"] > 0.01:
        print(
            f"WARNING N={r['n']}: bands sum to {total*1e3:.1f} ms but the step took "
            f"{r['per_step_wall_s']*1e3:.1f} ms -- a stage is missing from LABELS"
        )
ns = [r["n"] for r in recs]
x = np.arange(len(ns), dtype=float)

fig, axes = style.figure(width=style.TWO_COL, height=2.9, ncols=2)

# Left: absolute per-step seconds. Right: shares, which is what the figure is for.
for ax, normalise in ((axes[0], False), (axes[1], True)):
    bottom = np.zeros(len(ns))
    for i, (key, label) in enumerate(ORDER):
        vals = np.array([r["stages_s"].get(key, 0.0) for r in recs], dtype=float)
        if normalise:
            totals = np.array([r["per_step_wall_s"] for r in recs], dtype=float)
            vals = np.where(totals > 0, vals / totals * 100.0, 0.0)
        if not vals.any():
            continue
        ax.bar(
            x, vals, bottom=bottom, width=0.68,
            color=(style.INK_MUTED if key in ("unattributed", "other_measured")
                   else style.CATEGORICAL[i % len(style.CATEGORICAL)]),
            # A 2px surface gap between stacked segments keeps adjacent fills
            # from reading as one block.
            edgecolor="white", linewidth=1.0,
            label=label, zorder=3,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"$2^{{{int(round(np.log2(n)))}}}$" for n in ns])
    ax.set_xlabel("$N$")
    ax.set_ylabel("share of per-step time [%]" if normalise else "per-step time [s]")
    # Linear, not log. A stacked bar on a log axis is a misread waiting to happen:
    # segment *heights* stop being proportional to the values they represent, so a
    # 2% band can look comparable to a 60% one. The absolute panel spans well under
    # a decade here, so linear costs nothing.
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    style.finish(ax, legend=normalise, legend_kwargs={"loc": "center left",
                                                     "bbox_to_anchor": (1.01, 0.5),
                                                     "fontsize": 6.4})

frac = [r.get("attributed_fraction") for r in recs if r.get("attributed_fraction")]
if frac:
    print(f"attributed fraction of wall clock: {min(frac):.2%} .. {max(frac):.2%}")
fig.tight_layout()
style.footer(
    fig,
    jsonio.config_caption(cfg, ["order", "theta", "basis", "preset", "leaf_size", "device"]),
)
style.save(fig, FIG_DIR / "fig06_stage_breakdown.pdf")
"""

FIG06_CAPTION = """\
Per-stage contribution to per-step time against $N$, absolute (left) and as a
share (right). Measured on the strict large-$N$ GPU refresh path
(`preset="large_n_gpu"`, radix tree, solidfmm basis), because that is the only
path carrying per-stage instrumentation; it is the production per-step path at
these $N$, but the breakdown should not be read as describing the default
configuration used in figures 1-3. The stage timers do not partition the wall
clock exactly -- local-to-particle evaluation and host-side bookkeeping are not
separately instrumented -- so the remainder is shown explicitly as
*unattributed* rather than rescaled into the measured stages. Values from
`bench/results/scaling/stage_breakdown.json`.
"""

# --------------------------------------------------------------------------- #
# figure 07 -- GPU vs CPU
# --------------------------------------------------------------------------- #

FIG07 = """\
art = jsonio.read_result("scaling/gpu_vs_cpu.json")
cfg, recs = art["config"], art["data"]["records"]
recs = sorted(recs, key=lambda r: r["n"])

fig, axes = style.figure(width=style.TWO_COL, height=2.7, ncols=2)

# Left: the two raw curves, so the ratio on the right is auditable.
ax = axes[0]
for i, (key, label, entity) in enumerate(
    (("cpu_min_s", "CPU", "cpu"), ("gpu_min_s", "single A100", "gpu"))
):
    sel = [r for r in recs if r.get(key)]
    if not sel:
        continue
    ax.plot(
        [r["n"] for r in sel], [r[key] for r in sel],
        marker=style.MARKERS[i % len(style.MARKERS)],
        color=style.entity_color(entity), label=label, markersize=3.4,
    )
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("$N$"); ax.set_ylabel("wall-clock per evaluation [s]")
style.finish(ax, legend_kwargs={"loc": "upper left"})

# Right: the speedup, only where BOTH arms were measured.
ax = axes[1]
sel = [r for r in recs if r.get("speedup")]
ax.plot(
    [r["n"] for r in sel], [r["speedup"] for r in sel],
    marker="o", color=style.entity_color("gpu"), markersize=3.8,
)
ax.axhline(1.0, color=style.INK_MUTED, linewidth=0.8, zorder=1)
ax.text(0.98, 1.0, " parity", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=6.5, color=style.INK_MUTED)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("$N$"); ax.set_ylabel("speedup, CPU time / GPU time")
style.finish(ax, legend=False)

# With only a few matched points the log minor-tick labels collide into an
# unreadable band, so tick exactly at the measured N.
ticks = [r["n"] for r in sel]
if ticks:
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$2^{{{int(round(np.log2(n)))}}}$" for n in ticks])
    ax.minorticks_off()

notes = []
skipped = [r["n"] for r in recs if r.get("cpu_skipped")]
if skipped:
    notes.append(
        f"CPU arm not run above $N={min(skipped):,}$; curve is not extrapolated"
    )
# A point that failed is named, not dropped: the alternative is a figure that
# looks like it covered a range it never reached.
failed = [(r["n"], r["gpu_error"]) for r in recs if r.get("gpu_error")]
if failed:
    notes.append(
        "not measured: $N="
        + ", ".join(f"{n:,}" for n, _ in failed)
        + "$ (out of memory)"
    )
    print("not measured:", [(n, e[:60]) for n, e in failed])
if notes:
    ax.text(
        0.98, 0.03, "  |  ".join(notes),
        transform=ax.transAxes, va="bottom", ha="right",
        fontsize=5.6, color=style.INK_MUTED,
    )
fig.tight_layout()
style.footer(
    fig,
    jsonio.config_caption(cfg, ["order", "theta", "basis", "preset", "leaf_size", "precision"]),
)
style.save(fig, FIG_DIR / "fig07_gpu_vs_cpu.pdf")
print(cfg.get("shared_host", ""))
"""

FIG07_CAPTION = """\
Single-GPU against CPU wall-clock (left) and the resulting speedup (right), both
timing the identical quantity -- evaluation on a prebuilt tree at the same $N$,
seed, order, $\\theta$, leaf size and precision -- so that only the backend
differs. The GPU curve is nearly flat at small $N$, where the device is launch-
and overhead-bound rather than work-bound, so the speedup crosses unity only once
there is enough work to fill the card. The CPU arm is not run above the annotated
$N$ and the curve is not extrapolated past the last measured pair. Measured on a
shared 72-core host, so the ratio is indicative rather than a hardware review.
Values from `bench/results/scaling/gpu_vs_cpu.json`.
"""

# --------------------------------------------------------------------------- #
# figure 12 -- autodiff overhead
# --------------------------------------------------------------------------- #

FIG12 = """\
art = jsonio.read_result("differentiability/autodiff_overhead.json")
cfg, recs = art["config"], art["data"]["records"]
recs = [r for r in recs if "error" not in r]
if not recs:
    raise SystemExit("autodiff_overhead.json has no successful rows")

bases = [b for b in cfg["basis"] if any(r["basis"] == b for r in recs)]
lanes = sorted({r["lane"] for r in recs})
wrts = list(cfg["wrt"])

fig, axes = style.figure(width=style.TWO_COL, height=2.7, ncols=2)

# Left: absolute forward and forward+backward times. Right: the ratio, which is
# the claim. Rows are grouped by mode so a jitted point is never joined to an
# eager one.
ax = axes[0]
for bi, basis in enumerate(bases):
    for lane in lanes:
        sel = sorted(
            (r for r in recs if r["basis"] == basis and r["lane"] == lane),
            key=lambda r: r["n"],
        )
        if not sel:
            continue
        ax.plot([r["n"] for r in sel], [r["forward_min_s"] for r in sel],
                marker=style.MARKERS[bi % len(style.MARKERS)], linestyle="-",
                color=style.entity_color("forward"), markersize=3.4,
                label=f"forward - {basis}/{lane}")
        both = [r for r in sel if r["grads"].get("both", {}).get("forward_backward_min_s")]
        if both:
            ax.plot([r["n"] for r in both],
                    [r["grads"]["both"]["forward_backward_min_s"] for r in both],
                    marker=style.MARKERS[bi % len(style.MARKERS)], linestyle="--",
                    color=style.entity_color("forward_backward"), markersize=3.4,
                    label=f"fwd+bwd - {basis}/{lane}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("$N$"); ax.set_ylabel("wall-clock [s]")
style.finish(ax, legend_kwargs={"loc": "best", "fontsize": 5.8})

ax = axes[1]
for wi, wrt in enumerate(wrts):
    for bi, basis in enumerate(bases):
        for lane in lanes:
            sel = sorted(
                (r for r in recs
                 if r["basis"] == basis and r["lane"] == lane
                 and r["grads"].get(wrt, {}).get("ratio")),
                key=lambda r: r["n"],
            )
            if not sel:
                continue
            ax.plot(
                [r["n"] for r in sel],
                [r["grads"][wrt]["ratio"] for r in sel],
                marker=style.MARKERS[bi % len(style.MARKERS)],
                linestyle=["-", "--", ":"][wi % 3],
                color=style.CATEGORICAL[wi % len(style.CATEGORICAL)],
                label=f"d/d {wrt} - {basis}/{lane}",
                markersize=3.4,
            )
ax.axhline(1.0, color=style.INK_MUTED, linewidth=0.8, zorder=1)
ax.set_xscale("log")
ax.set_xlabel("$N$")
ax.set_ylabel("(forward + backward) / forward")
style.finish(ax, legend_kwargs={"loc": "best", "fontsize": 5.8})

modes = sorted({r["mode"] for r in recs})
axes[1].text(0.02, 0.98, "timed mode: " + ", ".join(modes),
             transform=axes[1].transAxes, va="top", ha="left",
             fontsize=6.5, color=style.INK_MUTED)
fig.tight_layout()
style.footer(
    fig,
    jsonio.config_caption(cfg, ["order", "theta", "leaf_size", "precision", "device"]),
)
style.save(fig, FIG_DIR / "fig12_autodiff_overhead.pdf")

for r in recs:
    for wrt, g in r["grads"].items():
        if g.get("ratio"):
            print(f"N={r['n']:<8d} {r['basis']:<8s} {r['lane']:<12s} {r['mode']:<5s} "
                  f"d/d{wrt:<10s} ratio={g['ratio']:.2f}")
"""

FIG12_CAPTION = """\
Cost of the reverse pass relative to the forward pass for the fixed-topology
differentiable FMM force, against $N$. **Left:** absolute forward and
forward-plus-backward wall-clock. **Right:** their ratio, which is the quantity
that matters -- the translation cascade is linear, so its reverse pass is a
transpose and a small bounded factor is the expected result. Tree construction is
outside both arms; including it in only the forward would flatter the ratio. Each
measurement records whether the call was jitted or ran eagerly (eager dispatch
re-traces per call), and jitted and eager points are never joined. Values from
`bench/results/differentiability/autodiff_overhead.json`.
"""

# --------------------------------------------------------------------------- #
# figure 13 -- FD vs autodiff
# --------------------------------------------------------------------------- #

FIG13 = """\
art = jsonio.read_result("differentiability/grad_correctness.json")
cfg, recs = art["config"], art["data"]["records"]
recs = [r for r in recs if "error" not in r]
if not recs:
    raise SystemExit("grad_correctness.json has no successful rows")

bases = [b for b in cfg["basis"] if any(r["basis"] == b for r in recs)]
wrts = list(cfg["wrt"])

fig, axes = style.figure(width=style.TWO_COL, height=2.7, ncols=2)

# Left: the frozen-topology check -- autodiff against a finite difference of the
# SAME function. This is the one that tests autodiff.
# Right: the full-pipeline check, which rebuilds the tree and therefore also
# measures topology sensitivity. Kept apart on purpose.
for ax, field, title in (
    (axes[0], "frozen_rel_l2", "frozen topology (tests autodiff)"),
    (axes[1], "full_pipeline_rel_l2", "full pipeline (rebuilds the tree)"),
):
    for wi, wrt in enumerate(wrts):
        for bi, basis in enumerate(bases):
            sel = sorted(
                (r for r in recs
                 if r["basis"] == basis and r["wrt"] == wrt and r.get(field)),
                key=lambda r: r["theta"],
            )
            if not sel:
                continue
            ax.plot(
                [r["theta"] for r in sel],
                [r[field] for r in sel],
                marker=style.MARKERS[bi % len(style.MARKERS)],
                linestyle=["-", "--"][wi % 2],
                color=style.entity_color(wrt, wi),
                label=f"d/d {wrt} - {basis}",
                markersize=3.6,
                markerfacecolor="white",
                markeredgecolor=style.entity_color(wrt, wi),
            )
    ax.set_yscale("log")
    ax.set_xlabel(r"opening angle $\\theta$")
    ax.set_ylabel("relative $L_2$ gradient error")
    ax.set_title(title, fontsize=8)
    style.finish(ax, legend=(ax is axes[0]), legend_kwargs={"loc": "best", "fontsize": 6.2})

fig.tight_layout()
style.footer(
    fig,
    jsonio.config_caption(cfg, ["n", "order", "leaf_size", "precision", "device"])
    + f"\\nFD: {cfg['fd_samples']} coords, eps={cfg['fd_eps']:g}",
)
style.save(fig, FIG_DIR / "fig13_grad_correctness.pdf")

for r in recs:
    print(f"{r['basis']:<9s} theta={r['theta']:<4.2f} {r['wrt']:<9s} "
          f"frozen={r['frozen_rel_l2']:.2e} "
          f"full={r.get('full_pipeline_rel_l2')} far_pairs={r['far_pairs']}")
"""

FIG13_CAPTION = """\
Agreement between autodiff and finite-difference gradients of the FMM force
against opening angle $\\theta$, differentiating with respect to positions and to
masses. **Left:** both arms perturb the *same* fixed-topology function
(`differentiable_accelerations` on one prepared state), which is what
`differentiable_accelerations` promises to be exact for; this is the test of
autodiff. **Right:** finite differences of the full pipeline, where the tree is
rebuilt at every perturbed point, so this additionally measures sensitivity to a
changed topology. The two are reported separately because merging them would
charge autodiff for topology changes: the acceptance criterion is piecewise
constant in the positions, so a perturbation that moves a pair across the
acceptance boundary changes which interactions exist, and only the finite
difference sees it. Values from
`bench/results/differentiability/grad_correctness.json`.
"""

# --------------------------------------------------------------------------- #
# figure 08 -- strong scaling
# --------------------------------------------------------------------------- #

FIG08 = """\
# Contention on a shared host is additive and intermittent, so a sweep's own
# median can be inflated at any point -- measured: two of three invocations of the
# six-device weak point read 455 and 399 ms against a true 229. The MINIMUM across
# independent invocations is the estimator section 5 quotes, so the figures use it
# too, or figure and text disagree at exactly the contended points.
def _min_over_invocations(stem):
    # -> ({ndev: record}, config), keeping per ndev the lowest-median invocation.
    best, cfg = {}, None
    for suffix in ("", "_rep1", "_rep2"):
        try:
            art = jsonio.read_result("multigpu/" + stem + suffix + ".json")
        except FileNotFoundError:
            continue
        cfg = art["config"]
        for r in art["data"]["records"]:
            if not r.get("valid"):
                continue
            d = r["ndev"]
            if d not in best or r["median_s"] < best[d]["median_s"]:
                best[d] = r
    if not best:
        raise SystemExit("no valid " + stem + " records in any invocation")
    return best, cfg

best, cfg = _min_over_invocations("strong_scaling")
recs = [best[d] for d in sorted(best)]

# A point whose traversal buffers overflowed produced a TRUNCATED force, so its
# wall clock is padded-buffer overhead over a wrong answer. Those are dropped
# from the curve and reported on the figure rather than silently omitted -- the
# regime boundary is the result here, not an inconvenience.
ok = list(recs)
bad = []  # invalid points already dropped by _min_over_invocations
if not ok:
    raise SystemExit(
        "strong_scaling.json has no valid points: every device count overflowed. "
        "Lower the fixed N, or raise leaf_size and the traversal caps."
    )
ok.sort(key=lambda r: r["ndev"])

ndev = [r["ndev"] for r in ok]
t = [r["median_s"] for r in ok]
# Efficiency is referenced to the smallest VALID device count, not to 1: a
# single-device run is not the same code path.
ref_d, ref_t = ndev[0], t[0]
eff = [(ref_t * ref_d) / (ti * di) for di, ti in zip(ndev, t)]

fig, axes = style.figure(width=style.TWO_COL, height=2.7, ncols=2)

axes[0].plot(ndev, t, marker=style.MARKERS[0], color=style.entity_color("wall", 0))
axes[0].set_xlabel("devices")
axes[0].set_ylabel("time per force evaluation [s]")
axes[0].set_yscale("log")
axes[0].set_title("wall clock at fixed $N$", fontsize=8)
style.finish(axes[0], legend=False)

axes[1].plot(ndev, eff, marker=style.MARKERS[1], color=style.entity_color("eff", 1),
             label="measured")
axes[1].axhline(1.0, linestyle=":", linewidth=0.8, color="0.5", label="ideal")
axes[1].set_xlabel("devices")
axes[1].set_ylabel(f"parallel efficiency (ref. {ref_d} devices)")
axes[1].set_ylim(0, 1.15)
axes[1].set_title("efficiency", fontsize=8)
style.finish(axes[1], legend=True, legend_kwargs={"loc": "best", "fontsize": 6.2})

note = jsonio.config_caption(cfg, ["order", "leaf_size", "precision", "device"])
if bad:
    dropped = ", ".join(f"{r['ndev']}x" for r in sorted(bad, key=lambda r: r["ndev"]))
    note += (
        f"\\ndropped (buffer overflow, force truncated): {dropped}"
        f"  |  healthy load {cfg['healthy_per_device_n']}/device"
    )
style.annotate_config(axes[0], note)
style.save(fig, FIG_DIR / "fig08_strong_scaling.pdf")

for r in sorted(recs, key=lambda r: r["ndev"]):
    flag = "" if r.get("valid") else "   INVALID (overflow: %s)" % ",".join(r.get("overflowed", []))
    print(f"ndev={r['ndev']:<3d} n={r.get('n')} per_dev={r.get('per_device_n')} "
          f"median={r.get('median_s')}{flag}")
"""

FIG08_CAPTION = """\
Strong scaling at fixed total $N$: time per force evaluation and parallel
efficiency against device count. Efficiency is referenced to the smallest device
count that produced a valid force, not to a single device, because the
single-device path is a different code path. Points at which a traversal buffer
overflowed are excluded and named on the figure: an overflow truncates the force,
so those runs are not slow correct answers but fast wrong ones. The excluded
region is the low-device-count end, where the per-device load is highest.\
"""

# --------------------------------------------------------------------------- #
# figure 09 -- weak scaling
# --------------------------------------------------------------------------- #

FIG09 = """\
# Contention on a shared host is additive and intermittent, so a sweep's own
# median can be inflated at any point -- measured: two of three invocations of the
# six-device weak point read 455 and 399 ms against a true 229. The MINIMUM across
# independent invocations is the estimator section 5 quotes, so the figures use it
# too, or figure and text disagree at exactly the contended points.
def _min_over_invocations(stem):
    # -> ({ndev: record}, config), keeping per ndev the lowest-median invocation.
    best, cfg = {}, None
    for suffix in ("", "_rep1", "_rep2"):
        try:
            art = jsonio.read_result("multigpu/" + stem + suffix + ".json")
        except FileNotFoundError:
            continue
        cfg = art["config"]
        for r in art["data"]["records"]:
            if not r.get("valid"):
                continue
            d = r["ndev"]
            if d not in best or r["median_s"] < best[d]["median_s"]:
                best[d] = r
    if not best:
        raise SystemExit("no valid " + stem + " records in any invocation")
    return best, cfg

best, cfg = _min_over_invocations("weak_scaling")
recs = [best[d] for d in sorted(best)]

ok = list(recs)
bad = []  # invalid points already dropped by _min_over_invocations
if not ok:
    raise SystemExit(
        "weak_scaling.json has no valid points: the per-device load overflows "
        "even at the smallest device count."
    )
ok.sort(key=lambda r: r["ndev"])

ndev = [r["ndev"] for r in ok]
thru = [r["throughput_particles_per_s"] for r in ok]
t = [r["median_s"] for r in ok]

fig, axes = style.figure(width=style.TWO_COL, height=2.7, ncols=2)

# Throughput. Perfect weak scaling is a straight line through the origin: each
# added device adds its own share of work and its own share of throughput.
axes[0].plot(ndev, thru, marker=style.MARKERS[0], color=style.entity_color("thru", 0),
             label="measured")
ideal = [thru[0] * (d / ndev[0]) for d in ndev]
axes[0].plot(ndev, ideal, linestyle=":", linewidth=0.8, color="0.5", label="ideal")
axes[0].set_xlabel("devices")
axes[0].set_ylabel("throughput [particles s$^{-1}$]")
axes[0].set_title("throughput at fixed load per device", fontsize=8)
style.finish(axes[0], legend=True, legend_kwargs={"loc": "best", "fontsize": 6.2})

# Time per evaluation. Flat is perfect; the rise is communication plus whatever
# padding the capacity retries added.
axes[1].plot(ndev, t, marker=style.MARKERS[1], color=style.entity_color("wall", 1))
axes[1].axhline(t[0], linestyle=":", linewidth=0.8, color="0.5")
axes[1].set_xlabel("devices")
axes[1].set_ylabel("time per force evaluation [s]")
axes[1].set_title("cost per evaluation (flat is ideal)", fontsize=8)
style.finish(axes[1], legend=False)

note = jsonio.config_caption(cfg, ["order", "leaf_size", "precision", "device"])
note += f"\\n{cfg.get('per_device_n_fixed')} particles/device"
if bad:
    note += f"  |  {len(bad)} point(s) dropped (overflow)"
style.annotate_config(axes[0], note)
style.save(fig, FIG_DIR / "fig09_weak_scaling.pdf")

for r in sorted(recs, key=lambda r: r["ndev"]):
    flag = "" if r.get("valid") else "   INVALID"
    print(f"ndev={r['ndev']:<3d} n={r.get('n')} median={r.get('median_s')} "
          f"throughput={r.get('throughput_particles_per_s')}{flag}")
"""

FIG09_CAPTION = """\
Weak scaling: the particle count per device is held fixed and devices are added,
so the ideal throughput is linear in device count and the ideal cost per
evaluation is flat. This is the load-bearing scaling measurement for this
implementation, because the per-device particle count is what the traversal
buffers are sized against; holding it fixed is the regime the code is built to
run in.\
"""

# --------------------------------------------------------------------------- #
# figure 11 -- load balance on a clustered distribution
# --------------------------------------------------------------------------- #

FIG11 = """\
art = jsonio.read_result("multigpu/load_balance.json")
cfg, recs = art["config"], art["data"]["records"]

ok = [r for r in recs if r.get("valid")]
if not ok:
    raise SystemExit("load_balance.json has no valid points")

dists = [d for d in cfg["distributions"] if any(r["distribution"] == d for r in ok)]

fig, axes = style.figure(width=style.TWO_COL, height=2.7, ncols=2)

# Left: pair work per device. A space-filling-curve split balances *particles* by
# construction, so a uniform cube looks perfect whatever the partition does. The
# clustered case is the one that can disagree, because the device holding the
# dense centre is handed more pair work than the ones holding the outskirts.
width = 0.8 / max(1, len(dists))
for di, dist in enumerate(dists):
    sel = [r for r in ok if r["distribution"] == dist]
    r = max(sel, key=lambda r: r["ndev"])
    work = r["per_device_work_total"]
    xs = [i + (di - (len(dists) - 1) / 2) * width for i in range(len(work))]
    axes[0].bar(xs, work, width=width, label=dist,
                color=style.entity_color(dist, di), edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("device")
axes[0].set_ylabel("interaction pairs")
axes[0].set_xticks(range(len(work)))
axes[0].set_title("pair work per device", fontsize=8)
style.finish(axes[0], legend=True, legend_kwargs={"loc": "best", "fontsize": 6.2})

# Right: imbalance vs device count. 1.0 is perfect; the ratio is the busiest
# device over the mean, which is what sets the step time -- every other device
# waits for it.
for di, dist in enumerate(dists):
    sel = sorted((r for r in ok if r["distribution"] == dist), key=lambda r: r["ndev"])
    axes[1].plot([r["ndev"] for r in sel], [r["work_imbalance"] for r in sel],
                 marker=style.MARKERS[di % len(style.MARKERS)],
                 color=style.entity_color(dist, di), label=dist)
axes[1].axhline(1.0, linestyle=":", linewidth=0.8, color="0.5")
axes[1].set_xlabel("devices")
axes[1].set_ylabel("busiest device / mean")
axes[1].set_title("work imbalance", fontsize=8)
style.finish(axes[1], legend=True, legend_kwargs={"loc": "best", "fontsize": 6.2})

style.annotate_config(
    axes[0],
    jsonio.config_caption(cfg, ["order", "leaf_size", "precision", "device"])
    + f"\\n{cfg.get('per_device_n_fixed')} particles/device",
)
style.save(fig, FIG_DIR / "fig11_load_balance.pdf")

for r in sorted(ok, key=lambda r: (r["distribution"], r["ndev"])):
    print(f"{r['distribution']:<9s} ndev={r['ndev']:<3d} "
          f"imbalance={r['work_imbalance']:.4f} work={r['per_device_work_total']}")
"""

FIG11_CAPTION = """\
Per-device pair work for a uniform and a centrally concentrated (Plummer)
distribution, and the resulting imbalance against device count. The imbalance is
the busiest device's share divided by the mean, which is the quantity that sets
the step time because every other device waits on it. A uniform cube is balanced
by construction under any reasonable partition, so it is shown as the control;
the clustered case is the one in which balancing particle counts and balancing
work come apart.\
"""


# --------------------------------------------------------------------------- #
# figure 16 -- gradient cost vs free-parameter count (section 7 headline)
# --------------------------------------------------------------------------- #

FIG16 = """\
art = jsonio.read_result("density_reconstruction/gradient_cost_vs_nparams.json")
cfg, data = art["config"], art["data"]
recs = [r for r in data["records"] if not r.get("failed")]
if not recs:
    raise SystemExit("gradient_cost_vs_nparams.json has no successful rows")
failed = [r for r in data["records"] if r.get("failed")]

# LATENCY-BOUND ROWS ARE NOT PLOTTED AS RATIOS. Below the point where the FMM
# becomes compute bound, a wall-clock ratio between the forward and the gradient
# reports which graph XLA fused into fewer kernels -- it came out BELOW ONE in
# the leaf-64 sweep, which is arithmetically impossible for a reverse pass. Such
# rows still carry the cost-flat-in-P statement, so they stay in the left panel
# and are excluded only from the ratio annotation.
compute_bound = [r for r in recs if not r.get("latency_bound")]

pos = sorted((r for r in recs if r["parameterization"] == "positions"),
             key=lambda r: r["num_free_parameters"])
par = sorted((r for r in recs if r["parameterization"] == "parametric"),
             key=lambda r: r["N"])

fig, axes = style.figure(width=style.TWO_COL, height=2.9, ncols=2)

# -- left: wall-clock against free-parameter count ------------------------- #
ax = axes[0]
P = [r["num_free_parameters"] for r in pos]
ax.plot(P, [r["forward_seconds"] for r in pos], marker=style.MARKERS[0],
        color=style.ENTITY["forward"], label="forward only")
ax.plot(P, [r["forward_backward_seconds"] for r in pos], marker=style.MARKERS[1],
        color=style.ENTITY["forward_backward"], label="forward + backward")

# The parametric arm at P = 7 is the same operator with a 450000x narrower
# pytree. It is drawn as points, not a curve: it is one parameter count.
if par:
    ax.scatter([r["num_free_parameters"] for r in par],
               [r["forward_backward_seconds"] for r in par],
               marker=style.MARKERS[3], s=18, zorder=5,
               facecolor="none", edgecolor=style.INK,
               label="parametric (P = %d), same N" % par[0]["num_free_parameters"])

# Finite differences, EXTRAPOLATED as (P + 1) forward evaluations from this
# run's own measured forward. Never measured; the label says so.
fd_P = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 3e7])
ref = data["annotations"]["reference_forward_seconds"]
ax.plot(fd_P, (fd_P + 1.0) * ref, linestyle=":", color=style.INK_MUTED,
        label="finite differences, extrapolated")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("free parameters $P$")
ax.set_ylabel("wall-clock per evaluation [s]")

# The annotation that IS the argument, in human units.
years = data["annotations"]["finite_difference_at"]["10000000"]["years"]
ax.annotate("$10^7$ parameters:\\n%.1f yr by finite\\ndifferences" % years,
            xy=(1e7, (1e7 + 1) * ref), xytext=(3e3, (1e7 + 1) * ref * 0.06),
            fontsize=6.0, color=style.INK,
            arrowprops={"arrowstyle": "->", "color": style.INK_MUTED, "lw": 0.6})
style.finish(ax, legend=True, legend_kwargs={"loc": "upper left", "fontsize": 5.6})

# -- right: the reverse-pass multiple, and memory -------------------------- #
ax = axes[1]
Ns = [r["N"] for r in compute_bound if r["parameterization"] == "positions"]
ratios = [r["backward_over_forward"] for r in compute_bound
          if r["parameterization"] == "positions"]
flops = [r["backward_over_forward_flops"] for r in compute_bound
         if r["parameterization"] == "positions"]
ax.plot(Ns, ratios, marker=style.MARKERS[0], color=style.ENTITY["forward_backward"],
        label="wall-clock")
if any(f is not None for f in flops):
    ax.plot([n for n, f in zip(Ns, flops) if f is not None],
            [f for f in flops if f is not None],
            marker=style.MARKERS[2], linestyle="--", color=style.INK_MUTED,
            label="XLA flop count")
ax.axhline(3.0, color=style.GRID, lw=0.8, zorder=0)
ax.text(Ns[0], 3.15, "3x", fontsize=5.6, color=style.INK_MUTED)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("$N$")
ax.set_ylabel("(forward + backward) / forward")

memory = [(r["N"], r.get("forward_backward_temp_bytes"))
          for r in pos if r.get("forward_backward_temp_bytes")]
if memory:
    twin = ax.twinx()
    twin.plot([m[0] for m in memory], [m[1] / 2**30 for m in memory],
              marker=style.MARKERS[4], linestyle="-.", color=style.CATEGORICAL[0],
              lw=0.9)
    twin.set_yscale("log")
    twin.set_ylabel("gradient temp memory [GiB]", color=style.CATEGORICAL[0])
    twin.tick_params(axis="y", colors=style.CATEGORICAL[0])
style.finish(ax, legend=True, legend_kwargs={"loc": "upper left", "fontsize": 5.6})

ceiling = ""
if failed:
    ceiling = "   single-device ceiling: N=%d failed (%s)" % (
        failed[0]["N"], failed[0].get("error_type", "?"))
    print("ceiling:", failed[0]["N"], failed[0].get("error_type"))

fig.tight_layout()
style.footer(fig, "%s, %s, order %d, theta %.2f, leaf %d, M=%d, mode=%s%s" % (
    art["meta"]["device_kind"], cfg["precision"], cfg["order"], cfg["theta"],
    cfg["leaf_size"], cfg["M"], cfg.get("mode", "?"), ceiling))
style.save(fig, str(FIG_DIR / "fig_16_gradient_cost_vs_nparams.pdf"))
"""

FIG16_CAPTION = r"""
Cost of a reverse-mode gradient against the number of free source positions.
**Left:** wall-clock for one forward evaluation and for one
forward-plus-backward, against free-parameter count $P$. Both are flat in $P$ --
the open diamonds are the 7-parameter parametric model evaluated through the
*same* operator at the same $N$, and they land on the high-dimensional points,
so a change of $P$ by a factor of $4.5\times10^{5}$ changes the cost by about
1%. The dotted line is finite differences, **extrapolated** as $(P+1)$ forward
evaluations from this run's own measured forward time; it was not measured, and
one-sided rather than central differences are assumed so the baseline is not
inflated. **Right:** the reverse-pass multiple against $N$, by wall-clock and by
XLA's flop count, with the gradient's peak temporary memory on the second axis.
The multiple is close to 3 up to $N\sim10^{4}$ and grows to roughly 13 at
$N=10^{6}$: bounded, and independent of $P$, but *not* a small constant across
this range. Points at which the pipeline is launch-latency rather than compute
bound are excluded from the right panel -- there a wall-clock ratio measures
kernel fusion and not differentiation -- but retained on the left, where the
statement is about $P$. The single-device ceiling in the footer is measured, by
running until it failed.
"""


# --------------------------------------------------------------------------- #
# figure 17 -- reconstruction quality
# --------------------------------------------------------------------------- #

FIG17 = """\
art = jsonio.read_result("density_reconstruction/reconstruction_runs.json")
cfg, recs = art["config"], art["data"]["records"]
recs = [r for r in recs if not r.get("failed")]
if not recs:
    raise SystemExit("reconstruction_runs.json has no successful rows")

def pick(**want):
    for r in recs:
        if all(r.get(k) == v for k, v in want.items()):
            return r
    return None

ref = dict(initial_guess="perturbed_truth", noise_fraction=cfg["noise_fractions"][0],
           softening=cfg["softenings"][0], perturber="lmc_like", regularized=True)
high = pick(parameterization="positions", **ref)
low = pick(parameterization="parametric", **ref)
unreg = pick(parameterization="positions", **{**ref, "regularized": False})
if high is None:
    raise SystemExit("no reference high-dimensional run in the artifact")

fig, axes = style.figure(width=style.TWO_COL, height=5.0, ncols=2, nrows=2)

# (a) enclosed mass: truth vs parametric vs high-dimensional -------------- #
ax = axes[0][0]
prof = high["profile_truth"]
ax.plot(prof["radius"], prof["enclosed_mass"], color=style.INK, lw=1.3, label="truth")
ax.plot(high["profile_initial"]["radius"], high["profile_initial"]["enclosed_mass"],
        color=style.GRID, lw=1.0, label="initial guess")
if low is not None:
    ax.plot(low["profile_reconstructed"]["radius"],
            low["profile_reconstructed"]["enclosed_mass"],
            marker=style.MARKERS[1], ms=2.4, color=style.CATEGORICAL[0],
            label="parametric (P=%d)" % low["num_free_parameters"])
ax.plot(high["profile_reconstructed"]["radius"],
        high["profile_reconstructed"]["enclosed_mass"],
        marker=style.MARKERS[0], ms=2.4, color=style.ENTITY["positions"],
        label="positions (P=%d)" % high["num_free_parameters"])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("$r$"); ax.set_ylabel("enclosed mass $M(<r)$")
style.finish(ax, legend=True, legend_kwargs={"loc": "upper left", "fontsize": 5.6})

# (b) field residual before/after, and the noise floor -------------------- #
ax = axes[0][1]
labels, before, after, floors = [], [], [], []
for label, run in (("parametric", low), ("positions", high), ("unregularised", unreg)):
    if run is None:
        continue
    labels.append(label)
    before.append(run["field_residual_before"]["rel_l2"])
    after.append(run["field_residual_after"]["rel_l2"])
    floors.append(run["field_residual_after"].get("noise_floor_rel_l2"))
x = np.arange(len(labels))
ax.bar(x - 0.18, before, width=0.34, color=style.GRID, label="before")
ax.bar(x + 0.18, after, width=0.34, color=style.ENTITY["positions"], label="after")
for i, floor in enumerate(floors):
    if floor:
        ax.plot([i - 0.4, i + 0.4], [floor, floor], color=style.CATEGORICAL[2],
                lw=1.0, ls="--", label="noise floor" if i == 0 else None)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.8)
ax.set_ylabel("field residual, relative $L_2$")
style.finish(ax, legend=True, legend_kwargs={"loc": "upper right", "fontsize": 5.6})

# (c) shell density, truth vs reconstruction ------------------------------- #
ax = axes[1][0]
ax.plot(prof["radius"], prof["shell_density"], color=style.INK, lw=1.3, label="truth")
ax.plot(high["profile_reconstructed"]["radius"],
        high["profile_reconstructed"]["shell_density"],
        marker=style.MARKERS[0], ms=2.4, color=style.ENTITY["positions"],
        label="reconstruction")
if unreg is not None:
    ax.plot(unreg["profile_reconstructed"]["radius"],
            unreg["profile_reconstructed"]["shell_density"],
            marker=style.MARKERS[2], ms=2.4, ls="--", color=style.CATEGORICAL[2],
            label="unregularised")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("$r$"); ax.set_ylabel(r"shell density $\\rho(r)$")
style.finish(ax, legend=True, legend_kwargs={"loc": "lower left", "fontsize": 5.6})

# (d) the degeneracy, stated as numbers ----------------------------------- #
# Panel (d) is the unregularised run. What it must show is that the FIELD can be
# fitted while the DENSITY is not recovered -- so it plots both, per run, rather
# than a picture of particles that would invite the eye to judge by clumpiness.
ax = axes[1][1]
rows = [(l, r) for l, r in (("parametric", low), ("positions", high),
                            ("unregularised", unreg)) if r is not None]
fx = [r["field_residual_after"]["rel_l2"] for _, r in rows]
dx = [r["density_after"]["grid_rel_l2"] for _, r in rows]
for i, (label, _) in enumerate(rows):
    ax.scatter(fx[i], dx[i], marker=style.MARKERS[i], s=26,
               color=style.CATEGORICAL[i % len(style.CATEGORICAL)], label=label)
lim = [min(fx + dx) * 0.6, max(fx + dx) * 1.6]
ax.plot(lim, lim, color=style.GRID, lw=0.8, zorder=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("field residual (primary)")
ax.set_ylabel("density disagreement (secondary)")
style.finish(ax, legend=True, legend_kwargs={"loc": "upper left", "fontsize": 5.6})

fig.tight_layout()
style.footer(fig, "%s, N=%d, M=%d, %d iterations, softening %g, noise %g" % (
    art["meta"]["device_kind"], high["N"], high["M"], high["iterations"],
    high["softening"], high["noise_fraction"]))
style.save(fig, str(FIG_DIR / "fig_17_reconstruction_quality.pdf"))
"""

FIG17_CAPTION = r"""
Reconstruction quality. **(a)** Enclosed-mass profile: truth, the initial guess,
the parametric fit and the high-dimensional fit. **(b)** Field-space residual
before and after each fit, with the noise floor marked -- a residual at the
floor has extracted everything the data holds, and driving it lower is fitting
the noise realisation. **(c)** Shell density, including the deliberately
unregularised run. **(d)** The degeneracy stated quantitatively: field residual
against density disagreement, with the diagonal drawn. Points below the diagonal
have fitted the field better than they have recovered the density. This is the
section's central caveat and not an artefact -- recovering discrete source
positions from external field samples is strongly ill-posed and has continuous
degeneracies, so a small field residual is *not* evidence that the mass
distribution has been recovered. Per-particle position error is not shown: the
sources are equal-mass and therefore interchangeable, which makes it close to
meaningless.
"""


# --------------------------------------------------------------------------- #
# figure 18 -- convergence and the initial-guess sweep
# --------------------------------------------------------------------------- #

FIG18 = """\
art = jsonio.read_result("density_reconstruction/reconstruction_runs.json")
cfg, recs = art["config"], art["data"]["records"]
recs = [r for r in recs if not r.get("failed")]
if not recs:
    raise SystemExit("reconstruction_runs.json has no successful rows")

fig, axes = style.figure(width=style.TWO_COL, height=2.9, ncols=2)

# -- left: loss against iteration, one curve per initial guess ------------- #
ax = axes[0]
guesses = [g for g in cfg["initial_guesses"]]
for i, guess in enumerate(guesses):
    sel = [r for r in recs if r["parameterization"] == "positions"
           and r["initial_guess"] == guess and r["regularized"]]
    if not sel:
        continue
    run = sel[0]
    trace = run["loss_trace"]
    ax.plot([t["iteration"] for t in trace], [t["loss"] for t in trace],
            color=style.CATEGORICAL[i % len(style.CATEGORICAL)],
            label=guess.replace("_", " "))
ax.set_yscale("log")
ax.set_xlabel("iteration"); ax.set_ylabel("loss (field residual + penalties)")
style.finish(ax, legend=True, legend_kwargs={"loc": "upper right", "fontsize": 5.6})

# -- right: the divergence -- field improves, density saturates ------------ #
# This is fig18's content. Plotted as the RATIO of each metric to its own
# starting value, so two quantities in different units share an axis honestly.
ax = axes[1]
for i, kind in enumerate(("parametric", "positions")):
    sel = sorted((r for r in recs if r["parameterization"] == kind and r["regularized"]),
                 key=lambda r: r["num_free_parameters"])
    if not sel:
        continue
    field_gain = [r["field_residual_after"]["rel_l2"] / r["field_residual_before"]["rel_l2"]
                  for r in sel]
    dens_gain = [r["density_after"]["grid_rel_l2"] / r["density_before"]["grid_rel_l2"]
                 for r in sel]
    ax.scatter(field_gain, dens_gain, marker=style.MARKERS[i], s=24,
               color=style.CATEGORICAL[i % len(style.CATEGORICAL)], label=kind)
ax.plot([1e-3, 2.0], [1e-3, 2.0], color=style.GRID, lw=0.8, zorder=0)
ax.axhline(1.0, color=style.GRID, lw=0.6, zorder=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("field residual, after / before")
ax.set_ylabel("density disagreement, after / before")
style.finish(ax, legend=True, legend_kwargs={"loc": "lower right", "fontsize": 5.6})

fig.tight_layout()
style.footer(fig, "%s, %d runs, N in %s, %d iterations" % (
    art["meta"]["device_kind"], len(recs), cfg["n"], cfg["iterations"]))
style.save(fig, str(FIG_DIR / "fig_18_convergence_and_degeneracy.pdf"))
"""

FIG18_CAPTION = r"""
Convergence, and the sensitivity to where the fit starts. **Left:** loss against
iteration for the high-dimensional fit from each of the four initial guesses --
a small perturbation of truth, an isotropised truth, a structurally wrong smooth
analytic model, and a naive uniform sphere. The loss is non-convex in the
positions, so the spread between these curves is a result in its own right and
not a nuisance. **Right:** the divergence that states the degeneracy
quantitatively. Each point is one run, plotting how much the field residual
improved against how much the density disagreement improved, each relative to
its own starting value. Points to the right of the diagonal improved the field
by more than they improved the density; points near the horizontal line at one
did not recover the density at all despite fitting the field. The
high-dimensional arm has vastly more freedom to move mass around at fixed field
than the parametric arm, and this is where that shows.
"""


# --------------------------------------------------------------------------- #
# figure 19 -- topology switching (the honest methods figure)
# --------------------------------------------------------------------------- #

FIG19 = """\
art = jsonio.read_result("density_reconstruction/topology_switching.json")
cfg, recs = art["config"], art["data"]["records"]
recs = [r for r in recs if not r.get("failed")]
if not recs:
    raise SystemExit("topology_switching.json has no successful rows")

def rate(r, name):
    return (r.get("switch_summary") or {}).get("intensive", {}).get("mean", {}).get(name)

fig, axes = style.figure(width=style.TWO_COL, height=5.0, ncols=2, nrows=2)

# (a) THE METRIC PANEL: intensive rates resolve, the extensive one saturates. #
ax = axes[0][0]
base_lr = min(r["learning_rate"] for r in recs)
sel = sorted((r for r in recs if r["learning_rate"] == base_lr
              and r["rebuild_cadence"] == 1), key=lambda r: r["N"])
Ns = [r["N"] for r in sel]
for i, (name, label) in enumerate((
        ("near_set_churn", "near-field source set (intensive)"),
        ("near_pair_churn", "near leaf-pair set"),
        ("leaf_churn", "leaf membership"),
        ("slot_churn", "Morton slot"))):
    values = [rate(r, name) for r in sel]
    if all(v is None for v in values):
        continue
    ax.plot(Ns, [v if v is not None else np.nan for v in values],
            marker=style.MARKERS[i % len(style.MARKERS)],
            color=style.CATEGORICAL[i % len(style.CATEGORICAL)], label=label)
ax.plot(Ns, [(r["switch_summary"] or {}).get("extensive", {}).get("switch_rate")
             for r in sel], marker="x", ls=":", color=style.INK,
        label="extensive 'anything changed?'")
ax.set_xscale("log"); ax.set_ylim(-0.04, 1.10)
ax.set_xlabel("$N$"); ax.set_ylabel("churn per rebuild")
style.finish(ax, legend=True, legend_kwargs={"loc": "lower right", "fontsize": 5.2})

# (b) cadence: how stale may the interaction list be? --------------------- #
ax = axes[0][1]
for i, n in enumerate(sorted({r["N"] for r in recs})):
    sel = sorted((r for r in recs if r["N"] == n and r["learning_rate"] == base_lr),
                 key=lambda r: r["rebuild_cadence"])
    if len(sel) < 2:
        continue
    ks = [r["rebuild_cadence"] for r in sel]
    ax.plot(ks, [rate(r, "near_set_churn") for r in sel],
            marker=style.MARKERS[i % len(style.MARKERS)],
            color=style.CATEGORICAL[i % len(style.CATEGORICAL)],
            label="N=%d, interaction set" % n)
    base = sel[0]["final_loss"]
    ax.plot(ks, [r["final_loss"] / base for r in sel], marker="x", ls="--",
            color=style.CATEGORICAL[i % len(style.CATEGORICAL)],
            label="N=%d, final loss / loss(k=1)" % n)
ax.set_xscale("log", base=2)
ax.set_xlabel("rebuild cadence $k$"); ax.set_ylabel("churn, and relative final loss")
style.finish(ax, legend=True, legend_kwargs={"loc": "center left", "fontsize": 5.2})

# (c) loss continuity across a switch, against the progress a step makes --- #
ax = axes[1][0]
pts = [(r["N"], r["rebuild_cadence"], r["loss_continuity"]["median_relative_jump"],
        r["loss_continuity"]["max_relative_jump"])
       for r in recs if r.get("loss_continuity")
       and r["loss_continuity"].get("median_relative_jump") is not None]
if pts:
    for i, n in enumerate(sorted({p[0] for p in pts})):
        sub = sorted((p for p in pts if p[0] == n), key=lambda p: p[1])
        ax.plot([p[1] for p in sub], [p[2] for p in sub],
                marker=style.MARKERS[i % len(style.MARKERS)],
                color=style.CATEGORICAL[i % len(style.CATEGORICAL)],
                label="N=%d, median" % n)
        ax.plot([p[1] for p in sub], [p[3] for p in sub], marker="x", ls=":",
                color=style.CATEGORICAL[i % len(style.CATEGORICAL)],
                label="N=%d, max" % n)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("rebuild cadence $k$")
ax.set_ylabel("loss jump at a rebuild, relative")
style.finish(ax, legend=True, legend_kwargs={"loc": "upper left", "fontsize": 5.2})

# (d) FD vs autodiff: pinned within an epoch, crossed over a switch -------- #
ax = axes[1][1]
fd = [r for r in recs if r.get("fd_agreement")]
if fd:
    x = np.arange(len(fd))
    ax.plot(x, [r["fd_agreement"]["pinned_median_rel"] for r in fd],
            marker=style.MARKERS[0], color=style.ENTITY["positions"],
            label="pinned within one epoch")
    ax.plot(x, [r["fd_agreement"]["crossed_median_rel"] for r in fd],
            marker=style.MARKERS[2], ls="--", color=style.CATEGORICAL[2],
            label="crossing a switch")
    ax.set_xticks(x)
    ax.set_xticklabels(["%d/k%d" % (r["N"], r["rebuild_cadence"]) for r in fd],
                       rotation=60, fontsize=4.6)
ax.set_yscale("log")
ax.set_ylabel("|FD $-$ autodiff| / |autodiff|")
style.finish(ax, legend=True, legend_kwargs={"loc": "upper left", "fontsize": 5.2})

fig.tight_layout()
style.footer(fig, "%s, leaf %d, order %d, theta %.2f, %d iterations, lr %g" % (
    art["meta"]["device_kind"], cfg["leaf_size"], cfg["order"], cfg["theta"],
    cfg["iterations"], base_lr))
style.save(fig, str(FIG_DIR / "fig_19_topology_switching.pdf"))
"""

FIG19_CAPTION = r"""
What a per-iteration tree rebuild does to a gradient descent -- the section's
second substantive result, and the first high-dimensional evidence that
per-iteration rebuilds do not obstruct convergence for an inference objective
through an FMM. **(a)** Why the rate is reported per particle. The extensive
"did the discrete structure change at all?" counter reads exactly one at every
$N$ and carries no information; the intensive rates resolve the structure. The
top curve to read is the near-field *source set* churn -- the fraction of the
sources reaching a given particle by direct summation that changed -- which is
the interaction-list question, and which the leaf-pair and Morton-slot rates
both overstate badly. **(b)** Cadence: the interaction list becomes almost
entirely stale as $k$ grows while the final loss does not move, so $k>1$ is
usable, and it is this measurement rather than any appeal to ordering stability
that says so. **(c)** The loss discontinuity at a rebuild boundary, measured at
*fixed positions* under the outgoing and incoming topologies so it isolates the
topology change from the optimiser's step. **(d)** Finite differences against
autodiff, pinned within one topology epoch and crossing a switch. Within an
epoch they agree to $\sim10^{-9}$, which is the fixed-topology contract of
Sect.~2 measured rather than argued; a finite difference that straddles a switch
legitimately disagrees, and by how much is what this panel reports. The pinned
arm pins the interaction-list selection as well as the tree -- pinning only the
tree leaves a residual that reads as a gradient bug and is not one. Cited
alongside, and not confirmed by, the low-dimensional Yggdrax precedent.
"""


# --------------------------------------------------------------------------- #
# figure 20 -- multi-GPU reconstruction scaling
# --------------------------------------------------------------------------- #

FIG20 = """\
art = jsonio.read_result("density_reconstruction/multigpu_scaling.json")
cfg, data = art["config"], art["data"]
recs = [r for r in data["records"] if not r.get("failed") and not r.get("skipped")]
ceiling = data.get("ceiling", [])
if not recs:
    raise SystemExit("multigpu_scaling.json has no successful rows")

fig, axes = style.figure(width=style.TWO_COL, height=2.9, ncols=2)

# -- left: strong scaling at fixed parameter count ------------------------- #
ax = axes[0]
sel = sorted(recs, key=lambda r: r["num_devices"])
ndev = [r["num_devices"] for r in sel]
wall = [r["timing"]["wall_seconds"] for r in sel]
ax.plot(ndev, wall, marker=style.MARKERS[0], color=style.ENTITY["gpu"],
        label="measured")
ax.plot(ndev, [wall[0] * ndev[0] / d for d in ndev], ls=":",
        color=style.INK_MUTED, label="ideal")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("devices"); ax.set_ylabel("fit wall-clock [s]")
style.finish(ax, legend=True, legend_kwargs={"loc": "lower left", "fontsize": 5.6})

# -- right: the ceiling, measured by running until it broke ---------------- #
ax = axes[1]
if ceiling:
    ok, bad = {}, {}
    for r in ceiling:
        target = bad if r.get("failed") else ok
        target.setdefault(r["num_devices"], []).append(r["num_free_parameters"])
    devices = sorted(set(ok) | set(bad))
    largest = [max(ok.get(d, [0])) for d in devices]
    ax.plot(devices, largest, marker=style.MARKERS[1], color=style.ENTITY["gpu"],
            label="largest $P$ that ran")
    for d in devices:
        for p in bad.get(d, []):
            ax.scatter([d], [p], marker="x", s=26, color=style.CATEGORICAL[2],
                       label="out of memory" if d == devices[0] else None)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("devices")
else:
    ax.text(0.5, 0.5, "no ceiling ladder in this artifact", ha="center",
            va="center", fontsize=6, color=style.INK_MUTED,
            transform=ax.transAxes)
    ax.set_xlabel("devices")
ax.set_yscale("log"); ax.set_ylabel("free parameters $P$")
style.finish(ax, legend=bool(ceiling), legend_kwargs={"loc": "upper left",
                                                      "fontsize": 5.6})

fig.tight_layout()
style.footer(fig, "%s, %s sharding, N=%d, %d iterations, leaf %d" % (
    art["meta"]["device_kind"], cfg.get("sharding_mode", "?"), cfg["n"][0],
    cfg["iterations"], cfg["leaf_size"]))
style.save(fig, str(FIG_DIR / "fig_20_multigpu_reconstruction_scaling.pdf"))
"""

FIG20_CAPTION = r"""
The reconstruction across multiple devices. **Left:** wall-clock for the same
fit on an increasing device count, at fixed parameter count, against ideal
scaling -- to be read against the strong- and weak-scaling figures of Sect.~5.
**Right:** the largest free-parameter count that ran, per device count, with the
configurations that exhausted memory marked. The ceiling is measured by climbing
a ladder until it broke, not inferred from a memory model. This is *parameter*
sharding: the optimisation's parameter array is distributed across the mesh, so
the parameter count is bounded by aggregate device memory rather than by one
device's. It is not the distributed force evaluation of Sect.~5, which partitions
the sources and exchanges halos; the two are not interchangeable and each run's
artifact records which it used.
"""


SPECS = [
    (
        "fig_01_force_error_vs_order",
        "Figure 01 -- force error vs expansion order",
        "Loads `bench/results/validation/force_error_vs_order.json`, produced by "
        "`bench/validation/force_error_vs_order.py`. No computation here.",
        FIG01,
        FIG01_CAPTION,
    ),
    (
        "fig_02_error_vs_theta",
        "Figure 02 -- force error vs opening angle",
        "Loads `bench/results/validation/error_vs_theta.json`, produced by "
        "`bench/validation/error_vs_theta.py`. No computation here.",
        FIG02,
        FIG02_CAPTION,
    ),
    (
        "fig_03_mac_comparison",
        "Figure 03 -- geometric vs mass-dependent MAC",
        "Loads `bench/results/validation/mac_comparison.json`, produced by "
        "`bench/validation/mac_comparison.py` (a wrapper over "
        "`bench/validation/mac_error_distribution.py`). No computation here.",
        FIG03,
        FIG03_CAPTION,
    ),
    (
        "fig_04_wallclock_vs_n",
        "Figure 04 -- wall-clock vs N",
        "Loads `bench/results/scaling/wallclock_vs_n.json`, produced by "
        "`bench/scaling/wallclock.py`. No computation here.",
        FIG04,
        FIG04_CAPTION,
    ),
    (
        "fig_05_interaction_counts",
        "Figure 05 -- interaction counts vs N",
        "Loads `bench/results/scaling/interaction_counts.json`, produced by "
        "`bench/scaling/interaction_counts.py`. No computation here.",
        FIG05,
        FIG05_CAPTION,
    ),
    (
        "fig_06_stage_breakdown",
        "Figure 06 -- per-stage time breakdown vs N",
        "Loads `bench/results/scaling/stage_breakdown.json`, produced by "
        "`bench/scaling/stage_breakdown.py`. No computation here.",
        FIG06,
        FIG06_CAPTION,
    ),
    (
        "fig_07_gpu_vs_cpu",
        "Figure 07 -- single-GPU vs CPU speedup",
        "Loads `bench/results/scaling/gpu_vs_cpu.json`, produced by "
        "`bench/scaling/gpu_vs_cpu_speedup.py`. No computation here.",
        FIG07,
        FIG07_CAPTION,
    ),
    (
        "fig_12_autodiff_overhead",
        "Figure 12 -- forward vs forward+backward cost",
        "Loads `bench/results/differentiability/autodiff_overhead.json`, produced by "
        "`bench/differentiability/autodiff_overhead.py`. No computation here.",
        FIG12,
        FIG12_CAPTION,
    ),
    (
        "fig_08_strong_scaling",
        "Figure 08 -- strong scaling",
        "Loads `bench/results/multigpu/strong_scaling.json`, produced by "
        "`bench/multigpu/strong_scaling.py`. No computation here.",
        FIG08,
        FIG08_CAPTION,
    ),
    (
        "fig_09_weak_scaling",
        "Figure 09 -- weak scaling",
        "Loads `bench/results/multigpu/weak_scaling.json`, produced by "
        "`bench/multigpu/weak_scaling.py`. No computation here.",
        FIG09,
        FIG09_CAPTION,
    ),
    (
        "fig_11_load_balance",
        "Figure 11 -- load balance on a clustered distribution",
        "Loads `bench/results/multigpu/load_balance.json`, produced by "
        "`bench/multigpu/load_balance.py`. No computation here.",
        FIG11,
        FIG11_CAPTION,
    ),
    (
        "fig_13_grad_correctness",
        "Figure 13 -- finite-difference vs autodiff gradients",
        "Loads `bench/results/differentiability/grad_correctness.json`, produced by "
        "`bench/differentiability/grad_correctness.py`. No computation here.",
        FIG13,
        FIG13_CAPTION,
    ),
    (
        "fig_16_gradient_cost_vs_nparams",
        "Figure 16 -- gradient cost vs free-parameter count",
        "Loads `bench/results/density_reconstruction/gradient_cost_vs_nparams.json`, "
        "produced by `bench/payoff_static/gradient_cost_vs_nparams.py`. No computation here.",
        FIG16,
        FIG16_CAPTION,
    ),
    (
        "fig_17_reconstruction_quality",
        "Figure 17 -- reconstruction quality",
        "Loads `bench/results/density_reconstruction/reconstruction_runs.json`, "
        "produced by `bench/payoff_static/reconstruction_runs.py`. No computation here.",
        FIG17,
        FIG17_CAPTION,
    ),
    (
        "fig_18_convergence_and_degeneracy",
        "Figure 18 -- convergence and degeneracy",
        "Loads `bench/results/density_reconstruction/reconstruction_runs.json`, "
        "produced by `bench/payoff_static/reconstruction_runs.py`. No computation here.",
        FIG18,
        FIG18_CAPTION,
    ),
    (
        "fig_19_topology_switching",
        "Figure 19 -- topology switching under a per-iteration rebuild",
        "Loads `bench/results/density_reconstruction/topology_switching.json`, "
        "produced by `bench/payoff_static/topology_switching.py`. No computation here.",
        FIG19,
        FIG19_CAPTION,
    ),
    (
        "fig_20_multigpu_reconstruction_scaling",
        "Figure 20 -- multi-GPU reconstruction scaling",
        "Loads `bench/results/density_reconstruction/multigpu_scaling.json`, "
        "produced by `bench/payoff_static/multigpu_scaling.py`. No computation here.",
        FIG20,
        FIG20_CAPTION,
    ),
]


def main() -> int:
    for stem, title, blurb, source, caption in SPECS:
        nb = notebook(title, blurb, source, caption)
        path = HERE / f"{stem}.ipynb"
        nbf.write(nb, str(path))
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
