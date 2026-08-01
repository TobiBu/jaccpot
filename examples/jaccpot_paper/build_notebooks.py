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
FIG_DIR = jsonio.repo_root() / "results" / "figures"
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
`results/validation/force_error_vs_order.json`.
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
    jsonio.config_caption(cfg, ["n", "basis", "leaf_size", "preset", "precision", "device"]),,
)
fig.tight_layout()
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
`results/validation/error_vs_theta.json`.
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
p90_key, p99_key = f"{prefix}p90", f"{prefix}p99"

ARM_LABEL = {
    "fixed": "geometric (sweep $\\\\theta$)",
    "mass": "mass-dependent, eq (16a) (sweep $\\\\epsilon$)",
    "mass_16b": "eq (16b), exact $O(N^2)$ $f_b$",
}
dists = sorted({r["distribution"] for r in recs})
orders = sorted({r["order"] for r in recs})

fig, axes = style.figure(width=style.TWO_COL, height=3.0, ncols=2)

# Left: the raw trade-off. Cost on x, error on y -- lower and further left is
# better, and the two arms' curves are directly comparable without any matching.
ax = axes[0]
for dist_i, dist in enumerate(dists):
    for arm in ("fixed", "mass", "mass_16b"):
        sel = sorted(
            (r for r in recs
             if r["arm"] == arm and r["distribution"] == dist and r[p90_key] > 0),
            key=lambda r: r["pair_work"],
        )
        if not sel:
            continue
        ax.plot(
            [r["pair_work"] for r in sel],
            [r[p90_key] for r in sel],
            marker=style.MARKERS[dist_i % len(style.MARKERS)],
            linestyle=["-", "--", ":"][dist_i % 3],
            color=style.entity_color({"fixed": "geometric"}.get(arm, arm)),
            label=f"{ARM_LABEL[arm]} - {dist}",
            markersize=3.2,
        )
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("pair work (far $\\\\times$ coeffs + near)")
ax.set_ylabel(f"90th-percentile scaled force error ({metric})")
style.finish(ax, legend_kwargs={"loc": "lower left", "fontsize": 5.6})

# Right: the head-to-head at MATCHED error. A ratio above 1 favours the mass MAC.
ax = axes[1]
if comps:
    for dist_i, dist in enumerate(dists):
        for arm in sorted({c.get("mass_arm", "mass") for c in comps}):
            sel = sorted(
                (c for c in comps
                 if c["distribution"] == dist
                 and c.get("mass_arm", "mass") == arm
                 and c.get("pair_work_ratio")),
                key=lambda c: c["matched_p90"],
            )
            if not sel:
                continue
            ax.plot(
                [c["matched_p90"] for c in sel],
                [c["pair_work_ratio"] for c in sel],
                marker=style.MARKERS[dist_i % len(style.MARKERS)],
                linestyle=["-", "--", ":"][dist_i % 3],
                color=style.entity_color(arm),
                label=f"{arm} - {dist}",
                markersize=3.2,
            )
    ax.axhline(1.0, color=style.INK_MUTED, linewidth=0.8, zorder=1)
    ax.text(
        0.98, 1.0, " parity", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=6.5, color=style.INK_MUTED,
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("matched 90th-percentile error")
    ax.set_ylabel("work ratio, geometric / mass-dependent")
    style.finish(ax, legend_kwargs={"loc": "best", "fontsize": 6})
    ratios = [c["pair_work_ratio"] for c in comps if c.get("pair_work_ratio")]
    if ratios:
        print(f"work ratio at matched error: min {min(ratios):.2f} max {max(ratios):.2f}")
        print("(> 1 favours the mass-dependent MAC)")
else:
    ax.text(0.5, 0.5, "no matched-error overlap\\nin this sweep",
            ha="center", va="center", transform=ax.transAxes, color=style.INK_MUTED)
    ax.set_axis_off()

style.footer(
    fig,
    jsonio.config_caption(cfg, ["n", "order", "leaf_size", "precision", "device"]),,
)
fig.tight_layout()
style.save(fig, FIG_DIR / "fig03_mac_comparison.pdf")
"""

FIG03_CAPTION = """\
Geometric against mass-dependent (Dehnen eq. 16a) multipole acceptance, on
clustered distributions. **Left:** the raw trade-off -- hardware-independent pair
work against the 90th-percentile scaled force error, each arm swept over its own
accuracy knob ($\\theta$ for the geometric criterion, $\\epsilon$ for the
mass-dependent one). **Right:** the two arms log-interpolated onto a common error
and compared there; a ratio above the parity line would favour the
mass-dependent criterion. **As measured, the mass-dependent criterion gives no net
compute advantage at matched error.** The eq. (16b) arm supplies the exact
$O(N^2)$ force scale $f_b$ and is included as a ceiling on what a better force-scale
estimator could buy, not as a runnable configuration. Nothing here is tuned to
favour either arm. Values from `results/validation/mac_comparison.json`.
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
    suffix = f"  ($\\\\alpha={alpha:.2f}$)" if alpha and np.isfinite(alpha) else ""
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
style.finish(ax, legend_kwargs={"loc": "upper left", "fontsize": 6.2})
style.footer(
    fig,
    jsonio.config_caption(cfg, ["order", "theta", "basis", "preset", "precision", "device", "seed"])
    + f"   exponent fitted for N >= {cfg.get('fit_min_n')}",
)
fig.tight_layout()
style.save(fig, FIG_DIR / "fig04_wallclock_vs_n.pdf")

for name, fit in sorted(fits.items()):
    if name.startswith(set_name):
        print(f"{name:<45s} alpha={fit['exponent']:.3f} R2={fit['r_squared']:.4f} "
              f"n={fit['n_points']} over N={fit.get('fit_min_n')}..{fit.get('fit_max_n')}")
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
algorithm bound. Values from `results/scaling/wallclock_vs_n.json`.
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

style.footer(
    fig,
    jsonio.config_caption(cfg, [\"order\", \"theta\", \"basis\", \"leaf_size\", \"device\", \"seed\"])
    + f\"   exponent fitted for N >= {cfg.get('fit_min_n')}\",
)
fig.tight_layout()
style.save(fig, FIG_DIR / \"fig05_interaction_counts.pdf\")

for name, fit in sorted(fits.items()):
    print(f\"{name:<12s} alpha={fit['exponent']:.3f} R2={fit['r_squared']:.4f} \"
          f\"n={fit['n_points']} over N={fit.get('fit_min_n')}..{fit.get('fit_max_n')}\")
"""

FIG05_CAPTION = """\
Accepted M2L (far) and P2P (near) interaction counts against $N$, with fitted
power-law exponents. These are properties of the tree and the acceptance
criterion, independent of hardware, and so are the cleanest statement of the
method's complexity -- wall-clock (figure 4) additionally mixes in memory
bandwidth and kernel launch overhead. **Right:** the same counts per particle,
where linear scaling appears as a flat line. Counts are read from the solver's
public runtime diagnostics. Values from
`results/scaling/interaction_counts.json`.
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

ORDER = [
    ("tree_build", "tree build"),
    ("p2m_m2m", "P2M + M2M"),
    ("traversal_setup", "traversal setup"),
    ("m2l", "M2L"),
    ("l2l", "L2L"),
    ("nearfield_p2p", "near field (P2P)"),
    ("other_measured", "other measured"),
    ("unattributed", "unattributed"),
]
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
    if not normalise:
        ax.set_yscale("log")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    style.finish(ax, legend=normalise, legend_kwargs={"loc": "center left",
                                                     "bbox_to_anchor": (1.01, 0.5),
                                                     "fontsize": 6.4})

frac = [r.get("attributed_fraction") for r in recs if r.get("attributed_fraction")]
if frac:
    print(f"attributed fraction of wall clock: {min(frac):.2%} .. {max(frac):.2%}")
style.footer(
    fig,
    jsonio.config_caption(cfg, ["order", "theta", "basis", "preset", "leaf_size", "device"]),,
)
fig.tight_layout()
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
`results/scaling/stage_breakdown.json`.
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

skipped = [r["n"] for r in recs if r.get("cpu_skipped")]
if skipped:
    ax.text(
        0.02, 0.98,
        f"CPU arm not run above $N={min(skipped):,}$\\n(curve is not extrapolated)",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=6.5, color=style.INK_MUTED,
    )
style.footer(
    fig,
    jsonio.config_caption(cfg, ["order", "theta", "basis", "preset", "leaf_size", "precision"]),,
)
fig.tight_layout()
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
Values from `results/scaling/gpu_vs_cpu.json`.
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
style.footer(
    fig,
    jsonio.config_caption(cfg, ["order", "theta", "leaf_size", "precision", "device"]),,
)
fig.tight_layout()
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
`results/differentiability/autodiff_overhead.json`.
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

style.footer(
    fig,
    jsonio.config_caption(cfg, ["n", "order", "leaf_size", "precision", "device"])
    + f"\\nFD: {cfg['fd_samples']} coords, eps={cfg['fd_eps']:g}",,
)
fig.tight_layout()
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
`results/differentiability/grad_correctness.json`.
"""

SPECS = [
    (
        "fig_01_force_error_vs_order",
        "Figure 01 -- force error vs expansion order",
        "Loads `results/validation/force_error_vs_order.json`, produced by "
        "`bench/validation/force_error_vs_order.py`. No computation here.",
        FIG01,
        FIG01_CAPTION,
    ),
    (
        "fig_02_error_vs_theta",
        "Figure 02 -- force error vs opening angle",
        "Loads `results/validation/error_vs_theta.json`, produced by "
        "`bench/validation/error_vs_theta.py`. No computation here.",
        FIG02,
        FIG02_CAPTION,
    ),
    (
        "fig_03_mac_comparison",
        "Figure 03 -- geometric vs mass-dependent MAC",
        "Loads `results/validation/mac_comparison.json`, produced by "
        "`bench/validation/mac_comparison.py` (a wrapper over "
        "`bench/validation/mac_error_distribution.py`). No computation here.",
        FIG03,
        FIG03_CAPTION,
    ),
    (
        "fig_04_wallclock_vs_n",
        "Figure 04 -- wall-clock vs N",
        "Loads `results/scaling/wallclock_vs_n.json`, produced by "
        "`bench/scaling/wallclock.py`. No computation here.",
        FIG04,
        FIG04_CAPTION,
    ),
    (
        "fig_05_interaction_counts",
        "Figure 05 -- interaction counts vs N",
        "Loads `results/scaling/interaction_counts.json`, produced by "
        "`bench/scaling/interaction_counts.py`. No computation here.",
        FIG05,
        FIG05_CAPTION,
    ),
    (
        "fig_06_stage_breakdown",
        "Figure 06 -- per-stage time breakdown vs N",
        "Loads `results/scaling/stage_breakdown.json`, produced by "
        "`bench/scaling/stage_breakdown.py`. No computation here.",
        FIG06,
        FIG06_CAPTION,
    ),
    (
        "fig_07_gpu_vs_cpu",
        "Figure 07 -- single-GPU vs CPU speedup",
        "Loads `results/scaling/gpu_vs_cpu.json`, produced by "
        "`bench/scaling/gpu_vs_cpu_speedup.py`. No computation here.",
        FIG07,
        FIG07_CAPTION,
    ),
    (
        "fig_12_autodiff_overhead",
        "Figure 12 -- forward vs forward+backward cost",
        "Loads `results/differentiability/autodiff_overhead.json`, produced by "
        "`bench/differentiability/autodiff_overhead.py`. No computation here.",
        FIG12,
        FIG12_CAPTION,
    ),
    (
        "fig_13_grad_correctness",
        "Figure 13 -- finite-difference vs autodiff gradients",
        "Loads `results/differentiability/grad_correctness.json`, produced by "
        "`bench/differentiability/grad_correctness.py`. No computation here.",
        FIG13,
        FIG13_CAPTION,
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
