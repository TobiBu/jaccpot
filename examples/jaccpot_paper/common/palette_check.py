"""Colour-vision and print checks for the paper's figure palettes.

A Python port of the six-check palette validator, so the palette in
:mod:`style` is verified by computation rather than by taste, and so a unit test
can assert it stays verified. Thresholds, the OKLab transform, and the
Machado-Oliveira-Fernandes (2009) severity-1.0 CVD matrices are taken from that
validator unchanged -- moving any of them would decalibrate the ΔE gates.

Checks
------
1. **Lightness band** -- OKLCH L inside the mode's band.
2. **Chroma floor** -- OKLCH C at or above the floor, so no hue reads as gray.
3. **CVD separation** -- OKLab ΔE×100 under protanopia and deuteranopia
   simulation. ``>= 8`` passes, ``6-8`` is a floor legal only with secondary
   encoding (direct labels, texture), below 6 fails.
4. **Normal-vision floor** -- unsimulated ΔE×100 must clear 15. A hard gate:
   secondary encoding does not excuse it.
5. **Contrast vs surface** -- WCAG ratio; below 3:1 obligates visible labels.

Ordinal (sequential ramp) checks replace 1-4, because a correct ramp spans the
lightness band by construction and its light steps sit below the chroma floor:
monotone lightness with ``ΔL >= 0.06`` per step, and the lightest step clearing
2:1 against the surface.

Run it directly to print a report::

    python -m examples.jaccpot_paper.common.palette_check
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Optional, Sequence

__all__ = [
    "contrast",
    "delta_e",
    "oklch",
    "validate_categorical",
    "validate_ordinal",
]

# -- thresholds (keep in lockstep with the reference validator) --------------
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}  # OKLCH L
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0  # OKLab ΔE×100, min(protan, deutan), adjacent
NORMAL_FLOOR = 15.0  # OKLab ΔE×100, unsimulated, worst pair
CONTRAST_MIN = 3.0  # WCAG vs surface
ORDINAL_MIN_DL = 0.06  # min OKLCH ΔL between adjacent ramp steps
ORDINAL_LIGHT_FLOOR = 2.0  # lightest ramp step: WCAG contrast vs surface

MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def _hex_to_srgb(value: str) -> tuple[float, float, float]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {value!r}")
    return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin(value: str) -> tuple[float, float, float]:
    return tuple(_s2lin(c) for c in _hex_to_srgb(value))  # type: ignore[return-value]


def _rel_lum(value: str) -> float:
    r, g, b = _lin(value)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colours."""

    hi, lo = sorted((_rel_lum(a), _rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab_from_lin(rgb: Sequence[float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklch(value: str) -> tuple[float, float]:
    """Return ``(L, C)`` in OKLCH for a hex colour."""

    _l, a, b = _oklab_from_lin(_lin(value))
    return _l, math.hypot(a, b)


def _simulate(value: str, kind: str) -> tuple[float, float, float]:
    r, g, b = _lin(value)
    m = MACHADO[kind]
    return tuple(  # type: ignore[return-value]
        min(1.0, max(0.0, row[0] * r + row[1] * g + row[2] * b)) for row in m
    )


def delta_e(a: str, b: str, kind: Optional[str] = None) -> float:
    """Euclidean OKLab distance ×100; ``kind=None`` is unsimulated vision."""

    pa = _oklab_from_lin(_simulate(a, kind) if kind else _lin(a))
    pb = _oklab_from_lin(_simulate(b, kind) if kind else _lin(b))
    return 100.0 * math.dist(pa, pb)


def validate_categorical(
    palette: Sequence[str],
    *,
    mode: Literal["light", "dark"] = "light",
    surface: str = "#ffffff",
    pairs: Literal["adjacent", "all"] = "adjacent",
) -> dict:
    """Run checks 1-5 on a categorical palette.

    ``pairs="all"`` is the right setting for scatter and small multiples, where
    any two series can end up side by side; ``"adjacent"`` suffices for lines and
    stacked bars, where only neighbours touch.
    """

    lo, hi = BAND[mode]
    report: list[tuple[str, str, str]] = []
    ok = True

    offband = [
        (c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi
    ]
    if offband:
        ok = False
    report.append(
        (
            "Lightness band",
            "FAIL" if offband else "PASS",
            (
                f"outside L {lo}-{hi}: {offband}"
                if offband
                else f"all {len(palette)} inside L {lo}-{hi}"
            ),
        )
    )

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    if lowc:
        ok = False
    report.append(
        (
            "Chroma floor",
            "FAIL" if lowc else "PASS",
            f"reads gray: {lowc}" if lowc else f"all {len(palette)} >= {CHROMA_FLOOR}",
        )
    )

    n = len(palette)
    if pairs == "all":
        pairlist = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairlist = [(i, i + 1) for i in range(n - 1)]

    worst: Optional[tuple[float, str, str, str]] = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    tritan = (
        min(delta_e(palette[i], palette[j], "tritan") for i, j in pairlist)
        if pairlist
        else 99.0
    )
    wd = worst[0] if worst else 99.0
    cvd_state = "PASS" if wd >= CVD_TARGET else ("WARN" if wd >= CVD_FLOOR else "FAIL")
    if cvd_state == "FAIL":
        ok = False
    report.append(
        (
            "CVD separation",
            cvd_state,
            (
                f"worst {pairs} {worst[3]}<->{worst[2]} dE {wd:.1f} ({worst[1]}) "
                f"- tritan {tritan:.1f}"
                if worst
                else "n/a"
            ),
        )
    )

    nworst: Optional[tuple[float, str, str]] = None
    for i, j in pairlist:
        d = delta_e(palette[i], palette[j])
        if nworst is None or d < nworst[0]:
            nworst = (d, palette[i], palette[j])
    nd = nworst[0] if nworst else 99.0
    if nd < NORMAL_FLOOR:
        ok = False
    report.append(
        (
            "Normal-vision floor",
            "PASS" if nd >= NORMAL_FLOOR else "FAIL",
            (
                f"worst {nworst[2]}<->{nworst[1]} dE {nd:.1f} (floor {NORMAL_FLOOR})"
                if nworst
                else "n/a"
            ),
        )
    )

    low = [
        (c, round(contrast(c, surface), 2))
        for c in palette
        if contrast(c, surface) < CONTRAST_MIN
    ]
    report.append(
        (
            "Contrast vs surface",
            "WARN" if low else "PASS",
            (
                f"below {CONTRAST_MIN}:1, needs visible labels: {low}"
                if low
                else f"all {len(palette)} >= {CONTRAST_MIN}:1"
            ),
        )
    )

    return {"ok": ok, "report": report, "worst_cvd": wd, "worst_normal": nd}


def validate_ordinal(ramp: Sequence[str], *, surface: str = "#ffffff") -> dict:
    """Run the sequential-ramp checks: monotone lightness and light-end contrast."""

    report: list[tuple[str, str, str]] = []
    ok = True
    lightness = [oklch(c)[0] for c in ramp]

    ascending = all(b > a for a, b in zip(lightness, lightness[1:]))
    descending = all(b < a for a, b in zip(lightness, lightness[1:]))
    monotone = ascending or descending
    if not monotone:
        ok = False
    report.append(
        (
            "Lightness monotonic",
            "PASS" if monotone else "FAIL",
            f"L = {[round(v, 3) for v in lightness]}",
        )
    )

    steps = [abs(b - a) for a, b in zip(lightness, lightness[1:])]
    worst_dl = min(steps) if steps else 99.0
    if worst_dl < ORDINAL_MIN_DL:
        ok = False
    report.append(
        (
            "Step separation",
            "PASS" if worst_dl >= ORDINAL_MIN_DL else "FAIL",
            f"min dL {worst_dl:.3f} (floor {ORDINAL_MIN_DL})",
        )
    )

    lightest = max(ramp, key=lambda c: oklch(c)[0])
    cr = contrast(lightest, surface)
    if cr < ORDINAL_LIGHT_FLOOR:
        ok = False
    report.append(
        (
            "Light-end contrast",
            "PASS" if cr >= ORDINAL_LIGHT_FLOOR else "FAIL",
            f"{lightest} vs {surface}: {cr:.2f}:1 (floor {ORDINAL_LIGHT_FLOOR})",
        )
    )
    return {"ok": ok, "report": report}


def format_report(name: str, result: dict) -> str:
    lines = [f"{name}: {'ALL CHECKS PASS' if result['ok'] else 'FAILED'}"]
    for check, state, detail in result["report"]:
        lines.append(f"  {state:<5} {check:<22} {detail}")
    return "\n".join(lines)


def _main() -> int:
    from . import style

    results = {
        "categorical (adjacent)": validate_categorical(style.CATEGORICAL),
        "categorical (all pairs)": validate_categorical(style.CATEGORICAL, pairs="all"),
        "sequential ramp": validate_ordinal(style.SEQUENTIAL),
    }
    failed = False
    for name, result in results.items():
        print(format_report(name, result))
        failed = failed or not result["ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
