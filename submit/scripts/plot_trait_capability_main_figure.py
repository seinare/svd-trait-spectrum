#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "results" / "trait_capability_main_figure"
ALPHAS = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
MODELS = {
    "llama32_1b_instruct": "Llama 1B",
    "llama32_3b_instruct": "Llama 3B",
    "llama31_8b_instruct": "Llama 8B",
    "qwen3_8b": "Qwen 8B",
}
CAP_MODEL = {
    "llama32_1b_instruct": "llama1b",
    "llama32_3b_instruct": "llama3b",
    "llama31_8b_instruct": "llama8b",
    "qwen3_8b": "qwen3_8b",
}
TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
]
CAPS = ["Factual Knowledge", "Language Understanding", "Deductive Reasoning"]
CAP_SHORT = {
    "Factual Knowledge": "Factual",
    "Language Understanding": "Language",
    "Deductive Reasoning": "Deductive",
}
COL_POS = "#d7191c"
COL_NEG = "#2c7bb6"
COL_GRAY = "#5f6368"
COL_LIGHT = "#eef1f4"
MODEL_MARK = ["circle", "square", "triangle", "diamond"]


def alpha_label(alpha: float) -> str:
    if abs(alpha) < 1e-12:
        return "0"
    return f"{alpha:.2f}".rstrip("0").rstrip(".")


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def sem(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def piecewise_slopes(points: dict[float, float]) -> tuple[float, float]:
    pos = [(a, points[a]) for a in ALPHAS if a > 0 and a in points]
    neg = [(-a, points[a]) for a in ALPHAS if a < 0 and a in points]
    b_pos = sum(x * y for x, y in pos) / sum(x * x for x, _y in pos) if pos else 0.0
    b_neg = sum(x * y for x, y in neg) / sum(x * x for x, _y in neg) if neg else 0.0
    return b_pos, b_neg


def read_trait_scores() -> dict[tuple[str, str], dict[float, float]]:
    out: dict[tuple[str, str], dict[float, float]] = {}
    for model in MODELS:
        path = ROOT / "docs/results/trait_alpha9_tables" / f"{model}_trait_alpha9_table.csv"
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                trait = row["trait"]
                out[(model, trait)] = {a: float(row[alpha_label(a)]) for a in ALPHAS if row.get(alpha_label(a))}
    return out


def read_capabilities() -> dict[tuple[str, str], dict[float, float]]:
    out: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    path = ROOT / "docs/results/eval6_all_models_alpha9_capability/capability_alpha_mle_errorbar_points.csv"
    wanted = set(CAP_MODEL.values())
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            dim = row["dimension"]
            if model in wanted and dim in CAPS:
                out[(model, dim)][round(float(row["alpha"]), 2)] = float(row["mean_mle"])
    return dict(out)


def read_paths() -> list[dict[str, object]]:
    rows = []
    path = ROOT / "docs/results/trait_capability_correlation/pooled_trait_on_capability_regression.csv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["term"] in CAPS:
                beta = float(row["standardized_beta"])
                if abs(beta) >= 0.30:
                    rows.append(
                        {
                            "trait": row["trait"],
                            "cap": row["term"],
                            "beta": beta,
                            "t": float(row["t_approx"]) if row["t_approx"] else 0.0,
                            "r2": float(row["r2"]),
                        }
                    )
    return rows


def read_task_weights() -> list[dict[str, object]]:
    rows = {}
    path = ROOT / "docs/results/eval6_all_models_alpha9_capability/task_capability_weight_pca2d.csv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["module"], row["task"])
            if key in rows:
                continue
            rows[key] = {
                "module": row["module"],
                "task": row["task"],
                "f": float(row["Factual Knowledge"]),
                "l": float(row["Language Understanding"]),
                "d": float(row["Deductive Reasoning"]),
            }
    return list(rows.values())


class SVG:
    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            "<defs>",
            '<marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#d7191c"/></marker>',
            '<marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#2c7bb6"/></marker>',
            "</defs>",
        ]

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, text: str, size: int = 12, weight: str = "400", anchor: str = "start", color: str = "#111", rotate: float | None = None) -> None:
        t = html.escape(text)
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}"{transform}>{t}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#333", width: float = 1.0, dash: str | None = None, marker: str | None = None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.2f}" fill="none"{d}{m}/>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none", stroke: str = "#222", width: float = 1.2, rx: float = 5) -> None:
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}"/>')

    def ellipse(self, x: float, y: float, w: float, h: float, fill: str = "white", stroke: str = "#222", width: float = 1.2) -> None:
        self.add(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{w/2:.1f}" ry="{h/2:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}"/>')

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "none", opacity: float = 1.0) -> None:
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.2f}"/>')

    def polyline(self, pts: list[tuple[float, float]], fill: str = "none", stroke: str = "#333", width: float = 1) -> None:
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.add(f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}"/>')

    def polygon(self, pts: list[tuple[float, float]], fill: str, stroke: str = "none", opacity: float = 1.0) -> None:
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.add(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.2f}"/>')

    def path(self, d: str, fill: str = "none", stroke: str = "#333", width: float = 1.0, opacity: float = 1.0) -> None:
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.2f}"/>')

    def marker(self, x: float, y: float, kind: str, fill: str, size: float = 5) -> None:
        if kind == "circle":
            self.circle(x, y, size, fill, "#333", 0.85)
        elif kind == "square":
            self.rect(x - size, y - size, size * 2, size * 2, fill, "#333", 0.6, 1)
        elif kind == "triangle":
            self.polygon([(x, y - size * 1.2), (x - size, y + size), (x + size, y + size)], fill, "#333", 0.85)
        else:
            self.polygon([(x, y - size), (x - size, y), (x, y + size), (x + size, y)], fill, "#333", 0.85)

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts))


def scale(v: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if abs(hi - lo) < 1e-12:
        return (out_lo + out_hi) / 2
    return out_lo + (v - lo) * (out_hi - out_lo) / (hi - lo)


def y_from_value(v: float, lo: float, hi: float, top: float, bottom: float) -> float:
    return scale(v, lo, hi, bottom, top)


def panel_a(svg: SVG, x: int, y: int, w: int, h: int, cap_slopes: dict[tuple[str, str], tuple[float, float]]) -> None:
    svg.text(x, y, "a", 24, "700")
    svg.text(x + 35, y, "alpha direction -> capability response", 17, "700")
    left, top, right, bottom = x + 55, y + 35, x + w - 20, y + h - 50
    vals = [v for pair in cap_slopes.values() for v in pair]
    lo, hi = min(vals + [-0.01]), max(vals + [0.01])
    m = max(abs(lo), abs(hi))
    lo, hi = -m * 1.15, m * 1.15
    svg.line(left, bottom, right, bottom, "#333", 1)
    svg.line(left, top, left, bottom, "#333", 1)
    z = y_from_value(0, lo, hi, top, bottom)
    svg.line(left, z, right, z, "#777", 0.8, "3,3")
    for tick in [-0.4, -0.2, 0, 0.2, 0.4]:
        if lo <= tick <= hi:
            yy = y_from_value(tick, lo, hi, top, bottom)
            svg.line(left - 4, yy, left, yy, "#333", 1)
            svg.text(left - 8, yy + 4, f"{tick:.1f}", 10, anchor="end")
    svg.text(left - 40, (top + bottom) / 2, "slope", 12, anchor="middle", rotate=-90)
    group_w = (right - left) / len(CAPS)
    bar_w = 20
    for i, cap in enumerate(CAPS):
        cx = left + group_w * (i + 0.5)
        pos_vals = [cap_slopes[(m, cap)][0] for m in MODELS]
        neg_vals = [cap_slopes[(m, cap)][1] for m in MODELS]
        for off, vals2, color in [(-13, pos_vals, COL_POS), (13, neg_vals, COL_NEG)]:
            mv = mean(vals2)
            yy = y_from_value(mv, lo, hi, top, bottom)
            y0 = y_from_value(0, lo, hi, top, bottom)
            svg.rect(cx + off - bar_w / 2, min(yy, y0), bar_w, abs(y0 - yy), color, "none", 0, 0)
            se = sem(vals2)
            ylo = y_from_value(mv - se, lo, hi, top, bottom)
            yhi = y_from_value(mv + se, lo, hi, top, bottom)
            svg.line(cx + off, ylo, cx + off, yhi, "#333", 0.8)
            for j, model in enumerate(MODELS):
                px = cx + off + (j - 1.5) * 4
                py = y_from_value(vals2[j], lo, hi, top, bottom)
                svg.marker(px, py, MODEL_MARK[j], "#ffffff", 3.2)
        svg.text(cx, bottom + 20, CAP_SHORT[cap], 12, anchor="middle")
    svg.rect(right - 95, top + 4, 10, 10, COL_POS, "none", 0, 0)
    svg.text(right - 80, top + 14, "positive", 11)
    svg.rect(right - 95, top + 22, 10, 10, COL_NEG, "none", 0, 0)
    svg.text(right - 80, top + 32, "negative", 11)


def panel_b(svg: SVG, x: int, y: int, w: int, h: int, trait_slopes: dict[tuple[str, str], tuple[float, float]]) -> None:
    svg.text(x, y, "b", 24, "700")
    svg.text(x + 35, y, "alpha direction -> TRAIT response", 17, "700")
    left, top, right, bottom = x + 115, y + 30, x + w - 25, y + h - 28
    trait_order = sorted(
        TRAITS,
        key=lambda t: abs(mean([trait_slopes[(m, t)][0] for m in MODELS])) + abs(mean([trait_slopes[(m, t)][1] for m in MODELS])),
        reverse=True,
    )
    vals = [v for pair in trait_slopes.values() for v in pair]
    m = max(abs(min(vals)), abs(max(vals)), 0.01) * 1.12
    lo, hi = -m, m
    zero = scale(0, lo, hi, left, right)
    svg.line(zero, top, zero, bottom, "#777", 0.8, "3,3")
    svg.line(left, bottom, right, bottom, "#333", 1)
    for tick in [-0.2, -0.1, 0, 0.1, 0.2]:
        if lo <= tick <= hi:
            xx = scale(tick, lo, hi, left, right)
            svg.line(xx, bottom, xx, bottom + 4, "#333", 1)
            svg.text(xx, bottom + 18, f"{tick:.1f}", 10, anchor="middle")
    row_h = (bottom - top) / len(trait_order)
    for i, trait in enumerate(trait_order):
        cy = top + row_h * (i + 0.5)
        svg.text(left - 8, cy + 4, trait, 11, anchor="end")
        pos_vals = [trait_slopes[(m, trait)][0] for m in MODELS]
        neg_vals = [trait_slopes[(m, trait)][1] for m in MODELS]
        for dy, vals2, color in [(-5, pos_vals, COL_POS), (5, neg_vals, COL_NEG)]:
            mv = mean(vals2)
            xx = scale(mv, lo, hi, left, right)
            svg.rect(min(zero, xx), cy + dy - 3.2, abs(xx - zero), 6.4, color, "none", 0, 0)
            svg.circle(xx, cy + dy, 4.2, color, "none", 0.95)
            for j, val in enumerate(vals2):
                svg.marker(scale(val, lo, hi, left, right), cy + dy + (j - 1.5) * 1.5, MODEL_MARK[j], "#ffffff", 2.7)
    svg.text((left + right) / 2, bottom + 34, "slope", 12, anchor="middle")


def panel_c(svg: SVG, x: int, y: int, w: int, h: int, paths: list[dict[str, object]]) -> None:
    svg.text(x, y, "c", 24, "700")
    svg.text(x + 35, y, "capability response -> TRAIT response", 17, "700")
    coeff = {(p["cap"], p["trait"]): float(p["beta"]) for p in paths}
    r2 = {p["trait"]: float(p["r2"]) for p in paths}
    # Include all columns but emphasize strong cells; this reads as a response-path coefficient matrix.
    all_coeff: dict[tuple[str, str], float] = {}
    path_csv = ROOT / "docs/results/trait_capability_correlation/pooled_trait_on_capability_regression.csv"
    with path_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["term"] in CAPS:
                all_coeff[(row["term"], row["trait"])] = float(row["standardized_beta"])
                r2[row["trait"]] = float(row["r2"])
    left, top = x + 135, y + 54
    cell_w, cell_h = 66, 58
    trait_order = ["Agreeableness", "Conscientiousness", "Extraversion", "Openness", "Narcissism", "Machiavellianism", "Neuroticism", "Psychopathy"]
    max_abs = max(abs(v) for v in all_coeff.values())
    def blend(color: str, strength: float) -> str:
        color = color.lstrip("#")
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        rr = round(255 - (255 - r) * strength)
        gg = round(255 - (255 - g) * strength)
        bb = round(255 - (255 - b) * strength)
        return f"#{rr:02x}{gg:02x}{bb:02x}"
    for j, trait in enumerate(trait_order):
        cx = left + j * cell_w + cell_w / 2
        label = trait if len(trait) <= 12 else trait[:11] + "."
        svg.text(cx, top - 8, label, 10, "600", anchor="middle", rotate=-35)
        svg.text(cx, top + len(CAPS) * cell_h + 26, f"{r2.get(trait, 0):.2f}", 10, anchor="middle", color="#555")
    svg.text(left - 48, top + len(CAPS) * cell_h + 26, "R²", 10, "600", anchor="end", color="#555")
    for i, cap in enumerate(CAPS):
        cy = top + i * cell_h + cell_h / 2
        svg.text(left - 18, cy + 4, CAP_SHORT[cap], 11, "600", anchor="end")
        for j, trait in enumerate(trait_order):
            beta = all_coeff[(cap, trait)]
            strength = min(abs(beta) / max_abs, 1.0)
            fill = blend(COL_POS if beta > 0 else COL_NEG, 0.18 + 0.72 * strength)
            stroke = "#111" if abs(beta) >= 0.30 else "#d0d0d0"
            sw = 1.2 if abs(beta) >= 0.30 else 0.5
            xx, yy = left + j * cell_w, top + i * cell_h
            svg.rect(xx, yy, cell_w - 4, cell_h - 4, fill, stroke, sw, 2)
            svg.text(xx + cell_w / 2 - 2, yy + cell_h / 2 + 4, f"{beta:+.2f}".replace("+", ""), 10, "700" if abs(beta) >= 0.30 else "400", anchor="middle")
    svg.rect(x + w - 116, y + h - 42, 16, 10, COL_POS, "none", 0, 0)
    svg.text(x + w - 96, y + h - 33, "positive beta", 10)
    svg.rect(x + w - 116, y + h - 24, 16, 10, COL_NEG, "none", 0, 0)
    svg.text(x + w - 96, y + h - 15, "negative beta", 10)
    svg.text(x + 35, y + h - 18, "Cells are standardized beta; dark borders mark |beta| >= 0.30.", 10, color="#555")


def panel_d(svg: SVG, x: int, y: int, w: int, h: int, tasks: list[dict[str, object]]) -> None:
    svg.text(x, y, "d", 24, "700")
    svg.text(x + 35, y, "task subitems in capability-demand space", 17, "700")
    top = y + 36
    size = min(w - 90, h - 75)
    x0, y0 = x + 60, top + size * 0.82
    fpt = (x0, y0)
    lpt = (x0 + size, y0)
    dpt = (x0 + size / 2, y0 - size * 0.866)
    svg.polygon([fpt, lpt, dpt], "#ffffff", "#333", 1.0)
    for t in [0.25, 0.5, 0.75]:
        # Iso-weight guide lines in barycentric coordinates.
        def xy(f: float, l: float, d: float) -> tuple[float, float]:
            return (f * fpt[0] + l * lpt[0] + d * dpt[0], f * fpt[1] + l * lpt[1] + d * dpt[1])
        svg.line(*xy(t, 1 - t, 0), *xy(t, 0, 1 - t), "#d3d7db", 0.6)
        svg.line(*xy(1 - t, t, 0), *xy(0, t, 1 - t), "#d3d7db", 0.6)
        svg.line(*xy(1 - t, 0, t), *xy(0, 1 - t, t), "#d3d7db", 0.6)
    colors = {"mmlu_pro": "#4c78a8", "mmlu_redux": "#f58518", "agieval": "#54a24b", "bbh": "#b279a2"}
    for r in tasks:
        f, l, d = float(r["f"]), float(r["l"]), float(r["d"])
        px = f * fpt[0] + l * lpt[0] + d * dpt[0]
        py = f * fpt[1] + l * lpt[1] + d * dpt[1]
        svg.circle(px, py, 3.5, colors.get(str(r["module"]), "#888"), "none", 0.65)
    svg.text(fpt[0] - 8, fpt[1] + 20, "Factual", 12, "600", anchor="middle")
    svg.text(lpt[0] + 8, lpt[1] + 20, "Language", 12, "600", anchor="middle")
    svg.text(dpt[0], dpt[1] - 10, "Deductive", 12, "600", anchor="middle")
    lx, ly = x + w - 110, y + h - 88
    for i, (mod, color) in enumerate(colors.items()):
        svg.circle(lx, ly + i * 18, 4, color, "none", 0.85)
        svg.text(lx + 10, ly + i * 18 + 4, mod.replace("_", "-"), 10)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trait_scores = read_trait_scores()
    caps = read_capabilities()

    trait_slopes: dict[tuple[str, str], tuple[float, float]] = {}
    for model in MODELS:
        for trait in TRAITS:
            base = trait_scores[(model, trait)][0.0]
            points = {a: trait_scores[(model, trait)][a] - base for a in ALPHAS}
            trait_slopes[(model, trait)] = piecewise_slopes(points)

    cap_slopes: dict[tuple[str, str], tuple[float, float]] = {}
    for trait_model, cap_model in CAP_MODEL.items():
        for cap in CAPS:
            points = caps[(cap_model, cap)]
            cap_slopes[(trait_model, cap)] = piecewise_slopes(points)

    with (OUT_DIR / "panel_a_capability_direction_slopes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "capability", "positive_slope", "negative_slope", "asymmetry"])
        writer.writeheader()
        for (model, cap), (pos, neg) in cap_slopes.items():
            writer.writerow({"model": model, "capability": cap, "positive_slope": pos, "negative_slope": neg, "asymmetry": pos - neg})

    with (OUT_DIR / "panel_b_trait_direction_slopes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "trait", "positive_slope", "negative_slope", "asymmetry"])
        writer.writeheader()
        for (model, trait), (pos, neg) in trait_slopes.items():
            writer.writerow({"model": model, "trait": trait, "positive_slope": pos, "negative_slope": neg, "asymmetry": pos - neg})

    svg = SVG(1500, 1050)
    svg.rect(0, 0, 1500, 1050, "#ffffff", "none", 0, 0)
    svg.text(40, 38, "Directional spectral perturbations jointly reshape capability and TRAIT response profiles", 23, "700")
    svg.text(40, 62, "Four dense instruct models with complete nine-point TRAIT data; red = positive alpha, blue = negative alpha.", 13, color="#555")
    panel_a(svg, 45, 105, 640, 365, cap_slopes)
    panel_b(svg, 765, 105, 670, 365, trait_slopes)
    panel_c(svg, 45, 545, 720, 430, read_paths())
    panel_d(svg, 815, 545, 620, 430, read_task_weights())
    # Model marker legend.
    lx, ly = 1045, 78
    for i, (model, label) in enumerate(MODELS.items()):
        svg.marker(lx + i * 105, ly, MODEL_MARK[i], "#ffffff", 4)
        svg.text(lx + i * 105 + 10, ly + 4, label, 10)
    out_svg = OUT_DIR / "trait_capability_main_figure.svg"
    svg.save(out_svg)

    report = [
        "# TRAIT-Capability Main Figure Draft",
        "",
        "This figure uses four dense instruct models with complete nine-point TRAIT data: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, and Qwen3-8B.",
        "",
        "- Panel a: piecewise directional slopes from alpha to the three fitted capability responses.",
        "- Panel b: piecewise directional slopes from alpha to TRAIT responses.",
        "- Panel c: pooled standardized capability-to-TRAIT response paths with model fixed effects; only paths with |beta| >= 0.30 are drawn.",
        "- Panel d: task subitems projected into the three-dimensional DS-judge capability demand simplex.",
        "",
        "The diagram should be described as a perturbation-response association model, not a causal SEM.",
    ]
    (OUT_DIR / "trait_capability_main_figure_notes.md").write_text("\n".join(report))
    print(out_svg)


if __name__ == "__main__":
    main()
