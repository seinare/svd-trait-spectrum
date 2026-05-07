#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Rectangle
from matplotlib.patches import PathPatch

from plot_trait_capability_main_figure import (
    ALPHAS,
    CAP_MODEL,
    CAP_SHORT,
    CAPS,
    COL_NEG,
    COL_POS,
    MODELS,
    OUT_DIR,
    ROOT,
    TRAITS,
    mean,
    piecewise_slopes,
    read_capabilities,
    read_task_weights,
    read_trait_scores,
)
from plot_trait_capability_main_figure_v2 import clean_task_label


TRAIT_SHORT = {
    "Openness": "Openness",
    "Conscientiousness": "Conscientious.",
    "Extraversion": "Extraversion",
    "Agreeableness": "Agreeableness",
    "Neuroticism": "Neuroticism",
    "Machiavellianism": "Machiavell.",
    "Narcissism": "Narcissism",
    "Psychopathy": "Psychopathy",
}
SEGMENTS = [
    (-0.2, -0.1, "[-.2,-.1]"),
    (-0.1, 0.0, "[-.1,0]"),
    (0.0, 0.1, "[0,.1]"),
    (0.1, 0.2, "[.1,.2]"),
]
CAP_TEXT = {
    "Factual Knowledge": "#238b45",
    "Language Understanding": "#2c7bb6",
    "Deductive Reasoning": "#6a51a3",
}
EVIL_TRAITS = {"Machiavellianism", "Narcissism", "Psychopathy"}
TRAIT_EVIL = "#8b1a1a"
TRAIT_OTHER = "#6baed6"


def segment_slope(points: dict[float, float], lo: float, hi: float) -> float:
    xs = [a for a in ALPHAS if lo - 1e-9 <= a <= hi + 1e-9 and a in points]
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    ys = [points[a] for a in xs]
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0


def segment_slopes(points: dict[float, float]) -> list[float]:
    return [segment_slope(points, lo, hi) for lo, hi, _label in SEGMENTS]


def build_cap_slopes() -> dict[tuple[str, str], tuple[float, float]]:
    caps = read_capabilities()
    out = {}
    for trait_model, cap_model in CAP_MODEL.items():
        for cap in CAPS:
            out[(trait_model, cap)] = piecewise_slopes(caps[(cap_model, cap)])
    return out


def build_cap_segment_slopes() -> dict[tuple[str, str], list[float]]:
    caps = read_capabilities()
    out = {}
    for trait_model, cap_model in CAP_MODEL.items():
        for cap in CAPS:
            out[(trait_model, cap)] = segment_slopes(caps[(cap_model, cap)])
    return out


def build_trait_segment_slopes() -> dict[tuple[str, str], list[float]]:
    scores = read_trait_scores()
    out = {}
    for model in MODELS:
        for trait in TRAITS:
            base = scores[(model, trait)][0.0]
            out[(model, trait)] = segment_slopes({a: scores[(model, trait)][a] - base for a in ALPHAS})
    return out


def build_trait_slopes() -> dict[tuple[str, str], tuple[float, float]]:
    scores = read_trait_scores()
    out = {}
    for model in MODELS:
        for trait in TRAITS:
            base = scores[(model, trait)][0.0]
            points = {a: scores[(model, trait)][a] - base for a in ALPHAS}
            out[(model, trait)] = piecewise_slopes(points)
    return out


def read_betas() -> dict[tuple[str, str], float]:
    path = ROOT / "docs/results/trait_capability_correlation/trait_capability_pearson.csv"
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["dimension"] in CAPS:
                out[(row["dimension"], row["trait"])] = float(row["pearson_r"])
    return out


def read_subtask_slopes() -> dict[tuple[str, str], tuple[float, float]]:
    raw: dict[tuple[str, str, str], dict[float, float]] = defaultdict(dict)
    path = ROOT / "docs/results/eval6_all_models_alpha9_capability/eval6_all_models_subtask_scores_long.csv"
    wanted = set(CAP_MODEL.values())
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] not in wanted:
                continue
            raw[(row["model"], row["module"], row["task"])][round(float(row["alpha"]), 2)] = float(row["value"])

    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (_model, module, task), vals in raw.items():
        if 0.0 not in vals:
            continue
        points = {a: vals[a] - vals[0.0] for a in ALPHAS if a in vals}
        if len(points) >= 5:
            grouped[(module, task)].append(piecewise_slopes(points))
    return {key: (mean([p for p, _ in pairs]), mean([n for _, n in pairs])) for key, pairs in grouped.items()}


def read_subtask_segment_slopes() -> dict[tuple[str, str], list[float]]:
    raw: dict[tuple[str, str, str], dict[float, float]] = defaultdict(dict)
    path = ROOT / "docs/results/eval6_all_models_alpha9_capability/eval6_all_models_subtask_scores_long.csv"
    wanted = set(CAP_MODEL.values())
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] not in wanted:
                continue
            raw[(row["model"], row["module"], row["task"])][round(float(row["alpha"]), 2)] = float(row["value"])

    grouped: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for (_model, module, task), vals in raw.items():
        if 0.0 not in vals:
            continue
        points = {a: vals[a] - vals[0.0] for a in ALPHAS if a in vals}
        if len(points) >= 5:
            grouped[(module, task)].append(segment_slopes(points))
    return {
        key: [mean([vals[i] for vals in pairs]) for i in range(len(SEGMENTS))]
        for key, pairs in grouped.items()
        if pairs
    }


def select_subtasks() -> list[dict[str, object]]:
    weights = {(r["module"], r["task"]): r for r in read_task_weights()}
    slopes = read_subtask_slopes()
    buckets: dict[str, list[tuple[float, tuple[str, str], tuple[float, float], dict[str, object]]]] = defaultdict(list)
    for key, w in weights.items():
        if key not in slopes:
            continue
        cap_weights = {
            "Factual Knowledge": float(w["f"]),
            "Language Understanding": float(w["l"]),
            "Deductive Reasoning": float(w["d"]),
        }
        dominant = max(cap_weights, key=cap_weights.get)
        score = max(abs(slopes[key][0]), abs(slopes[key][1])) + 0.20 * cap_weights[dominant]
        buckets[dominant].append((score, key, slopes[key], w))

    selected = []
    for cap in CAPS:
        for _score, key, pair, w in sorted(buckets[cap], reverse=True)[:4]:
            selected.append(
                {
                    "module": key[0],
                    "task": key[1],
                    "pos": pair[0],
                    "neg": pair[1],
                    "dominant_cap": cap,
                    "dominant_weight": max(float(w["f"]), float(w["l"]), float(w["d"])),
                }
            )
    return selected


def fmt(v: float) -> str:
    return f"{v:+.2f}".replace("+", "")


def side_response(vals: list[float], side: str) -> float:
    if side == "neg":
        return mean(vals[:2])
    return mean(vals[2:])


def module_label(module: str, task: str) -> str:
    return f"{module.replace('_', '-')}: {clean_task_label(task)}"


def draw_node(
    ax,
    xy,
    text,
    kind,
    width,
    height,
    edge="#222",
    face="white",
    fontsize=10,
    wrap=True,
    text_color="#111",
    zorder=10,
    face_alpha=1.0,
    pad=0.014,
    rounding=0.018,
):
    x, y = xy
    face_rgba = face
    if kind in {"circle", "ellipse"}:
        patch = Ellipse((x, y), width, height, facecolor=face_rgba, edgecolor=edge, lw=1.2, zorder=zorder, alpha=face_alpha)
    elif kind == "round":
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle=f"round,pad={pad},rounding_size={rounding}",
            facecolor=face_rgba,
            edgecolor=edge,
            lw=1.1,
            zorder=zorder,
            alpha=face_alpha,
        )
    else:
        patch = Rectangle((x - width / 2, y - height / 2), width, height, facecolor=face_rgba, edgecolor=edge, lw=1.0, zorder=zorder, alpha=face_alpha)
    ax.add_patch(patch)
    if wrap and len(text) > 17:
        words = text.split()
        mid = max(1, len(words) // 2)
        text = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight="semibold", color=text_color, zorder=zorder + 1)


def draw_edge(ax, start, end, color, lw, dashed, label=None, alpha=0.7, rad=0.0, label_t=0.5, label_dy=0.0, arrows=True):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>" if arrows else "-",
        mutation_scale=8 if arrows else 1,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        lw=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(patch)
    if label:
        x = start[0] + (end[0] - start[0]) * label_t
        y = start[1] + (end[1] - start[1]) * label_t + label_dy
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
            zorder=8,
        )


def draw_smooth_response_edge(
    ax,
    start,
    end,
    color,
    lw,
    dashed=True,
    label=None,
    alpha=0.7,
    label_t=0.5,
    label_dy=0.0,
    curve=0.48,
    label_alpha=0.92,
    label_z=8,
    zorder=3,
):
    """Response edge with horizontal out/in tangents and a smooth cubic middle."""
    sx, sy = start
    ex, ey = end
    dx = max((ex - sx) * curve, 0.08)
    verts = [
        (sx, sy),
        (sx + dx, sy),
        (ex - dx, ey),
        (ex, ey),
    ]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        lw=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    arrow_from = cubic_point(verts[0], verts[1], verts[2], verts[3], 0.985)
    ax.annotate(
        "",
        xy=end,
        xytext=arrow_from,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.0, "alpha": alpha, "mutation_scale": 8},
        zorder=zorder,
    )
    if label:
        lx, ly = cubic_point(verts[0], verts[1], verts[2], verts[3], label_t)
        ax.text(
            lx,
            ly + label_dy,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color=color,
            alpha=label_alpha,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": max(0.15, label_alpha * 0.85)},
            zorder=label_z,
        )


def draw_routed_edge(ax, start, end, color, lw, dashed, label=None, alpha=0.22, route="top", label_t=0.78, label_alpha=0.90, label_z=12):
    """Route a cross-layer edge around the middle layer through the outer margin."""
    sx, sy = start
    ex, ey = end
    outer_y = 0.86 if route == "top" else 0.16
    verts = [
        (sx, sy),
        (sx + 0.10, outer_y),
        (ex - 0.16, outer_y),
        (ex, ey),
    ]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        lw=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.annotate(
        "",
        xy=end,
        xytext=(ex - 0.012, ey),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.0, "alpha": alpha, "mutation_scale": 8},
        zorder=2,
    )
    if label:
        x = sx + (ex - sx) * label_t
        y = outer_y * 0.70 + ey * 0.30
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            weight="bold",
            color=color,
            alpha=label_alpha,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": max(0.15, label_alpha * 0.65)},
            zorder=label_z,
        )


def cubic_point(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    u = 1 - t
    return (
        u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
    )


def draw_same_column_curve(ax, start, end, color, lw, label=None, alpha=0.82, side_x=0.665, label_t=0.55):
    """Connect nodes in the same middle column with a right-side curve."""
    sx, sy = start
    ex, ey = end
    verts = [
        (sx, sy),
        (side_x, sy),
        (side_x, ey),
        (ex, ey),
    ]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, zorder=4)
    ax.add_patch(patch)
    ax.annotate(
        "",
        xy=end,
        xytext=(ex + 0.010, ey),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.0, "alpha": alpha, "mutation_scale": 8},
        zorder=4,
    )
    if label:
        lx, ly = cubic_point(verts[0], verts[1], verts[2], verts[3], label_t)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
            zorder=12,
        )


def draw_cross_column_curve(ax, start, end, color, lw, label=None, alpha=0.74, rad=0.16, label_t=0.56):
    sx, sy = start
    ex, ey = end
    mid_x = (sx + ex) / 2
    bend = rad * (1 if ey >= sy else -1)
    verts = [
        (sx, sy),
        (mid_x, sy + bend),
        (mid_x, ey - bend),
        (ex, ey),
    ]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="<->",
        mutation_scale=11,
        facecolor="none",
        edgecolor=color,
        color=color,
        lw=lw,
        alpha=alpha,
        zorder=4,
    )
    ax.add_patch(patch)
    if label:
        lx, ly = cubic_point(verts[0], verts[1], verts[2], verts[3], label_t)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
            zorder=12,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap_seg_slopes = build_cap_segment_slopes()
    trait_seg_slopes = build_trait_segment_slopes()
    betas = read_betas()
    subtasks = select_subtasks()
    subtask_seg_slopes = read_subtask_segment_slopes()

    fig, ax = plt.subplots(figsize=(15.8, 9.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.948,
        "Dashed lines denote alpha responses; solid lines denote correlations or DS-judge capability-demand links.",
        fontsize=11.5,
        color="#555",
        ha="left",
    )
    ax.text(0.11, 0.875, "alpha", fontsize=12, weight="bold", color="#555", ha="center")
    ax.text(0.35, 0.895, "capabilities", fontsize=12, weight="bold", color="#555", ha="center")
    ax.text(0.61, 0.895, "TRAIT", fontsize=12, weight="bold", color="#555", ha="center")
    ax.text(0.855, 0.875, "selected subtasks", fontsize=12, weight="bold", color="#555", ha="center")

    alpha_pos = (0.11, 0.64)
    alpha_neg = (0.11, 0.34)
    non_dark_traits = [t for t in TRAITS if t not in EVIL_TRAITS]
    dark_traits = [t for t in TRAITS if t in EVIL_TRAITS]
    trait_order = sorted(
        non_dark_traits,
        key=lambda t: max(abs(mean([trait_seg_slopes[(m, t)][i] for m in MODELS])) for i in range(len(SEGMENTS))),
        reverse=True,
    ) + sorted(
        dark_traits,
        key=lambda t: max(abs(mean([trait_seg_slopes[(m, t)][i] for m in MODELS])) for i in range(len(SEGMENTS))),
        reverse=True,
    )
    cap_pos = {
        "Factual Knowledge": (0.35, 0.80),
        "Language Understanding": (0.35, 0.68),
        "Deductive Reasoning": (0.35, 0.56),
    }
    trait_y = [0.640, 0.565, 0.490, 0.415, 0.340, 0.250, 0.165, 0.080]
    trait_pos = {trait: (0.61, trait_y[i]) for i, trait in enumerate(trait_order)}

    sub_pos = {}
    for i, r in enumerate(subtasks):
        cap = str(r["dominant_cap"])
        base_y = {"Factual Knowledge": 0.78, "Language Understanding": 0.56, "Deductive Reasoning": 0.34}[cap]
        idx = sum(1 for rr in subtasks[:i] if rr["dominant_cap"] == cap)
        sub_pos[str(r["task"])] = (0.885, base_y - idx * 0.048)

    # Background demand links.
    for r in subtasks:
        cap = str(r["dominant_cap"])
        task = str(r["task"])
        wt = float(r["dominant_weight"])
        draw_edge(
            ax,
            (cap_pos[cap][0] + 0.078, cap_pos[cap][1]),
            (sub_pos[task][0] - 0.105, sub_pos[task][1]),
            "#7f7f7f",
            0.55 + wt * 1.15,
            False,
            f"{wt:.2f}",
            alpha=0.24,
            rad=0.02,
            label_t=0.84,
            arrows=False,
        )

    def resp_color(v: float) -> str:
        return COL_POS if v >= 0 else COL_NEG

    # Alpha -> subtask response as low-priority pale context.
    sub_response = []
    for r in subtasks:
        key = (str(r["module"]), str(r["task"]))
        if key not in subtask_seg_slopes:
            continue
        for side in ["neg", "pos"]:
            val = side_response(subtask_seg_slopes[key], side)
            sub_response.append((abs(val), r, side, val))
    for _score, r, side, val in sorted(sub_response, key=lambda x: x[0], reverse=True)[:6]:
        task = str(r["task"])
        src = alpha_neg if side == "neg" else alpha_pos
        color = resp_color(val)
        draw_smooth_response_edge(
            ax,
            (src[0] + 0.052, src[1]),
            (sub_pos[task][0] - 0.110, sub_pos[task][1]),
            color,
            0.8,
            True,
            fmt(val),
            alpha=0.18,
            label_t=0.74,
            label_alpha=0.24,
            label_z=2,
            zorder=2,
            curve=0.55,
        )

    # Alpha -> capabilities.
    for cap, p in cap_pos.items():
        for side in ["neg", "pos"]:
            val = mean([side_response(cap_seg_slopes[(m, cap)], side) for m in MODELS])
            if abs(val) < 0.08:
                continue
            src = alpha_neg if side == "neg" else alpha_pos
            color = resp_color(val)
            draw_smooth_response_edge(
                ax,
                (src[0] + 0.052, src[1]),
                (p[0] - 0.078, p[1]),
                color,
                0.75 + min(abs(val) * 1.0, 1.4),
                True,
                fmt(val),
                0.64,
                label_t=0.43,
                curve=0.58,
            )

    # Alpha -> traits: strongest side-aggregated responses.
    trait_response = []
    for trait in trait_order:
        for side in ["neg", "pos"]:
            val = mean([side_response(trait_seg_slopes[(m, trait)], side) for m in MODELS])
            if abs(val) >= 0.07:
                trait_response.append((abs(val), trait, side, val))
    for rank, (_score, trait, side, val) in enumerate(sorted(trait_response, reverse=True)[:10]):
        p = trait_pos[trait]
        src = alpha_neg if side == "neg" else alpha_pos
        color = resp_color(val)
        draw_smooth_response_edge(
            ax,
            (src[0] + 0.052, src[1]),
            (p[0] - 0.064, p[1]),
            color,
            1.15 + min(abs(val), 1.2) * 0.9,
            True,
            fmt(val),
            alpha=0.42,
            label_t=0.36 if side == "neg" else 0.50,
            label_dy=(rank % 3 - 1) * 0.010,
            curve=0.55,
        )

    # Capability -> TRAIT correlations.
    beta_edges = sorted(((abs(beta), cap, trait, beta) for (cap, trait), beta in betas.items() if abs(beta) >= 0.30), reverse=True)
    for rank, (_ab, cap, trait, beta) in enumerate(beta_edges):
        color = COL_POS if beta > 0 else COL_NEG
        dist = abs(cap_pos[cap][1] - trait_pos[trait][1])
        dist_norm = min(dist / 0.75, 1.0)
        lw = 1.00 + 1.10 * min(_ab / 0.55, 1.0) + 2.10 * dist_norm
        alpha = 0.36 + 0.42 * dist_norm
        rad = 0.05 + 0.20 * dist_norm + 0.018 * (rank % 3)
        label_t = 0.62 if dist_norm > 0.55 else 0.52
        draw_cross_column_curve(
            ax,
            (cap_pos[cap][0] + 0.078, cap_pos[cap][1]),
            (trait_pos[trait][0] - 0.058, trait_pos[trait][1] + 0.020),
            color,
            lw,
            fmt(beta),
            alpha=alpha,
            rad=rad,
            label_t=0.68 if dist_norm > 0.55 else 0.58,
        )

    # Nodes last.
    draw_node(ax, alpha_pos, "+ alpha", "circle", 0.13, 0.080, COL_POS, "#fff4f4", 10.2)
    draw_node(ax, alpha_neg, "- alpha", "circle", 0.13, 0.080, COL_NEG, "#f3f8ff", 10.2)
    for cap, p in cap_pos.items():
        draw_node(ax, p, CAP_SHORT[cap], "ellipse", 0.15, 0.062, "#222", "white", 10, text_color=CAP_TEXT[cap])
    for trait, p in trait_pos.items():
        draw_node(
            ax,
            p,
            TRAIT_SHORT[trait],
            "round",
            0.112,
            0.024,
            "#222",
            "white",
            8.2,
            text_color=TRAIT_EVIL if trait in EVIL_TRAITS else TRAIT_OTHER,
            zorder=7,
            face_alpha=0.58,
            pad=0.007,
            rounding=0.012,
        )
    for r in subtasks:
        task = str(r["task"])
        mod = str(r["module"])
        draw_node(ax, sub_pos[task], module_label(mod, task), "rect", 0.205, 0.032, "#555", "white", 6.6, wrap=False)

    # Legend.
    lx, ly = 0.80, 0.16
    draw_edge(ax, (lx, ly), (lx + 0.055, ly), COL_POS, 1.5, True, None, 0.9, arrows=False)
    ax.text(lx + 0.068, ly, "positive response slope", va="center", fontsize=9.5)
    draw_edge(ax, (lx, ly - 0.035), (lx + 0.055, ly - 0.035), COL_NEG, 1.5, True, None, 0.9, arrows=False)
    ax.text(lx + 0.068, ly - 0.035, "negative response slope", va="center", fontsize=9.5)
    ax.add_patch(
        FancyArrowPatch(
            (lx, ly - 0.070),
            (lx + 0.055, ly - 0.070),
            arrowstyle="<->",
            mutation_scale=11,
            color=COL_POS,
            lw=2.8,
            alpha=0.9,
            zorder=3,
        )
    )
    ax.text(lx + 0.068, ly - 0.070, "correlation", va="center", fontsize=9.5)
    draw_edge(ax, (lx, ly - 0.105), (lx + 0.055, ly - 0.105), "#777", 1.5, False, None, 0.35, arrows=False)
    ax.text(lx + 0.068, ly - 0.105, "demand (deepseek-v4-pro judged)", va="center", fontsize=9.5)

    out_svg = OUT_DIR / "trait_capability_panel_c_standalone_mpl.svg"
    out_png = OUT_DIR / "trait_capability_panel_c_standalone_mpl.png"
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    with (OUT_DIR / "panel_c_standalone_selected_subtasks.csv").open("w", newline="") as f:
        fields = ["module", "task", "dominant_cap", "dominant_weight", "positive_slope", "negative_slope"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in subtasks:
            writer.writerow(
                {
                    "module": r["module"],
                    "task": r["task"],
                    "dominant_cap": r["dominant_cap"],
                    "dominant_weight": r["dominant_weight"],
                    "positive_slope": r["pos"],
                    "negative_slope": r["neg"],
                }
            )
    print(out_svg)
    print(out_png)


if __name__ == "__main__":
    main()
