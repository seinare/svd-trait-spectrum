#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

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
    SVG,
    TRAITS,
    alpha_label,
    mean,
    piecewise_slopes,
    read_capabilities,
    read_task_weights,
    read_trait_scores,
    scale,
    sem,
    y_from_value,
)


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
MODULE_COLORS = {"mmlu_pro": "#4c78a8", "mmlu_redux": "#f58518", "agieval": "#54a24b", "bbh": "#b279a2"}


def read_all_trait_capability_betas() -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    betas: dict[tuple[str, str], float] = {}
    r2: dict[str, float] = {}
    path = ROOT / "docs/results/trait_capability_correlation/pooled_trait_on_capability_regression.csv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["term"] in CAPS:
                betas[(row["term"], row["trait"])] = float(row["standardized_beta"])
                r2[row["trait"]] = float(row["r2"])
    return betas, r2


def build_trait_slopes() -> dict[tuple[str, str], tuple[float, float]]:
    scores = read_trait_scores()
    out = {}
    for model in MODELS:
        for trait in TRAITS:
            base = scores[(model, trait)][0.0]
            points = {a: scores[(model, trait)][a] - base for a in ALPHAS}
            out[(model, trait)] = piecewise_slopes(points)
    return out


def build_cap_slopes() -> dict[tuple[str, str], tuple[float, float]]:
    caps = read_capabilities()
    out = {}
    for trait_model, cap_model in CAP_MODEL.items():
        for cap in CAPS:
            out[(trait_model, cap)] = piecewise_slopes(caps[(cap_model, cap)])
    return out


def read_subtask_direction_slopes() -> dict[tuple[str, str], tuple[float, float]]:
    raw: dict[tuple[str, str, str], dict[float, float]] = defaultdict(dict)
    path = ROOT / "docs/results/eval6_all_models_alpha9_capability/eval6_all_models_subtask_scores_long.csv"
    wanted_models = set(CAP_MODEL.values())
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model not in wanted_models:
                continue
            alpha = round(float(row["alpha"]), 2)
            raw[(model, row["module"], row["task"])][alpha] = float(row["value"])

    per_task: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (_model, module, task), vals in raw.items():
        if 0.0 not in vals:
            continue
        points = {a: vals[a] - vals[0.0] for a in ALPHAS if a in vals}
        if len(points) >= 5:
            per_task[(module, task)].append(piecewise_slopes(points))

    return {
        key: (mean([p for p, _n in pairs]), mean([n for _p, n in pairs]))
        for key, pairs in per_task.items()
        if pairs
    }


def select_subtasks(n_per_module: int = 3) -> list[dict[str, object]]:
    weights = {(r["module"], r["task"]): r for r in read_task_weights()}
    slopes = read_subtask_direction_slopes()
    selected: list[dict[str, object]] = []
    for module in ["mmlu_pro", "mmlu_redux", "agieval", "bbh"]:
        candidates = []
        for key, pair in slopes.items():
            if key[0] != module or key not in weights:
                continue
            candidates.append((max(abs(pair[0]), abs(pair[1])), key, pair))
        for _score, key, pair in sorted(candidates, reverse=True)[:n_per_module]:
            w = weights[key]
            selected.append(
                {
                    "module": key[0],
                    "task": key[1],
                    "pos": pair[0],
                    "neg": pair[1],
                    "f": float(w["f"]),
                    "l": float(w["l"]),
                    "d": float(w["d"]),
                }
            )
    return selected


def clean_task_label(task: str) -> str:
    label = task
    for prefix in ["mmlu_pro_", "mmlu_redux_", "agieval_", "bbh_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("_", " ")
    return label[:22]


def draw_bar_pair(svg: SVG, cx: float, zero_y: float, vals_pos: list[float], vals_neg: list[float], lo: float, hi: float, top: float, bottom: float, bar_w: float = 18) -> None:
    for off, vals, color in [(-11, vals_pos, COL_POS), (11, vals_neg, COL_NEG)]:
        mv = mean(vals)
        yy = y_from_value(mv, lo, hi, top, bottom)
        svg.rect(cx + off - bar_w / 2, min(yy, zero_y), bar_w, abs(yy - zero_y), color, "none", 0, 0)
        err = sem(vals)
        svg.line(cx + off, y_from_value(mv - err, lo, hi, top, bottom), cx + off, y_from_value(mv + err, lo, hi, top, bottom), "#333", 0.8)


def panel_a_combined(svg: SVG, x: int, y: int, w: int, h: int, cap_slopes, trait_slopes) -> None:
    svg.text(x, y, "a", 24, "700")
    svg.text(x + 35, y, "directional alpha response in capabilities and TRAIT on a common slope axis", 17, "700")

    items: list[tuple[str, str, list[float], list[float]]] = []
    for cap in CAPS:
        items.append(("Capability", CAP_SHORT[cap], [cap_slopes[(m, cap)][0] for m in MODELS], [cap_slopes[(m, cap)][1] for m in MODELS]))
    trait_order = sorted(
        TRAITS,
        key=lambda t: abs(mean([trait_slopes[(m, t)][0] for m in MODELS])) + abs(mean([trait_slopes[(m, t)][1] for m in MODELS])),
        reverse=True,
    )
    for trait in trait_order:
        items.append(("TRAIT", TRAIT_SHORT[trait], [trait_slopes[(m, trait)][0] for m in MODELS], [trait_slopes[(m, trait)][1] for m in MODELS]))

    left, top, right, bottom = x + 175, y + 42, x + w - 70, y + h - 48
    vals = [v for _group, _label, pos, neg in items for v in pos + neg]
    m = max(abs(min(vals)), abs(max(vals)), 0.01) * 1.15
    lo, hi = -m, m
    zero_x = scale(0, lo, hi, left, right)
    svg.line(left, bottom, right, bottom, "#333", 1)
    svg.line(zero_x, top, zero_x, bottom, "#777", 0.8, "3,3")
    for tick in [-0.4, -0.2, 0, 0.2, 0.4]:
        if lo <= tick <= hi:
            xx = scale(tick, lo, hi, left, right)
            svg.line(xx, bottom, xx, bottom + 4, "#333", 0.8)
            svg.text(xx, bottom + 18, f"{tick:.1f}", 10, anchor="middle")
            svg.line(xx, top, xx, bottom, "#ececec", 0.5)
    svg.text((left + right) / 2, bottom + 36, "directional response slope", 12, "700", anchor="middle")
    row_h = (bottom - top) / len(items)
    last_group = None
    for i, (group, label, pos_vals, neg_vals) in enumerate(items):
        cy = top + row_h * (i + 0.5)
        if group != last_group:
            svg.text(x + 38, cy + 4, group, 11, "700", color="#555")
            last_group = group
        svg.text(left - 12, cy + 4, label, 10, anchor="end")
        for dy, vals2, color in [(-4.0, pos_vals, COL_POS), (4.0, neg_vals, COL_NEG)]:
            mv = mean(vals2)
            xx = scale(mv, lo, hi, left, right)
            svg.rect(min(zero_x, xx), cy + dy - 2.7, abs(xx - zero_x), 5.4, color, "none", 0, 0)
            svg.circle(xx, cy + dy, 3.2, color, "none", 0.95)
            err = sem(vals2)
            svg.line(scale(mv - err, lo, hi, left, right), cy + dy, scale(mv + err, lo, hi, left, right), cy + dy, "#333", 0.5)
    svg.rect(x + w - 140, y + 8, 11, 11, COL_POS, "none", 0, 0)
    svg.text(x + w - 124, y + 18, "positive alpha", 11)
    svg.rect(x + w - 140, y + 26, 11, 11, COL_NEG, "none", 0, 0)
    svg.text(x + w - 124, y + 36, "negative alpha", 11)


def panel_b_matrix(svg: SVG, x: int, y: int, w: int, h: int) -> None:
    svg.text(x, y, "b", 24, "700")
    svg.text(x + 35, y, "capability response -> TRAIT response", 17, "700")
    betas, r2 = read_all_trait_capability_betas()
    left, top = x + 155, y + 92
    cell_w, cell_h = 104, 50
    max_abs = max(abs(v) for v in betas.values())

    def blend(color: str, strength: float) -> str:
        color = color.lstrip("#")
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        rr = round(255 - (255 - r) * strength)
        gg = round(255 - (255 - g) * strength)
        bb = round(255 - (255 - b) * strength)
        return f"#{rr:02x}{gg:02x}{bb:02x}"

    label_lines = {
        "Openness": ["Open-", "ness"],
        "Conscientiousness": ["Conscient.", ""],
        "Extraversion": ["Extra-", "version"],
        "Agreeableness": ["Agree-", "ableness"],
        "Neuroticism": ["Neuro-", "ticism"],
        "Machiavellianism": ["Machiavell.", ""],
        "Narcissism": ["Narcissism", ""],
        "Psychopathy": ["Psycho-", "pathy"],
    }
    for j, trait in enumerate(TRAITS):
        cx = left + j * cell_w + cell_w / 2
        lines = label_lines[trait]
        svg.text(cx, top - 30, lines[0], 9, "700", anchor="middle")
        if lines[1]:
            svg.text(cx, top - 18, lines[1], 9, "700", anchor="middle")
        svg.text(cx, top + len(CAPS) * cell_h + 25, f"{r2.get(trait, 0):.2f}", 10, anchor="middle", color="#555")
    svg.text(left - 42, top + len(CAPS) * cell_h + 25, "R²", 10, "700", anchor="end", color="#555")
    for i, cap in enumerate(CAPS):
        cy = top + i * cell_h + cell_h / 2
        svg.text(left - 16, cy + 4, CAP_SHORT[cap], 11, "700", anchor="end")
        for j, trait in enumerate(TRAITS):
            beta = betas[(cap, trait)]
            strength = min(abs(beta) / max_abs, 1.0)
            fill = blend(COL_POS if beta > 0 else COL_NEG, 0.15 + 0.72 * strength)
            stroke = "#111" if abs(beta) >= 0.30 else "#d0d0d0"
            sw = 1.2 if abs(beta) >= 0.30 else 0.5
            xx, yy = left + j * cell_w, top + i * cell_h
            svg.rect(xx, yy, cell_w - 4, cell_h - 4, fill, stroke, sw, 2)
            svg.text(xx + cell_w / 2 - 2, yy + cell_h / 2 + 4, f"{beta:+.2f}".replace("+", ""), 10, "700" if abs(beta) >= 0.30 else "400", anchor="middle")
    svg.text(x + 35, y + h - 18, "Cells are standardized beta; dark borders mark |beta| >= 0.30.", 10, color="#555")


def draw_node(svg: SVG, x: float, y: float, text: str, w: float, h: float, fill: str = "#fff", stroke: str = "#222", ellipse: bool = False, rx: float = 7) -> None:
    if ellipse:
        svg.ellipse(x, y, w, h, fill, stroke, 1.2)
    else:
        svg.rect(x - w / 2, y - h / 2, w, h, fill, stroke, 1.1, rx)
    if len(text) > 18:
        parts = text.split()
        mid = len(parts) // 2
        svg.text(x, y - 3, " ".join(parts[:mid]), 10, "600", anchor="middle")
        svg.text(x, y + 10, " ".join(parts[mid:]), 10, "600", anchor="middle")
    else:
        svg.text(x, y + 4, text, 10, "600", anchor="middle")


def curve(svg: SVG, x1: float, y1: float, x2: float, y2: float, color: str, width: float, opacity: float = 0.65, dash: str | None = None) -> None:
    mx = (x1 + x2) / 2
    d = f"M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    svg.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width:.2f}" opacity="{opacity:.2f}"{extra}/>')


def line_label(svg: SVG, x1: float, y1: float, x2: float, y2: float, text: str, color: str = "#222", t: float = 0.52) -> None:
    mx = x1 + (x2 - x1) * t
    my = y1 + (y2 - y1) * t
    w = max(24, len(text) * 6 + 8)
    svg.rect(mx - w / 2, my - 8, w, 15, "#ffffff", "none", 0, 2)
    svg.text(mx, my + 4, text, 9, "700", anchor="middle", color=color)


def panel_c_paths(svg: SVG, x: int, y: int, w: int, h: int, cap_slopes, trait_slopes, subtasks) -> None:
    svg.text(x, y, "c", 24, "700")
    svg.text(x + 35, y, "perturbation-response association network", 17, "700")
    alpha_pos = (x + 85, y + 250)
    alpha_neg = (x + 85, y + 390)
    cap_pos = {
        "Factual Knowledge": (x + 335, y + 105),
        "Language Understanding": (x + 335, y + 210),
        "Deductive Reasoning": (x + 335, y + 315),
    }
    trait_pos = {trait: (x + 640, y + 56 + i * 50) for i, trait in enumerate(TRAITS)}
    sub_pos = {r["task"]: (x + 1040 + (i // 6) * 230, y + 70 + (i % 6) * 76) for i, r in enumerate(subtasks)}

    draw_node(svg, *alpha_pos, "+ alpha", 86, 42, "#fff5f5", COL_POS, True)
    draw_node(svg, *alpha_neg, "- alpha", 86, 42, "#f3f8ff", COL_NEG, True)
    for cap, pos in cap_pos.items():
        draw_node(svg, *pos, CAP_SHORT[cap], 128, 38, "#ffffff", "#222", True)
    for trait, pos in trait_pos.items():
        draw_node(svg, *pos, TRAIT_SHORT[trait], 132, 28, "#ffffff", "#222", False, rx=7)
    for r in subtasks:
        sx, sy = sub_pos[r["task"]]
        color = MODULE_COLORS.get(str(r["module"]), "#888")
        draw_node(svg, sx, sy, clean_task_label(str(r["task"])), 168, 24, "#ffffff", color, False, rx=0)

    # alpha -> capabilities
    for cap, (cx, cy) in cap_pos.items():
        pos_vals = [cap_slopes[(m, cap)][0] for m in MODELS]
        neg_vals = [cap_slopes[(m, cap)][1] for m in MODELS]
        for src, val, color in [(alpha_pos, mean(pos_vals), COL_POS), (alpha_neg, mean(neg_vals), COL_NEG)]:
            x1, y1, x2, y2 = src[0] + 45, src[1], cx - 66, cy
            curve(svg, x1, y1, x2, y2, color, 0.6 + min(abs(val) * 3.5, 2.1), 0.50, "5,4")
            line_label(svg, x1, y1, x2, y2, f"{val:+.2f}".replace("+", ""), color, 0.42)

    # alpha -> TRAIT, strongest six mean directional responses.
    trait_edges = []
    for trait in TRAITS:
        pv = mean([trait_slopes[(m, trait)][0] for m in MODELS])
        nv = mean([trait_slopes[(m, trait)][1] for m in MODELS])
        trait_edges.append((abs(pv), "pos", trait, pv))
        trait_edges.append((abs(nv), "neg", trait, nv))
    for _score, sign, trait, val in sorted(trait_edges, reverse=True)[:8]:
        src = alpha_pos if sign == "pos" else alpha_neg
        color = COL_POS if sign == "pos" else COL_NEG
        tx, ty = trait_pos[trait]
        x1, y1, x2, y2 = src[0] + 45, src[1], tx - 66, ty
        curve(svg, x1, y1, x2, y2, color, 0.45 + min(abs(val) * 4, 1.5), 0.18, "3,4")

    # alpha -> selected subtasks, pale.
    sub_edges = []
    for r in subtasks:
        sub_edges.append((abs(float(r["pos"])), r, "pos", float(r["pos"])))
        sub_edges.append((abs(float(r["neg"])), r, "neg", float(r["neg"])))
    for _score, r, sign, val in sorted(sub_edges, reverse=True)[:6]:
        sx, sy = sub_pos[r["task"]]
        src = alpha_pos if sign == "pos" else alpha_neg
        color = COL_POS if sign == "pos" else COL_NEG
        curve(svg, src[0] + 45, src[1], sx - 88, sy, color, 0.35 + min(abs(val) * 2, 1.0), 0.08, "2,6")

    # capability -> TRAIT associations.
    betas, _r2 = read_all_trait_capability_betas()
    for (cap, trait), beta in betas.items():
        if abs(beta) < 0.33:
            continue
        color = COL_POS if beta > 0 else COL_NEG
        cx, cy = cap_pos[cap]
        tx, ty = trait_pos[trait]
        x1, y1, x2, y2 = cx + 66, cy, tx - 70, ty
        curve(svg, x1, y1, x2, y2, color, 0.8 + min(abs(beta) * 1.9, 2.6), 0.72)
        line_label(svg, x1, y1, x2, y2, f"{beta:+.2f}".replace("+", ""), color, 0.56)

    # DS-judge capability demand -> subtasks.
    for r in subtasks:
        weights = [("Factual Knowledge", float(r["f"])), ("Language Understanding", float(r["l"])), ("Deductive Reasoning", float(r["d"]))]
        sx, sy = sub_pos[r["task"]]
        cap, wt = sorted(weights, key=lambda x: x[1], reverse=True)[0]
        cx, cy = cap_pos[cap]
        x1, y1, x2, y2 = cx + 66, cy, sx - 88, sy
        curve(svg, x1, y1, x2, y2, "#666666", 0.35 + wt * 1.3, 0.16)
        line_label(svg, x1, y1, x2, y2, f"{wt:.2f}", "#555", 0.74)
    svg.text(x + 40, y + h - 26, "Dashed lines are alpha responses; solid lines are capability-TRAIT correlations or DS-judge capability-demand links.", 10, color="#555")


def draw_triangle_standalone(tasks: list[dict[str, object]]) -> Path:
    svg = SVG(960, 760)
    svg.rect(0, 0, 960, 760, "#ffffff", "none", 0, 0)
    svg.text(40, 44, "Task subitems in three-dimensional capability-demand space", 22, "700")
    x0, y0, size = 130, 650, 540
    fpt, lpt, dpt = (x0, y0), (x0 + size, y0), (x0 + size / 2, y0 - size * 0.866)

    def xy(f: float, l: float, d: float) -> tuple[float, float]:
        return (f * fpt[0] + l * lpt[0] + d * dpt[0], f * fpt[1] + l * lpt[1] + d * dpt[1])

    svg.polygon([fpt, lpt, dpt], "#ffffff", "#222", 1.0)
    for t in [0.2, 0.4, 0.6, 0.8]:
        svg.line(*xy(t, 1 - t, 0), *xy(t, 0, 1 - t), "#cfd5da", 0.8)
        svg.line(*xy(1 - t, t, 0), *xy(0, t, 1 - t), "#cfd5da", 0.8)
        svg.line(*xy(1 - t, 0, t), *xy(0, 1 - t, t), "#cfd5da", 0.8)
        fx, fy = xy(t, 0, 1 - t)
        lx, ly = xy(1 - t, t, 0)
        dx, dy = xy(0, 1 - t, t)
        svg.text(fx - 18, fy + 3, f"{t:.1f}", 9, anchor="end", color="#666")
        svg.text(lx, ly + 22, f"{t:.1f}", 9, anchor="middle", color="#666")
        svg.text(dx + 18, dy + 3, f"{t:.1f}", 9, anchor="start", color="#666")
    for r in read_task_weights():
        px, py = xy(float(r["f"]), float(r["l"]), float(r["d"]))
        svg.circle(px, py, 3.8, MODULE_COLORS.get(str(r["module"]), "#888"), "none", 0.62)
    svg.line(fpt[0], fpt[1], lpt[0], lpt[1], "#222", 1.1)
    svg.line(lpt[0], lpt[1], dpt[0], dpt[1], "#222", 1.1)
    svg.line(dpt[0], dpt[1], fpt[0], fpt[1], "#222", 1.1)
    svg.text(fpt[0] - 16, fpt[1] + 38, "Factual", 14, "700", anchor="middle")
    svg.text(lpt[0] + 18, lpt[1] + 38, "Language", 14, "700", anchor="middle")
    svg.text(dpt[0], dpt[1] - 24, "Deductive", 14, "700", anchor="middle")
    lx, ly = 740, 470
    svg.text(lx - 4, ly - 20, "Module", 12, "700")
    for i, (mod, color) in enumerate(MODULE_COLORS.items()):
        svg.circle(lx, ly + i * 22, 5, color, "none", 0.85)
        svg.text(lx + 12, ly + i * 22 + 4, mod.replace("_", "-"), 11)
    out = OUT_DIR / "task_capability_simplex_standalone.svg"
    svg.save(out)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap_slopes = build_cap_slopes()
    trait_slopes = build_trait_slopes()
    subtasks = select_subtasks()

    svg = SVG(1500, 1530)
    svg.rect(0, 0, 1500, 1530, "#ffffff", "none", 0, 0)
    svg.text(42, 40, "Directional spectral perturbations link capability, TRAIT, and task responses", 23, "700")
    svg.text(42, 64, "Four dense instruct models with complete nine-point TRAIT data; positive alpha in red and negative alpha in blue.", 13, color="#555")
    panel_a_combined(svg, 55, 120, 1360, 330, cap_slopes, trait_slopes)
    panel_b_matrix(svg, 55, 505, 1360, 295)
    panel_c_paths(svg, 55, 850, 1360, 640, cap_slopes, trait_slopes, subtasks)
    out_svg = OUT_DIR / "trait_capability_main_figure_v2.svg"
    svg.save(out_svg)
    tri_svg = draw_triangle_standalone(subtasks)

    with (OUT_DIR / "panel_c_selected_subtasks.csv").open("w", newline="") as f:
        fields = ["module", "task", "positive_slope", "negative_slope", "factual_weight", "language_weight", "deductive_weight"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in subtasks:
            writer.writerow(
                {
                    "module": r["module"],
                    "task": r["task"],
                    "positive_slope": r["pos"],
                    "negative_slope": r["neg"],
                    "factual_weight": r["f"],
                    "language_weight": r["l"],
                    "deductive_weight": r["d"],
                }
            )
    notes = [
        "# Main Figure V2 Design",
        "",
        "Panel a combines alpha-direction response slopes for capabilities and TRAIT. The two subplots use separate scales.",
        "",
        "Panel b shows the full capability-to-TRAIT standardized coefficient matrix. Columns are TRAIT subdimensions, rows are the three fitted capability responses.",
        "",
        "Panel c is a perturbation-response association network. Red/blue encode positive/negative signs. Pale dashed lines are direct alpha-to-TRAIT/subtask response links; gray lines are DS-judge capability demand links from capabilities to selected subtasks.",
        "",
        "The task capability simplex is exported separately as `task_capability_simplex_standalone.svg`.",
    ]
    (OUT_DIR / "trait_capability_main_figure_v2_notes.md").write_text("\n".join(notes))
    print(out_svg)
    print(tri_svg)


if __name__ == "__main__":
    main()
