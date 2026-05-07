#!/usr/bin/env python3
"""All-model eval6 alpha9 capability analysis and SVG figures."""

from __future__ import annotations

import csv
import html
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
RAW_ROOTS = [ROOT / "docs" / "raw" / "eval6_llama", ROOT / "docs" / "raw" / "eval6_remote_sync"]
CAP_CSV = RESULTS / "capability_dimension" / "deepseek_v4_pro_eval6_batch10x10_20260504_141548.csv"
OUT = RESULTS / "eval6_all_models_alpha9_capability"
REPORT = OUT / "eval6_all_models_alpha9_capability.md"

MODELS = ["llama1b", "llama3b", "llama8b", "qwen3_8b", "qwen3_30b_a3b"]
MODEL_LABELS = {
    "llama1b": "Llama 3.2 1B Instruct",
    "llama3b": "Llama 3.2 3B Instruct",
    "llama8b": "Llama 3.1 8B Instruct",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_30b_a3b": "Qwen3 30B-A3B MoE",
}
MODEL_MARKERS = {
    "llama1b": "circle",
    "llama3b": "diamond",
    "llama8b": "square",
    "qwen3_8b": "triangle",
    "qwen3_30b_a3b": "cross",
}
MODEL_STROKES = {
    "llama1b": "#1d4ed8",
    "llama3b": "#15803d",
    "llama8b": "#7e22ce",
    "qwen3_8b": "#b91c1c",
    "qwen3_30b_a3b": "#c2410c",
}
MODULES = ["mmlu_pro", "mmlu_redux", "agieval", "bbh"]
ALPHAS = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
FIT_ALPHAS = [a for a in ALPHAS if a != 0.0]
XS = [round(-0.2 + i * 0.004, 3) for i in range(101)]
DIMENSIONS = ["Factual Knowledge", "Language Understanding", "Deductive Reasoning"]


def alpha_key(alpha: float) -> float:
    return round(float(alpha), 3)


def alpha_color(alpha: float) -> str:
    if abs(alpha) < 1e-12:
        return "#64748b"
    t = min(abs(alpha) / 0.2, 1.0)
    base = (248, 250, 252)
    target = (37, 99, 235) if alpha < 0 else (220, 38, 38)
    vals = [round(base[i] + (target[i] - base[i]) * t) for i in range(3)]
    return "#" + "".join(f"{v:02x}" for v in vals)


def infer_model(path: Path) -> str | None:
    text = str(path)
    if "Llama-3_2-1B-Instruct" in text or "llama1b" in text:
        return "llama1b"
    if "Llama-3_2-3B-Instruct" in text or "llama3b" in text:
        return "llama3b"
    if "Llama-3_1-8B-Instruct" in text or "llama8b" in text or "llama31_8b" in text:
        return "llama8b"
    if "Qwen3-30B-A3B" in text or "qwen3_30b_a3b" in text:
        return "qwen3_30b_a3b"
    if "Qwen3-8B" in text or "qwen3_8b" in text or "b968826d9c46" in text:
        return "qwen3_8b"
    return None


def infer_module(path: Path) -> str | None:
    parts = set(path.parts)
    for module in MODULES:
        if module in parts:
            return module
    return None


def infer_alpha(path: Path) -> float | None:
    text = str(path)
    patterns = [
        r"_alpha(-?\d+(?:\.\d+)?)_vllm",
        r"alpha(-?\d+(?:\.\d+)?)_vllm",
        r"alpha_([mp]?\d+p\d+)",
        r"alpha_(-?\d+(?:\.\d+)?)",
        r"alpha([mp]?\d+p\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        raw = match.group(1)
        if raw.startswith("m"):
            return -float(raw[1:].replace("p", "."))
        if raw.startswith("p"):
            return float(raw[1:].replace("p", "."))
        return alpha_key(float(raw))
    return None


def metric_for(module: str, task_row: dict[str, object]) -> tuple[str, float]:
    preferences = {
        "mmlu_pro": ["exact_match,custom-extract"],
        "mmlu_redux": ["exact_match,default"],
        "agieval": ["acc_norm,none", "acc,none"],
        "bbh": ["exact_match,flexible-extract", "exact_match,strict-match"],
    }[module]
    for key in preferences:
        value = task_row.get(key)
        if isinstance(value, int | float):
            return key, float(value)
    for key, value in task_row.items():
        if key == "alias" or key.endswith("_stderr"):
            continue
        if isinstance(value, int | float):
            return key, float(value)
    raise ValueError(f"no metric found for {module}: {task_row}")


def extract_scores() -> list[dict[str, object]]:
    rows_by_key = {}
    bad_rows = []
    for root in RAW_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("results_*.json")):
            model = infer_model(path)
            module = infer_module(path)
            alpha = infer_alpha(path)
            if model not in MODELS or module not in MODULES or alpha not in ALPHAS:
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                bad_rows.append({"source": str(path), "error": repr(exc)})
                continue
            for task, values in payload.get("results", {}).items():
                metric, value = metric_for(module, values)
                key = (model, module, alpha, task)
                rows_by_key[key] = {
                    "model": model,
                    "module": module,
                    "alpha": alpha,
                    "task": task,
                    "metric": metric,
                    "value": value,
                    "source": str(path),
                }
    if bad_rows:
        write_csv(OUT / "bad_json_files.csv", bad_rows)
    return [rows_by_key[k] for k in sorted(rows_by_key)]


def read_caps() -> dict[str, list[float]]:
    caps = {}
    with CAP_CSV.open() as handle:
        for row in csv.DictReader(handle):
            weights = [float(row[dim]) for dim in DIMENSIONS]
            total = sum(weights)
            if total > 1e-12:
                caps[row["task"]] = [w / total for w in weights]
    return caps


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_values(score_rows):
    values = defaultdict(lambda: defaultdict(dict))
    metrics = {}
    for row in score_rows:
        key = (row["model"], row["module"])
        values[key][row["task"]][float(row["alpha"])] = float(row["value"])
        metrics[(row["model"], row["module"], row["task"])] = row["metric"]
    return values, metrics


def coverage(values, caps):
    rows = []
    missing_rows = []
    for model in MODELS:
        for module in MODULES:
            tasks = [task for task, vals in values[(model, module)].items() if task in caps]
            complete = [task for task in tasks if set(ALPHAS).issubset(values[(model, module)][task])]
            for task in sorted(tasks):
                vals = values[(model, module)][task]
                missing = [a for a in ALPHAS if a not in vals]
                if missing:
                    missing_rows.append(
                        {
                            "model": model,
                            "module": module,
                            "task": task,
                            "available_alphas": " ".join(f"{a:g}" for a in sorted(vals)),
                            "missing_alphas": " ".join(f"{a:g}" for a in missing),
                        }
                    )
            rows.append(
                {
                    "model": model,
                    "module": module,
                    "complete_tasks": len(complete),
                    "available_tasks": len(tasks),
                    "complete": len(complete) == len(tasks) and len(tasks) > 0,
                }
            )
    return rows, missing_rows


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [x / div for x in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor:
                aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(n + 1)]
    return [aug[i][-1] for i in range(n)]


def projected_var(weights, sigmas):
    return max(sum((w * s) ** 2 for w, s in zip(weights, sigmas)), 1e-10)


def fit_means(obs, sigmas, ridge=1e-5):
    n = len(DIMENSIONS)
    xtx = [[0.0 for _ in range(n)] for _ in range(n)]
    xty = [0.0 for _ in range(n)]
    for weights, y in obs:
        wt = 1.0 / projected_var(weights, sigmas)
        for i in range(n):
            xty[i] += wt * weights[i] * y
            for j in range(n):
                xtx[i][j] += wt * weights[i] * weights[j]
    for i in range(n):
        xtx[i][i] += ridge
    return solve_linear(xtx, xty)


def nll_alpha(obs, means, sigmas):
    total = 0.0
    for weights, y in obs:
        res = y - dot(weights, means)
        var = projected_var(weights, sigmas)
        total += 0.5 * (math.log(2 * math.pi * var) + res * res / var)
    return total


def optimize_sigmas(obs, means, sigmas):
    z = [math.log(max(s, 1e-5) ** 2) for s in sigmas]
    best = nll_alpha(obs, means, [math.sqrt(math.exp(v)) for v in z])
    step = 0.08
    for _ in range(100):
        q = [math.exp(v) for v in z]
        sig = [math.sqrt(v) for v in q]
        grad_q = [0.0 for _ in q]
        for weights, y in obs:
            res = y - dot(weights, means)
            var = projected_var(weights, sig)
            common = 0.5 * (1.0 / var - res * res / (var * var))
            for k, w in enumerate(weights):
                grad_q[k] += common * w * w
        grad_z = [grad_q[k] * q[k] for k in range(len(q))]
        if math.sqrt(sum(g * g for g in grad_z)) < 1e-8:
            break
        local = step
        accepted = False
        for _ in range(24):
            cand_z = [max(-18.0, min(3.0, z[k] - local * grad_z[k])) for k in range(len(z))]
            cand_sig = [math.sqrt(math.exp(v)) for v in cand_z]
            cand = nll_alpha(obs, means, cand_sig)
            if cand <= best:
                z, best, accepted = cand_z, cand, True
                step = min(local * 1.05, 0.35)
                break
            local *= 0.5
        if not accepted:
            break
    return [math.sqrt(math.exp(v)) for v in z]


def fit_alpha_point(obs):
    sigmas = [0.05 for _ in DIMENSIONS]
    means = fit_means(obs, sigmas)
    last = None
    for _ in range(30):
        sigmas = optimize_sigmas(obs, means, sigmas)
        means = fit_means(obs, sigmas)
        current = nll_alpha(obs, means, sigmas)
        if last is not None and abs(last - current) < 1e-6:
            break
        last = current
    return means, sigmas, nll_alpha(obs, means, sigmas)


def fit_quadratic(points):
    s11 = s12 = s22 = t1 = t2 = 0.0
    for alpha, y in points:
        x1, x2 = alpha, alpha * alpha
        s11 += x1 * x1
        s12 += x1 * x2
        s22 += x2 * x2
        t1 += x1 * y
        t2 += x2 * y
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return 0.0, 0.0
    return (t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det


def curve(b, c, alpha):
    return b * alpha + c * alpha * alpha


def capability_fit(values, caps):
    point_rows = []
    coef_rows = []
    task_sets = {}
    for model in MODELS:
        model_tasks = []
        for module in MODULES:
            for task, vals in values[(model, module)].items():
                if task in caps and set(ALPHAS).issubset(vals):
                    model_tasks.append((module, task))
        task_sets[model] = model_tasks
        dim_points = defaultdict(list)
        for alpha in ALPHAS:
            obs = []
            for module, task in model_tasks:
                vals = values[(model, module)][task]
                obs.append((caps[task], vals[alpha] - vals[0.0]))
            means, sigmas, nll = fit_alpha_point(obs)
            for i, dim in enumerate(DIMENSIONS):
                if alpha != 0.0:
                    dim_points[dim].append((alpha, means[i]))
                point_rows.append(
                    {
                        "model": model,
                        "alpha": alpha,
                        "dimension": dim,
                        "mean_mle": means[i],
                        "sigma_mle": sigmas[i],
                        "nll": nll,
                        "tasks": len(model_tasks),
                    }
                )
        for dim in DIMENSIONS:
            b, c = fit_quadratic(dim_points[dim])
            for alpha in FIT_ALPHAS:
                coef_rows.append(
                    {
                        "model": model,
                        "dimension": dim,
                        "alpha": alpha,
                        "point_mean_mle": dict(dim_points[dim])[alpha],
                        "fit_mean": curve(b, c, alpha),
                        "linear_coef_b": b,
                        "quadratic_coef_c": c,
                    }
                )
    return point_rows, coef_rows, task_sets


def task_slopes(values, caps):
    rows = []
    for model in MODELS:
        for module in MODULES:
            for task, vals in values[(model, module)].items():
                if task not in caps or not set(ALPHAS).issubset(vals):
                    continue
                points = [(alpha, vals[alpha] - vals[0.0]) for alpha in FIT_ALPHAS]
                b, c = fit_quadratic(points)
                best_alpha = max(ALPHAS, key=lambda a: (vals[a], -abs(a)))
                klass = "positive" if best_alpha > 0 else "negative" if best_alpha < 0 else "neutral"
                row = {
                    "model": model,
                    "module": module,
                    "task": task,
                    "slope_task": b,
                    "quadratic_coef_c": c,
                    "best_alpha": best_alpha,
                    "class": klass,
                }
                for dim, val in zip(DIMENSIONS, caps[task]):
                    row[dim] = val
                rows.append(row)
    return rows


def percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def nice_ticks(lo, hi, n=5):
    if abs(hi - lo) < 1e-12:
        return [lo]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def marker(cx, cy, model, color, size=5.0):
    stroke = MODEL_STROKES[model]
    if MODEL_MARKERS[model] == "diamond":
        pts = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
        return f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{color}" stroke="{stroke}" stroke-width="0.8"/>'
    if MODEL_MARKERS[model] == "square":
        return f'<rect x="{cx-size:.1f}" y="{cy-size:.1f}" width="{2*size:.1f}" height="{2*size:.1f}" fill="{color}" stroke="{stroke}" stroke-width="0.8"/>'
    if MODEL_MARKERS[model] == "triangle":
        pts = [(cx, cy - size), (cx + size * 0.9, cy + size), (cx - size * 0.9, cy + size)]
        return f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{color}" stroke="{stroke}" stroke-width="0.8"/>'
    if MODEL_MARKERS[model] == "cross":
        return (
            f'<line x1="{cx-size:.1f}" y1="{cy-size:.1f}" x2="{cx+size:.1f}" y2="{cy+size:.1f}" stroke="{color}" stroke-width="2.2"/>'
            f'<line x1="{cx-size:.1f}" y1="{cy+size:.1f}" x2="{cx+size:.1f}" y2="{cy-size:.1f}" stroke="{color}" stroke-width="2.2"/>'
        )
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size:.1f}" fill="{color}" stroke="{stroke}" stroke-width="0.8"/>'


def write_dimension_errorbar_svg(path, dim, point_rows, coef_rows):
    width, height, pad = 1120, 680, 78
    dim_points = [r for r in point_rows if r["dimension"] == dim]
    vals = []
    for row in dim_points:
        m, s = float(row["mean_mle"]), float(row["sigma_mle"])
        vals.extend([m, m - s, m + s])
    y_min, y_max = percentile(vals, 0.01), percentile(vals, 0.99)
    pad_y = max((y_max - y_min) * 0.10, 0.005)
    y_min, y_max = y_min - pad_y, y_max + pad_y
    x_min, x_max = -0.225, 0.225
    offsets = dict(zip(MODELS, [-0.010, -0.005, 0.0, 0.005, 0.010]))

    def sx(x): return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)
    def sy(y): return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{pad}" y="34" font-family="Arial" font-size="19">{html.escape(dim)}: alpha effect with MLE error bars</text>')
    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#111827"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#111827"/>')
    for x in ALPHAS:
        parts.append(f'<line x1="{sx(x):.1f}" y1="{height-pad}" x2="{sx(x):.1f}" y2="{height-pad+6}" stroke="#111827"/>')
        parts.append(f'<text x="{sx(x)-13:.1f}" y="{height-pad+22}" font-family="Arial" font-size="10">{x:g}</text>')
    minor_start = math.floor(y_min / 0.01) * 0.01
    minor_end = math.ceil(y_max / 0.01) * 0.01
    tick = minor_start
    while tick <= minor_end + 1e-9:
        if y_min <= tick <= y_max:
            parts.append(f'<line x1="{pad-3}" y1="{sy(tick):.1f}" x2="{pad}" y2="{sy(tick):.1f}" stroke="#64748b" stroke-width="0.6"/>')
            parts.append(f'<line x1="{pad}" y1="{sy(tick):.1f}" x2="{width-pad}" y2="{sy(tick):.1f}" stroke="#f8fafc"/>')
        tick += 0.01
    for y in nice_ticks(y_min, y_max, 6):
        parts.append(f'<line x1="{pad}" y1="{sy(y):.1f}" x2="{width-pad}" y2="{sy(y):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="8" y="{sy(y)+4:.1f}" font-family="Arial" font-size="11">{y:+.3f}</text>')
    if y_min < 0 < y_max:
        parts.append(f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{width-pad}" y2="{sy(0):.1f}" stroke="#94a3b8" stroke-dasharray="4,4"/>')
    for row in dim_points:
        model, alpha = row["model"], float(row["alpha"])
        x = alpha + offsets[model]
        m, s = float(row["mean_mle"]), float(row["sigma_mle"])
        color = alpha_color(alpha)
        parts.append(f'<line x1="{sx(x):.1f}" y1="{sy(m-s):.1f}" x2="{sx(x):.1f}" y2="{sy(m+s):.1f}" stroke="{color}" stroke-width="1.2" opacity="0.85"/>')
        parts.append(marker(sx(x), sy(m), model, color, 4.8))
    lx, ly = width - 320, 54
    for i, model in enumerate(MODELS):
        parts.append(marker(lx, ly + i * 23, model, "#e2e8f0", 5.0))
        parts.append(f'<text x="{lx+18}" y="{ly+i*23+4}" font-family="Arial" font-size="12">{html.escape(MODEL_LABELS[model])}</text>')
    parts.append('<text x="78" y="58" font-family="Arial" font-size="12" fill="#475569">MLE points only; no quadratic smoothing. Minor y-axis ticks are spaced by 0.01.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def dim_coeff_map(coef_rows):
    out = defaultdict(dict)
    for row in coef_rows:
        out[(row["model"], row["dimension"])] = (float(row["linear_coef_b"]), float(row["quadratic_coef_c"]))
    return out


def predicted_task_delta(model, task, alpha, caps, coefs):
    weights = caps[task]
    total = 0.0
    for dim, w in zip(DIMENSIONS, weights):
        b, c = coefs[(model, dim)]
        total += w * curve(b, c, alpha)
    return total


def write_task_svg(path, title, module, task, values, caps, coefs):
    width, height, pad = 980, 600, 76
    all_pts = []
    for model in MODELS:
        vals = values[(model, module)].get(task, {})
        if set(ALPHAS).issubset(vals):
            base = vals[0.0]
            all_pts.extend((a, vals[a] - base) for a in ALPHAS)
            all_pts.extend((x, predicted_task_delta(model, task, x, caps, coefs)) for x in XS)
    y_min, y_max = percentile([y for _, y in all_pts], 0.01), percentile([y for _, y in all_pts], 0.99)
    pad_y = max((y_max - y_min) * 0.12, 0.006)
    y_min, y_max = y_min - pad_y, y_max + pad_y
    x_min, x_max = -0.21, 0.21

    def sx(x): return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)
    def sy(y): return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{pad}" y="34" font-family="Arial" font-size="18">{html.escape(title)}</text>')
    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#111827"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#111827"/>')
    for x in ALPHAS:
        parts.append(f'<line x1="{sx(x):.1f}" y1="{height-pad}" x2="{sx(x):.1f}" y2="{height-pad+6}" stroke="#111827"/>')
        parts.append(f'<text x="{sx(x)-13:.1f}" y="{height-pad+22}" font-family="Arial" font-size="10">{x:g}</text>')
    for y in nice_ticks(y_min, y_max, 6):
        parts.append(f'<line x1="{pad}" y1="{sy(y):.1f}" x2="{width-pad}" y2="{sy(y):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="8" y="{sy(y)+4:.1f}" font-family="Arial" font-size="11">{y:+.3f}</text>')
    if y_min < 0 < y_max:
        parts.append(f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{width-pad}" y2="{sy(0):.1f}" stroke="#94a3b8" stroke-dasharray="4,4"/>')
    for model in MODELS:
        vals = values[(model, module)].get(task, {})
        if not set(ALPHAS).issubset(vals):
            continue
        d = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(predicted_task_delta(model,task,x,caps,coefs)):.1f}" for i, x in enumerate(XS))
        parts.append(f'<path d="{d}" fill="none" stroke="{MODEL_STROKES[model]}" stroke-width="1.8" opacity="0.72"/>')
        base = vals[0.0]
        for alpha in ALPHAS:
            parts.append(marker(sx(alpha), sy(vals[alpha] - base), model, alpha_color(alpha), 4.4))
    lx, ly = width - 305, 54
    for i, model in enumerate(MODELS):
        parts.append(marker(lx, ly + i * 22, model, "#e2e8f0", 4.8))
        parts.append(f'<text x="{lx+18}" y="{ly+i*22+4}" font-family="Arial" font-size="12">{html.escape(MODEL_LABELS[model])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def pca_2d(points):
    means = [sum(p[j] for p in points) / len(points) for j in range(len(points[0]))]
    centered = [[p[j] - means[j] for j in range(len(p))] for p in points]
    d = len(points[0])
    cov = [[0.0 for _ in range(d)] for _ in range(d)]
    for p in centered:
        for i in range(d):
            for j in range(d):
                cov[i][j] += p[i] * p[j]
    for i in range(d):
        for j in range(d):
            cov[i][j] /= max(1, len(points) - 1)

    def mv(mat, vec):
        return [sum(row[i] * vec[i] for i in range(len(vec))) for row in mat]

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    def power(mat, seed):
        rng = random.Random(seed)
        v = [rng.random() - 0.5 for _ in range(d)]
        v = [x / (norm(v) or 1.0) for x in v]
        for _ in range(120):
            y = mv(mat, v)
            ny = norm(y)
            if ny < 1e-12:
                break
            v = [x / ny for x in y]
        return v, dot(v, mv(mat, v))

    v1, l1 = power(cov, 17)
    cov2 = [[cov[i][j] - l1 * v1[i] * v1[j] for j in range(d)] for i in range(d)]
    v2, l2 = power(cov2, 23)
    total = sum(cov[i][i] for i in range(d)) or 1.0
    return [(dot(p, v1), dot(p, v2)) for p in centered], [l1 / total, l2 / total]


def pca_fit(points):
    means = [sum(p[j] for p in points) / len(points) for j in range(len(points[0]))]
    centered = [[p[j] - means[j] for j in range(len(p))] for p in points]
    d = len(points[0])
    cov = [[0.0 for _ in range(d)] for _ in range(d)]
    for p in centered:
        for i in range(d):
            for j in range(d):
                cov[i][j] += p[i] * p[j]
    for i in range(d):
        for j in range(d):
            cov[i][j] /= max(1, len(points) - 1)

    def mv(mat, vec):
        return [sum(row[i] * vec[i] for i in range(len(vec))) for row in mat]

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    def power(mat, seed):
        rng = random.Random(seed)
        v = [rng.random() - 0.5 for _ in range(d)]
        v = [x / (norm(v) or 1.0) for x in v]
        for _ in range(120):
            y = mv(mat, v)
            ny = norm(y)
            if ny < 1e-12:
                break
            v = [x / ny for x in y]
        return v, dot(v, mv(mat, v))

    v1, l1 = power(cov, 31)
    cov2 = [[cov[i][j] - l1 * v1[i] * v1[j] for j in range(d)] for i in range(d)]
    v2, l2 = power(cov2, 37)
    total = sum(cov[i][i] for i in range(d)) or 1.0
    return {"mean": means, "components": [v1, v2], "variance_ratio": [l1 / total, l2 / total]}


def pca_project(points, fit):
    centered = [[p[j] - fit["mean"][j] for j in range(len(p))] for p in points]
    return [(dot(p, fit["components"][0]), dot(p, fit["components"][1])) for p in centered]


def tsne_2d(points, init, perplexity=30.0, seed=7):
    n = len(points)
    if n < 3:
        return init
    distances = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum((points[i][k] - points[j][k]) ** 2 for k in range(len(points[i])))
            distances[i][j] = distances[j][i] = dist
    target_h = math.log(min(perplexity, max(2.0, (n - 1) / 3)))
    cond = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row = []
        lo, hi = 1e-3, 1e3
        for _ in range(40):
            beta = (lo + hi) / 2
            vals = [0.0 if i == j else math.exp(-distances[i][j] * beta) for j in range(n)]
            z = sum(vals) or 1e-12
            probs = [v / z for v in vals]
            h = -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0)
            if h > target_h:
                lo = beta
            else:
                hi = beta
            row = probs
        cond[i] = row
    p = [[max((cond[i][j] + cond[j][i]) / (2 * n), 1e-12) for j in range(n)] for i in range(n)]
    rng = random.Random(seed)
    y = [[init[i][0] * 0.02 + (rng.random() - 0.5) * 1e-4, init[i][1] * 0.02 + (rng.random() - 0.5) * 1e-4] for i in range(n)]
    inc = [[0.0, 0.0] for _ in range(n)]
    for it in range(420):
        num = [[0.0 for _ in range(n)] for _ in range(n)]
        z = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = y[i][0] - y[j][0], y[i][1] - y[j][1]
                val = 1.0 / (1.0 + dx * dx + dy * dy)
                num[i][j] = num[j][i] = val
                z += 2 * val
        z = max(z, 1e-12)
        exag = 4.0 if it < 120 else 1.0
        lr = 100.0
        mom = 0.5 if it < 150 else 0.8
        grads = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            gx = gy = 0.0
            for j in range(n):
                if i == j:
                    continue
                q = max(num[i][j] / z, 1e-12)
                mult = 4.0 * (exag * p[i][j] - q) * num[i][j]
                gx += mult * (y[i][0] - y[j][0])
                gy += mult * (y[i][1] - y[j][1])
            grads[i] = [gx, gy]
        for i in range(n):
            inc[i][0] = mom * inc[i][0] + lr * grads[i][0]
            inc[i][1] = mom * inc[i][1] + lr * grads[i][1]
            y[i][0] += inc[i][0]
            y[i][1] += inc[i][1]
        mx, my = sum(v[0] for v in y) / n, sum(v[1] for v in y) / n
        for i in range(n):
            y[i][0] -= mx
            y[i][1] -= my
    return [(v[0], v[1]) for v in y]


def project_to_tsne_space(query_points, reference_points, reference_coords, k=12):
    projected = []
    for point in query_points:
        distances = []
        for idx, ref in enumerate(reference_points):
            d2 = sum((point[j] - ref[j]) ** 2 for j in range(len(point)))
            distances.append((d2, idx))
        distances.sort(key=lambda x: x[0])
        nearest = distances[: max(2, min(k, len(distances)))]
        if nearest[0][0] < 1e-14:
            projected.append(reference_coords[nearest[0][1]])
            continue
        sigma2 = max(nearest[-1][0], 1e-12)
        weights = [math.exp(-d2 / (2.0 * sigma2)) for d2, _ in nearest]
        total = sum(weights) or 1.0
        x = sum(w * reference_coords[idx][0] for w, (_, idx) in zip(weights, nearest)) / total
        y = sum(w * reference_coords[idx][1] for w, (_, idx) in zip(weights, nearest)) / total
        projected.append((x, y))
    return projected


def write_embedding_svg(path, title, rows, coords, x_label, y_label):
    width, height, pad = 1120, 760, 82
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad, y_pad = max((x_max - x_min) * 0.08, 0.01), max((y_max - y_min) * 0.08, 0.01)
    x_min, x_max, y_min, y_max = x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad

    def sx(x): return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)
    def sy(y): return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{pad}" y="34" font-family="Arial" font-size="18">{html.escape(title)}</text>')
    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#111827"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#111827"/>')
    for x in nice_ticks(x_min, x_max, 6):
        parts.append(f'<line x1="{sx(x):.1f}" y1="{pad}" x2="{sx(x):.1f}" y2="{height-pad}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{sx(x)-20:.1f}" y="{height-pad+22}" font-family="Arial" font-size="10">{x:+.2f}</text>')
    for y in nice_ticks(y_min, y_max, 6):
        parts.append(f'<line x1="{pad}" y1="{sy(y):.1f}" x2="{width-pad}" y2="{sy(y):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="10" y="{sy(y)+4:.1f}" font-family="Arial" font-size="10">{y:+.2f}</text>')
    for row, (x, y) in zip(rows, coords):
        alpha = float(row["best_alpha"])
        parts.append(marker(sx(x), sy(y), row["model"], alpha_color(alpha), 3.6))
    lx, ly = width - 320, 54
    for i, model in enumerate(MODELS):
        parts.append(marker(lx, ly + i * 22, model, "#e2e8f0", 4.8))
        parts.append(f'<text x="{lx+18}" y="{ly+i*22+4}" font-family="Arial" font-size="12">{html.escape(MODEL_LABELS[model])}</text>')
    parts.append(f'<text x="{width/2-36:.1f}" y="{height-18}" font-family="Arial" font-size="12">{html.escape(x_label)}</text>')
    parts.append(f'<text x="14" y="58" font-family="Arial" font-size="12">{html.escape(y_label)}</text>')
    parts.append('<text x="82" y="58" font-family="Arial" font-size="12" fill="#475569">Each point is one model-subtask. Color encodes alpha where that subtask reaches max score.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def response_vector(model, module, task, values):
    vals = values[(model, module)][task]
    base = vals[0.0]
    return [vals[alpha] - base for alpha in FIT_ALPHAS]


def predicted_vector(model, task, caps, coefs):
    return [predicted_task_delta(model, task, alpha, caps, coefs) for alpha in FIT_ALPHAS]


def dimension_unit_vector(model, dim, coefs):
    b, c = coefs[(model, dim)]
    return [curve(b, c, alpha) for alpha in FIT_ALPHAS]


def paired_embedding_rows(model, values, caps, coefs):
    rows = []
    pair_ids = []
    pair_index = 0
    for module in MODULES:
        for task, vals in sorted(values[(model, module)].items()):
            if task not in caps or not set(ALPHAS).issubset(vals):
                continue
            best_alpha = max(ALPHAS, key=lambda a: (vals[a], -abs(a)))
            common = {
                "model": model,
                "module": module,
                "task": task,
                "best_alpha": best_alpha,
                "pair_id": pair_index,
            }
            rows.append({**common, "kind": "actual", "label": task, "vector": response_vector(model, module, task, values)})
            rows.append({**common, "kind": "predicted", "label": task, "vector": predicted_vector(model, task, caps, coefs)})
            pair_ids.append(pair_index)
            pair_index += 1
    for dim in DIMENSIONS:
        rows.append(
            {
                "model": model,
                "module": "unit",
                "task": dim,
                "best_alpha": 0.0,
                "pair_id": "",
                "kind": "unit",
                "label": dim,
                "vector": dimension_unit_vector(model, dim, coefs),
            }
        )
    return rows


def write_paired_embedding_csv(path, rows, coords, x_name, y_name):
    out = []
    for row, (x, y) in zip(rows, coords):
        item = {
            "model": row["model"],
            "module": row["module"],
            "task": row["task"],
            "kind": row["kind"],
            "pair_id": row["pair_id"],
            "best_alpha": row["best_alpha"],
            x_name: x,
            y_name: y,
        }
        for alpha, val in zip(FIT_ALPHAS, row["vector"]):
            item[f"delta_{alpha:g}"] = val
        out.append(item)
    write_csv(path, out)


def write_paired_embedding_svg(path, title, rows, coords, x_label, y_label, var_text=""):
    width, height, pad = 1120, 780, 86
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad, y_pad = max((x_max - x_min) * 0.09, 0.01), max((y_max - y_min) * 0.09, 0.01)
    x_min, x_max, y_min, y_max = x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad

    def sx(x): return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)
    def sy(y): return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    by_pair = defaultdict(dict)
    unit_points = []
    for idx, row in enumerate(rows):
        if row["kind"] == "unit":
            unit_points.append((row, coords[idx]))
        else:
            by_pair[row["pair_id"]][row["kind"]] = (row, coords[idx])

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{pad}" y="34" font-family="Arial" font-size="18">{html.escape(title)}</text>')
    subtitle = "Actual and predicted points are connected per subtask; unit points are pure capability-effect vectors."
    if var_text:
        subtitle += " " + var_text
    parts.append(f'<text x="{pad}" y="58" font-family="Arial" font-size="12" fill="#475569">{html.escape(subtitle)}</text>')
    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#111827"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#111827"/>')
    for x in nice_ticks(x_min, x_max, 6):
        parts.append(f'<line x1="{sx(x):.1f}" y1="{pad}" x2="{sx(x):.1f}" y2="{height-pad}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{sx(x)-21:.1f}" y="{height-pad+22}" font-family="Arial" font-size="10">{x:+.2f}</text>')
    for y in nice_ticks(y_min, y_max, 6):
        parts.append(f'<line x1="{pad}" y1="{sy(y):.1f}" x2="{width-pad}" y2="{sy(y):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="10" y="{sy(y)+4:.1f}" font-family="Arial" font-size="10">{y:+.2f}</text>')

    for pair in by_pair.values():
        if "actual" not in pair or "predicted" not in pair:
            continue
        row, (ax, ay) = pair["actual"]
        _, (px, py) = pair["predicted"]
        color = alpha_color(float(row["best_alpha"]))
        parts.append(f'<line x1="{sx(ax):.1f}" y1="{sy(ay):.1f}" x2="{sx(px):.1f}" y2="{sy(py):.1f}" stroke="{color}" stroke-width="0.75" opacity="0.55"/>')
    for pair in by_pair.values():
        if "actual" in pair:
            row, (x, y) = pair["actual"]
            color = alpha_color(float(row["best_alpha"]))
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.3" fill="{color}" stroke="#111827" stroke-width="0.45" opacity="0.88"/>')
        if "predicted" in pair:
            row, (x, y) = pair["predicted"]
            color = alpha_color(float(row["best_alpha"]))
            parts.append(f'<rect x="{sx(x)-3.0:.1f}" y="{sy(y)-3.0:.1f}" width="6.0" height="6.0" fill="white" stroke="{color}" stroke-width="1.25" opacity="0.95"/>')
    for row, (x, y) in unit_points:
        cx, cy = sx(x), sy(y)
        parts.append(f'<path d="M {cx:.1f},{cy-8:.1f} L {cx+2.5:.1f},{cy-2.5:.1f} L {cx+8:.1f},{cy:.1f} L {cx+2.5:.1f},{cy+2.5:.1f} L {cx:.1f},{cy+8:.1f} L {cx-2.5:.1f},{cy+2.5:.1f} L {cx-8:.1f},{cy:.1f} L {cx-2.5:.1f},{cy-2.5:.1f} Z" fill="#111827"/>')
        short = {"Factual Knowledge": "FK", "Language Understanding": "LU", "Deductive Reasoning": "DR"}[row["task"]]
        parts.append(f'<text x="{cx+10:.1f}" y="{cy-7:.1f}" font-family="Arial" font-size="12" fill="#111827">{short}</text>')

    lx, ly = width - 300, 74
    parts.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="#94a3b8" stroke="#111827" stroke-width="0.45"/>')
    parts.append(f'<text x="{lx+15}" y="{ly+4}" font-family="Arial" font-size="12">actual subtask</text>')
    parts.append(f'<rect x="{lx-4}" y="{ly+17}" width="8" height="8" fill="white" stroke="#94a3b8" stroke-width="1.25"/>')
    parts.append(f'<text x="{lx+15}" y="{ly+25}" font-family="Arial" font-size="12">predicted subtask</text>')
    parts.append(f'<path d="M {lx},{ly+40} L {lx+2.5},{ly+45.5} L {lx+8},{ly+48} L {lx+2.5},{ly+50.5} L {lx},{ly+56} L {lx-2.5},{ly+50.5} L {lx-8},{ly+48} L {lx-2.5},{ly+45.5} Z" fill="#111827"/>')
    parts.append(f'<text x="{lx+15}" y="{ly+52}" font-family="Arial" font-size="12">capability unit</text>')
    parts.append(f'<text x="{width/2-36:.1f}" y="{height-18}" font-family="Arial" font-size="12">{html.escape(x_label)}</text>')
    parts.append(f'<text x="14" y="62" font-family="Arial" font-size="12">{html.escape(y_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def independent_capability_mean_map(point_rows):
    out = {}
    for row in point_rows:
        out[(row["model"], float(row["alpha"]), row["dimension"])] = float(row["mean_mle"])
    return out


def predicted_gain_independent(model, task, alpha, caps, mean_map):
    weights = caps[task]
    return sum(weights[i] * mean_map[(model, alpha, dim)] for i, dim in enumerate(DIMENSIONS))


def error_area_rows(model, values, caps, mean_map):
    rows = []
    for module in MODULES:
        for task, vals in sorted(values[(model, module)].items()):
            if task not in caps or not set(ALPHAS).issubset(vals):
                continue
            base = vals[0.0]
            gains = {alpha: vals[alpha] - base for alpha in ALPHAS}
            best_alpha = max(ALPHAS, key=lambda a: (gains[a], -abs(a)))
            actual_gain = gains[best_alpha]
            fitted_gain = predicted_gain_independent(model, task, best_alpha, caps, mean_map)
            errors = []
            for alpha in FIT_ALPHAS:
                pred = predicted_gain_independent(model, task, alpha, caps, mean_map)
                errors.append(gains[alpha] - pred)
            error_mse = sum(err * err for err in errors) / len(errors)
            error_rmse = math.sqrt(error_mse)
            row = {
                "model": model,
                "module": module,
                "task": task,
                "kind": "task",
                "best_alpha": best_alpha,
                "actual_gain": actual_gain,
                "gain_area_value": max(actual_gain, 0.0),
                "fitted_gain": fitted_gain,
                "gain_residual": actual_gain - fitted_gain,
                "error_mse": error_mse,
                "error_rmse": error_rmse,
            }
            for dim, val in zip(DIMENSIONS, caps[task]):
                row[dim] = val
            rows.append(row)
    for dim in DIMENSIONS:
        row = {
            "model": model,
            "module": "unit",
            "task": dim,
            "kind": "unit",
            "best_alpha": 0.0,
            "actual_gain": 0.0,
            "gain_area_value": 0.0,
            "fitted_gain": 0.0,
            "gain_residual": 0.0,
            "error_mse": 0.0,
            "error_rmse": 0.0,
        }
        for each in DIMENSIONS:
            row[each] = 1.0 if each == dim else 0.0
        rows.append(row)
    return rows


def write_error_area_csv(path, rows, coords):
    out = []
    for row, (x, y) in zip(rows, coords):
        item = {k: v for k, v in row.items() if k != "vector"}
        item["x2d"] = x
        item["y2d"] = y
        out.append(item)
    write_csv(path, out)


def write_error_area_svg(path, title, rows, coords, method_label, var_text=""):
    width, height, pad = 1080, 760, 84
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad, y_pad = max((x_max - x_min) * 0.09, 0.01), max((y_max - y_min) * 0.09, 0.01)
    x_min, x_max, y_min, y_max = x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad

    def sx(x):
        return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)

    def sy(y):
        return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    task_gains = [float(r["gain_area_value"]) for r in rows if r["kind"] == "task"]
    max_gain = max(task_gains) if task_gains else 1.0
    # Area is proportional to positive measured gain, so radius follows sqrt(gain).
    def gain_radius(gain):
        return 2.0 + 28.0 * math.sqrt(max(gain, 0.0) / max(max_gain, 1e-12))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{pad}" y="34" font-family="Arial" font-size="18">{html.escape(title)}</text>')
    subtitle = f"{method_label} of 3D capability-demand weights plus FK/LU/DR units. Color is alpha of maximum actual gain; same-color area is positive measured gain at that alpha."
    if var_text:
        subtitle += " " + var_text
    parts.append(f'<text x="{pad}" y="58" font-family="Arial" font-size="12" fill="#475569">{html.escape(subtitle)}</text>')

    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#111827"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#111827"/>')
    for x in nice_ticks(x_min, x_max, 6):
        parts.append(f'<line x1="{sx(x):.1f}" y1="{pad}" x2="{sx(x):.1f}" y2="{height-pad}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{sx(x)-22:.1f}" y="{height-pad+22}" font-family="Arial" font-size="10">{x:+.2f}</text>')
    for y in nice_ticks(y_min, y_max, 6):
        parts.append(f'<line x1="{pad}" y1="{sy(y):.1f}" x2="{width-pad}" y2="{sy(y):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="10" y="{sy(y)+4:.1f}" font-family="Arial" font-size="10">{y:+.2f}</text>')

    # Draw gain area first, then task point.
    for idx, row in enumerate(rows):
        if row["kind"] != "task":
            continue
        x, y = coords[idx]
        color = alpha_color(float(row["best_alpha"]))
        r = gain_radius(float(row["gain_area_value"]))
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r:.2f}" fill="{color}" opacity="0.18"/>')
    for idx, row in enumerate(rows):
        if row["kind"] != "task":
            continue
        x, y = coords[idx]
        color = alpha_color(float(row["best_alpha"]))
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.9" fill="{color}" stroke="#111827" stroke-width="0.45" opacity="0.94"/>')

    for idx, row in enumerate(rows):
        if row["kind"] != "unit":
            continue
        x, y = coords[idx]
        cx, cy = sx(x), sy(y)
        parts.append(f'<path d="M {cx:.1f},{cy-9:.1f} L {cx+2.8:.1f},{cy-2.8:.1f} L {cx+9:.1f},{cy:.1f} L {cx+2.8:.1f},{cy+2.8:.1f} L {cx:.1f},{cy+9:.1f} L {cx-2.8:.1f},{cy+2.8:.1f} L {cx-9:.1f},{cy:.1f} L {cx-2.8:.1f},{cy-2.8:.1f} Z" fill="#111827"/>')
        short = {"Factual Knowledge": "FK", "Language Understanding": "LU", "Deductive Reasoning": "DR"}[row["task"]]
        parts.append(f'<text x="{cx+11:.1f}" y="{cy-8:.1f}" font-family="Arial" font-size="12" fill="#111827">{short}</text>')

    lx, ly = width - 260, 86
    for idx, alpha in enumerate([-0.2, -0.1, 0.0, 0.1, 0.2]):
        x = lx + idx * 42
        parts.append(f'<circle cx="{x}" cy="{ly}" r="4" fill="{alpha_color(alpha)}" stroke="#111827" stroke-width="0.4"/>')
        parts.append(f'<text x="{x-14}" y="{ly+18}" font-family="Arial" font-size="10">{alpha:g}</text>')
    parts.append(f'<text x="{lx}" y="{ly-14}" font-family="Arial" font-size="12">best alpha</text>')
    for idx, frac in enumerate([0.25, 0.5, 1.0]):
        r = 2.0 + 28.0 * math.sqrt(frac)
        cy = ly + 58 + idx * 32
        parts.append(f'<circle cx="{lx+8}" cy="{cy}" r="{r:.2f}" fill="#64748b" opacity="0.18"/>')
        parts.append(f'<circle cx="{lx+8}" cy="{cy}" r="3.5" fill="#64748b" stroke="#111827" stroke-width="0.4"/>')
        parts.append(f'<text x="{lx+48}" y="{cy+4}" font-family="Arial" font-size="11">{frac:.2f} max gain</text>')
    parts.append(f'<text x="{width/2-82:.1f}" y="{height-26}" font-family="Arial" font-size="12">{html.escape(method_label)} axis 1</text>')
    parts.append(f'<text x="16" y="{pad-18}" font-family="Arial" font-size="12">{html.escape(method_label)} axis 2</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    caps = read_caps()
    score_rows = extract_scores()
    write_csv(OUT / "eval6_all_models_subtask_scores_long.csv", score_rows)
    values, _ = build_values(score_rows)
    cov, missing_rows = coverage(values, caps)
    write_csv(OUT / "coverage.csv", cov)
    write_csv(OUT / "missing_task_series.csv", missing_rows)
    point_rows, coef_rows, task_sets = capability_fit(values, caps)
    write_csv(OUT / "capability_alpha_mle_errorbar_points.csv", point_rows)
    write_csv(OUT / "capability_alpha_quadratic_curves.csv", coef_rows)
    slopes = task_slopes(values, caps)
    write_csv(OUT / "task_alpha_quadratic_slopes.csv", slopes)
    mean_map = independent_capability_mean_map(point_rows)

    images = []
    for dim in DIMENSIONS:
        path = OUT / f"dimension_{dim.lower().replace(' ', '_')}_errorbars.svg"
        write_dimension_errorbar_svg(path, dim, point_rows, coef_rows)
        images.append(path)

    coefs = dim_coeff_map(coef_rows)
    selected = [
        ("mmlu_pro", "mmlu_pro_math"),
        ("mmlu_pro", "mmlu_pro_computer_science"),
        ("mmlu_redux", "mmlu_redux_professional_law_generative"),
        ("agieval", "agieval_logiqa_en"),
        ("agieval", "agieval_sat_math"),
        ("bbh", "bbh_zeroshot_logical_deduction_three_objects"),
        ("bbh", "bbh_zeroshot_multistep_arithmetic_two"),
    ]
    for module, task in selected:
        if task not in caps:
            continue
        path = OUT / f"task_fit_{module}_{task}.svg"
        write_task_svg(path, f"{module}/{task}: observed deltas and capability fit", module, task, values, caps, coefs)
        images.append(path)

    embed_rows = slopes
    embed_points = [[float(r[dim]) for dim in DIMENSIONS] for r in embed_rows]
    pca_coords, var = pca_2d(embed_points)
    pca_out = []
    for row, (x, y) in zip(embed_rows, pca_coords):
        out = dict(row)
        out["pc1"] = x
        out["pc2"] = y
        pca_out.append(out)
    write_csv(OUT / "task_capability_weight_pca2d.csv", pca_out)
    pca_path = OUT / "task_capability_weight_pca2d.svg"
    write_embedding_svg(pca_path, f"Eval6 model-subtask PCA on 3D capability weights (PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%)", embed_rows, pca_coords, "PC1", "PC2")
    images.append(pca_path)
    tsne_coords = tsne_2d(embed_points, pca_coords)
    tsne_out = []
    for row, (x, y) in zip(embed_rows, tsne_coords):
        out = dict(row)
        out["tsne1"] = x
        out["tsne2"] = y
        tsne_out.append(out)
    write_csv(OUT / "task_capability_weight_tsne2d.csv", tsne_out)
    tsne_path = OUT / "task_capability_weight_tsne2d.svg"
    write_embedding_svg(tsne_path, "Eval6 model-subtask t-SNE on 3D capability weights", embed_rows, tsne_coords, "t-SNE 1", "t-SNE 2")
    images.append(tsne_path)

    paired_images = []
    paired_csvs = []
    gain_area_images = []
    gain_area_csvs = []
    for model in MODELS:
        rows = paired_embedding_rows(model, values, caps, coefs)
        actual_rows = [row for row in rows if row["kind"] == "actual"]
        projected_rows = [row for row in rows if row["kind"] != "actual"]
        actual_points = [row["vector"] for row in actual_rows]
        projected_points = [row["vector"] for row in projected_rows]

        pca_fit_obj = pca_fit(actual_points)
        actual_pca_coords = pca_project(actual_points, pca_fit_obj)
        projected_pca_coords = pca_project(projected_points, pca_fit_obj)
        pca_rows = actual_rows + projected_rows
        pca_coords = actual_pca_coords + projected_pca_coords
        var = pca_fit_obj["variance_ratio"]
        pca_csv = OUT / f"{model}_paired_actual_predicted_pca2d.csv"
        write_paired_embedding_csv(pca_csv, pca_rows, pca_coords, "pc1", "pc2")
        paired_csvs.append(pca_csv)
        pca_path = OUT / f"{model}_paired_actual_predicted_pca2d.svg"
        write_paired_embedding_svg(
            pca_path,
            f"{MODEL_LABELS[model]}: actual vs capability-predicted subtask response PCA",
            pca_rows,
            pca_coords,
            "PC1",
            "PC2",
            f"PCA fit uses actual subtask vectors only. PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%.",
        )
        paired_images.append(pca_path)
        actual_tsne_coords = tsne_2d(actual_points, actual_pca_coords, perplexity=22.0, seed=20260506 + MODELS.index(model))
        projected_tsne_coords = project_to_tsne_space(projected_points, actual_points, actual_tsne_coords)
        tsne_rows = actual_rows + projected_rows
        tsne_coords = actual_tsne_coords + projected_tsne_coords
        tsne_csv = OUT / f"{model}_paired_actual_predicted_tsne2d.csv"
        write_paired_embedding_csv(tsne_csv, tsne_rows, tsne_coords, "tsne1", "tsne2")
        paired_csvs.append(tsne_csv)
        tsne_path = OUT / f"{model}_paired_actual_predicted_tsne2d.svg"
        write_paired_embedding_svg(
            tsne_path,
            f"{MODEL_LABELS[model]}: actual vs capability-predicted subtask response t-SNE",
            tsne_rows,
            tsne_coords,
            "t-SNE 1",
            "t-SNE 2",
            "t-SNE fit uses actual subtask vectors only; predicted/unit points use neighbor interpolation into that space.",
        )
        paired_images.append(tsne_path)
        area_rows = error_area_rows(model, values, caps, mean_map)
        area_points = [[float(row[dim]) for dim in DIMENSIONS] for row in area_rows]
        area_pca_coords, area_var = pca_2d(area_points)
        area_pca_csv = OUT / f"{model}_capability_demand_gain_area_pca2d.csv"
        write_error_area_csv(area_pca_csv, area_rows, area_pca_coords)
        gain_area_csvs.append(area_pca_csv)
        area_pca_path = OUT / f"{model}_capability_demand_gain_area_pca2d.svg"
        write_error_area_svg(
            area_pca_path,
            f"{MODEL_LABELS[model]}: capability-demand PCA with measured gain area",
            area_rows,
            area_pca_coords,
            "PCA",
            f"PC1 {area_var[0]*100:.1f}%, PC2 {area_var[1]*100:.1f}%.",
        )
        gain_area_images.append(area_pca_path)

        area_tsne_coords = tsne_2d(area_points, area_pca_coords, perplexity=22.0, seed=20260516 + MODELS.index(model))
        area_tsne_csv = OUT / f"{model}_capability_demand_gain_area_tsne2d.csv"
        write_error_area_csv(area_tsne_csv, area_rows, area_tsne_coords)
        gain_area_csvs.append(area_tsne_csv)
        area_tsne_path = OUT / f"{model}_capability_demand_gain_area_tsne2d.svg"
        write_error_area_svg(
            area_tsne_path,
            f"{MODEL_LABELS[model]}: capability-demand t-SNE with measured gain area",
            area_rows,
            area_tsne_coords,
            "t-SNE",
        )
        gain_area_images.append(area_tsne_path)
    images.extend(paired_images)
    images.extend(gain_area_images)

    cov_rows = [[MODEL_LABELS[r["model"]], r["module"], f'{r["complete_tasks"]}/{r["available_tasks"]}', "yes" if r["complete"] else "no"] for r in cov]
    summary_rows = []
    for model in MODELS:
        for dim in DIMENSIONS:
            rows = [r for r in coef_rows if r["model"] == model and r["dimension"] == dim]
            b, c = float(rows[0]["linear_coef_b"]), float(rows[0]["quadratic_coef_c"])
            summary_rows.append([MODEL_LABELS[model], dim, f"{b:+.4f}", f"{c:+.4f}", f"{curve(b,c,0.2):+.4f}", f"{curve(b,c,-0.2):+.4f}"])
    missing_summary = []
    missing_by_model_module = defaultdict(int)
    for row in missing_rows:
        missing_by_model_module[(row["model"], row["module"])] += 1
    for (model, module), count in sorted(missing_by_model_module.items()):
        missing_summary.append([MODEL_LABELS[model], module, count])
    REPORT.write_text(
        "\n".join(
            [
                "# Eval6 All-Model Alpha9 Capability Analysis",
                "",
                "Scope: five models over alpha `[-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2]` and eval6 modules `mmlu_pro`, `mmlu_redux`, `agieval`, `bbh`.",
                "",
                "Capability weights use the DeepSeek eval6 judge table collapsed to three primary dimensions: Factual Knowledge, Language Understanding, and Deductive Reasoning. Weights are renormalized per subtask before fitting.",
                "",
                "For each model and alpha, capability means and sigmas are fitted by maximum likelihood over all complete eval6 subtasks. A zero-intercept quadratic is then fitted through nonzero alpha capability means. Task plots overlay observed subtask deltas with the capability-projected quadratic fit.",
                "",
                "## Coverage",
                "",
                md_table(["model", "module", "complete tasks", "complete"], cov_rows),
                "",
                "Incomplete task series are omitted from the capability fit and embedding plots. This currently affects Llama 3.1 8B BBH at alpha `-0.1` and Qwen3 30B-A3B AGIEval at alpha `-0.2`.",
                "",
                md_table(["model", "module", "incomplete task series"], missing_summary) if missing_summary else "No incomplete task series.",
                "",
                "## Capability Quadratic Summary",
                "",
                md_table(["model", "dimension", "linear b", "quadratic c", "fit +0.2", "fit -0.2"], summary_rows),
                "",
                "## Figures",
                "",
                *[f"![{path.stem}]({path})" for path in images],
                "",
                "## Paired Embedding CSVs",
                "",
                *[f"- `{path}`" for path in paired_csvs],
                "",
                "## Gain Area Embedding CSVs",
                "",
                *[f"- `{path}`" for path in gain_area_csvs],
                "",
            ]
        )
    )
    print(REPORT)
    for path in images:
        print(path)


if __name__ == "__main__":
    main()
