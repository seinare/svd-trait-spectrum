#!/usr/bin/env python3
"""Fit eval6 capability curves per model and benchmark module."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs" / "raw" / "eval6_llama"
RESULTS = ROOT / "docs" / "results"
CAP_CSV = RESULTS / "capability_dimension" / "deepseek_v4_pro_eval6_batch10x10_20260504_141548.csv"
OUT = RESULTS / "eval6_module_capability_curves"
REPORT = OUT / "llama1b_3b_eval6_module_capability_curves.md"

MODELS = ["llama1b", "llama3b"]
MODEL_LABELS = {
    "llama1b": "Llama 3.2 1B Instruct",
    "llama3b": "Llama 3.2 3B Instruct",
}
MODEL_COLORS = {"llama1b": "#2563eb", "llama3b": "#16a34a"}
MODULES = ["mmlu_pro", "mmlu_redux", "agieval", "bbh"]
ALPHAS = [-0.2, -0.1, 0.0, 0.1, 0.2]
FIT_ALPHAS = [a for a in ALPHAS if a != 0.0]
XS = [round(-0.2 + i * 0.004, 3) for i in range(101)]
DIMENSIONS = [
    "Factual Knowledge",
    "Language Understanding",
    "Inductive Reasoning",
    "Deductive Reasoning",
    "Mathematical Computation",
    "Structural Analysis",
    "Ethical & Safety Judgment",
]
P = len(DIMENSIONS) * 2


def model_from_path(path: Path) -> str:
    text = str(path)
    if "llama1b_alpha5_eval6" in text:
        return "llama1b"
    if "llama3b_alpha5_eval6" in text:
        return "llama3b"
    raise ValueError(f"cannot infer model from {path}")


def module_from_path(path: Path) -> str:
    parts = set(path.parts)
    for module in MODULES:
        if module in parts:
            return module
    raise ValueError(f"cannot infer module from {path}")


def alpha_from_path(path: Path) -> float:
    for part in path.parts:
        match = re.search(r"_alpha(-?\d+(?:\.\d+)?)_vllm$", part)
        if match:
            return round(float(match.group(1)), 3)
    raise ValueError(f"cannot infer alpha from {path}")


def metric_for(module: str, task_row: dict[str, float]) -> tuple[str, float]:
    preferences = {
        "mmlu_pro": ["exact_match,custom-extract"],
        "mmlu_redux": ["exact_match,default"],
        "agieval": ["acc_norm,none", "acc,none"],
        "bbh": ["exact_match,flexible-extract", "exact_match,strict-match"],
    }[module]
    for key in preferences:
        if key in task_row:
            return key, float(task_row[key])
    for key, value in task_row.items():
        if key == "alias" or key.endswith("_stderr"):
            continue
        if isinstance(value, int | float):
            return key, float(value)
    raise ValueError(f"no metric found for {module}: {task_row}")


def extract_scores() -> list[dict[str, object]]:
    rows = []
    for path in sorted(RAW.rglob("results_*.json")):
        model = model_from_path(path)
        module = module_from_path(path)
        alpha = alpha_from_path(path)
        if alpha not in ALPHAS:
            continue
        payload = json.loads(path.read_text())
        for task, values in payload["results"].items():
            metric, value = metric_for(module, values)
            rows.append(
                {
                    "model": model,
                    "module": module,
                    "alpha": alpha,
                    "task": task,
                    "metric": metric,
                    "value": value,
                    "source": str(path),
                }
            )
    return rows


def read_caps() -> dict[str, dict[str, object]]:
    out = {}
    with CAP_CSV.open() as handle:
        for row in csv.DictReader(handle):
            out[row["task"]] = {
                "group": row["group"],
                "weights": [float(row[dim]) for dim in DIMENSIONS],
            }
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def design_row(alpha: float, weights: list[float]) -> list[float]:
    row = []
    for weight in weights:
        row.extend([weight * alpha, weight * alpha * alpha])
    return row


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


def projected_variance(alpha: float, weights: list[float], taus: list[float]) -> float:
    return max(alpha * alpha * sum((w * tau) ** 2 for w, tau in zip(weights, taus)), 1e-12)


def predict(beta: list[float], x: list[float]) -> float:
    return sum(beta[i] * x[i] for i in range(P))


def curve(beta: list[float], dim_idx: int, alpha: float) -> float:
    return beta[2 * dim_idx] * alpha + beta[2 * dim_idx + 1] * alpha * alpha


def task_prediction(beta: list[float], weights: list[float], alpha: float) -> float:
    return sum(weights[i] * curve(beta, i, alpha) for i in range(len(DIMENSIONS)))


def fit_beta(obs, taus: list[float], ridge: float = 1e-5) -> list[float]:
    xtx = [[0.0 for _ in range(P)] for _ in range(P)]
    xty = [0.0 for _ in range(P)]
    for _, alpha, weights, x, y in obs:
        w = 1.0 / projected_variance(alpha, weights, taus)
        for i in range(P):
            xty[i] += w * x[i] * y
            for j in range(P):
                xtx[i][j] += w * x[i] * x[j]
    for i in range(P):
        xtx[i][i] += ridge
    return solve_linear(xtx, xty)


def nll(obs, beta: list[float], taus: list[float]) -> float:
    total = 0.0
    for _, alpha, weights, x, y in obs:
        res = y - predict(beta, x)
        var = projected_variance(alpha, weights, taus)
        total += 0.5 * (math.log(2 * math.pi * var) + res * res / var)
    return total


def optimize_taus(obs, beta: list[float], taus: list[float]) -> list[float]:
    z = [math.log(max(t, 1e-6) ** 2) for t in taus]
    best = nll(obs, beta, [math.sqrt(math.exp(v)) for v in z])
    step = 0.12
    for _ in range(120):
        q = [math.exp(v) for v in z]
        tau_tmp = [math.sqrt(v) for v in q]
        grad_q = [0.0 for _ in DIMENSIONS]
        for _, alpha, weights, x, y in obs:
            res = y - predict(beta, x)
            var = projected_variance(alpha, weights, tau_tmp)
            common = 0.5 * (1.0 / var - res * res / (var * var))
            for k, weight in enumerate(weights):
                grad_q[k] += common * alpha * alpha * weight * weight
        grad_z = [grad_q[k] * q[k] for k in range(len(q))]
        if math.sqrt(sum(g * g for g in grad_z)) < 1e-7:
            break
        local = step
        accepted = False
        for _ in range(24):
            cand_z = [max(-18.0, min(4.0, z[k] - local * grad_z[k])) for k in range(len(z))]
            cand_taus = [math.sqrt(math.exp(v)) for v in cand_z]
            cand = nll(obs, beta, cand_taus)
            if cand <= best:
                z, best, accepted = cand_z, cand, True
                step = min(local * 1.05, 0.5)
                break
            local *= 0.5
        if not accepted:
            break
    return [math.sqrt(math.exp(v)) for v in z]


def fit_module(score_rows: list[dict[str, object]], caps: dict[str, dict[str, object]], model: str, module: str):
    values = defaultdict(dict)
    metrics = {}
    for row in score_rows:
        if row["model"] == model and row["module"] == module:
            values[row["task"]][float(row["alpha"])] = float(row["value"])
            metrics[row["task"]] = row["metric"]
    tasks = sorted(t for t, vals in values.items() if set(ALPHAS).issubset(vals) and t in caps)
    obs = []
    for task in tasks:
        weights = caps[task]["weights"]
        base = values[task][0.0]
        for alpha in FIT_ALPHAS:
            y = values[task][alpha] - base
            obs.append((task, alpha, weights, design_row(alpha, weights), y))
    if len(obs) < P:
        raise ValueError(f"not enough observations for {model}/{module}: {len(obs)}")
    taus = [0.2 for _ in DIMENSIONS]
    beta = fit_beta(obs, taus)
    last = None
    for _ in range(24):
        taus = optimize_taus(obs, beta, taus)
        beta = fit_beta(obs, taus)
        current = nll(obs, beta, taus)
        if last is not None and abs(last - current) < 1e-5:
            break
        last = current
    residuals = []
    for task, alpha, weights, x, y in obs:
        pred = predict(beta, x)
        sigma = math.sqrt(projected_variance(alpha, weights, taus))
        residuals.append((task, alpha, y, pred, y - pred, sigma))
    return {
        "model": model,
        "module": module,
        "tasks": tasks,
        "metrics": metrics,
        "beta": beta,
        "taus": taus,
        "nll": nll(obs, beta, taus),
        "residuals": residuals,
        "values": values,
    }


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if abs(hi - lo) < 1e-12:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def write_svg(path: Path, title: str, lines, scatters=None, y_label="delta") -> None:
    width, height, pad = 900, 540, 70
    pts = []
    for _, _, line in lines:
        pts.extend(line)
    for _, points in scatters or []:
        pts.extend(points)
    xs = [p[0] for p in pts] or [-0.2, 0.2]
    ys = [p[1] for p in pts] or [-0.01, 0.01]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_y - min_y, 1e-6)
    min_y -= 0.1 * span
    max_y += 0.1 * span
    sx = lambda x: pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)
    sy = lambda y: height - pad - (y - min_y) / (max_y - min_y) * (height - 2 * pad)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    out.append('<rect width="100%" height="100%" fill="white"/>')
    out.append(f'<text x="{pad}" y="34" font-family="Arial" font-size="18">{html.escape(title)}</text>')
    out.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#222"/>')
    out.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#222"/>')
    for x in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        out.append(f'<line x1="{sx(x):.1f}" y1="{height-pad}" x2="{sx(x):.1f}" y2="{height-pad+6}" stroke="#222"/>')
        out.append(f'<text x="{sx(x)-12:.1f}" y="{height-pad+22}" font-family="Arial" font-size="11">{x:g}</text>')
    for y in nice_ticks(min_y, max_y, 6):
        out.append(f'<line x1="{pad-6}" y1="{sy(y):.1f}" x2="{pad}" y2="{sy(y):.1f}" stroke="#222"/>')
        out.append(f'<text x="8" y="{sy(y)+4:.1f}" font-family="Arial" font-size="11">{y:+.3f}</text>')
        out.append(f'<line x1="{pad}" y1="{sy(y):.1f}" x2="{width-pad}" y2="{sy(y):.1f}" stroke="#eee"/>')
    if min_y < 0 < max_y:
        out.append(f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{width-pad}" y2="{sy(0):.1f}" stroke="#999" stroke-dasharray="4,4"/>')
    out.append(f'<text x="{width/2-20:.1f}" y="{height-18}" font-family="Arial" font-size="12">alpha</text>')
    out.append(f'<text x="12" y="{pad-18}" font-family="Arial" font-size="12">{html.escape(y_label)}</text>')
    for idx, (label, color, line) in enumerate(lines):
        path_data = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(line))
        out.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{width-260}" y="{58+idx*18}" font-family="Arial" font-size="12" fill="{color}">{html.escape(label)}</text>')
    for color, points in scatters or []:
        for x, y in points:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{color}"/>')
    out.append("</svg>")
    path.write_text("\n".join(out))


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    score_rows = extract_scores()
    caps = read_caps()
    write_csv(OUT / "eval6_llama1b_3b_subtask_scores_long.csv", score_rows)

    fits = {}
    coverage_rows = []
    coef_rows = []
    curve_rows = []
    tau_rows = []
    residual_rows = []
    for model in MODELS:
        for module in MODULES:
            fit = fit_module(score_rows, caps, model, module)
            fits[(model, module)] = fit
            coverage_rows.append({
                "model": model,
                "module": module,
                "complete_tasks": len(fit["tasks"]),
                "expected_capability_tasks": sum(1 for row in caps.values() if row["group"] == module),
                "nll": fit["nll"],
            })
            for i, dim in enumerate(DIMENSIONS):
                coef_rows.append({
                    "model": model,
                    "module": module,
                    "dimension": dim,
                    "linear_coef_b": fit["beta"][2 * i],
                    "quadratic_coef_c": fit["beta"][2 * i + 1],
                    "tau_mle": fit["taus"][i],
                    "total_nll": fit["nll"],
                })
                tau_rows.append({
                    "model": model,
                    "module": module,
                    "dimension": dim,
                    "tau_mle": fit["taus"][i],
                    "total_nll": fit["nll"],
                })
                for alpha in FIT_ALPHAS:
                    curve_rows.append({
                        "model": model,
                        "module": module,
                        "dimension": dim,
                        "alpha": alpha,
                        "fit_delta": curve(fit["beta"], i, alpha),
                        "linear_coef_b": fit["beta"][2 * i],
                        "quadratic_coef_c": fit["beta"][2 * i + 1],
                        "tau_mle": fit["taus"][i],
                        "total_nll": fit["nll"],
                    })
            for task, alpha, observed, pred, residual, sigma in fit["residuals"]:
                nll_i = 0.5 * (math.log(2 * math.pi * sigma * sigma) + residual * residual / (sigma * sigma))
                residual_rows.append({
                    "model": model,
                    "module": module,
                    "task": task,
                    "alpha": alpha,
                    "observed_delta": observed,
                    "fit_delta": pred,
                    "residual": residual,
                    "sigma": sigma,
                    "nll": nll_i,
                })

    write_csv(OUT / "coverage.csv", coverage_rows)
    write_csv(OUT / "capability_fk_coefficients_by_module.csv", coef_rows)
    write_csv(OUT / "capability_curve_points_by_module.csv", curve_rows)
    write_csv(OUT / "dimension_tau_mle_by_module.csv", tau_rows)
    write_csv(OUT / "task_likelihood_errors_by_module.csv", residual_rows)

    images = []
    for module in MODULES:
        for i, dim in enumerate(DIMENSIONS):
            lines = []
            for model in MODELS:
                fit = fits[(model, module)]
                lines.append((MODEL_LABELS[model], MODEL_COLORS[model], [(x, curve(fit["beta"], i, x)) for x in XS]))
            path = OUT / f"{module}_dimension_{dim.lower().replace(' ', '_').replace('&', 'and')}.svg"
            write_svg(path, f"{module}: {dim}", lines, y_label="capability delta")
            images.append(path)
        # Representative task: the one with the largest mean absolute observed delta in 1B.
        fit0 = fits[("llama1b", module)]
        by_task = defaultdict(list)
        for task, alpha, observed, *_ in fit0["residuals"]:
            by_task[task].append((alpha, observed))
        task = max(by_task, key=lambda t: sum(abs(v) for _, v in by_task[t]) / len(by_task[t]))
        lines = []
        scatters = []
        for model in MODELS:
            fit = fits[(model, module)]
            weights = caps[task]["weights"]
            lines.append((MODEL_LABELS[model], MODEL_COLORS[model], [(x, task_prediction(fit["beta"], weights, x)) for x in XS]))
            base = fit["values"][task][0.0]
            scatters.append((MODEL_COLORS[model], sorted((a, fit["values"][task][a] - base) for a in ALPHAS)))
        path = OUT / f"{module}_task_{task}.svg"
        write_svg(path, f"{module}: {task}", lines, scatters=scatters, y_label="task delta")
        images.append(path)

    fit_summary = []
    for model in MODELS:
        for module in MODULES:
            fit = fits[(model, module)]
            at02 = sorted(
                ((DIMENSIONS[i], curve(fit["beta"], i, 0.2)) for i in range(len(DIMENSIONS))),
                key=lambda item: item[1],
            )
            fit_summary.append([
                MODEL_LABELS[model],
                module,
                len(fit["tasks"]),
                f'{fit["nll"]:.2f}',
                f'{min(fit["taus"]):.4f}-{max(fit["taus"]):.4f}',
                at02[0][0],
                f'{at02[0][1]:+.4f}',
                at02[-1][0],
                f'{at02[-1][1]:+.4f}',
            ])

    lines = [
        "# Eval6 Module-Specific Capability Curves\n",
        "Scope: Llama 3.2 1B/3B Instruct, alpha `-.2, -.1, 0, .1, .2`. Each benchmark module is fitted separately: MMLU-Pro, MMLU-Redux, AGIEval, and BBH.",
        "",
        "For every subtask, the fitted target is `score(alpha)-score(0)`. The module-specific model is `mu=sum_k w_task,k f_k(alpha)`, with `f_k(alpha)=b_k alpha + c_k alpha^2`. Each capability dimension has an independent `tau_k`, and the likelihood variance is `alpha^2 * sum_k (w_task,k tau_k)^2`.",
        "",
        "Metrics: MMLU-Pro uses `exact_match,custom-extract`; MMLU-Redux uses `exact_match,default`; AGIEval uses `acc_norm,none`; BBH uses `exact_match,flexible-extract`.",
        "\n## Fit Summary\n",
        md_table(["model", "module", "tasks", "NLL", "tau range", "most negative dim at .2", "delta", "most positive dim at .2", "delta"], fit_summary),
        "\n## Outputs\n",
        f"- Scores long table: `{OUT / 'eval6_llama1b_3b_subtask_scores_long.csv'}`",
        f"- Coefficients: `{OUT / 'capability_fk_coefficients_by_module.csv'}`",
        f"- Curve points: `{OUT / 'capability_curve_points_by_module.csv'}`",
        f"- Dimension tau: `{OUT / 'dimension_tau_mle_by_module.csv'}`",
        f"- Task likelihood errors: `{OUT / 'task_likelihood_errors_by_module.csv'}`",
        "\n## Figures\n",
    ]
    for image in images:
        lines.append(f"![{image.stem}]({image})")
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
