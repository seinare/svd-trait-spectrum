#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "results" / "trait_capability_correlation"

ALPHAS = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
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
DIMENSIONS = ["Factual Knowledge", "Language Understanding", "Deductive Reasoning"]

MODEL_MAP = {
    "llama32_1b_instruct": "llama1b",
    "llama32_3b_instruct": "llama3b",
    "llama31_8b_instruct": "llama8b",
    "qwen3_8b": "qwen3_8b",
}


def alpha_label(alpha: float) -> str:
    if abs(alpha) < 1e-12:
        return "0"
    return f"{alpha:.2f}".rstrip("0").rstrip(".")


def read_trait_tables() -> dict[tuple[str, float, str], float]:
    values: dict[tuple[str, float, str], float] = {}
    for trait_model in MODEL_MAP:
        path = ROOT / "docs" / "results" / "trait_alpha9_tables" / f"{trait_model}_trait_alpha9_table.csv"
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                trait = row["trait"]
                for alpha in ALPHAS:
                    text = row.get(alpha_label(alpha), "")
                    if text:
                        values[(trait_model, round(alpha, 2), trait)] = float(text)
    return values


def read_capability_points() -> dict[tuple[str, float, str], float]:
    path = ROOT / "docs" / "results" / "eval6_all_models_alpha9_capability" / "capability_alpha_mle_errorbar_points.csv"
    values: dict[tuple[str, float, str], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            alpha = round(float(row["alpha"]), 2)
            dim = row["dimension"]
            if model in set(MODEL_MAP.values()) and dim in DIMENSIONS:
                values[(model, alpha, dim)] = float(row["mean_mle"])
    return values


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def matvec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(v * xv for v, xv in zip(row, x)) for row in a]


def invert(a: list[list[float]], ridge: float = 0.0) -> list[list[float]]:
    n = len(a)
    aug = []
    for i, row in enumerate(a):
        left = [float(v) for v in row]
        left[i] += ridge
        aug.append(left + [1.0 if i == j else 0.0 for j in range(n)])
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor:
                aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def ols(x: list[list[float]], y: list[float]) -> dict[str, object]:
    xt = transpose(x)
    xtx = matmul(xt, x)
    xty = matvec(xt, y)
    try:
        inv = invert(xtx)
    except ValueError:
        inv = invert(xtx, ridge=1e-8)
    beta = matvec(inv, xty)
    pred = matvec(x, beta)
    resid = [yi - pi for yi, pi in zip(y, pred)]
    n = len(y)
    p = len(beta)
    sse = sum(r * r for r in resid)
    mean_y = sum(y) / n
    sst = sum((yi - mean_y) ** 2 for yi in y)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    df = max(n - p, 1)
    sigma2 = sse / df
    se = [math.sqrt(max(sigma2 * inv[i][i], 0.0)) for i in range(p)]
    return {"beta": beta, "se": se, "pred": pred, "resid": resid, "r2": r2, "sse": sse, "df": df}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(x: list[float], y: list[float]) -> float:
    sx, sy = sd(x), sd(y)
    if sx == 0 or sy == 0:
        return 0.0
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / ((len(x) - 1) * sx * sy)


def zscore(xs: list[float]) -> tuple[list[float], float, float]:
    m, s = mean(xs), sd(xs)
    if s == 0:
        return [0.0 for _ in xs], m, s
    return [(x - m) / s for x in xs], m, s


def build_long_rows() -> list[dict[str, object]]:
    traits = read_trait_tables()
    caps = read_capability_points()
    rows: list[dict[str, object]] = []
    for trait_model, cap_model in MODEL_MAP.items():
        for alpha in ALPHAS:
            base_trait_alpha = 0.0
            for trait in TRAITS:
                score = traits[(trait_model, round(alpha, 2), trait)]
                base = traits[(trait_model, base_trait_alpha, trait)]
                row = {
                    "trait_model": trait_model,
                    "capability_model": cap_model,
                    "alpha": alpha,
                    "trait": trait,
                    "trait_score": score,
                    "delta_trait": score - base,
                }
                for dim in DIMENSIONS:
                    row[dim] = caps[(cap_model, round(alpha, 2), dim)]
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_long_rows()
    nonzero = [r for r in rows if abs(float(r["alpha"])) > 1e-12]

    write_csv(
        OUT_DIR / "trait_capability_points_long.csv",
        rows,
        ["trait_model", "capability_model", "alpha", "trait", "trait_score", "delta_trait"] + DIMENSIONS,
    )

    pooled_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    models = list(MODEL_MAP)

    for trait in TRAITS:
        data = [r for r in nonzero if r["trait"] == trait]
        y_raw = [float(r["delta_trait"]) for r in data]
        y, _ym, _ys = zscore(y_raw)
        dim_z = {}
        for dim in DIMENSIONS:
            dim_z[dim], _m, _s = zscore([float(r[dim]) for r in data])

        x = []
        for i, r in enumerate(data):
            # Standardized capability columns plus model fixed effects.
            row = [1.0] + [dim_z[dim][i] for dim in DIMENSIONS]
            row += [1.0 if r["trait_model"] == m else 0.0 for m in models[1:]]
            x.append(row)
        fit = ols(x, y)
        names = ["intercept"] + DIMENSIONS + [f"model_fe:{m}" for m in models[1:]]
        for name, beta, se in zip(names, fit["beta"], fit["se"]):
            pooled_rows.append(
                {
                    "trait": trait,
                    "term": name,
                    "standardized_beta": beta,
                    "se": se,
                    "t_approx": beta / se if se else "",
                    "r2": fit["r2"],
                    "n": len(data),
                    "df": fit["df"],
                }
            )
        for r, pred, resid in zip(data, fit["pred"], fit["resid"]):
            pred_rows.append(
                {
                    "trait_model": r["trait_model"],
                    "alpha": r["alpha"],
                    "trait": trait,
                    "delta_trait": r["delta_trait"],
                    "pred_z": pred,
                    "resid_z": resid,
                }
            )
        for dim in DIMENSIONS:
            corr_rows.append(
                {
                    "trait": trait,
                    "dimension": dim,
                    "pearson_r": pearson([float(r[dim]) for r in data], y_raw),
                    "n": len(data),
                }
            )

    write_csv(
        OUT_DIR / "pooled_trait_on_capability_regression.csv",
        pooled_rows,
        ["trait", "term", "standardized_beta", "se", "t_approx", "r2", "n", "df"],
    )
    write_csv(OUT_DIR / "trait_capability_pearson.csv", corr_rows, ["trait", "dimension", "pearson_r", "n"])
    write_csv(OUT_DIR / "pooled_predictions_long.csv", pred_rows, ["trait_model", "alpha", "trait", "delta_trait", "pred_z", "resid_z"])

    per_model_rows: list[dict[str, object]] = []
    for trait in TRAITS:
        for model in models:
            data = [r for r in nonzero if r["trait"] == trait and r["trait_model"] == model]
            y = [float(r["delta_trait"]) for r in data]
            x = [[1.0] + [float(r[dim]) for dim in DIMENSIONS] for r in data]
            fit = ols(x, y)
            for name, beta, se in zip(["intercept"] + DIMENSIONS, fit["beta"], fit["se"]):
                per_model_rows.append(
                    {
                        "trait_model": model,
                        "trait": trait,
                        "term": name,
                        "beta_raw": beta,
                        "se": se,
                        "r2": fit["r2"],
                        "n": len(data),
                    }
                )
    write_csv(OUT_DIR / "per_model_trait_on_capability_regression.csv", per_model_rows, ["trait_model", "trait", "term", "beta_raw", "se", "r2", "n"])

    # Leave-one-model-out prediction on standardized pooled design without model fixed effects.
    lomo_rows: list[dict[str, object]] = []
    for trait in TRAITS:
        all_data = [r for r in nonzero if r["trait"] == trait]
        for heldout in models:
            train = [r for r in all_data if r["trait_model"] != heldout]
            test = [r for r in all_data if r["trait_model"] == heldout]
            y_train_raw = [float(r["delta_trait"]) for r in train]
            y_train, ym, ys = zscore(y_train_raw)
            dim_stats = {}
            for dim in DIMENSIONS:
                xs = [float(r[dim]) for r in train]
                _z, m, s = zscore(xs)
                dim_stats[dim] = (m, s)
            x_train = [[1.0] + [((float(r[dim]) - dim_stats[dim][0]) / dim_stats[dim][1] if dim_stats[dim][1] else 0.0) for dim in DIMENSIONS] for r in train]
            fit = ols(x_train, y_train)
            sq = []
            for r in test:
                xx = [1.0] + [((float(r[dim]) - dim_stats[dim][0]) / dim_stats[dim][1] if dim_stats[dim][1] else 0.0) for dim in DIMENSIONS]
                pred_z = sum(b * v for b, v in zip(fit["beta"], xx))
                pred = pred_z * ys + ym
                sq.append((float(r["delta_trait"]) - pred) ** 2)
            rmse = math.sqrt(sum(sq) / len(sq))
            lomo_rows.append({"trait": trait, "heldout_model": heldout, "rmse_delta_trait": rmse, "n_test": len(test)})
    write_csv(OUT_DIR / "leave_one_model_out_rmse.csv", lomo_rows, ["trait", "heldout_model", "rmse_delta_trait", "n_test"])

    capability_terms = [r for r in pooled_rows if r["term"] in DIMENSIONS]
    top = sorted(capability_terms, key=lambda r: abs(float(r["standardized_beta"])), reverse=True)[:12]
    report = [
        "# TRAIT-Capability First-Layer Correlation Analysis",
        "",
        "Scope: four dense instruct models with complete nine-point TRAIT data: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, and Qwen3-8B.",
        "",
        "Response variable is `Delta TRAIT = TRAIT(alpha) - TRAIT(0)`. Predictors are the three fitted capability latent responses at the same model and alpha. The main pooled model uses standardized TRAIT deltas and standardized capability predictors, plus model fixed effects. Alpha=0 rows are excluded from regression because both sides are mechanically zero.",
        "",
        "## Strongest Standardized Pooled Terms",
        "",
        "| trait | capability | beta | approx t | R2 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in top:
        report.append(
            f"| {r['trait']} | {r['term']} | {float(r['standardized_beta']):.3f} | {float(r['t_approx']):.2f} | {float(r['r2']):.3f} |"
        )
    report.extend(
        [
            "",
            "## Files",
            "",
            "- `pooled_trait_on_capability_regression.csv`: standardized pooled coefficients with model fixed effects.",
            "- `trait_capability_pearson.csv`: simple pointwise correlations.",
            "- `per_model_trait_on_capability_regression.csv`: raw within-model OLS coefficients.",
            "- `leave_one_model_out_rmse.csv`: leave-one-model-out predictive RMSE on TRAIT deltas.",
        ]
    )
    (OUT_DIR / "trait_capability_correlation_report.md").write_text("\n".join(report))
    print(OUT_DIR)


if __name__ == "__main__":
    main()
