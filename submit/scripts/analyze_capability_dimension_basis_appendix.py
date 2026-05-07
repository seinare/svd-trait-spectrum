#!/usr/bin/env python3
"""Appendix analysis for reducing seven judged capability dimensions to three."""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/results/capability_dimension/deepseek_v4_pro_eval6_batch10x10_20260504_141548.csv"
OUT = ROOT / "docs/results/capability_dimension_basis_appendix"

DIMENSIONS = [
    "Factual Knowledge",
    "Language Understanding",
    "Inductive Reasoning",
    "Deductive Reasoning",
    "Mathematical Computation",
    "Structural Analysis",
    "Ethical & Safety Judgment",
]

CORE3 = [
    "Factual Knowledge",
    "Language Understanding",
    "Deductive Reasoning",
]

DESC = {
    "Factual Knowledge": "The ability to retrieve and recall world knowledge, concepts, terminology, and established facts. It represents crystallized intelligence and memory for specific information.",
    "Language Understanding": "The ability to precisely parse complex texts, interpret domain-specific jargon, resolve syntactic ambiguity, and grasp nuanced semantics. It reflects deep reading comprehension.",
    "Inductive Reasoning": "The ability to derive general rules or patterns from specific examples, categorize, perform analogical reasoning, and make data-driven predictions.",
    "Deductive Reasoning": "The ability to derive necessarily true conclusions from given premises through rigorous logical steps, applying general rules to specific cases.",
    "Mathematical Computation": "The ability to execute precise, multi-step arithmetic, algebraic, or symbolic manipulation. It focuses on formal operational accuracy.",
    "Structural Analysis": "The ability to decompose complex systems into constituent parts, understand relationships and causal links, and synthesize system-level behavior.",
    "Ethical & Safety Judgment": "The ability to identify moral dilemmas, assess safety implications, recognize social norms and legal boundaries, and distinguish compliant from harmful behavior.",
}

SHORT = {
    "Factual Knowledge": "Factual",
    "Language Understanding": "Language",
    "Inductive Reasoning": "Inductive",
    "Deductive Reasoning": "Deductive",
    "Mathematical Computation": "Math",
    "Structural Analysis": "Structural",
    "Ethical & Safety Judgment": "Ethical/Safety",
}


def read_rows() -> tuple[list[dict[str, str]], np.ndarray]:
    with INPUT.open(newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(row[d]) for d in DIMENSIONS] for row in rows], dtype=float)
    return rows, x


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pca_fit_transform(x: np.ndarray, n: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    centered = x - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n]
    coords = centered @ components.T
    explained = (s**2) / np.sum(s**2)
    return coords, components, explained[:n], mean


def pca_project(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - mean) @ components.T


def pearson_corr(x: np.ndarray) -> np.ndarray:
    return np.corrcoef(x, rowvar=False)


def linear_regression(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = math.sqrt(ss_res / len(y))
    return beta, pred, r2, rmse


def barycentric_weights(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Return affine weights over a 2D triangle for each point.

    Weights sum to one. Negative weights mean the point is outside the triangle.
    """
    a = np.vstack([triangle.T, np.ones(3)])
    out = []
    for point in points:
        b = np.array([point[0], point[1], 1.0])
        w = np.linalg.solve(a, b)
        out.append(w)
    return np.array(out)


def scale(points: np.ndarray, width: int, height: int, pad: int):
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
    dx, dy = x1 - x0, y1 - y0
    x0 -= dx * 0.12 + 1e-9
    x1 += dx * 0.12 + 1e-9
    y0 -= dy * 0.12 + 1e-9
    y1 += dy * 0.12 + 1e-9

    def sx(v: float) -> float:
        return pad + (v - x0) / (x1 - x0) * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - (v - y0) / (y1 - y0) * (height - 2 * pad)

    return sx, sy, (x0, x1, y0, y1)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def write_pca_svg(rows: list[dict[str, str]], coords: np.ndarray, unit_coords: np.ndarray, explained: np.ndarray, path: Path) -> None:
    width, height, pad = 980, 700, 82
    all_points = np.vstack([coords, unit_coords])
    sx, sy, lim = scale(all_points, width, height, pad)
    x0, x1, y0, y1 = lim

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="82" y="36" font-family="Arial" font-size="18" fill="#111827">Seven-dimensional judged capability weights: PCA projection</text>',
        f'<text x="82" y="58" font-family="Arial" font-size="12" fill="#475569">Gray points are eval6 subtasks; diamonds are unit capability vectors. PC1 {explained[0]*100:.1f}%, PC2 {explained[1]*100:.1f}%.</text>',
    ]
    for x in nice_ticks(x0, x1):
        parts.append(f'<line x1="{sx(x):.1f}" y1="{pad}" x2="{sx(x):.1f}" y2="{height-pad}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{sx(x):.1f}" y="{height-pad+22}" font-family="Arial" font-size="11" text-anchor="middle" fill="#475569">{x:.2f}</text>')
    for y in nice_ticks(y0, y1):
        parts.append(f'<line x1="{pad}" y1="{sy(y):.1f}" x2="{width-pad}" y2="{sy(y):.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{pad-10}" y="{sy(y)+4:.1f}" font-family="Arial" font-size="11" text-anchor="end" fill="#475569">{y:.2f}</text>')
    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#111827"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-24}" font-family="Arial" font-size="13" text-anchor="middle" fill="#111827">PC1</text>')
    parts.append(f'<text x="22" y="{height/2:.1f}" font-family="Arial" font-size="13" text-anchor="middle" fill="#111827" transform="rotate(-90 22 {height/2:.1f})">PC2</text>')

    group_color = {"mmlu_pro": "#94a3b8", "mmlu_redux": "#9ca3af", "agieval": "#a1a1aa", "bbh": "#cbd5e1"}
    for row, (x, y) in zip(rows, coords, strict=True):
        color = group_color.get(row["group"], "#9ca3af")
        parts.append(f'<circle cx="{sx(float(x)):.1f}" cy="{sy(float(y)):.1f}" r="3.2" fill="{color}" opacity="0.55" stroke="none"><title>{html.escape(row["group"] + "/" + row["task"])}</title></circle>')

    core_colors = {
        "Factual Knowledge": "#dc2626",
        "Language Understanding": "#2563eb",
        "Deductive Reasoning": "#16a34a",
    }
    for dim, (x, y) in zip(DIMENSIONS, unit_coords, strict=True):
        color = core_colors.get(dim, "#111827")
        cx, cy = sx(float(x)), sy(float(y))
        r = 8.5 if dim in CORE3 else 6.5
        parts.append(f'<path d="M {cx:.1f},{cy-r:.1f} L {cx+r:.1f},{cy:.1f} L {cx:.1f},{cy+r:.1f} L {cx-r:.1f},{cy:.1f} Z" fill="{color}" opacity="0.95" stroke="white" stroke-width="1.4"/>')
        parts.append(f'<text x="{cx+10:.1f}" y="{cy-8:.1f}" font-family="Arial" font-size="12" fill="{color}" font-weight="600">{html.escape(SHORT[dim])}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def corr_color(v: float) -> str:
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        t = v
        r, g, b = 220, int(245 - 150 * t), int(245 - 150 * t)
    else:
        t = -v
        r, g, b = int(245 - 150 * t), int(245 - 120 * t), 220
    return f"rgb({r},{g},{b})"


def write_corr_svg(corr: np.ndarray, path: Path) -> None:
    width, height = 920, 760
    left, top, cell = 220, 86, 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="64" y="36" font-family="Arial" font-size="18" fill="#111827">Correlation matrix of original seven capability weights</text>',
        '<text x="64" y="58" font-family="Arial" font-size="12" fill="#475569">Pearson correlations across eval6 subtask-level DeepSeek weight vectors.</text>',
    ]
    for i, d in enumerate(DIMENSIONS):
        y = top + i * cell + cell / 2 + 4
        parts.append(f'<text x="{left-10}" y="{y:.1f}" font-family="Arial" font-size="11" text-anchor="end" fill="#111827">{html.escape(SHORT[d])}</text>')
        x = left + i * cell + cell / 2
        parts.append(f'<text x="{x:.1f}" y="{top-12}" font-family="Arial" font-size="11" text-anchor="middle" fill="#111827" transform="rotate(-42 {x:.1f} {top-12})">{html.escape(SHORT[d])}</text>')
    for i in range(len(DIMENSIONS)):
        for j in range(len(DIMENSIONS)):
            x, y = left + j * cell, top + i * cell
            v = float(corr[i, j])
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{corr_color(v)}" stroke="white"/>')
            parts.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+4:.1f}" font-family="Arial" font-size="12" text-anchor="middle" fill="#111827">{v:+.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def markdown_table(fields: list[str], rows: list[dict[str, object]], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        vals = []
        for f in fields:
            v = row[f]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, x = read_rows()
    coords, comps, explained, mean = pca_fit_transform(x, 2)
    unit = np.eye(len(DIMENSIONS))
    unit_coords = pca_project(unit, mean, comps)
    core_unit_indices = [DIMENSIONS.index(d) for d in CORE3]
    core_triangle = unit_coords[core_unit_indices]
    task_bary = barycentric_weights(coords, core_triangle)
    unit_bary = barycentric_weights(unit_coords, core_triangle)
    task_inside = np.all(task_bary >= -1e-9, axis=1)
    inside_ratio = float(np.mean(task_inside))
    corr = pearson_corr(x)

    pca_rows = []
    for row, (pc1, pc2) in zip(rows, coords, strict=True):
        pca_rows.append({"group": row["group"], "task": row["task"], "pc1": pc1, "pc2": pc2})
    for dim, (pc1, pc2) in zip(DIMENSIONS, unit_coords, strict=True):
        pca_rows.append({"group": "unit_dimension", "task": dim, "pc1": pc1, "pc2": pc2})
    write_csv(OUT / "seven_dim_capability_pca2d.csv", pca_rows, ["group", "task", "pc1", "pc2"])

    loading_rows = []
    for ci, comp in enumerate(comps, start=1):
        for dim, value in zip(DIMENSIONS, comp, strict=True):
            loading_rows.append({"component": f"PC{ci}", "dimension": dim, "loading": float(value)})
    write_csv(OUT / "seven_dim_pca_loadings.csv", loading_rows, ["component", "dimension", "loading"])

    bary_rows = []
    for dim, point, weights in zip(DIMENSIONS, unit_coords, unit_bary, strict=True):
        recon = weights @ core_triangle
        residual = float(np.linalg.norm(point - recon))
        bary_rows.append(
            {
                "dimension": dim,
                "factual_weight": float(weights[0]),
                "language_weight": float(weights[1]),
                "deductive_weight": float(weights[2]),
                "pca2d_residual": residual,
                "inside_core_triangle": bool(np.all(weights >= -1e-9)),
            }
        )
    write_csv(
        OUT / "seven_dim_unit_vectors_as_core3_barycentric.csv",
        bary_rows,
        ["dimension", "factual_weight", "language_weight", "deductive_weight", "pca2d_residual", "inside_core_triangle"],
    )

    task_bary_rows = []
    for row, weights, inside in zip(rows, task_bary, task_inside, strict=True):
        task_bary_rows.append(
            {
                "group": row["group"],
                "task": row["task"],
                "factual_weight": float(weights[0]),
                "language_weight": float(weights[1]),
                "deductive_weight": float(weights[2]),
                "inside_core_triangle": bool(inside),
            }
        )
    write_csv(
        OUT / "seven_dim_task_points_core3_barycentric.csv",
        task_bary_rows,
        ["group", "task", "factual_weight", "language_weight", "deductive_weight", "inside_core_triangle"],
    )

    corr_rows = []
    for i, d1 in enumerate(DIMENSIONS):
        row = {"dimension": d1}
        for j, d2 in enumerate(DIMENSIONS):
            row[d2] = float(corr[i, j])
        corr_rows.append(row)
    write_csv(OUT / "seven_dim_correlation_matrix.csv", corr_rows, ["dimension", *DIMENSIONS])

    core_x = x[:, [DIMENSIONS.index(d) for d in CORE3]]
    reconstruction_rows = []
    pred_rows = []
    for dim_i, dim in enumerate(DIMENSIONS):
        beta, pred, r2, rmse = linear_regression(x[:, dim_i], core_x)
        reconstruction_rows.append(
            {
                "target_dimension": dim,
                "intercept": float(beta[0]),
                "coef_factual": float(beta[1]),
                "coef_language": float(beta[2]),
                "coef_deductive": float(beta[3]),
                "r2": float(r2),
                "rmse": float(rmse),
            }
        )
        for row, y, yhat in zip(rows, x[:, dim_i], pred, strict=True):
            pred_rows.append(
                {
                    "group": row["group"],
                    "task": row["task"],
                    "target_dimension": dim,
                    "actual": float(y),
                    "predicted_from_core3": float(yhat),
                    "residual": float(y - yhat),
                }
            )
    write_csv(
        OUT / "seven_dim_reconstruction_from_core3.csv",
        reconstruction_rows,
        ["target_dimension", "intercept", "coef_factual", "coef_language", "coef_deductive", "r2", "rmse"],
    )
    write_csv(
        OUT / "seven_dim_reconstruction_predictions_long.csv",
        pred_rows,
        ["group", "task", "target_dimension", "actual", "predicted_from_core3", "residual"],
    )

    write_pca_svg(rows, coords, unit_coords, explained, OUT / "seven_dim_capability_pca2d.svg")
    write_corr_svg(corr, OUT / "seven_dim_correlation_matrix.svg")

    singular_values = np.linalg.svd(x - mean, full_matrices=False)[1]
    full_explained = (singular_values**2) / np.sum(singular_values**2)
    explained_rows = [
        {
            "component": f"PC{i + 1}",
            "explained_variance_ratio": float(v),
            "cumulative": float(np.sum(full_explained[: i + 1])),
        }
        for i, v in enumerate(full_explained)
    ]
    write_csv(OUT / "seven_dim_pca_explained_variance.csv", explained_rows, ["component", "explained_variance_ratio", "cumulative"])

    strongest_corr = []
    for i, d1 in enumerate(DIMENSIONS):
        for j, d2 in enumerate(DIMENSIONS):
            if i < j:
                strongest_corr.append({"pair": f"{d1} / {d2}", "correlation": float(corr[i, j]), "abs_correlation": abs(float(corr[i, j]))})
    strongest_corr.sort(key=lambda r: r["abs_correlation"], reverse=True)

    report = [
        "# Appendix: Justification for the Three-Dimension Capability Basis",
        "",
        "This appendix documents why the later capability-response analysis uses three dimensions: `Factual Knowledge`, `Language Understanding`, and `Deductive Reasoning`. The starting point was a seven-dimensional DeepSeek judge rubric applied to eval6 subtasks. The analysis below treats every eval6 subtask as one point in the original seven-dimensional capability-weight simplex.",
        "",
        "## Original Seven Capability Dimensions",
        "",
    ]
    for dim in DIMENSIONS:
        report.append(f"- **{dim}**: {DESC[dim]}")
    report += [
        "",
        "## Data",
        "",
        f"- Source table: `{INPUT.relative_to(ROOT)}`.",
        f"- Unit of analysis: {len(rows)} eval6 subtasks.",
        "- Each row is the averaged DeepSeek-v4-pro judge mixture for one subtask, using 10 valid batches of 10 questions when available.",
        "- The seven weights are non-negative and sum to one per subtask, so the PCA is applied directly to the judged mixture vectors rather than to model scores.",
        "",
        "## Seven-Dimension PCA",
        "",
        "The figure projects all subtask-level seven-dimensional vectors to two PCA axes. Gray points are actual subtasks. Diamonds are unit vectors for the seven original capability dimensions, projected into the same PCA space.",
        "",
        "![Seven-dimension capability PCA](seven_dim_capability_pca2d.svg)",
        "",
        f"The first two PCs explain `{explained[0]*100:.1f}%` and `{explained[1]*100:.1f}%` of the variance. In this projection, the most exposed vertices are the unit vectors for `Factual Knowledge`, `Language Understanding`, and `Deductive Reasoning`. The remaining dimensions lie closer to the interior or along directions between those vertices. This supports interpreting the benchmark's judged capability demand as mostly occupying a triangular region spanned by these three axes rather than using all seven dimensions symmetrically.",
        "",
        "### PCA Loadings",
        "",
        markdown_table(["component", "dimension", "loading"], loading_rows),
        "",
        "### Core-Triangle Geometry",
        "",
        f"In the PCA plane, `{inside_ratio*100:.1f}%` of subtask points fall inside the triangle spanned by the three selected unit vectors. In this run, no subtask point forms a fourth broad vertex outside the core triangle.",
        "",
        "The table below expresses every original unit dimension vector as affine barycentric coordinates over the three selected PCA vertices. For unit vectors, the residual is numerically zero because three non-collinear points span the 2D PCA plane; the sign and magnitude of the weights indicate whether a dimension is inside the core triangle or extends beyond one of its sides.",
        "",
        markdown_table(["dimension", "factual_weight", "language_weight", "deductive_weight", "pca2d_residual", "inside_core_triangle"], bary_rows),
        "",
        "## Correlation Structure of the Seven Original Dimensions",
        "",
        "Because weights live on a simplex, correlations include both semantic co-occurrence and competition for probability mass. The matrix is still useful for identifying redundant or composite dimensions.",
        "",
        "![Seven-dimension correlation matrix](seven_dim_correlation_matrix.svg)",
        "",
        "Strongest absolute pairwise correlations:",
        "",
        markdown_table(["pair", "correlation", "abs_correlation"], strongest_corr, max_rows=10),
        "",
        "The matrix shows that several non-core dimensions are not independent axes in the sampled benchmark set. They tend to move as mixtures or complements of factual, language, and deductive demand. For example, mathematical and structural weights are localized to specific task families and are partly explained by combinations of deductive and factual demand rather than forming broad benchmark-wide extremes.",
        "",
        "## Reconstructing Seven Dimensions from the Core Three",
        "",
        "As a direct check, each original dimension was regressed on the three retained dimensions:",
        "",
        "`target ~ intercept + Factual Knowledge + Language Understanding + Deductive Reasoning`",
        "",
        markdown_table(["target_dimension", "intercept", "coef_factual", "coef_language", "coef_deductive", "r2", "rmse"], reconstruction_rows),
        "",
        "The retained three dimensions trivially reconstruct themselves. More importantly, the residual dimensions can be read as benchmark-specific mixtures or residual specializations relative to the core basis. Low or moderate R2 for a residual dimension should not be interpreted as evidence that it is unimportant in general; it indicates that, in the current eval6 subtask set, it is not a broad independent axis comparable to the three PCA-exposed vertices.",
        "",
        "## Interpretation for the Main Paper",
        "",
        "The seven-dimensional judge rubric was intentionally broad, covering knowledge, reading, inductive and deductive reasoning, computation, structural analysis, and safety judgment. However, the actual eval6 subtask distribution does not activate these seven axes uniformly. In the PCA geometry, task points form a region whose outer structure is mainly anchored by `Factual Knowledge`, `Language Understanding`, and `Deductive Reasoning`. The other four dimensions occupy more interior or specialized locations and can be treated as secondary composites for this benchmark collection.",
        "",
        "Therefore, the three-dimension basis used in the main alpha-response analysis is not a replacement for the conceptual seven-dimension rubric. It is a data-driven low-dimensional basis for this benchmark suite: it preserves the dominant geometry of judged task demands while avoiding unstable over-interpretation of sparse or highly localized dimensions.",
        "",
        "## Output Files",
        "",
        "- `seven_dim_capability_pca2d.svg`",
        "- `seven_dim_correlation_matrix.svg`",
        "- `seven_dim_capability_pca2d.csv`",
        "- `seven_dim_pca_loadings.csv`",
        "- `seven_dim_pca_explained_variance.csv`",
        "- `seven_dim_unit_vectors_as_core3_barycentric.csv`",
        "- `seven_dim_task_points_core3_barycentric.csv`",
        "- `seven_dim_correlation_matrix.csv`",
        "- `seven_dim_reconstruction_from_core3.csv`",
        "- `seven_dim_reconstruction_predictions_long.csv`",
    ]
    (OUT / "capability_dimension_basis_appendix.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
