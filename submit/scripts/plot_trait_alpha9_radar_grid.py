#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path('/Users/wangzhanhui/Documents/New project/svd-trait-spectrum')
IN_DIR = ROOT / 'docs/results/trait_alpha9_tables'
OUT_DIR = ROOT / 'docs/results/trait_alpha9_radar_grid'
ALPHAS = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
TRAITS = [
    'Openness',
    'Conscientiousness',
    'Extraversion',
    'Agreeableness',
    'Neuroticism',
    'Machiavellianism',
    'Narcissism',
    'Psychopathy',
]
MODELS = [
    ('llama32_1b_instruct', 'Llama-3.2-1B Inst.'),
    ('llama32_3b_instruct', 'Llama-3.2-3B Inst.'),
    ('llama31_8b_instruct', 'Llama-3.1-8B Inst.'),
    ('qwen3_8b', 'Qwen3-8B'),
    ('qwen3_30b_a3b', 'Qwen3-30B-A3B'),
]
COLORS = {
    -0.2: '#08306b',
    -0.15: '#2171b5',
    -0.1: '#6baed6',
    -0.05: '#c6dbef',
    0.0: '#222222',
    0.05: '#fdd0a2',
    0.1: '#fc8d59',
    0.15: '#e34a33',
    0.2: '#99000d',
}
R_MIN = 0.14
R_MAX = 0.72
R_TICKS = [0.2, 0.4, 0.6]


def alpha_col(a: float) -> str:
    return '0' if abs(a) < 1e-12 else f'{a:g}'


def load_table(model: str) -> dict[float, dict[str, float]]:
    path = IN_DIR / f'{model}_trait_alpha9_table.csv'
    out = {a: {} for a in ALPHAS}
    if not path.exists():
        return {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            trait = row['trait']
            if trait not in TRAITS:
                continue
            for a in ALPHAS:
                val = (row.get(alpha_col(a)) or '').strip()
                if val:
                    out[a][trait] = float(val)
    return {a: vals for a, vals in out.items() if all(t in vals for t in TRAITS)}


def write_long(all_scores: dict[str, dict[float, dict[str, float]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / 'trait_alpha9_radar_scores_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'alpha', 'trait', 'trait_score'])
        w.writeheader()
        for model, scores in all_scores.items():
            for a in ALPHAS:
                for t in TRAITS:
                    v = scores.get(a, {}).get(t)
                    w.writerow({'model': model, 'alpha': f'{a:g}', 'trait': t, 'trait_score': '' if v is None else f'{v:.6f}'})
    with (OUT_DIR / 'trait_alpha9_radar_coverage.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'available_alpha_count', 'available_alphas', 'missing_alphas'])
        w.writeheader()
        for model, _ in MODELS:
            scores = all_scores[model]
            avail = [a for a in ALPHAS if a in scores]
            miss = [a for a in ALPHAS if a not in scores]
            w.writerow({
                'model': model,
                'available_alpha_count': len(avail),
                'available_alphas': ' '.join(f'{a:g}' for a in avail),
                'missing_alphas': ' '.join(f'{a:g}' for a in miss),
            })


def unit_vertices(scale: float = 1.0) -> list[tuple[float, float]]:
    verts = []
    for i in range(len(TRAITS)):
        theta = math.pi / 2 - 2 * math.pi * i / len(TRAITS)
        verts.append((scale * math.cos(theta), scale * math.sin(theta)))
    return verts


def scaled_radius(value: float) -> float:
    return max(0.0, min(1.0, (value - R_MIN) / (R_MAX - R_MIN)))


def value_points(values: list[float]) -> list[tuple[float, float]]:
    return [(scaled_radius(v) * x, scaled_radius(v) * y) for v, (x, y) in zip(values, unit_vertices(1.0))]


def draw_polygon_grid(ax) -> None:
    ax.set_aspect('equal')
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.30, 1.18)
    ax.axis('off')
    base = unit_vertices(1.0)
    for x, y in base:
        ax.plot([0, x], [0, y], color='#d7d7d7', lw=0.75, zorder=0)
    for tick in R_TICKS:
        s = scaled_radius(tick)
        ring = unit_vertices(s)
        ring.append(ring[0])
        ax.plot([p[0] for p in ring], [p[1] for p in ring], color='#d4d4d4', lw=0.78, zorder=0)
        ax.text(s + 0.035, 0.012, f'{tick:.1f}', fontsize=7.3, color='#666', ha='left', va='center')
    outer = unit_vertices(1.0)
    outer.append(outer[0])
    ax.plot([p[0] for p in outer], [p[1] for p in outer], color='#999999', lw=0.95, zorder=0)
    for trait, (x, y) in zip(TRAITS, base):
        label_x = 1.045 * x
        label_y = 1.045 * y
        ha = 'center'
        if label_x > 0.15:
            ha = 'left'
        elif label_x < -0.15:
            ha = 'right'
        va = 'center'
        if label_y > 0.75:
            va = 'bottom'
        elif label_y < -0.75:
            va = 'top'
        ax.text(label_x, label_y, trait, fontsize=7.6, ha=ha, va=va, color='#111')


def plot_panel(ax, model: str, label: str, scores: dict[float, dict[str, float]]) -> None:
    draw_polygon_grid(ax)
    ax.set_title(label, fontsize=11.3, fontweight='bold', pad=12)
    if not scores:
        ax.text(0, 0.06, 'missing\nTRAIT alpha9', ha='center', va='center', fontsize=12, color='#777')
        return
    for a in ALPHAS:
        if a not in scores:
            continue
        vals = [scores[a][t] for t in TRAITS]
        pts = value_points(vals)
        pts.append(pts[0])
        lw = 1.18
        line_alpha = 0.88 if a == 0 else 0.80
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[a], lw=lw, alpha=line_alpha, zorder=3 if a == 0 else 2)
        if a == 0:
            ax.fill([p[0] for p in pts], [p[1] for p in pts], color=COLORS[a], alpha=0.055, zorder=1)
    missing = [a for a in ALPHAS if a not in scores]
    if missing:
        ax.text(0, -1.18, 'missing: ' + ', '.join(f'{a:g}' for a in missing), ha='center', fontsize=7.5, color='#9a3412')


def render_grid(all_scores: dict[str, dict[float, dict[str, float]]], models, name: str, title: str):
    n = len(models)
    fig = plt.figure(figsize=(16.2, 9.35 if n == 5 else 8.55))
    positions = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5)] if n == 5 else [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]
    axes = []
    for pos, (model, label) in zip(positions, models):
        ax = fig.add_subplot(*pos)
        plot_panel(ax, model, label, all_scores[model])
        axes.append(ax)
    if n == 5:
        blank = fig.add_subplot(2, 3, 6)
        blank.axis('off')
    handles = []
    labels = []
    for a in ALPHAS:
        h, = axes[0].plot([], [], color=COLORS[a], lw=1.35, alpha=0.88 if a == 0 else 0.85, label=f'{a:g}')
        handles.append(h)
        labels.append(f'{a:g}')
    fig.legend(handles, labels, title='alpha', frameon=False, ncol=3, loc='lower right', bbox_to_anchor=(0.965, 0.065))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.985)
    fig.text(0.03, 0.024, f'Radial scale is clipped to [{R_MIN:.2f}, {R_MAX:.2f}] to emphasize alpha-induced differences. Score = P(high1)+P(high2).', fontsize=9.6, color='#444')
    fig.tight_layout(rect=[0.02, 0.065, 0.98, 0.95])
    png = OUT_DIR / f'{name}.png'
    svg = OUT_DIR / f'{name}.svg'
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def render_individual(all_scores: dict[str, dict[float, dict[str, float]]]) -> None:
    for model, label in MODELS:
        fig = plt.figure(figsize=(5.7, 5.45))
        ax = fig.add_subplot(1, 1, 1)
        plot_panel(ax, model, label, all_scores[model])
        handles = []
        labs = []
        for a in ALPHAS:
            h, = ax.plot([], [], color=COLORS[a], lw=1.35, label=f'{a:g}')
            handles.append(h)
            labs.append(f'{a:g}')
        fig.legend(handles, labs, title='alpha', frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.0), fontsize=8)
        fig.tight_layout(rect=[0, 0.09, 1, 1])
        fig.savefig(OUT_DIR / f'{model}_trait_alpha9_radar.png', dpi=300, bbox_inches='tight')
        fig.savefig(OUT_DIR / f'{model}_trait_alpha9_radar.svg', bbox_inches='tight')
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_scores = {m: load_table(m) for m, _ in MODELS}
    write_long(all_scores)
    p1, s1 = render_grid(all_scores, MODELS, 'trait_alpha9_radar_grid_5models', 'TRAIT Personality Scores Across Nine Alpha Points')
    complete_models = [(m, l) for m, l in MODELS if len(all_scores[m]) == len(ALPHAS)]
    p2, s2 = render_grid(all_scores, complete_models, 'trait_alpha9_radar_grid_complete_models', 'TRAIT Personality Scores Across Nine Alpha Points')
    render_individual(all_scores)
    print(p1)
    print(s1)
    print(p2)
    print(s2)
    print(OUT_DIR / 'trait_alpha9_radar_coverage.csv')


if __name__ == '__main__':
    main()
