# Qwen3-8B Result Status

Canonical existing run:

- `/data1/xjh/runs/svd-trait-spectrum/xjha6_sbj9_20260501_181602`

Targeted supplement run:

- `/data1/xjh/runs/svd-trait-spectrum/xjha6_qwen_missing_lmeval_cot_20260502_064202`

Raw archive:

- `/data1/xjh/runs/svd-trait-spectrum/archive/qwen3_existing_and_targeted_20260502.raw.tgz`

## Existing standard / BFCL / judge results

| alpha | GSM8K | MATH | ARC | DROP | BFCL first | BFCL retry | TruthfulQA | HaluEval | AdvBench |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.4 | 0.13 | 0.15 | 0.70 | 0.21 | 0.005 | 0.015 | pending | pending | pending |
| -0.3 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| -0.2 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| -0.1 | 0.84 | 0.45 | 0.90 | 0.63 | 0.095 | 0.395 | 0.86 | 1.39 | 2.89 |
| 0.0 | 0.87 | 0.42 | 0.89 | 0.68 | 0.220 | 0.510 | 0.93 | 1.35 | 2.93 |
| 0.1 | 0.80 | 0.38 | 0.91 | 0.73 | 0.245 | 0.580 | 1.00 | 1.41 | 2.86 |
| 0.2 | 0.71 | 0.35 | 0.85 | 0.60 | 0.075 | 0.270 | 0.98 | 1.41 | 2.83 |
| 0.3 | 0.25 | 0.21 | 0.70 | 0.27 | 0.130 | 0.265 | 0.79 | 1.09 | 2.89 |
| 0.4 | 0.08 | 0.15 | 0.27 | 0.06 | 0.005 | 0.015 | 0.77 | 0.51 | 2.38 |

## Targeted supplement plan

The supplement launcher fills the known gaps without writing into the old result directory:

| GPU | Work |
| --- | --- |
| 0 | `alpha=-0.4` judge, plus lm-eval and standard-cot for `alpha=-0.4,0,0.3` |
| 1 | `alpha=-0.3` standard/BFCL/judge, plus lm-eval and standard-cot for `alpha=-0.3,-0.1,0.1` |
| 2 | `alpha=-0.2` standard/BFCL/judge, plus lm-eval and standard-cot for `alpha=-0.2,0.2,0.4` |

At the time this summary was written, the supplement tasks were still running and had not reported hard failures. Treat the pending cells above as incomplete until the supplement run finishes and is parsed into a final table.

## Observed pattern

The current complete points show a broad optimum near alpha 0 to 0.1 for Qwen3-8B. Large negative alpha damages GSM8K/MATH/BFCL while preserving some ARC performance; large positive alpha degrades standard reasoning and BFCL, with alpha 0.4 showing a sharp collapse across ARC and DROP.

