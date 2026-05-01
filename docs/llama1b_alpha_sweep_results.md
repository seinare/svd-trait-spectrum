# Llama-3.2-1B Alpha Sweep Results

Canonical raw run:

- Sweep: `/data1/xjh/runs/pruning_codex/llama1b_6alpha_5modules_20260502_060314`
- Alpha 0 baseline: `/data1/xjh/runs/pruning_codex/llama1b_alpha0_five_modules_20260502_045240`
- Raw archive: `/data1/xjh/runs/pruning_codex/archive/llama1b_6alpha_5modules_20260502_060314.raw.tgz`

## Standard

| alpha | GSM8K | MATH | ARC | DROP |
| ---: | ---: | ---: | ---: | ---: |
| -0.3 | 0.15 | 0.16 | 0.23 | 0.12 |
| -0.2 | 0.20 | 0.21 | 0.35 | 0.25 |
| -0.1 | 0.17 | 0.28 | 0.43 | 0.24 |
| 0.0 | 0.19 | 0.19 | 0.48 | 0.23 |
| 0.1 | 0.08 | 0.12 | 0.49 | 0.23 |
| 0.2 | 0.01 | 0.12 | 0.43 | 0.13 |
| 0.3 | 0.01 | 0.10 | 0.08 | 0.11 |

## BFCL

| alpha | first-pass | retry |
| ---: | ---: | ---: |
| -0.3 | 0.010 | 0.010 |
| -0.2 | 0.055 | 0.085 |
| -0.1 | 0.040 | 0.090 |
| 0.0 | 0.000 | 0.010 |
| 0.1 | 0.000 | 0.045 |
| 0.2 | 0.000 | 0.005 |
| 0.3 | 0.000 | 0.000 |

## Judge

| alpha | TruthfulQA | HaluEval | AdvBench |
| ---: | ---: | ---: | ---: |
| -0.3 | 0.49 | 0.50 | 2.70 |
| -0.2 | 0.58 | 0.89 | 2.74 |
| -0.1 | 0.56 | 1.21 | 2.83 |
| 0.0 | 0.70 | 1.36 | 2.83 |
| 0.1 | 0.71 | 1.35 | 2.75 |
| 0.2 | 0.62 | 1.11 | 2.79 |
| 0.3 | 0.62 | 0.78 | 2.76 |

## lm-eval

| alpha | mean | MMLU | GPQA | AGIEval | HellaSwag | IFEval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.3 | 0.325 | 0.372 | 0.250 | 0.229 | 0.431 | 0.257 |
| -0.2 | 0.347 | 0.395 | 0.256 | 0.247 | 0.445 | 0.336 |
| -0.1 | 0.362 | 0.419 | 0.245 | 0.249 | 0.456 | 0.396 |
| 0.0 | 0.375 | 0.429 | 0.263 | 0.262 | 0.456 | 0.442 |
| 0.1 | 0.368 | 0.421 | 0.264 | 0.260 | 0.448 | 0.412 |
| 0.2 | 0.337 | 0.383 | 0.258 | 0.238 | 0.429 | 0.368 |
| 0.3 | 0.289 | 0.318 | 0.270 | 0.214 | 0.393 | 0.274 |

## Standard CoT

| alpha | GSM8K acc | GSM8K format err | GSM8K length err | MATH acc | MATH format err | MATH length err |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -0.3 | 0.15 | 0.07 | 0.04 | 0.16 | 0.31 | 0.31 |
| -0.2 | 0.31 | 0.03 | 0.01 | 0.19 | 0.20 | 0.19 |
| -0.1 | 0.33 | 0.03 | 0.00 | 0.28 | 0.14 | 0.12 |
| 0.0 | 0.37 | 0.03 | 0.02 | 0.31 | 0.12 | 0.11 |
| 0.1 | 0.28 | 0.02 | 0.01 | 0.37 | 0.23 | 0.21 |
| 0.2 | 0.20 | 0.12 | 0.07 | 0.14 | 0.50 | 0.46 |
| 0.3 | 0.03 | 0.24 | 0.11 | 0.02 | 0.57 | 0.47 |

## Notes

The overall best region is around alpha 0 to -0.1 depending on the module. Strong positive perturbations degrade most metrics, especially standard reasoning, BFCL, lm-eval, and CoT format stability. CoT MATH is particularly sensitive to positive alpha because format and length errors rise sharply.

