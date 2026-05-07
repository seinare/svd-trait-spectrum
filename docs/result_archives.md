# Result Archives

This repository keeps scripts and compact summaries in git. Large raw result trees are archived on the experiment hosts to avoid committing generated JSON/log payloads.

## Raw archives

| Model / run | Host | Archive | Source directories | Notes |
| --- | --- | --- | --- | --- |
| Llama-3.2-1B-Instruct alpha sweep | `xjhl4` | `/data1/xjh/runs/pruning_codex/archive/llama1b_6alpha_5modules_20260502_060314.raw.tgz` | `/data1/xjh/runs/pruning_codex/llama1b_6alpha_5modules_20260502_060314`, `/data1/xjh/runs/pruning_codex/llama1b_alpha0_five_modules_20260502_045240` | Completed for standard, BFCL, judge, lm-eval, and standard-cot. |
| Llama-3.2-3B-Instruct alpha sweep | `xjhl4` | not yet tarred | `/data1/xjh/runs/pruning_codex/llama3b_7alpha_5modules_20260502_070618` | Completed for standard, BFCL, judge, lm-eval, and standard-cot across alpha -0.3 to 0.3. |
| TRAIT personality sweep | `xjhl4` | not yet tarred | `/data1/xjh/runs/pruning_codex/trait_3models_4gpu_20260502_130048` | Completed for Llama-3.2-1B, Llama-3.2-3B, and Qwen3-8B on their measured alpha grids. |
| Qwen3-8B partial and targeted supplement | `xjha6` | `/data1/xjh/runs/svd-trait-spectrum/archive/qwen3_existing_and_targeted_20260502.raw.tgz` | `/data1/xjh/runs/svd-trait-spectrum/xjha6_sbj9_20260501_181602`, `/data1/xjh/runs/svd-trait-spectrum/xjha6_qwen_missing_lmeval_cot_20260502_064202` | Existing `standard -> bfcl -> judge` results plus the targeted missing lm-eval/cot supplement run. |
| Qwen3-8B TP=2 residual supplement | `xjha6` | not yet tarred | `/data1/xjh/runs/svd-trait-spectrum/xjha6_qwen_residual_tp2_20260502_103759` | Fills `-0.4` judge and remaining Qwen lm-eval / standard-cot gaps; still running while the current tables were generated. |

## Overwrite check

The targeted Qwen supplement writes to `/data1/xjh/runs/svd-trait-spectrum/xjha6_qwen_missing_lmeval_cot_20260502_064202`. The existing Qwen result directory `/data1/xjh/runs/svd-trait-spectrum/xjha6_sbj9_20260501_181602` was kept read-only by convention and was not used as an output root for the supplement launcher.

An accidentally started full Qwen rerun wrote only to its own directory, `/data1/xjh/runs/svd-trait-spectrum/xjha6_qwen_full_9alpha_5modules_20260502_055828`, and was stopped before being treated as canonical data.
