# Data Bundle

This directory contains selected lightweight data tables needed to inspect or reproduce the main analyses in the submit bundle. It does not include model checkpoints, vLLM caches, Slurm logs, full generated-answer dumps, or large matrix-level singular-value dumps.

## Directory Layout

### `benchmark_scores/`

Core benchmark score summaries across model, perturbation alpha, module, and subtask.

- `all_module_metrics_long.csv`: long-form module-level scores for the main benchmark blocks.
- `all_module_summary_wide.csv`: compact wide table of module-level scores.
- `all_mmlu_subtasks_alpha.csv`, `all_agieval_subtasks_alpha.csv`, `all_gpqa_subtasks_alpha.csv`: subtask score tables for MMLU, AGIEval, and GPQA.
- `all_lm_eval_alpha_subtasks.csv`, `all_lm_eval_subtasks_long.csv`: lm-eval subtask tables.
- `all_coverage_matrix.csv`: result coverage matrix.
- `qwen_moe_lm_eval_mmlu20_agieval_gpqa.csv`: Qwen3-30B-A3B MoE lm-eval summary for MMLU20, AGIEval English, and GPQA.

### `capability_judge/`

DeepSeek-v4-pro judged capability-demand weights and seven-dimension basis diagnostics.

- `eval6_deepseek_v4_pro_capability_weights.csv`: averaged three-dimension capability weights for Eval6 subtasks.
- `eval6_deepseek_v4_pro_capability_coverage.csv`: coverage of valid judge responses.
- `eval6_deepseek_v4_pro_capability_raw.jsonl.gz`: compressed raw judge responses for the Eval6 capability-weight run.
- `seven_dim_*.csv`: appendix data showing how the original seven capability dimensions project into the retained three-dimensional basis.

### `eval6_capability_fit/`

Data used for fitting Eval6 subtask perturbation responses to three capability dimensions.

- `eval6_all_models_subtask_scores_long.csv`: raw Eval6 subtask score series across models and alpha values.
- `capability_alpha_mle_errorbar_points.csv`: per-alpha maximum-likelihood fitted capability shifts and uncertainty scales.
- `capability_alpha_quadratic_curves.csv`: fitted alpha-response curves for capability dimensions.
- `capability_5panel_mean_sigma.csv`, `capability_independent_points_long.csv`, `capability_independent_nll.csv`: module-wise independent capability fits.
- `task_alpha_quadratic_slopes.csv`: direct subtask-level alpha-response fits.
- `task_capability_weight_pca2d.csv`, `task_capability_weight_tsne2d.csv`: capability-demand embedding coordinates.
- `capability_curve_points_by_module.csv`, `capability_fk_coefficients_by_module.csv`, `dimension_tau_mle_by_module.csv`, `task_likelihood_errors_by_module.csv`: module-separated capability fit data.
- `eval6_capability_5panel_independent.md`: table-form report for the five-panel capability fit.

### `trait/`

TRAIT score tables and capability-TRAIT association data.

- `*_trait_alpha9_table.csv`: per-model TRAIT scores over the nine-point alpha grid.
- `trait_alpha9_radar_scores_long.csv`: long-form TRAIT scores used for radar plots.
- `trait_alpha9_coverage.csv`, `trait_alpha9_radar_coverage.csv`: coverage tables.
- `trait_capability_points_long.csv`: aligned capability and TRAIT deltas across model-alpha points.
- `trait_capability_pearson.csv`: Pearson correlations between capability responses and TRAIT responses.
- `pooled_trait_on_capability_regression.csv`, `per_model_trait_on_capability_regression.csv`, `pooled_predictions_long.csv`, `leave_one_model_out_rmse.csv`: regression diagnostics.

### `svd_spectrum/`

Singular-value spectrum summaries used to analyze layer-wise spectral nonuniformity.

- `combined_depth_alpha_svd_summary.csv`: depth-binned alpha perturbation statistics.
- `combined_layer_alpha_svd_summary.csv`: layer-level singular-value statistics.
- `table_*.csv`: compact report tables for top singular value and Gini changes.
- `svd_alpha_spectrum_report.md`, `svd_alpha_layer_nonuniformity_report.md`: narrative summaries.

The full matrix-level singular-value dump is intentionally excluded because it is large and not needed for the main manuscript tables and figures.

### `fineweb_kl/`

FineWeb 128k-token distribution-shift summaries.

- `fineweb_distribution_kl_128k_long.csv`: alpha-level KL results.
- `fineweb_distribution_kl_128k_summary.csv`: compact summary.
- `fineweb_distribution_kl_128k_report.md`: report text.

### `magnitude_beta/`

Magnitude/beta perturbation score summaries for Llama 3.2 1B Instruct.

- `magnitude_beta_lm_eval_subtasks_long.csv`: long-form beta sweep results.
- `magnitude_beta_mmlu_subtasks.csv`, `magnitude_beta_agieval_subtasks.csv`: subtask summaries.
- `magnitude_beta_mmlu_subtasks.md`, `magnitude_beta_agieval_subtasks.md`: table reports.

### `figure_support/`

Small tables used by the capability-TRAIT main figure.

- `panel_a_alpha_segment_slopes.csv`: local alpha-response slopes.
- `panel_b_trait_direction_slopes.csv`: TRAIT directional slopes.
- `panel_c_standalone_selected_subtasks.csv`: selected subtask links for the path diagram.
- `trait_capability_abc_report.md`: report text documenting the figure construction.

## Notes

- Scores are stored as produced by the project scripts. Some files contain deltas relative to alpha 0, while others contain absolute benchmark scores; inspect column names before combining tables.
- Negative alpha corresponds to spectral smoothing and positive alpha to spectral spiking under the Matthew singular-value operator.
- The compressed `.jsonl.gz` file can be inspected with `gzip -cd <file> | head`.
