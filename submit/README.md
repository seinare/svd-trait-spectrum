# SVD Trait Spectrum Submit Bundle

This folder contains the core code used for the SVD perturbation experiments and the capability/personality analyses. It is intentionally a lightweight code bundle: large raw outputs, model checkpoints, caches, and API keys are not included.

## Project Summary

The project studies how singular-value perturbations of LLM MLP `up_proj` and `down_proj` matrices affect task performance, fitted capability dimensions, TRAIT personality scores, and output-distribution stability. The main perturbation is the energy-conserving Matthew operator:

```text
log s_i' = mean(log s) + (1 + alpha) * (log s_i - mean(log s))
```

Positive `alpha` sharpens the singular-value spectrum and negative `alpha` smooths it, while preserving the geometric mean of singular values in log space. Magnitude perturbation scripts are also included for beta-scaling variants.

## Folder Contents

- `pyproject.toml`, `uv.lock`: uv environment definition.
- `scripts/eval_script1_standard.py`: standard benchmark block, including GSM8K, MATH, ARC, MMLU, GPQA, AGIEval, HellaSwag, and IFEval-style tasks used in the project pipeline.
- `scripts/eval_script1_standard_cot.py`: isolated CoT benchmark script for controlled CoT runs.
- `scripts/eval_script2_bfcl.py`: BFCL/function-calling evaluation.
- `scripts/eval_script3_judge.py`: judge-model evaluation using an OpenAI-compatible API such as DeepSeek.
- `scripts/eval_script4_lm_eval_tasks.py`: reduced `lm-eval` suite.
- `scripts/eval_script5_trait_personality.py`: TRAIT softmax log-likelihood scoring.
- `scripts/eval_script6_lm_eval_full_subtasks.py`: Eval6 full subtask suite for MMLU-Pro, MMLU-Redux, AGIEval, and BBH.
- `scripts/eval_fineweb_bpb_matthew.py`: FineWeb validation BPB evaluation.
- `scripts/eval_fineweb_distribution_kl_matthew.py`: perturbation-induced token distribution KL evaluation.
- `scripts/analyze_*.py`, `scripts/summarize_*.py`: capability fitting, spectrum statistics, TRAIT summaries, and result table generation.
- `scripts/plot_*.py`: figure generation for capability error bars, capability-TRAIT relationships, radar plots, and singular-value spectra.
- `scripts/submit_*.py`, `scripts/run_fineweb_bpb_worker.sh`: representative Slurm/job-launch helpers.

## Environment

Install with uv from this folder:

```bash
cd submit
uv sync
```

For Hugging Face or judge-model access, set environment variables instead of hardcoding keys:

```bash
export HF_TOKEN=...
export DEEPSEEK_API_KEY=...
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
```

The judge scripts default to an OpenAI-compatible API and can use DeepSeek by setting:

```bash
export DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

## Smoke-Test Commands

Run a small standard benchmark smoke test at `alpha=0`:

```bash
uv run python scripts/eval_script1_standard.py \
  --model_id Qwen/Qwen3-8B \
  --alpha 0 \
  --smoke \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.60 \
  --max_model_len 4096 \
  --max_tokens 256 \
  --output_root results/smoke/standard
```

Run an Eval6 subtask smoke test:

```bash
uv run python scripts/eval_script6_lm_eval_full_subtasks.py \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --alpha 0 \
  --limit 5 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.60 \
  --output_root results/smoke/eval6
```

Run TRAIT scoring with softmax log-likelihood:

```bash
uv run python scripts/eval_script5_trait_personality.py \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --alpha 0 \
  --limit 20 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.60 \
  --output_root results/smoke/trait
```

Run judge-scored evaluation:

```bash
uv run python scripts/eval_script3_judge.py \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --alpha 0 \
  --smoke \
  --judge_model v4-flash \
  --judge_base_url "${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}" \
  --output_root results/smoke/judge
```

## Full-Run Examples

Run one alpha point for the five main modules:

```bash
MODEL=meta-llama/Llama-3.2-1B-Instruct
ALPHA=-0.2

uv run python scripts/eval_script1_standard.py --model_id "$MODEL" --alpha "$ALPHA" --output_root results/full/standard
uv run python scripts/eval_script2_bfcl.py --model_id "$MODEL" --alpha "$ALPHA" --output_root results/full/bfcl
uv run python scripts/eval_script3_judge.py --model_id "$MODEL" --alpha "$ALPHA" --judge_model v4-flash --output_root results/full/judge
uv run python scripts/eval_script4_lm_eval_tasks.py --model_id "$MODEL" --alpha "$ALPHA" --output_root results/full/lm_eval
uv run python scripts/eval_script6_lm_eval_full_subtasks.py --model_id "$MODEL" --alpha "$ALPHA" --output_root results/full/eval6
```

For multi-GPU Slurm runs, use the included `submit_*.py` scripts as templates and adjust model paths, partition names, GPU counts, and output roots for the target cluster.

## Analysis and Figure Commands

After syncing raw Eval6 outputs into `docs/raw` and judge capability weights into `docs/results/capability_dimension`, run:

```bash
uv run python scripts/analyze_eval6_all_models_alpha9.py
uv run python scripts/analyze_eval6_capability_5panel_independent.py
uv run python scripts/plot_eval6_capability_3x5_independent.py
```

For TRAIT summaries and radar plots:

```bash
uv run python scripts/summarize_trait_alpha9_tables.py
uv run python scripts/plot_trait_alpha9_radar_grid.py
```

For capability-TRAIT correlation figures:

```bash
uv run python scripts/analyze_trait_capability_correlations.py
uv run python scripts/plot_trait_capability_ab_figure.py
uv run python scripts/plot_trait_capability_panel_c_mpl.py
```

For singular-value spectrum analysis:

```bash
uv run python scripts/analyze_svd_alpha_spectrum.py
uv run python scripts/plot_svd_alpha_spectrum.py
```

## Notes

- Temporary perturbed model directories should be placed under fast local scratch and deleted after each evaluation. The evaluation scripts support this through `--tmp_root`, `--prepared_model_dir`, and `--save_prepared_model_dir`.
- Qwen3 thinking mode is disabled in the non-CoT standard script. CoT behavior is isolated in `eval_script1_standard_cot.py`.
- The scripts expect normal CUDA/vLLM availability. Use `--local_files_only` when models are already present on disk.
