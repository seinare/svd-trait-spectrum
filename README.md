# SVD Trait Spectrum: Llama Matthew Perturbation

This repository explores the "Energy-Conserving Matthew Operator" (Matthew Perturbation) on Llama-3.2 models. By applying SVD-based spectral scaling on MLP layers, we investigate how model "personality"—specifically reasoning, factuality, and tool-calling—shifts across the smoothing-sharpening spectrum.

## Repository Structure
- `scripts/`: Implementation of the Matthew Operator and evaluation pipelines.
  - `eval_script1_standard.py`: Reasoning & Math (GSM8K, MATH, ARC-Challenge, DROP).
  - `eval_script1_standard_cot.py`: CoT-only GSM8K and MATH evaluation with stricter final-answer parsing.
  - `eval_script2_bfcl.py`: Tool-Calling (Berkeley Function Calling Leaderboard).
  - `eval_script3_judge.py`: LLM-as-a-judge (TruthfulQA, HaluEval, AdvBench).
  - `eval_script4_lm_eval_tasks.py`: lm-evaluation-harness tasks grouped by evaluation meaning: knowledge understanding (MMLU), hard science reasoning (GPQA), exam reasoning (AGIEval), commonsense reasoning (HellaSwag), and instruction following (IFEval).
- `results/`: Multi-tier evaluation results in JSON format.
  - `standard/`: Mechanical evaluation metrics.
  - `bfcl/`: Tool calling compliance and recovery metrics.
  - `judge/`: Factual truthfulness and safety scores with bad-case logs.

## Evaluation layout

The evaluation surface is split by semantics rather than by implementation convenience:

1. `eval_script1_standard.py`: direct-answer cognitive tasks. It runs GSM8K, MATH, ARC-Challenge, and DROP without CoT.
2. `eval_script1_standard_cot.py`: isolated CoT tasks. It currently keeps only GSM8K and MATH because Llama 3.2 format control was unstable on broader CoT coverage.
3. `eval_script2_bfcl.py`: tool-use syntax and recovery. It runs BFCL v3 simple, checks AST/tool-call correctness, and retries malformed calls up to 3 attempts.
4. `eval_script3_judge.py`: judge-scored factuality, groundedness, and safety. It generates answers for TruthfulQA, HaluEval, and AdvBench, then scores them with a configurable judge model. The default judge model is `v4-flash`, which is resolved to the API model name `deepseek-v4-flash`; set the key with `DEEPSEEK_API_KEY` or pass `--api_key`.
5. `eval_script4_lm_eval_tasks.py`: lm-evaluation-harness coverage grouped as `mmlu`, `gpqa`, `agieval`, `hellaswag`, and `ifeval`. The default `all` preset runs those groups in that order; `requested` is kept as an alias.

## Multi-GPU and additional benchmark usage

Use `uv sync` to build the environment. The project pins Python to 3.11 because Triton needs Python development headers at runtime; on systems where `/usr/include/python*` is missing, use `uv python install 3.11` before `uv sync`.

The vLLM-backed scripts accept `--tensor_parallel_size`, `--gpu_memory_utilization`, `--max_model_len`, and `--enforce_eager`. When a Slurm allocation exposes more than one compatible GPU, set `--tensor_parallel_size` to the number of visible GPUs. Use `--enforce_eager` on clusters where Torch Inductor/Triton compile paths fail.

Example additional benchmark smoke run:

```bash
python scripts/eval_script4_lm_eval_tasks.py \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
  --alpha 0.0 \
  --preset gpqa \
  --preset hellaswag \
  --smoke \
  --tensor_parallel_size 1
```

For full requested coverage, omit `--smoke`; by default script 4 expands `--preset all` in this order:

1. Knowledge understanding: MMLU subject splits.
2. Hard science reasoning: GPQA main/extended/diamond zeroshot.
3. Exam reasoning: AGIEval English subtasks only.
4. Commonsense reasoning: HellaSwag.
5. Instruction following: IFEval.

The old `--preset requested` name is kept as an alias for `--preset all`.

Example judge smoke run:

```bash
export DEEPSEEK_API_KEY=...

uv run python scripts/eval_script3_judge.py \
  --model_id Qwen/Qwen3-8B \
  --alpha 0.0 \
  --judge_model v4-flash \
  --smoke \
  --num_samples 1
```

Qwen3-8B smoke-tested command on `wzq`:

```bash
export HF_HOME=/data1/cse12111103/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER=PCI_BUS_ID

uv run python scripts/eval_script4_lm_eval_tasks.py \
  --model_id Qwen/Qwen3-8B \
  --alpha 0.0 \
  --preset hellaswag \
  --smoke \
  --backend vllm \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.75 \
  --max_model_len 2048 \
  --enforce_eager \
  --batch_size auto
```

## Current Status
See [status.md](status.md) for detailed progress, findings, and technical challenges encountered during the experiments.
