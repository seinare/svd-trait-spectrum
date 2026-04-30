# SVD Trait Spectrum: Llama Matthew Perturbation

This repository explores the "Energy-Conserving Matthew Operator" (Matthew Perturbation) on Llama-3.2 models. By applying SVD-based spectral scaling on MLP layers, we investigate how model "personality"—specifically reasoning, factuality, and tool-calling—shifts across the smoothing-sharpening spectrum.

## Repository Structure
- `scripts/`: Implementation of the Matthew Operator and evaluation pipelines.
  - `eval_script1_standard.py`: Reasoning & Math (GSM8K, MATH, GPQA, DROP).
  - `eval_script2_bfcl.py`: Tool-Calling (Berkeley Function Calling Leaderboard).
  - `eval_script3_judge.py`: LLM-as-a-judge (TruthfulQA, HaluEval, AdvBench).
- `results/`: Multi-tier evaluation results in JSON format.
  - `standard/`: Mechanical evaluation metrics.
  - `bfcl/`: Tool calling compliance and recovery metrics.
  - `judge/`: Factual truthfulness and safety scores with bad-case logs.

## Current Status
See [status.md](status.md) for detailed progress, findings, and technical challenges encountered during the experiments.
