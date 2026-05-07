#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


MODEL_ID = "/data1/xjh/code/pruning/models/Llama-3.2-1B-Instruct"


def stage_command(repo: Path, run_root: Path, stage: str, args: argparse.Namespace) -> list[str]:
    py = repo / ".venv" / "bin" / "python"
    common = [
        "--alpha", "0",
        "--gpu", "0",
        "--model_id", MODEL_ID,
        "--local_files_only",
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--enforce_eager",
    ]
    if stage == "standard":
        return [
            str(py), "scripts/eval_script1_standard.py",
            *common,
            "--max_model_len", str(args.max_model_len),
            "--max_tokens", str(args.standard_max_tokens),
            "--output_root", str(run_root / "standard"),
        ]
    if stage == "standard_cot":
        return [
            str(py), "scripts/eval_script1_standard_cot.py",
            *common,
            "--max_model_len", str(args.cot_max_model_len),
            "--max_tokens", str(args.cot_max_tokens),
            "--thinking_budget", str(args.thinking_budget),
            "--output_root", str(run_root / "standard_cot"),
        ]
    if stage == "bfcl":
        return [
            str(py), "scripts/eval_script2_bfcl.py",
            *common,
            "--max_model_len", str(args.max_model_len),
            "--output_root", str(run_root / "bfcl"),
        ]
    if stage == "judge":
        return [
            str(py), "scripts/eval_script3_judge.py",
            *common,
            "--max_model_len", str(args.max_model_len),
            "--judge_model", args.judge_model,
            "--judge_workers", str(args.judge_workers),
            "--output_root", str(run_root / "judge"),
        ]
    if stage == "lm_eval":
        return [
            str(py), "scripts/eval_script4_lm_eval_tasks.py",
            *common,
            "--max_model_len", str(args.max_model_len),
            "--backend", "vllm",
            "--preset", "requested",
            "--batch_size", args.lm_eval_batch_size,
            "--output_root", str(run_root / "lm_eval"),
        ]
    raise ValueError(f"unknown stage: {stage}")


def write_stage_script(repo: Path, run_root: Path, stage: str, gpu: int, args: argparse.Namespace) -> Path:
    cmd = stage_command(repo, run_root, stage, args)
    script = run_root / f"run_{stage}.sh"
    log = run_root / "logs" / f"{stage}.log"
    status = run_root / "status" / f"{stage}.status"
    timing = run_root / "timing.tsv"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo))}",
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"export CUDA_VISIBLE_DEVICES={gpu}",
        "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "export VLLM_NO_USAGE_STATS=1",
        "export HF_HOME=${HF_HOME:-/data1/xjh/.cache/huggingface}",
        "export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/data1/xjh/.cache/huggingface/hub}",
        "export HF_DATASETS_CACHE=/data1/xjh/.cache/huggingface/datasets",
        "export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}",
        "export UV_CACHE_DIR=${UV_CACHE_DIR:-/data1/xjh/.cache/uv}",
        "if [ -z \"${HF_TOKEN:-}\" ] && [ -f /data1/xjh/.config/svd/hf_token ]; then",
        "  export HF_TOKEN=$(tr -d '\\n' < /data1/xjh/.config/svd/hf_token)",
        "fi",
        "if [ -z \"${DEEPSEEK_API_KEY:-}\" ] && [ -f /data1/xjh/.config/svd/deepseek_api_key ]; then",
        "  export DEEPSEEK_API_KEY=$(tr -d '\\n' < /data1/xjh/.config/svd/deepseek_api_key)",
        "fi",
        f"echo stage={stage} gpu={gpu} start=$(date -Is) | tee {shlex.quote(str(status))}",
        "t0=$(date +%s)",
        f"{shlex.join(cmd)} > {shlex.quote(str(log))} 2>&1",
        "t1=$(date +%s)",
        f"printf '%s\\t%s\\t%s\\t%s\\n' {shlex.quote(stage)} {gpu} \"$t0\" \"$t1\" >> {shlex.quote(str(timing))}",
        f"echo stage={stage} done=$(date -Is) seconds=$((t1-t0)) | tee -a {shlex.quote(str(status))}",
    ]
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    return script


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data1/xjh/code/pruning_codex"))
    parser.add_argument("--run_root", type=Path, default=None)
    parser.add_argument("--gpus", type=str, default="2,3,4,5,6")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--standard_max_tokens", type=int, default=256)
    parser.add_argument("--cot_max_model_len", type=int, default=8192)
    parser.add_argument("--cot_max_tokens", type=int, default=8192)
    parser.add_argument("--thinking_budget", type=int, default=1024)
    parser.add_argument("--judge_model", type=str, default="v4-flash")
    parser.add_argument("--judge_workers", type=int, default=20)
    parser.add_argument("--lm_eval_batch_size", type=str, default="auto")
    args = parser.parse_args()

    stages = ["standard", "standard_cot", "bfcl", "judge", "lm_eval"]
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(stages):
        raise SystemExit(f"Need at least {len(stages)} GPUs, got {gpus}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_root or Path(f"/data1/xjh/runs/pruning_codex/llama1b_alpha0_five_modules_{stamp}")
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "status").mkdir(parents=True, exist_ok=True)
    (run_root / "pids").mkdir(parents=True, exist_ok=True)
    (run_root / "timing.tsv").write_text("")

    for stage, gpu in zip(stages, gpus):
        script = write_stage_script(args.repo, run_root, stage, gpu, args)
        nohup_log = run_root / "logs" / f"{stage}.nohup.log"
        with nohup_log.open("wb") as fh:
            proc = subprocess.Popen(["nohup", "bash", str(script)], stdout=fh, stderr=subprocess.STDOUT)
        (run_root / "pids" / f"{stage}.pid").write_text(str(proc.pid) + "\n")
        print(f"{stage}\tgpu={gpu}\tpid={proc.pid}\tlog={run_root / 'logs' / (stage + '.log')}")
    print(f"RUN_ROOT={run_root}")


if __name__ == "__main__":
    main()
