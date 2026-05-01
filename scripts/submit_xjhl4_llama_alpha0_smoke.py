#!/usr/bin/env python3
import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


MODELS = {
    "llama32_1b": "/data1/xjh/code/pruning/models/Llama-3.2-1B-Instruct",
    "llama32_3b": "/data1/xjh/code/pruning/models/Llama-3.2-3B-Instruct",
}


def build_lane(repo: Path, run_root: Path, model_key: str, gpu: int, args: argparse.Namespace) -> str:
    model_id = MODELS[model_key]
    out = run_root / model_key
    py = repo / ".venv" / "bin" / "python"

    direct_vllm_common = [
        "--alpha", "0",
        "--gpu", "0",
        "--model_id", model_id,
        "--local_files_only",
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--max_model_len", str(args.max_model_len),
        "--enforce_eager",
    ]

    standard = [
        str(py), "scripts/eval_script1_standard.py",
        *direct_vllm_common,
        "--num_samples", str(args.standard_samples),
        "--max_tokens", str(args.standard_max_tokens),
        "--output_root", str(out / "standard"),
    ]
    bfcl = [
        str(py), "scripts/eval_script2_bfcl.py",
        *direct_vllm_common,
        "--num_samples", str(args.bfcl_samples),
        "--output_root", str(out / "bfcl"),
    ]
    judge = [
        str(py), "scripts/eval_script3_judge.py",
        *direct_vllm_common,
        "--num_samples", str(args.judge_samples),
        "--judge_model", args.judge_model,
        "--judge_workers", str(args.judge_workers),
        "--output_root", str(out / "judge"),
    ]
    lm_eval = [
        str(py), "scripts/eval_script4_lm_eval_tasks.py",
        "--alpha", "0",
        "--gpu", str(gpu),
        "--model_id", model_id,
        "--local_files_only",
        "--backend", "vllm",
        "--preset", "requested",
        "--limit", str(args.lm_eval_limit),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--max_model_len", str(args.max_model_len),
        "--enforce_eager",
        "--output_root", str(out / "lm_eval"),
    ]

    steps = [
        ("standard", standard),
        ("bfcl", bfcl),
        ("judge", judge),
        ("lm_eval", lm_eval),
    ]

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo))}",
        f"mkdir -p {shlex.quote(str(out))}/logs",
        f"echo lane={model_key} gpu={gpu} start=$(date -Is) | tee {shlex.quote(str(out))}/lane.status",
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
        "if [ -z \"${DEEPSEEK_API_KEY:-}\" ]; then",
        "  echo 'DEEPSEEK_API_KEY is required for judge smoke' >&2",
        "  exit 2",
        "fi",
        f": > {shlex.quote(str(out))}/timing.tsv",
    ]

    for name, cmd in steps:
        log = out / "logs" / f"{name}.log"
        lines.extend(
            [
                f"echo step={name} start=$(date -Is) | tee -a {shlex.quote(str(out))}/lane.status",
                "t0=$(date +%s)",
                f"{shlex.join(cmd)} > {shlex.quote(str(log))} 2>&1",
                "t1=$(date +%s)",
                f"printf '%s\\t%s\\t%s\\n' {shlex.quote(name)} \"$t0\" \"$t1\" >> {shlex.quote(str(out))}/timing.tsv",
                f"echo step={name} done=$(date -Is) seconds=$((t1-t0)) | tee -a {shlex.quote(str(out))}/lane.status",
            ]
        )
    lines.append(f"echo lane={model_key} done=$(date -Is) | tee -a {shlex.quote(str(out))}/lane.status")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data1/xjh/code/pruning_codex"))
    parser.add_argument("--run_root", type=Path, default=None)
    parser.add_argument("--gpu_1b", type=int, default=2)
    parser.add_argument("--gpu_3b", type=int, default=3)
    parser.add_argument("--standard_samples", type=int, default=20)
    parser.add_argument("--bfcl_samples", type=int, default=20)
    parser.add_argument("--judge_samples", type=int, default=10)
    parser.add_argument("--judge_workers", type=int, default=8)
    parser.add_argument("--lm_eval_limit", type=int, default=2)
    parser.add_argument("--judge_model", type=str, default="v4-flash")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--standard_max_tokens", type=int, default=256)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_root or Path(f"/data1/xjh/runs/pruning_codex/llama_alpha0_smoke_{stamp}")
    run_root.mkdir(parents=True, exist_ok=True)

    lanes = [("llama32_1b", args.gpu_1b), ("llama32_3b", args.gpu_3b)]
    for model_key, gpu in lanes:
        lane_script = run_root / f"run_{model_key}.sh"
        lane_script.write_text(build_lane(args.repo, run_root, model_key, gpu, args))
        lane_script.chmod(0o755)
        log = run_root / f"{model_key}.nohup.log"
        with log.open("wb") as fh:
            proc = subprocess.Popen(["nohup", "bash", str(lane_script)], stdout=fh, stderr=subprocess.STDOUT)
        print(f"{model_key}\tgpu={gpu}\tpid={proc.pid}\tlog={log}")
    print(f"RUN_ROOT={run_root}")


if __name__ == "__main__":
    main()
