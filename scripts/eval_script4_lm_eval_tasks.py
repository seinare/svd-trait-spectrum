import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Knowledge and broad academic understanding. Keep a representative 20-subject
# subset so lm-eval remains the broad benchmark without dominating wall time.
MMLU_SUBTASKS = [
    "mmlu_abstract_algebra",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_physics",
    "mmlu_computer_security",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_physics",
    "mmlu_international_law",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_philosophy",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
]

# Hard science reasoning. These are gated on Hugging Face and require HF_TOKEN.
GPQA_SUBTASKS = ["gpqa_main_zeroshot", "gpqa_extended_zeroshot", "gpqa_diamond_zeroshot"]

# English exam-style analytical reasoning.
AGIEVAL_SUBTASKS = [
    "agieval_aqua_rat",
    "agieval_logiqa_en",
    "agieval_lsat_ar",
    "agieval_lsat_lr",
    "agieval_lsat_rc",
    "agieval_sat_en",
    "agieval_sat_en_without_passage",
    "agieval_sat_math",
]

COMMONSENSE_REASONING_TASKS = ["hellaswag"]
INSTRUCTION_FOLLOWING_TASKS = ["ifeval"]

EVAL_GROUPS = [
    {
        "name": "mmlu",
        "label": "knowledge_understanding",
        "description": "MMLU subject splits",
        "tasks": MMLU_SUBTASKS,
    },
    {
        "name": "gpqa",
        "label": "hard_science_reasoning",
        "description": "GPQA main/extended/diamond zeroshot",
        "tasks": GPQA_SUBTASKS,
    },
    {
        "name": "agieval",
        "label": "exam_reasoning",
        "description": "AGIEval English-language AQuA, LogiQA, LSAT, and SAT subtasks",
        "tasks": AGIEVAL_SUBTASKS,
    },
    {
        "name": "hellaswag",
        "label": "commonsense_reasoning",
        "description": "HellaSwag commonsense completion",
        "tasks": COMMONSENSE_REASONING_TASKS,
    },
    {
        "name": "ifeval",
        "label": "instruction_following",
        "description": "IFEval instruction-following compliance",
        "tasks": INSTRUCTION_FOLLOWING_TASKS,
    },
]

PRESETS = {
    group["name"]: group["tasks"] for group in EVAL_GROUPS
}
PRESETS["all"] = [task for group in EVAL_GROUPS for task in group["tasks"]]
PRESETS["requested"] = PRESETS["all"]

GROUP_LABELS = {group["name"]: group["label"] for group in EVAL_GROUPS}
GROUP_DESCRIPTIONS = {group["name"]: group["description"] for group in EVAL_GROUPS}


def apply_svd_energy_matthew(model, alpha=0.0):
    if alpha == 0.0:
        return
    import torch
    from tqdm import tqdm

    print(f"Applying SVD: alpha={alpha} on ['up_proj', 'down_proj']")
    with torch.no_grad():
        for layer in tqdm(model.model.layers, desc="SVD"):
            for name in ["up_proj", "down_proj"]:
                proj = getattr(layer.mlp, name)
                W = proj.weight.data.float()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                L = torch.log(torch.clamp(S, min=1e-9))
                M = torch.mean(L)
                L_new = L + alpha * (L - M)
                S_new = torch.exp(L_new)
                W_new = (U @ torch.diag(S_new) @ Vh).to(torch.bfloat16)
                proj.weight.data.copy_(W_new)


def expand_tasks(task_args, preset_args):
    tasks = []
    for preset in preset_args:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Available: {', '.join(sorted(PRESETS))}")
        tasks.extend(PRESETS[preset])
    tasks.extend(task_args)
    seen = set()
    return [task for task in tasks if not (task in seen or seen.add(task))]


def summarize_task_groups(task_args, preset_args, expanded_tasks):
    explicit_groups = []
    for preset in preset_args:
        if preset in ("all", "requested"):
            explicit_groups.extend(group["name"] for group in EVAL_GROUPS)
        else:
            explicit_groups.append(preset)

    seen_groups = set()
    groups = []
    for name in explicit_groups:
        if name in seen_groups:
            continue
        seen_groups.add(name)
        groups.append(
            {
                "name": name,
                "label": GROUP_LABELS.get(name, name),
                "description": GROUP_DESCRIPTIONS.get(name, ""),
                "tasks": PRESETS[name],
            }
        )

    custom_tasks = [task for task in task_args if task in expanded_tasks]
    if custom_tasks:
        groups.append(
            {
                "name": "custom",
                "label": "custom_tasks",
                "description": "Tasks supplied with --task",
                "tasks": custom_tasks,
            }
        )
    return groups


def build_lm_eval_command(args, model_dir, output_dir, tasks):
    if args.backend == "vllm":
        model_args = [
            f"pretrained={model_dir}",
            "dtype=bfloat16",
            "trust_remote_code=True",
            f"tensor_parallel_size={args.tensor_parallel_size}",
            f"gpu_memory_utilization={args.gpu_memory_utilization}",
            f"max_model_len={args.max_model_len}",
        ]
        if args.enforce_eager:
            model_args.append("enforce_eager=True")
    else:
        model_args = [
            f"pretrained={model_dir}",
            "dtype=bfloat16",
            "trust_remote_code=True",
        ]
        if args.parallelize_hf:
            model_args.append("parallelize=True")

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        args.backend,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        args.batch_size,
        "--output_path",
        output_dir,
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(args.num_fewshot)])
    if args.device and args.backend == "hf":
        cmd.extend(["--device", args.device])
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--preset", action="append", default=None, choices=sorted(PRESETS))
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--parallelize_hf", action="store_true")
    parser.add_argument("--output_root", type=str, default="results/lm_eval/eval_script4_lm_eval_tasks")
    parser.add_argument("--tmp_root", type=str, default="/tmp")
    parser.add_argument("--prepared_model_dir", type=str, default=None)
    parser.add_argument("--save_prepared_model_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    presets = args.preset if args.preset is not None else ["all"]
    tasks = expand_tasks(args.task, presets)
    task_groups = summarize_task_groups(args.task, presets, tasks)
    if args.smoke and args.limit is None:
        args.limit = 5
    if args.device is None:
        args.device = f"cuda:{args.gpu}"

    model_id = args.model_id
    model_name_safe = model_id.split("/")[-1].replace(".", "_")
    run_name = f"{model_name_safe}_alpha{args.alpha}_{args.backend}"
    output_dir = os.path.join(args.output_root, run_name)
    if os.path.exists(output_dir) and not args.force:
        print(f"Output exists: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_source = args.prepared_model_dir or model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cleanup_dir = None
    if args.prepared_model_dir:
        eval_model_dir = args.prepared_model_dir
    elif args.alpha == 0.0 and not args.save_prepared_model_dir:
        eval_model_dir = model_id
    else:
        eval_model_dir = args.save_prepared_model_dir or tempfile.mkdtemp(
            prefix=f"eval_script4_{model_name_safe}_alpha{args.alpha}_",
            dir=args.tmp_root,
        )
        if not args.save_prepared_model_dir:
            cleanup_dir = eval_model_dir
        device = f"cuda:{args.gpu}"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=args.local_files_only,
        )
        apply_svd_energy_matthew(model, args.alpha)
        os.makedirs(eval_model_dir, exist_ok=True)
        model.save_pretrained(eval_model_dir)
        tokenizer.save_pretrained(eval_model_dir)
        del model
        torch.cuda.empty_cache()

    try:
        metadata = {
            "method": "eval_script4_lm_eval_tasks",
            "alpha": args.alpha,
            "model_id": model_id,
            "backend": args.backend,
            "task_groups": task_groups,
            "tasks": tasks,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            "batch_size": args.batch_size,
            "prepared_model_dir": args.prepared_model_dir,
            "evaluation_style": "lm_eval_native_non_cot",
        }
        with open(os.path.join(output_dir, "run_config.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        cmd = build_lm_eval_command(args, eval_model_dir, output_dir, tasks)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
