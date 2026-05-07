#!/usr/bin/env python3
"""Heavy lm-eval suite with full subtasks for MMLU-Pro, MMLU-Redux, AGIEval, and BBH.

This script mirrors eval_script4_lm_eval_tasks.py but uses a broader task set.
It keeps BBH on non-CoT zeroshot tasks by default to match the non-CoT eval
policy used by the main suite.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MMLU_PRO_SUBTASKS = [
    "mmlu_pro_biology",
    "mmlu_pro_business",
    "mmlu_pro_chemistry",
    "mmlu_pro_computer_science",
    "mmlu_pro_economics",
    "mmlu_pro_engineering",
    "mmlu_pro_health",
    "mmlu_pro_history",
    "mmlu_pro_law",
    "mmlu_pro_math",
    "mmlu_pro_other",
    "mmlu_pro_philosophy",
    "mmlu_pro_physics",
    "mmlu_pro_psychology",
]

MMLU_REDUX_SUBTASKS = [
    "mmlu_redux_abstract_algebra_generative",
    "mmlu_redux_anatomy_generative",
    "mmlu_redux_astronomy_generative",
    "mmlu_redux_business_ethics_generative",
    "mmlu_redux_clinical_knowledge_generative",
    "mmlu_redux_college_biology_generative",
    "mmlu_redux_college_chemistry_generative",
    "mmlu_redux_college_computer_science_generative",
    "mmlu_redux_college_mathematics_generative",
    "mmlu_redux_college_medicine_generative",
    "mmlu_redux_college_physics_generative",
    "mmlu_redux_computer_security_generative",
    "mmlu_redux_conceptual_physics_generative",
    "mmlu_redux_econometrics_generative",
    "mmlu_redux_electrical_engineering_generative",
    "mmlu_redux_elementary_mathematics_generative",
    "mmlu_redux_formal_logic_generative",
    "mmlu_redux_global_facts_generative",
    "mmlu_redux_high_school_biology_generative",
    "mmlu_redux_high_school_chemistry_generative",
    "mmlu_redux_high_school_computer_science_generative",
    "mmlu_redux_high_school_european_history_generative",
    "mmlu_redux_high_school_geography_generative",
    "mmlu_redux_high_school_government_and_politics_generative",
    "mmlu_redux_high_school_macroeconomics_generative",
    "mmlu_redux_high_school_mathematics_generative",
    "mmlu_redux_high_school_microeconomics_generative",
    "mmlu_redux_high_school_physics_generative",
    "mmlu_redux_high_school_psychology_generative",
    "mmlu_redux_high_school_statistics_generative",
    "mmlu_redux_high_school_us_history_generative",
    "mmlu_redux_high_school_world_history_generative",
    "mmlu_redux_human_aging_generative",
    "mmlu_redux_human_sexuality_generative",
    "mmlu_redux_international_law_generative",
    "mmlu_redux_jurisprudence_generative",
    "mmlu_redux_logical_fallacies_generative",
    "mmlu_redux_machine_learning_generative",
    "mmlu_redux_management_generative",
    "mmlu_redux_marketing_generative",
    "mmlu_redux_medical_genetics_generative",
    "mmlu_redux_miscellaneous_generative",
    "mmlu_redux_moral_disputes_generative",
    "mmlu_redux_moral_scenarios_generative",
    "mmlu_redux_nutrition_generative",
    "mmlu_redux_philosophy_generative",
    "mmlu_redux_prehistory_generative",
    "mmlu_redux_professional_accounting_generative",
    "mmlu_redux_professional_law_generative",
    "mmlu_redux_professional_medicine_generative",
    "mmlu_redux_professional_psychology_generative",
    "mmlu_redux_public_relations_generative",
    "mmlu_redux_security_studies_generative",
    "mmlu_redux_sociology_generative",
    "mmlu_redux_us_foreign_policy_generative",
    "mmlu_redux_virology_generative",
    "mmlu_redux_world_religions_generative",
]

MMLU_REDUX_AGGREGATES = [
    "mmlu_redux_generative",
    "mmlu_redux_humanities_generative",
    "mmlu_redux_other_generative",
    "mmlu_redux_social_sciences_generative",
    "mmlu_redux_spanish",
    "mmlu_redux_spanish_generative",
    "mmlu_redux_stem_generative",
]

AGIEVAL_SUBTASKS = [
    "agieval_aqua_rat",
    "agieval_gaokao_biology",
    "agieval_gaokao_chemistry",
    "agieval_gaokao_chinese",
    "agieval_gaokao_english",
    "agieval_gaokao_geography",
    "agieval_gaokao_history",
    "agieval_gaokao_mathcloze",
    "agieval_gaokao_mathqa",
    "agieval_gaokao_physics",
    "agieval_jec_qa_ca",
    "agieval_jec_qa_kd",
    "agieval_logiqa_en",
    "agieval_logiqa_zh",
    "agieval_lsat_ar",
    "agieval_lsat_lr",
    "agieval_lsat_rc",
    "agieval_sat_en",
    "agieval_sat_en_without_passage",
    "agieval_sat_math",
]

BBH_ZEROSHOT_SUBTASKS = [
    "bbh_zeroshot_boolean_expressions",
    "bbh_zeroshot_causal_judgement",
    "bbh_zeroshot_date_understanding",
    "bbh_zeroshot_disambiguation_qa",
    "bbh_zeroshot_dyck_languages",
    "bbh_zeroshot_formal_fallacies",
    "bbh_zeroshot_geometric_shapes",
    "bbh_zeroshot_hyperbaton",
    "bbh_zeroshot_logical_deduction_five_objects",
    "bbh_zeroshot_logical_deduction_seven_objects",
    "bbh_zeroshot_logical_deduction_three_objects",
    "bbh_zeroshot_movie_recommendation",
    "bbh_zeroshot_multistep_arithmetic_two",
    "bbh_zeroshot_navigate",
    "bbh_zeroshot_object_counting",
    "bbh_zeroshot_penguins_in_a_table",
    "bbh_zeroshot_reasoning_about_colored_objects",
    "bbh_zeroshot_ruin_names",
    "bbh_zeroshot_salient_translation_error_detection",
    "bbh_zeroshot_snarks",
    "bbh_zeroshot_sports_understanding",
    "bbh_zeroshot_temporal_sequences",
    "bbh_zeroshot_tracking_shuffled_objects_five_objects",
    "bbh_zeroshot_tracking_shuffled_objects_seven_objects",
    "bbh_zeroshot_tracking_shuffled_objects_three_objects",
    "bbh_zeroshot_web_of_lies",
    "bbh_zeroshot_word_sorting",
]

BBH_FEWSHOT_SUBTASKS = [task.replace("bbh_zeroshot_", "bbh_fewshot_") for task in BBH_ZEROSHOT_SUBTASKS]
BBH_COT_ZEROSHOT_SUBTASKS = [task.replace("bbh_zeroshot_", "bbh_cot_zeroshot_") for task in BBH_ZEROSHOT_SUBTASKS]
BBH_COT_FEWSHOT_SUBTASKS = [task.replace("bbh_zeroshot_", "bbh_cot_fewshot_") for task in BBH_ZEROSHOT_SUBTASKS]

EVAL_GROUPS = [
    {
        "name": "mmlu_pro",
        "label": "mmlu_pro_full",
        "description": "MMLU-Pro 14 subject subtasks.",
        "tasks": MMLU_PRO_SUBTASKS,
    },
    {
        "name": "mmlu_redux",
        "label": "mmlu_redux_full_subjects",
        "description": "MMLU-Redux subject-level generative subtasks, excluding aggregate rollups.",
        "tasks": MMLU_REDUX_SUBTASKS,
    },
    {
        "name": "agieval",
        "label": "agieval_full",
        "description": "All atomic AGIEval subtasks, including English and Chinese exam subsets.",
        "tasks": AGIEVAL_SUBTASKS,
    },
    {
        "name": "bbh",
        "label": "bbh_zeroshot_full",
        "description": "BBH zeroshot non-CoT subtasks.",
        "tasks": BBH_ZEROSHOT_SUBTASKS,
    },
]

PRESETS = {group["name"]: group["tasks"] for group in EVAL_GROUPS}
PRESETS["mmlu_redux_aggregates"] = MMLU_REDUX_AGGREGATES
PRESETS["bbh_fewshot"] = BBH_FEWSHOT_SUBTASKS
PRESETS["bbh_cot_zeroshot"] = BBH_COT_ZEROSHOT_SUBTASKS
PRESETS["bbh_cot_fewshot"] = BBH_COT_FEWSHOT_SUBTASKS
PRESETS["all"] = [task for group in EVAL_GROUPS for task in group["tasks"]]
PRESETS["requested"] = PRESETS["all"]

GROUP_LABELS = {group["name"]: group["label"] for group in EVAL_GROUPS}
GROUP_DESCRIPTIONS = {group["name"]: group["description"] for group in EVAL_GROUPS}
GROUP_LABELS.update(
    {
        "mmlu_redux_aggregates": "mmlu_redux_aggregate_rollups",
        "bbh_fewshot": "bbh_fewshot_full",
        "bbh_cot_zeroshot": "bbh_cot_zeroshot_full",
        "bbh_cot_fewshot": "bbh_cot_fewshot_full",
    }
)
GROUP_DESCRIPTIONS.update(
    {
        "mmlu_redux_aggregates": "Optional MMLU-Redux aggregate rollup tasks.",
        "bbh_fewshot": "Optional BBH fewshot non-CoT subtasks.",
        "bbh_cot_zeroshot": "Optional BBH zeroshot CoT subtasks.",
        "bbh_cot_fewshot": "Optional BBH fewshot CoT subtasks.",
    }
)


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


def validate_tasks(tasks):
    try:
        from lm_eval.tasks import TaskManager
    except Exception as exc:  # noqa: BLE001
        print(f"Task validation skipped: failed to import lm_eval TaskManager: {exc}", file=sys.stderr)
        return {"validated": False, "missing": []}

    manager = TaskManager()
    available = set(manager.all_tasks)
    missing = [task for task in tasks if task not in available]
    if missing:
        raise RuntimeError(
            "The current lm-eval install is missing requested tasks:\n"
            + "\n".join(f"- {task}" for task in missing)
        )
    return {"validated": True, "missing": []}


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
    parser.add_argument("--output_root", type=str, default="results/lm_eval/eval_script6_lm_eval_full_subtasks")
    parser.add_argument("--tmp_root", type=str, default="/tmp")
    parser.add_argument("--prepared_model_dir", type=str, default=None)
    parser.add_argument("--save_prepared_model_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_task_validation", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    presets = args.preset if args.preset is not None else ["all"]
    tasks = expand_tasks(args.task, presets)
    task_groups = summarize_task_groups(args.task, presets, tasks)
    if not args.skip_task_validation:
        validate_tasks(tasks)
    if args.smoke and args.limit is None:
        args.limit = 5
    if args.device is None:
        args.device = f"cuda:{args.gpu}"

    model_id = args.model_id
    model_name_safe = Path(model_id).name.replace(".", "_")
    run_name = f"{model_name_safe}_alpha{args.alpha}_{args.backend}"
    output_dir = os.path.join(args.output_root, run_name)
    if os.path.exists(output_dir) and not args.force:
        result_files = []
        for root, _, files in os.walk(output_dir):
            result_files.extend(f for f in files if f.startswith("results_") and f.endswith(".json"))
        if result_files:
            print(f"Output exists with results: {output_dir}")
            return
        raise RuntimeError(f"Output exists without lm-eval results; pass --force or remove it: {output_dir}")
    if os.path.exists(output_dir) and args.force:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run_config = {
        "model_id": model_id,
        "alpha": args.alpha,
        "backend": args.backend,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "presets": presets,
        "tasks": tasks,
        "task_count": len(tasks),
        "task_groups": task_groups,
        "prepared_model_dir": args.prepared_model_dir,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    if args.prepared_model_dir:
        model_dir = args.prepared_model_dir
    elif args.alpha == 0.0:
        model_dir = model_id
    else:
        if args.save_prepared_model_dir:
            work_dir = args.save_prepared_model_dir
            os.makedirs(work_dir, exist_ok=True)
        else:
            os.makedirs(args.tmp_root, exist_ok=True)
            work_dir = tempfile.mkdtemp(prefix=f"eval_script6_{model_name_safe}_alpha{args.alpha}_", dir=args.tmp_root)
        print(f"Preparing perturbed model in {work_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": args.device},
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        apply_svd_energy_matthew(model, args.alpha)
        model.save_pretrained(work_dir, max_shard_size="4GB")
        tokenizer.save_pretrained(work_dir)
        model_dir = work_dir

    cmd = build_lm_eval_command(args, model_dir, output_dir, tasks)
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        if args.prepared_model_dir is None and args.alpha != 0.0 and not args.save_prepared_model_dir:
            shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
