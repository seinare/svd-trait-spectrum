import argparse
import json
import os
import re
import shutil
import tempfile
from fractions import Fraction

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


# ======== SVD PERTURBATION ========
def apply_svd_energy_matthew(model, alpha=0.0):
    if alpha == 0.0:
        return
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


# ======== PARSING UTILS ========
def extract_boxed_answer(text):
    match = re.search(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", text)
    if match:
        return match.group(1).strip(), "boxed"
    return None, None


def strip_thinking(text):
    if "</think>" in text:
        return text.split("</think>")[-1].strip(), True, True
    if "<think>" in text:
        return text.split("<think>", 1)[-1].strip(), True, False
    return text.strip(), False, False


def normalize(s):
    if s is None:
        return ""
    s = s.replace("\\ ", " ")
    s = s.replace(" ", "").replace("\n", "").lower()
    s = s.replace("\\\\", "\\").replace("\\!", "").replace("\\,", "").replace("\\", "")
    s = re.sub(r"text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"[^a-z0-9./+-]", "", s)
    return s


def extract_number(s):
    if s is None:
        return None
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?", s)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def parse_number(s):
    num = extract_number(s)
    if num is None:
        return None
    try:
        return Fraction(num)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_choice_label(value):
    match = re.fullmatch(r"\s*([A-Za-z0-9]+)\s*[\).:]?\s*", str(value))
    return match.group(1).upper() if match else None


def is_equivalent(pred, gt, task):
    if pred is None or gt is None:
        return False

    if task == "ARC-Challenge":
        pred_label = normalize_choice_label(pred)
        gt_label = normalize_choice_label(gt)
        return pred_label is not None and gt_label is not None and pred_label == gt_label

    if task == "GSM8K":
        pred_num = parse_number(pred)
        gt_num = parse_number(gt)
        return pred_num is not None and gt_num is not None and pred_num == gt_num

    pred_norm = normalize(pred)
    gt_norm = normalize(gt)
    if pred_norm == gt_norm:
        return True

    if task == "MATH":
        pred_num = parse_number(pred)
        gt_num = parse_number(gt)
        return pred_num is not None and gt_num is not None and pred_num == gt_num

    if task == "DROP":
        pred_num = parse_number(pred)
        gt_num = parse_number(gt)
        if pred_num is not None and gt_num is not None:
            return pred_num == gt_num
        return False

    return False


def extract_fallback_answer(text, task):
    final_text, has_think, closed_think = strip_thinking(text)
    pred, source = extract_boxed_answer(final_text)
    if pred is not None:
        return pred, source, final_text, has_think, closed_think

    if task == "ARC-Challenge":
        matches = re.findall(r"(?:answer|option|choice)\s*[:：]?\s*\b([A-Za-z0-9]+)\b", final_text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper(), "final_option", final_text, has_think, closed_think

    if task in {"GSM8K", "MATH"}:
        matches = re.findall(r"[-+]?\\d[\\d,]*(?:\\.\\d+)?(?:/\\d+)?", final_text)
        if matches:
            return matches[-1].replace(",", ""), "final_number", final_text, has_think, closed_think

    lines = [line.strip() for line in final_text.splitlines() if line.strip()]
    if lines:
        last = re.sub(r"^(?:final answer|answer)\\s*[:：]\\s*", "", lines[-1], flags=re.IGNORECASE).strip()
        if 0 < len(last) <= 120:
            return last, "final_line", final_text, has_think, closed_think

    return None, None, final_text, has_think, closed_think


def render_chat_prompt(tokenizer, messages, thinking_budget=None):
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    if thinking_budget is not None:
        template_kwargs["thinking_budget"] = thinking_budget
    return tokenizer.apply_chat_template(
        messages,
        **template_kwargs,
    )


# ======== DATASETS ========
def load_eval_datasets(num_samples):
    print("Loading datasets...")
    data = {}

    ds = load_dataset("gsm8k", "main", split="test")
    gsm8k = []
    for row in ds.select(range(min(num_samples, len(ds)))):
        ans = row["answer"].split("#### ")[-1].strip()
        gsm8k.append({"problem": row["question"], "ground_truth": ans})
    data["GSM8K"] = gsm8k

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math = []
    for row in ds.select(range(min(num_samples, len(ds)))):
        gt, _ = extract_boxed_answer(row["solution"])
        math.append({"problem": row["problem"], "ground_truth": gt or row["answer"]})
    data["MATH"] = math

    return data


def run_evaluation(
    model_dir,
    data,
    tokenizer,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    max_model_len=4096,
    max_tokens=8192,
    thinking_budget=None,
    enforce_eager=False,
    debug_samples=5,
    correct_debug_samples=3,
):
    print("Generating responses for standard_cot mode...")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )

    sys_msg = "Think through the problem carefully, then put your final answer inside \\boxed{}."

    try:
        results = {}
        for task, items in data.items():
            prompts = []
            for item in items:
                msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": item["problem"]}]
                prompts.append(render_chat_prompt(tokenizer, msgs, thinking_budget=thinking_budget))

            outputs = llm.generate(
                prompts,
                SamplingParams(temperature=0.0, max_tokens=max_tokens),
                use_tqdm=True,
            )

            format_errors = 0
            wrong_answers = 0
            correct_answers = 0
            parse_sources = {}
            has_think_count = 0
            closed_think_count = 0
            length_finish_count = 0
            debug = []
            correct_debug = []

            for i, out in enumerate(outputs):
                completion = out.outputs[0]
                out_text = completion.text.strip()
                pred, source, final_text, has_think, closed_think = extract_fallback_answer(out_text, task)
                gt = items[i]["ground_truth"]
                finish_reason = getattr(completion, "finish_reason", None)
                if finish_reason == "length" and source == "final_line":
                    pred = None
                    source = None

                if has_think:
                    has_think_count += 1
                if closed_think:
                    closed_think_count += 1
                if finish_reason == "length":
                    length_finish_count += 1
                if source:
                    parse_sources[source] = parse_sources.get(source, 0) + 1

                if pred is None:
                    format_errors += 1
                    outcome = "format_error"
                elif is_equivalent(pred, gt, task):
                    correct_answers += 1
                    outcome = "correct"
                else:
                    wrong_answers += 1
                    outcome = "wrong"

                if len(debug) < debug_samples and outcome != "correct":
                    debug.append(
                        {
                            "index": i,
                            "outcome": outcome,
                            "ground_truth": gt,
                            "prediction": pred,
                            "parse_source": source,
                            "finish_reason": finish_reason,
                            "has_think": has_think,
                            "closed_think": closed_think,
                            "final_text_preview": final_text[:800],
                            "raw_preview": out_text[:1200],
                        }
                    )
                elif outcome == "correct" and len(correct_debug) < correct_debug_samples:
                    correct_debug.append(
                        {
                            "index": i,
                            "ground_truth": gt,
                            "prediction": pred,
                            "parse_source": source,
                            "finish_reason": finish_reason,
                            "has_think": has_think,
                            "closed_think": closed_think,
                            "final_text_preview": final_text[:800],
                            "raw_preview": out_text[:1200],
                        }
                    )

            total = len(items)
            results[task] = {
                "Total": total,
                "Format_Error_Rate": format_errors / total,
                "Wrong_Answer_Rate": wrong_answers / total,
                "Correct_Rate": correct_answers / total,
                "Has_Think_Rate": has_think_count / total,
                "Closed_Think_Rate": closed_think_count / total,
                "Length_Finish_Rate": length_finish_count / total,
                "Parse_Sources": parse_sources,
                "Debug_Examples": debug,
                "Correct_Debug_Examples": correct_debug,
            }
    finally:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        del llm
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--thinking_budget", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--output_root", type=str, default="results/standard_cot/eval_script1_standard_cot")
    parser.add_argument("--tmp_root", type=str, default="/tmp")
    parser.add_argument("--prepared_model_dir", type=str, default=None)
    parser.add_argument("--save_prepared_model_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--debug_samples", type=int, default=5)
    parser.add_argument("--correct_debug_samples", type=int, default=3)
    args = parser.parse_args()

    model_id = args.model_id
    model_name_safe = model_id.split("/")[-1].replace(".", "_")
    device = f"cuda:{args.gpu}"
    out_path = os.path.join(args.output_root, f"{model_name_safe}_res_alpha{args.alpha}.json")
    if not args.smoke and os.path.exists(out_path):
        return

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
            prefix=f"eval_script1_standard_cot_{model_name_safe}_alpha{args.alpha}_",
            dir=args.tmp_root,
        )
        if not args.save_prepared_model_dir:
            cleanup_dir = eval_model_dir
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
        num_samples = args.num_samples if args.num_samples is not None else (2 if args.smoke else 100)
        data = load_eval_datasets(num_samples)
        scores_cot = run_evaluation(
            eval_model_dir,
            data,
            tokenizer,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            enforce_eager=args.enforce_eager,
            debug_samples=args.debug_samples,
            correct_debug_samples=args.correct_debug_samples,
        )
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    final_res = {
        "config": {
            "method": "eval_script1_standard_cot",
            "alpha": args.alpha,
            "model_id": model_id,
            "num_samples": num_samples,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "thinking_budget": args.thinking_budget,
            "enforce_eager": args.enforce_eager,
            "prepared_model_dir": args.prepared_model_dir,
            "save_prepared_model_dir": args.save_prepared_model_dir,
            "eval_model_dir": eval_model_dir,
            "thinking_mode": "cot_enabled",
        },
        "scores_cot": scores_cot,
    }

    os.makedirs(args.output_root, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_res, f, indent=2)


if __name__ == "__main__":
    main()
