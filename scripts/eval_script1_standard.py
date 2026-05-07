import torch
import json
import argparse
import os
import re
import shutil
import tempfile
from fractions import Fraction
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ======== SVD PERTURBATION ========
def apply_svd_energy_matthew(model, alpha=0.0):
    if alpha == 0.0: return
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
        return match.group(1).strip()
    return None

def normalize(s):
    if s is None: return ""
    s = s.replace("\\ ", " ")
    s = s.replace(" ", "").replace("\\n", "").lower()
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
    if pred is None or gt is None: return False
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
    if task in {"MATH", "DROP"}:
        pred_num = parse_number(pred)
        gt_num = parse_number(gt)
        return pred_num is not None and gt_num is not None and pred_num == gt_num
    return False

def render_chat_prompt(tokenizer, messages, enable_thinking=None):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)

def strip_thinking(text):
    return text.split("</think>")[-1] if "</think>" in text else text

def extract_fallback_answer(text, task):
    final_text = strip_thinking(text).strip()
    pred = extract_boxed_answer(final_text)
    if pred is not None:
        return pred, "boxed"
    if task == "ARC-Challenge":
        matches = re.findall(r"(?:answer|option|choice)\s*[:：]?\s*\b([A-Za-z0-9]+)\b", final_text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper(), "final_option"
    if task in {"GSM8K", "MATH", "DROP"}:
        num = extract_number(final_text)
        if num is not None:
            return num, "final_number"
    lines = [line.strip() for line in final_text.splitlines() if line.strip()]
    if lines:
        last = re.sub(r"^(?:final answer|answer)\s*[:：]\s*", "", lines[-1], flags=re.IGNORECASE).strip()
        if 0 < len(last) <= 120:
            return last, "final_line"
    return None, "format_error"

# ======== DATASETS ========
def load_eval_datasets(num_samples):
    print("Loading datasets...")
    data = {}
    
    # GSM8K
    ds = load_dataset("gsm8k", "main", split="test")
    gsm8k = []
    for row in ds.select(range(min(num_samples, len(ds)))):
        ans = row["answer"].split("#### ")[-1].strip()
        gsm8k.append({"problem": row["question"], "ground_truth": ans})
    data["GSM8K"] = gsm8k
    
    # MATH
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math = []
    for row in ds.select(range(min(num_samples, len(ds)))):
        gt = extract_boxed_answer(row["solution"]) or row["answer"]
        math.append({"problem": row["problem"], "ground_truth": gt})
    data["MATH"] = math
    
    # ARC-Challenge
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc = []
    for row in ds.select(range(min(num_samples, len(ds)))):
        choices = row["choices"]
        formatted_choices = "\n".join(
            f"{label}) {text}" for label, text in zip(choices["label"], choices["text"])
        )
        prob = f"{row['question']}\n{formatted_choices}\nThis is multiple choice. Put only the correct option label inside \\boxed{{}}."
        arc.append({"problem": prob, "ground_truth": row["answerKey"]})
    data["ARC-Challenge"] = arc
    
    # DROP
    ds = load_dataset("drop", split="validation")
    drop = []
    for row in ds.select(range(min(num_samples, len(ds)))):
        prob = f"Context: {row['passage']}\\nQuestion: {row['question']}"
        drop.append({"problem": prob, "ground_truth": row["answers_spans"]["spans"][0]})
    data["DROP"] = drop
    
    return data

def run_evaluation(
    model_dir,
    data,
    tokenizer,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    max_model_len=4096,
    max_tokens=256,
    enforce_eager=False,
    debug_samples=5,
):
    print("Generating responses for no_cot mode...")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )
    
    sys_msg = "Solve the problem directly. Do not output any reasoning steps. You MUST put your final answer inside \\boxed{}."
    try:
        results = {}
        for task, items in data.items():
            prompts = []
            for item in items:
                msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": item["problem"]}]
                prompts.append(render_chat_prompt(tokenizer, msgs, enable_thinking=False))

            outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=True)

            format_errors = 0
            wrong_answers = 0
            correct_answers = 0
            parse_sources = {}
            debug_examples = []

            for i, out in enumerate(outputs):
                out_text = out.outputs[0].text.strip()
                pred, source = extract_fallback_answer(out_text, task)
                gt = items[i]["ground_truth"]
                parse_sources[source] = parse_sources.get(source, 0) + 1

                if pred is None:
                    format_errors += 1
                    bucket = "format"
                elif is_equivalent(pred, gt, task):
                    correct_answers += 1
                    bucket = "correct"
                else:
                    wrong_answers += 1
                    bucket = "wrong"

                if bucket != "correct" and len(debug_examples) < debug_samples:
                    debug_examples.append({
                        "index": i,
                        "bucket": bucket,
                        "ground_truth": gt,
                        "prediction": pred,
                        "parse_source": source,
                        "finish_reason": out.outputs[0].finish_reason,
                        "output": out_text[:1000],
                    })

            total = len(items)
            results[task] = {
                "Total": total,
                "Format_Error_Rate": format_errors / total,
                "Wrong_Answer_Rate": wrong_answers / total,
                "Correct_Rate": correct_answers / total,
                "Parse_Sources": parse_sources,
                "Debug_Examples": debug_examples,
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
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--output_root", type=str, default="results/standard/eval_script1_standard")
    parser.add_argument("--tmp_root", type=str, default="/tmp")
    parser.add_argument("--prepared_model_dir", type=str, default=None)
    parser.add_argument("--save_prepared_model_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    model_name_safe = model_id.split("/")[-1].replace(".", "_")
    device = f"cuda:{args.gpu}"
    out_path = os.path.join(args.output_root, f"{model_name_safe}_res_alpha{args.alpha}.json")
    if not args.smoke and os.path.exists(out_path):
        return

    tokenizer_source = args.prepared_model_dir or model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    cleanup_dir = None
    if args.prepared_model_dir:
        eval_model_dir = args.prepared_model_dir
    elif args.alpha == 0.0 and not args.save_prepared_model_dir:
        eval_model_dir = model_id
    else:
        eval_model_dir = args.save_prepared_model_dir or tempfile.mkdtemp(
            prefix=f"eval_script1_{model_name_safe}_alpha{args.alpha}_",
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

        results_no_cot = run_evaluation(
            eval_model_dir,
            data,
            tokenizer,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            enforce_eager=args.enforce_eager,
        )
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
    
    final_res = {
        "config": {
            "method": "eval_script1",
            "alpha": args.alpha,
            "model_id": model_id,
            "num_samples": num_samples,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "enforce_eager": args.enforce_eager,
            "prepared_model_dir": args.prepared_model_dir,
            "save_prepared_model_dir": args.save_prepared_model_dir,
            "eval_model_dir": eval_model_dir,
            "thinking_mode": "disabled",
        },
        "scores_no_cot": results_no_cot
    }
    
    os.makedirs(args.output_root, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_res, f, indent=2)

if __name__ == "__main__":
    main()
