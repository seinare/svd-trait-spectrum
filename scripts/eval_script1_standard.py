import torch
import json
import argparse
import os
import re
import shutil
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
    s = s.replace(" ", "").replace("\\n", "").lower()
    s = s.replace("\\\\", "\\").replace("\\!", "").replace("\\,", "")
    return s

def is_equivalent(pred, gt):
    if pred is None or gt is None: return False
    return normalize(pred) == normalize(gt)

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
    
    # GPQA
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    gpqa = []
    for row in ds.select(range(min(num_samples, len(ds)))):
        prob = f"{row['Question']}\\nA) {row['Incorrect Answer 1']}\\nB) {row['Incorrect Answer 2']}\\nC) {row['Incorrect Answer 3']}\\nThis is multiple choice, answer with the exact text of the correct choice."
        gpqa.append({"problem": prob, "ground_truth": row["Correct Answer"]})
    data["GPQA"] = gpqa
    
    # DROP
    ds = load_dataset("drop", split="validation")
    drop = []
    for row in ds.select(range(min(num_samples, len(ds)))):
        prob = f"Context: {row['passage']}\\nQuestion: {row['question']}"
        drop.append({"problem": prob, "ground_truth": row["answers_spans"]["spans"][0]})
    data["DROP"] = drop
    
    return data

def run_evaluation(model_dir, data, tokenizer, mode="cot"):
    print(f"Generating responses for {mode} mode...")
    llm = LLM(model=model_dir, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=4096)
    
    if mode == "cot":
        sys_msg = "Solve the problem step-by-step. You MUST put your final answer inside \\boxed{}."
        max_tokens = 512
    else:
        sys_msg = "Solve the problem directly. Do not output any reasoning steps. You MUST put your final answer inside \\boxed{}."
        max_tokens = 64
        
    results = {}
    for task, items in data.items():
        prompts = []
        for item in items:
            msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": item["problem"]}]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=True)
        
        format_errors = 0
        wrong_answers = 0
        correct_answers = 0
        
        for i, out in enumerate(outputs):
            out_text = out.outputs[0].text.strip()
            pred = extract_boxed_answer(out_text)
            gt = items[i]["ground_truth"]
            
            if pred is None:
                format_errors += 1
            elif is_equivalent(pred, gt) or (normalize(gt) in normalize(pred)):
                correct_answers += 1
            else:
                wrong_answers += 1
                
        total = len(items)
        results[task] = {
            "Total": total,
            "Format_Error_Rate": format_errors / total,
            "Wrong_Answer_Rate": wrong_answers / total,
            "Correct_Rate": correct_answers / total
        }
        
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
    args = parser.parse_args()

    model_id = args.model_id
    model_name_safe = model_id.split("/")[-1].replace(".", "_")
    device = f"cuda:{args.gpu}"
    
    out_path = f"results/eval_script1_standard/{model_name_safe}_res_alpha{args.alpha}.json"
    if not args.smoke and os.path.exists(out_path):
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    apply_svd_energy_matthew(model, args.alpha)
    
    temp_dir = f"/tmp/eval_script1_tmp_{args.alpha}"
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    del model
    torch.cuda.empty_cache()
    
    num_samples = 2 if args.smoke else 100
    data = load_eval_datasets(num_samples)
    
    results_cot = run_evaluation(temp_dir, data, tokenizer, mode="cot")
    results_no_cot = run_evaluation(temp_dir, data, tokenizer, mode="no_cot")
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    final_res = {
        "config": {"method": "eval_script1", "alpha": args.alpha},
        "scores_cot": results_cot,
        "scores_no_cot": results_no_cot
    }
    
    os.makedirs("results/eval_script1_standard", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_res, f, indent=2)

if __name__ == "__main__":
    main()
